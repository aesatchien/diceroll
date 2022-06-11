import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
import time
from copy import deepcopy
import re

#from dicecharts import make_acplot, make_histogram, make_ac_comparison_plot

# -------- NUMERICAL SECTION -----------
def dice_list(nd_list=((1, 6)), bonus=0, debug=False, time_limit=10):
    '''

    Calculates the probability density function for list of dice rolls of arbitrary type
    Cuts large rolls into chunks - otherwise this is really short
     :param nd_list: list of (n,d) tuples
    '''

    start_time = time.time()
    time_list = []
    # on my computer this scales as below
    n_tot = sum([n for n, d in nd_list])
    d_avg = (sum([d * n for n, d in nd_list])) / n_tot
    time_guesstimate = 0.5 * d_avg ** n_tot / 1000
    # abort if it's going to take too long
    if debug:
        print(f"Unchunked time guesstimate is {time_guesstimate / 1000:.2f} s", flush=True)
        if time_guesstimate > time_limit * 1000:
            print(f'Unchunked guesstimate {time_guesstimate / 1000:.2f}s exceeds time limit of 10s...')
            # return []

    # if n is too high, split it up and do the joint dist for the speed advantage
    if nd_list[0][0] > 12:
        chunk = 6
    else:
        chunk = 5
    if nd_list[0][0] > chunk and len(nd_list) < 2:  # empirically chunking at 5 or 6 seems best?
        print(f'Splitting in chunks of {chunk} to save processing time...')
        n = nd_list[0][0]
        d = nd_list[0][1]
        dist_list = []

        for ix in range(n // chunk):
            dist_list.append(dice_roll(chunk, d, 0))
        if n % chunk > 0:
            dist_list.append(dice_roll(n % chunk, d, 0))
        dist = np.array(joint_dist(dist_list, debug=debug))
        dist[:, 0] += bonus
        return dist

    all_tuples = []
    for n, d in nd_list:
        # set up the die from 1 to n and add bonus first
        die = list(range(1, d + 1))
        for ix in range(n):
            all_tuples.append(die)
    time_list.append(['create tuples', time.time() - start_time])
    # most of the work is done here, takes about %50 of the time. unpack and then product them all together
    all_tuples = list(it.product(*all_tuples))
    time_list.append(['it product', time.time() - start_time])
    # next hardest bit is summing them all up, takes about 40% of the time
    sums = list(map(sum, all_tuples))
    time_list.append(['sums', time.time() - start_time])
    # the conversion to np array takes about 10% of the time
    array_sums = np.array(sums)
    time_list.append(['numpy conversion', time.time() - start_time])
    # add in the bonus
    array_sums += bonus
    # getting the actual probability distribution is trivial with the numpy calls
    unique, counts = np.unique(array_sums, return_counts=True)
    time_list.append(['numpy unique counts', time.time() - start_time])
    if debug:
        for ix, item in enumerate(time_list):
            print(
                f'{item[0]:20s}: {1000 * item[1]:6.2f}   Delta: {1000 * (item[1] - time_list[max(ix - 1, 0)][1]):6.2f}')
    return np.array(list(zip(unique, counts / len(array_sums))))


# helper for the one that takes an arbitrary dice list
def dice_roll(n, d, bonus=0, extra_n=None, extra_d=None, debug=False, time_limit=10):
    """

    Calculates the probability density function for n dice rolls of same type
    This is simple and starts to seriously bog down if passed more than 6 dice

    """
    # on my computer this scales as below
    time_guesstimate = 0.5 * d ** n / 1000
    # quit if it's going to take too long
    if debug:
        print(f"Time guestimate is {time_guesstimate:.2f} ms")
        if time_guesstimate > time_limit * 1000:
            print(f'Guestimate {time_guesstimate / 1000:.2f}s exceeds time limit of 10s.  Aborting')
            return []
    # set up the die from 1 to n and add bonus first
    die = list(range(1, d + 1))
    # most of the work is done here, takes about %50 of the time
    all_tuples = list(it.product(die, repeat=n))
    # next hardest bit is summing them all up, takes about 40% of the time
    sums = list(map(sum, all_tuples))
    # the conversion to np array takes about 10% of the time
    array_sums = np.array(sums)
    # add in the bonus
    array_sums += bonus
    # getting the actual probability distribution is trivial with the numpy calls
    unique, counts = np.unique(array_sums, return_counts=True)
    return np.array(list(zip(unique, counts / len(array_sums))))


# Generating a joint distribution - this probably takes the longest of anything.  will need to profile
# Why is this version so much faster than dice_list?
def joint_dist(dists_list, debug=False):
    """
    Takes a list of distributions [[damage_1, probability_1],...] , breaks down into all
    damages and all probabilities, gets their products, then applies appropriate operator
    to recombine into one large distribution.  Much more elegant in Mathematica.
    :param dists_list:
    :return: numpy array of damages and probabilities
    """
    start_time = time.time()
    time_list = []
    all_damage_tuples = []
    all_probability_tuples = []
    for dist in dists_list:
        all_damage_tuples.append([damage for damage, probability in dist])
        all_probability_tuples.append([probability for damage, probability in dist])
    time_list.append(['create tuples', time.time() - start_time])
    # most of the work is done here, takes about %50 of the time. unpack and then product them all together
    all_damage_tuples = list(it.product(*all_damage_tuples))
    all_probability_tuples = list(it.product(*all_probability_tuples))
    time_list.append(['it products', time.time() - start_time])

    # next hardest bit is summing them all up, takes about 40% of the time
    # python has a sum, but no prod function, so use np
    damages = list(map(sum, all_damage_tuples))
    probabilities = np.prod(all_probability_tuples, axis=1)
    time_list.append(['sums & prods', time.time() - start_time])
    # and humpty-dumpty is back together again
    pdist = list(zip(damages, probabilities))
    # groupby just seemed the easiest way to take care of my problem of double values
    df = pd.DataFrame(pdist, columns=['damage', 'probability'])
    final_df = df.groupby(['damage']).sum().reset_index()
    time_list.append(['groupby', time.time() - start_time])
    if debug:
        for ix, item in enumerate(time_list):
            print(
                f'{item[0]:20s}: {1000 * item[1]:6.2f}   Delta: {1000 * (item[1] - time_list[max(ix - 1, 0)][1]):6.2f}')

    return np.array(final_df)


def pmdf(d, dice_type='normal'):
    """
    Calculates the probability density function for a dice roll of different type
    :param type:str normal, lucky (best of 2) or unlucky (worst of 2)
    :returns: pdf of the dice roll, eg the 'lucky d20' that takes the best of two d20 rolls

    """
    if dice_type == 'normal':
        return dice_list([(1, 20)], 0, debug=False)
    else:
        die = list(range(1, d + 1))
        all_tuples = list(it.product(die, repeat=2))
        if dice_type == 'lucky':
            array = np.array(list(map(max, all_tuples)))
        elif dice_type == 'unlucky':
            array = np.array(list(map(min, all_tuples)))
        else:
            return dice_list([(1, 20)], 0, debug=False)
        unique, counts = np.unique(array, return_counts=True)
        return np.array(list(zip(unique, counts / len(array))))


# pathfinder for hitting PF2e rules for hitting CRB 445
def to_hit(hit_bonus, AC, dice_type='normal', digits=6):
    message = ""
    hit_probablity = 0
    if AC - hit_bonus > 29:
        message, hit_probablity = "Unhittable", 0
    elif AC - hit_bonus <= -9:
        message, hit_probablity = "Unmissable ", 1
    elif AC - hit_bonus <= 1:
        message, hit_probablity = "Autohit 1-P(1)", 1 - pmdf(20, dice_type)[0][1]
    elif AC - hit_bonus >= 20:
        message, hit_probablity = "Hail Mary P(20)", pmdf(20, dice_type)[-1][1]
    elif AC - hit_bonus > 0:
        message, hit_probablity = "Calculate hit", sum(x[1] for x in pmdf(20, dice_type)[AC - hit_bonus - 1:])
    return message, round(hit_probablity, digits)


# 2e crit calculation  CRB p.445 overrides p.278
def to_crit(hit_bonus, AC, dice_type='normal', digits=6):
    message = ""
    crit_probablity = 0
    message = ""
    crit_probablity = 0
    if AC - hit_bonus <= -9:
        message, crit_probablity = "Autocrit 1-P(1)", 1 - pmdf(20, dice_type)[0][1]
    elif AC - hit_bonus >= 21:
        message, crit_probablity = "Uncrittable", 0
    elif AC - hit_bonus >= 10:
        message, crit_probablity = "HM Crit P(20)", pmdf(20, dice_type)[-1][1]
    elif AC - hit_bonus > -9:
        message, crit_probablity = "Calculate crit", sum(x[1] for x in pmdf(20, dice_type)[10 + AC - hit_bonus - 1:])
    return message, round(crit_probablity, digits)


# PF2e is way easier then 1e on crit calculations...
def hitstats(hit_bonus=None, AC=None, nd_list=((1, 6)), bonus=0, crit=None, critmult=2, deadly=None, extran=None,
             extrad=None, dice_type='normal', debug=False):
    dice_dist = 0
    crit_dist = 0
    hit_message, hit_probablity = to_hit(hit_bonus, AC, dice_type=dice_type)
    crit_message, crit_probablity = to_crit(hit_bonus, AC, dice_type=dice_type)
    dice_dist = dice_list(nd_list, bonus)
    crit_dist = dice_dist.copy()
    dice_dist[:, 1] *= (hit_probablity - crit_probablity)
    crit_dist[:, 1] *= crit_probablity
    crit_dist[:, 0] *= 2
    if deadly is not None:
        deadly_dist = dice_list(deadly)
        # need to concatenate this to crit_dist and normalize - because you don't double it

    # smart way to drop the zeroes (could use np or df as well)
    if crit_probablity < 1E-6:
        crit_dist = np.array([[0, 0]])
    if hit_probablity < 1E-6 or (hit_probablity - crit_probablity) < 1E-6:
        dice_dist = np.array([[0, 0]])
    pdist = np.array([[0, 1 - hit_probablity]])
    pdist = np.concatenate((pdist, dice_dist, crit_dist))
    if debug:
        print(f'Hit Prob: {hit_probablity}  Crit Prob: {crit_probablity} Sum of pdmf: {np.sum(pdist[:, 1]):.3f}' +
              f' Mean dmg: {np.sum(np.prod(pdist, axis=1)):.3f}')
    # groupby just seemed the easiest way to take care of my problem of double values
    df = pd.DataFrame(pdist, columns=['damage', 'probability'])
    df = df.groupby(['damage']).sum().reset_index()
    return np.array(df)


def multiple_attack(attack_list=None, title=None, speed_boost=False, output='histogram'):
    """
    The basic function to group multiple attacks and give a plot thereof
    :param attack_list: list of keyword arguments to the hitstats function
    :param title: title for the chart
    :return: charts or np/pandas objects
    """
    dist_list = []
    pdist = []
    average_damage = 0

    if speed_boost:
        # take a shortcut and don't bother to do the joint probability distribution
        for kwargs in attack_list:
            dmg_dist = hitstats(**kwargs)
            dist_list.append(dmg_dist)
            average_damage += np.dot(dmg_dist[:, 0], dmg_dist[:, 1])

        miss_chance = np.product([x[0][1] for x in dist_list]) # just take the products of all of them
        if miss_chance > 0.99999:
            adjusted_average_damage = 0
        else:
            adjusted_average_damage = average_damage / (1 - miss_chance)
        pdist = np.array([[0, miss_chance],[adjusted_average_damage , 1-miss_chance]])

    else:
        # compute the full probability distribution
        for kwargs in attack_list:
            dist_list.append(hitstats(**kwargs))

        if len(attack_list) > 1:
            pdist = joint_dist(dist_list)
        else:
            pdist = dist_list[0]

    label = make_label(attack_list)
    if title is None:
        title = label

    if output == 'histogram':
        make_histogram([pdist], title=title, description=label)
    elif output == 'pdist':
        return pdist
    elif output == 'df':
        return pd.DataFrame(pdist, columns=['damage', 'probability'])
    return

def ac_scan(attack_list=None, title=None, description=None, start_ac=0, end_ac=45, speed_boost=False, exacting=False, output='df'):
    data = []
    columns = ['AC', 'MISS_CHANCE', 'AVG_DAMAGE', 'STDEV']
    attack_copy = deepcopy(attack_list)  # don't want to change the original
    for ac in range(start_ac, end_ac + 1):
        for x in attack_copy:
            x['AC'] = ac
        if exacting:  # kluge so I can run a comparison with exacting strike (always the last one given)
            hit_bonus = attack_list[0]['hit_bonus']
            attack_copy[-1]['hit_bonus'] = hit_bonus - 5- int(round(5 * to_hit(hit_bonus=hit_bonus, AC=ac, dice_type='normal')[-1]))
        df = multiple_attack(attack_copy, speed_boost=speed_boost, output='df')
        mean = df['damage'].dot(df['probability'])
        stdev = np.round(np.sqrt(np.power(df['damage'] - mean, 2).dot(df['probability'])), 2)
        # miss = 1-to_hit(x['hit_bonus'], ac)[1]
        miss = df['probability'][0]
        data.append([ac, miss, mean, stdev])
    df = pd.DataFrame(data, columns=columns)
    if output == 'df':
        return df
    if output == 'plot':
        # print(f'Title is {title}',flush=True)
        desc = make_label(attack_copy) if description is None else description
        make_acplot(df, title=title, description=desc, start_ac=start_ac, end_ac=end_ac)


def ac_scan_comparison(attack_list1=None, attack_list2=None, title = None, description_1='', description_2='', speed_boost=False,
                       exacting=(False,False), start_ac=0, end_ac=45, output='df'):
    df1 = ac_scan(attack_list=attack_list1, output='df', start_ac=start_ac, end_ac=end_ac, speed_boost=speed_boost, exacting=exacting[0])
    df2 = ac_scan(attack_list=attack_list2, output='df', start_ac=start_ac, end_ac=end_ac, speed_boost=speed_boost, exacting=exacting[1])
    df = pd.concat([df1, df2], axis=1)
    df.columns = columns = ['AC', 'MISS_CHANCE_1', 'AVG_DAMAGE_1', 'STDEV_1', 'AC2', 'MISS_CHANCE_2', 'AVG_DAMAGE_2', 'STDEV_2']
    df = df.drop(['AC2'], axis=1)
    df['DELTA_DAMAGE'] = df1['AVG_DAMAGE'] - df2['AVG_DAMAGE']
    df['DELTA_MISS_CHANCE'] = df1['MISS_CHANCE'] - df2['MISS_CHANCE']
    df['MISS_CHANCE_1'] = df1['MISS_CHANCE']
    df['MISS_CHANCE_2'] = df2['MISS_CHANCE']

    if output == 'df':
        return df
    if output == 'plot':
        df['MISS_CHANCE_1'] = 100 * df['MISS_CHANCE_1']
        df['MISS_CHANCE_2'] = 100 * df['MISS_CHANCE_2']
        df['DELTA_MISS_CHANCE'] = 100 * df['DELTA_MISS_CHANCE']
        label1 = description_1 + ": " + re.sub(r'vs \d*AC ', '', make_label(attack_list1))
        label2 = description_2 + ": " + re.sub(r'vs \d*AC ', '', make_label(attack_list2))
        make_ac_comparison_plot(df, title, label1, label2)


# --------  HELPER FUNCTIONS / WRAPPERS -----------
def make_label(attack_list=None, exacting=False):
    """
    Parse the attack list to get a nicely formatted text description
    :param attack_list: takes the form of [{'hit_bonus':15, 'AC':20, 'nd_list':[(6,8)], 'bonus':4, 'dice_type':'normal'}]
    :return:
    """
    label = ''
    for ix, kwargs in enumerate(attack_list):
        nd = ''
        for iy, die in enumerate(kwargs['nd_list']):
            if iy > 0:
                nd += "+"
            nd += str(die[0]) + 'd' + str(die[1])
        # break after a certain number of attacks ? not at them moment
        if ix > 0 and ix % 2 == 0:
            label += ", "
        elif ix > 0:
            label += ", "  # could make this a comma, but ATM i'd prefer a list
        if 'hit_bonus' in kwargs.keys():
            label += f"+{kwargs['hit_bonus']} vs {kwargs['AC']}AC @ {nd}+{kwargs['bonus']}"
        else:
            label += f"{nd}+{kwargs['bonus']}"
        if 'dice_type' in kwargs.keys():
            if kwargs['dice_type']!= 'normal':
                label += f" ({kwargs['dice_type']})"
    return label.lstrip()


def multi_histogram(attack_list=None, title=None):
    description = make_label(attack_list)
    pdist_list = []
    for attack in attack_list:
        pdist = multiple_attack([attack], output='pdist')
        pdist_list.append(pdist)
    make_histogram(pdist_list, title=description, description=None)
    #return pdist_list



# -------- LOOKING GOOD SECTION - PLOTS AND SUCH -----------
def make_histogram(pdist_list, title='Probablilty Test', description=None):
    """
        Returns a pyplot of the distribution
        note that you can pass this a df or a numpy array (same thing)
    """

    plt.figure(num=None, figsize=(10, 3), dpi=100)
    plt.grid(False)

    for ix, pdist in enumerate(pdist_list):
        font_color = 'C'+str(ix)
        df = pd.DataFrame(pdist, columns=['damage', 'probability'])
        mean = df['damage'].dot(df['probability'])
        stdev = np.sqrt(np.power(df['damage'] - mean, 2).dot(df['probability']))

        plt.bar('damage', 'probability', data=df, width=(0.8-0.1*ix), alpha = 0.9) # alpha=(ix+1)/len(pdist_list)
        # plt.legend(['hit_prob','crit_prob'], loc='upper left')

        description = description.replace(', ', '\n') if description else None
        #description = description.replace(', ', '!!!') if description else None
        text_top = 0.7 -.2*ix
        text_line_height = 0.07
        annotation_offset = 0
        line_count = 0
        if description:
            plt.annotate(text=description, xy=(.85 - .005 * (len(description.splitlines()[0])), text_top),
                         xycoords='axes fraction', fontsize=11)
            annotation_offset = 2 * text_line_height

        if min(df['damage']) < 1:
            plt.annotate(text=f"Miss Chance: {df['probability'][0]:4.2f}", xy=(.79, text_top - annotation_offset),
                         xycoords='axes fraction', fontsize=11, color = font_color)
            line_count+=1
        plt.annotate(text=f'Mean: {mean:4.2f}', xy=(.84, text_top - line_count*text_line_height - annotation_offset),
                     xycoords='axes fraction', fontsize=11, color = font_color)
        line_count += 1
        plt.annotate(text=f'Stdev: {stdev:4.2f}', xy=(.84, text_top - line_count*text_line_height - annotation_offset),
                     xycoords='axes fraction', fontsize=11, color = font_color)

    plt.title(title, fontsize=12)
    plt.xlabel('damage')
    plt.ylabel('probability')
    plt.show()


def make_acplot(df, title=None, description=None, start_ac=0, end_ac=45):
    """
        Returns a pyplot of the AC test
        note that you can pass this a df or a numpy array (same thing)
    """
    # columns = ['AC', 'MISS_CHANCE', 'AVG_DAMAGE', 'STDEV']

    title = 'PF2e AC Plot' if title is None else 'PF2e AC Plot for ' + title
    title = title.replace(f' vs {end_ac}AC', '')
    description = description.replace(f' vs {end_ac}AC', '')

    df['MISS_CHANCE'] = df['MISS_CHANCE'] * 100

    ax = df.plot('AC', 'AVG_DAMAGE', style='ro', figsize=(10, 4), legend=None)
    # ax.set_xlim(start_ac-1, end_ac+1)
    ax.set_ylim(0, 1.1 * np.max(df['AVG_DAMAGE']))
    ax.set_ylabel('Average Damage (hp)', color='red')
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    for a, b in zip(df['AC'], df['AVG_DAMAGE']):
        ax.annotate(str(int(round(b))), xy=(a, b), xytext=(-2, 5), textcoords='offset points', color='red', fontsize=8)
        # plt.text(a, b, str(int(round(b))), color='red', fontsize=8)

    ax2 = df.plot('AC', 'MISS_CHANCE', secondary_y=True, color='blue', ax=ax, legend=None)
    ax2.set_ylabel('% Chance of Missing all Attacks', color='blue')
    ax2.spines['right'].set_color('blue')
    ax2.axes.yaxis.set_ticklabels([])
    ax2.tick_params(axis='y', colors='blue')
    ax2.fill_between(df['AC'], 0, df['MISS_CHANCE'], facecolor='xkcd:blue', alpha=0.1) #'#87CEFA'
    ax2.get_yaxis().set_tick_params(which='both', direction='in')
    ax2.set_ylim(0, 100)
    # ax.set_xlim(start_ac - 1, end_ac + 1)

    align_yaxis(ax, 0, ax2, 0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)  # prevents ax from hiding ax2
    plt.rcParams["figure.dpi"] = 100
    plt.suptitle(title, y=0.99, fontsize=12)
    plt.title(description, fontsize=11)
    ax.set_xlabel('opponent AC')
    plt.show()


def make_ac_comparison_plot(df, title, label1, label2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)
    plt.suptitle('Comparison between two AC scans:\n' + label1 + ' (ATT1)\n' + label2 + ' (ATT2)', fontsize=12, y=1.06)

    df.plot('AC', 'AVG_DAMAGE_1', ax=ax1, style='ro', markersize=4.5)
    df.plot('AC', 'AVG_DAMAGE_2', ax=ax1, style='ro', markersize=4, fillstyle='none', mew=0.75)
    df.plot('AC', 'DELTA_DAMAGE', ax=ax1, color='r')
    df.plot('AC', 'MISS_CHANCE_1', ax=ax2, style='bo', markersize=4.5)
    df.plot('AC', 'MISS_CHANCE_2', ax=ax2, style='bo', markersize=4, fillstyle='none', mew=0.75)
    df.plot('AC', 'DELTA_MISS_CHANCE', ax=ax2, color='b')

    ax1.fill_between(df['AC'], 0, df['DELTA_DAMAGE'], facecolor='xkcd:pink', alpha=0.2)
    ax1.set_ylabel('Positive is good for #1', color='red')
    ax2.set_ylabel('Positive is bad for #1', color='b')
    ax2.fill_between(df['AC'], 0, df['DELTA_MISS_CHANCE'], facecolor='xkcd:blue', alpha=0.2)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax2.get_yaxis().set_tick_params(which='both', direction='in')
    # ylims=np.max(np.abs(df['DELTA_DAMAGE']))
    # ax1.set_ylim(-ylims,ylims)
    ax1.legend()  # fix the symbols in legends not showing up in pandas plots
    ax2.legend()
    plt.subplots_adjust(wspace=0.08)
    plt.show()


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)
