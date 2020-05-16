from os.path import basename
from statistics import mean
from typing import Mapping
import re
import numpy
from matplotlib import colors
import matplotlib.pyplot as plt
from config import OCCUPIED, CELL_COLORS, BRIGHT, DARK
import seaborn as sns


""" Part A"""

CellState = int
Color = str


def _create_cmap(cell_colors: Mapping[CellState, Color]):
    """

    :param cell_colors: maps a number (cell state) to a string color.
    :return:
    """
    cell_states = list(cell_colors.keys())
    # map colors
    cols = [color for _, color in sorted(zip(cell_states, [cell_colors[n] for n in cell_states]))]
    nums = sorted(cell_states)
    bound = nums + [nums[-1] + 1]
    stat = cols
    # create colorMap
    cmap = colors.ListedColormap(stat)
    norm = colors.BoundaryNorm(boundaries=bound, ncolors=cmap.N)
    return cmap, norm


def show_mat(mat):
    fig, ax = plt.subplots()
    cmap, norm = _create_cmap(CELL_COLORS)

    board = numpy.ones((len(mat), len(mat))) * DARK
    board[::2, ::2] = BRIGHT
    board[1::2, 1::2] = BRIGHT
    ax.imshow(board, interpolation='nearest', cmap=cmap)
    # set queens
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == OCCUPIED:
                # queen = u'\u2655'
                queen_color = 'black'  # if (i - j) % 2 == 0 else 'white'
                ax.text(j, i,  '♕', size=30, ha='center', va='center', color=queen_color)
    ax.set(xticks=[], yticks=[])
    ax.axis('image')
    plt.show()


def part_a_demonstration():
    ch = [0,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,0]
    board = [[0 for _ in range(8)] for _ in range(8)]
    row = 0
    for i in range(0, len(ch), 3):
        g1, g2, g3 = ch[i:i + 3]
        col = int(''.join(map(str, [g1, g2, g3])), 2)
        board[row][col] = 1
        row += 1
    show_mat(board)


def show_distribution():
    """ show running time distribution of GA runs and brute force runs"""
    def get_vals(path):
        o = re.compile(r"-> run for ([0-9.]*) sec and ([0-9]*) gen.*")
        gens = []
        times = []
        with open(path, 'r') as f:
            for line in f:
                result = o.match(line)
                if result:
                    time, gen = float(result.group(1)), int(result.group(2))
                    gens.append(gen)
                    times.append(time)
            return times, gens
    ga_times, ga_gens = get_vals('part_a_data/GA')
    bf_times, bf_gens = get_vals('part_a_data/brute-force')
    print(mean(ga_times))
    print(mean(bf_times))

    # f, axes = plt.subplots(2, 1)
    # sns.kdeplot(ga_times, shade=True, label="GA", ax=axes[0])
    # sns.kdeplot(bf_times, shade=True, label="brute-force", ax=axes[1])

    sns.kdeplot(ga_times, shade=True, label="GA")
    sns.kdeplot(bf_times, shade=True, label="brute-force")
    # sns.distplot(ga_times, hist=False, rug=True, label="GA", ax=axes[0])
    # sns.distplot(bf_times, hist=False, rug=True, label="brute-force", ax=axes[1])

    plt.legend()
    plt.title('running-time distribution (100 samples)')
    plt.xlabel('time [seconds]')
    plt.ylabel('density')
    plt.show()


""" Part B """


def get_scores(path, title=None, error=False):
    o = re.compile(r"gen: ([0-9]*) fit: ([0-9\-]*).*")
    mean = re.compile(r".*mean: ([0-9\-]*).*")
    gens = []
    fits = []
    means = []
    with open(path, 'r') as f:
        for line in f:
            result = o.match(line)
            m = mean.match(line)
            if result:
                result = o.match(line)
                gen, err = int(result.group(1)), int(result.group(2))

                if error:
                    err = 298 - err if err > 0 else -err
                elif err <= 0:
                    err += 298
                # if err > 0:  # data is in fitness
                #     if error:  # show error
                #         err = 298 - err
                # else:  # data is in error
                #     if error:  # show error
                #         err = -err
                #     else:  # show fitness
                #         err += 298

                # if err < 0:
                #     err += 298
                gens.append(gen)
                fits.append(err)
            if m:
                val = int(m.group(1))
                # val = val + 298 if val < 0 else val
                if error:
                    val = 298 - val if val > 0 else -val
                elif val <= 0:
                    val += 298

                means.append(val)

    return gens, fits, means
    # fig, ax = plt.subplots()
    # ax.plot(gens, fits)
    # if means:
    #     ax.plot(gens, means)
    # ax.set(xlabel='generations', ylabel='error' if error else 'fitness',
    #        title=title if title else 'part b ({})'.format(basename(path)))
    # plt.show()


def show_graph(path, **kargs):
    title = kargs.get('title', None)
    error = kargs.get('error', False)

    gens, fits, means = get_scores(path, **kargs)
    fig, ax = plt.subplots()
    ax.plot(gens, fits, label='best fitness')
    if means:
        ax.plot(gens, means, label='mean fitness')
    ax.set(xlabel='generations', ylabel='error' if error else 'fitness score',
           title=title if title else 'part b ({})'.format(basename(path)))
    plt.legend(loc='lower right')
    plt.show()


def show_improvement():
    fig, ax = plt.subplots()
    ax.set(xlabel='generations', ylabel='fitness score', title='max population fitness (35 runs)')
    outs = range(1,34)
    paths = [f'part_b_data/out{i}' for i in outs]
    for path, i in zip(paths, outs):
        gens, fits, means = get_scores(path)
        ax.plot(gens, fits, label=f'{i}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    # show_graph(path='part_a_data/out2', error=True)

    # part B brute force
    show_distribution()

    # part B tries
    show_improvement()

    # part B mean jump examples
    show_graph(path ='part_b_data/out30', title='max population fitness (pop size=100)')
    show_graph(path ='part_b_data/out34', title='max population fitness (pop size=500)')
