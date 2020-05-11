from os.path import basename
from typing import Mapping
import re
import numpy
from matplotlib import colors
import matplotlib.pyplot as plt
from config import OCCUPIED, CELL_COLORS, BRIGHT, DARK

""" Part A"""

CellState = int
Color = str


def create_cmap(cell_colors: Mapping[CellState, Color]):
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
    cmap, norm = create_cmap(CELL_COLORS)

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


""" Part B """


def show_graph(path):
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
                gen, err = int(result.group(1)), -int(result.group(2))
                if err < 0:
                    err += 298
                gens.append(gen)
                fits.append(err)
            if m:
                val = -int(m.group(1))
                val = val + 298 if val < 0 else val
                means.append(val)
    fig, ax = plt.subplots()
    ax.plot(gens, fits)
    if means:
        ax.plot(gens, means)
    ax.set(xlabel='gens', ylabel='error',
           title='part b ({})'.format(basename(path)))
    plt.show()


def part_a():
    ch = [0,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,0]
    board = [[0 for _ in range(8)] for _ in range(8)]
    row = 0
    for i in range(0, len(ch), 3):
        g1, g2, g3 = ch[i:i + 3]
        col = int(''.join(map(str, [g1, g2, g3])), 2)
        board[row][col] = 1
        row += 1
    show_mat(board)


def part_b(name):
    show_graph('part_b_data/{}'.format(name))


if __name__ == '__main__':
    # part_a()
    part_b(name='out1')
