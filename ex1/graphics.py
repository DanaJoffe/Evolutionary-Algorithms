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
    o = re.compile(r"gen: ([0-9]*) fit: ([0-9\-]*)")
    gens = []
    fits = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('gen'):
                result = o.match(line)
                gen, err = int(result.group(1)), -int(result.group(2))
                if err < 0:
                    err += 300
                gens.append(gen)
                fits.append(err)
    fig, ax = plt.subplots()
    ax.plot(gens, fits)
    ax.set(xlabel='gens', ylabel='error',
           title='part b ({})'.format(basename(path)))
    plt.show()


# def t1():
#     size = 8
#     chessboard = numpy.zeros((size, size))
#
#     chessboard[1::2, 0::2] = 1
#     chessboard[0::2, 1::2] = 1
#
#     plt.imshow(chessboard, cmap='binary')
#
#     for _ in range(20):
#         i, j = numpy.random.randint(0, 8, 2)
#         plt.text(i, j, '♕', fontsize=20, ha='center', va='center', color='black' if (i - j) % 2 == 0 else 'white')
#
#     plt.show()
#
#
# def t2():
#     board = numpy.zeros((8, 8, 3))
#     board += 0.5  # "Black" color. Can also be a sequence of r,g,b with values 0-1.
#     board[::2, ::2] = 1  # "White" color
#     board[1::2, 1::2] = 1  # "White" color
#
#     positions = [1, 5, 7, 2, 0, 3, 6, 4]
#
#     fig, ax = plt.subplots()
#     ax.imshow(board, interpolation='nearest')
#
#     for y, x in enumerate(positions):
#         # Use "family='font name'" to change the font
#         ax.text(x, y, u'\u2655', size=30, ha='center', va='center')
#
#     ax.set(xticks=[], yticks=[])
#     ax.axis('image')
#
#     plt.show()


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


def part_b(name='out1'):
    show_graph('part_b_data/{}'.format(name))


if __name__ == '__main__':
    # part_a()
    part_b(name='out8')



