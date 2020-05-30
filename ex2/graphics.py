from os.path import basename
from statistics import mean
from typing import Mapping
import re
import numpy
from matplotlib import colors
import matplotlib.pyplot as plt
import os
from graphviz import Digraph
import matplotlib.image as mpimg

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


def show_distribution():
    import seaborn as sns
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


def get_scores(path, error=False):
    o = re.compile(r"gen: ([0-9]*) fit: ([0-9\-]*).*")
    mean = re.compile(r".*mean: ([0-9\-]*).*")

    o2 = re.compile(r"generation: ([0-9]*), best score: ([0-9\-]*).*")
    mean2 = re.compile(r".*mean score: ([0-9\-]*).*")

    gens = []
    fits = []
    means = []
    with open(path, 'r') as f:
        for line in f:
            result = o.match(line) or o2.match(line)
            m = mean.match(line) or mean2.match(line)
            if result:
                gen, err = int(result.group(1)), int(result.group(2))

                if error:
                    err = 298 - err if err > 0 else -err
                elif err <= 0:
                    err += 298
                gens.append(gen)
                fits.append(err)
            if m:
                val = int(m.group(1))
                if error:
                    val = 298 - val if val > 0 else -val
                elif val <= 0:
                    val += 298

                means.append(val)

    return gens, fits, means


def show_graph(path, loc='lower right', **kargs):
    title = kargs.get('title', None)
    error = kargs.get('error', False)

    gens, fits, means = get_scores(path, **kargs)
    fig, ax = plt.subplots()
    ax.plot(gens, fits, label='best fitness')
    if means:
        ax.plot(gens, means, label='mean fitness')
    ax.set(xlabel='generations', ylabel='error' if error else 'fitness score',
           title=title if title else 'part b ({})'.format(basename(path)))
    plt.legend(loc=loc)
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


######################################################################


def plot_tree(head, name='result'):
    tree = head.get_nodes()
    repr = {comp: str(i) for i, comp in enumerate(tree)}

    dot = Digraph(comment='GP')
    # add nodes
    for node, rep in repr.items():
        dot.node(rep, node.symbol)
    for node in tree:
        for child in node.children:
            dot.edge(repr[node], repr[child])

    with open("graph_string.dot", 'w') as f:
        f.write(str(dot.source))
    os.system(f"dot -Tpng graph_string.dot -o {name}.png")
    os.remove("graph_string.dot")
    img = mpimg.imread(f'{name}.png')
    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    pass


