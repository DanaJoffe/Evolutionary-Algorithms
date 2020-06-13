import math
import operator
import random

import graphviz
import numpy
from deap import base, creator, gp
from deap import base
from deap import tools
from deap import algorithms

from globals import titles, operators, load_data
from lib.GeneticProgrammingAPI import plot_tree

features = len(titles)
data = load_data()

# y = df.iloc[:, 0]
# X = df.iloc[:, 1:]
# for label, (i, row) in zip(y, X.iterrows()):

    # row = row.tolist()


pset = gp.PrimitiveSet("MAIN", features)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)


# def protectedDiv(left, right):
#     try:
#         return left / right
#     except ZeroDivisionError:
#         return 1
# pset = gp.PrimitiveSet("MAIN", 1)
# pset.addPrimitive(operator.add, 2)
# pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(operator.mul, 2)
# pset.addPrimitive(protectedDiv, 2)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

kwargs = {f'ARG{i}': t for i, t in zip(range(features), titles)}
pset.renameArguments(**kwargs)

# creator.create("Individual", gp.PrimitiveTree)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)  # , pset=pset)

toolbox = base.Toolbox()
# how to create an individual
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# how to create the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x

    df = data.sample(n=500) # frac=0.2
    # df = data
    points = df.iterrows()
    # real_func = lambda x: x**4 + x**3 + x**2 + x
    # real_func = lambda x: x * (x+1)


    # sqerrors = ((func(x) - real_func(x)) **2 for x in points)
    sqerrors = ((func( *(row[1:].tolist()) ) - row[0]) **2 for (i, row) in points)
    return math.fsum(sqerrors) / features,


# todo: change input
toolbox.register("evaluate", evalSymbReg, points=None)#[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
# CROSSOVER: one point crossover with uniform probability over all the nodes
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# MUTATION: a uniform probability mutation which may append a new full sub-tree to a node
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# in order to avoid bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
# stats_size = tools.Statistics(len)
# mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# mstats.register("avg", numpy.mean)
# mstats.register("std", numpy.std)
# mstats.register("min", numpy.min)
# mstats.register("max", numpy.max)
#
# pop = toolbox.population(n=300)
# hof = tools.HallOfFame(1)
# pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
#                                halloffame=hof, verbose=True)


def create_dot(edges, labels, operators=operators):
    from graphviz import Digraph
    dot = Digraph(comment='GP')
    # add nodes
    for rep, sym in labels.items():
        sym = operators.get(sym, sym)
        dot.node(str(rep), str(sym))
    for ed in edges:
        # for child in node.children:
        dot.edge(str(ed[0]), str(ed[1]))
    return str(dot.source)


def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    cr = 0.5
    mr = 0.1
    gens = 2
    # print log
    pop, log = algorithms.eaSimple(pop, toolbox, cr, mr, gens, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(hof.items[0])
    # expr = toolbox.individual()
    nodes, edges, labels = gp.graph(hof.items[0])
    dot = create_dot(edges, labels)
    plot_tree(dot=dot)

    return pop, log, hof


if __name__ == "__main__":
    main()


# toolbox = base.Toolbox()
# toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
# # toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
#
#
# # expr = toolbox.individual()
# # nodes, edges, labels = gp.graph(expr)
#
#
# expr = gp.genFull(pset, min_=1, max_=3)
# tree = gp.PrimitiveTree(expr)
# print(str(tree))
#
# function = gp.compile(tree, pset)
# # function(1, 2)
#
# import matplotlib.pyplot as plt
# import networkx as nx #graphviz_layout
# from networkx import graph

# [...] Execution of code that produce a tree expression


# g = nx.Graph()
# g.add_nodes_from(nodes)
# g.add_edges_from(edges)
# pos = nx.graphviz_layout(g, prog="dot")
#
# nx.draw_networkx_nodes(g, pos)
# nx.draw_networkx_edges(g, pos)
# nx.draw_networkx_labels(g, pos, labels)
# plt.show()


### Graphviz Section ###
# import pygraphviz as pgv
#
# g = pgv.AGraph()
# g.add_nodes_from(nodes)
# g.add_edges_from(edges)
# g.layout(prog="dot")
#
# for i in nodes:
#     n = g.get_node(i)
#     n.attr["label"] = labels[i]
#
# g.draw("tree.pdf")















