import os
from GeneticAlgoAPI.early_convergence_avoidance import KeepAvgFarFromBest, ECA
from GeneticAlgoAPI.fitness_function import MistakesBasedFitnessFunc
from GeneticAlgoAPI.genetic_algorithm import ApplyElitism, GeneticAlgorithm
from GeneticAlgoAPI.selection_strategy import RankSelection
from GeneticProgrammingAPI.gp_crossover import GPCrossover
from GeneticProgrammingAPI.gp_mutation import GPMutation
from graphics import plot_tree
from run_ga import build_and_run, get_time_units
from timeit import default_timer as timer
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from GeneticProgrammingAPI.Component import Variable, Operator, Constant, Condition, PLUS, MULTIPLY, SQUARED, MINUS, \
    SUBTRACT, DIVIDE, IF_ELSE
from GeneticProgrammingAPI.gp_chromosome import GPChromosome


def get_chromo_min_nodes(min_nodes, components):
    chromo = GPChromosome(components, max_depth=5, max_nodes=10)
    while chromo.nodes < min_nodes:
        chromo = GPChromosome(components, max_depth=5, max_nodes=10)
    return chromo


def main():
    components = [Variable('x'), Operator(lambda x,y: x+y, symbol='+'), Constant(6, symbol='6'),
                  Condition(lambda x,y,z: y if x else z, symbol='if else'), Operator(lambda x,y: x*y, symbol='*')]

    chromo1 = get_chromo_min_nodes(2, components)
    chromo2 = get_chromo_min_nodes(2, components)
    # c2 = chromo.__copy__()
    plot_tree(chromo1)
    plot_tree(chromo2)
    c, = GPCrossover().pair_chromosomes((chromo1, chromo2))

    plot_tree(c)

    x = 0


#################################################### The Algorithm ####################################################


"""
questions:
1) how do the algorithm 'learns' the constants? when we decide a node is a constant, we randomly choose the value? from what range? 
2) in crossover - if chromosomes are size 1, what is the action?
3) in crossover - can we concat the whole second parent to the first? (is the head optional?)
4) in mutation - in what scenario we're doing gaussian noise? and can we switch from constant to variable? (both arity=0)
5) in mutation - what is the recommended std for the gaussian noise?
6) constants are always floats? if they can be integers - how the gaussian noise works?
7) in mutation: arity preservation OR arity disruption?

next to implement: mutation

"""

LABEL = 'label'

# dataset = [{'x': i,
#             LABEL: (i*6+6)*6+i*i}
#            for i in range(100)]

dataset = [{'x': x,
            'y': y,
            LABEL: z}
           for x, y, z in [(3, 6, 16), (4, 12, 45), (5, 10, 48), (2, 9, 13.5)]
           ]

components = (Variable('x'), Variable('y'), Constant(range=(4, 8), integer=True),
              PLUS, MULTIPLY, SQUARED, MINUS, SUBTRACT, DIVIDE)
max_depth=5
max_nodes=20


class DomainChromosome(GPChromosome):
    def __init__(self, components=components, max_depth=max_depth, max_nodes=max_nodes, **kwargs):
        super().__init__(components, max_depth=max_depth, max_nodes=max_nodes, **kwargs)


class DomainGP(RankSelection, GPCrossover, GPMutation, ApplyElitism,
                    MistakesBasedFitnessFunc, GeneticAlgorithm):
    def __init__(self, elitism, *args, components=components, **kwargs):
        GeneticAlgorithm.__init__(self, *args, **kwargs)
        self.elitism = elitism
        self.components = components

    @classmethod
    def calc_mistakes(cls, chromosome: DomainChromosome):
        # count mistakes
        # return sum(1 for item in dataset if chromosome.calc(**item) != item[LABEL])
        return sum(abs(chromosome.calc(**item) - item[LABEL]) for item in dataset)


def measure_time(func):
    """ returns running time of func """
    def inner(*args, **kwargs):
        start = timer()
        output = func(*args, **kwargs)
        end = timer()
        return end-start, output
    return inner


def run_algo():
    mutation_rate = .005  # .001
    crossover_rate = .81  # .75
    population_size = 500  # 100
    elitism_count = 2

    eca = ECA()#(mr=.1, cr=1, gen=5, dist_from_avg=1)
    time, chromo, gen = build_and_run(eca, mutation_rate, crossover_rate, population_size, elitism_count, DomainGP,
                                      DomainChromosome)
    time, unit = get_time_units(time)
    print("run for {} {}".format(time, unit))
    plot_tree(chromo.head)
    return time, gen


if __name__ == '__main__':
    run_algo()
    # main()
