import random
from GeneticAlgoAPI.chromosome import Chromosome, ListChromosomeBase
from GeneticAlgoAPI.crossover_strategy import CrossoverStrategy, SinglePointCrossover
from GeneticAlgoAPI.genetic_algorithm import GeneticAlgorithm, ApplyElitism
from GeneticAlgoAPI.mutation_strategy import MutationStrategy, BinaryMutation
from GeneticAlgoAPI.population import Population
import numpy as np

from GeneticAlgoAPI.selection_strategy import SelectionStrategy, RouletteWheelSelection
from config import ELITISM, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE
from run_ga import run


class myChromo(ListChromosomeBase):
    def __init__(self):
        super().__init__(10)
        # if representation:
        #     self.genome = representation
        #     # todo: change distribution to numpy.random.uniform
        #
        #     # binary rep: bin(x)
        #     # representation: bit string of length 10
        #     length = 10
        #     num = random.randint(0, 2 ** length - 1)
        #     bit_string = bin(num)[2:]
        #     bit_string = '0' * (length - len(bit_string)) + bit_string
        #     bit_list = [int(num) for num in list(bit_string)]
        #     self.genome = bit_list
        # else:
        #     self.genome = representation

        # x = 0  # empty
        # x |= 1 << 19  # set bit 19
        # x &= ~(1 << 19)  # clear bit 19
        # x ^= 1 << 19  # toggle bit 19
        # x = ~x  # invert *all* bits, all the way to infinity
        # mask = ((1 << 20) - 1)  # define a 20 bit wide mask
        # x &= mask  # ensure bits 20 and higher are 0
        # x ^= mask  # invert only bits 0 through 19
        #
        # (x >> 19) & 1  # test bit 19
        # (x >> 16) & 0xf  # get bits 16 through 20.



    # def __getitem__(self, k):
    #     return self.genome.__getitem__(k)
    #
    # def __setitem__(self, key, value):
    #     self.genome.__setitem__(key, value)
    #
    # def __len__(self):
    #     return self.genome.__len__()

    # def set_fitness_score(self, score):
    #     self.fitness = score

    # def get_fitness(self):
    #     return self.fitness


# inheritance order is important!!
class myGA(RouletteWheelSelection, SinglePointCrossover, BinaryMutation, ApplyElitism, GeneticAlgorithm):
    def __init__(self, elitism=ELITISM, mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                 population_size=POPULATION_SIZE):
        super().__init__()
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size

        # @positive
    def fitness_func(self, population):
        """

        :param population:
        :return: integer
        """
        # mistakes-based fitness function
        return {chromo: -1 * self._calc_mistakes(chromo) for chromo in population}

        # goal: bit vector of 11111.....
        # return {chromo: -1 * sum(chromo) for chromo in population}

        # goal: bit vector of 10101010... or 010101010...
        # return {ch: sum([abs(gene1 - gene2) for gene1, gene2 in zip(ch, ch[1:])])
        #         for ch in population}

        # goal: 1111100000
        # goal_vec = [1,1,1,1,1,0,0,0,0,0]
        # return {ch: sum([1 if gene == goal_gene else 0 for gene, goal_gene in zip(ch, goal_vec)])
        #         for ch in population}

    def get_stop_cond(self, population):
        # MAX_FITTNESS = 10  # 10
        # return f.genome == [0,0,0,0,0,0,0,0,0,0] # f.get_fitness() == MAX_FITTNESS#
        # f = population.get_fittest()
        return self._calc_mistakes(population.get_fittest()) == 0

    def _calc_mistakes(self, chromosome):
        # goal = [1,1,1,1,1,0,0,0,0,0]
        # return sum([1 for gene1, gene2 in zip(chromosome, goal) if gene1 != gene2])
        return sum([1 - gene for gene in chromosome])

        # return sum([1 for gene1, gene2 in zip(chromosome, chromosome[1:]) if gene1 == gene2])

# def evaluate(population, gen):
#     fittest = population.get_fittest()
#     f = fittest.get_fitness()
#     print("gen: {} fit: {} chromo: {}".format(str(gen), f, str(fittest)))


# def run_ga(ga, population):
#     ga.set_fitness_scores(population)
#     gen = 0
#     evaluate(population, gen)
#     while not ga.get_stop_cond(population):  # f < MAX_FITTNESS:
#         gen += 1
#         elite = ga.apply_elitism(population)
#         parents = ga.selection(population)
#         population = ga.crossover(parents)
#         population = ga.mutation(population)
#         population.add_chromosome(elite)
#         ga.set_fitness_scores(population)
#
#         evaluate(population, gen)


def main():
    # parameters
    mutation_rate = MUTATION_RATE
    crossover_rate = CROSSOVER_RATE
    population_size = POPULATION_SIZE
    elitism_count = ELITISM

    ga = myGA(elitism_count, mutation_rate, crossover_rate, population_size)
    population = Population()
    population.init_population(population_size, myChromo)

    (time, unit), ch = run(ga, population)
    print("run for {} {}".format(time, unit))


if __name__ == '__main__':
    main()

    """
    FLOW:
    n = population size
    - set initialize population size n
    - set Fitness Function
     do:
    - Selection protocol: choose parents (k vectors)
    - Crossover method: how to mate k parents together and yield m offsprings
    - Mutation method: takes 1 chromosome return 1 chromosome 
    
    NOTES:
    - GA algorithm doesn't know the domain!!!!
    - who knows the domain: Chromosome, crossover, mutation 
    """





