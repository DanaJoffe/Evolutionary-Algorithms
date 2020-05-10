import math
import random
from abc import ABC
from math import log, ceil
from statistics import mean
from timeit import default_timer as timer
from GeneticAlgoAPI.chromosome import ListChromosomeBase, IntChromosome, Chromosome
from GeneticAlgoAPI.crossover_strategy import SinglePointCrossover, UniformCrossover, TwoParentsTwoChildren, \
    CrossoverStrategy
from GeneticAlgoAPI.fitness_function import MistakesBasedFitnessFunc, AbsoluteFitness
from GeneticAlgoAPI.genetic_algorithm import GeneticAlgorithm, ApplyElitism
from GeneticAlgoAPI.mutation_strategy import BinaryMutation, MutationStrategy
from GeneticAlgoAPI.population import Population
from GeneticAlgoAPI.selection_strategy import RouletteWheelSelection
from config import MUTATION_RATE, CROSSOVER_RATE, POPULATION_SIZE, ELITISM
from run_ga import build_and_run, get_time_units, evaluate

""" Recreate 3 Shakespeare's sentences """

sentence = "to be or not to be that is the question. " \
           "whether tis nobler in the mind to suffer. " \
           "the slings and arrows of outrageous fortune. " \
           "or to take arms against a sea of troubles and by opposing end them. " \
           "to die to sleep. " \
           "no more. and by a sleep to say we end. " \
           "the heartache and the thousand natural shocks."

ab = "abcdefghijklmnopqrstuvwxyz ."
ab_size = len(ab)
bits_per_char = ceil(log(ab_size, 2))
chromo_size = len(sentence) * bits_per_char


def binary_english_dict(binary_num):
    assert 0 <= binary_num < ab_size
    return ab[binary_num]


class ShakespeareChromosome(IntChromosome):
    def __init__(self, **kargs):
        super().__init__(chromo_size, **kargs)

        # if this isn't a copy but a new random chromosome
        if 'genome' not in kargs:
            # choose random letter
            for w in range(len(sentence)):
                i = w * bits_per_char
                self[i:i+bits_per_char] = random.randint(0, ab_size - 1)

    def __str__(self):
        mask = 2 ** bits_per_char - 1
        s = ''
        for i in range(len(sentence)):
            # j = bits_per_char * i
            # s += binary_english_dict(self[j:j + bits_per_char])
            s += binary_english_dict(((mask << bits_per_char * i) & self.genome) >> bits_per_char * i)
        return s

        # chromo_sentence = [binary_english_dict(((mask << bits_per_char * i) & self.genome) >> bits_per_char * i)
        #                    for i in range(len(sentence))]
        # return ''.join(self.to_sentence())


class CustomTextBinaryMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome):
        new_chromo = chromosome.__copy__()
        for w in range(len(sentence)):
            if random.random() < self.mutation_rate:
                pos = random.randint(0, bits_per_char)
                i = w * bits_per_char + pos
                new_chromo[i] = 1 - chromosome[i]
        return new_chromo


class CustomTextLetterMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome):
        new_chromo = chromosome.__copy__()
        for w in range(len(sentence)):
            if random.random() < self.mutation_rate:
                start = w * bits_per_char
                end = start + bits_per_char
                # raffle a new number (letter)
                new_letter = random.randint(0, ab_size - 1)
                new_chromo[start:end] = new_letter
        return new_chromo


class CustomTextUniformCrossover(TwoParentsTwoChildren, CrossoverStrategy):
    def pair_chromosomes(self, chromosomes):
        """ do crossover and return offsprings"""
        if len(chromosomes) != 2:
            raise Exception("must contain 2 parents")
        ch1, ch2 = chromosomes

        offspring1 = ch1.__copy__()
        offspring2 = ch2.__copy__()
        for w in range(len(sentence)):
            # take gene from other parent with probability 0.5
            if random.random() < 0.5:
                # pos = random.randint(0, bits_per_char)
                s = w * bits_per_char
                e = s + bits_per_char
                offspring1[s:e] = ch2[s:e]
                offspring2[s:e] = ch1[s:e]
        return offspring1, offspring2


class ShakespeareGA(RouletteWheelSelection, CustomTextUniformCrossover, CustomTextLetterMutation, ApplyElitism,
                    AbsoluteFitness, GeneticAlgorithm):
    def __init__(self, elitism=ELITISM,
                 mutation_rate=MUTATION_RATE,
                 crossover_rate=CROSSOVER_RATE,
                 population_size=POPULATION_SIZE):
        super().__init__()
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.max_fitness = len(sentence)

    def __str__(self):
        # import inspect
        return str(ShakespeareGA.__mro__) + '\n' + \
               "elitism={}\nmutation_rate={}\ncrossover_rate={}\npopulation_size={}".format(self.elitism,
                                                                                             self.mutation_rate,
                                                                                             self.crossover_rate,
                                                                                             self.population_size) \
               + "\nadd scale down to selection"

    def calc_corrects(self, chromosome):
        chromo_sentence = str(chromosome)
        return sum([1 if l1 == l2 else 0 for l1, l2 in zip(chromo_sentence, sentence)])


def run(ga, population):
    original_mr = ga.mutation_rate
    original_cr = ga.crossover_rate
    stop_extra_mutate = 0
    ga.set_fitness_scores(population)
    gen = 0
    evaluate(population, gen, ga)
    while not ga.get_stop_cond(population):
        gen += 1
        if gen > stop_extra_mutate:
            ga.mutation_rate = original_mr
            ga.crossover_rate = original_cr

        elite = ga.apply_elitism(population)
        parents = ga.selection(population)
        population = ga.crossover(parents, population.get_size())
        population = ga.mutation(population)
        population.add_chromosome(elite)
        ga.set_fitness_scores(population)

        evaluate(population, gen, ga)

        # deal with early convergence
        f = population.get_fittest().get_fitness()
        m = mean(ch.get_fitness() for ch in population)
        if f - m < 1:
            ga.mutation_rate = .08
            ga.crossover_rate = 1
            stop_extra_mutate = gen + 1

        # if early convergence is found - increase mutation rate
        # w = population.get_least_fit().get_fitness()
        # b = population.get_fittest().get_fitness()
        # if population.get_least_fit().get_fitness() >= population.get_fittest().get_fitness() - 2:  ##run for 6.82 minutes , gen: 3844
    return population.get_fittest(), gen


def main():
    mutation_rate = .001
    crossover_rate = .75
    population_size = 100
    elitism_count = 8

    """
    goal: finish in 10 minutes.
    - 10 minutes run = 2.5 minutes to get 80% = 240,
                       7.5 minutes to get 20% = 60
    assumptions:
    - 80% of the text is found at the first 1/4 of the time.
    - 20% of the text is found at the last 3/4 of the time.
    
    indications for a good run:
    - fit 150 in ~800 gens
    
    previous results:
    - fit 200 in 2460 gens (10 gens = 2.5 sec) => out3
    - fit 240 in 3000 gens (10 gens = 2 sec) => out5
    - fit 240 in 1726 gens => 5.7 minutes (10 gens ~ 2 sec) => out6
 
    """

    """
    1) (GOOD) now - running the old BinaryMutation method. save result.             (out11)
    2) (NOT GOOD) run. config: selection changed, chromosomes compared by address.  (out12) -> makes selection take longer.
    3) (VERY BAD) (a) change selection based on genomes, (b) change __str__ in GA, (c) run. -> 13,000 gens and no answer
    4) => smaller population

    """

    ga = ShakespeareGA(elitism_count, mutation_rate, crossover_rate, population_size)
    population = Population()
    population.init_population(population_size, ShakespeareChromosome)

    print(ga)

    start = timer()
    chromo, gen = run(ga, population)
    end = timer()
    time, unit = get_time_units(end - start)

    print("run for {} {}, {} gens".format(time, unit, gen))


if __name__ == '__main__':
    # for _ in range(10):
    #     print(random.randint(0, 3))

    main()




