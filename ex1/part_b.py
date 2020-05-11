import random
from math import log, ceil
from typing import Tuple
from GeneticAlgoAPI.chromosome import IntChromosome, Chromosome
from GeneticAlgoAPI.crossover_strategy import SinglePointCrossover, UniformCrossover, TwoParentsTwoChildren, \
    CrossoverStrategy
from GeneticAlgoAPI.early_convergence_avoidance import KeepAvgFarFromBest
from GeneticAlgoAPI.fitness_function import AbsoluteFitness
from GeneticAlgoAPI.genetic_algorithm import GeneticAlgorithm, ApplyElitism
from GeneticAlgoAPI.mutation_strategy import BinaryMutation, MutationStrategy
from GeneticAlgoAPI.population import Population
from GeneticAlgoAPI.selection_strategy import RouletteWheelSelection, RankSelection
from run_ga import build_and_run, get_time_units

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
            s += binary_english_dict(((mask << bits_per_char * i) & self.genome) >> bits_per_char * i)
        return s


class CustomTextBinaryMutation(MutationStrategy):
    """ go over the letters and change one random bit in every letter """
    def mutate(self, chromosome: Chromosome):
        new_chromo = chromosome.__copy__()
        for w in range(len(sentence)):
            if random.random() < self.mutation_rate:
                pos = random.randint(0, bits_per_char)
                i = w * bits_per_char + pos
                new_chromo[i] = 1 - chromosome[i]
        return new_chromo


class CustomTextLetterMutation(MutationStrategy):
    """ go over the letters and choose a random letter to replace the current one """
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
    """ choose a parent from which to take each letter. don't break the existing letters """
    def pair_chromosomes(self, chromosomes: Tuple):
        """ do crossover and return offsprings"""
        if len(chromosomes) != 2:
            raise Exception("must contain 2 parents")
        ch1, ch2 = chromosomes

        offspring1 = ch1.__copy__()
        offspring2 = ch2.__copy__()
        for w in range(len(sentence)):
            # take letter from other parent with probability 0.5
            if random.random() < 0.5:
                s = w * bits_per_char
                e = s + bits_per_char
                offspring1[s:e] = ch2[s:e]
                offspring2[s:e] = ch1[s:e]
        return offspring1, offspring2


class ShakespeareGA(RankSelection, CustomTextUniformCrossover, CustomTextLetterMutation, ApplyElitism,
                    AbsoluteFitness, GeneticAlgorithm):
    def __init__(self, elitism, *args, **kwargs):
        GeneticAlgorithm.__init__(self, *args, **kwargs)
        self.elitism = elitism
        self.max_fitness = len(sentence)

    def __str__(self):
        return str(ShakespeareGA.__mro__) + '\n' + \
               "elitism={}\nmutation_rate={}\ncrossover_rate={}\npopulation_size={}".format(self.elitism,
                                                                                             self.mutation_rate,
                                                                                             self.crossover_rate,
                                                                                             self.population_size) \
               + "\nadd scale down to selection"

    def calc_corrects(self, chromosome):
        chromo_sentence = str(chromosome)
        return sum([1 if l1 == l2 else 0 for l1, l2 in zip(chromo_sentence, sentence)])


def main():
    mutation_rate = .001
    crossover_rate = .75
    population_size = 500
    elitism_count = 2

    # early convergence avoidance mechanism
    eca = KeepAvgFarFromBest(mr=.08, cr=1, gen=1, dist_from_avg=1)
    time, chromo, gen = build_and_run(eca, mutation_rate, crossover_rate, population_size, elitism_count,
                                      ShakespeareGA, ShakespeareChromosome)
    time, unit = get_time_units(time)
    print("run for {} {}, {} gens".format(time, unit, gen))


if __name__ == '__main__':
    main()

    # x =\
    # "run for 4.6651660716666665 minutes, 3716 gens\n"\
    # "run for 3.746307553333333 minutes, 2652 gens\n"\
    # "run for 3.883907578333333 minutes, 2773 gens\n"\
    # "run for 4.8536751166666665 minutes, 3483 gens\n"\
    # "run for 8.694523064999998 minutes, 4948 gens\n"\
    # "run for 4.197982960000002 minutes, 3177 gens\n"\
    # "run for 3.1772619033333322 minutes, 2660 gens\n"\
    # "run for 5.006165533333334 minutes, 3930 gens\n"\
    # "run for 2.9011833766666695 minutes, 2160 gens\n"\
    # "run for 4.604555205000005 minutes, 3344 gens"

    # print(x)

    # minuts = []
    # gens = []
    # for line in x.split('\n'):
    #     parts = line.split(' ')
    #     try:
    #         minuts.append(float(parts[2]))
    #         gens.append(int(parts[4]))
    #     except:
    #         gens.append(int(parts[1]))
    # print("avg time: {} minutes, {} generations".format(mean(minuts), mean(gens)))



""""
-> best & worst -2, +2 gens:
    gen: 4169 fit: 298 mean: 296.29 chromo: to be or not to be that is the question. whether tis nobler in the mind to suffer. the slings and arrows of outrageous fortune. or to take arms against a sea of troubles and by opposing end them. to die to sleep. no more. and by a sleep to say we end. the heartache and the thousand natural shocks.
    run for 7.224326696666667 minutes
"""

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

