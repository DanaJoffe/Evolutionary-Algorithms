import random
from abc import ABC
from math import log, ceil
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

# sentence = "adriana: ay, ay, antipholus, look strange and frown." \
#            "some other mistress hath thy sweet aspects;" \
#            "i am not adriana, nor thy wife."

# sentence = "i love amir"
# sentence = "li"
sentence = "to be or not to be that is the question. " \
           "whether tis nobler in the mind to suffer. " \
           "the slings and arrows of outrageous fortune. " \
           "or to take arms against a sea of troubles and by opposing end them. " \
           "to die to sleep. " \
           "no more. and by a sleep to say we end. " \
           "the heartache and the thousand natural shocks."

english_ab_size = 26
punctuation_ab = [' ', '.']
ab_size = len(punctuation_ab) + english_ab_size
bits_per_char = ceil(log(ab_size, 2))
chromo_size = len(sentence) * bits_per_char


def binary_english_dict(binary_num):
    """
    0-25: english small letters. ('a'=97)
    26-27: punctuation

    """
    binary_num = binary_num % ab_size
    punctuation = {num: char for num, char in zip(list(range(26, 26+len(punctuation_ab))), punctuation_ab)}
    if binary_num < 26:
        return chr(binary_num + 97)
    return punctuation[binary_num]  # .get(binary_num, '$')


class ShakespeareChromosome(IntChromosome):
    def __init__(self):
        super().__init__(chromo_size)  # len(sentence))  # chromo_size = 300 * 5 = 1500 bits
        # int8 = 8 bits, one letter
        # 28 chars, 5 bits (32 options). 8 bits = 255 options

    def __str__(self):
        mask = 2 ** bits_per_char - 1
        s = ''
        for i in range(len(sentence)):
            s += binary_english_dict(((mask << bits_per_char * i) & self.genome) >> bits_per_char * i)
        return s

        # chromo_sentence = [binary_english_dict(((mask << bits_per_char * i) & self.genome) >> bits_per_char * i)
        #                    for i in range(len(sentence))]
        # return ''.join(self.to_sentence())


class CustomTextMutation(MutationStrategy):
    def mutate(self, chromosome: Chromosome):
        new_chromo = chromosome.__copy__()
        for w in range(len(sentence)):
            if random.random() < self.mutation_rate:
                pos = random.randint(0, bits_per_char)
                i = w * bits_per_char + pos
                new_chromo[i] = 1 - chromosome[i]
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


class ShakespeareGA(RouletteWheelSelection, CustomTextUniformCrossover, CustomTextMutation, ApplyElitism,
                    MistakesBasedFitnessFunc, GeneticAlgorithm):
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
               "elitism={}\nmutation_rate={}\ncrossover_rate={}\n population_size={}".format(self.elitism,
                                                                                             self.mutation_rate,
                                                                                             self.crossover_rate,
                                                                                             self.population_size)

    def calc_mistakes(self, chromosome):
        chromo_sentence = str(chromosome)
        return sum([1 if l1 != l2 else 0 for l1, l2 in zip(chromo_sentence, sentence)])

    # def calc_corrects(self, chromosome):
    #     chromo_sentence = str(chromosome)
    #     return sum([1 if l1 == l2 else 0 for l1, l2 in zip(chromo_sentence, sentence)])


def run(ga, population):
    ga.set_fitness_scores(population)
    gen = 0
    evaluate(population, gen, ga)
    while not ga.get_stop_cond(population):
        gen += 1
        elite = ga.apply_elitism(population)
        parents = ga.selection(population)
        population = ga.crossover(parents)
        population = ga.mutation(population)
        population.add_chromosome(elite)
        ga.set_fitness_scores(population)

        # v = {str(ch) for ch in population}
        # if len(v) < 0.5 * population.get_size():
        # if ga.max_score > -50:
            # ga.mutation_rate = .05
            # ga.crossover_rate = .8
        # else:
        #     ga.mutation_rate = .01
        #     ga.crossover_rate = .75

        # gen: 1089 fit: -98

        evaluate(population, gen, ga)
    return population.get_fittest()


def main():
    mutation_rate = .008
    crossover_rate = .75
    population_size = 90
    elitism_count = 6

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

    ga = ShakespeareGA(elitism_count, mutation_rate, crossover_rate, population_size)
    population = Population()
    population.init_population(population_size, ShakespeareChromosome)

    print(ga)

    start = timer()
    chromo = run(ga, population)
    end = timer()
    time, unit = get_time_units(end - start)

    print("run for {} {}".format(time, unit))
    print(chromo)


if __name__ == '__main__':
    main()
