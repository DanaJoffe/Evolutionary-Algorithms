import math
from abc import ABC, abstractmethod
from statistics import mean

from GeneticAlgoAPI.genetic_algorithm import GeneticAlgorithm
from GeneticAlgoAPI.population import Population


class ECA(object):
    """
    Early Convergence Avoidance

    default implementation: do nothing (don't avoid early convergence)
    """
    def before_start(self, ga: GeneticAlgorithm):
        """ called before the algorithm starts running """

    def start_generation(self, gen: int, ga:GeneticAlgorithm, population: Population):
        """ called with every generation that starts """

    def end_generation(self, gen: int, ga:GeneticAlgorithm, population: Population):
        """ called with every generation that ends """


class AddDisruption(ECA):
    def __init__(self, mr=.1, cr=1, gen=1):
        """
        when early convergence is detected, change mutation rate & crossover rate to higher values for limited time.

        :param mr: new mutation rate when adding noise.
        :param cr: new crossover rate when adding noise.
        :param gen: number of generations to apply noise addition.
        """
        # the generation in which the disruption should stop
        self.stop_disruption = 0
        self.original_mr = None
        self.original_cr = None
        self.increased_mutation_rate = mr
        self.increased_crossover_rate = cr
        self.apply_to_generations = gen

    def __str__(self):
        return f"\nAddDisruption:" \
               f"increased_mutation_rate: {self.increased_mutation_rate} " \
               f"increased_crossover_rate: {self.increased_crossover_rate} " \
               f"apply_to_generations: {self.apply_to_generations} "

    def before_start(self, ga: GeneticAlgorithm):
        """ save algorithm's parameters """
        self.original_mr = ga.mutation_rate
        self.original_cr = ga.crossover_rate

    def start_generation(self, gen: int, ga: GeneticAlgorithm, population: Population):
        """ check if disruption should be stopped """
        if gen > self.stop_disruption:
            ga.mutation_rate = self.original_mr
            ga.crossover_rate = self.original_cr

    def end_generation(self, gen: int, ga: GeneticAlgorithm, population: Population):
        """ check if early convergence is found and disruption (high rates of crossover & mutation) should be added """
        if self.check_for_ec(gen, ga, population):
            ga.mutation_rate = self.increased_mutation_rate
            ga.crossover_rate = self.increased_crossover_rate
            self.stop_disruption = gen + self.apply_to_generations

    @abstractmethod
    def check_for_ec(self, gen: int, ga: GeneticAlgorithm, population: Population):
        """ returns True if early convergence was found and disruption should be added """
        raise NotImplemented


class KeepAvgFarFromBest(AddDisruption):
    def __init__(self, dist_from_avg=1, **kargs):
        super().__init__(**kargs)
        self.dist_from_avg = dist_from_avg

    def __str__(self):
        return super().__str__() + f"\nKeepAvgFarFromBest: dist_from_avg: {self.dist_from_avg}"

    def check_for_ec(self, gen: int, ga:GeneticAlgorithm, population: Population):
        f = population.get_fittest().get_fitness()
        m = mean(ch.get_fitness() for ch in population)
        return f - m < self.dist_from_avg


class GraduallyIncrease(KeepAvgFarFromBest):
    def __init__(self, inc_mr_factor=0.0001, inc_cr_factor=0, steps=50, max_mr=math.inf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mr_factor = inc_mr_factor
        self.cr_factor = inc_cr_factor
        self.steps = steps
        self.max_mr = max_mr

    def start_generation(self, gen: int, ga: GeneticAlgorithm, population: Population):
        super().start_generation(gen, ga, population)  # GraduallyIncrease, self
        if gen % self.steps == 0:
            self.original_mr = min(self.original_mr + self.mr_factor, self.max_mr)
            self.original_cr += self.cr_factor

    def __str__(self):
        return f"\nGraduallyIncrease:" \
               f"mr_factor: {self.mr_factor} " \
               f"cr_factor: {self.cr_factor} " \
               f"max_mr: {self.max_mr} " \
               f"steps: {self.steps} "


class KeepWorstFarFromBest(AddDisruption):
    def __init__(self, dist=1, **kargs):
        super().__init__(**kargs)
        self.dist = dist

    def __str__(self):
        return super().__str__() + f"\nKeepWorstFarFromBest: dist: {self.dist}"

    def check_for_ec(self, gen: int, ga:GeneticAlgorithm, population: Population):
        w = population.get_least_fit().get_fitness()
        b = population.get_fittest().get_fitness()
        return b - w < self.dist


# todo: re-write this function
class VarianceDisp(AddDisruption):
    def __init__(self, dist = 5, **kargs):#less then 5 different chromosomes in the population indicates EC
        super().__init__(**kargs)
        self.dist = dist

    def __str__(self):
        return super().__str__() + f"\nVarianceDisp: dist: {self.dist}"

    def check_for_ec(self, gen: int, ga: GeneticAlgorithm, population: Population):
        counterdiff = 0
        diffchromosomes = []
        for chromosome in population:
            if not (diffchromosomes.__contains__(chromosome)):
                diffchromosomes.append(chromosome)
                counterdiff += 1
        return counterdiff < self.dist
