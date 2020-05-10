from abc import ABC, abstractmethod
from typing import Type, TypeVar, List, Union, Iterable
from GeneticAlgoAPI.chromosome import Chromosome
import numpy as np

# class Population(ABC):
#     """ population of Chromosomes """
#
#     # @abstractmethod
#     # def get_fittest(self) -> Type[Chromosome]:
#     #     """ returns the fittest chromosome """
#
#     @abstractmethod
#     def add_chromosome(self, chromo: Type[Chromosome]) -> None:
#         """ add new chromosome to the population """
#
#     @abstractmethod
#     def calc_fitness_scores(self):
#         """ set fitness score to each chromosome in population """


class Population(object):
    """ population of Chromosomes """
    def __init__(self):
        self.population: List[Chromosome] = []

    def get_fittest(self) -> Chromosome:
        """ returns the fittest chromosome """
        fitness = np.array([ch.get_fitness() for ch in self.population])
        return self.population[fitness.argmax()]

    def get_least_fit(self) -> Chromosome:
        """ returns the least fit chromosome """
        fitness = np.array([ch.get_fitness() for ch in self.population])
        return self.population[fitness.argmin()]

    def add_chromosome(self, chromosomes: Union[Chromosome, Iterable[Chromosome]]) -> None:
        """ add new chromosome to the population """
        if isinstance(chromosomes, Chromosome):
            self.population.append(chromosomes)
        else:
            for chromo in chromosomes:
                self.population.append(chromo)

    def remove_chromosome(self, chromo: Chromosome) -> None:
        self.population.remove(chromo)

    def get_size(self):
        return self.population.__len__()

    def __iter__(self):
        return self.population.__iter__()

    def init_population(self, size: int, chromo_type: Type[Chromosome]):
        for _ in range(size):
            self.add_chromosome(chromo_type())

