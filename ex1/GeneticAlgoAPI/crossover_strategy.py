import random
from abc import ABC, abstractmethod
from typing import Tuple, Generic, Type, TypeVar
from GeneticAlgoAPI.chromosome import Chromosome


class CrossoverStrategy(ABC):
    @abstractmethod
    def get_parents_amount(self) -> int:
        """ """
        raise NotImplemented

    @abstractmethod
    def get_offsprings_amount(self) -> int:
        """ """
        raise NotImplemented

    @abstractmethod
    def pair_chromosomes(self, chromosomes: Tuple[Chromosome]) -> Tuple[Chromosome]:
        """ do crossover and return offsprings"""
        raise NotImplemented


class TwoParentsTwoChildren(object):
    def get_parents_amount(self):
        return 2

    def get_offsprings_amount(self):
        return 2


class SinglePointCrossover(TwoParentsTwoChildren, CrossoverStrategy):
    # the class knows Chromosome's representation
    # def get_parents_amount(self):
    #     return 2
    #
    # def get_offsprings_amount(self):
    #     return 2

    def pair_chromosomes(self, chromosomes):
        """ do crossover and return offsprings"""
        if len(chromosomes) != 2:
            raise Exception("must contain 2 parents")
        ch1, ch2 = chromosomes

        crossover_point = random.randint(0, len(ch1) - 1)
        offspring1 = ch1.__copy__()
        offspring2 = ch2.__copy__()
        offspring1[crossover_point:] = ch2[crossover_point:]
        offspring2[crossover_point:] = ch1[crossover_point:]
        return offspring1, offspring2


class UniformCrossover(TwoParentsTwoChildren, CrossoverStrategy):
    # the class knows Chromosome's representation
    # def get_parents_amount(self):
    #     return 2
    #
    # def get_offsprings_amount(self):
    #     return 2

    def pair_chromosomes(self, chromosomes):
        """ do crossover and return offsprings"""
        if len(chromosomes) != 2:
            raise Exception("must contain 2 parents")
        ch1, ch2 = chromosomes

        offspring1 = ch1.__copy__()
        offspring2 = ch2.__copy__()
        for i in range(len(ch1)):
            # take gene from other parent with probability 0.5
            if random.random() < 0.5:
                offspring1[i] = ch2[i]
                offspring2[i] = ch1[i]
        return offspring1, offspring2

