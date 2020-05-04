from abc import ABC, abstractmethod
from typing import Mapping
from GeneticAlgoAPI.chromosome import Chromosome
from GeneticAlgoAPI.population import Population


class FitnessFuncBase(ABC):
    @abstractmethod
    def fitness_func(self, population: Population) -> Mapping[Chromosome, int]:
        """
        used by set_fitness_scores to calculate and set a fitness score to all chromosomes.

        notes:
         - best score is highest score.
         - this function knows the true class of the chromosome.
         - individual chromosome fitness can be calculated based on itself or based on all-population tournaments.
        """
        raise NotImplemented


class MistakesBasedFitnessFunc(FitnessFuncBase):
    def fitness_func(self, population):
        # mistakes-based fitness function
        return {chromo: -1 * self.calc_mistakes(chromo) for chromo in population}

    def get_stop_cond(self, population):
        return self.calc_mistakes(population.get_fittest()) == 0

    @abstractmethod
    def calc_mistakes(self, chromosome):
        """ domain dependant """
        raise NotImplemented


class AbsoluteFitness(FitnessFuncBase):
    """  requires max_fitness member """
    max_fitness = None

    def fitness_func(self, population):
        # corrects-based fitness function
        return {chromo: self.calc_corrects(chromo) for chromo in population}

    def get_stop_cond(self, population):
        return self.calc_corrects(population.get_fittest()) == self.max_fitness

    @abstractmethod
    def calc_corrects(self, chromosome):
        raise NotImplemented


