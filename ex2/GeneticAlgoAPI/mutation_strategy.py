import random
from abc import ABC

from GeneticAlgoAPI.chromosome import Chromosome, VectorChromosome


class MutationStrategy(ABC):
    mutation_rate = None

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """  """
        raise NotImplemented


class BinaryMutation(MutationStrategy):
    def mutate(self, chromosome: VectorChromosome):
        new_chromo = chromosome.__copy__()
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                new_chromo[i] = 1 - chromosome[i]
        return new_chromo
