import random

import numpy

from lib.GeneticAlgoAPI.mutation_strategy import MutationStrategy
from lib.GeneticProgrammingAPI.component import Constant
from lib.GeneticProgrammingAPI.gp_chromosome import GPChromosome


class GPMutation(MutationStrategy):
    components = None
    # todo: choose std for gaussian noise
    _noise_std = 1.0

    # todo: arity preservation VS arity disruption

    def mutate(self, chromosome: GPChromosome):
        new_chromo = chromosome.__copy__()

        for node in new_chromo.genome:
            if random.random() < self.mutation_rate:
                components = [c for c in self.components if c.arity == node.arity]
                chosen = random.choice(components)()
                # todo: modify 'if' statement (choose between random assignment & gaussian noise)
                if (type(node.content) is Constant and type(chosen) is Constant) and random.random() < 0.5:
                    # add gaussian noise
                    chosen.value = node.content.value + numpy.random.normal(scale=self._noise_std)
                node.set_content(chosen)
        return chromosome
