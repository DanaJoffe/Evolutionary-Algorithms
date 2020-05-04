import math
import random
from abc import ABC, abstractmethod
from typing import Mapping, Type, Dict, List, Tuple

import numpy

from GeneticAlgoAPI.chromosome import Chromosome
from GeneticAlgoAPI.crossover_strategy import CrossoverStrategy
from GeneticAlgoAPI.fitness_function import FitnessFuncBase
from GeneticAlgoAPI.mutation_strategy import MutationStrategy
from GeneticAlgoAPI.population import Population
from GeneticAlgoAPI.selection_strategy import SelectionStrategy


class GeneticAlgorithm(SelectionStrategy, CrossoverStrategy, MutationStrategy, FitnessFuncBase, ABC):
    population_size = None
    crossover_rate = None

    max_score = -math.inf

    def set_fitness_scores(self, population: Population):
        scores = self.fitness_func(population)
        for chromo, score in scores.items():
            chromo.set_fitness_score(score)

            # track the fittest score
            if score > self.max_score:
                self.max_score = score

    @abstractmethod
    def get_stop_cond(self, population: Population):
        """ returns True if the algorithm should stop"""
        raise NotImplemented

    def mutation(self, population: Population) -> Population:
        """  """
        new_pop = Population()
        for ch in population:
            new_pop.add_chromosome(self.mutate(ch))
        return new_pop

    def crossover(self, parents: List[Tuple[Chromosome]]) -> Population:
        """
        get list of parents to mate, return the new population

        NOTE: when population (size - elitism amount) doesn't divide in (offspring amount) then the remainder
         offsprings won't enter the new population.
        """
        new_population = Population()
        for chromosomes in parents:
            # do crossover with probability self.crossover_rate
            if random.random() < self.crossover_rate:
                offsprings = self.pair_chromosomes(chromosomes)
            else:
                offsprings = chromosomes
            for ch in offsprings:
                new_population.add_chromosome(ch)
                # if there are too many offsprings then stop adding them to population
                if new_population.get_size() == self.population_size:
                    break
        return new_population

    def selection(self, population: Population) -> List[Tuple[Chromosome]]:
        """
        select parents for crossover

        implements basic selection method: 1) create a pool of chromosomes, 2) select the amount needed for mating.
        """
        parents = []
        pairings = math.ceil(population.get_size() / self.get_offsprings_amount())
        self.set_selection_pool([ch for ch in population])
        for _ in range(pairings):
            parents.append(tuple([self.select() for _ in range(self.get_parents_amount())]))
        return parents


class ApplyElitism(object):
    """ requires self.elitism parameter """
    elitism = None

    def apply_elitism(self, population):
        """
        return best chromosomes and remove worst chromosomes

        NOTE: changing population
        :param population:
        :return:
        """
        elite = []
        if self.elitism > 0:
            chromos, scores = [], []
            for ch in population:
                chromos.append(ch)
                scores.append(ch.get_fitness())
            scores = numpy.array(scores)

            # elitism: add fittest chromosomes to new population, remove worst chromosomes
            for _ in range(self.elitism):
                # add chromosome with best score
                indx = int(numpy.argmax(scores))
                elite.append(chromos[indx])
                # remove chromosome with best score (so it won't be picked again)
                scores = numpy.delete(scores, indx)
                del chromos[indx]
                # remove chromosome with worst score from population
                indx = int(numpy.argmin(scores))
                scores = numpy.delete(scores, indx)
                population.remove_chromosome(chromos[indx])
                del chromos[indx]
        return elite


def positive(fitness_func):
    """ force a fitness_func to return positive scores, larger then 0"""
    def inner(*args, **kwargs):
        output = fitness_func(*args, **kwargs)
        if any(val <= 0 for val in output.values()):
            abs_max = max([abs(val) for val in output.values()])
            for k, v in output.items():
                output[k] = v + abs_max + 1
        return output
    return inner


