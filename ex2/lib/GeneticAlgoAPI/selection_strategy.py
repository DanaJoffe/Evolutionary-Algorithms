import random
from abc import ABC, abstractmethod
from typing import List
from lib.GeneticAlgoAPI import Chromosome


class SelectionStrategy(ABC):
    @abstractmethod
    def set_selection_pool(self, chromosomes_pool: List[Chromosome]) -> None:
        """  """
        raise NotImplemented

    @abstractmethod
    def select(self) -> Chromosome:
        """ return the selected chromosome """
        raise NotImplemented


class RouletteWheelSelection(SelectionStrategy):
    def __init__(self, *args, **kwargs):
        self.chromosomes_pool = None
        self.roulette = None

    def select(self):
        pick = random.randint(0, len(self.roulette)-1)
        return self.chromosomes_pool[self.roulette[pick]]

    def __force_positive_scores(self, scores):
        """ if there are negative scores - shift all the sores above zero. if all scores are positive
         but some are zero - add 1 """
        if any(val < 0 for val in scores.values()):
            abs_max = max([abs(val) for val in scores.values()])
            for k, v in scores.items():
                scores[k] = v + abs_max + 1
        # make it possible to everyone to be selected
        elif any(val == 0 for val in scores.values()):
            for k, v in scores.items():
                scores[k] = v + 1
        return scores

    def __scale_down(self, scores):
        """ make all scores shift down, s.t. the worst score will be 1.
        assumption: all scores are > 0 """
        worst_score = min([val for val in scores.values()])
        for k, v in scores.items():
            scores[k] = v - worst_score + 1
        return scores

    def set_selection_pool(self, chromosomes_pool):
        chrom_to_fit = {chrom: chrom.get_fitness() for chrom in chromosomes_pool}
        chrom_to_fit = self.__force_positive_scores(chrom_to_fit)
        chrom_to_fit = self.__scale_down(chrom_to_fit)

        self.chromosomes_pool = list(chrom_to_fit.keys())
        self.roulette = []
        for i, chromosome in enumerate(self.chromosomes_pool):
            self.roulette += [i] * chrom_to_fit[chromosome]


class RankSelection(RouletteWheelSelection):
    def set_selection_pool(self, chromosomes_pool):
        scores = list({chrom.get_fitness() for chrom in chromosomes_pool})
        scores.sort()
        new_scores = {score: i+1 for i, score in enumerate(scores)}
        chrom_to_fit = {chrom: new_scores[chrom.get_fitness()] for chrom in chromosomes_pool}
        self.chromosomes_pool = list(chrom_to_fit.keys())
        self.roulette = []
        for i, chromosome in enumerate(self.chromosomes_pool):
            self.roulette += [i] * chrom_to_fit[chromosome]
