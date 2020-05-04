import random
from abc import ABC, abstractmethod
from typing import List
from GeneticAlgoAPI.chromosome import Chromosome


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
    def __init__(self):
        self.chromosomes_pool = None
        self.roulette = None

    def select(self):
        pick = random.randint(0, len(self.roulette)-1)
        return self.chromosomes_pool[self.roulette[pick]]

    def __force_positive_scores(self, scores):
        if any(val <= 0 for val in scores.values()):  # todo: think if it should be < or <=
            abs_max = max([abs(val) for val in scores.values()])
            for k, v in scores.items():
                scores[k] = v + abs_max + 1

        # abs_max = max([abs(val) for val in scores.values()])
        # for k, v in scores.items():
        #     scores[k] = abs_max + v
        return scores

    def set_selection_pool(self, chromosomes_pool):
        chrom_to_fit = {chrom: chrom.get_fitness() for chrom in chromosomes_pool}
        chrom_to_fit = self.__force_positive_scores(chrom_to_fit)

        self.chromosomes_pool = list(chrom_to_fit.keys())
        self.roulette = []
        for i, chromosome in enumerate(self.chromosomes_pool):
            self.roulette += [i] * chrom_to_fit[chromosome]
