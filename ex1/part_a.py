from statistics import mean

from GeneticAlgoAPI.chromosome import ListChromosomeBase, IntChromosome
from GeneticAlgoAPI.crossover_strategy import SinglePointCrossover, UniformCrossover
from GeneticAlgoAPI.early_convergence_avoidance import KeepAvgFarFromBest
from GeneticAlgoAPI.fitness_function import MistakesBasedFitnessFunc
from GeneticAlgoAPI.genetic_algorithm import GeneticAlgorithm, ApplyElitism
from GeneticAlgoAPI.mutation_strategy import BinaryMutation
from GeneticAlgoAPI.selection_strategy import RouletteWheelSelection
from config import EMPTY, OCCUPIED, CELL_COLORS
from graphics import show_mat
from run_ga import build_and_run, get_time_units
from timeit import default_timer as timer

""" Find a solution to the 8 Queens Problem """


class EightQueensChromosome(ListChromosomeBase):
    def __init__(self, **kargs):
        super().__init__(24, **kargs)  # 3 bits for queen, total 8 queens

    # # for IntChromosome
    # def __str__(self):
    #     s = ''
    #     for i in range(0, len(self), 3):
    #         genes = self[i:i + 3]
    #         s_ = str(bin(genes))[2:]
    #         s_ = '0' * (3 - len(s_)) + s_
    #         s += s_
    #     return s

    def get_queens_locations(self):
        locations = []
        row = 0
        for i in range(0, len(self), 3):
            genes = self[i:i + 3]
            col = int(''.join(map(str, genes)), 2)  # for IntChromosome: genes
            locations.append((row, col))
            row += 1
        return locations

    def to_matrix(self):
        board = [[EMPTY for _ in range(8)] for _ in range(8)]
        for row, col in self.get_queens_locations():
            board[row][col] = OCCUPIED
        return board


class EightQueensGA(RouletteWheelSelection, SinglePointCrossover, BinaryMutation, ApplyElitism,
                    MistakesBasedFitnessFunc, GeneticAlgorithm):
    def __init__(self, elitism, *args, **kwargs):
        GeneticAlgorithm.__init__(self, *args, **kwargs)
        self.elitism = elitism

    @classmethod
    def calc_mistakes(cls, chromosome):
        locations = chromosome.get_queens_locations()
        quines_in_columns_count = {col: 0 for col in range(8)}
        for row, col in locations:
            quines_in_columns_count[col] += 1

        # each queen is by default in a different row, so collision can happen in column & diagonal
        column_collisions = sum([n*(n-1)/2 for n in quines_in_columns_count.values()])
        diagonal_collisions = sum([1 for i, (r1, c1) in enumerate(locations)
                                   for r2, c2 in locations[i+1:]
                                   if abs(r1-r2) == abs(c1-c2)])

        return int(column_collisions + diagonal_collisions)


def measure_time(func):
    """ returns running time of func """
    def inner(*args, **kwargs):
        start = timer()
        output = func(*args, **kwargs)
        end = timer()
        return end-start, output
    return inner


@measure_time
def brute_force():
    run = 1
    while True:
        chromo = EightQueensChromosome()
        if EightQueensGA.calc_mistakes(chromo) == 0:
            break
        run += 1
    # print(f"it took {run} random chromosomes to find a solution")
    return run


def main():
    mutation_rate = .001
    crossover_rate = .75
    population_size = 100
    elitism_count = 2

    eca = KeepAvgFarFromBest(mr=.1, cr=1, gen=5, dist_from_avg=1)
    time, chromo, gen = build_and_run(eca, mutation_rate, crossover_rate, population_size, elitism_count, EightQueensGA,
                                      EightQueensChromosome)
    time, unit = get_time_units(time)
    print("run for {} {}".format(time, unit))
    show_mat(chromo.to_matrix())
    return time, gen


if __name__ == '__main__':
    main()

    """ RUN 100 TIME GA & BF """
    # for _ in range(100):
    #     print("================================")
    #     time, gen = brute_force()
    #     print(f"-> run for {time} sec and {gen} gen")

    # times_gens = [main() for _ in range(100)]
    # times_gens = [brute_force() for _ in range(100)]
    # print(f"avg time: {mean([t[0] for t in times_gens])} seconds, avg gens: {mean([t[1] for t in times_gens])}")


