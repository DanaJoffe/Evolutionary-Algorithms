from GeneticAlgoAPI.chromosome import ListChromosomeBase
from GeneticAlgoAPI.crossover_strategy import SinglePointCrossover
from GeneticAlgoAPI.fitness_function import MistakesBasedFitnessFunc
from GeneticAlgoAPI.genetic_algorithm import GeneticAlgorithm, ApplyElitism
from GeneticAlgoAPI.mutation_strategy import BinaryMutation
from GeneticAlgoAPI.selection_strategy import RouletteWheelSelection
from config import MUTATION_RATE, CROSSOVER_RATE, POPULATION_SIZE, ELITISM, EMPTY, OCCUPIED, CELL_COLORS
from graphics import show_mat
from run_ga import build_and_run


""" Create a solution to the 8 Quines Problem """


class EightQueensChromosome(ListChromosomeBase):
    def __init__(self):
        super().__init__(24)

    def get_queens_locations(self):
        locations = []
        row = 0
        for i in range(0, len(self), 3):
            g1, g2, g3 = self[i:i + 3]
            col = int(''.join(map(str, [g1, g2, g3])), 2)
            locations.append((row, col))
            row += 1
        return locations

    def to_matrix(self):
        board = [[EMPTY for _ in range(8)] for _ in range(8)]
        row = 0
        for i in range(0, len(self), 3):
            g1, g2, g3 = self[i:i + 3]
            col = int(''.join(map(str, [g1, g2, g3])), 2)
            board[row][col] = OCCUPIED
            row += 1
        return board


class EightQueensGA(RouletteWheelSelection, SinglePointCrossover, BinaryMutation, ApplyElitism,
                    MistakesBasedFitnessFunc, GeneticAlgorithm):
    def __init__(self, elitism=ELITISM,
                 mutation_rate=MUTATION_RATE,
                 crossover_rate=CROSSOVER_RATE,
                 population_size=POPULATION_SIZE):
        super().__init__()
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size

    def calc_mistakes(self, chromosome):
        locations = []
        row = 0
        quines_in_columns_count = {col: 0 for col in range(8)}
        # iterate over triplets of genes
        for i in range(0, len(chromosome), 3):
            g1, g2, g3 = chromosome[i:i + 3]
            col = int(''.join(map(str, [g1, g2, g3])), 2)
            quines_in_columns_count[col] += 1
            locations.append((row, col))
            row += 1
        # each quine is by default in a different row, so collision can happen in column & diagonal
        column_collisions = sum([n*(n-1)/2 for n in quines_in_columns_count.values()])
        diagonal_collisions = sum([1 for i, (r1, c1) in enumerate(locations)
                                   for r2, c2 in locations[i+1:]
                                   if abs(r1-r2) == abs(c1-c2)])

        return int(column_collisions + diagonal_collisions)


def main():
    mutation_rate = .1
    crossover_rate = CROSSOVER_RATE
    population_size = POPULATION_SIZE
    elitism_count = ELITISM

    (time, unit), chromo = build_and_run(mutation_rate, crossover_rate, population_size, elitism_count,
                                         EightQueensGA, EightQueensChromosome)

    print("run for {} {}".format(time, unit))
    show_mat(chromo.to_matrix())


if __name__ == '__main__':
    main()


