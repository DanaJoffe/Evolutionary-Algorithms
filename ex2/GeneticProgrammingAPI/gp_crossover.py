import random
from typing import Tuple
from GeneticAlgoAPI.crossover_strategy import CrossoverStrategy, TwoParentsTwoChildren, TwoParentsOneChild
from GeneticProgrammingAPI.Node import Node
from GeneticProgrammingAPI.gp_chromosome import GPChromosome
from graphics import plot_tree


class GPCrossover(TwoParentsOneChild, CrossoverStrategy):
    def pair_chromosomes(self, chromosomes: Tuple[GPChromosome]):
        """ choose edge = choose node that is not the head (every node has one incoming edge) """
        assert len(chromosomes) == 2, "must contain 2 parents"
        ch1, ch2 = chromosomes[0], chromosomes[1]

        offspring1 = ch1.__copy__()
        offspring2 = ch2.__copy__()
        max_nodes = offspring1.max_nodes
        max_depth = offspring1.max_depth

        children = [node for node in offspring1.genome if node.parent]
        if not children:
            # todo: one node parent..?
            return offspring1,
        random_child: Node = random.choice(children)
        current_level = random_child.level
        current_nodes = offspring1.head.nodes - random_child.nodes
        # todo: is the head optional?
        optional_nodes = [node for node in offspring2.genome
                          if node.depth + current_level <= max_depth and current_nodes + node.nodes <= max_nodes]
        new_child = random.choice(optional_nodes)
        random_child.replace_node(new_child)

        assert offspring1.head.depth <= max_depth
        assert offspring1.head.nodes <= max_nodes
        # Note: leftovers weren't treated
        return offspring1,


