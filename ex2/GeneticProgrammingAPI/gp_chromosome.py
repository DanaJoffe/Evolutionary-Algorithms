import math
import random
from typing import List, Iterable

from GeneticAlgoAPI.chromosome import Chromosome
from GeneticProgrammingAPI.component import Component
from GeneticProgrammingAPI.node import Node, iterate_tree


class GPChromosome(Chromosome):
    def __init__(self, components: Iterable[Component], max_depth=math.inf, max_nodes=math.inf, head=None):
        self.max_nodes = max_nodes
        self.max_depth = max_depth  # depth = number of edges in a row
        self._components = components
        if head:
            self.head = head
        else:
            self.head = Node(random.choice(components)())
            self._init_nodes(self.head, 0, [1 + self.head.arity])
            self.head.level = 0
            leafs = [node for node in iterate_tree(self.head) if not node.children]
            for leaf in leafs:
                leaf.depth = 0
                leaf.nodes = 1

    @property
    def genome(self) -> List[Node]:
        return [node for node in iterate_tree(self.head)]

    def get_nodes(self):
        return self.genome

    @property
    def nodes(self) -> int:
        return self.head.nodes

    def _init_nodes(self, node: Node, total_depth: int, nodes_counter):
        if node.arity == 0:
            return  # 0, 1
        components = self._components
        if total_depth == self.max_depth - 1:
            # limit components to arity=0
            components = [c for c in components if c.arity == 0]
        for _ in range(node.arity):
            # limit components so that max_nodes is kept
            components = [c for c in components if nodes_counter[0] + c.arity <= self.max_nodes]
            child = Node(random.choice(components)())
            node.add_child(child)
            nodes_counter[0] += child.arity
            self._init_nodes(child, total_depth + 1, nodes_counter)

    def calc(self, *args, **kwargs):
        return self.head.calc(*args, **kwargs)

    def __copy__(self):
        class_type = self.__class__
        instance = class_type(components=self._components, max_depth=self.max_depth, max_nodes=self.max_nodes,
                              head=self.head.__copy__())
        # sanity check
        # assert instance.head.level == 0
        # for n in instance.genome:
        #     if n.parent:
        #         assert n.level == n.parent.level + 1
        return instance

    def __str__(self):
        # return f"tree: {self.genome}"
        return f"tree: {str(self.head)}"

