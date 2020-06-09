from typing import Optional, List

from lib.GeneticProgrammingAPI.component import Component


class Node(object):
    """ tree node """
    def __init__(self, content):
        self._content: Component = content

        self.children: List[Node] = []  # [Node(0) for _ in range(sons_num)]
        self._nodes = 1  # subtree nodes (including self)
        self._depth = 0  # subtree depth
        self.parent: Optional[Node] = None
        self._level = 0  # relative to head

    @property
    def content(self):
        return self._content

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, value):
        """ rolling setter: rolling down the tree """
        self._level = value
        for child in self.children:
            child.level = self._level + 1

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, value):
        """ rolling setter: rolling up the tree """
        self._depth = value
        if self.parent:
            self.parent.depth = max(child.depth for child in self.parent.children) + 1

    @property
    def nodes(self) -> int:
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        """ rolling setter: rolling up the tree """
        self._nodes = value
        if self.parent:
            self.parent.nodes = sum(child.nodes for child in self.parent.children) + 1

    def _set_parent(self, parent: "Node"):
        self.parent = parent

    def add_child(self, child: "Node"):
        self.children.append(child)
        child._set_parent(self)
        return self

    def replace_node(self, other: "Node"):
        assert self.parent is not None, "must have parent"
        self.parent._replace_child(self, other)

    def _replace_child(self, child: "Node", other: "Node"):
        """ replace self's child-subtree with other subtree """
        child.parent = None
        other.parent = self
        # todo: improve
        for i, node in enumerate(self.children):
            if node == child:
                self.children[i] = other
                break

        # update nodes amount & subtree depth in parent (self)
        other.nodes = other.nodes  # updates parent
        other.depth = other.depth  # updates parent
        # update level in new child subtree (other)
        other.level = self.level + 1
        # update level in old child subtree (child)
        child.level = 0

    def set_content(self, content):
        self._content = content

    def __copy__(self) -> "Node":
        copy: Node = Node(self._content.__copy__())
        for child in self.children:
            # set children and parents
            copy.add_child(child.__copy__())
        copy._nodes = self._nodes
        copy._depth = self._depth
        copy._level = self._level
        return copy

    # Component

    @property
    def arity(self):
        return self._content.arity

    @property
    def symbol(self):
        return self._content.symbol

    def __repr__(self):
        return self._content.__repr__()

    def calc(self, *args, **kwargs):
        # recursive call
        values = [child.calc(*args, **kwargs) for child in self.children]
        assert len(values) == self.arity, f"children amount is wrong"
        return self._content.calc(values, *args, **kwargs)

    def __str__(self):
        # recursive call
        values = [child.__str__() for child in self.children]
        assert len(values) == self.arity, f"children amount is wrong"
        ret = self._content.__str__().format(*values)
        return ret

    def __call__(self, *args, **kwargs) -> "Component":
        """ create this component"""
        return self._content.__call__()

    def get_nodes(self):
        return [node for node in iterate_tree(self)]


def iterate_tree(head: Node):
    yield head
    for child in head.children:
        yield from iterate_tree(child)

