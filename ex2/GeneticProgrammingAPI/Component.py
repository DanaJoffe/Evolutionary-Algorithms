

class Node(object):
    """ tree node """
    def __init__(self):
        self.sons = []  # [Node(0) for _ in range(sons_num)]

    def add_son(self, son: "Node"):
        self.sons.append(son)


class Component(Node):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity
        self.inputs = 0  # should be arity's size


class Variable(Component):
    def __init__(self, name: str):
        super().__init__(arity=0)
        self.name = name


class Operator(Component):
    def __init__(self, op_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_func = op_func


class Constant(Component):
    def __init__(self, value):
        super().__init__(arity=0)
        self.value = value


class Condition(Component):
    def __init__(self, cond_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_func = cond_func


