import math
import random
from abc import ABC, abstractmethod
from inspect import signature
from typing import Optional, Tuple

""" basics """


class Component(ABC):
    def __init__(self, arity: int, symbol: str = '', string_repr=None):
        super().__init__()
        self.arity = arity
        self.symbol = symbol
        self.string_repr = string_repr

    def __repr__(self):
        return self.symbol

    @abstractmethod
    def calc(self, values, *args, **kwargs):
        raise NotImplemented

    def __str__(self):
        if self.string_repr:
            return self.string_repr
        return self.symbol

    def __call__(self, *args, **kwargs) -> "Component":
        """ create this component"""
        return self._create()

    def _create(self) -> "Component":
        return self.__copy__()

    @abstractmethod
    def __copy__(self) -> "Component":
        raise NotImplemented


class Variable(Component):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, arity=0, symbol=name, **kwargs)
        self.name = name

    def calc(self, values, *args, **kwargs):
        value = kwargs.get(self.name, None)
        assert value is not None, f"missing variable {self.name}"
        return value

    def __copy__(self):
        class_type = self.__class__
        instance = class_type(self.name)
        return instance
        # return Variable(self.name)


class Constant(Component):
    def __init__(self, value=None, range: Tuple[int, int]=None, integer=False, **kwargs):
        self._value = value
        self._range = range
        self._integer = integer
        if value:
            self.value = value
        elif integer:
            # raise Exception("Integer Constant - Not Supported")
            self.value = random.randint(*range)
        else:
            self.value = random.uniform(*range)
        super().__init__(arity=0, symbol=str(self.value), *kwargs)

    def calc(self, *args, **kwargs):
        return self.value

    def __copy__(self):
        class_type = self.__class__
        instance = class_type(self.value)
        return instance
        # return Constant(self.value)

    def _create(self) -> "Component":
        return Constant(self._value, self._range, self._integer)


class Function(Component):
    def __init__(self, func, *args, **kwargs):
        params = signature(func).parameters
        super().__init__(*args, arity=len(params), **kwargs)
        self.func = func

    def calc(self, values, *args, **kwargs):
        return self.func(*values)

    def __copy__(self):
        class_type = self.__class__
        instance = class_type(self.func, symbol=self.symbol, string_repr=self.string_repr)
        return instance


""" name typedef """


class Operator(Function):
    pass


class Condition(Function):
    pass


""" support string representation """


class BinaryOperator(Operator):
    def __str__(self):
        ret = '({}' + self.symbol + '{})'
        return ret


class PrefixUnaryOperator(Operator):
    def __str__(self):
        ret = '(' + self.symbol + '{})'
        return ret


class PostfixUnaryOperator(Operator):
    def __str__(self):
        ret = '({}' + self.symbol + ')'
        return ret


PLUS     = BinaryOperator(lambda x, y: x + y, symbol='+')
SUBTRACT = BinaryOperator(lambda x, y: x - y, symbol='-')
MULTIPLY = BinaryOperator(lambda x, y: x * y, symbol='*')
DIVIDE   = BinaryOperator(lambda x, y: x / y if y != 0 else math.inf, symbol='/')

MINUS   = PrefixUnaryOperator(lambda x: -x, symbol='-')
SQUARED = PostfixUnaryOperator(lambda x: x**2, symbol='^2')

IF_ELSE = Condition(lambda x,y,z: y if x else z, symbol='if else', string_repr='(if {} then {} else {})')


