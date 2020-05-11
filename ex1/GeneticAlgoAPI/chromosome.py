import random
from abc import ABC, abstractmethod

"""
shouldn't inherit from Chromosome class, but from it's subclasses.
"""


class IndexedObject(ABC, object):
    @abstractmethod
    def __getitem__(self, k):
        """  """
        raise NotImplemented

    @abstractmethod
    def __setitem__(self, key, value):
        """  """
        raise NotImplemented

    @abstractmethod
    def __len__(self):
        """  """
        raise NotImplemented


class Chromosome(IndexedObject, ABC):
    fitness = None
    genome = None

    def set_fitness_score(self, score):
        """  """
        self.fitness = score

    def get_fitness(self):
        """ """
        return self.fitness

    def __repr__(self):
        fitness = self.get_fitness()
        if fitness:
            return "({}) {}".format(str(fitness), str(self))
        return str(self)

    @abstractmethod
    def __copy__(self):
        raise NotImplemented


class ListChromosomeBase(Chromosome, ABC):
    """ chromosome that have a list of ints (bits) as inner representation """
    def __init__(self, length, genome=None):
        """ initialize random bit list """
        self.genome = genome
        if genome is None:
            num = random.randint(0, 2 ** length - 1)
            bit_string = bin(num)[2:]
            bit_string = '0' * (length - len(bit_string)) + bit_string
            bit_list = [int(num) for num in list(bit_string)]
            self.genome = bit_list

    def __getitem__(self, k):
        return self.genome.__getitem__(k)

    def __setitem__(self, key, value):
        self.genome.__setitem__(key, value)

    def __len__(self):
        return self.genome.__len__()

    def __str__(self):
        return ','.join(str(v) for v in self.genome)

    def __copy__(self):
        class_type = self.__class__
        instance = class_type(genome=self.genome.copy())
        return instance


class IntChromosome(Chromosome):
    def __init__(self, length, genome=None):
        self.length = length
        self.genome = genome
        # initialize random bit list
        if genome is None:
            self.genome = random.randint(0, 2 ** length - 1)

    def _get_bit(self, start, stop):
        total_bits = stop - start
        ones = 2 ** total_bits - 1
        mask = ones << start
        num = mask & self.genome
        return num >> start

    def __getitem__(self, key):
        if type(key) is slice:
            if key.step:
                raise NotImplemented
            start = key.start if key.start else 0
            stop = key.stop if key.stop else self.length
            return self._get_bit(start, stop)
        # get k-th bit
        return self._get_bit(key, key+1)

    def _set_bit(self, start, stop, value):
        # turn-off bits
        total_bits = stop - start
        mask = (2 ** total_bits - 1) << start
        ones = 2 ** self.length - 1
        mask = ones ^ mask
        self.genome = self.genome & mask

        # copy value
        value = value << start
        self.genome = self.genome | value

    def __setitem__(self, key, value):
        """
        :param key: position
        :param value: bit value
        """
        if type(key) is slice:
            if key.step:
                raise NotImplemented
            start = key.start if key.start else 0
            stop = key.stop if key.stop else self.length
            self._set_bit(start, stop, value)
        else:
            self._set_bit(key, key+1, value)

    def __len__(self):
        return self.length

    def __copy__(self):
        class_type = self.__class__
        instance = class_type(genome=self.genome)
        return instance

    def __str__(self):
        raise NotImplemented


