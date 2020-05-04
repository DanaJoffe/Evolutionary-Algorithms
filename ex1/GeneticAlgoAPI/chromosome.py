import random
from abc import ABC, abstractmethod
from math import log

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
    def __init__(self, length):
        """ initialize random bit list """
        # todo: change distribution to numpy.random.uniform

        # binary rep: bin(x)
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
        instance = class_type()
        instance.genome = self.genome.copy()
        return instance


class IntChromosome(Chromosome):
    # int8 = 8 bits, one letter
    # 28 chars, 5 bits (32 options). 8 bits = 255 options

    """ every cell holds int8 number """

    def __init__(self, length):
        self.length = length
        """ initialize random bit list """
        # todo: change distribution to numpy.random.uniform

        # binary rep: bin(x)
        num = random.randint(0, 2 ** length - 1)
        self.genome = num

    def _get_bit(self, pos):
        mask = 1 << pos
        num = mask & self.genome
        return num >> pos

    def __getitem__(self, k):
        # a[start:stop]  # items start through stop-1
        # a[start:]  # items start through the rest of the array
        # a[:stop]  # items from the beginning through stop-1
        # a[:]  # a copy of the whole array

        if type(k) is slice:
            if k.step:
                raise NotImplemented

            total_bits = k.stop - k.start
            ones = 2 ** total_bits - 1
            mask = ones << k.start
            num = mask & self.genome
            return num >> k.start
            # return [self._get_bit(pos) for pos in range(k.start, k.stop)]
        # get k-th bit
        return self._get_bit(k)

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

        # assert 0 <= key <= self.length
        # assert value == 0 or value == 1
        # mask = 1 << key
        # if value == 1:
        #     self.genome = self.genome | mask
        # else:
        #     ones = 2 ** self.length - 1
        #     mask = ones ^ mask
        #     self.genome = self.genome & mask

    def __setitem__(self, key, value):
        """
        :param key: position
        :param value: bit value
        """
        if type(key) is slice:
            if key.step:
                raise NotImplemented

            self._set_bit(key.start, key.stop, value)

            # # turn-off bits
            # total_bits = key.stop - key.start
            # mask = (2 ** total_bits - 1) << key.start
            # ones = 2 ** self.length - 1
            # mask = ones ^ mask
            # self.genome = self.genome & mask
            #
            # # copy value
            # value = value << key.start
            # self.genome = self.genome | value
            #
            # # for k, v in zip(range(key.start, key.stop), value):
            # #     self._set_bit(k, v)
        else:
            self._set_bit(key, key+1, value)
            # self._set_bit(key, value)

    def __len__(self):
        return self.length  # self.genome.__len__()

    def __copy__(self):
        class_type = self.__class__
        instance = class_type()
        instance.genome = self.genome
        return instance



