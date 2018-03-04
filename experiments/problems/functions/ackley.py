import math
import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class Ackley(Function):

    def function(self, x):
        first = -20. * np.exp(-0.2 * np.sqrt(np.sum(np.power(x, 2))/float(len(x))))
        second = np.exp(np.sum(np.cos(np.multiply(2 * np.pi, x)))/float(len(x)))
        return first - second + 20. + np.e

    def getBounds(self):
        return Bounds(-32.768, 32.768)

    def test(self):
        assert(self.testOfFunct([1, 1]) == self.function([1, 1]))

    def testOfFunct(self, x):
        firstSum = 0.0
        secondSum = 0.0
        for c in x:
            firstSum += c ** 2.0
            secondSum += math.cos(2.0 * math.pi * c)
        n = float(len(x))
        return -20.0 * math.exp(-0.2 * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + 20 + math.e
