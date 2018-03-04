import math
import numpy as np

from mlpy.numberGenerator.bounds import Bounds
from experiments.problems.functions.structure.function import Function

class SchafferF6(Function):

    def function(self, x):
        x_squared = np.power(x, 2)
        one_index_sum = x_squared[:-1] + x_squared[1:]
        numerator = np.power(np.sin(one_index_sum), 2) - 0.5
        denominator = np.power(1 + (0.001 * (one_index_sum)), 2)
        return np.sum(0.5 + np.divide(numerator, denominator))

    def getBounds(self):
        return Bounds(-100, 100)