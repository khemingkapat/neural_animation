from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self, compound=None):
        self.compound = compound

    @abstractmethod
    def activate(self, input):
        pass

    @abstractmethod
    def derivative(self):
        pass


class ReLU(Activation):
    def activate(self, input):
        return np.maximum(input, 0)

    def derivative(self, z):
        return z > 0


class SoftMax(Activation):
    def activate(self, input):
        return np.exp(input) / sum(np.exp(input))

    def derivative(self, z):
        result = np.diag(z)
        for i in range(z.size):
            for j in range(z.size):
                if i == j:
                    result[i, j] = z[i] * (1 - z[i])
                else:
                    result[i, j] = -z[i] * z[j]
        return result
