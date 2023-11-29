from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self, compound=None):
        self.compound = compound

    @abstractmethod
    def activate(self, **kwargs):
        pass

    @abstractmethod
    def derivative(self, **kwargs):
        pass


class ReLU(Activation):
    def activate(self, input, **kwargs):
        return np.maximum(input, 0)

    def derivative(self, z, **kwargs):
        return z > 0


class SoftMax(Activation):
    def activate(self, input, **kwargs):
        return np.exp(input) / sum(np.exp(input))

    def derivative(self, z, **kwargs):
        print(f"{z.shape=}")
        result = np.zeros((z.size, z.size))
        for i in range(z.size):
            for j in range(z.size):
                if i == j:
                    result[i, j] = z[i] * (1 - z[i])
                else:
                    result[i, j] = -z[i] * z[j]
        return result
