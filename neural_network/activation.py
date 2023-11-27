from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
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
        return int(z > 0)


class SoftMax(Activation):
    def activate(self, input):
        return np.exp(input) / sum(np.exp(input))

    def derivative(self, z):
        return z
