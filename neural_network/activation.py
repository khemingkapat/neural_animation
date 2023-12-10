import numpy as np


class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def activate(self, input):
        self.input = input
        return self.activation(self.input)

    def derivative(self):
        return self.activation_prime(self.input)


class PassThrough(Activation):
    def __init__(self):
        pass_through = lambda x: x
        pass_through_prime = lambda x: 1
        super().__init__(pass_through, pass_through_prime)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: x > 0
        super().__init__(relu, relu_prime)


class SigMoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / 1 + np.exp(-x)
        sigmoid_prime = lambda x: x * (1 - x)
        super().__init__(sigmoid, sigmoid_prime)
