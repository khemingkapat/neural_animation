import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weight = np.random.rand(n_neurons, n_inputs) - 0.5
        self.bias = np.random.rand(n_neurons, 1) - 0.5
        self.activation = activation

    def forward(self, input):
        return self.activation.activate(self.weight.dot(input) + self.bias)
