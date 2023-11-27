from .layer import Layer
import numpy as np


class Network:
    def __init__(self, n_inputs, n_neurons, activations):
        self.n_layers = len(n_neurons)
        self.layers = []
        all_neurons = [n_inputs] + n_neurons
        for i in range(1, len(n_neurons) + 1):
            self.layers.append(
                Layer(all_neurons[i - 1], all_neurons[i], activations[i - 1])
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.z = x

    @property
    def result(self):
        return np.argmax(self.z, 0)

    def backward(self, x, y):
        pass
