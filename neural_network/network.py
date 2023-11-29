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

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer.forward(x, **kwargs)
        self.z = x
        return self.z

    @property
    def result(self):
        return np.argmax(self.z, 0)

    def gradient_descent(self, x, y, learning_rate, iterations):
        for iter in range(iterations):
            self.forward(x)
            out_grad = (2 * (self.z - y)) / y.size
            mse = np.power((self.z - y), 2).mean()
            print(f"{iter=} <- -> {mse=}")

            for l in self.layers[::-1]:
                out_grad = l.backward(out_grad, learning_rate)

        return np.power((y - self.z), 2).mean()
