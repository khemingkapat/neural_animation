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
        return self.z

    def accuracy(self, y):
        prediction = np.argmax(self.z, 0)
        print(self.z)
        print(prediction)
        return (prediction == y).mean()

    def gradient_descent(self, x, y, learning_rate, iterations):
        y_true = np.eye(x.shape[0])[y].T
        for iter in range(iterations):
            self.forward(x)

            out_grad = (2 / y.shape[0]) * (self.z - y_true)
            mse = ((y_true - self.z) ** 2).mean()
            if iter % 10 == 0:
                print(f"iter {iter} ==> mse = {mse}")
                print(f"Accuracy ==> {self.accuracy(y)}")
                print("-" * 30)
            for l in self.layers[::-1]:
                out_grad = l.backward(out_grad, learning_rate)

        return np.power((y - self.z), 2).mean()
