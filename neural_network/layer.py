import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weight = np.random.rand(n_neurons, n_inputs) - 0.5
        self.bias = np.random.rand(n_neurons, 1) - 0.5
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.output = self.activation.activate(self.weight.dot(input) + self.bias)
        return self.output

    def backward(self, out_grad, learning_rate):
        weight_grad = out_grad.dot(self.input.T)
        self.weight -= (
            weight_grad * learning_rate * self.activation.derivative(self.output)
        )
        self.bias -= learning_rate * out_grad * self.activation.derivative(self.output)
        return self.weight.T.dot(out_grad)
