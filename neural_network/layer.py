import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weight = np.random.rand(n_neurons, n_inputs) - 0.5
        self.bias = np.random.rand(n_neurons, 1) - 0.5
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.linear_output = self.weight.dot(input) + self.bias
        self.output = self.activation.activate(self.linear_output)
        return self.output

    def backward(self, out_grad, learning_rate):
        _, m = self.input.shape
        # print(m)
        act_grad = np.multiply(out_grad, self.activation.derivative()) / m
        weight_grad = act_grad.dot(self.input.T)
        # print(weight_grad.sum())
        bias_grad = act_grad.sum(axis=1).reshape(self.bias.shape)
        self.weight -= learning_rate * weight_grad
        self.bias -= learning_rate * bias_grad
        return self.weight.T.dot(act_grad)
