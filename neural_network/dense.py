import numpy as np
from .layer import Layer
import pandas as pd


class Dense(Layer):
    def __init__(self, input_size, output_size, model_path=None):
        if not model_path:
            self.weight = np.random.rand(output_size, input_size) - 0.5
            self.bias = np.random.rand(output_size, 1) - 0.5
        else:
            self.weight = pd.read_csv(f"{model_path}_weight.csv").values[:, 1:]
            self.bias = pd.read_csv(f"{model_path}_bias.csv").values[:, 1:]

    def forward(self, input):
        self.input = input
        return self.weight.dot(self.input) + self.bias

    def backward(self, out_grad, learning_rate):
        weight_grad = out_grad.dot(self.input.T)
        self.weight -= learning_rate * weight_grad
        self.bias -= learning_rate * out_grad
        return self.weight.T.dot(out_grad)
