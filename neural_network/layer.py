import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, out_grad, learning_rate):
        pass

    def __str__(self):
        return f"{self.__class__.__name__} layer"
