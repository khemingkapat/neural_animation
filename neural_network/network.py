import numpy as np
from .dense import Dense
import pandas as pd


class Network:
    def __init__(self, neurons, acts, model_path=None):
        self.layers = []
        if model_path:
            for idx in range(len(neurons) - 1):
                self.layers.append(
                    Dense(neurons[idx], neurons[idx + 1], f"{model_path}/layer{idx+1}")
                )
                self.layers.append(acts[idx])
        else:
            for idx in range(len(neurons) - 1):
                self.layers.append(Dense(neurons[idx], neurons[idx + 1]))
                self.layers.append(acts[idx])

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def gradient_descent(self, X, Y, iteration, learning_rate):
        def mse(y, y_pred):
            return np.mean(np.power(y - y_pred, 2))

        def mse_prime(y, y_pred):
            return 2 * (y_pred - y) / np.size(y)

        for _ in range(iteration):
            err = 0
            for x, y in zip(X, Y):
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                y_true = np.eye(10)[y].T.reshape(-1, 1)
                err += mse(y_true, output)
                grad = mse_prime(y_true, output)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

    def save_model(self):
        for idx, layer in enumerate(self.layers[::2], start=1):
            pd.DataFrame(layer.weight).to_csv(f"./network/layer{idx}_weight.csv")
            pd.DataFrame(layer.bias).to_csv(f"./network/layer{idx}_bias.csv")
