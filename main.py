from neural_network import *
import numpy as np
import pandas as pd


def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))


def mse_prime(y, y_pred):
    return 2 * (y_pred - y) / np.size(y)


data = pd.read_csv("./train.csv").values[:100]
np.random.shuffle(data)

X = data[:, 1:].reshape(-1, 784, 1) / 255
Y = data[:, 0].reshape(-1, 1, 1)
print(np.unique(Y))


network = [Dense(28 * 28, 10), Tanh(), Dense(10, 10), Tanh()]

epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    err = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)

        y_true = np.eye(10)[y].T.reshape(-1, 1)
        err += mse(y_true, output)
        grad = mse_prime(y_true, output)

        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    err /= len(X)
    print(f"epoch {epoch} error = {err}")

print("-" * 30 + "after trained" + "-" * 30)
for x, y in list(zip(X, Y))[:20]:
    output = x
    for layer in network:
        output = layer.forward(output)

    print(f"actual y = {y}")
    print(f"prediction = {np.argmax(output)}")
    print("-" * 50)
