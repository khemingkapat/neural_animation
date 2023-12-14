from neural_network import *
import numpy as np
import pandas as pd

data = pd.read_csv("./train.csv").values
np.random.shuffle(data)
data = data[:10000]

X = data[:, 1:].reshape(-1, 784, 1) / 255
Y = data[:, 0].reshape(-1, 1, 1)

network = Network([28 * 28, 10, 10], [Tanh(), Tanh()])
network.gradient_descent(X, Y, 1000, 0.1)
network.save_model()
