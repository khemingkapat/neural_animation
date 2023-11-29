from neural_network import *
import numpy as np

nw = Network(10, [5, 5, 3], [ReLU(), ReLU(), ReLU(), SoftMax()])


sample_data = np.random.rand(10, 1)
y = np.zeros((3, 1))
y[2] = 1

print(nw.gradient_descent(sample_data, y, 0.5, 20))
