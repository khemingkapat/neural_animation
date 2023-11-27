from neural_network import *
import numpy as np

nw = Network(10, [3, 5], [ReLU(), SoftMax()])
sample_data = np.random.rand(10, 1)

nw.forward(sample_data)
print(nw.layers[0].weight.shape)
