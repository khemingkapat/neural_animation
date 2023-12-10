from neural_network import *
import numpy as np
import pandas as pd

x = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
y = np.array([0, 1, 1, 0])

nw = Network(2, [3, 2], [ReLU(), SigMoid()])
nw.gradient_descent(x, y, 0.2, 1000)
