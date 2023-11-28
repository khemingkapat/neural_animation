from neural_network import *
import numpy as np

np.random.seed(0)
l1 = Layer(5, 5, ReLU())
print(l1.weight)
sample_data = np.random.rand(5, 1)

y = np.zeros((5, 1))
y[1] = 1
for _ in range(20):
    result = l1.forward(sample_data)
    err = ((y - result) ** 2).sum() / y.size
    print(f"{err=}")
    print("-" * 50)
    out_grad = (2 * (result - y)) / y.size
    dx = l1.backward(out_grad, 0.1)
