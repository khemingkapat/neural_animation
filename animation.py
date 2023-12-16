from matplotlib import animation
import numpy as np
import matplotlib

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import seaborn as sns

fig = plt.figure()
data_list = []
for _ in range(100):
    data_list.append(np.random.randn(10, 10))


def init():
    sns.heatmap(np.zeros((10, 10)), vmax=0.8, cbar=False)


def animate(i):
    data = data_list[i]
    sns.heatmap(data, vmax=0.8, cbar=False)


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=20)
writer = animation.PillowWriter(fps=20)
anim.save("heatmap.gif", writer=writer)
