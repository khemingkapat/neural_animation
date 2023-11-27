import numpy as np
import matplotlib

# matplotlib.use("Agg") # useful for a webserver case where you don't want to ever visualize the result live.
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


# Change to reflect your file location!
# plt.rcParams["animation.ffmpeg_path"] = "./ffmpeg.exe"


# Fixing random state for reproducibility
np.random.seed(19680801)


metadata = dict(title="Movie", artist="codinglikemad")
writer = PillowWriter(fps=15, metadata=metadata)

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

plt.xlim(-10, 10)
plt.ylim(-10, 10)


def func(x, y, t):
    return (np.cos(x / 2 + t) ** 2 + np.sin(y / 2 + t) ** 2) - 1


xvec = np.linspace(0, 2 * np.pi, 1000)
yvec = np.linspace(0, 2 * np.pi, 1000)

xlist, ylist = np.meshgrid(xvec, yvec)

rlist = np.sqrt(np.square(xlist) + np.square(ylist))

with writer.saving(fig, "exp3d.gif", 100):
    for tval in np.linspace(0, 20, 160):
        print(tval)
        zval = func(xlist, ylist, tval)
        ax.set_zlim(-1, 1)
        ax.plot_surface(xlist, ylist, zval, cmap=cm.viridis)

        writer.grab_frame()
        plt.cla()
