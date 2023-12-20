import numpy as np
from mycolorpy import colorlist as mcp

data = np.random.randn(10)
print(data.shape)
color1 = mcp.gen_color_normalized(cmap="inferno", data_arr=data)
print(color1)
