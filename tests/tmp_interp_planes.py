# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interpn

x1 = np.arange(10)
x2 = np.arange(10)
# arr = np.arange(10)
arr = x1 + x2[:, np.newaxis]
# arr = np.arange(10)
interp_x = 3           # Only one value on the x1-axis
interp_y = np.arange(10)    # A range of values on the x2-axis

# Note the following two lines that are used to set up the
# interpolation points as a 10x2 array!
interp_mesh = np.array(np.meshgrid(interp_x, interp_y))
interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((10, 2))

# Perform the interpolation
points = (x1, x2)
interp_arr = interpn(points, arr, interp_points)

# Set up grid for plotting
X, Y = np.meshgrid(x1, x2)

# Plot the result
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, arr, rstride=1, cstride=1, cmap=cm.jet,
                       linewidth=0, alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.scatter(interp_x * np.ones(interp_y.shape), interp_y, interp_arr, s=20,
           c='k', depthshade=False)
plt.xlabel('x1')
plt.ylabel('x2')

plt.show()