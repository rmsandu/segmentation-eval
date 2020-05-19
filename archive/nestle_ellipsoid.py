# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import math
from scripts.ellipsoid_inner_outer import get_surface_points
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nestle


def plot_ellipsoid_3d(ell, ax):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = ell.ctr + np.dot(ell.axes,
                                                      [x[i,j],y[i,j],z[i,j]])

    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='#2980b9', alpha=0.2)



dir_name = r"C:\tmp_patients\Pat_MAV_BE_B02_\Study_0\Series_7\CAS-One Recordings\2019-07-28_19-33-55\Segmentations\SeriesNo_28\SegmentationNo_0"
A, spacing = get_surface_points(dir_name)

npoints = 4000
# A = np.array([[0.25, 1., 0.5],
#               [1., 0.25, 0.5],
#               [0.5, 1., 0.25]])
ell_gen = nestle.Ellipsoid([0., 0., 0.], np.dot(A.T, A))
points = ell_gen.samples(npoints)
pointvol = ell_gen.vol / npoints

# Find bounding ellipsoid(s)
ells = nestle.bounding_ellipsoids(points, pointvol)

# plot
fig = plt.figure(figsize=(10., 10.))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='.')
for ell in ells:
    plot_ellipsoid_3d(ell, ax)

fig.tight_layout()
plt.show()