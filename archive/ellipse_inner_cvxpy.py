# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy.linalg as la
from matplotlib.patches import Ellipse
from math import pi, cos, sin, degrees
import numpy as np
import cvxpy as cp
from scipy.linalg import null_space, norm
# random seed
np.random.seed(0)


def ellipse_angle_of_rotation(a):
    a, b, c = a[0][0], a[0][1], a[1][1]
    if b == 0:
        if a < c:
            return 0
        else:
            return np.pi/2
    else:
        if a < c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

#%% Problem data.
n = 2
px = [0, .5, 2, 3, 1]
py = [0, 1, 1.5, .5, -.5]
# pz = [0, 2, -3.5, 2.8, 3.2]
m = np.size(px)
pxint = sum(px) / m
pyint = sum(py) / m
# pzint = sum(pz) / m
#TODO
# px = [px; px(0)];
# py = [py; py(0)];
px = [0, .5, 2, 3, 1, 0]
py = [0, 1, 1.5, .5, -.5, 0]
# pz = [0, 2, -3.5, 2.8, -3.2, 0]
px = np.asarray(px)
py = np.asarray(py)
# pz = np.asarray(pz)
#%%
# generate A, b
A = np.zeros((m, n))
b = np.zeros((m, 1))

for i in range(0, m):
    A[i, :] = null_space(np.asmatrix([px[i+1] - px[i], py[i+1] - py[i]])).T
    b[i] = A[i, :] * .5*(np.asmatrix([px[i+1] + px[i], py[i+1] + py[i]])).T
    if A[i, :] * np.asmatrix([pxint, pyint]).T - b[i] > 0:
        A[i, :] = -A[i, :]
        b[i] = -b[i]

#%% Construct the problem.
# cvx_begin

#     variable B(n,n) symmetric
#     variable d(n)
#     maximize( det_rootn( B ) )
#     subject to
#        for i = 1:m
#            norm( B*A(i,:)', 2 ) + A(i,:)*d <= b(i);
#        end
# cvx_end
B = cp.Variable((n, n), symmetric=True)
d = cp.Variable(n)
objective = cp.Maximize(cp.log_det(B))
constraint = [cp.norm(B * A[i, :].T, 2) + A[i, :] * d <= b[i] for i in range(m)]

prob = cp.Problem(objective, constraint)
result = prob.solve(verbose=True)

#%% PLOT THE RESULTS

fig, ax = plt.subplots()
U, D, V = la.svd(np.asarray(B.value))
rx, ry = 1. / np.sqrt(D)
U, D, V = la.svd(np.asarray(B.value) / 2)
rx_outer, ry_outer = 1. / np.sqrt(D)
xcenter = d.value[0]
ycenter = d.value[1]
plt.plot(px, py, 'b--')
angle = ellipse_angle_of_rotation(B.value)
ellipse = Ellipse((xcenter, ycenter), rx*2, ry*2, angle=degrees(angle), fill=False, color='red')
ellipse_outer = Ellipse((xcenter, ycenter), rx_outer*2, ry_outer*2, angle=degrees(angle), fill=False, color='green')
ax.add_patch(ellipse)
ax.add_patch(ellipse_outer)
plt.ylim([-1, 4])
plt.xlim([-1, 4])
plt.show()
# %% 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(px, py, pz, c='orange', label='Ablation Segmentation')
# plt.show()