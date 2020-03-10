# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
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
# pz = [0, 2, 2.5, 1, 3]

m = np.size(px)
pxint = sum(px) / m
pyint = sum(py) / m
#TODO
# px = [px; px(0)];
# py = [py; py(0)];
px = [0, .5, 2, 3, 1,  0]
py = [0, 1, 1.5, .5, -.5, 0]
px = np.asarray(px)
py = np.asarray(py)

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
xcenter = d.value[0]
ycenter = d.value[1]
plt.plot(px, py, 'r-')
angle = ellipse_angle_of_rotation(B.value)
ellipse = Ellipse((xcenter, ycenter), rx*2, ry*2, angle=degrees(angle), fill=False)

ax.add_patch(ellipse)
plt.show()
