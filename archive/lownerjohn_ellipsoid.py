# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

##
#  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
#
#  File:      lownerjohn_ellipsoid.py
#
#  Purpose:
#  Computes the Lowner-John inner and outer ellipsoidal
#  approximations of a polytope.
#
#  Note:
#  To plot the solution the Python package matplotlib is required.
#
#  References:
#    [1] "Lectures on Modern Optimization", Ben-Tal and Nemirovski, 2000.
#    [2] "MOSEK modeling manual", 2018
##
from scipy.spatial import ConvexHull
import polytope as pc
import DicomReader as Reader
import scripts.ellipsoid_inner_outer as ell
import sys
from math import sqrt, ceil, log
from mosek.fusion import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.linalg import null_space, norm

'''
Models the convex set 

  S = { (x, t) \in R^n x R | x >= 0, t <= (x1 * x2 * ... * xn)^(1/n) }

using three-dimensional power cones
'''
def geometric_mean(M, x, t):
    n = int(x.getSize())
    if n==1:
      M.constraint(Expr.sub(t, x), Domain.lessThan(0.0))
    else:
      t2 = M.variable()
      M.constraint(Var.hstack(t2, x.index(n-1), t), Domain.inPPowerCone(1-1.0/n))
      geometric_mean(M, x.slice(0,n-1), t2)


'''
 Purpose: Models the hypograph of the n-th power of the
 determinant of a positive definite matrix. See [1,2] for more details.

   The convex set (a hypograph)

   C = { (X, t) \in S^n_+ x R |  t <= det(X)^{1/n} },

   can be modeled as the intersection of a semidefinite cone

   [ X, Z; Z^T Diag(Z) ] >= 0  

   and a number of rotated quadratic cones and affine hyperplanes,

   t <= (Z11*Z22*...*Znn)^{1/n}  (see geometric_mean).
'''
def det_rootn(M, t, n):
    # Setup variables
    Y = M.variable(Domain.inPSDCone(2 * n))

    # Setup Y = [X, Z; Z^T , diag(Z)]
    X   = Y.slice([0, 0], [n, n])
    Z   = Y.slice([0, n], [n, 2 * n])
    DZ  = Y.slice([n, n], [2 * n, 2 * n])

    # Z is lower-triangular
    M.constraint(Z.pick([[i,j] for i in range(n) for j in range(i+1,n)]), Domain.equalsTo(0.0))
    # DZ = Diag(Z)
    M.constraint(Expr.sub(DZ, Expr.mulElm(Z, Matrix.eye(n))), Domain.equalsTo(0.0))

    # t^n <= (Z11*Z22*...*Znn)
    geometric_mean(M, DZ.diag(), t)

    # Return an n x n PSD variable which satisfies t <= det(X)^(1/n)
    return X

'''
  The inner ellipsoidal approximation to a polytope 

     S = { x \in R^n | Ax < b }.

  maximizes the volume of the inscribed ellipsoid,

     { x | x = C*u + d, || u ||_2 <= 1 }.

  The volume is proportional to det(C)^(1/n), so the
  problem can be solved as 

    maximize         t
    subject to       t       <= det(C)^(1/n)
                || C*ai ||_2 <= bi - ai^T * d,  i=1,...,m
                C is PSD

  which is equivalent to a mixed conic quadratic and semidefinite
  programming problem.
'''
def lownerjohn_inner(A, b):
    with Model("lownerjohn_inner") as M:
        M.setLogHandler(sys.stdout)
        m, n = len(A), len(A[0])

        # Setup variables
        t = M.variable("t", 1, Domain.greaterThan(0.0))
        C = det_rootn(M, t, n)
        d = M.variable("d", n, Domain.unbounded())

        # (b-Ad, AC) generate cones
        M.constraint("qc", Expr.hstack(Expr.sub(b, Expr.mul(A, d)), Expr.mul(A, C)),
                     Domain.inQCone())

        # Objective: Maximize t
        M.objective(ObjectiveSense.Maximize, t)

        M.solve()

        M.writeTask('lj-inner2.task')
        M.writeTask('lj-inner2.ptf')
        C, d = C.level(), d.level()
        return ([C[i:i + n] for i in range(0, n * n, n)], d)

'''
  The outer ellipsoidal approximation to a polytope given 
  as the convex hull of a set of points

    S = conv{ x1, x2, ... , xm }

  minimizes the volume of the enclosing ellipsoid,

    { x | || P*x-c ||_2 <= 1 }

  The volume is proportional to det(P)^{-1/n}, so the problem can
  be solved as

    maximize         t
    subject to       t       <= det(P)^(1/n)
                || P*xi - c ||_2 <= 1,  i=1,...,m
                P is PSD.
'''
def lownerjohn_outer(x):
    with Model("lownerjohn_outer") as M:
        M.setLogHandler(sys.stdout)
        m, n = len(x), len(x[0])

        # Setup variables
        t = M.variable("t", 1, Domain.greaterThan(0.0))
        P = det_rootn(M, t, n)
        c = M.variable("c", n, Domain.unbounded())

        # (1, Px-c) in cone
        M.constraint("qc",
                     Expr.hstack(Expr.ones(m),
                                 Expr.sub(Expr.mul(x, P),
                                          Var.reshape(Var.repeat(c, m), [m, n])
                                          )
                                 ),
                     Domain.inQCone())

        # Objective: Maximize t
        M.objective(ObjectiveSense.Maximize, t)
        M.solve()

        M.writeTask('lj-outer2.task')
        M.writeTask('lj-outer2.ptf')
        P, c = P.level(), c.level()
        return ([P[i:i + n] for i in range(0, n * n, n)], c)

##########################################################################


if __name__ == '__main__':

    # Vertices of a pentagon in 2D
    # p = [[0., 0.], [1., 3.], [5.5, 4.5], [7., 4.], [7., 1.], [3., -2.]]
    file_path_ablation = r"C:\tmp_patients\Pat_G12\Study_20200506\Series_003\CAS-One Recordings\2020-05-09_16-17-12\Segmentations\SeriesNo_27\SegmentationNo_0"
    dcm_img, reader = Reader.read_dcm_series(file_path_ablation)
    p = ell.get_surface_points(dcm_img)
    hull = ConvexHull(p)
    min_points = hull.min_bound
    max_points = hull.max_bound
    pts = np.zeros((len(min_points), 2))
    for i, el in enumerate(min_points):
        pts[i, 0] = min_points[i]
        pts[i, 1] = max_points[i]
    nVerts = len(p)
    polytope_Ab = pc.box2poly(pts)
    A = polytope_Ab.A
    b = polytope_Ab.b

  #%%
    # The hyperplane representation of the same polytope
    # A = [[-p[i][1] + p[i - 1][1], p[i][0] - p[i - 1][0]]
    #      for i in range(len(p))]
    # b = [A[i][0] * p[i][0] + A[i][1] * p[i][1] for i in range(len(p))]

    Po, co = lownerjohn_outer(p)
    Ci, di = lownerjohn_inner(A, b)

    # Visualization
    try:
        import numpy as np
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Polygon
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_patch(patches.Polygon(p, fill=False, color="red"))
        # The inner ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
        x = Ci[0][0] * np.cos(theta) + Ci[0][1] * np.sin(theta) + di[0]
        y = Ci[1][0] * np.cos(theta) + Ci[1][1] * np.sin(theta) + di[1]
        ax.plot(x, y)
        # The outer ellipse
        x, y = np.meshgrid(np.arange(-1.0, 8.0, 0.025),
                           np.arange(-3.0, 6.5, 0.025))
        ax.contour(x, y,
                   (Po[0][0] * x + Po[0][1] * y - co[0]) ** 2 + (Po[1][0] * x + Po[1][1] * y - co[1]) ** 2, [1])
        ax.autoscale_view()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # plt.show()
        fig.savefig('ellipsoid.png')
#%%
    except:
            print("Inner:")
            print("  C = ", Ci)
            print("  d = ", di)
            print("Outer:")
            print("  P = ", Po)
            print("  c = ", co)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='orange', label='Ablation Segmentation')
# ell.plot_ellipsoid(Po, co, 'blue', ax)
# ell.plot_ellipsoid(Ci, di, 'green', ax)
# plt.legend(loc='best')
# plt.show()
# fig.savefig('ellipsoid_dcm_3d.png')
# timestr = time.strftime("%H%M%S-%Y%m%d")
# file_dir = r"C:\develop\segmentation-eval\figures"
# filepath = os.path.join(file_dir, 'ellipsoid_' + timestr)
# gh.save(filepath, width=12, height=12, tight=True)

