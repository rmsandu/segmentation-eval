# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
from mpl_toolkits.mplot3d import axes3d
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from scipy.spatial import ConvexHull


def random_point_ellipsoid(a, b, c, x0, y0, z0):
    """Generate a random point on an ellipsoid defined by a,b,c"""
    u = np.random.rand()
    v = np.random.rand()
    theta = u * 2.0 * np.pi
    phi = np.arccos(2.0 * v - 1.0)
    sinTheta = np.sin(theta);
    cosTheta = np.cos(theta);
    sinPhi = np.sin(phi);
    cosPhi = np.cos(phi);
    rx = a * sinPhi * cosTheta;
    ry = b * sinPhi * sinTheta;
    rz = c * cosPhi;
    return rx, ry, rz


def random_point_ellipse(W, d):
    # random angle
    alpha = 2 * np.pi * np.random.random()
    # vector on that angle
    pt = np.array([np.cos(alpha), np.sin(alpha)])
    # Ellipsoidize it
    return W @ pt + d


def GetRandom(dims, Npts):
    if dims == 2:
        W = sklearn.datasets.make_spd_matrix(2)
        d = np.array([2, 3])
        points = np.array([random_point_ellipse(W, d) for i in range(Npts)])
    elif dims == 3:
        points = np.array([random_point_ellipsoid(3, 5, 7, 2, 3, 3) for i in range(Npts)])
    else:
        raise Exception("dims must be 2 or 3!")
    noise = np.random.multivariate_normal(mean=[0] * dims, cov=0.2 * np.eye(dims), size=Npts)
    return points + noise


def GetHull(points):
    dim = points.shape[1]
    hull = ConvexHull(points)
    A = hull.equations[:, 0:dim]
    b = hull.equations[:, dim]
    return A, -b, hull  # Negative moves b to the RHS of the inequality


def Plot(points, hull, B, d):
    fig = plt.figure()
    if points.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1])
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        display_points = np.array([random_point_ellipse([[1, 0], [0, 1]], [0, 0]) for i in range(100)])
        display_points = display_points @ B + d
        ax.scatter(display_points[:, 0], display_points[:, 1])
    elif points.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        display_points = np.array([random_point_ellipsoid(1, 1, 1, 0, 0, 0) for i in range(len(points))])
        display_points = display_points @ B + d
        ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2])
        return ax
    plt.show()


def FindMaximumVolumeInscribedEllipsoid(points):
    """Find the inscribed ellipsoid of maximum volume. Return its matrix-offset form."""
    dim = points.shape[1]
    A, b, hull = GetHull(points)

    B = cp.Variable((dim, dim), PSD=True)  # Ellipsoid
    d = cp.Variable(dim)  # Center

    constraints = [cp.norm(B @ A[i], 2) + A[i] @ d <= b[i] for i in range(len(A))]
    prob = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
    optval = prob.solve()
    if optval == np.inf:
        raise Exception("No solution possible!")
    print(f"Optimal value: {optval}")

    # ax = Plot(points, hull, B.value, d.value)
    # return B.value, d.value, ax

    return B.value, d.value


if __name__ == '__main__':
    FindMaximumVolumeInscribedEllipsoid(GetRandom(dims=2, Npts=100))
    FindMaximumVolumeInscribedEllipsoid(GetRandom(dims=3, Npts=100))
