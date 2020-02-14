# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
from os import walk
from mpl_toolkits.mplot3d import Axes3D
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

import DicomReader


def mvee(points, tol=0.0001, flag='outer'):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N
    #  inner ellipse: if d < 1+tol_dist
    #  outer ellipse : while err > tol:
    if flag == 'inner':
        while err < tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
    elif flag == 'outer':
        while err > tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
    c = u * points
    A = la.inv(points.T * np.diag(u) * points - c.T * c) / d
    return np.asarray(A), np.squeeze(np.asarray(c))


def plot_ellipsoid(A, centroid, color):
    U, D, V = la.svd(A)
    rx, ry, rz = 1. / np.sqrt(D)
    print(rx, ry, rz)
    u, v = np.mgrid[0:2 * pi:20j, -pi / 2:pi / 2:10j]
    x = rx * cos(u) * cos(v)
    y = ry * sin(u) * cos(v)
    z = rz * sin(v)
    E = np.dstack((x, y, z))
    E = np.dot(E, V) + centroid
    x, y, z = np.rollaxis(E, axis=-1)
    ax.plot_wireframe(x, y, z, cstride=1, rstride=1, color=color, alpha=0.2)


def contour_list(dirList):
    """
    Retrieves all contours of DICOM images in given file path.
    :param filepath: File path where DICOM images are stored
    :return: List of contours in array format.
    """
    img = DicomReader.read_dcm_series(dirList, False)
    contour = sitk.LabelContour(img, fullyConnected=False)
    contours = sitk.GetArrayFromImage(contour)
    return contours


if __name__ == '__main__':

    # points = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 3], [0, 0, -3], [0, 2, 0], [0, -2, 0]])
    points = np.array([[0.53135758, -0.25818091, -0.32382715],
                       [0.58368177, -0.3286576, -0.23854156, ],
                       [0.28741533, -0.03066228, -0.94294771],
                       [0.65685862, -0.09220681, -0.60347573],
                       [0.63137604, -0.22978685, -0.27479238],
                       [0.59683195, -0.15111101, -0.40536606],
                       [0.68646128, -0.046802, -0.68407367],
                       [0.62311759, -0.0101013, -0.1863324],
                       [0.62311759, -0.2101013, -0.1863324]])
    # onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    # dirList = r"C:\develop\Alblation-SSM-master\Alblation-SSM-master\ablation_segmentations\SeriesNo_3\SegmentationNo_2"
    pi = np.pi
    sin = np.sin
    cos = np.cos
    dir_name = r"C:\develop\Alblation-SSM-master\Alblation-SSM-master\ablation_segmentations"
    file_idx = 0

    # for (dirpath, dirnames, filenames) in walk(dir_name):
    #     print('dirpath: ', dirpath)
    #     print('dirnames: ', dirnames)
    #     try:
    #         vertices = contour_list(dirpath)
    #     except Exception:
    #         continue
    #     vertices_locations = vertices.nonzero()
    #     vertices_unravel = list(zip(vertices_locations[0], vertices_locations[1], vertices_locations[2]))
    #     vertices_list = [list(vertices_unravel[i]) for i in range(0, len(vertices_unravel), 10)]
    #     points = np.array(vertices_list)

    A_outer, centroid_outer = mvee(points, flag='outer')
    A_inner, centroid_inner = mvee(points, flag='inner')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='g')
    plot_ellipsoid(A_outer, centroid_outer, 'blue')
    plot_ellipsoid(A_inner, centroid_inner, 'red')
    dir_save = 'C:\develop\Alblation-SSM-master'
    plt.savefig(os.path.join(dir_save, 'ablation_ellipsoids') + str(file_idx) + '.png', dpi=300)
    plt.show()
    # file_idx +=1
    # plt.close()
    # todo: get volume from skimage including spacing
    #
