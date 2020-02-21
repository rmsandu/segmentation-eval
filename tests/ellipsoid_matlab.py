# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
from matplotlib.patches import Ellipse
import os
from skimage.draw import ellipsoid, ellipsoid_stats, ellipse
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


def plot_3D_ellipsoid(A, centroid, color):
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
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax.plot_wireframe(x, y, z, cstride=1, rstride=1, color=color, alpha=0.2)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')


def plot_2D_ellipse(A, centroid):
    U, D, V = la.svd(A)
    rx, ry, rz = 1. / np.sqrt(D)
    # eigenvectors are the coefficients of an ellipse in general form
    # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
    # finding center of ellipse [eqn.19 and 20] from (**)

    phi = .5 * np.arctan((2. * ry) / (rx - rz))

    u = 1.  # x-position of the center
    v = 0.5  # y-position of the center
    a = rx  # radius on the x-axis
    b = ry  # radius on the y-axis
    t_rot = phi  # rotation angle

    t = np.linspace(0, 2 * pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])
    # 2-D rotation matrix

    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

    plt.plot(u + Ell[0, :], v + Ell[1, :])  # initial ellipse
    plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'darkorange')  # rotated ellipse
    plt.grid(color='lightgray', linestyle='--')
    plt.show()


def contour_list(dirList):
    """
    Retrieves all contours of DICOM images in given file path.
    :param filepath: File path where DICOM images are stored
    :return: List of contours in array format.
    """
    img = DicomReader.read_dcm_series(dirList, False)
    spacing = img.GetSpacing()
    print('Img Spacing: ', img.GetSpacing())
    print('Img Size: ', img.GetSize())
    contour = sitk.LabelContour(img, fullyConnected=False)
    contours = sitk.GetArrayFromImage(contour)
    return contours, spacing


if __name__ == '__main__':

    # onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    dirList = r"C:\develop\Alblation-SSM-master\Alblation-SSM-master\ablation_segmentations\SeriesNo_3\SegmentationNo_2"
    pi = np.pi
    sin = np.sin
    cos = np.cos
    dir_name = r"C:\develop\Alblation-SSM-master\Alblation-SSM-master\ablation_segmentations"
    file_idx = 0
    vertices, spacing = contour_list(dirList)
    vertices_locations = vertices.nonzero()
    vertices_unravel = list(zip(vertices_locations[0], vertices_locations[1], vertices_locations[2]))
    vertices_list = [list(vertices_unravel[i]) for i in range(0, len(vertices_unravel), 10)]
    points = np.array(vertices_list)

    # for (dirpath, dirnames, filenames) in walk(dir_name):
    #     print('dirpath: ', dirpath)
    #     print('dirnames: ', dirnames)
    #     try:
    #         vertices = get_surface_points(dirpath)
    #     except Exception:
    #         continue
    #     vertices_locations = vertices.nonzero()
    #     vertices_unravel = list(zip(vertices_locations[0], vertices_locations[1], vertices_locations[2]))
    #     vertices_list = [list(vertices_unravel[i]) for i in range(0, len(vertices_unravel), 10)]
    #     points = np.array(vertices_list)

    A_outer, centroid_outer = mvee(points, flag='outer')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    A_inner, centroid_inner = mvee(points, flag='inner')

    U, D, V = la.svd(A_inner)
    rx, ry, rz = 1. / np.sqrt(D)
    ellip_array = ellipsoid(rx, ry, rz, spacing=spacing)

    #%% 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='g')
    plot_3D_ellipsoid(A_outer, centroid_outer, 'blue')
    plot_3D_ellipsoid(A_inner, centroid_inner, 'red')
    # dir_save = 'C:\develop\Alblation-SSM-master'
    # plt.savefig(os.path.join(dir_save, 'ablation_ellipsoids') + str(file_idx) + '.png', dpi=300)
    # plt.show()
    # file_idx +=1
    # plt.close()
    # todo: get volume from skimage including spacing
    #   1. apply spacing
    #   2. plot a slice in 2D using the im_array
    #   3. compute the volume...
    #   2.a --> plot a slice with contour points 2.b --> plot the minim ellipse  2.c. --> plot the maximum ellipse
