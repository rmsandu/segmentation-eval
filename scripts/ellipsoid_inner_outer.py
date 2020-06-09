# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import time
import VolumeMetrics
from mpl_toolkits.mplot3d import Axes3D
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from skimage.draw import ellipsoid

import DicomReader
import scripts.inner_ellipsoid as inner_ellipsoid
import utils.graphing as gh


def mvee(points, tol=0.001):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """

    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    u = np.ones(N) / N
    err = 1 + tol
    while err > tol:
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = la.norm(new_u - u)
        u = new_u

    c = u * points  # center of ellipsoid
    A = la.inv(points.T * np.diag(u) * points - c.T * c) / d

    # U, D, V = la.svd(np.asarray(A))
    # rx, ry, rz = 1. / np.sqrt(D)
    #
    # return rx, ry, rz
    return np.asarray(A), np.squeeze(np.asarray(c))


def volume_ellipsoid(a, b, c):
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')
    abc = [a, b, c]
    abc.sort(reverse=True)
    a = abc[0]
    b = abc[1]
    c = abc[2]

    # Volume
    vol = (4 * np.pi * a * b * c) / 3

    return vol / 1000


def volume_ellipsoid_spacing(a, b, c, spacing):
    """

    :param a: major semi-axis of an ellipsoid
    :param b: least semi-axis of an ellipsoid
    :param c:  minor semi-axis of an ellipsoid
    :param spacing: spacing of a grid, tuple like e.g. (1,  1, 1)
    :return: volume of an ellipsoid in ml, taking spacing into account
    """
    ellipsoid_array = ellipsoid(a, b, c, spacing)
    ellipsoid_non_zero = ellipsoid_array.nonzero()
    num_voxels = len(list(zip(ellipsoid_non_zero[0],
                              ellipsoid_non_zero[1],
                              ellipsoid_non_zero[2])))
    volume_object_ml = (num_voxels * spacing[0] * spacing[1] * spacing[2]) / 1000

    return volume_object_ml


def get_surface_points(dcm_img):
    """
    :param img_file: DICOM like image in SimpleITK format
    :return: surface points of a 3d volume
    """
    contour = sitk.LabelContour(dcm_img, fullyConnected=False)
    contours = sitk.GetArrayFromImage(contour)
    vertices_locations = contours.nonzero()
    vertices_unravel = list(zip(vertices_locations[0], vertices_locations[1], vertices_locations[2]))
    vertices_list = [list(vertices_unravel[i]) for i in range(0, len(vertices_unravel))]

    surface_points = np.array(vertices_list)
    return surface_points


def plot_ellipsoid(A, centroid, color, ax):
    """

    :param A: matrix
    :param centroid: center
    :param color: color
    :param ax: axis
    :return:
    """
    centroid = np.asarray(centroid)
    A = np.asarray(A)
    U, D, V = la.svd(A)
    rx, ry, rz = 1. / np.sqrt(D)
    u, v = np.mgrid[0:2 * np.pi:20j, -np.pi / 2:np.pi / 2:10j]
    x = rx * np.cos(u) * np.cos(v)
    y = ry * np.sin(u) * np.cos(v)
    z = rz * np.sin(v)
    E = np.dstack((x, y, z))
    E = np.dot(E, V) + centroid
    x, y, z = np.rollaxis(E, axis=-1)
    ax.plot_wireframe(x, y, z, cstride=1, rstride=1, color=color, alpha=0.2)
    ax.set_zlabel('Z-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_xlabel('X-Axis')


def get_ellipsoid_fit_volumes(img_file):
    if isinstance(img_file, str) is True:
        img = DicomReader.read_dcm_series(img_file, False)
    else:
        img = img_file
    try:
        spacing = img.GetSpacing()
    except Exception:
        print('not a DICOM Image. Please provide a DICOM IMG in SimpleITK format')
        return
    points = get_surface_points(dcm_img=img)
    # A, centroid = mvee(p)  # outer ellipsoid
    # U, D, V = la.svd(np.asarray(A))
    # rx, ry, rz = 1. / np.sqrt(D)
    B, centroid = inner_ellipsoid.FindMaximumVolumeInscribedEllipsoid(points)
    # volume_outer_ellipsoid = volume_ellipsoid(rx, ry, rz)
    volume_outer_ellipsoid = np.nan
    volume_inner_ellipsoid = np.sqrt(la.det(B) / 1000) * 4.19

    return volume_outer_ellipsoid, volume_inner_ellipsoid


def get_radii_from_matrix(A):
    """

    :param A: matrix describing ellipsoid
    :return: the radii
    """
    U, D, V = la.svd(np.asarray(A))
    rx, ry, rz = 1. / np.sqrt(D)
    return rx, ry, rz


if __name__ == '__main__':

    dir_name_ablation = r"D:\MAVERRIC\MAVERRIC_segmented_validated\Pat_B02_\Study_0\Series_7\CAS-One Recordings\2019-04-17_14-13-14\Segmentations\SeriesNo_28\SegmentationNo_0"
    dcm_img, reader = DicomReader.read_dcm_series(dir_name_ablation)
    points = get_surface_points(dcm_img)
    B_inner, centroid_inner, ax = inner_ellipsoid.FindMaximumVolumeInscribedEllipsoid(points)
    A_outer, centroid_outer = mvee(points)
    spacing = [0.75, 0.75, 1]
#%%
    vol_obj = VolumeMetrics.VolumeMetrics()
    vol_obj.set_image_object(ablation_segmentation=dcm_img)
    volume_ablation_eav = vol_obj.get_volume_ml(dcm_img)
    print('Éffective Ablation Volume with spacing into account [ml]:', volume_ablation_eav)

    rx_outer, ry_outer, rz_outer = get_radii_from_matrix(A_outer)
    vol_formula_outer = volume_ellipsoid(rx_outer, ry_outer, rz_outer)
    vol_spacing_outer = volume_ellipsoid_spacing(rx_outer, ry_outer, rz_outer, spacing)
    print('Outer Ellipsoid volume:', vol_formula_outer)
    print('Outer Ellipsoid volume with spacing ellipsoid scikit-image:', vol_spacing_outer)

    # rx_inner, ry_inner, rz_inner = B_inner[0, 0], B_inner[1, 1], B_inner[2,2]
    vol_formula_inner = np.sqrt(la.det(B_inner) / 1000) * 4.19
    # vol_spacing_inner = volume_ellipsoid_spacing(rx_inner, ry_inner, rz_inner, spacing)
    print('Inner Ellipsoid volume:', vol_formula_inner)



#%% PLOT
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', label='Ablation Segmentation')
    plot_ellipsoid(A_outer, centroid_outer, 'green', ax)
    # plot_ellipsoid(B_inner, centroid_inner, 'orange', ax)

    plt.legend(loc='best')
    plt.show()
    # timestr = time.strftime("%H%M%S-%Y%m%d")
    # file_dir = r"C:\develop\segmentation-eval\figures"
    # filepath = os.path.join(file_dir, 'ellipsoid_' + timestr)
    # gh.save(filepath, width=12, height=12, tight=True)

