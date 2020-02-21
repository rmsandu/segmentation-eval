# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import SimpleITK as sitk
import numpy as np
import numpy.linalg as la
from skimage.draw import ellipsoid

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
    # return np.asarray(A), np.squeeze(np.asarray(c))
    U, D, V = la.svd(np.asarray(A))
    rx, ry, rz = 1. / np.sqrt(D)
    # return the radii
    return rx, ry, rz


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


def get_surface_points(img_file):
    """

    :param img_file: DICOM like image in SimpleITK format
    :return: surface points of a 3d volume
    """
    if isinstance(img_file, str) is True:
        img = DicomReader.read_dcm_series(img_file, False)
    else:
        img = img_file
    try:
        spacing = img.GetSpacing()
    except Exception:
        print('not a DICOM Image. Please provide a Dicom IMG in SimpleITK format')

    contour = sitk.LabelContour(img, fullyConnected=False)
    contours = sitk.GetArrayFromImage(contour)
    vertices_locations = contours.nonzero()
    vertices_unravel = list(zip(vertices_locations[0], vertices_locations[1], vertices_locations[2]))
    vertices_list = [list(vertices_unravel[i]) for i in range(0, len(vertices_unravel), 10)]
    surface_points = np.array(vertices_list)
    return surface_points, spacing


def volume_outer_ellipsoid(img):
    """

    :param img: DICOM SimpleITK image
    :return: volume in ml
    """
    surface_points, spacing = get_surface_points(img)
    try:
        rx, ry, rz = mvee(surface_points, flag='outer')
    except Exception:
        print('The points cannot be approximated with an ellipsoid...returning NaN for volume')
        return np.nan
    vol_formula_outer = volume_ellipsoid_spacing(rx, ry, rz, spacing=spacing)
    return vol_formula_outer


def volume_inner_ellipsoid(img):
    """

    :param img: DICOM SimpleITK image
    :return: volume in ml
    """
    surface_points, spacing = get_surface_points(img)
    try:
        rx, ry, rz = mvee(surface_points, flag='inner')
    except Exception:
        print('The points cannot be approximated with an ellipsoid...returning NaN for volume')
        return np.nan
    vol_formula_inner = volume_ellipsoid_spacing(rx, ry, rz, spacing=spacing)
    return vol_formula_inner


if __name__ == '__main__':

    dir_name=r"C:\develop\Alblation-SSM-master\Alblation-SSM-master\ablation_segmentations\SeriesNo_3\SegmentationNo_2"
    points, spacing = get_surface_points(dir_name)
    rx, ry, rz = mvee(points, flag='outer')
    vol_formula_outer = volume_ellipsoid(rx, ry, rz)
    vol_spacing_outer = volume_ellipsoid_spacing(rx, ry, rz, spacing)
    print('volume formula:', vol_formula_outer)
    print('volume spacing ellipsoid scikit-image:', vol_spacing_outer)
    rx, ry, rz = mvee(points, flag='inner')
    vol_formula_inner = volume_ellipsoid(rx, ry, rz)
    vol_spacing_inner = volume_ellipsoid_spacing(rx, ry, rz, spacing)
    print('volume formula:', vol_formula_inner)
    print('volume spacing ellipsoid scikit-image:', vol_spacing_inner)
