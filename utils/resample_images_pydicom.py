# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""


import pydicom
import scipy
import numpy as np

def resample_pydicom_img(self, image, scan, new_spacing=[1, 1, 1]):
    id = 0
    imgs_to_process = np.load(output_path + 'fullimages_{}.npy'.format(id))
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing


print
"Shape before resampling\t", imgs_to_process.shape
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1, 1, 1])
print
"Shape after resampling\t", imgs_after_resamp.shape