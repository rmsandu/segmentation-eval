# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:00:45 2018

@author: Raluca Sandu
"""

import SimpleITK as sitk


def paste_roi_image(image, image_roi):
    ''' Usage: newImage = resize_image(source_img_plan,roi_tumor_mask) '''
    size = image.GetSize()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    # black 3D image
    output_image = sitk.Image(size, sitk.sitkInt32)
    output_image.SetOrigin(origin)
    output_image.SetSpacing(spacing)
    output_image.SetDirection(direction)    
    destinationIndex = image.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    pasted_img = sitk.Paste(output_image, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)
    return pasted_img