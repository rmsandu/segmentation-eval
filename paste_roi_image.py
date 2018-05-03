# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:00:45 2018

@author: Raluca Sandu
"""

import SimpleITK as sitk


def paste_roi_image(image_plan, image_validation, image_roi):
    ''' Usage: newImage = resize_image(source_img_plan, source_img_validation, roi_tumor_mask) '''
        
    sizeP = image_plan.GetSize()
    sizeV = image_validation.GetSize()
    # we assume that the number of rol and cols  is always 512x512
    if sizeP[2] > sizeV[2]:
        newSize = (sizeP[0], sizeP[1], sizeP[2])
    else:
        newSize = (sizeP[0], sizeP[1], sizeV[2])
        
    originV = image_validation.GetOrigin()   
    originP = image_plan.GetOrigin()
    newOrigin = ()
    for idx, val in enumerate(originP):
        if  originP[idx] < originV[idx]:
            newOrigin = newOrigin + (originP[idx],)
        else:
            newOrigin = newOrigin + (originV[idx],)
    
    spacingP = image_plan.GetSpacing()
    directionP = image_plan.GetDirection()
    
    # black 3D image
    outputImage = sitk.Image(newSize, sitk.sitkInt32)
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(spacingP)
    outputImage.SetDirection(directionP)    
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)

    return pasted_img