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
    if sizeP[2] > sizeV[2]:
        newSize = (sizeP[0], sizeP[1], sizeP[2])
    else:
        newSize = (sizeP[0], sizeP[1], sizeV[2])
     
    if sizeP[1]  > sizeV[1]:
        newSize = (sizeP[0], sizeP[1], newSize[2])
    else:
        newSize = (sizeP[0], sizeV[1], newSize[2])
    
    if sizeP[0] > sizeV[0]:
        newSize = (sizeP[0], newSize[1], newSize[2])
    else:
        newSize = (sizeV[0], newSize[1], newSize[2])
        
        
        
    originV = image_validation.GetOrigin()   
    originP = image_plan.GetOrigin()
    newOrigin = ()
    for idx, val in enumerate(originP):
        if  abs(originP[idx]) > abs(originV[idx]):
            newOrigin = newOrigin + (originP[idx],)
        else:
            newOrigin = newOrigin + (originV[idx],)
     
    # we assume we have the same spacing as the images have been taken with the same scanner    
    spacingP = image_plan.GetSpacing() 
#    newOriginResampled = (newOrigin[0]*newSize[0]*spacingP[0], newOrigin[1]*newSize[1]*spacingP[1],\
#                        newOrigin[2]*newSize[2]*spacingP[2])
    directionP = image_plan.GetDirection()
    
    # re-cast the pixel type of the roi mask
    pixelID = image_plan.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType( pixelID )
    image_roi = caster.Execute(image_roi)
 
    # black 3D image
    outputImage = sitk.Image(newSize, sitk.sitkInt16)
#    outputImage.SetPixelAsComplexFloat64() 
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(spacingP)
    outputImage.SetDirection(directionP)    
    # transform from physical point to index the origin of the ROI image
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)

    return pasted_img