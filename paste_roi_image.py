# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:00:45 2018

@author: Raluca Sandu
"""

import SimpleITK as sitk


def paste_roi_image(image_source, image_roi):
    ''' Usage: newImage = resize_image(source_img_plan, roi_mask) '''
        
    newSize = image_source.GetSize()
    newOrigin = image_source.GetOrigin()   
    # we assume we have the same spacing as the images have been taken with the same scanner    
    newSpacing = image_source.GetSpacing() 
    newDirection = image_source.GetDirection()
    
    # re-cast the pixel type of the roi mask
    pixelID = image_source.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType( pixelID )
    image_roi = caster.Execute(image_roi)
 
    # black 3D image
    outputImage = sitk.Image(newSize, sitk.sitkInt16)
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(newSpacing)
    outputImage.SetDirection(newDirection)    
    # transform from physical point to index the origin of the ROI image
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)

    return pasted_img