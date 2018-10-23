# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:00:45 2018

@author: Raluca Sandu
"""

import SimpleITK as sitk


def paste_roi_image(image_source, image_roi):
    """ Resize ROI binary mask to size, dimension, origin of its source/original img.
        Usage: newImage = resize_image(source_img_plan, roi_mask)
    """   
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
    # TO DO: modify the pixel type!!
    outputImage = sitk.Image(newSize, sitk.sitkInt16)
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(newSpacing)
    outputImage.SetDirection(newDirection)    
    # transform from physical point to index the origin of the ROI image
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    # paste the roi mask into the re-sized image
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)
    return pasted_img


def paste_roi_imageMaxSize(image_plan, image_validation, image_roi,  tumor_path,ablation_path, flag_source_img):

    """ Resize all the masks to the same dimensions, spacing and origin.
        Usage: newImage = resize_image(source_img_plan, source_img_validation, ROI(ablation/tumor)_mask)
        1. translate to same origin
        2. largest number of slices and interpolate the others.
        3. same resolution 1x1x1 mm3 - resample
        4. (physical space)

    """
        # 1. translate to same origin
        # 2.
    sizeP = image_plan.GetSize()
    sizeV = image_validation.GetSize()
    # we assume that the number of rol and cols  is always 512x512
    # create a new Size (x,y,z) and set the number of slices with the max number of slice from the two
    if sizeP[2] > sizeV[2]:
        newSize = (sizeP[0], sizeP[1], sizeP[2])
    else:
        newSize = (sizeP[0], sizeP[1], sizeV[2])
        
    originV = image_validation.GetOrigin()   
    originP = image_plan.GetOrigin()
    
    # create a new origin tuple format
    newOrigin = ()
    for idx, val in enumerate(originP):
        if originP[idx] < originV[idx]:
            newOrigin = newOrigin + (originP[idx],)
        else:
            newOrigin = newOrigin + (originV[idx],)
       
    # re-cast the pixel type of the roi mask
    pixelID = image_plan.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType( pixelID )
    image_roi = caster.Execute(image_roi)

    spacingP = image_plan.GetSpacing()
    spacingV = image_validation.GetSpacing()

    if (spacingP != spacingV):
        print('the spacing of the 2 images differ')
        print('ablation_path',ablation_path)
        print('tumor_path', tumor_path)

    # set spacing from the source image that the mask was derived from
    if flag_source_img == 0:
        # tumor mask so use image plan where it was derived from
        image = image_plan
    elif flag_source_img == 1:
        image = image_validation

    spacingP = image.GetSpacing()
    directionP = image.GetDirection()
    outputImage = sitk.Image(newSize, sitk.sitkInt16)
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(spacingP)
    outputImage.SetDirection(directionP)    
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)
    
    return pasted_img
