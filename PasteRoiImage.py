# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:00:45 2018

@author: Raluca Sandu
"""
import numpy as np
import collections
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
    caster.SetOutputPixelType(pixelID)
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


def paste_roi_imageMaxSize(images):
    """ Resize all the masks to the same dimensions, spacing and origin.
        Usage: newImage = resize_image(source_img_plan, source_img_validation, ROI(ablation/tumor)_mask)
        1. translate to same origin
        2. largest number of slices and interpolate the others.
        3. same resolution 1x1x1 mm3 - resample
        4. (physical space)
        Slice Thickness (0018,0050)
        ImagePositionPatient (0020,0032)
        ImageOrientationPatient (0020,0037)
        PixelSpacing (0028,0030)
        Frame Of Reference UID (0020,0052)
    """
    # %% Define tuple to store the images
    tuple_resized_imgs = collections.namedtuple('tuple_resized_imgs',
                                                ['img_plan',
                                                 'img_validation',
                                                 'ablation_mask',
                                                 'tumor_mask'])
    # %% Create Reference image with zero origin, identity direction cosine matrix and isotropic dimension
    dimension = images[0].GetDimension()  #
    reference_physical_size = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = [512] * dimension
    reference_origin = np.zeros(dimension)
    data = [images.image_plan, images.image_validation, images.ablation_image, images.tumor_image]
    for img in data:
        reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                      zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]
    # reference_spacing = [1,1,1]
    reference_image = sitk.Image(reference_size, images[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    # %%  Apply transforms
    data_resized = []
    for img in data:
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)
        # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
        # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
        # no new labels are introduced.
        resampled_img = sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
        data_resized.append(resampled_img)

    # assuming the order stays the same, reassigng back to tuple
    resized_imgs = tuple_resized_imgs(img_plan=data_resized[0],
                                      img_validation=data_resized[1],
                                      ablation_mask=data_resized[2],
                                      tumor_mask=data_resized[3])
    return resized_imgs
    # %%

    # sizeP = image_plan.GetSize()
    # sizeV = image_validation.GetSize()
    # # we assume that the number of rol and cols  is always 512x512
    # # create a new Size (x,y,z) and set the number of slices with the max number of slice from the two
    # if sizeP[2] > sizeV[2]:
    #     newSize = (sizeP[0], sizeP[1], sizeP[2])
    # else:
    #     newSize = (sizeP[0], sizeP[1], sizeV[2])
    #
    # originV = image_validation.GetOrigin()
    # originP = image_plan.GetOrigin()
    #
    # # create a new origin tuple format
    # newOrigin = ()
    # for idx, val in enumerate(originP):
    #     if originP[idx] < originV[idx]:
    #         newOrigin = newOrigin + (originP[idx],)
    #     else:
    #         newOrigin = newOrigin + (originV[idx],)
    #
    # # re-cast the pixel type of the roi mask
    # pixelID = image_plan.GetPixelID()
    # caster = sitk.CastImageFilter()
    # caster.SetOutputPixelType( pixelID )
    # image_roi = caster.Execute(image)
    #
    # spacingP = image_plan.GetSpacing()
    # spacingV = image_validation.GetSpacing()
    #
    #
    #
    # # set spacing from the source image that the mask was derived from
    # if flag_source_img == 0:
    #     # tumor mask so use image plan where it was derived from
    #     image = image_plan
    # elif flag_source_img == 1:
    #     image = image_validation
    #
    # spacingP = image.GetSpacing()
    # directionP = image.GetDirection()
    # outputImage = sitk.Image(newSize, sitk.sitkInt16)
    # outputImage.SetOrigin(newOrigin)
    # outputImage.SetSpacing(spacingP)
    # outputImage.SetDirection(directionP)
    # destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    # pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)
    #
    # return pasted_img
