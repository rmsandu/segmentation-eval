# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:00:45 2018

@author: Raluca Sandu
"""
import numpy as np
from recordclass import recordclass
import SimpleITK as sitk
import PasteROI_Segmentation2OriginalSize as PasteROI

def print_image_dimensions(image, name):
    """
    Print Image Parameters.
    :param image: SimpleITK object image
    :param name: Name of the image as defined by the user
    :return:
    """
    print(name + ' RESIZED:')
    print(name + ' direction: ', image.GetDirection())
    print(name + ' origin: ', image.GetOrigin())
    print(name + ' spacing: ', image.GetSpacing())
    print(name + ' size: ', image.GetSize())
    print(name + ' pixel: ', image.GetPixelIDTypeAsString())


def resize_resample_images(images, reference_spacing, reference_size, print_flag=False):
    """ Resize all the images to the same dimensions and space.
        Usage: newImage = resize_image(tuple_images(img_plan, img_validation, ablation_mask, tumor_mask), [1.0, 1.0, 1.0], [512, 512, 500])
        1. Create a Blank (black) Reference IMAGE where we will "copy" all the rest of the images
        2. Paste (same spacing) or Resample (different spacing) GT Segmentations to the same size as the Original Images
        3. Resample all 4 images in the same spacing and physical space
    """
    # %% Define tuple to store the images
    tuple_resized_imgs = recordclass('tuple_resized_imgs',
                                                ['img_plan',
                                                 'img_validation',
                                                 'ablation_mask',
                                                 'tumor_mask'])
    # %% Create Reference image with zero origin, identity direction cosine matrix and isotropic dimension
    dimension = images.img_plan.GetDimension()  #
    reference_direction = np.identity(dimension).flatten()
    reference_origin = np.zeros(dimension)
    reference_image = sitk.Image(reference_size, images.img_plan.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    #%% PRINT DIMENSIONS
    if print_flag:
        print_image_dimensions(reference_image, 'REFERENCE IMAGE')
        print_image_dimensions(images.tumor_mask, 'TUMOR Original')
        print_image_dimensions(images.ablation_mask, 'ABLATION Original')
        print_image_dimensions(images.img_plan, 'IMAGE PLAN ORIGINAL')
        print_image_dimensions(images.img_validation, 'IMAGE VALIDATION ORIGINAL')

    #%% RESIZE the ROI SEGMENTATIONS before RESAMPLING and RESIZING TO REFERENCE IMAGE
    if images.img_plan.GetSpacing() == images.tumor_mask.GetSpacing():
        ## use PASTE if the original img and segmentation have the same spacing
        images.tumor_mask = (PasteROI.paste_roi_image(images.img_plan, images.tumor_mask))
        images.ablation_mask = (PasteROI.paste_roi_image(images.img_validation, images.ablation_mask))
    else:
        ## Resample the segmentations to the size of their original image
        images.tumor_mask = PasteROI.resize_segmentation(images.img_plan, images.tumor_mask)
        images.ablation_mask = PasteROI.resize_segmentation(images.img_validation, images.ablation_mask)

    # PRINT DIMENSIONS AFTER RESIZING GT Segmentation MASKS ROI
    if print_flag:
        print_image_dimensions(images.tumor_mask, 'TUMOR ROI RESIZED')
        print_image_dimensions(images.ablation_mask, 'ABLATION ROI RESIZED')

    #%%  APPLY TRANSFORMS TO RESIZE AND RESAMPLE ALL THE DATA IN THE SAME SPACE
    data = [images.img_plan, images.img_validation, images.ablation_mask, images.tumor_mask]
    data_resized = []
    for idx,img in enumerate(data):
        # Set Transformation
        transformTranslation = sitk.AffineTransform(dimension) # use affine transform with 3 dimensions
        transformTranslation.SetMatrix(img.GetDirection()) # set the cosine direction matrix
        transformTranslation.SetTranslation(np.array(img.GetOrigin() - reference_origin))
        transformTranslation.SetCenter(reference_center)
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(np.array(transformTranslation.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transformTranslation)
        centered_transform.AddTransform(centering_transform)
        # set all  output image parameters: origin, spacing, direction, starting index, and size with RESAMPLE
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetTransform(centered_transform)
        resampler.SetDefaultPixelValue(0)
        if idx==0 or idx==1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif idx==2 or idx==3:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_img = resampler.Execute(img)
        data_resized.append(resampled_img)
        if print_flag:
            print_image_dimensions(resampled_img, 'RESAMPLED IMAGE ' + str(idx))

    # assuming the order stays the same, reassigng back to tuple
    resized_imgs = tuple_resized_imgs(img_plan=data_resized[0],
                                      img_validation=data_resized[1],
                                      ablation_mask=data_resized[2],
                                      tumor_mask=data_resized[3])
    return resized_imgs
