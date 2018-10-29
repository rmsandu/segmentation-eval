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
        Usage: newImage = paste_roi_image(source_img_plan, roi_mask)
        !!! We assume that the mask has the same dimensions as the file it has been derived from. !!!
    """
    newSize = image_source.GetSize()
    newOrigin = image_source.GetOrigin()
    # we assume we have the same spacing as the images have been taken with the same scanner    
    newSpacing = image_roi.GetSpacing()
    newDirection = image_roi.GetDirection()

    if image_source.GetSpacing() != image_roi.GetSpacing():
        print('the spacing of the source and derived mask differ')
    # re-cast the pixel type of the roi mask
    pixelID = image_source.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    image_roi = caster.Execute(image_roi)

    # black 3D image
    outputImage = sitk.Image(newSize, image_source.GetPixelIDValue())
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(newSpacing)
    outputImage.SetDirection(newDirection)
    # img.TransformContinuousIndexToPhysicalPoint
    # destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    # paste the roi mask into the re-sized image
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)
    return pasted_img

def recast_pixel_val(image_source, image_roi):
    pixelID = image_source.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    image_roi = caster.Execute(image_roi)
    return image_roi

def resize_resample_images(images):
    """ Resize all the images to the same dimensions, spacing and origin.
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
    try:
        img5 = images.img_plan + images.tumor_mask
    except Exception:
        print('images dont occupy the same physical space')

    dimension = images.img_plan.GetDimension()  #
    reference_direction = np.identity(dimension).flatten()
    reference_size_x = 512
    reference_origin = np.zeros(dimension)
    data = [images.img_plan, images.img_validation, images.ablation_mask, images.tumor_mask]

    reference_physical_size = np.zeros(dimension)
    for img in data:
        reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                      zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    reference_spacing = [reference_physical_size[0] / (reference_size_x - 1)] * dimension
    reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    reference_image = sitk.Image(reference_size, images.img_plan.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    print('reference:')
    print('segmentation direction', reference_image.GetDirection())
    print('segmentation origin', reference_image.GetOrigin())
    print('segmentation spacing', reference_image.GetSpacing())
    print('segmentation size', reference_image.GetSize())

    print('TUMOR Original:')
    print('segmentation direction', images.tumor_mask.GetDirection())
    print('segmentation origin', images.tumor_mask.GetOrigin())
    print('segmentation spacing', images.tumor_mask.GetSpacing())
    print('segmentation size', images.tumor_mask.GetSize())
    print('ABLATION Original:')
    print('ablation direction', images.ablation_mask.GetDirection())
    print('ablation origin', images.ablation_mask.GetOrigin())
    print('ablation spacing', images.ablation_mask.GetSpacing())
    print('ablation size', images.ablation_mask.GetSize())
    print('IMAGE PLAN ORIGINAL:')
    print('image direction', images.img_plan.GetDirection())
    print('image origin', images.img_plan.GetOrigin())
    print('image spacing', images.img_plan.GetSpacing())
    print('image size', images.img_plan.GetSize())
    #%% Paste the GT segmentation masks before transformation
    # tumor_mask_paste = (paste_roi_image(images.img_plan, images.tumor_mask))
    # ablation_mask_paste = (paste_roi_image(images.img_validation, images.ablation_mask))
    # images.tumor_mask = tumor_mask_paste
    # images.ablation_mask = ablation_mask_paste

    # %%  Apply transforms
    data_resized = []

    for idx,img in enumerate(data):
        #%% Set Transformation
        transformTranslation = sitk.AffineTransform(dimension) # use affine transform with 3 dimensions
        transformTranslation.SetMatrix(img.GetDirection()) # set the cosine direction matrix
        # transformTranslation.SetTranslation(reference_origin - np.array(img.GetOrigin())) # set the translation.
        transformTranslation.SetTranslation(np.array(img.GetOrigin() - reference_origin))
        transformTranslation.SetCenter(reference_center)
        print('Transform Matrix3:', transformTranslation.GetMatrix())
        print('Params after translation', transformTranslation.GetParameters())
        print('translation3', transformTranslation.GetTranslation())
        # transformScaling = sitk.AffineTransform(dimension)
        # matrix = np.array(transformScaling.GetMatrix()).reshape((dimension, dimension))
        # size = img.GetSize()
        # spacing = img.GetSpacing()
        # physical_size = [size[0]*spacing[0], size[1]*spacing[1], size[2]*spacing[2]]
        # x_scale, y_scale, z_scale = reference_physical_size[0]/physical_size[0], reference_physical_size[1]/physical_size[1], reference_physical_size[2]/physical_size[2]
        #
        # matrix[0, 0] = x_scale
        # matrix[1, 1] = y_scale
        # matrix[2, 2] = z_scale
        #
        # transformScaling.SetMatrix(matrix.ravel())
        # completeTransform = sitk.Transform(transformTranslation)
        # completeTransform.AddTransform(transformScaling)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(np.array(transformTranslation.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transformTranslation)
        centered_transform.AddTransform(centering_transform)
        print('Params after Centering', centered_transform.GetParameters())
        #%% set all  output image parameters: origin, spacing, direction, starting index, and size.
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        # resampler.SetDefaultPixelValue(img.GetPixelIDValue())
        # resampler.SetSize(reference_image.GetSize())
        # resampler.SetOutputSpacing(reference_image.GetSpacing())
        # resampler.SetOutputOrigin(reference_image.GetOrigin())
        # resampler.SetOutputDirection(reference_image.GetDirection())
        resampler.SetTransform(centered_transform)
        if idx==0 or idx==1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif idx==1 or idx==3:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        print('Image: ', idx)
        print('BEFORE RESAMPLING:')
        print('direction', img.GetDirection())
        print('origin', img.GetOrigin())
        print('spacing', img.GetSpacing())
        print('size', img.GetSize())
        print('center', np.array(
            img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0)))

        resampled_img = resampler.Execute(img)

        print('Image: ', idx)
        print('AFTER RESAMPLING:')
        print('direction', resampled_img.GetDirection())
        print('origin', resampled_img.GetOrigin())
        print('spacing', resampled_img.GetSpacing())
        print('size', resampled_img.GetSize())
        print('center', np.array(
            resampled_img.TransformContinuousIndexToPhysicalPoint(np.array(resampled_img.GetSize()) / 2.0)))

        data_resized.append(resampled_img)



    # assuming the order stays the same, reassigng back to tuple
    resized_imgs = tuple_resized_imgs(img_plan=data_resized[0],
                                      img_validation=data_resized[1],
                                      ablation_mask=data_resized[2],
                                      tumor_mask=data_resized[3])
    return resized_imgs
