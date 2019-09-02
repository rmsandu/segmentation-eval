import SimpleITK as sitk

def recast_pixel_val(image_source, image_roi):
    """
    Recast pixel value to be the same for segmentation and original image, othewise SimpleITK complains.
    :param image_source:
    :param image_roi:
    :return:
    """
    pixelID = image_source.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    image_roi = caster.Execute(image_roi)
    return image_roi


def paste_roi_image(image_source, image_roi, reference_size=None):
    """ Resize ROI binary mask to size, dimension, origin of its source/original img.
        Usage: newImage = paste_roi_image(source_img_plan, roi_mask)
        Use only if the image segmentation ROI has the same spacing as the image source
    """
    # get the size and the origin from the source image
    if reference_size:
        newSize = reference_size
    else:
        newSize = image_source.GetSize()

    newOrigin = image_source.GetOrigin()
    # get the spacing and the direction from the mask or the image if they are identical
    newSpacing = image_source.GetSpacing()
    newDirection = image_source.GetDirection()

    # re-cast the pixel type of the roi mask
    image_roi = recast_pixel_val(image_source, image_roi)

    # black 3D image
    outputImage = sitk.Image(newSize, image_source.GetPixelIDValue())
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(newSpacing)
    outputImage.SetDirection(newDirection)
    # img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    # paste the roi mask into the re-sized image
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)

    return pasted_img


def resample_segmentations(image_source, image_roi):
    """
    If the spacing of the segmentation is different from its original image, use RESAMPLE
    Resample parameters:  identity transformation, zero as the default pixel value, and nearest neighbor interpolation
    (assuming here that the origin of the original segmentation places it in the correct location with respect to the original image)
    :param image_source:
    :param image_roi:
    :return: new_segmentation of the image_roi
    """
    # image_roi = recast_pixel_val(image_source, image_roi)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_source)  # the ablation mask
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_img = resampler.Execute(image_roi)  # the tumour mask
    return resampled_img


if __name__ == '__main__':
    pass
