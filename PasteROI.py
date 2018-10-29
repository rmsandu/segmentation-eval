import SimpleITK as sitk
import numpy as np

def paste_roi_image(image_source, image_roi):
    """ Resize ROI binary mask to size, dimension, origin of its source/original img.
        Usage: newImage = paste_roi_image(source_img_plan, roi_mask)
        !!! We assume that the mask has the same dimensions as the file it has been derived from. !!!
    """
    # get the size and the origin from the source image
    newSize = image_source.GetSize()
    newOrigin = image_source.GetOrigin()
    # get the spacing and the direction from the mask
    newSpacing = image_roi.GetSpacing()
    newDirection = image_roi.GetDirection()

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
    print('DestinationIndex:', destinationIndex)
    print('DestinationPhysical:', image_roi.GetOrigin())
    return pasted_img

def recast_pixel_val(image_source, image_roi):
    pixelID = image_source.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    image_roi = caster.Execute(image_roi)
    return image_roi

def resize_segmentation(image_source, image_roi):

    image_roi = recast_pixel_val(image_source, image_roi)

    new_segmentation = sitk.Resample(image_roi, image_source.GetSize(),
                                     sitk.Transform(),
                                     sitk.sitkNearestNeighbor,
                                     image_source.GetOrigin(),
                                     image_source.GetSpacing(),
                                     image_source.GetDirection(),
                                     0,
                                     image_roi.GetPixelID())
    return new_segmentation