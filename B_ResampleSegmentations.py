# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import SimpleITK as sitk
import numpy as np
import scipy


class ResizeSegmentation(object):

    def __init__(self, ablation_segmentation, tumor_segmentation, ablation_source_ct):

        self.tumor_segmentation = tumor_segmentation
        self.ablation_segmentation = ablation_segmentation
        self.ablation_source_ct = ablation_source_ct

    def resample_segmentation(self):
        """
        If the spacing of the segmentation is different from its original image, use RESAMPLE
        Resample parameters:  identity transformation, zero as the default pixel value, and nearest neighbor interpolation
        (assuming here that the origin of the original segmentation places it in the correct location w.r.t  original image)
        :return: new_segmentation of the image_roi
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.ablation_source_ct)  # the ablation mask
        resampler.SetDefaultPixelValue(0)
        # use NearestNeighbor interpolation for the ablation&tumor segmentations so no new labels are generated
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetSize(self.ablation_source_ct.GetSize())
        resampler.SetOutputSpacing(self.ablation_source_ct.GetSpacing())
        resampler.SetOutputDirection(self.ablation_source_ct.GetDirection())
        resampled_tumor = resampler.Execute(self.tumor_segmentation)  # the tumour mask
        resampled_ablation = resampler.Execute(self.ablation_segmentation)  # the ablation mask
        return resampled_tumor, resampled_ablation

    def resample_segmentation_pydicom(self, scan, new_spacing=[1, 1, 1]):
        """

        :param scan:
        :param new_spacing:
        :return:
        """
        image = self.tumor_segmentation
        id = 0
        output_path = r""
        imgs_to_process = np.load(output_path + 'fullimages_{}.npy'.format(id))

        # def resample(image, scan, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing
        spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
        spacing = np.array(list(spacing))

        resize_factor = spacing / new_spacing
        new_real_shape = self.image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

        return image, new_spacing

    def recast_pixel_val(self, image_source, image_roi):
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

    def paste_roi_image(self, image_source, image_roi, reference_size=None):
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
        image_roi = ResizeSegmentation.recast_pixel_val(image_source, image_roi)

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
