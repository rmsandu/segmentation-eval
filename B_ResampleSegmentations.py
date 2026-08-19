# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import SimpleITK as sitk


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
