# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:17:41 2017

@author: Raluca Sandu
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
import DicomReader as Reader


class VolumeMetrics:

    def __init__(self):
        self.tumorSegm = None
        self.ablationSegm = None
        self.volume_tumor = None
        self.volume_ablation = None
        self.volume_residual = None
        self.coverage_ratio = None
        self.dice = None
        self.jaccard = None
        self.volumetric_overlap_error = None
        self.volume_similarity = None

    def set_image_object(self, ablation_filepath, tumor_filepath):
        self.tumorSegm = Reader.read_dcm_series(tumor_filepath, False)
        self.ablationSegm = Reader.read_dcm_series(ablation_filepath, False)

    def get_volume_ml(self, image):
        x_spacing, y_spacing, z_spacing = image.GetSpacing()
        image_nda = sitk.GetArrayFromImage(image)
        imageSegm_nda_NonZero = image_nda.nonzero()
        num_voxels = len(list(zip(imageSegm_nda_NonZero[0],
                                  imageSegm_nda_NonZero[1],
                                  imageSegm_nda_NonZero[2])))
        if 0 >= num_voxels:
            raise Exception('The mask image does not seem to contain an object.')
        volume_object_ml = (num_voxels * x_spacing * y_spacing * z_spacing) / 1000
        return volume_object_ml

    def get_volume_residual_coverage(self):
        tumor_nda = sitk.GetArrayFromImage(self.tumorSegm)
        ablation_nda = sitk.GetArrayFromImage(self.ablationSegm)
        # get the coordinates of the non-zero values from the binary masks
        # shape is array[slice,col,row]
        tumor_voxels_non_zero = np.transpose(np.nonzero(tumor_nda))
        ablation_voxels_non_zero = np.transpose(np.nonzero(ablation_nda))
        # transform into tuple-set
        tumor_set = set([tuple(x) for x in tumor_voxels_non_zero])
        ablation_set = set([tuple(x) for x in ablation_voxels_non_zero])
        # perform intersection on the common voxel coordinates between tumor and ablation
        intersection_tumor_ablation = np.array([x for x in tumor_set & ablation_set])
        num_voxels_intersection_non_zero = len(intersection_tumor_ablation)

        # Get the spacing
        x_spacing, y_spacing, z_spacing = self.tumorSegm.GetSpacing()
        # volume_residual = volume_tumor - volume_intersection ablation and tumor
        volume_intersection = (num_voxels_intersection_non_zero * x_spacing * y_spacing * z_spacing) / 1000
        volume_tumor = self.get_volume_ml(self.tumorSegm)
        volume_residual = volume_tumor - volume_intersection
        coverage_ratio = 1 - volume_residual/volume_tumor
        # coverage_ratio = 1- volume_intersection / volume_tumor
        return volume_residual, coverage_ratio

    def set_volume_metrics(self):
        self.volume_tumor = self.get_volume_ml(self.tumorSegm)
        self.volume_ablation = self.get_volume_ml(self.ablationSegm)
        self.volume_residual, self.coverage_ratio = self.get_volume_residual_coverage()

        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        try:
            overlap_measures_filter.Execute(self.tumorSegm, self.ablationSegm)
        except Exception as e:
            # print error message if the filter cannot be executed.
            # this is the case when the images are not in the same space
            print(repr(e))

        self.dice = overlap_measures_filter.GetDiceCoefficient()
        self.jaccard = overlap_measures_filter.GetDiceCoefficient()
        self.volumetric_overlap_error = 1. - overlap_measures_filter.GetJaccardCoefficient()
        self.volume_similarity = overlap_measures_filter.GetVolumeSimilarity()

    def get_volume_metrics_df(self):
        volume_metrics_dict = {
            'Tumour Volume (ml)': self.volume_tumor,
            'Ablation volume (ml)': self.volume_ablation,
            'Tumour residual volume (ml)': self.volume_residual,
            'Dice': self.dice,
            'Jaccard': self.jaccard,
            'Volume Overlap Error': self.volumetric_overlap_error,
            'Volume Similarity': self.volume_similarity,
            'Tumour coverage ratio': self.coverage_ratio
        }

        return pd.DataFrame(data=volume_metrics_dict, index=list(range(1)),
                            columns=[name for name in volume_metrics_dict.keys()])
