# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:17:41 2017

@author: Raluca Sandu
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
import readDICOMFiles as Reader


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


    def set_image_object(self, ablationFilepath, tumorFilepath):
        self.tumorSegm = Reader.read_dcm_series(ablationFilepath)
        self.ablationSegm = Reader.read_dcm_series(tumorFilepath)


    def get_volume_ml(self, image):
        x_spacing, y_spacing, z_spacing = image.GetSpacing()
        image_nda = sitk.GetArrayFromImage(image)
        imageSegm_nda_NonZero = image_nda.nonzero()
        num_surface_voxels = len(list(zip(imageSegm_nda_NonZero[0],
                                          imageSegm_nda_NonZero[1],
                                          imageSegm_nda_NonZero[2])))
        if 0 >= num_surface_voxels:
            raise Exception('The mask image does not seem to contain an object.')     
        volume_object_ml = (num_surface_voxels * x_spacing * y_spacing * z_spacing)/1000
        return volume_object_ml


    def get_volume_residual(self):
        # Find the set difference of two arrays.
        # Return the sorted, unique values in ar1 that are not in ar2.
        tumorSegm_nda = sitk.GetArrayFromImage(self.tumorSegm)
        ablationSegm_nda = sitk.GetArrayFromImage(self.ablationSegm)
        difference_voxels = len(np.setdiff1d(tumorSegm_nda, ablationSegm_nda))
        x_spacing, y_spacing, z_spacing = self.tumorSegm.GetSpacing()
        volume_residual = (difference_voxels * x_spacing * y_spacing * z_spacing)/1000
        return volume_residual

    def set_volume_metrics(self):
        self.volume_tumor = self.get_volume_ml(self.tumorSegm)
        self.volume_ablation = self.get_volume_ml(self.ablationSegm)
        self.volume_residual = self.get_volume_residual()

        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(self.tumorSegm, self.ablationSegm)

        self.dice = overlap_measures_filter.GetDiceCoefficient()
        self.jaccard = overlap_measures_filter.GetDiceCoefficient()
        self.volumetric_overlap_error = 1. - overlap_measures_filter.GetJaccardCoefficient()
        self.volume_similarity = overlap_measures_filter.GetVolumeSimilarity()

    def get_volume_metrics_df(self):
        volumemetrics_dict = {
            'Tumour Volume (ml)': self.volume_tumor,
            'Ablation volume (ml)': self.volume_ablation,
            'Tumour residual volume (ml)': self.volume_residual,
            'Dice': self.dice,
            'Jaccard': self.jaccard,
            'Volume Overlap Error': self.volumetric_overlap_error,
            'Volume Similarity': self.volume_similarity,
            ' Tumour coverage ratio': self.coverage_ratio
        }

        return pd.DataFrame(volumemetrics_dict)



