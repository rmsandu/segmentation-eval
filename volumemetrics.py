# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:17:41 2017

@author: Raluca Sandu
"""

import numpy as np
import pandas as pd
from enum import Enum
import readDICOMFiles as Reader
from medpy import metric
import SimpleITK as sitk

class VolumeMetrics():


    def __init__(self, ablationFilepath , tumorFilepath):

        self.tumorSegm = Reader.read_dcm_series(ablationFilepath)
        self.ablationSegm = Reader.read_dcm_series(tumorFilepath)
        self.tumorSegm_nda = self.tumorSegm.GetArrayFromImage()
        self.ablationSegm_nda = self.ablationSegm.GetArrayFromImage()
        

       
    def get_volume_ml(self, image):
        
        x_spacing, y_spacing, z_spacing = image.GetSpacing()
        image_nda = sitk.GetArrayFromImage(image)
        imageSegm_nda_NonZero = image_nda.nonzero()
        num_surface_voxels = len(list(zip( imageSegm_nda_NonZero[0], imageSegm_nda_NonZero[1], imageSegm_nda_NonZero[2])))
                
        if 0 >= num_surface_voxels:
            raise Exception('The mask image does not seem to contain an object.')     
        volume_object_ml = (num_surface_voxels * x_spacing * y_spacing * z_spacing)/1000
        return volume_object_ml
    

    def get_volume_residual(self):
        # Find the set difference of two arrays.
        # Return the sorted, unique values in ar1 that are not in ar2.
        difference_voxels = len(np.setdiff1d(self.tumorSegm_nda, self.ablationSegm_nda))
        x_spacing, y_spacing, z_spacing = self.tumorSegm.GetSpacing()
        volume_residual = (difference_voxels * x_spacing * y_spacing * z_spacing)/1000
        return volume_residual
    
    
    def overlap_metrics(self):
        
        class OverlapMeasures(Enum):
            volume_tumor, volume_ablation, volume_residual,\
            dice, jaccard, volume_similarity, \
            volumetric_overlap_error, relative_vol_difference = range(8)
        
        overlap_results = np.zeros((1,len(OverlapMeasures.__members__.items())))  
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(self.tumorSegm, self.ablationSegm)
        overlap_results[0, OverlapMeasures.volume_tumor] = self.get_volume_ml(self.tumorSegm)
        overlap_results[0, OverlapMeasures.volume_ablation] = self.get_volume_ml(self.ablationSegm)
        overlap_results[0, OverlapMeasures.volume_residual] = self.get_volume_residual()
        overlap_results[0,OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
        overlap_results[0,OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
        overlap_results[0,OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
        overlap_results[0,OverlapMeasures.volumetric_overlap_error.value] = 1. - overlap_measures_filter.GetJaccardCoefficient()
        overlap_results[0,OverlapMeasures.relative_vol_difference.value] = metric.ravd(self.ablationSegm_nda, self.tumdorSegm_nda)
        
        self.overlap_results_df = pd.DataFrame(data=overlap_results, index = list(range(1)), 
                                  columns=[name for name, _ in OverlapMeasures.__members__.items()])
        self.overlap_results_df.columns = [' Tumour Volume (ml)', ' Ablation volume (ml)', \
                                           ' Tumour residual volume (ml)', \
                                           'Dice', 'Jaccard', 'Volume Similarity', 
                                           'Volume Overlap Error', 'Relative Volume Difference']
          

    def get_VolumeMetrics(self):
        return self.overlap_results_df