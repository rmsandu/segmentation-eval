# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:17:41 2017

@author: Raluca Sandu
"""

import numpy as np
import pandas as pd
from enum import Enum
from medpy import metric
import SimpleITK as sitk
#%%
class VolumeMetrics(object):
    
    def __init__(self,maskFile, referenceFile):
        
        reference = sitk.ReadImage(referenceFile, sitk.sitkUInt8)
        mask = sitk.ReadImage(maskFile,sitk.sitkUInt8)
        
        ref_array = sitk.GetArrayFromImage(reference) # convert the sitk image to numpy array 
        seg_array = sitk.GetArrayFromImage(mask)
        
        class OverlapMeasures(Enum):
            dice, jaccard, volume_similarity, volumetric_overlap_error, relative_vol_difference = range(5)
        
        overlap_results = np.zeros((1,len(OverlapMeasures.__members__.items())))  
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        
        overlap_measures_filter.Execute(reference, mask)
        overlap_results[0,OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
        overlap_results[0,OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
        overlap_results[0,OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
        overlap_results[0,OverlapMeasures.volumetric_overlap_error.value] = 1. - overlap_measures_filter.GetJaccardCoefficient()
        overlap_results[0,OverlapMeasures.relative_vol_difference.value] = metric.ravd(seg_array, ref_array)
        
        self.overlap_results_df = pd.DataFrame(data=overlap_results, index = list(range(1)), 
                                  columns=[name for name, _ in OverlapMeasures.__members__.items()])
        self.overlap_results_df.columns = ['Dice', 'Jaccard', 'Volume Similarity', 'Volume Overlap Error', 'Relative Volume Difference']
        

    
    #%%

    def get_VolumeMetrics(self):
        return self.overlap_results_df