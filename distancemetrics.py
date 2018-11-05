# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:43:49 2017

Mauerer Distance Map for the tumor Object (deemed as "tumor in this particular case")
Algorithm Pipeline :
    1. compute the contour surface of the object (
    face+edge+vertex connectivity : fullyConnected=True
    face connectivity only : fullyConnected=False
    2. convert from SimpleITK format to Numpy Array Img
    3. remove the zeros from the contour of the object, NOT from the distance map
    4. compute the number of 1's pixels in the contour
    5. instantiate the Signed Mauerer Distance map for the object (negative numbers also)
        # Multiply the binary surface segmentations with the distance maps. The resulting distance
        # maps contain non-zero values only on the surface (they can also contain zero on the surface)
@author: Raluca Sandu
"""
import numpy as np
import pandas as pd
from enum import Enum
import SimpleITK as sitk
import DicomReader as Reader



class DistanceMetrics(object):

    def __init__(self, ablation_path, tumor_path):
        ''' Read the images from the filepaths'''

        tumor_segmentation, series_reader = Reader.read_dcm_series(tumor_path)
        ablation_segmentation, series_reader = Reader.read_dcm_series(ablation_path)
        # Reader.print_dimensions_img('tumor', tumor_segmentation)
        # Reader.print_dimensions_img('ablation', ablation_segmentation)
        ''' init the enum fields for surface dist measures computer with simpleitk'''
        class SurfaceDistanceMeasuresITK(Enum):
            hausdorff_distance, max_distance, min_surface_distance, mean_surface_distance,\
             median_surface_distance, std_surface_distance = range(6)

        surface_distance_results = np.zeros((1, len(SurfaceDistanceMeasuresITK.__members__.items())))
        #%%
        tumor_surface = sitk.LabelContour(tumor_segmentation, fullyConnected=True)
        
        tumor_surface_array = sitk.GetArrayFromImage(tumor_surface)
        tumor_surface_array_NonZero = tumor_surface_array.nonzero()

        self.num_tumor_surface_pixels = len(list(zip(tumor_surface_array_NonZero[0],
                                                     tumor_surface_array_NonZero[1],
                                                     tumor_surface_array_NonZero[2])))
        # check if there is actually an object present
        if 0 >= self.num_tumor_surface_pixels:
            raise Exception('The mask image does not seem to contain an object.')
            
        # init signed mauerer distance as tumor metrics from SimpleITK
        self.tumor_distance_map = sitk.SignedMaurerDistanceMap(tumor_segmentation,
                                                               squaredDistance=False,
                                                               useImageSpacing=True)
        
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        try:
            hausdorff_distance_filter.Execute(tumor_segmentation, ablation_segmentation)
        except Exception as e:
            print(repr(e))
        surface_distance_results[0, SurfaceDistanceMeasuresITK.hausdorff_distance.value] =\
            hausdorff_distance_filter.GetHausdorffDistance()
            
        #%%
        ''' Mauerer Distance Map for the Ablation Object '''
        ablation_surface = sitk.LabelContour(ablation_segmentation)
        ablation_surface_mask_array = sitk.GetArrayFromImage(ablation_surface)
        ablation_mask_array_NonZero = ablation_surface_mask_array.nonzero()
        
        # Get the number of pixels in the mask surface by counting all pixels that are non-zero
        self.num_ablation_surface_pixels = len(list(zip(ablation_mask_array_NonZero[0],
                                                        ablation_mask_array_NonZero[1],
                                                        ablation_mask_array_NonZero[2])))
        
        if 0 >= self.num_ablation_surface_pixels:
            raise Exception('The mask image does not seem to contain an object.')
        # init Mauerer Distance
        self.ablation_distance_map = sitk.SignedMaurerDistanceMap(ablation_segmentation,
                                                                  squaredDistance=False,
                                                                  useImageSpacing=True)
 
        ablation_distance_map_array = sitk.GetArrayFromImage(self.ablation_distance_map)
       
        # compute the contours multiplied with the euclidean distances 
        self.ablation2tumor_distance_map = ablation_distance_map_array*tumor_surface_array
          
        # remove the zeros from the surface contour(indexes) from the distance maps '''
        self.surface_distances = list(self.ablation2tumor_distance_map[tumor_surface_array_NonZero]/-255) 

        #%% 
        ''' Compute the surface distances max, min, mean, median, std '''
        surface_distance_results[0, SurfaceDistanceMeasuresITK.max_distance.value] = np.max(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.min_surface_distance.value] = \
            np.min(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.mean_surface_distance.value] = \
            np.mean(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.median_surface_distance.value] = \
            np.median(self.surface_distances)
        surface_distance_results[0, SurfaceDistanceMeasuresITK.std_surface_distance.value] = \
            np.std(self.surface_distances)
        
        # Save to DataFrame
        self.surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index = list(range(1)),
                                           columns=[name for name, _ in SurfaceDistanceMeasuresITK.__members__.items()])
        # change the name of the columns
        self.surface_distance_results_df.columns = ['Hausdorff_AT', 'Maximum_AT', 'Minimum_AT', 'Mean_AT', 'Median_AT', 'Std_AT']

    #%%    
    def get_SitkDistances(self):
        return self.surface_distance_results_df
    
    def get_ablation_dist_map(self):
        return self.ablation2tumor_distance_map
    
    def get_surface_distances(self):
        return self.surface_distances