# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:55:23 2018

@author: Raluca Sandu
"""

import numpy as np
import pandas as pd
from enum import Enum
from medpy import metric
import SimpleITK as sitk
import matplotlib.pyplot as plt
#%%
class DistanceMetrics(object):

    
    def __init__(self, maskfile , referencefile):
        
        
        ''' Read the images from the filepaths'''
        reference_segmentation = sitk.ReadImage(referencefile, sitk.sitkUInt8)
        segmentation = sitk.ReadImage(maskfile,sitk.sitkUInt8) 
        
        ''' init the enum fields for surface dist measures computer with simpleitk'''
        class SurfaceDistanceMeasuresITK(Enum):
            hausdorff_distance, max_distance, min_surface_distance, mean_surface_distance, median_surface_distance, std_surface_distance = range(6)
        
        
        class MedpyMetricDists(Enum):
            hausdorff_distanceMedPy, avg_surface_distanceMedPy, avg_symmetric_surface_distanceMedPy = range(3)
        
        surface_distance_results = np.zeros((1,len(SurfaceDistanceMeasuresITK.__members__.items())))
        surface_dists_Medpy = np.zeros((1,len(MedpyMetricDists.__members__.items())))
        
        #%%
        ''''
        Mauerer Distance Map for the Reference Object (deemed as "tumor in this particular case")
        Algorithm Pipeline :
            1. compute the contour surface of the object (6-pixel-connectivity neighborhood)
            2. convert from SimpleITK format to Numpy Array Img
            3. remove the zeros from the contour of the object, NOT from the distance map
            4. compute the number of 1's pixels in the contour
            5. instantiate the Signed Mauerer Distance map for the object (negative numbers also)
        '''
        reference_surface = sitk.LabelContour(reference_segmentation)
        reference_surface_array = sitk.GetArrayFromImage(reference_surface)
        reference_surface_array_NonZero = reference_surface_array.nonzero()
        self.num_reference_surface_pixels = len(list(zip(reference_surface_array_NonZero[0], reference_surface_array_NonZero[1], reference_surface_array_NonZero[2])))
                        
        # init signed mauerer distance as reference metrics from SimpleITK
        self.reference_distance_map = sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True)
        
        ''''compute Hausdorff distance also known as maximum symmetric distance
        - doesn't matter whether is computed from mask to reference (ablation, tumor) as it selects the maximum amongs the two
        '''
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(reference_segmentation, segmentation)
        surface_distance_results[0,SurfaceDistanceMeasuresITK.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
            
        #%%
        ''' Mauerer Distance Map for the Mask Object (deemed as "ablation" in this particular case)'''
        segmented_surface_mask = sitk.LabelContour(segmentation) #aka the ablation file
        segmented_surface_mask_array = sitk.GetArrayFromImage(segmented_surface_mask)
        surface_mask_array_NonZero = segmented_surface_mask_array.nonzero()
        
        # Get the number of pixels in the mask surface by counting all pixels that are non-zero
        self.num_segmented_surface_pixels = len(list(zip(surface_mask_array_NonZero[0],surface_mask_array_NonZero[1], surface_mask_array_NonZero[2])))
        
        # init Mauerer Distance
        self.mask_distance_map = sitk.SignedMaurerDistanceMap(segmentation, squaredDistance=False, useImageSpacing=True)
        #%%        
        # Multiply the binary surface segmentations with the distance maps. The resulting distance
        # maps contain non-zero values only on the surface (they can also contain zero on the surface)
        mask_distance_map_array = sitk.GetArrayFromImage(self.mask_distance_map)
        reference_distance_map_array = sitk.GetArrayFromImage(self.reference_distance_map)
        
        '''compute the contours multiplied with the euclidean distances'''
        self.seg2ref_distance_map = mask_distance_map_array*reference_surface_array
        self.ref2seg_distance_map = reference_distance_map_array*segmented_surface_mask_array
 #%%           
        '''
        - plot the distances as heatmap one slice for verification
        '''
        if flag_plot is True:
            fig1, ax1= plt.subplots()
            z = int(np.floor(np.shape(reference_distance_map_array)[0]/2))
            heatmap = ax1.imshow(reference_distance_map_array[z,:,:])
            plt.title('Distance Map for Tumor. 1 Slice Visualization')
            plt.colorbar(heatmap)
            
            fig2, ax2= plt.subplots()
            z = int(np.floor(np.shape(self.ref2seg_distance_map)[0]/2))
            heatmap = ax2.imshow(self.ref2seg_distance_map[z,:,:]/255)
            plt.title('Tumor to Ablation Surface Distances. 1 Slice Visualization')
            plt.colorbar(heatmap)
            
            fig3, ax3= plt.subplots()
            z = int(np.floor(np.shape(mask_distance_map_array)[0]/2))
            heatmap = ax3.imshow(mask_distance_map_array[z,:,:])
            plt.title('Distance Map for Ablation. 1 Slice Visualization')
            plt.colorbar(heatmap)
            
            fig4, ax4= plt.subplots()
            z = int(np.floor(np.shape(self.seg2ref_distance_map)[0]/2))
            heatmap = ax4.imshow(self.seg2ref_distance_map[z,:,:]/-255)
            plt.title('Ablation To Tumor Surface Distances. 1 Slice Visualization')
            plt.colorbar(heatmap)
        
#%%        
        '''remove the zeros from the surface contour(indexes) from the distance maps '''
        self.seg2ref_distances = list(self.seg2ref_distance_map[reference_surface_array_NonZero]/-255) 
        self.ref2seg_distances = list(self.ref2seg_distance_map[surface_mask_array_NonZero]/255) 

        if flag_symmetric is True:
            self.surface_distances = self.seg2ref_distances + self.ref2seg_distances
        if flag_mask2reference is True:
            self.surface_distances = self.seg2ref_distances
        if flag_reference2mask is True:
            self.surface_distances = self.ref2seg_distances
        #%% 
        ''' Compute the surface distances max, min, mean, median, std '''
        surface_distance_results[0,SurfaceDistanceMeasuresITK.max_distance.value] = np.max(self.surface_distances)
        surface_distance_results[0,SurfaceDistanceMeasuresITK.min_surface_distance.value] = np.min(self.surface_distances)
        surface_distance_results[0,SurfaceDistanceMeasuresITK.mean_surface_distance.value] = np.mean(self.surface_distances)
        surface_distance_results[0,SurfaceDistanceMeasuresITK.median_surface_distance.value] = np.median(self.surface_distances)
        surface_distance_results[0,SurfaceDistanceMeasuresITK.std_surface_distance.value] = np.std(self.surface_distances)
        
        # Save to DataFrame
        self.surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index = list(range(1)),
                                      columns=[name for name, _ in SurfaceDistanceMeasuresITK.__members__.items()])
        
        # change the name of the columns
        if flag_symmetric is True:
            self.surface_distance_results_df.columns = ['Hausdorff', 'Maximum Symmetric', 'Minimum Symmetric', 'Mean Symmetric', 'Median Symmetric', 'Std']
        
        if flag_reference2mask is True:
            self.surface_distance_results_df.columns = ['Hausdorff_TA', 'Maximum_TA', 'Minimum_TA', 'Mean_TA', 'Median_TA', 'Std_TA']

        if flag_mask2reference is True:
            self.surface_distance_results_df.columns = ['Hausdorff_AT', 'Maximum_AT', 'Minimum_AT', 'Mean_AT', 'Median_AT', 'Std_AT']
            
        #%%
        ''' use MedPy library for comparision with SimpleITK that values are in the same range'''
        img_array = sitk.GetArrayFromImage(reference_segmentation)
        seg_array = sitk.GetArrayFromImage(segmentation)
        # reverse array in the order x, y, z
        img_array_rev = np.flip(img_array,2)
        seg_array_rev = np.flip(seg_array,2)
        vxlspacing = segmentation.GetSpacing()
        
        # use the MedPy metric library
        surface_dists_Medpy[0,MedpyMetricDists.hausdorff_distanceMedPy.value] = metric.binary.hd(seg_array_rev,img_array_rev, voxelspacing=vxlspacing)
        surface_dists_Medpy[0,MedpyMetricDists.avg_surface_distanceMedPy.value] = metric.binary.asd(seg_array_rev,img_array_rev, voxelspacing=vxlspacing)
        surface_dists_Medpy[0,MedpyMetricDists.avg_symmetric_surface_distanceMedPy.value] = metric.binary.assd(seg_array_rev,img_array_rev, voxelspacing=vxlspacing)
        self.surface_dists_Medpy_df = pd.DataFrame(data=surface_dists_Medpy, index = list(range(1)),
                                      columns=[name for name, _ in MedpyMetricDists.__members__.items()])

    #%%
    
    def get_Distances(self):
        # convert to DataFrame
        metrics_all = pd.concat([self.surface_dists_Medpy_df, self.surface_distance_results_df], axis=1)
        return(metrics_all)
    
    def get_SitkDistances(self):
        return self.surface_distance_results_df
    
    def get_MedPyDistances(self):
        return self.surface_dists_Medpy_df
    
    def get_avg_symmetric_dist(self):
        return (self.n1*self.avg_ref + self.n2*self.avg_seg)/(self.n1+self.n2)
    
    def get_std_symmetric_dist(self):
        return self.symmetric_std
    
    def get_mask_dist_map(self):
        return self.seg2ref_distance_map
    
    def get_reference_dist_map(self):
        return self.ref2seg_distance_map 
    
    def get_ref2seg_distances(self):
        return self.ref2seg_distances 
    
    def get_seg2ref_distances(self):
        return self.seg2ref_distances
    
    def get_all_distances(self):
        return self.all_surface_distances