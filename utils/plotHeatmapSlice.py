# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:45:57 2018

@author: Raluca Sandu
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
def plotHeatMapDistances(reference_distance_map_array, mask_distance_map_array,seg2ref_distance_map,ref2seg_distance_map):
    
    fig1, ax1= plt.subplots()
    z = int(np.floor(np.shape(reference_distance_map_array)[0]/2))
    heatmap = ax1.imshow(reference_distance_map_array[z,:,:])
    plt.title('Distance Map for Tumor. 1 Slice Visualization')
    plt.colorbar(heatmap)
    
    fig2, ax2= plt.subplots()
    z = int(np.floor(np.shape(ref2seg_distance_map)[0]/2))
    heatmap = ax2.imshow(ref2seg_distance_map[z,:,:]/255)
    plt.title('Tumor to Ablation Surface Distances. 1 Slice Visualization')
    plt.colorbar(heatmap)
    #TO DO: save the picture offline
    
    fig3, ax3= plt.subplots()
    z = int(np.floor(np.shape(mask_distance_map_array)[0]/2))
    heatmap = ax3.imshow(mask_distance_map_array[z,:,:])
    plt.title('Distance Map for Ablation. 1 Slice Visualization')
    plt.colorbar(heatmap)
    # TO DO: save the picture offline
    
    fig4, ax4= plt.subplots()
    z = int(np.floor(np.shape(seg2ref_distance_map)[0]/2))
    heatmap = ax4.imshow(seg2ref_distance_map[z,:,:]/-255)
    plt.title('Ablation To Tumor Surface Distances. 1 Slice Visualization')
    plt.colorbar(heatmap)
    
    # TO DO : save the picture offfline