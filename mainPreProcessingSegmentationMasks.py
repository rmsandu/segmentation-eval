# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:25:13 2018

@author: Raluca Sandu
"""

import numpy as np
import SimpleITK as sitk
import read_dcm_series as reader
import write_dcm_series_v2 as writeDCM
from paste_roi_image import paste_roi_image
import resampling_hu_dcm as hu
import matplotlib.pyplot as plt

#%%
folder_path_tumor = "C:/develop/data/Segmentation_tumor"
folder_path_plan  = "C:/develop/data/Source_CT_Plan"
folder_path_ablation = "C:/develop/data/Segmentation_ablation"
folder_path_validation = "C:/develop/data/Source_CT_Validation"

tumor_mask = reader.read_dcm_series( folder_path_tumor )
ablation_mask = reader.read_dcm_series( folder_path_ablation )
source_img_plan= reader.read_dcm_series( folder_path_plan )
source_img_validation = reader.read_dcm_series( folder_path_validation )
#%%
reader.print_dimensions_img('tumor', tumor_mask)
print('\n')
reader.print_dimensions_img('ablation', ablation_mask)
print('\n')
reader.print_dimensions_img('plan', source_img_plan)
print('\n')
reader.print_dimensions_img('validation', source_img_validation)
#%%
''' resize the Segmentation Mask to the dimensions of the source images they were derived from '''
resizedTumorMask = paste_roi_image( source_img_plan, tumor_mask )
resizedAblationMask = paste_roi_image( source_img_validation, ablation_mask)

writeDCM.writeDICOMSeries(folder_path_plan ,"C:/develop/data/Resized_Segmentation_Tumor/", resizedTumorMask)
writeDCM.writeDICOMSeries(folder_path_validation,"C:/develop/data/Resized_Segmentation_Ablation/", resizedAblationMask)

#%%
idxSlice = 122
labelTumor = 1

img_source_nda = sitk.GetArrayFromImage( source_img_plan ) 
img_validation_nda = sitk.GetArrayFromImage( source_img_validation )
mask_tumor_nda = sitk.GetArrayFromImage( resizedTumorMask )
mask_ablation_nda = sitk.GetArrayFromImage( resizedAblationMask )

pydicom_scans = reader.load_scan( folder_path_plan )
imgHU_pydicom = hu.get_pixels_hu( pydicom_scans )

indexes_all = np.nonzero(mask_tumor_nda)

img_T1_255 = sitk.Cast(sitk.RescaleIntensity(source_img_plan), sitk.sitkUInt8)
seg = sitk.Cast(sitk.RescaleIntensity(resizedTumorMask), sitk.sitkUInt8)

OverlaidImg = sitk.LabelOverlay(img_T1_255, seg)
OverlaidImg_nda = sitk.GetArrayFromImage(OverlaidImg)
#%%
#
#oneSlice_source = imgHU_pydicom[122].astype(np.int32)
oneSlice_source =  img_source_nda[122,:,:]
oneSlice_tumor = mask_tumor_nda[122,:,:].astype(np.float)
oneSlice_ablation = mask_ablation_nda[70,:,:].astype(np.float)
#
AblationOverlay = oneSlice_ablation
TumorOverlay = oneSlice_tumor
AblationOverlay[ AblationOverlay==0 ] = np.nan
TumorOverlay[TumorOverlay ==0 ] = np.nan
#%%
plt.figure()
im1 = plt.imshow(oneSlice_source, cmap=plt.cm.gray, interpolation='none')
im2 = plt.imshow(TumorOverlay, cmap='RdYlBu', alpha=0.3, interpolation='none')
im3 = plt.imshow(AblationOverlay, cmap='jet', alpha=0.4, interpolation='none')
plt.grid(False)
