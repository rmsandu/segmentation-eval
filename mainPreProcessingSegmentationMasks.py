# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:25:13 2018

@author: Raluca Sandu
"""



import SimpleITK as sitk
import read_dcm_series as reader
import write_dcm_series_v2 as writeDCM
from paste_roi_image import paste_roi_image
from sitkShowSlice import sitk_show

#%%
folder_path_tumor = "C:/develop/data/Segmentation_tumor"
tumor_mask = reader.read_dcm_series(folder_path_tumor)
folder_path_ablation = "C:/develop/data/Segmentation_ablation"
ablation_mask = reader.read_dcm_series(folder_path_ablation)
source_img_plan= reader.read_dcm_series("C:/develop/data/Source_CT_Plan")
source_img_validation = reader.read_dcm_series("C:/develop/data/Source_CT_Validation")
#%%
reader.print_dimensions_img('tumor', tumor_mask)
print('\n')
reader.print_dimensions_img('ablation', ablation_mask)
print('\n')
reader.print_dimensions_img('plan', source_img_plan)
print('\n')
reader.print_dimensions_img('validation', source_img_validation)


#%%
# resize the Segmentation Mask to the dimensions of the source images they were derived from
resizedTumorMask = paste_roi_image(source_img_plan, tumor_mask)
resizedAblationMask = paste_roi_image( source_img_validation, ablation_mask)

#writeDCM.writeDICOMSeries("C:/develop/data/Source_CT_Plan","C:/develop/data/Resized_Segmentation_Tumor/", resizedTumorMask)
#writeDCM.writeDICOMSeries("C:/develop/data/Source_CT_Plan","C:/develop/data/Resized_Segmentation_Ablation/", resizedAblationMask)
# TO DO:
    # - adapt the number of slices for the ablation mask
    # - it seems that the mask for the tumor is not copied at the right location . but why?
    # - the ablation mask is copied on different slices. make it the same number of slices as the source img it was derived from (CT validation)
    # - check if you can run eval segmentation metrics if the tumor and ablation mask have different sizes (nr. of slices)
    #
#sitk.Show(resizedAblationMask)
#sitk.Show(resizedTumorMask)
#%%
idxSlice = 122
labelTumor = 1
imgOriginal = source_img_plan[:,:,idxSlice]
imgMask = resizedTumorMask[:,:,idxSlice]

img_T1_255 = sitk.Cast(sitk.RescaleIntensity(source_img_plan), sitk.sitkUInt8)
seg = sitk.Cast(sitk.RescaleIntensity(resizedTumorMask), sitk.sitkUInt8)
OverlaidImg = sitk.LabelOverlay(img_T1_255, seg, opacity=0.2)

sitk_show(OverlaidImg[:,:,idxSlice])