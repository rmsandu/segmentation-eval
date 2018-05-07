# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:25:13 2018

@author: Raluca Sandu
"""


import SimpleITK as sitk
import pydicom as dicom
from paste_roi_image import paste_roi_image
import read_dcm_series as reader
import write_dcm_series_v2 as writeDCM

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
resizedTumorMask = paste_roi_image(source_img_plan, source_img_validation, tumor_mask)
writeDCM.writeDICOMSeries("C:/develop/data/Source_CT_Plan","C:/develop/data/Resized_Segmentation_Tumor/", resizedTumorMask)
resizedAblationMask = paste_roi_image(source_img_plan, source_img_validation, ablation_mask)
writeDCM.writeDICOMSeries("C:/develop/data/Source_CT_Plan","C:/develop/data/Resized_Segmentation_Ablation/", resizedAblationMask)
# TO DO: adapt the number of slices
#sitk.Show(resizedAblationMask)
#sitk.Show(resizedTumorMask)
#%%
new_folder = "C:\develop\data\Resized_Segmentation"