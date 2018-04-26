# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:45:50 2018

@author: Raluca Sandu
"""
import os
import SimpleITK as sitk
import pydicom as dicom
from paste_roi_image import paste_roi_image
import resampling_hu_dcm as resample
#%%

def read_dcm_series(folder_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def print_dimensions_img(title,image):
    print('Dimensions of ' + title + ' image:', image.GetSize())
    print('Spacing of ' +title + ' image:', image.GetSpacing())
    print('Origin of '+title + ' image:', image.GetOrigin())
    print('Direction of '+title +' image:',image.GetDirection())
    print('Pixel ID Value:',image.GetPixelIDValue())


#%%
folder_path_tumor = "C:/develop/data/Segmentation_tumor"
tumor_mask = read_dcm_series(folder_path_tumor)
folder_path_ablation = "C:/develop/data/Segmentation_ablation"
ablation_mask = read_dcm_series(folder_path_ablation)
source_img_plan = read_dcm_series("C:/develop/data/Source_CT_Plan")
source_img_validation = read_dcm_series("C:/develop/data/Source_CT_Validation")
#%%
print_dimensions_img('tumor', tumor_mask)
print('\n')
print_dimensions_img('ablation', ablation_mask)
print('\n')
print_dimensions_img('plan', source_img_plan)
print('\n')
print_dimensions_img('validation', source_img_validation)
#%%
'''a different way of reading slice by slice and ordering slices based on index'''
files = os.listdir(folder_path_tumor )
slices = [dicom.read_file(os.path.join(folder_path_tumor , filename)) for filename in files]
slices.sort(key = lambda x: int(x.InstanceNumber))

patient = resample.load_scan(folder_path_tumor)
imgs_hu = resample.get_pixels_hu(patient)

#%%
print("Slice Thickness: %f" % slices[0].SliceThickness)
print("Pixel Spacing (row, col, slices): (%f, %f) " % (slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]))

#%%
resizedTumorMask = paste_roi_image(source_img_plan,tumor_mask)
sitk.Show(resizedTumorMask)
resizedAblationMask = paste_roi_image(source_img_validation,ablation_mask)
sitk.Show(resizedAblationMask)


