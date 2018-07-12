# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:57:34 2018

@author: Raluca Sandu
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from DicomReader import read_dcm_series 
from splitAllPaths import splitall

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def str_to_vector(needle_str):
    needle_vector = np.array([float(i) for i in needle_str.split()])
    return needle_vector


rootdir = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_23042018"
df_data = pd.read_excel(
        "C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_23042018\Patients_MWA_IR_SegmentationPaths.xlsx",
        dtype=str)

radii = df_data["Radii"][0]
radii_vector = str_to_vector(radii.strip("'[']"))
translation = df_data["Translation"][0]
rotation = df_data["Rotation"][0]
ep_needle_str = df_data["ValidationEntryPoint"][0]
ep_needle = str_to_vector(ep_needle_str.strip('[]'))
tp_needle_str = df_data["ValidationTargetPoint"][0]
tp_needle = str_to_vector(tp_needle_str.strip('[]'))
path_source = df_data["ValidationAblationPath"][0]
path_mask = df_data["AblationPath"][0]
#%%
#directory_path = "C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_23042018\Pat_Amez-Droz Anne-Marie_0013086812_2017-08-04_08-19-05\Study_0\Series_8"
image_source = read_dcm_series(os.path.join(rootdir,path_source))
all_paths = splitall(path_source)
image_mask = read_dcm_series(os.path.join(rootdir, all_paths[0], path_mask))
vector = tp_needle - ep_needle
ablation_needle_coords = unit_vector(vector)

#%%
newSize = image_source.GetSize()
newOrigin = image_source.GetOrigin()
# we assume we have the same spacing as the images have been taken with the same scanner    
newSpacing = image_source.GetSpacing() 
newDirection = image_source.GetDirection()
# create black 3D image with the size and origin of the plan CT
outputImage = sitk.Image(newSize, sitk.sitkInt16)
outputImage.SetOrigin(newOrigin)
outputImage.SetSpacing(newSpacing)
outputImage.SetDirection(newDirection)    

#%% find the origin of the ablation ellipsoid
origin_ellipse = outputImage.TransformPhysicalPointToIndex(tp_needle)
x_radius = outputImage.TransformPhysicalPointToIndex(tp_needle + radii_vector[0])
y_radius = outputImage.TransformPhysicalPointToIndex(tp_needle + radii_vector[1])
z_radius = outputImage.TransformPhysicalPointToIndex(tp_needle + radii_vector[2])
# TODO: translate and rotate
# apply the rotation angle (rotate alongside the needle)
# iterate using the ellipsoid equation. set the pixel value to 255
# plot nda
# plot GIF
# compute metrics