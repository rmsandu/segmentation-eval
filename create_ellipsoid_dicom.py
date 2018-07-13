# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:57:34 2018

@author: Raluca Sandu
"""
import os
import math
import cv2
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from DicomReader import read_dcm_series 
from splitAllPaths import splitall
import matplotlib.pyplot as plt
from angle_2vectors import angle_between
#%%
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
img_nda = sitk.GetArrayFromImage(outputImage)
perpendicular_axis_ep = outputImage.TransformIndexToPhysicalPoint((0,0,0))
perpendicular_axis_tp = outputImage.TransformIndexToPhysicalPoint(img_nda.shape)

rotation_angle = angle_between(ep_needle,tp_needle, perpendicular_axis_ep, perpendicular_axis_tp)
#%% find the origin of the ablation ellipsoid
origin_ellipse = outputImage.TransformPhysicalPointToIndex(tp_needle)
z_radius = outputImage.TransformPhysicalPointToIndex(tp_needle + radii_vector[0])
x_radius = outputImage.TransformPhysicalPointToIndex(tp_needle + radii_vector[1])
y_radius = outputImage.TransformPhysicalPointToIndex(tp_needle + radii_vector[2])

start_slice = min(x_radius[2], y_radius[2], z_radius[2])
end_slice = max(x_radius[2], y_radius[2], z_radius[2])
#%%
height = 512
width = 512
#y,x = np.mgrid[:height,:width]
h = origin_ellipse[0]
k = origin_ellipse[1]

a, b = (z_radius[0], x_radius[1]) # Semi-major and semi-minor axis
theta = math.radians(90.0) # TODO: find out Ellipse rotation (radians)

angle_between(ep_needle,tp_needle, perpendicular_axis_ep, perpendicular_axis_tp)
inner_scale = 0.6 # Scale of the inner full-white ellipse

ellipse_outer = ((h,k), (a*2, b*2), math.degrees(rotation_angle))

transparency = np.zeros((height, width), np.uint8)
e = cv2.ellipse(transparency, ellipse_outer, 255, -1, cv2.LINE_AA)
ellipse_sitk = sitk.GetImageFromArray(e)
cv2.imshow('image',e)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# 1. check if we are on the right slice of the 
# if yes modify if not set image_ellipse to image_slice
# 2. ellipse equation. set the pixels.
# 3. transform the image back to simple itk type
# TODO: get the sign of the angle!!!
# iterate throuch each slice and add the ellipse on it.
writer = sitk.ImageFileWriter()
file_name = "ellipse_"
folder_output = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\ellipse"

for i in range(outputImage.GetDepth()):

        image_slice = outputImage[:, :, i]
            
        if (i >= start_slice and i <= end_slice):
            img_nda = sitk.GetArrayFromImage(image_slice)
            ellipse_outer = ((origin_ellipse[0], origin_ellipse[0]),
                             (a*2, b*2), rotation_angle)
            ellipse_nda = cv2.ellipse(img_nda, ellipse_outer, 255, -1, cv2.LINE_AA)
            # replice the slice with the created ellipse
            image_slice = sitk.GetImageFromArray(ellipse_nda)

        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        # (0020, 0032) image position patient determines the 3D spacing between slices.
        # Image Position (Patient)
        image_slice.SetMetaData("0020|0032",
                                '\\'.join(map(str, image_source.TransformIndexToPhysicalPoint((0, 0, i)))))
        image_slice.SetMetaData("0020,0013", str(i))  # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.normpath(folder_output + '/' + file_name + str(i) + '.dcm'))
        writer.Execute(image_slice)

#%%
plt.figure()
img_source_nda = sitk.GetArrayFromImage(image_source)
im1 = plt.imshow(img_source_nda[170,:,:], cmap=plt.cm.gray, interpolation='none')  
# ellipse_overlay = e
# ellipse_overlay [ellipse_overlay  == 0] = np.nan
# im2 = plt.imshow(ellipse_overlay, cmap='RdYlBu', alpha=0.3, interpolation='none')
# apply the rotation angle (rotate alongside the needle)
# iterate using the ellipsoid equation. set the pixel value to 255
# plot nda
# plot GIF
# compute metrics