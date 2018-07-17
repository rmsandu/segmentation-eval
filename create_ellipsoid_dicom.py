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
#%% extract information from  Excel list/DataFrame 
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
translation_str = df_data["Translation"][0]
rotation = df_data["Rotation"][0]
path_source = df_data["ValidationAblationPath"][0]
path_mask = df_data["AblationPath"][0]
# needle trajectory saved as str(list)
translation = str_to_vector(translation_str.strip("'[']"))
ep_needle_str = df_data["ValidationEntryPoint"][0]
ep_needle = str_to_vector(ep_needle_str.strip('[]'))
tp_needle_str = df_data["ValidationTargetPoint"][0]
tp_needle = str_to_vector(tp_needle_str.strip('[]'))

#%% Read the DICOM Images.
# source validation
image_source = read_dcm_series(os.path.join(rootdir, path_source))
all_paths = splitall(path_source)
# ablation mask
#image_mask = read_dcm_series(os.path.join(rootdir, all_paths[0], path_mask))
image_mask = read_dcm_series(r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_23042018_anonymized\Pat_GTDB_11\Trajectory0\Resized_Ablation_Segmentation")
#%% # create black 3D image with the size and origin of the source CT
newSize = image_source.GetSize()
newOrigin = image_source.GetOrigin()
# we assume we have the same spacing as the images have been taken with the same scanner    
newSpacing = image_source.GetSpacing() 
newDirection = image_source.GetDirection()
outputImage = sitk.Image(newSize, sitk.sitkInt16)
outputImage.SetOrigin(newOrigin)
outputImage.SetSpacing(newSpacing)
outputImage.SetDirection(newDirection)    
img_nda = sitk.GetArrayFromImage(outputImage)
#%% compute translation
translated_tp_needle = np.zeros(shape=(1,3))
translated_tp_needle[0] = tp_needle[0] + translation[1]
translated_tp_needle[1] = tp_needle[1] + translation[0]

#%% compute rotation angle of ellipsoid. 
# TODO: compute the signed rotation angle. check if axis correct
perpendicular_axis_ep = outputImage.TransformIndexToPhysicalPoint((0,0,0))
perpendicular_axis_tp = outputImage.TransformIndexToPhysicalPoint(img_nda.shape)
rotation_angle = angle_between(ep_needle, tp_needle, perpendicular_axis_ep, perpendicular_axis_tp)

#%% find the origin of the ablation ellipsoid
# should the translation be applied to the needle origin?
# TODO: which is the starting slice of the needle?. should be the needle origin?
origin_ellipse = outputImage.TransformPhysicalPointToIndex(translated_tp_needle)
z_radius = outputImage.TransformPhysicalPointToIndex(translated_tp_needle + radii_vector[2])
x_radius = outputImage.TransformPhysicalPointToIndex(translated_tp_needle + radii_vector[1])
y_radius = outputImage.TransformPhysicalPointToIndex(translated_tp_needle + radii_vector[0])

#start_slice = min(x_radius[2], y_radius[2], z_radius[2])
start_slice = origin_ellipse[2]
end_slice = max(x_radius[2], y_radius[2], z_radius[2])
#%%
height = 512
width = 512

# TODO: find the correct axis
# TODO: find out Ellipse rotation (radians)
semi_major, semi_minor = (int(radii_vector[0]), int(radii_vector[1]))

angle_between(ep_needle,tp_needle, perpendicular_axis_ep, perpendicular_axis_tp)
inner_scale = 0.6 # Scale of the inner full-white ellipse

ellipse_outer = ((origin_ellipse[0], origin_ellipse[1]), (semi_major*2, semi_minor*2), math.degrees(rotation_angle))
transparency = np.zeros((height, width), np.uint8)
ellipse_nda_slice = cv2.ellipse(transparency, ellipse_outer, 255, -1, cv2.LINE_AA)
ellipse_sitk = sitk.GetImageFromArray(ellipse_nda_slice)
cv2.imshow('image', ellipse_nda_slice)
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
#            cv2.ellipse(ResultImage, (centerX,centerY), (width,height), 0, 0, 180, yellow, 2)
            ellipse_outer = ((origin_ellipse[0], origin_ellipse[0]),
                             (semi_major*2, semi_major*2), rotation_angle)
            ellipse_nda = cv2.ellipse(img_nda, ellipse_outer, 255, -1, cv2.LINE_AA)
            # replice the slice with the created ellipse
            image_slice = sitk.GetImageFromArray(ellipse_nda)

#        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
#        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
#        # Setting the type to CT preserves the slice location.
#        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
#        # (0020, 0032) image position patient determines the 3D spacing between slices.
#        # Image Position (Patient)
#        image_slice.SetMetaData("0020|0032",
#                                '\\'.join(map(str, image_source.TransformIndexToPhysicalPoint((0, 0, i)))))
#        image_slice.SetMetaData("0020,0013", str(i))  # Instance Number
#
#        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
#        writer.SetFileName(os.path.normpath(folder_output + '/' + file_name + str(i) + '.dcm'))
#        writer.Execute(image_slice)

#%% plot slices overlaid for verification purposes.
# TODO: show ablation overlay from ablation mask
# TODO: show ellipsoid ablation
plt.figure()
img_source_nda = sitk.GetArrayFromImage(image_source)
img_mask_nda = sitk.GetArrayFromImage(image_mask)
slice_tumor = img_source_nda[origin_ellipse[2],:,:].astype(np.float)
slice_ablation = img_mask_nda[origin_ellipse[2],:,:].astype(np.float)
slice_ellipse = ellipse_nda_slice.astype(np.float)

im1 = plt.imshow(slice_tumor, cmap=plt.cm.gray, interpolation='none')  
# display only the tumor/ablation which value is 255 (white). set the rest to nan
slice_ablation [slice_ablation == 0] = np.nan
# plot the overlaid ablation
im2 = plt.imshow(slice_ablation, cmap='RdYlBu', alpha=0.1, interpolation='none')
# plot the created ellipse
slice_ellipse [slice_ellipse == 0] = np.nan
im3 = plt.imshow(slice_ellipse, cmap = 'YlGnBu', alpha=0.2, interpolation='none')
# apply the rotation angle (rotate alongside the needle)
# iterate using the ellipsoid equation. set the pixel value to 255
# plot nda
# plot GIF
# compute metrics