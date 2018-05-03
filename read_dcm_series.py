# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:45:50 2018

@author: Raluca Sandu
"""
import os
import numpy as np
import SimpleITK as sitk
import pydicom as dicom
#%%

def read_dcm_series_MetaData(folder_path):
     # currently not functional
    series_reader = sitk.ImageSeriesReader()
    series_IDs = series_reader.GetGDCMSeriesIDs(folder_path)
     
    if not series_IDs:
        print("ERROR: given directory \""+folder_path+"\" does not contain a DICOM series.")
        return('ERROR: no DICOM series in given directory')
    series_file_names = series_reader.GetGDCMSeriesFileNames(folder_path, series_IDs[0])
    
    image_reader = sitk.ImageFileReader()
    image_reader.LoadPrivateTagsOn()
    image_list = []
    for file_name in series_file_names:
        image_reader.SetFileName(file_name)
        image_list.append(image_reader.Execute())

    # Pasting all of the slices into a 3D volume requires 2D image slices and not 3D slices
    # The volume's origin and direction are taken from the first slice and the spacing from
    # the difference between the first two slices. Note that we are assuming we are
    # dealing with a volume represented by axial slices (z spacing is difference between images).
    image_list2D = [image[:,:,0] for image in image_list]
    image3D = sitk.JoinSeries(image_list2D, image_list[0].GetOrigin()[2], image_list[1].GetOrigin()[2] - image_list[0].GetOrigin()[2])
    image3D.SetDirection(image_list[0].GetDirection())
     
    return image3D



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



def load_scan(path):
    # Load the scans in given folder path
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
#    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
