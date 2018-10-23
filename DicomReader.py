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


def read_dcm_series(folder_path):

    try:
        if next(os.walk(folder_path), None) is None:
            # single DICOM File
            image = sitk.ReadImage(os.path.normpath(folder_path), sitk.sitkInt16)
            return image
    except Exception:
        print('Non-readable DICOM Data: ', folder_path)
        return None
    # DICOM Series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(os.path.normpath(folder_path))
    reader.SetFileNames(dicom_names)
    try:
        image = reader.Execute()
        return image
    except Exception:
        print('Non-readable DICOM Data: ', folder_path)
        return None


def print_dimensions_img(title,image):
    print('Dimensions of ' + title + ' image:', image.GetSize())
    print('Spacing of ' + title + ' image:', image.GetSpacing())
    print('Origin of '+ title + ' image:', image.GetOrigin())
    print('Direction of ' + title + ' image:',image.GetDirection())
    print('Pixel ID Value:', image.GetPixelIDValue())


def load_scan(path):
    # Load the scans in given folder path
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
#    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except Exception:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
