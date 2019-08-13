# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:45:50 2018

@author: Raluca Sandu
"""
import os
import numpy as np
import SimpleITK as sitk
import pydicom
#%%


def read_dcm_series(folder_path, reader_flag=True):
    """
    Read DICOM Series/Single Image from a folder path into a SimpleITK Image Object.
    :param folder_path: directory address containing DICOM Images
    :param reader_flag:
    :return: SimpleITK Image Object
    """

    try:
        if next(os.walk(folder_path), None) is None:
            # single DICOM File
            image = sitk.ReadImage(os.path.normpath(folder_path), sitk.sitkInt16)
            return image, None
    except Exception:
        print('Non-readable DICOM Data: ', folder_path)
        return None
    # DICOM Series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(os.path.normpath(folder_path))
    reader.SetFileNames(dicom_names)
    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    try:
        image = reader.Execute()
        if reader_flag:
            return image, reader
        else:
            return image
    except Exception:
        print('Non-readable DICOM Data: ', folder_path)
        if reader_flag:
            return None, None
        else:
            return None

def read_single_dcm(path):
    try:
        ds = pydicom.read_file(path)
    except Exception:
        return None
    return ds


def read_dcm_series_pydicom(path):
    # Load the scans in given folder path
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
#    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except Exception:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
