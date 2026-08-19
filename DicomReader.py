# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:45:50 2018

@author: Raluca Sandu
"""
import os

import SimpleITK as sitk

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
