# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:40:46 2018

@author: Raluca Sandu
"""

import SimpleITK as sitk

import sys, time, os

# Read the original series. First obtain the series file names using the
# image series reader. Then read each image using an image reader that is
# set to load all DICOM tags (public+private). The resulting images contain
# their DICOM meta-data dictionaries.
def writeDICOMSeries(folderInput, folderOutput, img_pasted):
    data_directory = folderInput
    series_reader = sitk.ImageSeriesReader()
    series_IDs = series_reader.GetGDCMSeriesIDs(data_directory)
    if not series_IDs:
        print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
        sys.exit(1)
    series_file_names = series_reader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])
    
    image_reader = sitk.ImageFileReader()
    image_reader.LoadPrivateTagsOn()
    image_list = []
    for file_name in series_file_names:
        image_reader.SetFileName(file_name)
        image_list.append(image_reader.Execute())

    
    
    writer = sitk.ImageFileWriter()
    # Use the study/seriers/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    
    for i in range(img_pasted.GetDepth()):
        image_slice = img_pasted[:,:,i]
        original_slice = image_list[i]
        # Copy the meta-data except the PixelSpacing, PatientPosition, SliceThickness
#        if k!="0028|0030" and k!= "0018|5100" and k!="0018|0050" :
        for k in original_slice.GetMetaDataKeys():
            image_slice.SetMetaData(k, original_slice.GetMetaData(k))
            
        # Set relevant keys indicating the change, modify or remove private tags as needed
        # Each of the UID components is a number (cannot start with zero) and separated by a '.'
        image_slice.SetMetaData("0008|0031", modification_time)
        image_slice.SetMetaData("0008|0021", modification_date)
        image_slice.SetMetaData("0008|0008", "DERIVED\SECONDARY")
        # We create a unique series ID using the date and time.
        image_slice.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time)
        # Write to the output directory and add the extension dcm if not there, to force writing is in DICOM format.
        writer.SetFileName(os.path.join(folderOutput, os.path.basename(series_file_names[i])) + ('' if os.path.splitext(series_file_names[i])[1] == '.dcm' else '.dcm'))
        writer.Execute(image_slice)