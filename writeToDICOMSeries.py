# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:40:46 2018

@author: Raluca Sandu
"""
import time, os
import SimpleITK as sitk

# Read the original series. First obtain the series file names using the
# image series reader. Then read each image using an image reader that is
# set to load all DICOM tags (public+private). The resulting images contain
# their DICOM meta-data dictionaries.
def writeDICOMSeries(folderOutput, img_pasted):

    # Use the study/seriers/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = img_pasted.GetDirection()
#    spacing = img_pasted.GetSpacing()
    

    series_tag_values = [("0008|0031", modification_time), # Series Time
                  ("0008|0021", modification_date), # Series Date
                  ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                  ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                  ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                    direction[1],direction[4],direction[7])))),
                  ("0008|103e", "ResizedSegmentationMask")] # Series Description
    
    for i in range(img_pasted.GetDepth()):
        image_slice = img_pasted[:,:,i]
        # Tags shared by the series
        for tag, value in series_tag_values:
            # Slice specific tags
            image_slice.SetMetaData(tag,value)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        # (0020, 0032) image position patient determines the 3D spacing between slices.
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, img_pasted.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020,0013", str(i)) # Instance Number
        image_slice.SetMetaData("0008|0008", "DERIVED\SECONDARY")
        
        # Copy the meta-data except the PixelSpacing, PatientPosition, SliceThickness
#        if k!="0028|0030" and k!= "0018|5100" and k!="0018|0050" :
#        for k in original_slice.GetMetaDataKeys():
#            image_slice.SetMetaData(k, original_slice.GetMetaData(k))
#            
         # Write to the output directory and add the extension dcm, to force writing in DICOM format.   
        writer.SetFileName(os.path.normpath(folderOutput + '/' + 'ablationSegm' + str(i)  + '.dcm'))
        writer.Execute(image_slice)