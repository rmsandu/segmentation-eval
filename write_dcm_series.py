# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:07:23 2018

@author: Raluca Sandu
"""

import SimpleITK as sitk

def write_to_file(image):
    
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for i in range(image.GetDepth()):
        image_slice = image[:,:,i]
        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName('newTumorMask'+str(i)+'.dcm')
        writer.Execute(image_slice)