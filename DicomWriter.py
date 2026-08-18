# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:40:46 2018

@author: Raluca Sandu

Writes a SimpleITK image out as a DICOM series, one file per slice.
"""
import os

import SimpleITK as sitk


class DicomWriter:

    def __init__(self, image=None, folder_output=None, file_name=None, series_reader=None):
        """
        :type: image in SimpleITK format
        :type folder_output: folder path to write the DICOM Series Files
        :type file_name: string specifying the filename, ablation or tumor z.B.
        :type patient_id: string number denoting unique patient ID
        """
        self.image = image
        self.folder_output = folder_output
        self.file_name = file_name
        self.series_reader = series_reader

    def save_image_to_file(self):
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        # writer.SetKeepOriginalImageUID()
        for i in range(self.image.GetDepth()):
            image_slice = self.image[:, :, i]
            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            writer.SetFileName(os.path.normpath(self.folder_output + '/' + self.file_name + str(i) + '.dcm'))
            writer.Execute(image_slice)






