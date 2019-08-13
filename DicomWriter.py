# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:40:46 2018

@author: Raluca Sandu
# Read the original series. First obtain the series file names using the
# image series reader. Then read each image using an image reader that is
# set to load all DICOM tags (public+private). The resulting images contain
# their DICOM meta-data dictionaries.
Reading the DICOM series is a three step process: first obtain the series ID, then obtain the file names associated
with the series ID, and finally use the series reader to read the images.
By default the DICOM meta-data dicitonary for each of the slices is not read.
In this example we configure the series reader to load the meta-data dictionary including all of the private tags.
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






