# -*- coding: utf-8 -*--1`
"""
@author: Raluca Sandu
"""
import pydicom
import SimpleITK as sitk

# TODO: add reading for dicom image object with pydicom library
# TODO: add study, referring physician, studydate, seriesinnstanceuid, etc
class ReadDicomTags:

    @staticmethod
    def print_image_dimensions(image, name):
        """
        Print Image Parameters.
        :param image: SimpleITK object image
        :param name: Name of the image as defined by the user
        :return:
        """
        print(name + ' direction: ', image.GetDirection())
        print(name + ' origin: ', image.GetOrigin())
        print(name + ' spacing: ', image.GetSpacing())
        print(name + ' size: ', image.GetSize())
        print(name + ' pixel: ', image.GetPixelIDTypeAsString())
