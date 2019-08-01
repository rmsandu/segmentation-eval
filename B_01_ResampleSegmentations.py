# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import DicomReader as Reader
import DicomWriter as DicomWriter
from recordclass import recordclass
import PasteROI_Segmentation2OriginalSize as PasteROI


class ResizeSegmentations:

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

    def __init__(self, tumor_segmentation, ablation_segmentation, root_path_to_save, flag_plot_id):
        self.root_path_to_save = root_path_to_save
        self.flag_plot_id = flag_plot_id
        self.tumor_segmentation = tumor_segmentation
        self.ablation_segmentation = ablation_segmentation

        self.new_filepaths = []
        self.ablation_paths_resized = []
        self.tumor_paths_resized = []
        images_resampled = recordclass('images_resampled', ['ablation_segmentation', 'tumor_segmentation'])
        images_resampled.tumor_segmentation = PasteROI.resample_segmentations(self.ablation_segmentation,
                                                                              self.tumor_segmentation)
        images_resampled.ablation_segmentation = self.ablation_segmentation
