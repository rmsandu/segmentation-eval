# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:42:27 2018

@author: Raluca Sandu
"""

import os
import readDICOMFiles as Reader
import writeToDICOMSeries as WriterClass
import paste_roi_image as PasteROI


class ResizedSegmentations:

    def __init__(self, df_folderpaths):
        self.ablation_paths = df_folderpaths['Ablation Segmentation Path'].tolist()
        self.tumor_paths = df_folderpaths['Tumor Segmentation Path'].tolist()
        self.folder_path_plan = df_folderpaths['Plan Images Path'].tolist()
        self.folder_path_validation = df_folderpaths['Validation Images Path'].tolist()
        self.patients = df_folderpaths['PatientID']

    def save_images_to_disk(self):

        for idx, ablation_path in enumerate(self.ablation_paths):
            tumor_mask = Reader.read_dcm_series(self.tumor_paths[idx])
            ablation_mask = Reader.read_dcm_series(ablation_path)
            source_img_plan = Reader.read_dcm_series(self.folder_path_plan[idx])
            source_img_validation = Reader.read_dcm_series(self.folder_path_validation[idx])
            # resize the Segmentation Mask to the dimensions of the source images they were derived from '''
            resized_tumor_mask = PasteROI.paste_roi_imageMaxSize(source_img_plan, source_img_validation,
                                                                 tumor_mask)
            resized_ablation_mask = PasteROI.paste_roi_imageMaxSize(source_img_plan, source_img_validation,
                                                                    ablation_mask)

            # create new folder path in C:/develop/data/PatID
            root_path = r"C:/develop/data/"
            parent_directory = os.path.join(root_path, 'Pat_GTDB_' + str(self.patients[idx]))
            child_directory_tumor = os.path.join(parent_directory, 'Resized_Tumor_Segmentation')
            child_directory_ablation = os.path.join(parent_directory, 'Resized_Ablation_Segmentation')

            if not os.path.exists(parent_directory):
                os.makedirs(parent_directory)
                if not os.path.exists(child_directory_tumor):
                    os.makedirs(child_directory_tumor)
                if not os.path.exists(child_directory_ablation):
                    os.makedirs(child_directory_ablation)

            # Save the Re-sized Segmentations to DICOM Series
            obj_writer1 = WriterClass.DicomWriter(resized_tumor_mask, source_img_plan,
                                                  child_directory_tumor,
                                                  'tumorSegm', str(self.patients[idx]))
            obj_writer1.save_image_to_file()

            obj_writer2 = WriterClass.DicomWriter(resized_ablation_mask, source_img_validation,
                                                  child_directory_ablation,
                                                  'ablationSegm', str(self.patients[idx]))
            obj_writer2.save_image_to_file()

