# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:42:27 2018

@author: Raluca Sandu
"""

import os
import DicomReader as Reader
import DicomWriter as DicomWriter
import PasteRoiImage as PasteRoi


class ResizeSegmentations:

    def __init__(self, df_folderpaths):
        self.ablation_paths = df_folderpaths[' Ablation Segmentation Path'].tolist()
        self.tumor_paths = df_folderpaths[' Tumour Segmentation Path'].tolist()
        self.folder_path_plan = df_folderpaths['Plan Images Path'].tolist()
        self.folder_path_validation = df_folderpaths['Validation Images Path'].tolist()
        self.trajectoryID = df_folderpaths['TrajectoryID'].tolist()
        self.patients = df_folderpaths['PatientID'].tolist()
        # tolist() might be unnecessary

    def save_images_to_disk(self, root_path_to_save):

        for idx, ablation_path in enumerate(self.ablation_paths):

            print(str(self.patients[idx]))
            tumor_mask = Reader.read_dcm_series(self.tumor_paths[idx])
     
            ablation_mask = Reader.read_dcm_series(ablation_path)
            source_img_plan = Reader.read_dcm_series(self.folder_path_plan[idx])
            source_img_validation = Reader.read_dcm_series(self.folder_path_validation[idx])
            # resize the Segmentation Mask to the dimensions of the source images they were derived from '''
            resized_tumor_mask = PasteRoi.paste_roi_imageMaxSize(source_img_plan, source_img_validation,
                                                                 tumor_mask)
            resized_ablation_mask = PasteRoi.paste_roi_imageMaxSize(source_img_plan, source_img_validation,
                                                                    ablation_mask)

            parent_directory = os.path.join(root_path_to_save,
                                            'Pat_GTDB_' + str(self.patients[idx]))

            child_directory_trajectory = os.path.join(parent_directory, 'Trajectory' + str(self.trajectoryID[idx]))
            child_directory_tumor = os.path.join(parent_directory,
                                                 child_directory_trajectory,
                                                 'Resized_Tumor_Segmentation')
            child_directory_ablation = os.path.join(parent_directory,
                                                    child_directory_trajectory,
                                                    'Resized_Ablation_Segmentation')

            if not os.path.exists(parent_directory):
                os.makedirs(parent_directory)
                os.makedirs(child_directory_trajectory)
                os.makedirs(child_directory_tumor)
                os.makedirs(child_directory_ablation)
            else:
                # the patient folder already exists, create new trajectory folder with lesion and ablation folder
                if not os.path.exists(child_directory_trajectory):
                    os.makedirs(child_directory_trajectory)
                if not os.path.exists(child_directory_tumor):
                    # trajectory folder exists, re-write segmentations
                    os.makedirs(child_directory_tumor)
                    os.makedirs(child_directory_ablation)

            # Save the Re-sized Segmentations to DICOM Series
            obj_writer1 = DicomWriter.DicomWriter(resized_tumor_mask, source_img_plan,
                                                  child_directory_tumor,
                                                  'tumorSegm', str(self.patients[idx]))
            obj_writer1.save_image_to_file()

            obj_writer2 = DicomWriter.DicomWriter(resized_ablation_mask, source_img_validation,
                                                  child_directory_ablation,
                                                  'ablationSegm', str(self.patients[idx]))
            obj_writer2.save_image_to_file()

