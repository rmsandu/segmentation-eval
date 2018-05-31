# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:42:27 2018

@author: Raluca Sandu
"""

import os
import pandas as pd
import DicomReader as Reader
import DicomWriter as DicomWriter
import PasteRoiImage as PasteRoi


class ResizeSegmentations:

    def __init__(self, df_folderpaths):
        self.df_folderpaths = df_folderpaths
        self.new_filepaths = []

    def save_images_to_disk(self, root_path_to_save):
        ablation_paths = self.df_folderpaths[' Ablation Segmentation Path'].tolist()
        tumor_paths = self.df_folderpaths[' Tumour Segmentation Path'].tolist()
        folder_path_plan = self.df_folderpaths['Plan Images Path'].tolist()
        folder_path_validation = self.df_folderpaths['Validation Images Path'].tolist()
        trajectoryID = self.df_folderpaths['TrajectoryID'].tolist()
        patients = self.df_folderpaths['PatientID'].tolist()

        for idx, ablation_path in enumerate(ablation_paths):

            tumor_mask = Reader.read_dcm_series(tumor_paths[idx])
            ablation_mask = Reader.read_dcm_series(ablation_path)
            source_img_plan = Reader.read_dcm_series(folder_path_plan[idx])
            source_img_validation = Reader.read_dcm_series(folder_path_validation[idx])

            # resize the Segmentation Mask to the dimensions of the source images they were derived from '''
            resized_tumor_mask = PasteRoi.paste_roi_imageMaxSize(source_img_plan, source_img_validation,
                                                                 tumor_mask)
            resized_ablation_mask = PasteRoi.paste_roi_imageMaxSize(source_img_plan, source_img_validation,
                                                                    ablation_mask)
            # create folder directories to write the new segmentations
            parent_directory = os.path.join(root_path_to_save,
                                            'Pat_GTDB_' + str(patients[idx]))
            child_directory_trajectory = os.path.join(parent_directory, 'Trajectory' + str(trajectoryID[idx]))
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
            # save new filepaths to dictionary
            dict_paths = {
                    ' Tumour Segmentation Path Resized':  child_directory_tumor,
                    ' Ablation Segmentation Path Resized': child_directory_ablation
                }
            self.new_filepaths.append(dict_paths)

            # Save the Re-sized Segmentations to DICOM Series
            obj_writer1 = DicomWriter.DicomWriter(resized_tumor_mask, source_img_plan,
                                                  child_directory_tumor,
                                                  'tumorSegm', str(patients[idx]))
            obj_writer1.save_image_to_file()

            obj_writer2 = DicomWriter.DicomWriter(resized_ablation_mask, source_img_validation,
                                                  child_directory_ablation,
                                                  'ablationSegm', str(patients[idx]))
            obj_writer2.save_image_to_file()

    def get_new_filepaths(self):
        df_new_filepaths = pd.DataFrame(self.new_filepaths)
        df_final = pd.concat([self.df_folderpaths, df_new_filepaths], axis=1)
        return df_final
