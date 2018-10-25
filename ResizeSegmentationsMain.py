# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:42:27 2018

@author: Raluca Sandu
"""

import os
import collections
import pandas as pd
import DicomReader as Reader
import DicomWriter as DicomWriter
import PasteRoiImage as PasteRoi


class ResizeSegmentations:

    def __init__(self, df_folderpaths):
        self.df_folderpaths = df_folderpaths
        self.new_filepaths = []

    def save_images_to_disk(self, root_path_to_save):
        # TODO: change to single and enumerate outside the function!
        ablation_paths = self.df_folderpaths['AblationPath'].tolist()
        tumor_paths = self.df_folderpaths['TumorPath'].tolist()
        folder_path_plan = self.df_folderpaths['PlanTumorPath'].tolist()
        patients_names = self.df_folderpaths['PatientName'].tolist()
        # folder_path_plan = self.df_folderpaths['SourceTumorPath'].tolist()
        folder_path_validation = self.df_folderpaths['ValidationAblationPath'].tolist()
        trajectoryID = self.df_folderpaths['NeedleNr'].tolist()
        patients = self.df_folderpaths['PatientID'].tolist()
        # TODO: create artificial patient id (e.g. 1, 2, 3, etc...)
        
        for idx in range(0,len(ablation_paths)):
            flag_tumor =  False
            flag_ablation = False
            if not (str(tumor_paths[idx])=='nan') and not(str(ablation_paths[idx])=='nan'): # if both paths exists
                tumor_mask, tumor_reader = Reader.read_dcm_series(tumor_paths[idx])
                source_img_plan, img_plan_reader = Reader.read_dcm_series(folder_path_plan[idx])
                ablation_mask, ablation_reader = Reader.read_dcm_series(ablation_paths[idx])
                source_img_validation, img_validation_reader = Reader.read_dcm_series(folder_path_validation[idx])
                # check if any of the variables are empty, then skip, else resize the metrics
                # execute the condition when true and all image sources could be read
                if not(not(tumor_mask and ablation_mask and source_img_plan and source_img_validation)):
                    # resize the Segmentation Mask to the dimensions of the source images they were derived from
                    flag_tumor = True
                    flag_ablation = True
                    tuple_imgs = collections.namedtuple('tuple_imgs',
                                                        ['img_plan',
                                                         'img_validation',
                                                         'ablation_mask',
                                                         'tumor_mask'])
                    images = tuple_imgs(source_img_plan, source_img_validation, ablation_mask, tumor_mask)
                    images_resized = PasteRoi.paste_roi_imageMaxSize(images)

                    #%% create folder directories to write the new segmentations
                    parent_directory = os.path.join(root_path_to_save,
                                                    "Pat_GTDB_" + str(patients_names[idx])+ '_' + str(patients[idx]))

                    child_directory_trajectory = os.path.join(parent_directory, "Trajectory" + str(trajectoryID[idx]))

                    child_directory_tumor = os.path.join(parent_directory,
                                                         child_directory_trajectory,
                                                         "Resized_Tumor_Segmentation")
                    child_directory_ablation = os.path.join(parent_directory,
                                                            child_directory_trajectory,
                                                            "Resized_Ablation_Segmentation")
                    child_directory_img_plan = os.path.join(parent_directory, 'CT_Planning')
                    child_directory_img_validation = os.path.join(parent_directory, 'CT_Validation')

                    if not os.path.exists(parent_directory):
                        os.makedirs(parent_directory)
                        os.makedirs(child_directory_trajectory)
                        os.makedirs(child_directory_tumor)
                        os.makedirs(child_directory_ablation)
                        os.makedirs(child_directory_img_plan)
                        os.makedirs(child_directory_img_validation)
                    else:
                        os.makedirs(child_directory_img_validation)
                        os.makedirs(child_directory_img_plan)
                        # the patient folder already exists, create new trajectory folder with lesion and ablation folder
                        if not os.path.exists(child_directory_trajectory):
                            os.makedirs(child_directory_trajectory)
                        if not os.path.exists(child_directory_tumor):
                            # trajectory folder exists, re-write segmentations
                            os.makedirs(child_directory_tumor)
                            os.makedirs(child_directory_ablation)

                    #%% Save the Re-sized Segmentations to DICOM Series
                    obj_writer = DicomWriter.DicomWriter(image=images_resized.tumor_mask,
                                                         folder_output=child_directory_tumor,
                                                         file_name="tumorSegm",
                                                         patient_id=str(patients[idx]),
                                                         series_reader=tumor_reader)
                    obj_writer.save_mask_image_to_file()

                    obj_writer1 = DicomWriter.DicomWriter(image=images_resized.ablation_mask,
                                                         folder_output=child_directory_ablation,
                                                         file_name="ablationSegm",
                                                         patient_id=str(patients[idx]),
                                                         series_reader=ablation_reader)
                    obj_writer1.save_mask_image_to_file()

                    obj_writer2 = DicomWriter.DicomWriter(image=images_resized.img_plan,
                                                          folder_output=child_directory_img_plan,
                                                          file_name="CT_Plan",
                                                          patient_id=str(patients[idx]),
                                                          series_reader=img_plan_reader)
                    obj_writer2.save_source_img_to_file()

                    obj_writer3 = DicomWriter.DicomWriter(image=images_resized.img_validation,
                                                          folder_output=child_directory_img_validation,
                                                          file_name="CT_Validation",
                                                          patient_id=str,
                                                          series_reader=img_validation_reader)
                    obj_writer3.save_source_img_to_file()



            #%% save new filepaths to dictionary
            if flag_tumor is False :
                child_directory_tumor = None
            if flag_ablation  is False:
                child_directory_ablation = None
            dict_paths = {
                    " Tumour Segmentation Path Resized":  child_directory_tumor,
                    " Ablation Segmentation Path Resized": child_directory_ablation
                }
            self.new_filepaths.append(dict_paths)

    def get_new_filepaths(self):
        df_new_filepaths = pd.DataFrame(self.new_filepaths)
        df_final = pd.concat([self.df_folderpaths, df_new_filepaths], axis=1)
        return df_final
