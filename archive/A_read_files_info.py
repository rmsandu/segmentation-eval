# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:43:49 2018
@author: Raluca Sandu
1. read excel file output from xml-processing/mainExtractTrajectories.
2. gather patient segmentations mask filepaths
3. resize segmentations so quantitative eval metrics can be computed --> output: images & updated excel with new filepaths
4. read this file with the paths of the resized files compute the eval metrics (optional, run script C_mainDistanceVolumeMetrics)

"""

import os
import pandas as pd
import B_Resize_Resample_Images as ResizerClass
import readInputKeyboard

pd.options.mode.chained_assignment = None


def redcap_info(df_folderpaths):
    input_filepaths_maverric_key = readInputKeyboard.getNonEmptyString(
        "Filepath to CSV/Excel with input segmentations paths : ")
    # add the maverric study id
    df_maverric_key = pd.read_excel(input_filepaths_maverric_key)
    col_patient_id = df_maverric_key["ID nr"]
    df_maverric_key["maverric no"] = df_maverric_key["maverric no"].str.upper()
    df_maverric_key["ID_4nr"] = df_maverric_key["ID nr"].apply(lambda x: x.split('-')[1])
    dict_maverric_keys = dict(zip(df_maverric_key["ID_4nr"], df_maverric_key["maverric no"]))
    # read maverric key and ID nr colum
    # patient - take last 4 numbers before the dash.
    # match with new patient ID

    df_folderpaths["PatientID"] = df_folderpaths["PatientID"].astype(str)
    patient_ids = df_folderpaths["PatientID"].tolist()
    patient_maverric_id_list = []
    for idx, patient_id in enumerate(patient_ids):
        # look for the patient id in the dict_maverric_keys
        id_4nr = str(patient_id)[-4:]
        maverric_id = dict_maverric_keys.get(id_4nr)
        if maverric_id is not None:
            patient_maverric_id_list.append(maverric_id.upper())
        else:
            print('Patient Key ID not found:', patient_id)
            patient_maverric_id_list.append(None)
    df_folderpaths['MAVERRIC_ID'] = patient_maverric_id_list


if __name__ == '__main__':

    rootdir = os.path.normpath(readInputKeyboard.getNonEmptyString("Root Directory File Path with Patient Datasets: "))

    input_filepaths = readInputKeyboard.getNonEmptyString("Filepath to CSV/Excel with input segmentations paths: ")
    df_folderpaths = pd.read_excel(input_filepaths, sheet_name='Trajectories', converters={'PatientID': str})

    folder_path_new_resized_images = readInputKeyboard.getNonEmptyString(
        "FilePath to Folder for saving the resized segmentations: ")

    flag_resize_only_segmentations = readInputKeyboard.getChoiceYesNo(
        "Do you want to resize only and the segmentation mask?", ['Y', 'N'])

    flag_match_with_patient_studyID = readInputKeyboard.getChoiceYesNo(
        "Match patient id from input file with patient id from another file?", ['Y', 'N'])

    flag_extract_max_size = readInputKeyboard.getChoiceYesNo(
        'Do you want to resize the segmentations mask images and their sources to the same sizes?', ['Y', 'N'])

    flag_plot_id = readInputKeyboard.getChoice(
        "Do you want to plot metrics per single lesion or single needle trajectory?", ['l', 'n'])

    # read the excel with the maverric id and match patients id on maverric study id
    # optional step
    if flag_match_with_patient_studyID is True:
        redcap_info(df_folderpaths)
        # todo: is the dataframe updated with pointer reference (???)
    print('Finished Reading Input Files....')
    # %% 2. Call The ResizerClass to Resize Segmentations
    resize_object = ResizerClass.ResizeSegmentations(df_folderpaths=df_folderpaths,
                                                     root_path_to_save=folder_path_new_resized_images,
                                                     flag_extract_max_size=False,
                                                     flag_resize_only_segmentations=flag_resize_only_segmentations,
                                                     flag_plot_id=flag_plot_id)
    # call function to resize images
    resize_object.I_call_resize_resample_all_images()
    # update excel with filepaths of resized images
    df_resized_filepaths = resize_object.get_new_filepaths()

    # write the filepaths of the new resized to disk, in the folder of the resized images
    filename = 'FilepathsResizedGTSegmentations' + '.xlsx'
    filepathExcel_resized_segmentations = os.path.join(folder_path_new_resized_images, filename)
    writer = pd.ExcelWriter(filepathExcel_resized_segmentations)
    df_resized_filepaths.to_excel(writer, index=False)
    writer.save()
    if flag_resize_only_segmentations is True:
        print("Finished Re-sizing and Re-writing the segmentation masks to disk...")
    else:
        print("Finished Re-sizing and Re-writing the segmentations masks and the source images to the disk...")

    print("Resizing segmentations finished. Continue with Calculating Distance and Volume Metrics next...")
