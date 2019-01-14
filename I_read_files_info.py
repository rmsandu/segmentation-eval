# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:43:49 2018
@author: Raluca Sandu
1. read excel file output from xml-processing/mainExtractTrajectories.
2. gather patient tumor/ablation filepaths
3. resize segmentations (new-filepaths) --> output: images & updated excel with new filepaths
4. read this file (or just the dataframe) and compute the metrics

Missing scripts (TODO):
 - registration
 - create DICOM Predicted Ellipsoid Mask
"""
# %%
import os
import time
import pandas as pd
import III_mainDistanceVolumeMetrics as Metrics
import II_Resize_Resample_Images as ResizerClass
pd.options.mode.chained_assignment = None

# %% 1. Read Excel File with Patient File Info Path
rootdir = r""
input_filepaths = r""

df_folderpaths = pd.read_excel(input_filepaths)
#%% add the maverric id
input_filepaths_maverric_key = r""
df_maverric_key = pd.read_excel(input_filepaths_maverric_key)
col_patient_id = df_maverric_key["ID nr"]
df_maverric_key["ID_4nr"] = df_maverric_key["ID nr"].apply(lambda x: x.split('-')[1])
dict_maverric_keys = dict(zip(df_maverric_key["ID_4nr"], df_maverric_key["maverric no"]))
# read maverric key and ID nr colum
# patient - take last 4 numbers before the dash.
# match with new patient ID
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

# %% 2. Call The ResizerClass to Resize Segmentations
folder_path_saving_resized = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\resized"
resize_object = ResizerClass.ResizeSegmentations(df_folderpaths=df_folderpaths,
                                                 root_path_to_save=folder_path_saving_resized,
                                                 flag_extract_max_size=False,
                                                 flag_resize_only_segmentations=True)
# add a new flag: flag_resize_only_segmentations
resize_object.I_call_resize_resample_all_images()
df_resized_filepaths = resize_object.get_new_filepaths()

# TODO: apply registration from the registration matrix...somewhere in the future.

timestr = time.strftime("%H%M%S-%Y%m%d")
# filename = 'FilepathsResizedGTSegmentations' + '_' + timestr + '.xlsx'
filename = 'FilepathsResizedGTSegmentations' + '.xlsx'
filepathExcel = os.path.join(folder_path_saving_resized, filename)
# filepathExcel = os.path.join(rootdir, filename)
writer = pd.ExcelWriter(filepathExcel)
df_resized_filepaths.to_excel(writer, index=False)
writer.save()
print("success")

# %% 4. call distance metrics. Plot HIstograms
df_final = pd.read_excel(
    r"C:\PatientDatasets_GroundTruth_Database\Stockholm\resized\FilepathsResizedGTSegmentations.xlsx")
rootdir_plots = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\plots"
# ablations = df_final['AblationPath']
# tumors = df_final['TumorPath']
ablations = df_final["Ablation Segmentation Path Resized"].tolist()
tumors = df_final["Tumour Segmentation Path Resized"].tolist()
trajectories = df_final['NeedleNr']
pats = df_final['MAVERRIC_ID']
Metrics.main_distance_volume_metrics(df_final, pats, ablations, tumors, trajectories, rootdir_plots)

