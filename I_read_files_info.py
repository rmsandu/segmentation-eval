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
import mainDistanceVolumeMetrics as Metrics
import II_Resize_Resample_Images as ResizerClass

pd.options.mode.chained_assignment = None

# %% 1. Read Excel File with Patient File Info Path
rootdir = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\3d_segmentation_maverric\maverric"
input_filepaths = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\3d_segmentation_maverric\maverric\MAVERRIC_Stockholm_October_all_patients.xlsx"
df_folderpaths = pd.read_excel(input_filepaths)

# %% 2. Call The ResizerClass to Resize Segmentations
folder_path_saving = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\resized"
resize_object = ResizerClass.ResizeSegmentations(df_folderpaths=df_folderpaths,
                                                 root_path_to_save=folder_path_saving,
                                                 flag_extract_max_size=False)
resize_object.I_call_resize_resample_all_images()
df_resized_filepaths = resize_object.get_new_filepaths()
# %% 3. Apply Registration
# TODO: apply registration
timestr = time.strftime("%H%M%S-%Y%m%d")
# filename = 'FilepathsResizedGTSegmentations' + '_' + timestr + '.xlsx'
filename = 'FilepathsResizedGTSegmentations' + '.xlsx'
filepathExcel = os.path.join(folder_path_saving, filename)
# filepathExcel = os.path.join(rootdir, filename)
writer = pd.ExcelWriter(filepathExcel)
df_resized_filepaths.to_excel(writer, index=False)
writer.save()
print("success")

# %% 4. call distance metrics. Plot HIstograms
df_final = pd.read_excel(
    r"C:\PatientDatasets_GroundTruth_Database\Stockholm\resized\FilepathsResizedGTSegmentations.xlsx")
# df_final = df_resized_filepaths
df_new1 = df_final[[' Ablation Segmentation Path Resized',
                    ' Tumour Segmentation Path Resized',
                    'PatientID',
                    'NeedleNr']]
# df_new1.rename(columns={' Ablation Segmentation Path Resized': ' Ablation Segmentation Path',
#                         ' Tumour Segmentation Path Resized': ' Tumour Segmentation Path'}, inplace=True)

rootdir = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\plots"
# call the metrics script
Metrics.main_distance_volume_metrics(df_new1, rootdir)
df_patientdata = df_final
