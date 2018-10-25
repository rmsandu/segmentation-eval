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
import ResizeSegmentationsMain as ReaderWriterClass

pd.options.mode.chained_assignment = None

# %% 1. Read Excel File with Patient File Info Path
rootdir = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\3d_segmentation_maverric\maverric"
input_filepaths = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\3d_segmentation_maverric\maverric\MAVERRIC_Stockholm_October_all_patients.xlsx"
df_folderpaths = pd.read_excel(input_filepaths)

# %% 2. Call The ReaderWriterClass to Resize Segmentations
resize_object = ReaderWriterClass.ResizeSegmentations(df_folderpaths)
# TODO: create new folder for resized segmentations-append_datetime_ and the name of the excel file.
folder_path_saving = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\resized"
resize_object.save_images_to_disk(folder_path_saving)
df_resized_filepaths = resize_object.get_new_filepaths()
# %% 3. Apply Registration
# TODO: apply registration
# %%
timestr = time.strftime("%H%M%S-%Y%m%d")
filename = 'FilepathsResizedGTSegmentations' + '_' + timestr + '.xlsx'
filepathExcel = os.path.join(folder_path_saving, filename)
# filepathExcel = os.path.join(rootdir, filename)
writer = pd.ExcelWriter(filepathExcel)
df_resized_filepaths.to_excel(writer, index=False)
writer.save()
print("success")

# %% 4. call distance metrics
df_final = pd.read_excel(
    r"C:\PatientDatasets_GroundTruth_Database\Stockholm\maverric_processed_no_registration\Filepaths_Resized_GTSegmentations_Stockholm_June.xlsx")
df_new1 = df_final[[' Ablation Segmentation Path Resized',
                    ' Tumour Segmentation Path Resized',
                    'PatientID',
                    'TrajectoryID',
                    'Pathology']]
# df_new1.rename(columns={' Ablation Segmentation Path Resized': ' Ablation Segmentation Path',
#                         ' Tumour Segmentation Path Resized': ' Tumour Segmentation Path'}, inplace=True)

rootdir = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\plots"
Metrics.main_distance_volume_metrics(df_new1, rootdir)

df_patientdata = df_final
