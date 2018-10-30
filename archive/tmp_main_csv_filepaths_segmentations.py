# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:30:11 2018

@author: Raluca Sandu
"""

import pandas as pd
import mainDistanceVolumeMetrics as Metrics
pd.options.mode.chained_assignment = None
#%%

file_2018 = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\FilepathsResizedGTSegmentations_114101-20180531.xlsx"
file_2017 = r"C:\PatientDatasets_GroundTruth_Database\GroundTruthDB_2017\FilepathsGTSegmentations2017_120657-20180531.xlsx"

df_2018 = pd.read_excel(file_2018)
df_2017 = pd.read_excel(file_2017)

df_new1 = df_2018[[' Ablation Segmentation Path Resized',
                  ' Tumour Segmentation Path Resized',
                  'PatientID',
                  'NeedleNr',
                  'Pathology']]
df_new1.rename(columns={' Ablation Segmentation Path Resized': ' Ablation Segmentation Path',
                        ' Tumour Segmentation Path Resized': ' Tumour Segmentation Path'}, inplace=True)

# create new dataframe with selected columns
# df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
df_final = df_2017.append(df_new1)
df_final = df_final.reset_index()
#%%
# call distance metrics
df_final = pd.read_excel(r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_MAVERRIC_Danderyd_June\FilepathsResizedGTSegmentations_121717-20180704.xlsx")
df_new1 = df_final[[' Ablation Segmentation Path Resized',
                  ' Tumour Segmentation Path Resized',
                  'PatientID',
                  'NeedleNr',
                  'Pathology']]
df_new1.rename(columns={' Ablation Segmentation Path Resized': ' Ablation Segmentation Path',
                        ' Tumour Segmentation Path Resized': ' Tumour Segmentation Path'}, inplace=True)
rootdir = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_MAVERRIC_Danderyd_June"
Metrics.main_distance_volume_metrics(df_new1, rootdir)
#df_patientdata = df_final
#def main_distance_volume_metrics(df_patientdata, rootdir, FLAG_SAVE_TO_EXCEL=True)