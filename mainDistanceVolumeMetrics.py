# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:43:36 2017

@author: Raluca Sandu
"""
import os
import time
import pandas as pd
import plotHistDistances as pm
from distancemetrics import DistanceMetrics
from volumemetrics import VolumeMetrics
import DistancesVolumes_twinAxes as twinAxes
import distances_boxplots_all_lesions as bpLesions


def main_distance_volume_metrics(df_patientdata, rootdir, FLAG_SAVE_TO_EXCEL=True):
    
    ablations = df_patientdata[' Ablation Segmentation Path'].tolist()
    tumors = df_patientdata[' Tumour Segmentation Path'].tolist()
    trajectory = df_patientdata['TrajectoryID']
    pats = df_patientdata['PatientID'].tolist()
    df_metrics_all = pd.DataFrame()
    distanceMaps_allPatients = []
    #%%
    # iterate through the lesions&ablations segmentations paths
    for idx, seg in enumerate(tumors):
        # call function to compute distance metrics
        evalmetrics = DistanceMetrics(ablations[idx], tumors[idx])
        df_distances_1set = evalmetrics.get_SitkDistances()
        # call function to compute volume metrics
        evaloverlap = VolumeMetrics()
        evaloverlap.set_image_object(ablations[idx], tumors[idx])
        evaloverlap.set_volume_metrics()
        df_volumes_1set = evaloverlap.get_volume_metrics_df()
        df_metrics = pd.concat([df_volumes_1set, df_distances_1set], axis=1)
        df_metrics_all = df_metrics_all.append(df_metrics)
        distanceMap = evalmetrics.get_surface_distances()
        distanceMaps_allPatients.append(distanceMap)
        num_surface_pixels = evalmetrics.num_tumor_surface_pixels
        #  plot the color coded histogram of the distances
        title = 'Ablation to Tumor Euclidean Distances'
        pm.plotHistDistances(pats[idx], trajectory[idx], rootdir, distanceMap, num_surface_pixels, title)
    #%%
    # add the Distance Map to the input dataframe. to be written to Excel
    df_patientdata['DistanceMaps'] = distanceMaps_allPatients
    df_patients_sorted = df_patientdata.sort_values(['PatientID'], ascending=True)
    data_distances_to_plot = df_patients_sorted['DistanceMaps'].tolist()
    data_volumes_to_plot = df_metrics_all[' Tumour coverage ratio']
    # plot Boxplot per patient
    bpLesions.plotBoxplots(data_distances_to_plot, rootdir)
    # plot distances in Boxplot vs Tumor Coverage Ratio
    twinAxes.plotBoxplots(data_distances_to_plot, data_volumes_to_plot, rootdir)
    #%% 
    ''' save to excel the average of the distance metrics '''
    if FLAG_SAVE_TO_EXCEL:
        print('writing to Excel....')
        print(rootdir)
        df_metrics_all.index = list(range(len(df_metrics_all)))
        df_final = pd.concat([df_patientdata, df_metrics_all], axis=1)
        timestr = time.strftime("%H%M%S-%Y%m%d")
        filename = 'DistanceVolumeMetrics_Pooled_' + title + '-' + timestr + '.xlsx'
        filepath_excel = os.path.join(rootdir, filename)
        writer = pd.ExcelWriter(filepath_excel)
        df_final.to_excel(writer, index=False, float_format='%.2f')
