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


def main_distance_volume_metrics(df_patientdata, pats, ablations, tumors, trajectories, rootdir,
                                 pathology='Metastases',
                                 FLAG_SAVE_TO_EXCEL=True,
                                 title='Ablation to Tumor Euclidean Distances'):

    df_patientdata['Pathology'] = pathology
    # pats = df_patientdata['PatientID'].tolist()
    df_metrics_all = pd.DataFrame()
    distanceMaps_allPatients = []
    # %% CALL Distance and Volume Metrics and save them to DataFrame, then to CSV
    # iterate through the lesions&ablations segmentations paths
    for idx in range(0, len(tumors)):
        if not (str(tumors[idx]) == 'nan') and not (str(ablations[idx]) == 'nan'):
            # TODO: update patient id in the plots
            # call function to compute distance metrics
            try:
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
                pm.plotHistDistances(pat_name=pats[idx],
                                     trajectory_idx=trajectories[idx],
                                     pathology=pathology[idx],
                                     rootdir=rootdir,
                                     distanceMap=distanceMap,
                                     num_voxels=num_surface_pixels,
                                     title=title)
            except Exception:
                # append empty dataframe
                numRows, numCols = df_metrics_all.shape
                numRows = 1
                df_empty = pd.DataFrame(index=range(numRows), columns=range(numCols))
                df_metrics_all = df_metrics_all.append(df_empty)
                # append empty DataFrame
                distanceMaps_allPatients.append([])
                # TODO: need to set the columns as well
                print(str(pats[idx]), 'error computing the distances and volumes')
                continue
        else:
            # if the segmentation path and ablation are empty
            numRows, numCols = df_metrics_all.shape
            numRows = 1
            df_empty = pd.DataFrame(index=range(numRows), columns=range(numCols))
            df_metrics_all = df_metrics_all.append(df_empty)
            # append empty DataFrame
            distanceMaps_allPatients.append([])
    # %%
    # add the Distance Map to the input dataframe. to be written to Excel
    df_patientdata['DistanceMaps'] = distanceMaps_allPatients
    df_metrics_all.index = list(range(len(df_metrics_all)))
    df_final = pd.concat([df_patientdata, df_metrics_all], axis=1)
    df_patients_sorted = df_final.sort_values(['PatientID'], ascending=True)
    # data_distances_to_plot = df_patients_sorted['DistanceMaps'].tolist()
    # data_volumes_to_plot = df_patients_sorted['Tumour coverage ratio']
    # plot Boxplot per patient
    # bpLesions.plotBoxplots(data_distances_to_plot, rootdir)
    # plot distances in Boxplot vs Tumor Coverage Ratio
    # twinAxes.plotBoxplots(data_distances_to_plot, data_volumes_to_plot, rootdir)
    # %%
    ''' save to excel the average of the distance metrics '''
    if FLAG_SAVE_TO_EXCEL:
        print('writing to Excel....', rootdir)
        timestr = time.strftime("%H%M%S-%Y%m%d")
        filename = 'DistanceVolumeMetrics_Pooled_' + title + '-' + timestr + '.xlsx'
        filepath_excel = os.path.join(rootdir, filename)
        writer = pd.ExcelWriter(filepath_excel)
        df_patients_sorted.to_excel(writer, index=False, float_format='%.2f')
        writer.save()
