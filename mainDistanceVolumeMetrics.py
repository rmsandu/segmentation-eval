# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:43:36 2017

@author: Raluca Sandu
"""
import os, re, time
import pandas as pd
import plotHistDistances as pm
#import plotBoxplotSurface as bp
from distancemetrics import DistanceMetrics
from volumemetrics import VolumeMetrics
import DistancesVolumes_twinAxes as twinAxes
import distances_boxplots_all_lesions as bpLesions


def main_distance_metrics(df_patientdata, rootdir, FLAG_SAVE_TO_EXCEL=True):
    
    ablations = df_patientdata['AblationFile'].tolist()
    reference = df_patientdata['TumorFile'].tolist()
    pats = df_patientdata['PatientName']
    pat_ids = []

    df_metrics_all = pd.DataFrame()
    distanceMaps_allPatients =[]
    #%%
    # iterate through the lesions&ablations segmentations paths 
    # call function to compute distance metrics
    # call function to compute volume metrics
        
    for idx, seg in enumerate(reference):
        
        evalmetrics = DistanceMetrics(ablations[idx], reference[idx])
        evaloverlap = VolumeMetrics(ablations[idx], reference[idx])
        df_distances_1set = evalmetrics.get_SitkDistances()
        df_volumes_1set = evaloverlap.volume_metrics
        df_metrics = pd.concat([df_volumes_1set, df_distances_1set], axis=1)
        df_metrics_all = df_metrics_all.append(df_metrics)
        distanceMap = evalmetrics.get_surface_distances()
        distanceMaps_allPatients.append(distanceMap)
        num_surface_pixels = evalmetrics.num_tumor_surface_pixels

        #%%
        '''extract the patient id from the folder/file path'''
        # where pats[idx] contains the name of the folder
        # and pat_id is extrated from the folder path, find the numeric index written in the folder/file path, assume that is the "true" patient ID
        try:
            pat_id_str = re.findall('\\d+', pats[idx])
            pat_id = int(pat_id_str[0])
            pat_ids.append(pat_id)
        except Exception:
            print('numeric data not found in the file name')
            pat_id = "p1" + str(idx)
            pat_ids.append(pat_id)
        # plot the color coded histogram of the distances
        title = 'Ablation to Tumor Euclidean Distances'
        pm.plotHistDistances(pats[idx], pat_id, rootdir,  distanceMap, num_surface_pixels, title)

    #%%
    '''plot boxplots of the distanceMaps for each lesion'''
    # sort rows ascending based on pat_id
    df_patientdata['PatientID'] = pat_ids
    df_patientdata['DistanceMaps'] = distanceMaps_allPatients
    df_patients_sorted = df_patientdata.sort_values(['PatientID'], ascending=True)
    data_toplot = df_patients_sorted['DistanceMaps'].tolist()
    bpLesions.plotBoxplots(data_toplot, rootdir)
    # twinAxes.plotBoxplots(data_toplot,rootdir)
    #%% 
    ''' save to excel the average of the distance metrics '''
    if FLAG_SAVE_TO_EXCEL:
        df_metrics_all.index = list(range(len(df_metrics_all)))
        df_final = pd.concat([df_patientdata, df_metrics_all], axis=1)
        timestr = time.strftime("%H%M%S-%Y%m%d")
        filename = 'DistanceVolumeMetrics_Pooled_' + title + '-' + timestr + '.xlsx'
        filepathExcel = os.path.join(rootdir, filename)
        writer = pd.ExcelWriter(filepathExcel)
        df_final.to_excel(writer, index=False, float_format='%.2f')
