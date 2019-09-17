# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:43:36 2017

@author: Raluca Sandu
"""
import os
import time
import pandas as pd
import plotHistDistances as pm
from B_ResampleSegmentations import ResizeSegmentation
from distancemetrics import DistanceMetrics, AxisMetrics
from volumemetrics import VolumeMetrics


def main_distance_volume_metrics(patient_id, source_ct_ablation, source_ct_tumor,
                                 ablation_segmentation, tumor_segmentation,
                                 lesion_id, ablation_date, dir_plots,
                                 FLAG_SAVE_TO_EXCEL=True, title='Ablation to Tumor Euclidean Distances'):
    # %% Get Surface Distances
    surface_distance_metrics = DistanceMetrics(ablation_segmentation, tumor_segmentation)
    if surface_distance_metrics.num_tumor_surface_pixels > 0:
        df_distances_1set = surface_distance_metrics.get_SitkDistances()
        distanceMap = surface_distance_metrics.get_surface_distances()
        num_surface_pixels = surface_distance_metrics.num_tumor_surface_pixels
    else:
        df_distances_1set = None
    # %% Get Ablation Evaluation Metrics (distance and gray intensity)
    resizer = ResizeSegmentation(source_ct_ablation, ablation_segmentation)
    ablation_segmentation_resized = resizer.resample_segmentation()
    ablation_axis_metrics = AxisMetrics(source_ct_ablation, ablation_segmentation_resized)
    if ablation_axis_metrics.error_flag is False:
        df_ablation_metrics_1set = ablation_axis_metrics.get_axis_metrics_df()
    else:
        df_ablation_metrics_1set = None
    # %% call function to compute volume metrics
    evaloverlap = VolumeMetrics()
    evaloverlap.set_image_object(ablation_segmentation, tumor_segmentation)
    evaloverlap.set_volume_metrics()
    if evaloverlap.error_flag is False:
        df_volumes_1set = evaloverlap.get_volume_metrics_df()
    else:
        df_volumes_1set = None
    # %% PLOT the color coded histogram of the distances
    if (df_distances_1set is not None) and (distanceMap is not None):
        try:
            perc_smaller_equal_than_0, perc_0_5, perc_greater_than_5 = pm.plotHistDistances(pat_name=patient_id,
                                                                                            lesion_id=lesion_id,
                                                                                            rootdir=dir_plots,
                                                                                            distanceMap=distanceMap,
                                                                                            num_voxels=num_surface_pixels,
                                                                                            title=title,
                                                                                            ablation_date=ablation_date)

        except Exception:
            print(patient_id, 'error plotting the distances and volumes')
    else:
        return

    SurfaceDistances_raw_numbers = {
        'patient_id': patient_id,
        'lesion_id': lesion_id,
        'ablation_date': ablation_date,
        'number_nonzero_surface_pixels': num_surface_pixels,
        'SurfaceDistances_Tumor2Ablation': distanceMap
    }

    SurfaceDistances_percentages = {
        'safety_margin_distribution: x<=0mm [%]': perc_smaller_equal_than_0,
        'safety_margin_distribution: 0<x<=5mm [%]': perc_0_5,
        'safety_margin_distribution: 5mm<x [%]': perc_greater_than_5
    }

    # %% Set UP the Final DataFrame by concatenating all the features extracted
    SurfaceDistances_dict_list = []
    SurfaceDistances_dict_list.append(SurfaceDistances_raw_numbers)
    df_SurfaceDistances = pd.DataFrame(SurfaceDistances_dict_list)

    patient_data = {'patient_id': patient_id,
                    'lesion_id': lesion_id,
                    'ablation_date': ablation_date}
    patient_list = []
    patient_list.append(patient_data)
    patient_df = pd.DataFrame(patient_list)
    SurfaceDistances_percentages_list = []
    SurfaceDistances_percentages_list.append(SurfaceDistances_percentages)
    df_SurfaceDistances_percentages = pd.DataFrame(SurfaceDistances_percentages_list)

    df_metrics = pd.concat(
        [patient_df, df_volumes_1set, df_distances_1set, df_ablation_metrics_1set, df_SurfaceDistances_percentages],
        axis=1)
    # %%  save to excel the average of the distance metrics '''
    if FLAG_SAVE_TO_EXCEL:
        timestr = time.strftime("%H%M%S-%Y%m%d")
        lesion_id_str = str(lesion_id)
        lesion_id = lesion_id_str.split('.')[0]
        filename = str(patient_id) + '_' + str(lesion_id) + '_' + 'AblationDate_' + str(
            ablation_date) + '_DistanceVolumeMetrics' + timestr + '.xlsx'
        filepath_excel = os.path.join(dir_plots, filename)
        writer = pd.ExcelWriter(filepath_excel)
        df_metrics.to_excel(writer, sheet_name='AT_metrics', index=False, float_format='%.4f')
        df_SurfaceDistances.to_excel(writer, sheet_name="SurfaceDistances", index=False, float_format="%.4f")
        writer.save()
        print('writing to Excel....', dir_plots)
