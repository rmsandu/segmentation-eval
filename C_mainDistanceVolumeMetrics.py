# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:43:36 2017

@author: Raluca Sandu
"""
import os
import time

import pandas as pd

import scripts.plot_ablation_margin_hist as pm
from DistanceMetrics import DistanceMetrics, RadiomicsMetrics
from VolumeMetrics import VolumeMetrics


def main_distance_volume_metrics(patient_id, source_ct_ablation, source_ct_tumor,
                                 ablation_segmentation, tumor_segmentation_resampled,
                                 lesion_id, ablation_date, dir_plots,
                                 FLAG_SAVE_TO_EXCEL=True, title='Ablation to Tumor Euclidean Distances',
                                 calculate_volume_metrics=False, calculate_radiomics=False):
    # %% Get Surface Distances between tumor and ablation segmentations

    surface_distance_metrics = DistanceMetrics(ablation_segmentation, tumor_segmentation_resampled)
    if surface_distance_metrics.num_tumor_surface_pixels > 0:
        df_distances_1set = surface_distance_metrics.get_SitkDistances()
        distanceMap = surface_distance_metrics.get_surface_distances()
        num_surface_pixels = surface_distance_metrics.num_tumor_surface_pixels
    else:
        df_distances_1set = None
        distanceMap = None
        num_surface_pixels = None
    # %% Get Radiomics Metrics (shape and intensity)
    # ABLATION
    ablation_radiomics_metrics = RadiomicsMetrics(source_ct_ablation, ablation_segmentation)
    if ablation_radiomics_metrics.error_flag is False:
        df_ablation_metrics_1set = ablation_radiomics_metrics.get_axis_metrics_df()
        new_columns_name = df_ablation_metrics_1set.columns + '_ablation'
        df_ablation_metrics_1set.columns = new_columns_name
    else:
        df_ablation_metrics_1set = None
    # TUMOR
    tumor_radiomics_metrics = RadiomicsMetrics(source_ct_tumor, tumor_segmentation_resampled)
    if tumor_radiomics_metrics.error_flag is False:
        df_tumor_metrics_1set = tumor_radiomics_metrics.get_axis_metrics_df()
        new_columns_name = df_tumor_metrics_1set.columns + '_tumor'
        df_tumor_metrics_1set.columns = new_columns_name
    else:
        df_tumor_metrics_1set = None
    surface_distance_metrics = DistanceMetrics(ablation_segmentation, tumor_segmentation_resampled)
    if surface_distance_metrics.num_tumor_surface_pixels > 0:
        df_distances_1set = surface_distance_metrics.get_SitkDistances()
        distanceMap = surface_distance_metrics.get_surface_distances()
        num_surface_pixels = surface_distance_metrics.num_tumor_surface_pixels
    else:
        df_distances_1set = None
        distanceMap = None
        num_surface_pixels = None
    # %% Get Radiomics Metrics (shape and intensity)
    if calculate_radiomics:
        # ABLATION
        ablation_radiomics_metrics = RadiomicsMetrics(source_ct_ablation, ablation_segmentation)
        if ablation_radiomics_metrics.error_flag is False:
            df_ablation_metrics_1set = ablation_radiomics_metrics.get_axis_metrics_df()
            new_columns_name = df_ablation_metrics_1set.columns + '_ablation'
            df_ablation_metrics_1set.columns = new_columns_name
        else:
            df_ablation_metrics_1set = None
        # TUMOR
        tumor_radiomics_metrics = RadiomicsMetrics(source_ct_tumor, tumor_segmentation_resampled)
        if tumor_radiomics_metrics.error_flag is False:
            df_tumor_metrics_1set = tumor_radiomics_metrics.get_axis_metrics_df()
            new_columns_name = df_tumor_metrics_1set.columns + '_tumor'
            df_tumor_metrics_1set.columns = new_columns_name
        else:
            df_tumor_metrics_1set = None
    else:
        df_ablation_metrics_1set = None
        df_tumor_metrics_1set = None
#added surface_distance library, fixed bug in resampling


    # %% call function to compute volume metrics
    if calculate_volume_metrics:
        evaloverlap = VolumeMetrics()
        evaloverlap.set_image_object(ablation_segmentation, tumor_segmentation_resampled)
        evaloverlap.set_volume_metrics()
        if evaloverlap.error_flag is False:
            df_volumes_1set = evaloverlap.get_volume_metrics_df()
        else:
            df_volumes_1set = None
    else:
        df_volumes_1set = None
    # %% PLOT the color coded histogram of the distances
    if (df_distances_1set is not None) and (distanceMap is not None) and (num_surface_pixels is not None):
        try:
            perc_smaller_equal_than_0, perc_0_5, perc_greater_than_5 = pm.plot_histogram_surface_distances(
                pat_name=patient_id,
                lesion_id=lesion_id,
                rootdir=dir_plots,
                distanceMap=distanceMap,
                num_voxels=num_surface_pixels,
                title=title,
                ablation_date=ablation_date,
                flag_to_plot=True)

        except Exception:
            print(patient_id, ' error plotting the distances and volumes')
            perc_smaller_equal_than_0, perc_0_5, perc_greater_than_5 = None, None, None
    else:
        perc_smaller_equal_than_0, perc_0_5, perc_greater_than_5 = None, None, None

    SurfaceDistances_raw_numbers = {
        'patient_id': patient_id,
        'lesion_id': lesion_id,
        'ablation_date': ablation_date,
        'number_nonzero_surface_pixels': num_surface_pixels,
        'SurfaceDistances_Tumor2Ablation': distanceMap
    }

    SurfaceDistances_percentages = {
        'safety_margin_distribution_0': perc_smaller_equal_than_0,
        'safety_margin_distribution_5': perc_0_5,
        'safety_margin_distribution_10': perc_greater_than_5
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
        [patient_df, df_volumes_1set, df_distances_1set, df_ablation_metrics_1set, df_tumor_metrics_1set,
         df_SurfaceDistances_percentages], axis=1)
    # todo delete the next lien
    # df_metrics = pd.concat([patient_df, df_volumes_1set], axis=1)
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
        # df_SurfaceDistances.to_excel(writer, sheet_name="SurfaceDistances", index=False, float_format="%.4f")
        writer.save()
        print('writing to Excel....', dir_plots)
