# -*- coding: utf-8 -*-
"""
Shared multivariate (Power, Time) -> Predicted Ablation Volume interpolation,
used by scripts/interpolation_volumes_tumor_size_marker.py and
import_from_csv/predict_PAV_brochure_interpolation.py.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def predict_ablation_volume_griddata(df_ablation, df_radiomics, ablation_volume_col):
    """
    Predict ablation volume for each row of df_radiomics by linearly
    interpolating the brochure's (Power, Time) -> ablation-volume grid.

    :param df_ablation: brochure/reference data with 'Power', 'Time_Duration_Applied',
        and `ablation_volume_col` columns.
    :param df_radiomics: measured data with 'Power' and 'Time_Duration_Applied' columns
        to predict a volume for.
    :param ablation_volume_col: name of the known-volume column in df_ablation.
    :return: 1D array of predicted ablation volumes, one per row of df_radiomics.
    """
    points_power = np.asarray(df_ablation['Power']).reshape((len(df_ablation), 1))
    points_time = np.asarray(df_ablation['Time_Duration_Applied']).reshape((len(df_ablation), 1))
    power_and_time_brochure = np.hstack((points_power, points_time))
    ablation_vol_brochure = np.asarray(df_ablation[ablation_volume_col]).reshape((len(df_ablation), 1))

    grid_x = np.array(pd.to_numeric(df_radiomics['Power'].to_numpy(), errors='coerce')).reshape(-1, 1)
    grid_y = np.array(pd.to_numeric(df_radiomics['Time_Duration_Applied'].to_numpy(), errors='coerce')).reshape(-1, 1)
    power_and_time_effective = np.asarray(np.hstack((grid_x, grid_y)))

    ablation_vol_interpolated = griddata(power_and_time_brochure, ablation_vol_brochure,
                                         power_and_time_effective, method='linear')
    return ablation_vol_interpolated.reshape(len(df_radiomics), )
