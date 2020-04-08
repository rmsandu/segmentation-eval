# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def interpolation_fct(df_ablation, df_radiomics):
    """

    :param df_ablation:
    :param df_radiomics:
    :return:
    """
    # perform interpolation as a function of  power and time (multivariate interpolation)
    points_power = np.asarray(df_ablation['Power']).reshape((len(df_ablation), 1))
    points_time = np.asarray(df_ablation['Time_Duration_Applied']).reshape((len(df_ablation), 1))
    power_and_time_brochure = np.hstack((points_power, points_time))
    ablation_vol_brochure = np.asarray(df_ablation['Predicted Ablation Volume (ml)']).reshape((len(df_ablation), 1))
    df_radiomics.dropna(subset=['Power', 'Time_Duration_Applied'], inplace=True)
    grid_x = df_radiomics['Power'].to_numpy()
    grid_y = df_radiomics['Time_Duration_Applied'].to_numpy()
    grid_x = np.array(pd.to_numeric(grid_x, errors='coerce'))
    grid_y = np.array(pd.to_numeric(grid_y, errors='coerce'))
    grid_x = grid_x.reshape(len(grid_x), 1)
    grid_y = grid_y.reshape(len(grid_y), 1)
    power_and_time_effective = np.asarray(np.hstack((grid_x, grid_y)))

    ablation_vol_interpolated_brochure = griddata(power_and_time_brochure, ablation_vol_brochure,
                                                  power_and_time_effective, method='linear')
    ablation_vol_interpolated_brochure = ablation_vol_interpolated_brochure.reshape(len(df_radiomics), )
    ablation_vol_measured = np.asarray(df_radiomics['Ablation Volume [ml]']).reshape(len(df_radiomics), )
    # sanity check that both EAV and PAV have the same length (even if NaNs present)
    if len(ablation_vol_interpolated_brochure) != len(ablation_vol_measured):
        print("something's not right")
        return None
    return ablation_vol_interpolated_brochure


if __name__ == '__main__':
    df_ablation_brochure = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_power_time.xlsx")
    df_radiomics_ = df_radiomics[(df_radiomics['Comments'].isnull())]
    # %% ACCULIS
    df_acculis = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Angyodinamics (Acculis)']
    df_acculis.reset_index(inplace=True)
    df_radiomics_acculis = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_acculis.reset_index(inplace=True)
    ablation_vol_interpolated_brochure_acculis = interpolation_fct(df_acculis, df_radiomics_acculis)
    # %% COVIDIEN
    df_covidien = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Covidien (Covidien MWA)']
    df_covidien.reset_index(inplace=True)
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics_covidien.reset_index(inplace=True)
    ablation_vol_interpolated_brochure_covidien = interpolation_fct(df_covidien, df_radiomics_covidien)
    # %% AMICA
    df_amica = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Amica (Probe)']
    df_amica.reset_index(inplace=True)
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    df_radiomics_amica.set_index(inplace=True)
    ablation_vol_interpolated_brochure_amica = interpolation_fct(df_covidien, df_radiomics_amica)
