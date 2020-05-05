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
    # df_radiomics.dropna(subset=['Power', 'Time_Duration_Applied'], inplace=True)
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
    # ablation_vol_measured = np.asarray(df_radiomics['Ablation Volume [ml]']).reshape(len(df_radiomics), )
    # # sanity check that both EAV and PAV have the same length (even if NaNs present)
    # if len(ablation_vol_interpolated_brochure) != len(ablation_vol_measured):
    #     print("something's not right")
    #     return None
    return ablation_vol_interpolated_brochure


if __name__ == '__main__':
    df_ablation_brochure = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_May2020.xlsx")
    # df_radiomics['Energy [kj]'] = (df_radiomics['Time_Duration_Applied'] * df_radiomics['Power']) / 1000

    # %% ACCULIS
    df_acculis = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_acculis = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    ablation_vol_interpolated_brochure_acculis = interpolation_fct(df_acculis, df_radiomics_acculis)
    df_radiomics_acculis['Ablation Volume (manufacturers)'] = ablation_vol_interpolated_brochure_acculis
    # %% COVIDIEN
    df_covidien = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']
    ablation_vol_interpolated_brochure_covidien = interpolation_fct(df_covidien, df_radiomics_covidien)
    # %% AMICA
    df_amica = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Amica (Probe)']
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    ablation_vol_interpolated_brochure_amica = interpolation_fct(df_covidien, df_radiomics_amica)
    # %%
    # df.loc[df.my_channel > 20000, 'my_channel'] = 0
    df_radiomics.loc[
        df_radiomics.Device_name == 'Angyodinamics (Acculis)', 'Predicted_Ablation_Volume'] = ablation_vol_interpolated_brochure_acculis
    df_radiomics.loc[
        df_radiomics.Device_name == 'Covidien (Covidien MWA)', 'Predicted_Ablation_Volume'] = ablation_vol_interpolated_brochure_covidien
    df_radiomics.loc[
        df_radiomics.Device_name == 'Amica (Probe)', 'Predicted_Ablation_Volume'] = ablation_vol_interpolated_brochure_amica

    filepath_excel = 'Radiomics_MAVERRIC_May2020.xlsx'
    writer = pd.ExcelWriter(filepath_excel)
    df_radiomics.to_excel(writer, sheet_name='radiomics', index=False)
    writer.save()
