# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import pandas as pd

from utils.interpolation import predict_ablation_volume_griddata


def interpolation_fct(df_ablation, df_radiomics):
    """
    Compute the Predicted Ablation Value by linear interpolation using Power (Watts) and Time (seconds) using griddata
    :param df_ablation:
    :param df_radiomics:
    :return: Predicted Ablation Volume (ablation_vol_interpolated)
    """
    return predict_ablation_volume_griddata(df_ablation, df_radiomics,
                                            ablation_volume_col='Predicted_Ablation_Volume')


if __name__ == '__main__':
    df_ablation_brochure = pd.read_excel("/path/to/data/Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel("/path/to/data/radiomics_population.xlsx")

    # %% ACCULIS
    df_acculis = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_acculis = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    # call the interpolation functions
    ablation_vol_interpolated_brochure_acculis = interpolation_fct(df_acculis, df_radiomics_acculis)
    # %% COVIDIEN
    df_covidien = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']
    ablation_vol_interpolated_brochure_covidien = interpolation_fct(df_covidien, df_radiomics_covidien)
    # %% AMICA
    df_amica = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Amica (Probe)']
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    ablation_vol_interpolated_brochure_amica = interpolation_fct(df_amica, df_radiomics_amica)

    # replace in the dataframe the interpolated PAV at the exact location according to the MWA devices
    df_radiomics.loc[
        df_radiomics.Device_name == 'Angyodinamics (Acculis)', 'Predicted_Ablation_Volume'] = \
        ablation_vol_interpolated_brochure_acculis
    df_radiomics.loc[
        df_radiomics.Device_name == 'Covidien (Covidien MWA)', 'Predicted_Ablation_Volume'] = \
        ablation_vol_interpolated_brochure_covidien
    df_radiomics.loc[
        df_radiomics.Device_name == 'Amica (Probe)', 'Predicted_Ablation_Volume'] = \
        ablation_vol_interpolated_brochure_amica

    filepath_excel = 'radiomics_predicted_ablation_volume.xlsx'
    with pd.ExcelWriter(filepath_excel) as writer:
        df_radiomics.to_excel(writer, sheet_name='radiomics', index=False)
    print('Computed Predicted_Ablation_Volume for each MWA device (covidien, amica, angiodynamics)')
