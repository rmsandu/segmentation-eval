# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


# read the ablation params from the brochure.
# interpolate if value not available. Nan if outside the range.
# read the tumor axis and average them out.
# add 10 and 5 mm to the radius
# plug in into the sphere formula
#plot against the predictive volume


def interpolation_fct(df_ablation, df_radiomics, title, fontsize=24, flag_tumor=None):
    """

    :param df_ablation:
    :param df_radiomics:
    :param title:
    :param fontsize:
    :param flag:
    :param flag_energy_axis:
    :return:
    """
    # perform interpolation as a function of  power and time (multivariate interpolation)
    points_power = np.asarray(df_ablation['Power']).reshape((len(df_ablation), 1))
    points_time = np.asarray(df_ablation['Time_Duration_Applied']).reshape((len(df_ablation), 1))
    power_and_time_brochure = np.hstack((points_power, points_time))
    ablation_vol_brochure = np.asarray(df_ablation['Ablation Volume [ml]_brochure']).reshape((len(df_ablation), 1))
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
    #%% PLOT Tumor Volume vs PAV (per colours subcapsular vs. non-subcapsular)
    # groupby needed.
    fig, ax = plt.subplots()
    if flag_tumor == 'Tumour Volume [ml]':
        tumor_volume = df_radiomics['Tumour Volume [ml]']
    elif flag_tumor == 'Tumour Volume + 10mm margin [ml]':
        tumor_volume = df_radiomics['Tumour Volume + 10mm margin [ml]']
    subcapsular = df_radiomics['Proximity_to_surface']
    df = pd.DataFrame(data=dict(x=ablation_vol_interpolated_brochure, y=tumor_volume, subcapsular=subcapsular))
    df.dropna(inplace=True)
    grouped = df.groupby(subcapsular)


if __name__ == '__main__':
    df_brochure = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_ablation_curated.xlsx")
    # TODO: extract Tumor Volume using getMeshVolumeFeatureValue() from pyRadiomics
    # radius_tumor = (df_radiomics['least_axis_length_tumor'] +
    #                 df_radiomics['minor_axis_length_tumor'] + df_radiomics['major_axis_length_tumor']) / 3
    # we have volume of sphere in voxels...we can try to find out the radius from there, rather than averaging out the 3 radii of an ellipsoid
    radius = ((3 * df_radiomics['Tumour Volume [ml]'])/4*pi) ** (1. / 3)
    df_radiomics['Tumor_Radius'] = radius
    df_radiomics['Tumour Volume + 10mm margin [ml]'] = (4 * pi * (df_radiomics['Tumor_Radius'] + 10) ** 3) / 3000
    df_radiomics['Tumor Volume + 5mm margin [ml]'] = (4 * pi * (df_radiomics['Tumor_Radius'] + 5) ** 3) / 3000
