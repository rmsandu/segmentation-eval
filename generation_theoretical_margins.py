# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn import linear_model

import utils.graphing as gh


def interpolation_fct(df_ablation, df_radiomics, title, fontsize=24, flag_tumor=None, lin_regr=False):
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
    # %% PLOT Tumor Volume vs PAV (per colours subcapsular vs. non-subcapsular)
    # groupby needed.
    fig, ax = plt.subplots()
    if flag_tumor == 'Tumour Volume [ml]':
        tumor_volume = df_radiomics['Tumour Volume [ml]']
    elif flag_tumor == 'Tumour Volume + 10mm margin [ml]':
        tumor_volume = df_radiomics['Tumour Volume + 10mm margin [ml]']
    subcapsular = df_radiomics['Proximity_to_surface']
    df = pd.DataFrame(data=dict(x=tumor_volume, y=ablation_vol_interpolated_brochure, subcapsular=subcapsular))
    df.dropna(inplace=True)
    grouped = df.groupby(subcapsular)
    labels = ['Subcapsular Tumors', 'Deep Tumors']
    for i, (name, group) in enumerate(grouped):
        plt.scatter(group.x, group.y, alpha=0.5, label=labels[i], s=100)
    plt.legend(title=title + '(n = ' + str(len(df)) + ' )', title_fontsize=fontsize, fontsize=fontsize)
    plt.ylabel('Predicted Ablation Volume [ml]', fontsize=fontsize)
    plt.xlabel(flag_tumor, fontsize=fontsize)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    ax.tick_params(axis='y', labelsize=fontsize, color='k')
    ax.tick_params(axis='x', labelsize=fontsize, color='k')
    plt.tick_params(labelsize=fontsize, color='black')
    if lin_regr is True:
        X = np.asarray(df.x).reshape(len(df), 1)
        Y = np.asarray(df.y).reshape(len(df), 1)
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        SS_tot = np.sum((Y - np.mean(Y)) ** 2)
        residuals = Y - regr.predict(X)
        SS_res = np.sum(residuals ** 2)
        r_squared = 1 - (SS_res / SS_tot)
        correlation_coef = np.corrcoef(X[:, 0], Y[:, 0])[0, 1]
        label_r2 = r'$R^2:{0:.2f}$'.format(r_squared)
        label_r = r'$r: {0:.2f}$'.format(correlation_coef)
        ax.tick_params(axis='y', labelsize=fontsize, color='k')
        ax.tick_params(axis='x', labelsize=fontsize, color='k')
        plt.tick_params(labelsize=fontsize, color='black')
        plt.plot([], [], ' ', label=label_r)
        plt.plot([], [], ' ', label=label_r2)
        reg = plt.plot(X, regr.predict(X), color='black', linewidth=1.5, label='Linear Regression')
        plt.legend(title=title, title_fontsize=fontsize, fontsize=fontsize)
    plt.show()
    figpath = os.path.join("figures", title +  '_PAV_vs_' + flag_tumor)
    gh.save(figpath, width=12, height=12, ext=["png"], close=True, tight=True, dpi=600)


if __name__ == '__main__':
    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_ablation_curated.xlsx")
    radius = ((3 * df_radiomics['Tumour Volume [ml]'] * 1000) / 4 * pi) ** (1. / 3)
    df_radiomics['Tumor_Radius'] = radius
    df_radiomics['Tumour Volume + 10mm margin [ml]'] = (4 * pi * (df_radiomics['Tumor_Radius'] + 10) ** 3) / 3000
    df_radiomics['Tumor Volume + 5mm margin [ml]'] = (4 * pi * (df_radiomics['Tumor_Radius'] + 5) ** 3) / 3000

    df_amica = df_ablation[df_ablation['Device_name'] == 'Amica (Probe)']
    df_amica.reset_index(inplace=True)
    df_angyodinamics = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']
    df_angyodinamics.reset_index(inplace=True)
    df_covidien = df_ablation[df_ablation['Device_name'] == 'Covidien (Covidien MWA)']
    df_covidien.reset_index(inplace=True)
    # df_radiomics = df_radiomics[(df_radiomics['Energy [kj]'] > 0) & (df_radiomics['Energy [kj]'] <= 100)]
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    df_radiomics_amica.reset_index(inplace=True)
    df_radiomics_angyodinamics = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_angyodinamics.reset_index(inplace=True)
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics_covidien.reset_index(inplace=True)

    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, 'Solero', fontsize=24,
                      flag_tumor='Tumour Volume + 10mm margin [ml]')
    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, 'Solero', fontsize=24,
                      flag_tumor='Tumour Volume [ml]')
    interpolation_fct(df_amica, df_radiomics_amica, 'Amica', fontsize=24,
                      flag_tumor='Tumour Volume + 10mm margin [ml]')
    interpolation_fct(df_covidien, df_radiomics_covidien, 'Covidien', fontsize=24,
                      flag_tumor='Tumour Volume + 10mm margin [ml]')
    interpolation_fct(df_amica, df_radiomics_amica, 'Amica', fontsize=24,
                      flag_tumor='Tumour Volume [ml]')
    interpolation_fct(df_covidien, df_radiomics_covidien, 'Covidien', fontsize=24,
                      flag_tumor='Tumour Volume [ml]')

