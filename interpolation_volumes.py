# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn import linear_model

import utils.graphing as gh


# plt.style.use('ggplot')


def interpolation_fct(df_ablation, df_radiomics, title, fontsize=24, flag_size=False, flag=None,
                      flag_energy_axis=False):
    # perform interpolation as a function of  power and time (multivariate interpolation)
    points_power = np.asarray(df_ablation['Power']).reshape((len(df_ablation), 1))
    points_time = np.asarray(df_ablation['Time_Duration_Applied']).reshape((len(df_ablation), 1))
    points = np.hstack((points_power, points_time))
    values = np.asarray(df_ablation['Ablation Volume [ml]_brochure']).reshape((len(df_ablation), 1))
    df_radiomics.dropna(subset=['Power', 'Time_Duration_Applied'], inplace=True)
    grid_x = np.asarray(df_radiomics['Power']).reshape((len(df_radiomics), 1))
    grid_y = np.asarray(df_radiomics['Time_Duration_Applied']).reshape((len(df_radiomics), 1))
    xi = np.hstack((grid_x, grid_y))
    ablation_vol_interpolated_brochure = griddata(points, values, xi, method='linear')

    # PREDICTED VS MEASURED
    ablation_vol_measured = np.asarray(df_radiomics['Ablation Volume [ml]']).reshape(len(df_radiomics), 1)
    if flag == 'Tumour Volume [ml]':
        size_values = np.asarray(df_radiomics['Tumour Volume [ml]']).reshape(len(df_radiomics), 1)
    elif flag == 'No. chemo cycles':
        # df_radiomics['no_chemo_cycle'] = df_radiomics['no_chemo_cycle'] +  1
        size_values = np.asarray(df_radiomics['no_chemo_cycle']).reshape(len(df_radiomics), 1)

    fig, ax = plt.subplots()

    if flag_size is True:
        size = np.asarray([100 * (n + 1) for n in size_values]).reshape(len(size_values), 1)
        size_mask = ~np.isnan(size)
        size = size[size_mask]
        sc = ax.scatter(ablation_vol_interpolated_brochure, ablation_vol_measured, color='steelblue', marker='o',
                        alpha=0.7, s=size)
        legend_1 = ax.legend(*sc.legend_elements("sizes", num=5, func=lambda x: x / 100 - 1, color='steelblue'),
                             title=flag, labelspacing=1.5, borderpad=0.75, handletextpad=2,
                             fontsize=fontsize - 2, loc='upper right')
        legend_1.get_title().set_fontsize(str(fontsize))
        ax.add_artist(legend_1)
    elif flag_energy_axis:
        ax2 = ax.twiny()
        ax.scatter(ablation_vol_interpolated_brochure, ablation_vol_measured, color='steelblue', marker='o', s=100)
        ax.set_ylabel('Effective Ablation Volume [ml]', fontsize=fontsize)
        ax.set_xlabel('Predicted Ablation Volume Brochure [ml]', fontsize=fontsize)
        ax.tick_params(axis='x', labelcolor='steelblue')
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 100])
        energy = df_radiomics['Energy [kj]']
        ax2.scatter(energy, ablation_vol_measured, color='purple', marker='*', s=100, alpha=0.5)
        ax2.set_xlabel('Energy [kj]', color='purple', fontsize=fontsize)
        ax2.tick_params(axis='x', colors='purple')
        ax2.set_ylim([0, 100])
        ax2.set_xlim([0, 100])
    else:
        sc = plt.scatter(ablation_vol_interpolated_brochure, ablation_vol_measured, color='steelblue', marker='o',
                         s=100)
    plt.ylabel('Effective Ablation Volume [ml]', fontsize=fontsize)
    plt.xlabel('Predicted Ablation Volume Brochure [ml]', fontsize=fontsize)
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    # get the data ready for linear regression
    X = ablation_vol_interpolated_brochure.reshape(len(ablation_vol_interpolated_brochure), 1)
    Y = ablation_vol_measured.reshape(len(ablation_vol_measured), 1)
    mask = ~np.isnan(X) & ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)
    nr_samples = len(X)
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
    # plt.plot([], [], ' ', label=label_r)
    # plt.plot([], [], ' ', label=label_r2)
    if flag is not None:
        reg = plt.plot(X, regr.predict(X), color='orange', linewidth=1.5)
        plt.legend(fontsize=fontsize, loc='upper left', title=title + ' (n = ' + str(nr_samples) + ' )',
                   title_fontsize=fontsize, labelspacing=0)
        figpathHist = os.path.join("figures",
                                   title + flag + '_ablation_vol_interpolated')
    else:
        reg = plt.plot(X, regr.predict(X), color='orange', linewidth=1.5, label='Linear Regression')
        plt.plot([], [], ' ', label='n = ' + str(nr_samples))
        plt.legend(fontsize=fontsize, loc='upper left', title=title, title_fontsize=fontsize - 2)
        figpathHist = os.path.join("figures", title + '_ablation_vol_interpolated')
    gh.save(figpathHist, width=12, height=12, ext=["png"], close=True, tight=True, dpi=600)

    return ablation_vol_interpolated_brochure


if __name__ == '__main__':
    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_111119.xlsx")
    # sort values
    df_ablation.sort_values(by=['Energy_brochure'], inplace=True)
    df_amica = df_ablation[df_ablation['Device_name'] == 'Amica (Probe)']
    df_angyodinamics = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']
    df_covidien = df_ablation[df_ablation['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics.sort_values(by=['Energy [kj]'], inplace=True)
    df_radiomics = df_radiomics[(df_radiomics['Energy [kj]'] > 0) & (df_radiomics['Energy [kj]'] <= 100)]
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    df_radiomics_angyodinamics = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']

    # flag_options : 1. flag == 'No. chemo cycles' 2. flag == 'Tumour Volume [ml]'
    interpolation_fct(df_amica, df_radiomics_amica, 'Amica', flag_size=True, flag='Tumour Volume [ml]')
    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, 'Solero', flag_size=True, flag='Tumour Volume [ml]')
    interpolation_fct(df_covidien, df_radiomics_covidien, 'Covidien', flag_size=True, flag='Tumour Volume [ml]')
