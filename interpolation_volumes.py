# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata
from sklearn import linear_model

import utils.graphing as gh


def interpolation_fct(df_ablation, df_radiomics, title, single_interpolation=True, flag_size=False):
    if single_interpolation is True:
        # perform interpolation as a function of energy
        df_radiomics.dropna(subset=['Energy [kj]'], inplace=True)
        x = np.asarray(df_ablation['Energy_brochure'])
        y = np.asarray(df_ablation['Ablation Volume [ml]_brochure'])
        f = interp1d(x, y, fill_value="extrapolate")
        df_radiomics.dropna(subset=['Ablation Volume [ml]'], inplace=True)
        energy = np.asarray(df_radiomics['Energy [kj]'])
        print('No of samples for ' + title + ': ', len(energy))
        ablation_vol_interpolated_brochure = f(energy)
        fig, ax = plt.subplots()
        plt.plot(x, y, 'o', energy, f(energy), '*')
        plt.legend(['data ' + title, 'linear interpolation'], loc='best')
        plt.xlabel('Energy [kJ]')
        plt.ylim([0, 120])
        plt.xlim([0, 80])
        plt.ylabel('Effective Ablation Volume Brochure [ml]')
        plt.grid('on')
        plt.show()
        figpathHist = os.path.join("figures", title + '_interpolation')
        gh.save(figpathHist, width=8, height=8, ext=["png"], close=True)
    else:
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
    tumor_vol = np.asarray(df_radiomics['Tumour Volume [ml]']).reshape(len(df_radiomics), 1)
    mask = ~np.isnan(ablation_vol_measured)
    # tumor_vol_size = tumor_vol[mask]
    # print('length_tumor_vol', len(tumor_vol_size))
    # print('length ablation vol measured', len(ablation_vol_measured))
    fig, ax = plt.subplots()
    if flag_size is True:
        size = np.asarray([50 * n for n in tumor_vol]).reshape(len(tumor_vol), 1)
        size_mask = ~np.isnan(size)
        size = size[size_mask]
        sc = ax.scatter(ablation_vol_interpolated_brochure, ablation_vol_measured, color='steelblue', marker='o',
                        alpha=0.7, s=size)
        plt.show()
        legend_1 = ax.legend(*sc.legend_elements("sizes", num=5, func=lambda x: x / 50, color='steelblue'),
                             title='Tumor Volume [ml]', labelspacing=1.5, borderpad=1.5, handletextpad=3.5,
                             fontsize=18, loc='upper right')
        legend_1.get_title().set_fontsize('18')
        ax.add_artist(legend_1)
    else:
        sc = plt.scatter(ablation_vol_interpolated_brochure, ablation_vol_measured, color='steelblue', marker='o',
                         s=100)
    plt.ylabel('Effective Ablation Volume [ml] for ' + title, fontsize=20)
    plt.xlabel('Predicted Ablation Volume Brochure [ml] for ' + title, fontsize=20)
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    # plt.title(title + '  Nr. Samples: ' + str(len(ablation_vol_measured)))
    X = ablation_vol_interpolated_brochure.reshape(len(ablation_vol_interpolated_brochure), 1)
    Y = ablation_vol_measured.reshape(len(ablation_vol_measured), 1)
    mask = ~np.isnan(X) & ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    SS_tot = np.sum((Y - np.mean(Y)) ** 2)
    residuals = Y - regr.predict(X)
    SS_res = np.sum(residuals ** 2)
    r_squared = 1 - (SS_res / SS_tot)
    correlation_coef = np.corrcoef(X[:, 0], Y[:, 0])[0, 1]
    label_r2 = r'$R^2:{0:.2f}$'.format(r_squared)
    label_r = r'$r: {0:.2f}$'.format(correlation_coef)
    ax.tick_params(axis='y', labelsize=20, color='k')
    ax.tick_params(axis='x', labelsize=20, color='k')
    plt.tick_params(labelsize=20, color='black')
    reg = plt.plot(X, regr.predict(X), color='orange', linewidth=1.5, label='Linear Regression')
    plt.plot([], [], ' ', label=label_r)
    plt.plot([], [], ' ', label=label_r2)
    plt.legend(fontsize=20, loc='upper left')
    if flag_size:
        figpathHist = os.path.join("figures",
                                   title + '_measured_vs_predicted_volume_power_time_interpolation_tumor_volume')
    else:
        figpathHist = os.path.join("figures", title + '_measured_vs_predicted_volume_power_time_interpolation')
    gh.save(figpathHist, width=11, height=11, ext=["png"], close=True, tight=True, dpi=600)

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

    interpolation_fct(df_amica, df_radiomics_amica, 'Amica', single_interpolation=False, flag_size=False)
    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, 'Angyodinamics (Solero)',
                      single_interpolation=False, flag_size=False)
    interpolation_fct(df_covidien, df_radiomics_covidien, 'Covidien', single_interpolation=False, flag_size=False)
