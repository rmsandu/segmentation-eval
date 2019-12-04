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


def interpolation_fct(df_ablation, df_radiomics, title, fontsize=24, flag=None,
                      flag_energy_axis=False):
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
    # %% PLOT

    fig, ax = plt.subplots()
    if flag == 'Tumour Volume [ml]':
        size_values = np.asarray(df_radiomics['Tumour Volume [ml]']).reshape(len(df_radiomics), )
        df = pd.DataFrame(data=dict(x=ablation_vol_interpolated_brochure, y=ablation_vol_measured, sizes=size_values))
        df.dropna(inplace=True)
        bins = np.arange(start=0, stop=30, step=6)
        grouped = df.groupby(np.digitize(df.sizes, bins, right=True))
        sizes = [150 * (i + 1.) for i in range(5)]
        labels = ['0-5', '5-10', '15-20', '20-25', '25-30']
        nr_samples = len(df)
        for i, (name, group) in enumerate(grouped):
            plt.scatter(group.x, group.y, s=sizes[i], alpha=0.5, label=labels[name-1])
        legend1 = ax.legend(labelspacing=1, borderpad=0.75, title='Tumor Volume [ml]',
                            handletextpad=1.5, loc='upper right', fontsize=fontsize - 2, title_fontsize=fontsize - 2)
        ax.add_artist(legend1)

    elif flag == 'No. chemo cycles':
        size_values = np.asarray(df_radiomics['no_chemo_cycle']).reshape(len(df_radiomics), )
        df = pd.DataFrame()
        df['x'] = ablation_vol_interpolated_brochure
        df['y'] = ablation_vol_measured
        df['sizes'] = size_values
        df.dropna(inplace=True)
        nr_samples = len(df)
        bins = np.arange(start=0, stop=12, step=4)
        print(np.digitize(df.sizes, bins, right=True))
        grouped = df.groupby(np.digitize(df.sizes, bins, right=True))
        sizes = [150 * (i + 1.) for i in range(4)]
        labels = ['0', '1-4', '5-8', '9-12']
        for i, (name, group) in enumerate(grouped):
            plt.scatter(group.x, group.y, s=sizes[i], alpha=0.5, label=labels[name])
        legend1 = ax.legend(labelspacing=1, borderpad=0.75, title='Chemotherapy Cycles',
                   handletextpad=1.5, loc='upper right', fontsize=fontsize-2, title_fontsize=fontsize-2)
        ax.add_artist(legend1)

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
    reg = plt.plot(X, regr.predict(X), color='black', linewidth=1.5, linestyle='--', label='Linear Regression')
    if flag is not None:
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
        textstr = title + ' (n = ' + str(nr_samples) + ' )'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
        figpathHist = os.path.join("figures",
                                   title + flag + '_ablation_vol_interpolated')
    else:
        plt.plot([], [], ' ')
        plt.legend(fontsize=fontsize, loc='upper left', title=title + ' (n = ' + str(nr_samples) + ' )',
                   title_fontsize=fontsize, labelspacing=0)
        figpathHist = os.path.join("figures", title + '_ablation_vol_interpolated')
    gh.save(figpathHist, width=12, height=12, ext=["png"], close=True, tight=True, dpi=600)

    return ablation_vol_interpolated_brochure


if __name__ == '__main__':
    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_ablation_curated.xlsx")
    # sort values
    # df_ablation.sort_values(by=['Energy_brochure'], inplace=True)
    df_amica = df_ablation[df_ablation['Device_name'] == 'Amica (Probe)']
    df_angyodinamics = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']
    df_covidien = df_ablation[df_ablation['Device_name'] == 'Covidien (Covidien MWA)']
    # df_radiomics = df_radiomics[(df_radiomics['Energy [kj]'] > 0) & (df_radiomics['Energy [kj]'] <= 100)]
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    df_radiomics_angyodinamics = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']

    # flag_options : 1. flag == 'No. chemo cycles' 2. flag == 'Tumour Volume [ml]'
    interpolation_fct(df_amica, df_radiomics_amica, 'Amica', flag='Tumour Volume [ml]')
    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, 'Solero', flag='Tumour Volume [ml]')
    interpolation_fct(df_covidien, df_radiomics_covidien, 'Covidien', flag='Tumour Volume [ml]')
