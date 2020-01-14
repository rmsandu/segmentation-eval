# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata
from sklearn import linear_model

import utils.graphing as gh

def interpolation_fct(df_ablation, df_radiomics, title, fontsize=24):
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

    # %% PLOT BOXPLOTS
    # boxplots_PAV_EAV.plot_boxplots_volumes(ablation_vol_interpolated_brochure, ablation_vol_measured, flag_subcapsular='all')
    # %% PLOT SCATTER PLOTS
    # fig, ax = plt.subplots()
    # df.loc[df.my_channel > 20000, 'my_channel'] = 0
    df_radiomics.loc[df_radiomics.no_chemo_cycle > 0, 'no_chemo_cycle'] = 'Yes'
    df_radiomics.loc[df_radiomics.no_chemo_cycle == 0, 'no_chemo_cycle'] = 'No'
    # create new pandas DataFrame for easier plotting
    df_chemo = pd.DataFrame()
    df_chemo['PAV'] = ablation_vol_interpolated_brochure
    df_chemo['EAV'] = ablation_vol_measured
    df_chemo['Subcapsular'] = df_radiomics['Proximity_to_surface']
    df_chemo['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df_chemo.dropna(inplace=True)
    df_chemo['R(EAV:PAV)'] = df_chemo['EAV'] / df_chemo['PAV']
    nr_samples = len(df_chemo)
    Y = np.asarray(df_chemo['R(EAV:PAV)']).reshape(len(df_chemo), 1)
    X = np.asarray(df_chemo['Energy (kJ)']).reshape(len(df_chemo), 1)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    SS_tot = np.sum((Y - np.mean(Y)) ** 2)
    residuals = Y - regr.predict(X)
    SS_res = np.sum(residuals ** 2)
    r_squared = 1 - (SS_res / SS_tot)
    correlation_coef = np.corrcoef(X[:, 0], Y[:, 0])[0, 1]
    label_r2 = r'$R^2:{0:.2f}$'.format(r_squared)
    label_r = r'$r: {0:.2f}$'.format(correlation_coef)
    # actually plot
    # sns.set(font_scale=2)
    p = sns.lmplot(x="PAV", y="EAV", hue="Subcapsular", data=df_chemo, markers=["*", "s"],
                    ci=None, scatter_kws={"s": 150, "alpha": 0.5},  line_kws = {'label': 'red'},
                   legend=True, legend_out = False)
    #
    ax = p.axes[0, 0]
    ax.legend(fontsize=24, title_fontsize=24, title='Acculis')
    leg = ax.get_legend()
    L_labels = leg.get_texts()
    label_line_1 = r'$R^2:{0:.2f}$'.format(0.02)
    label_line_2 = r'$R^2:{0:.2f}$'.format(0.1)
    L_labels[0].set_text(label_line_1)
    L_labels[1].set_text(label_line_2)
    L_labels[2].set_text('Deep Tumors')
    L_labels[3].set_text('Subcapsular Tumors')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('Predicted Ablation Volume (mL)', fontsize=24)
    plt.ylabel('Effective Ablation Volume (mL)', fontsize=24)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(labelsize=fontsize, color='k', width=2, length=10)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    # sns.set_context("paper")
    figpath = os.path.join("figures", 'Acculis_' + '_EAV_PAV_subcapsular_groups')

    gh.save(figpath , width=12, height=12, ext=["png"], close=True, tight=True, dpi=600)



if __name__ == '__main__':
    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_ablation_curated.xlsx")

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

    # flag_options : 1. flag == 'No. chemo cycles' 2. flag == 'Tumour Volume [ml]'

    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, 'Acculis')

    # interpolation_fct(df_amica, df_radiomics_amica, 'Amica', flag_energy_axis=False, flag_lin_regr=True)
    # interpolation_fct(df_covidien, df_radiomics_covidien, 'Covidien',  flag_energy_axis=False, flag_lin_regr=True)
