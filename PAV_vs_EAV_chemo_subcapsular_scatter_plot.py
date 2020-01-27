# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import griddata

import plot_boxplots_PAV_vs_EAV
import utils.graphing as gh


def interpolation_fct(df_ablation, df_radiomics, device='Acculis', fontsize=24, flag_hue='subcapsular'):
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
    plot_boxplots_PAV_vs_EAV.plot_boxplots_volumes(ablation_vol_interpolated_brochure, ablation_vol_measured,
                                                   flag_subcapsular=False)
    # %% PLOT SCATTER PLOTS
    df_radiomics.loc[df_radiomics.no_chemo_cycle > 0, 'no_chemo_cycle'] = 'Yes'
    df_radiomics.loc[df_radiomics.no_chemo_cycle == 0, 'no_chemo_cycle'] = 'No'
    # create new pandas DataFrame for easier plotting
    df = pd.DataFrame()
    df['Tumor_Vol'] = df_radiomics['Tumour Volume [ml]']
    df['PAV'] = ablation_vol_interpolated_brochure
    df['EAV'] = ablation_vol_measured
    df['Subcapsular'] = df_radiomics['Proximity_to_surface']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['Chemotherapy'] = df_radiomics['no_chemo_cycle']
    df['Proximity_to_vessels'] = df_radiomics['Proximity_to_vessels']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))
    if flag_hue == 'chemotherapy':
        p = sns.lmplot(y="R(EAV:PAV)", x="Tumor_Vol", hue="Chemotherapy", data=df, markers=["o", "D"],
                       palette=['mediumvioletred', 'darkgreen'],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.5}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        chemo_true = df[df['Chemotherapy'] == 'Yes']
        chemo_false = df[df['Chemotherapy'] == 'No']
        slope, intercept, r_1, p_value, std_err = stats.linregress(chemo_false['Energy (kJ)'],
                                                                   chemo_false['R(EAV:PAV)'])
        slope, intercept, r_2, p_value, std_err = stats.linregress(chemo_true['Energy (kJ)'],
                                                                   chemo_true['R(EAV:PAV)'])
        label_2 = 'Chemotherapy: No'
        label_3 = 'Chemotherapy: Yes'
        ax = p.axes[0, 0]

    elif flag_hue == 'subcapsular':
        p = sns.lmplot(y="R(EAV:PAV)", x="Energy (kJ)", hue="Subcapsular", data=df, markers=["*", "s"],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.5}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        subcapsular_true = df[df['Subcapsular'] == False]
        subcapsular_false = df[df['Subcapsular'] == True]
        slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['Energy (kJ)'],
                                                                   subcapsular_false['R(EAV:PAV)'])
        slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['Energy (kJ)'],
                                                                   subcapsular_true['R(EAV:PAV)'])
        label_2 = 'Deep Tumors'
        label_3 = 'Subcapsular'
        ax = p.axes[0, 0]

    elif flag_hue == 'vessels':
        p = sns.lmplot(y="R(EAV:PAV)", x="Energy (kJ)", hue="Proximity_to_vessels", data=df, markers=["*", "s"],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.5}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        subcapsular_true = df[df['Proximity_to_vessels'] == False]
        subcapsular_false = df[df['Proximity_to_vessels'] == True]
        slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['Energy (kJ)'],
                                                                   subcapsular_false['R(EAV:PAV)'])
        slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['Energy (kJ)'],
                                                                   subcapsular_true['R(EAV:PAV)'])
        label_2 = 'Proximity to vessels: No'
        label_3 = 'Proximity to vessels: Yes'
        ax = p.axes[0, 0]

    else:
        # NO GROUPING SELECTED
        # sns.set_palette(sns.cubehelix_palette(8, start=2, rot =0, dark=0, light=.95, reverse=True))
        slope, intercept, r_square, p_value, std_err = stats.linregress(df['R(EAV:PAV)'], df['Energy (kJ)'])
        ax = sns.regplot(x="Energy (kJ)", y="R(EAV:PAV)", data=df, color=sns.xkcd_rgb["medium green"],
                         line_kws={'label': r'$R^2:{0:.4f}$'.format(r_square)},
                         scatter_kws={"s": 150, "alpha": 0.6})

    if flag_hue in ['vessels', 'subcapsular', 'chemotherapy']:
        ax.legend(fontsize=24, title_fontsize=24, title=device)
        leg = ax.get_legend()
        L_labels = leg.get_texts()
        label_line_1 = r'$R^2:{0:.2f}$'.format(r_1)
        label_line_2 = r'$R^2:{0:.2f}$'.format(r_2)
        L_labels[0].set_text(label_line_1)
        L_labels[1].set_text(label_line_2)
        L_labels[2].set_text(label_2)
        L_labels[3].set_text(label_3)
    else:
        ax.legend(fontsize=24, title_fontsize=24, title=device)
    plt.ylabel('R(EAV:PAV)', fontsize=24)
    plt.xlabel('Energy (kJ)', fontsize=24)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(labelsize=fontsize, color='k', width=2, length=10)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    figpath = os.path.join("figures", device + '_ratio_EAV_PAV_groups_' + flag_hue)
    gh.save(figpath, width=12, height=12, ext=["png"], close=True, tight=True, dpi=600)


if __name__ == '__main__':
    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_Acculis_MAVERRIC_22012020.xlsx")
    df_acculis = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']
    df_acculis.reset_index(inplace=True)
    # df_radiomics = df_radiomics[df_radiomics['Proximity_to_surface'] == False]
    df_radiomics = df_radiomics[(df_radiomics['Comments'].isnull())]
    df_radiomics_acculis = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_acculis.reset_index(inplace=True)
    # flag_hue='chemotherapy'
    # %% extract the needle error
    interpolation_fct(df_acculis, df_radiomics_acculis, 'Acculis MWA System', flag_hue='simple')
