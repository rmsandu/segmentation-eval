# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.graphing as gh


def draw_pie(dist,
             xpos,
             ypos,
             size,
             ax=None,
             colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    k = 0
    for r1, r2 in zip(pie[:-1], pie[1:]):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()
        xy = np.column_stack([x, y])
        ax.scatter([xpos], [ypos], marker=xy, s=size, color=colors[k], alpha=0.5)
        k += 1

    return ax


def call_plot_pies(df_radiomics, title=None, flag_plot_type=None, flag_overlap=None):
    """

    PREDICTED VS MEASURED SCATTER PIE CHART with distances represented
    :param flag_overlap:
    :param flag_plot_type:
    :param df_radiomics:
    :param title:
    :return:
    """
    fontsize = 18
    ablation_vol_interpolated_brochure = np.asanyarray(df_radiomics['Predicted_Ablation_Volume']).reshape(
        len(df_radiomics), 1)
    ablation_vol_measured = np.asarray(df_radiomics['Ablation Volume [ml]']).reshape(len(df_radiomics), 1)
    df_radiomics['MEV-MIV'] = df_radiomics['Outer Ellipsoid Volume'] - df_radiomics['Inner Ellipsoid Volume']
    df_radiomics['R(EAV:PAV)'] = df_radiomics['Ablation Volume [ml]'] / df_radiomics['Predicted_Ablation_Volume']

    ratios_0 = df_radiomics.safety_margin_distribution_0.tolist()
    ratios_5 = df_radiomics.safety_margin_distribution_5.tolist()
    ratios_10 = df_radiomics.safety_margin_distribution_10.tolist()

    # %% ACTUALLY PLOT STUFF
    fig, ax = plt.subplots()
    if flag_plot_type == 'PAV_EAV':
        for idx, val in enumerate(ablation_vol_interpolated_brochure):
            xs = ablation_vol_interpolated_brochure[idx]
            ys = ablation_vol_measured[idx]
            ratio_0 = ratios_0[idx] / 100
            ratio_5 = ratios_5[idx] / 100
            ratio_10 = ratios_10[idx] / 100
            if ~(np.isnan(xs)) and ~(np.isnan(ys)):
                draw_pie([ratio_0, ratio_5, ratio_10], xs, ys, 500, colors=['red', 'orange', 'green'], ax=ax)
        plt.ylabel('Effective Ablation Volume (mL)', fontsize=fontsize)
        plt.xlabel('Predicted Ablation Volume (mL)', fontsize=fontsize)
        plt.xlim([0, 80])
        plt.ylim([0, 80])

    elif flag_plot_type == 'MEV_MIV':
        # drop the rows where MIV > MEV
        # since the minimum inscribed ellipsoid (MIV) should always be smaller than the maximum enclosing ellipsoid (MEV)
        df_radiomics = df_radiomics[df_radiomics['Outer Ellipsoid Volume'] < 150]
        print('Nr Samples used for Outer Ellipsoid Volume < 150 ml:', len(df_radiomics))
        df_radiomics = df_radiomics[df_radiomics['MEV-MIV'] >= 0]
        print('Nr Samples used for MEV-MIV >=0 :', len(df_radiomics))
        r_eav_pav = np.asarray(df_radiomics['R(EAV:PAV)']).reshape(len(df_radiomics), 1)
        mev_miv = np.asarray(df_radiomics['MEV-MIV']).reshape(len(df_radiomics), 1)
        for idx, val in enumerate(mev_miv):
            xs = mev_miv[idx]
            ys = r_eav_pav[idx]
            ratio_0 = ratios_0[idx] / 100
            ratio_5 = ratios_5[idx] / 100
            ratio_10 = ratios_10[idx] / 100
            if ~(np.isnan(xs)) and ~(np.isnan(ys)):
                draw_pie([ratio_0, ratio_5, ratio_10], xs, ys, 500, colors=['red', 'orange', 'green'], ax=ax)
        plt.xlabel('Ablation Volume Irregularity (MEV-MIV) (mL)', fontsize=fontsize)
        plt.ylabel('R(EAV:PAV)', fontsize=fontsize)
        # plt.xlim([0, 80])
        # plt.ylim([0, 80])
        # ax.set_xscale('log')
    # %% EDIT THE PLOTS with colors
    red_patch = mpatches.Patch(color='red', label='Ablation Margin ' + r'$x < 0$' + 'mm')
    orange_patch = mpatches.Patch(color='orange', label='Ablation Margin ' + r'$0 \leq x < 5$' + 'mm')
    green_patch = mpatches.Patch(color='darkgreen', label='Ablation Margin ' + r'$x \geq 5$' + 'mm')
    plt.legend(handles=[red_patch, orange_patch, green_patch], fontsize=fontsize, loc='best',
               title=title, title_fontsize=fontsize+1)
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
    # textstr = title
    # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
    ax.tick_params(axis='y', labelsize=fontsize, color='k')
    ax.tick_params(axis='x', labelsize=fontsize, color='k')
    plt.tick_params(labelsize=fontsize, color='black')

    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'pie_charts_' + flag_plot_type + timestr)
    gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)


if __name__ == '__main__':
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_May192020.xlsx")
    # df_acculis = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    # df_radiomics_acculis = df_acculis[df_acculis['Inclusion_Margin_Analysis'] == 1]
    df_radiomics_all = df_radiomics[df_radiomics['Inclusion_Energy_PAV_EAV'] == True]

    # %% SELECT DEEP (aka  non-SUBCAPSULAR TUMORS) because we are only interested in plotting those for the moment
    df_radiomics_all = df_radiomics_all[df_radiomics_all['Proximity_to_surface'] == False]

    # %% SEPARATE THE MARGIN DISTRIBUTION
    df_radiomics_all.dropna(subset=['safety_margin_distribution_0',
                                    'safety_margin_distribution_5',
                                    'safety_margin_distribution_10'],
                            inplace=True)
    print('Nr Samples used for margin distribution available initally:', len(df_radiomics_all))
    # %% PLOT
    # Overlap measurements: Dice score, Volume Overlap Error,  Tumour residual volume [ml]
    call_plot_pies(df_radiomics_all, title='Non-Subcapsular Tumors', flag_plot_type='MEV_MIV')
    call_plot_pies(df_radiomics_all, title='Non-Subcapsular Tumors', flag_plot_type='PAV_EAV')

    # %% LATERAL ERROR Needle Plotting
    # df_radiomics_acculis.dropna(subset=['ValidationTargetPoint'], inplace=True)
    #
    # df_radiomics_acculis['center_tumor'] = df_radiomics_acculis[
    #     ['center_of_mass_x_tumor', 'center_of_mass_y_tumor', 'center_of_mass_z_tumor']].values.tolist()
    #
    # df_radiomics_acculis['TP_needle_1'] = df_radiomics_acculis['ValidationTargetPoint'].map(lambda x: x[1:len(x) - 1])
    # df_radiomics_acculis['TP_needle'] = df_radiomics_acculis['TP_needle_1'].map(
    #     lambda x: np.array([float(i) for i in x.split()]))
    # list_errors_needle = []
    # for row in df_radiomics_acculis.itertuples():
    #     try:
    #         needle_error = np.linalg.norm(row.center_tumor - row.TP_needle)
    #     except Exception:
    #         needle_error = np.nan
    #     list_errors_needle.append(needle_error)
    # print(len(list_errors_needle))
    # df_radiomics_acculis['needle_error'] = list_errors_needle
    # df_radiomics_acculis['needle_error'] = df_radiomics_acculis['LateralError']
