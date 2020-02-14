# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import utils.graphing as gh


def draw_pie(dist,
             xpos,
             ypos,
             size,
             ax=None,
             colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

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


def interpolation_fct(df_ablation, df_radiomics, title=None, flag_needle_error=False, flag_overlap='Dice'):
    """
    Interpolate the missing ablation volumes using the power and time from the brochure
    :param df_ablation:
    :param df_radiomics:
    :param title:
    :return:
    """
    fontsize = 20
    # perform interpolation as a function of  power and time (multivariate interpolation)
    # EXTRACT VALUES FROM THE MWA BROCHURE
    points_power = np.asarray(df_ablation['Power']).reshape((len(df_ablation), 1))
    points_time = np.asarray(df_ablation['Time_Duration_Applied']).reshape((len(df_ablation), 1))
    points = np.hstack((points_power, points_time))
    values = np.asarray(df_ablation['Predicted Ablation Volume (ml)']).reshape((len(df_ablation), 1))
    # EXTRACT VALUES FROM RADIOMICS (Effective measured values)
    df_radiomics.dropna(subset=['Power', 'Time_Duration_Applied'], inplace=True)
    grid_x = np.asarray(df_radiomics['Power']).reshape((len(df_radiomics), 1))
    grid_y = np.asarray(df_radiomics['Time_Duration_Applied']).reshape((len(df_radiomics), 1))
    xi = np.hstack((grid_x, grid_y))
    ablation_vol_interpolated_brochure = griddata(points, values, xi, method='linear')

    # PREDICTED VS MEASURED SCATTER PIE CHART
    ablation_vol_measured = np.asarray(df_radiomics['Ablation Volume [ml]']).reshape(len(df_radiomics), 1)
    ratios_0 = df_radiomics.safety_margin_distribution_0.tolist()
    ratios_5 = df_radiomics.safety_margin_distribution_5.tolist()
    ratios_10 = df_radiomics.safety_margin_distribution_10.tolist()

    # ACTUALLY PLOT STUFF
    fig, ax = plt.subplots()
    if flag_needle_error is False:
        for idx, val in enumerate(ablation_vol_interpolated_brochure):
            xs = ablation_vol_interpolated_brochure[idx]
            ys = ablation_vol_measured[idx]
            ratio_0 = ratios_0[idx] / 100
            ratio_5 = ratios_5[idx] / 100
            ratio_10 = ratios_10[idx] / 100
            if ~(np.isnan(xs)) and ~(np.isnan(ys)):
                draw_pie([ratio_0, ratio_5, ratio_10], xs, ys, 500, colors=['red', 'orange', 'green'], ax=ax)
        plt.ylabel('Effective Ablation Volume (mL)', fontsize=fontsize)
        plt.xlabel('Predicted Ablation Volume Brochure (mL)', fontsize=fontsize)
        plt.xlim([0, 80])
        plt.ylim([0, 80])
    else:
        y = ablation_vol_measured / ablation_vol_interpolated_brochure  # ratio EAV/PAV
        fig_title = '_ratio_EAV_PAV'
        ylabel_text = 'R(EAV:PAV)'
        needle_error = np.asarray(df_radiomics['needle_error']).reshape(len(df_radiomics), 1)
        if flag_overlap == 'Dice':
            y = np.asarray(df_radiomics['Dice']).reshape(len(df_radiomics), 1)
            fig_title = '_Dice_'
            ylabel_text = 'Dice Score'
        if flag_overlap == 'Volume Overlap Error':
            y = np.asarray(df_radiomics['Volume Overlap Error']).reshape(len(df_radiomics), 1)
            fig_title = '_Volume Overlap Error'
            ylabel_text = 'Volume Overlap Error'
        if flag_overlap == 'Tumour residual volume [ml]':
            y = np.asarray(df_radiomics['Tumour residual volume [ml]']).reshape(len(df_radiomics), 1)
            # tumor_radius = np.asarray(df_radiomics['major_axis_length_tumor']/2).reshape(len(df_radiomics), 1)
            # y_normalized = df_radiomics['Tumour residual volume [ml]'] / df_radiomics['Tumour Volume [ml]']
            # x_normalized = needle_error/tumor_radius
            # y = np.asarray(y_normalized).reshape(len(df_radiomics), 1)
            # x = np.asarray(x_normalized).reshape(len(df_radiomics), 1)
            fig_title = '_Tumour residual volume [ml]'
            ylabel_text = 'Tumour residual volume (mL)'
        for idx, val in enumerate(ablation_vol_interpolated_brochure):
            ys = y[idx]
            xs = needle_error[idx]
            ratio_0 = ratios_0[idx] / 100
            ratio_5 = ratios_5[idx] / 100
            ratio_10 = ratios_10[idx] / 100
            if ~(np.isnan(xs)) and ~(np.isnan(ys)):
                draw_pie([ratio_0, ratio_5, ratio_10], xs, ys, 500, colors=['red', 'orange', 'green'], ax=ax)
        # ax.set_xscale('log')
        plt.ylabel(ylabel_text, fontsize=fontsize + 2)
        plt.xlabel('Lateral Error (mm)', fontsize=fontsize + 2)
        plt.xlim([-0.2, 6])
        plt.ylim([-0.2, 3])

    red_patch = mpatches.Patch(color='red', label='Ablation Margin ' + r'$x < 0$' + 'mm')
    orange_patch = mpatches.Patch(color='orange', label='Ablation Margin ' + r'$0 \leq  x \leq 5$' + 'mm')
    green_patch = mpatches.Patch(color='darkgreen', label='Ablation Margin ' + r'$x > 5$' + 'mm')
    plt.legend(handles=[red_patch, orange_patch, green_patch], fontsize=fontsize, loc='best',
               title=title, title_fontsize=21)
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
    # textstr = title
    # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
    ax.tick_params(axis='y', labelsize=fontsize, color='k')
    ax.tick_params(axis='x', labelsize=fontsize, color='k')
    plt.tick_params(labelsize=fontsize, color='black')
    # plt.xlim([8, 46])
    # plt.ylim([-0.2, 3.5])
    if flag_needle_error is True:
        figpath = os.path.join("figures", title + "_pie_charts_euclidean_error" + fig_title)
    else:
        figpath = os.path.join("figures", title + "_pie_charts")
    gh.save(figpath, width=14, height=10, close=True, dpi=600, tight=True)
    plt.show()


if __name__ == '__main__':
    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_Acculis_MAVERRIC_22012020.xlsx")
    df_acculis = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']

    #%% SELECT DEEP / SUBCAPSULAR TUMORS
    df_radiomics = df_radiomics[df_radiomics['Proximity_to_surface'] == False]

    #%% SEPARATE THE MARGIN DISTRIBUTION
    df_radiomics.dropna(subset=['safety_margin_distribution_0',
                                'safety_margin_distribution_5',
                                'safety_margin_distribution_10'],
                        inplace=True)
    df_radiomics = df_radiomics[(df_radiomics['Energy [kj]'] > 0) & (df_radiomics['Energy [kj]'] <= 100)]
    df_radiomics = df_radiomics[(df_radiomics['Comments'].isnull())]
    df_radiomics_acculis = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']

    df_radiomics_acculis.dropna(subset=['ValidationTargetPoint'], inplace=True)

    df_radiomics_acculis['center_tumor'] = df_radiomics_acculis[
        ['center_of_mass_x_tumor', 'center_of_mass_y_tumor', 'center_of_mass_z_tumor']].values.tolist()

    df_radiomics_acculis['TP_needle_1'] = df_radiomics_acculis['ValidationTargetPoint'].map(lambda x: x[1:len(x) - 1])
    df_radiomics_acculis['TP_needle'] = df_radiomics_acculis['TP_needle_1'].map(
        lambda x: np.array([float(i) for i in x.split()]))
    list_errors_needle = []
    for row in df_radiomics_acculis.itertuples():
        try:
            needle_error = np.linalg.norm(row.center_tumor - row.TP_needle)
        except Exception:
            needle_error = np.nan
        list_errors_needle.append(needle_error)
    print(len(list_errors_needle))
    df_radiomics_acculis['needle_error'] = list_errors_needle
    df_radiomics_acculis['needle_error'] = df_radiomics_acculis['LateralError']

    #%% PLOT
    # Overlap measurements: Dice score, Volume Overlap Error,  Tumour residual volume [ml]
    interpolation_fct(df_acculis, df_radiomics_acculis, title='Deep Tumors', flag_needle_error=True,
                      flag_overlap='Tumour residual volume [ml]')
