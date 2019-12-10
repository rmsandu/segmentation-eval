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
        fig, ax = plt.subplots(figsize=(10,8))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
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


def interpolation_fct(df_ablation, df_radiomics, title=None):
    """
    Interpolate the missing ablation volumes using the power and time from the brochure
    :param df_ablation:
    :param df_radiomics:
    :param title:
    :param lin_regr:
    :param ratio_0:
    :param ratio_5:
    :param ratio_10:
    :return:
    """
    fontsize = 22
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

    # PREDICTED VS MEASURED SCATTER PIE CHART
    ablation_vol_measured = np.asarray(df_radiomics['Ablation Volume [ml]']).reshape(len(df_radiomics), 1)
    ratios_0 = df_radiomics.safety_margin_distribution_0.tolist()
    ratios_5 = df_radiomics.safety_margin_distribution_5.tolist()
    ratios_10 = df_radiomics.safety_margin_distribution_10.tolist()

    fig, ax = plt.subplots()
    for idx, val in enumerate(ablation_vol_interpolated_brochure):
        xs = ablation_vol_interpolated_brochure[idx]
        ys = ablation_vol_measured[idx]
        ratio_0 = ratios_0[idx] / 100
        ratio_5 = ratios_5[idx] / 100
        ratio_10 = ratios_10[idx] / 100
        draw_pie([ratio_0, ratio_5, ratio_10], xs, ys, 500, colors=['red', 'orange', 'green'], ax=ax)

    nr_samples = len(ablation_vol_measured)
    plt.ylabel('Effective Ablation Volume [ml]', fontsize=fontsize)
    plt.xlabel('Predicted Ablation Volume Brochure [ml]', fontsize=fontsize)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    red_patch = mpatches.Patch(color='red', label='Ablation Surface Margin ' + r'$x < 0$' + 'mm')
    orange_patch = mpatches.Patch(color='orange', label='Ablation Surface Margin ' + r'$0 \leq  x \leq 5$' + 'mm')
    green_patch = mpatches.Patch(color='darkgreen', label='Ablation Surface Margin ' + r'$x > 5$' + 'mm')
    plt.legend(handles=[red_patch, orange_patch, green_patch], fontsize=fontsize, loc='best')
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray')
    textstr = title
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
    ax.tick_params(axis='y', labelsize=fontsize, color='k')
    ax.tick_params(axis='x', labelsize=fontsize, color='k')
    plt.tick_params(labelsize=fontsize, color='black')
    figpath = os.path.join("figures", title + "_pie_charts")
    gh.save(figpath, width=14, height=10, close=True, dpi=600, tight=True)
    plt.show()


if __name__ == '__main__':

    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_ablation_curated.xlsx")
    # sort values
    df_ablation.sort_values(by=['Energy_brochure'], inplace=True)
    df_radiomics = df_radiomics[df_radiomics['Proximity_to_surface'] == True]
    df_radiomics.dropna(subset=['safety_margin_distribution_0',
                                'safety_margin_distribution_5',
                                'safety_margin_distribution_10'],
                                inplace=True)
    df_amica = df_ablation[df_ablation['Device_name'] == 'Amica (Probe)']
    df_angyodinamics = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']
    df_covidien = df_ablation[df_ablation['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics.sort_values(by=['Energy [kj]'], inplace=True)
    df_radiomics = df_radiomics[(df_radiomics['Energy [kj]'] > 0) & (df_radiomics['Energy [kj]'] <= 100)]
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    df_radiomics_angyodinamics = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']

    interpolation_fct(df_amica, df_radiomics_amica, title='Amica (Subcapsular)')
    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, title='Solero (Subcapsular)')
    interpolation_fct(df_covidien, df_radiomics_covidien, title='Covidien  (Subcapsular)')


    # fig, ax = plt.subplots()
    # draw_pie([0.2, 0.5, 0.3], 1, 1, 1000, colors=['red', 'orange', 'green'], ax=ax)
    # draw_pie([0.14, 0.1, 0.5], 2, 2, 1000, colors=['red', 'orange', 'green'], ax=ax)

