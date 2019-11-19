# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import matplotlib
import utils.graphing as gh
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from sklearn import linear_model
import numpy as np
import pandas as pd


def draw_pie_markers(xs, ys, ratios, sizes, colors, title):
    """

    :param xs:
    :param ys:
    :param ratios:
    :param sizes:
    :param colors:
    :param title:
    :return:
    """
    # assert sum(ratios) <= 1, 'sum of ratios needs to be < 1'
    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker': xy, 's': np.abs(xy).max() ** 2 * np.array(sizes), 'facecolor': color})
    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker)
    plt.ylabel('Effective Ablation Volume [ml] for ' + title, fontsize=20)
    plt.xlabel('Predicted Ablation Volume Brochure [ml] for ' + title, fontsize=20)
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    plt.show()
    figpath = os.path.join("figures", )
    gh.save(figpath, width=11, height=11, ext=["png"], close=True, tight=True, dpi=600)


def interpolation_fct(df_ablation, df_radiomics, title=None, lin_regr=False, ratios=ratios):
    """

    :param df_ablation:
    :param df_radiomics:
    :param title:
    :param lin_regr:
    :return:
    """

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
    fig, ax = plt.subplots()
    draw_pie_markers(xs=ablation_vol_interpolated_brochure,
                     ys=ablation_vol_interpolated_brochure,
                     ratios=ratios,
                     sizes=[300, 300, 300],
                     colors=['green', 'red', 'orange'],
                     title=title)

    if lin_regr is True:
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
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.xaxis.ticks.set_color('black')
        matplotlib.rc('axes', labelcolor='black')


if __name__ == '__main__':

    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_111119.xlsx")
    # sort values
    df_ablation.sort_values(by=['Energy_brochure'], inplace=True)
    df_ablation.dropna(subset=['safety_margin_distribution_0',
                               'safety_margin_distribution_5', 'safety_margin_distribution_10'],
                                inplace=True)
    df_amica = df_ablation[df_ablation['Device_name'] == 'Amica (Probe)']
    df_angyodinamics = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']
    df_covidien = df_ablation[df_ablation['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics.sort_values(by=['Energy [kj]'], inplace=True)
    df_radiomics = df_radiomics[(df_radiomics['Energy [kj]'] > 0) & (df_radiomics['Energy [kj]'] <= 100)]
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    df_radiomics_angyodinamics = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']

    ratio_0 = df_amica.safety_margin_distribution_0
    ratio_5 = df_amica.safety_margin_distribution_5
    ratio_10 = df_angyodinamics.safety_margin_distribution_10
    ratios = [ratio_0, ratio_5, ratio_10]
    interpolation_fct(df_amica, df_radiomics_amica, title='Amica', lin_regr=False, ratios=ratios)
    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, title='Angyodinamics (Solero)', lin_regr=False,
                      ratios=ratios)
    interpolation_fct(df_covidien, df_radiomics_covidien, title='Covidien', lin_regr=False, ratios=ratios)


    fig, ax = plt.subplots()
    xs = np.random.rand(3)
    ys = np.random.rand(3)
    draw_pie_markers(xs=xs,
                     ys=ys,
                     ratios=[.3, .2, .5],
                     sizes=[300, 300, 300],
                     colors=['green', 'red', 'orange'])
# draw_pie_markers(xs=np.random.rand(2),
#               ys=np.random.rand(2),
#               ratios=[.33, .66],
#               sizes=[100, 120],
#               colors=['blue', 'yellow'])
# draw_pie_markers(xs=np.random.rand(2),
#               ys=np.random.rand(2),
#               ratios=[.33, .25],
#               sizes=[50, 75],
#               colors=['maroon', 'brown'])
# plt.show()
