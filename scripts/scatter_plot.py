# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model

import utils.graphing as gh

sns.set(style="ticks")
# plt.style.use('ggplot')


def scatter_plot(df1,  **kwargs):
    """
    df, x_data, y_data, title, x_label=False, y_label='', lin_reg=''
    :param df:
    :param x_data:
    :param x_data:
    :param title:
    :param x_label:
    :param x_label:
    :param lin_reg:
    :return:
    """
    fig, ax = plt.subplots()
    fontsize = 24
    if kwargs.get('x_data') is None:
        print('No X input data to plot')
        return
    if kwargs.get('y_data') is None:
        print('No Y input data to plot')
        return
    df_to_plot = df1.copy()
    # drop NaNs from both  x and y
    # df_to_plot.dropna(inplace=True)
    df_to_plot.dropna(subset=[kwargs["y_data"], kwargs["x_data"]], inplace=True)
    if kwargs.get('colormap') is not None:
        X_scatter = df_to_plot[kwargs['x_data']]
        Y_scatter = df_to_plot[kwargs['y_data']]
        t = df_to_plot[kwargs['colormap']]
        plt.scatter(X_scatter, Y_scatter, c=t, cmap='viridis', s=20)
        cbar = plt.colorbar()
        cbar.ax.set_title(kwargs['colormap'], fontsize=8)
    else:
        df_to_plot.plot.scatter(x=kwargs["x_data"], y=kwargs["y_data"], s=100, alpha=0.7, color='purple', marker='*')
    if kwargs.get('size') is not None:
        size = np.asarray(50 * (df_to_plot[kwargs['size']] + 1)).reshape(len(df_to_plot), 1)
        size_mask = ~np.isnan(size)
        size = size[size_mask]
        # color = 'steelblue'
        fig, ax = plt.subplots()
        x = df_to_plot[kwargs['x_data']]
        y = df_to_plot[kwargs['y_data']]
        sc = ax.scatter(x, y, s=size, alpha=0.6,  marker='*', color='purple')
        legend_1 = ax.legend(*sc.legend_elements("sizes", num=6, func=lambda x: x / 50 - 1, color='steelblue'),
                             title=kwargs['size'], labelspacing=1.5, borderpad=1.5, handletextpad=3.5,
                             fontsize=fontsize, loc='upper right')
        legend_1.get_title().set_fontsize(str(fontsize))
        ax.add_artist(legend_1)

    if kwargs.get('lin_reg') is not None:
        X = np.array(df_to_plot[kwargs['x_data']])
        Y = np.array(df_to_plot[kwargs['y_data']])
        regr = linear_model.LinearRegression()
        X = X.reshape(len(X), 1)
        Y = Y.reshape(len(Y), 1)
        regr.fit(X, Y)
        SS_tot = np.sum((Y-np.mean(Y))**2)
        residuals = Y - regr.predict(X)
        SS_res = np.sum(residuals**2)
        r_squared = 1-(SS_res/SS_tot)
        correlation_coef = np.corrcoef(X[:, 0], Y[:, 0])[0, 1]
        label_r2 = r'$R^2:{0:.2f}$'.format(r_squared)
        label_r = r'$r: {0:.2f}$'.format(correlation_coef)
        plt.plot(X, regr.predict(X), color='orange', linewidth=1.5, label='Linear Regression')
        plt.plot([], [], ' ', label='n = ' + str(len(kwargs['x_data'])))
        plt.plot([], [], ' ', label=label_r)
        plt.plot([], [], ' ', label=label_r2)
        if kwargs.get('legend_title'):
            plt.legend(fontsize=fontsize, loc='best', title=kwargs['legend_title'], title_fontsize=fontsize)
        else:
            plt.legend(fontsize=fontsize, loc='upper left')

    if kwargs.get('x_lim') is not None and kwargs.get('y_lim') is not None:
        plt.xlim([0, kwargs['x_lim']])
        plt.ylim([0, kwargs['y_lim']])
    title = kwargs['title']
    # plt.tick_params(labelsize=fontsize, color='black')
    # ax.tick_params(axis='y', labelsize=fontsize, color='k')
    # ax.tick_params(axis='x', labelsize=fontsize, color='k')
    # ax.xaxis.label.set_color('black')
    # ax.yaxis.label.set_color('black')
    # matplotlib.rc('axes', labelcolor='black')
    figpathHist = os.path.join("figures", title)

    if kwargs.get('x_label') is not None and kwargs.get('y_label') is None:
        plt.xlabel(kwargs['x_label'], fontsize=fontsize, color='k')
        plt.ylabel(kwargs['y_data'], fontsize=fontsize, color='k')
    elif kwargs.get('y_label') is not None and kwargs.get('x_label') is None:
        plt.ylabel(kwargs['y_label'], fontsize=fontsize, color='k')
        plt.xlabel(kwargs['x_data'], fontsize=fontsize, color='k')
    elif kwargs.get('x_label') is None and kwargs.get('y_label') is None:
        plt.xlabel(kwargs['x_data'], fontsize=fontsize, color='k')
        plt.ylabel(kwargs['y_data'], fontsize=fontsize, color='k')
    gh.save(figpathHist, width=12, height=12, ext=["png"], tight=True, close=True, dpi=600)
    plt.close('all')


def scatter_plot_groups(df1_no_outliers):

    groups = df1_no_outliers.groupby('Proximity_to_vessels')
    fig, ax = plt.subplots()
    lesion_per_device = []
    device_name_grp = []
    for name, group in groups:
        ax.plot(group["Energy [kj]"], group["Ablation Volume [ml]"], marker='o', linestyle='', ms=14, label=name)
        lesion_per_device.append(len(group))
        device_name_grp.append(name)
    L = ax.legend()
    L_labels = L.get_texts()

    for idx, L in enumerate(L_labels):
        L.set_text(device_name_grp[idx] + ' N=' + str(lesion_per_device[idx]))

    plt.xlabel('Energy [kJ]', fontsize=20, color='black')
    plt.ylabel('Ablation Volume [ml]', fontsize=20, color='black')
    plt.title("Ablation Volume vs Energy Grouped by Proximity to Vessels", fontsize=20, color='black')
    plt.tick_params(labelsize=20, color='black')
    plt.legend(title_fontsize=20)
    ax.tick_params(colors='black', labelsize=20)

    figpathHist = os.path.join("figures", "Ablation Volume vs Energy Grouped by Proximity to Vessels.")
    gh.save(figpathHist, width=18, height=16, ext=['png'], close=True)
    # %% group by device name
    groups = df1_no_outliers.groupby('Device_name')
    fig, ax = plt.subplots()
    lesion_per_device = []
    device_name_grp = []
    for name, group in groups:
        ax.plot(group["Energy [kj]"], group["Ablation Volume [ml]"], marker='o', linestyle='', ms=14, label=name)
        lesion_per_device.append(len(group))
        device_name_grp.append(name)
    L = ax.legend()
    L_labels = L.get_texts()

    for idx, L in enumerate(L_labels):
        L.set_text(str(device_name_grp[idx]) + ' N=' + str(lesion_per_device[idx]))

    plt.title('Ablation Volume [ml] vs Energy [kJ] per MWA Device Type.', fontsize=20)
    plt.xlabel('Energy [kJ]', fontsize=20, color='black')
    plt.ylabel('Ablation Volume [ml]', fontsize=20, color='black')
    plt.tick_params(labelsize=20, color='black')
    plt.legend(title_fontsize=20)
    ax.tick_params(colors='black', labelsize=20)
    figpathHist = os.path.join("figures", "Ablation Volume vs  Energy per MWA Device Category.")
    gh.save(figpathHist, width=18, height=16, ext=['png'], close=True)

    # chemotherapy
    groups = df1_no_outliers.groupby('chemo_before_ablation')
    fig, ax = plt.subplots()
    lesion_per_device = []
    device_name_grp = []
    for name, group in groups:
        ax.plot(group["Energy [kj]"], group["Ablation Volume [ml]"], marker='o', linestyle='', ms=14, label=name)
        lesion_per_device.append(len(group))
        device_name_grp.append(name)
    L = ax.legend()
    L_labels = L.get_texts()

    for idx, L in enumerate(L_labels):
        L.set_text(str(device_name_grp[idx]) + ' N=' + str(lesion_per_device[idx]))

    plt.title('Ablation Volume [ml] vs Energy [kJ] grouped by Chemotherapy Treatment before Ablation.', fontsize=20)
    plt.xlabel('Energy [kJ]', fontsize=20, color='black')
    plt.ylabel('Ablation Volume [ml]', fontsize=20, color='black')
    plt.tick_params(labelsize=20, color='black')
    plt.legend(title_fontsize=20)
    ax.tick_params(colors='black', labelsize=20)
    figpathHist = os.path.join("figures", "Ablation Volume vs  Energy grouped by chemotherapy.")
    gh.save(figpathHist, width=18, height=16, ext=['png'], close=True)




