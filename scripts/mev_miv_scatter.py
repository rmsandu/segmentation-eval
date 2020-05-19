# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

import utils.graphing as gh


def plot_mev_miv(df_radiomics):
    """
    Plot a 3-subplot of pav vs eav, subcapsular and chemo
    :param df_radiomics:
    :return:
    """
    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['MWA Systems'] = df_radiomics['Device_name']
    df['MIV'] = df_radiomics['Inner Ellipsoid Volume']
    df['MEV'] = df_radiomics['Outer Ellipsoid Volume']
    df['MEV-MIV'] = df['MEV'] - df['MIV']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']

    # drop outer volumes larger than 150 because they are probably erroneous
    df = df[df['MEV'] < 150]
    # drop the rows where MIV > MEV
    # since the minimum inscribed ellipsoid (MIV) should always be smaller than the maximum enclosing ellipsoid (MEV)
    df = df[df['MEV-MIV'] >= 0]

    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 2, figsize=(20, 20))
    # %% histogram MEV-MIV
    sns.distplot(df['MEV-MIV'], color=sns.xkcd_rgb["reddish"], hist_kws={"ec": 'black'},
                 axlabel='Ablation Surface Irregularity Distribution (MEV-MIV)', ax=axes[0])
    # %%   R (EAV:PAV) on y-axis and MEV-MIV on the x-axis
    slope, intercept, r_square, p_value, std_err = stats.linregress(df['R(EAV:PAV)'], df['MEV-MIV'])
    p = sns.regplot(y="R(EAV:PAV)", x="MEV-MIV", data=df, scatter_kws={"s": 100, "alpha": 0.5},
                    color=sns.xkcd_rgb["reddish"],
                    line_kws={'label': r'$R^2:{0:.2f}$'.format(r_square)}, ax=axes[1])
    axes[1].legend(loc='best')
    for ax in axes:
        # ax.set(adjustable='box', aspect='equal')
        ax.set(aspect='auto', adjustable='box')
        # ax.tick_params(axis='both', which='major', labelsize=8)
    # actually save the picture to the disk
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'Ratio_EAV-PAV_MEV-MIV_difference_' + timestr)
    gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)

    # %% linear regression graph between MEV and MIV
    slope, intercept, r_square, p_value, std_err = stats.linregress(df['MEV'], df['MIV'])
    p = sns.regplot(x="MEV", y="MIV", data=df, scatter_kws={"s": 100, "alpha": 0.5},
                    color=sns.xkcd_rgb["reddish"],
                    line_kws={'label': r'$R^2:{0:.2f}$'.format(r_square)})
    plt.legend(loc='best')
    plt.xlim([0, 150])
    plt.ylim([0, 150])
    plt.xlabel('Minimum Enclosing Ellipsoid Volume - MEV (mL)')
    plt.ylabel('Maximum Inscribed Ellipsoid Volume - MIV (mL)')
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'All_3MWA_MEV_MIV_' + timestr)
    gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)

    # %% Graph with PAV -x and EAV- y and marker size as the difference between MEV-MIV

    sns.relplot(x="PAV", y="EAV", alpha=0.5, size='MEV-MIV',
                color=sns.xkcd_rgb["reddish"], sizes=(100, 500),
                data=df)
    plt.xlim([0, 80])
    plt.ylim([0, 80])
    plt.xlabel('Predicted Ablation Volume - PAV  (mL)')
    plt.ylabel('Effective Ablation Volume - EAV  (mL)')

    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'All_3MWA_MEV_MIV_difference_size' + timestr)
    gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)

