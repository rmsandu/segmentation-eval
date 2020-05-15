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
    df['MIV'] = df_radiomics['Outer Ellipsoid Volume'] / 3
    df['MEV'] = df_radiomics['Outer Ellipsoid Volume']
    df['MEV-MIV'] = df['MEV'] - df['MIV']

    df = df[df['MEV'] < 150]

    slope, intercept, r_square, p_value, std_err = stats.linregress(df['MEV'], df['MIV'])
    p = sns.regplot(x="MEV", y="MIV", data=df, scatter_kws={"s": 100, "alpha": 0.5},
                    color=sns.xkcd_rgb["reddish"],
                    line_kws={'label': r'$R^2:{0:.2f}$'.format(r_square)})
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'All_3MWA_MEV_MIV_' + timestr)
    gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)

    sns.relplot(x="PAV", y="EAV", alpha=0.5, size='MEV-MIV',
                color=sns.xkcd_rgb["reddish"], sizes=(100, 500),
                data=df)
    plt.xlim([0, 80])
    plt.ylim([0, 80])

    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'All_3MWA_MEV_MIV_difference_' + timestr)
    gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)
