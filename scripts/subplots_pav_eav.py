# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import utils.graphing as gh


def plot_subplots(df_radiomics):
    """
    Plot a 3-subplot of pav vs eav, subcapsular and chemo
    :param df_radiomics:
    :return: plot fo png file
    """
    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 3, figsize=(20, 20))

    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['MWA Systems'] = df_radiomics['Device_name']
    df['Proximity_to_surface'] = df_radiomics['Proximity_to_surface']
    df['Chemotherapy'] = df_radiomics['chemo_before_ablation']
    df['Chemo_yes'] = df['EAV']
    df['Chemo_no'] = df['EAV']
    df['Subcapsular'] = df['EAV']
    df['Non-Subcapsular'] = df['EAV']
    df.loc[
        df.Proximity_to_surface == False, 'Subcapsular'] = np.nan  # only keep those with value true, ie subcapsular
    df.loc[df.Proximity_to_surface == True, 'Non-Subcapsular'] = np.nan
    df.loc[
        df.Chemotherapy == 'No', 'Chemo_yes'] = np.nan
    df.loc[df.Chemotherapy == 'Yes', 'Chemo_no'] = np.nan  # chemo no

    print('Nr Samples used:', str(len(df)))

    # 1st plot PAV vs EAV with lin regr
    slope, intercept, r_square, p_value, std_err = stats.linregress(df['EAV'], df['PAV'])
    sns.regplot(x="PAV", y="EAV", data=df, scatter_kws={"s": 11, "alpha": 0.6},
                color=sns.xkcd_rgb["violet"],
                line_kws={'label': r'$R^2:{0:.2f}$'.format(r_square)}, ax=axes[0])
    axes[0].legend(fontsize=8, loc='best')
    axes[0].set_ylabel('EAV (ml)', fontsize=8)
    axes[0].set_xlabel('PAV (ml)', fontsize=8)
    # Subcapsular 2nd plot
    subcapsular_false = df[df['Proximity_to_surface'] == False]
    subcapsular_true = df[df['Proximity_to_surface'] == True]
    slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['PAV'],
                                                               subcapsular_false['EAV'])
    slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['PAV'],
                                                               subcapsular_true['EAV'])
    sns.regplot(y="Non-Subcapsular", x="PAV", data=df, scatter_kws={"s": 11, "alpha": 0.6},
                line_kws={'label': r'Non-subcapsular: $R^2={0:.2f}$'.format(r_1)},
                ax=axes[1])
    sns.regplot(y="Subcapsular", x="PAV", data=df, scatter_kws={"s": 11, "alpha": 0.6},
                color=sns.xkcd_rgb["orange"], line_kws={'label': r'Subcapsular: $R^2={0:.2f}$'.format(r_2)},
                ax=axes[1])
    axes[1].legend(fontsize=8, loc='best')
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    axes[1].set_xlabel('PAV (ml)', fontsize=8)
    axes[1].set_title('Predicted (PAV) vs Effective Ablation Volume (EAV) for 3 MWA Devices', fontsize=10)

    # Chemo 3rd plot
    chemo_false = df[df['Chemotherapy'] == 'No']
    chemo_true = df[df['Chemotherapy'] == 'Yes']
    x1 = chemo_false['PAV']
    y1 = chemo_false['EAV']
    x2 = chemo_true['PAV']
    y2 = chemo_true['EAV']
    slope, intercept, r_1, p_value, std_err = stats.linregress(x1, y1)
    slope, intercept, r_2, p_value, std_err = stats.linregress(x2, y2)
    sns.regplot(y='Chemo_yes', x="PAV", data=df, scatter_kws={"s": 11, "alpha": 0.6}, color=sns.xkcd_rgb["teal green"],
                ax=axes[2], line_kws={'label': r'Chemo:yes $R^2={0:.2f}$'.format(r_2)})
    sns.regplot(y='Chemo_no', x="PAV", data=df, scatter_kws={"s": 11, "alpha": 0.6}, color=sns.xkcd_rgb["slate grey"],
                ax=axes[2], line_kws={'label': r'Chemo:no $R^2={0:.2f}$'.format(r_1)})
    axes[2].legend(fontsize=8, loc='best')
    axes[2].set_yticklabels([])
    axes[2].set_ylabel('')
    axes[2].set_xlabel('PAV (ml)', fontsize=8)

    # add major title to subplot
    # f.suptitle('Predicted (PAV) vs Effective Ablation Volume (EAV) for 3 MWA Devices', fontsize=10)

    # set the axes limits and new ticks
    axes[2].set_xlim([0, 81])
    axes[1].set_xlim([0, 81])
    axes[0].set_xlim([0, 81])

    axes[2].set_ylim([0, 81])
    axes[1].set_ylim([0, 81])
    axes[0].set_ylim([0, 81])

    axes[0].xaxis.set_ticks(np.arange(0, 81, 20))
    axes[1].xaxis.set_ticks(np.arange(0, 81, 20))
    axes[2].xaxis.set_ticks(np.arange(0, 81, 20))

    plt.subplots_adjust(wspace=0.1)

    # set the fontsize of the ticks of the subplots
    for ax in axes:
        ax.set(adjustable='box', aspect='equal')
        # ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=8)

    # save the figure
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'All_3MWA_SUbcapsular_Chemo_' + timestr)
    gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)
