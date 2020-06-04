# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_mev_miv(df_radiomics):
    """
    Plot a 3-subplot of pav vs eav, subcapsular and chemo
    :param df_radiomics:
    :return:
    """
    font = {'family': 'DejaVu Sans',
            'size': 18}
    matplotlib.rc('font', **font)

    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['MWA Systems'] = df_radiomics['Device_name']
    df['MIV'] = df_radiomics['Inner Ellipsoid Volume']
    df['MEV'] = df_radiomics['Outer Ellipsoid Volume']
    df['MEV-MIV'] = df['MEV'] - df['MIV']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']


    fig, ax = plt.subplots(figsize=(12, 12))
    sns.distplot(df['Energy (kJ)'],  hist_kws={"ec": 'black', "align": "mid"},
                 axlabel='Energy', ax=ax)
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'Energy_distribution_' + timestr + '.png')
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    # drop outer volumes larger than 150 because they are probably erroneous
    df = df[df['MEV'] < 150]
    # drop the rows where MIV > MEV
    # since the minimum inscribed ellipsoid (MIV) should always be smaller than the maximum enclosing ellipsoid (MEV)
    df = df[df['MEV-MIV'] >= 0]
    min_val = int(min(df['MEV-MIV']))
    max_val = int(max(df['MEV-MIV']))
    print('Min Val Mev-Miv:', min_val)
    print('Max Val Mev-Miv:', max_val)
    print('nr of samples for mev-miv:', len(df))

    # %% histogram MEV-MIV
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.distplot(df['MEV-MIV'], color=sns.xkcd_rgb["reddish"], hist_kws={"ec": 'black',  "align": "mid"},
                 axlabel='Distribution of Ablation Volume Irregularity (MEV-MIV) (mL)', ax=ax)

    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'MEV-MIV_distribution_' + timestr + '.png')
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()

    fig1, ax1 = plt.subplots(figsize=(12, 12))
    sns.distplot(df['MEV'], color=sns.xkcd_rgb["reddish"], hist_kws={"ec": 'black'},
                 axlabel='MEV', ax=ax1)
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'MEV_distribution_' + timestr + '.png')
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()

    fig1, ax2 = plt.subplots(figsize=(12, 12))
    sns.distplot(df['MIV'], color=sns.xkcd_rgb["reddish"], hist_kws={"ec": 'black'},
                 axlabel='MIV', ax=ax2)
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'MIV_distribution_' + timestr + '.png')
    plt.savefig(figpath, dpi=300)
    plt.close()

    fig1, ax3 = plt.subplots(figsize=(12, 12))
    sns.distplot(df['EAV'], color=sns.xkcd_rgb["reddish"], hist_kws={"ec": 'black'},
                 axlabel='EAV', ax=ax3)
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'EAV_distribution_' + timestr + '.png')
    plt.savefig(figpath, dpi=300)
    plt.close()

    fig1, ax4 = plt.subplots(figsize=(12, 12))
    sns.distplot(df['PAV'], color=sns.xkcd_rgb["reddish"], hist_kws={"ec": 'black'},
                 axlabel='PAV', ax=ax4)
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'PAV_distribution_' + timestr + '.png')
    plt.savefig(figpath, dpi=300)
    plt.close()

    # %%   R (EAV:PAV) on y-axis and MEV-MIV on the x-axis
    fig1, ax5 = plt.subplots(figsize=(12, 12))
    slope, intercept, r_square, p_value, std_err = stats.linregress(df['R(EAV:PAV)'], df['MEV-MIV'])
    print('p-val mev miv energy:', p_value )
    print()
    p = sns.regplot(y="R(EAV:PAV)", x="MEV-MIV", data=df, scatter_kws={"s": 100, "alpha": 0.5},
                    color=sns.xkcd_rgb["reddish"],
                    line_kws={'label': r'$r = {0:.2f}$'.format(r_square)}, ax=ax5)
    plt.xlabel('MEV-MIV (mL)')
    plt.legend()

    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", 'Ratio_EAV-PAV_MEV-MIV_difference_' + timestr)
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()
    # %% linear regression graph between MEV and MIV
    # slope, intercept, r_square, p_value, std_err = stats.linregress(df['MEV'], df['MIV'])
    # p = sns.regplot(x="MEV", y="MIV", data=df, scatter_kws={"s": 100, "alpha": 0.5},
    #                 color=sns.xkcd_rgb["reddish"],
    #                 line_kws={'label': r'$R^2:{0:.2f}$'.format(r_square)})
    # plt.legend(loc='best')
    # plt.xlim([0, 150])
    # plt.ylim([0, 150])
    # plt.xlabel('Minimum Enclosing Ellipsoid Volume - MEV (mL)')
    # plt.ylabel('Maximum Inscribed Ellipsoid Volume - MIV (mL)')
    # timestr = time.strftime("%H%M%S-%Y%m%d")
    # figpath = os.path.join("figures", 'All_3MWA_MEV_MIV_' + timestr)
    # gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)

    # %% Graph with PAV -x and EAV- y and marker size as the difference between MEV-MIV

    # sns.relplot(x="PAV", y="EAV", alpha=0.5, size='MEV-MIV',
    #             color=sns.xkcd_rgb["reddish"], sizes=(100, 500),
    #             data=df)
    # plt.xlim([0, 80])
    # plt.ylim([0, 80])
    # plt.xlabel('Predicted Ablation Volume - PAV  (mL)')
    # plt.ylabel('Effective Ablation Volume - EAV  (mL)')
    #
    # timestr = time.strftime("%H%M%S-%Y%m%d")
    # figpath = os.path.join("figures", 'All_3MWA_MEV_MIV_difference_size' + timestr)
    # gh.save(figpath, ext=["png"], width=12, height=12, close=True, tight=True, dpi=300)
