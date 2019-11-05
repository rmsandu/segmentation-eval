# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro

import utils.graphing as gh

sns.set(style="ticks")
plt.style.use('ggplot')


def plot_boxplots_chemo(df):
    """
    boxplot chemotherapy
    :param df: dataframe
    :return: ttest values
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    df_chemo = df.copy()
    df_chemo['Ablation Volume [ml] / Energy [kJ]'] = df_chemo['Ablation Volume [ml]'] / df_chemo['Energy [kj]']
    df_chemo.dropna(subset=['Ablation Volume [ml] / Energy [kJ]'], inplace=True)
    df_chemo.dropna(subset=['chemo_before_ablation'], inplace=True)
    df_chemo['chemo_before_ablation'].replace('No', False, inplace=True)
    df_chemo['chemo_before_ablation'].replace('Yes', True, inplace=True)

    df.dropna(subset=['Ablation Volume [ml]'], inplace=True)
    df.dropna(subset=['chemo_before_ablation'], inplace=True)
    df['chemo_before_ablation'].replace('No', False, inplace=True)
    df['chemo_before_ablation'].replace('Yes', True, inplace=True)
    # ttest
    no_chemo_df = df_chemo[df_chemo['chemo_before_ablation'] == False]
    no_chemo = no_chemo_df['Ablation Volume [ml]'].tolist()
    chemo_df = df_chemo[df_chemo['chemo_before_ablation'] == True]
    chemo = chemo_df['Ablation Volume [ml]'].tolist()

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.hist(no_chemo)
    plt.title('No Chemotherapy')
    plt.ylabel('Ablation Volume [ml]')
    figpathHist = os.path.join("figures", "histogram ablation volumes no chemo")
    gh.save(figpathHist, ext=['png'], close=True)
    fig1, ax = plt.subplots(figsize=(12, 10))
    plt.hist(chemo)
    plt.title('Chemotherapy')
    plt.ylabel('Ablation Volume [ml] ')
    figpathHist = os.path.join("figures", "histogram ablation volumes chemo")
    gh.save(figpathHist, ext=['png'], close=True)

    print('no of tumors with chemo:', str(len(chemo)))
    print('no of tumors with no chemo:', str(len(no_chemo)))
    #
    stat, p_chemo = shapiro(chemo)

    # interpret
    alpha_chemo = 0.05
    if p_chemo > alpha_chemo:
        msg = 'Sample Chemo looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample Chemo does not look Gaussian (reject H0)'
    print(msg)

    stat, p_no_chemo = shapiro(no_chemo)

    # interpret
    alpha_no_chemo = 0.05
    if p_no_chemo > alpha_no_chemo:
        msg = 'Sample No Chemo looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample No Chemo does not look Gaussian (reject H0)'
    print(msg)

    if p_no_chemo < alpha_no_chemo and p_chemo < alpha_chemo:
        t, p = stats.mannwhitneyu(chemo, no_chemo)
        print('mann withney u test applied for samples coming from a non Gaussian distribution:')
        print("t = " + str(t))
        print("p = " + str(p))
    else:
        t, p = stats.ttest_ind(chemo, no_chemo)
        print('ttest applied for samples coming from a Gaussian distribution:')
        print("t = " + str(t))
        print("p = " + str(p))

    fig, ax = plt.subplots(figsize=(12, 10))
    bp_dict = df.boxplot(column=['Ablation Volume [ml]'],
                         ax=ax,
                         notch=True,
                         by='chemo_before_ablation',
                         patch_artist=True,
                         return_type='both')
    ax.set_xlabel('')
    plt.show()
    for row_key, (ax, row) in bp_dict.iteritems():
        for i, box in enumerate(row['fliers']):
            box.set_marker('o')
        for i, box in enumerate(row['boxes']):
            if i == 0:
                box.set_facecolor('Purple')
                box.set_edgecolor('DarkMagenta')
            else:
                box.set_facecolor('LightPink')
                box.set_edgecolor('HotPink')
        for i, box in enumerate(row['medians']):
            box.set_color(color='Black')
            box.set_linewidth(2)
        for i, box in enumerate(row['whiskers']):
            box.set_color(color='Black')
            box.set_linewidth(2)
    xticklabels = ['No Chemotherapy before Ablation', 'Chemotherapy Administered before Ablation']
    xtickNames = plt.setp(ax, xticklabels=xticklabels)
    plt.setp(xtickNames, fontsize=10, color='black')
    plt.ylim([-2, 120])
    plt.ylabel('Ablation Volume [ml]', fontsize=12, color='k')
    plt.tick_params(labelsize=10, color='black')
    ax.tick_params(colors='black', labelsize=10, color='k')
    ax.set_ylim([-2, 120])
    plt.xlabel('')
    fig.suptitle('')
    plt.title('')
    # plt.title('Comparison of Ratio (Ablation Volumes [ml] : Energy [kJ]) from MAVERRIC Dataset by Chemotherapy', fontsize=12)
    plt.title('Comparison of Ablation Volumes [ml] from MAVERRIC Dataset by Chemotherapy',
              fontsize=12)
    figpathHist = os.path.join("figures", "boxplot ablation volumes by chemo before ablation")
    gh.save(figpathHist, ext=['png'], close=True)


def plot_boxplots_volumes(df):
    """
    BOXPLOTS ABLATION VOLUMES
    :param df:
    :return:
    """
    # ttest
    df_volumes = df.copy()
    df_volumes.dropna(subset=['Ablation Volume [ml]'], inplace=True)
    df_volumes.dropna(subset=['Ablation Volume [ml] (manufacturers)'], inplace=True)
    ablation_vol = df_volumes['Ablation Volume [ml]'].tolist()
    ablation_vol_brochure = df_volumes['Ablation Volume [ml] (manufacturers)'].tolist()

    stat, p_brochure = shapiro(ablation_vol_brochure)
    # interpret
    alpha_brochure = 0.05
    if p_brochure > alpha_brochure:
        msg = 'Sample Ablation Volume Brochure looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample Ablation Volume Brochure does not look Gaussian (reject H0)'
    print(msg)

    stat, p_voxel = shapiro(ablation_vol)
    # interpret
    alpha_voxel = 0.05
    if p_voxel > alpha_voxel:
        msg = 'Sample Ablation Volume looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample Ablation Volume does not look Gaussian (reject H0)'
    print(msg)

    if p_voxel < alpha_voxel and p_brochure < alpha_brochure:
        t, p = stats.mannwhitneyu(ablation_vol, ablation_vol_brochure)
        print('mann withney u test applied for samples coming from a non Gaussian distribution:')
        print("t = " + str(t))
        print("p = " + str(p))
    else:
        t, p = stats.ttest_ind(ablation_vol, ablation_vol_brochure)
        print('ttest applied for samples coming from a Gaussian distribution:')
        print("t = " + str(t))
        print("p = " + str(p))

    fig, ax = plt.subplots(figsize=(12, 10))
    bp_dict = df.boxplot(column=['Ablation Volume [ml]', 'Ablation Volume [ml] (parametrized_formula)',
                                 'Ablation Volume [ml] (manufacturers)'],
                         ax=ax,
                         notch=True,
                         patch_artist=True,
                         return_type='both'
                         )
    ax.set_xlabel('')
    row = bp_dict.lines
    # for idx,row in enumerate(lines):
    for i, box in enumerate(row['fliers']):
        box.set_marker('o')
        # box.set_edgecolor('RoyalBlue')
    for i, box in enumerate(row['boxes']):
        if i == 0:
            box.set_facecolor('Blue')
            box.set_edgecolor('MediumBlue')
        elif i == 1:
            box.set_facecolor('BlueViolet')
            box.set_edgecolor('BlueViolet')
        elif i == 2:
            box.set_facecolor('DeepSkyBlue')
            box.set_edgecolor('DodgerBlue')

    for i, box in enumerate(row['medians']):
        box.set_color(color='Black')
        box.set_linewidth(2)
    for i, box in enumerate(row['whiskers']):
        box.set_color(color='Black')
        box.set_linewidth(2)

    xticklabels = ['Ablation Volume [ml] (Voxel-Based)', 'Ablation Volume [ml] (Ellipsoid Formula)',
                   'Ablation Volume [ml] (Manufacturers Brochure)']
    xtickNames = plt.setp(ax, xticklabels=xticklabels)
    plt.setp(xtickNames, fontsize=10, color='black')
    plt.ylim([-2, 100])
    plt.ylabel('Ablation Volume [ml]', fontsize=14, color='k')
    plt.tick_params(labelsize=10, color='black')
    ax.tick_params(colors='black', labelsize=10, color='k')
    ax.set_ylim([-2, 100])
    plt.title('Comparison of Ablation Volumes [ml] from MAVERRIC Dataset', fontsize=16)
    figpathHist = os.path.join("figures", "boxplot volumes")
    gh.save(figpathHist, ext=['png'], close=True)


def plot_boxplots_subcapsular(df):
    """

    :param df:
    :return:
    """
    # plot energy
    df['Ablation Volume [ml] / Energy [kJ]'] = df['Ablation Volume [ml]'] / df['Energy [kj]']
    df.dropna(subset=['Ablation Volume [ml] / Energy [kJ]'], inplace=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    bp_dict = df.boxplot(column=['Ablation Volume [ml] / Energy [kJ]'], by='Proximity_to_surface',
                         ax=ax,
                         notch=True,
                         patch_artist=True,
                         return_type='both'
                         )

    for row_key, (ax, row) in bp_dict.iteritems():
        ax.set_xlabel('')
        for i, box in enumerate(row['fliers']):
            box.set_marker('o')
            # box.set_edgecolor('RoyalBlue')
        for i, box in enumerate(row['boxes']):
            box.set_facecolor('CornflowerBlue')
            box.set_edgecolor('DarkBlue')
        for i, box in enumerate(row['medians']):
            box.set_color(color='Black')
            box.set_linewidth(2)
        for i, box in enumerate(row['whiskers']):
            box.set_color(color='Black')

    xticklabels = ['Non-subcapsular Tumors', 'Subcapsular Tumors']
    xtickNames = plt.setp(ax, xticklabels=xticklabels)
    plt.setp(xtickNames, fontsize=10, color='black')
    # plt.ylim([-2, 100])
    plt.ylabel('Ablation Volume / Energy  [mL/kJ] ', fontsize=14, color='k')
    plt.tick_params(labelsize=10, color='black')
    ax.tick_params(colors='black', labelsize=10, color='k')
    # ax.set_ylim([-2, 100])
    title = 'Comparison of Ratio(Ablation Volume: Energy) [mL/kJ]'
    plt.title(title, fontsize=16)
    figpathHist = os.path.join("figures", "boxplot ablation volumes subcapsular non subcapsular")
    gh.save(figpathHist, ext=['png'], close=True)

    non_subcapsular = df[df['Proximity_to_surface'] == False]
    non_subcapsular['Ablation Volume [ml] / Energy [kJ]'] = non_subcapsular[
        'Ablation Volume [ml] / Energy [kJ]'].replace(np.inf, np.nan)
    non_subcapsular.dropna(subset=['Ablation Volume [ml] / Energy [kJ]'], inplace=True)
    non_subcapsular_volume = np.asarray(non_subcapsular['Ablation Volume [ml] / Energy [kJ]'].tolist())
    subcapsular = df[df['Proximity_to_surface'] == True]
    subcapsular['Ablation Volume [ml] / Energy [kJ]'] = subcapsular[
        'Ablation Volume [ml] / Energy [kJ]'].replace(np.inf, np.nan)
    subcapsular.dropna(subset=['Ablation Volume [ml] / Energy [kJ]'], inplace=True)
    subcapsular_volume = np.asarray(subcapsular['Ablation Volume [ml] / Energy [kJ]'].tolist())

    print('no of subcapsular tumors:', str(len(subcapsular_volume)))
    print('no of non-subcapsular tumors:', str(len(non_subcapsular_volume)))

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.hist(non_subcapsular_volume)
    title = 'Non Subcapsular Lesions'
    plt.title(title)
    plt.ylabel('Ablation Volume / Energy ')
    figpathHist = os.path.join("figures", "histogram" + title)
    gh.save(figpathHist, ext=['png'], close=True)
    fig1, ax = plt.subplots(figsize=(12, 10))
    plt.hist(subcapsular_volume)
    title = 'Subcapsular Lesions'
    plt.title(title)
    plt.ylabel('Ablation Volume [ml] ')
    figpathHist = os.path.join("figures", "histogram" + title)
    gh.save(figpathHist, ext=['png'], close=True)

    # shapiro test for normality
    stat, p_subcapsular = shapiro(subcapsular_volume)
    stat, p_non_subcapsular = shapiro(non_subcapsular_volume)
    alpha = 0.05
    if p_subcapsular > alpha:
        msg = 'Sample Subcapsular Ablation Vol looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample Subcapsular Ablation Vol does not look Gaussian (reject H0)'
    print(msg)

    if p_non_subcapsular > alpha:
        msg = 'Sample Non-Subcapsular Ablation Vol looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample Non-Subcapsular Ablation Vol does not look Gaussian (reject H0)'
    print(msg)

    # if p_subcapsular < alpha or p_non_subcapsular < alpha:
    t, p = stats.mannwhitneyu(non_subcapsular_volume, subcapsular_volume)
    print('ABLATION VOLUME / ENERGY')
    print('mann withney u test applied for samples coming from a non Gaussian distribution:')
    print("t = " + str(t))
    print("p = " + str(p))
    # elif p_subcapsular > alpha and p_non_subcapsular > alpha:
    t, p = stats.ttest_ind(non_subcapsular_volume, subcapsular_volume, equal_var=False)
    print('ABLATION VOLUME / ENERGY')
    print('ttest applied for samples coming from a Gaussian distribution:')
    print("t = " + str(t))
    print("p = " + str(p))
