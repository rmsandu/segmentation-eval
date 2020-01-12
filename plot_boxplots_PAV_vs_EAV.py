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


def plot_boxplots_volumes(ablation_vol_brochure, ablation_vol_measured, flag_subcapsular=None):
    """
    """
    # drop the nans 
    effective_ablation_vol = ablation_vol_measured[~np.isnan(ablation_vol_measured)]
    predicted_ablation_vol = ablation_vol_brochure[~np.isnan(ablation_vol_brochure)]

    stat, p_brochure = shapiro(predicted_ablation_vol)
    # interpret
    alpha_brochure = 0.05
    if p_brochure > alpha_brochure:
        msg = 'Sample Ablation Volume Brochure looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample Ablation Volume Brochure does not look Gaussian (reject H0)'
    print(msg)

    stat, p_voxel = shapiro(effective_ablation_vol)
    # interpret
    alpha_voxel = 0.05
    if p_voxel > alpha_voxel:
        msg = 'Sample Ablation Volume looks Gaussian (fail to reject H0)'
    else:
        msg = 'Sample Ablation Volume does not look Gaussian (reject H0)'
    print(msg)

    if p_voxel < alpha_voxel and p_brochure < alpha_brochure:
        t, p = stats.mannwhitneyu(effective_ablation_vol, predicted_ablation_vol)
        print('mann withney u test applied for samples coming from a non Gaussian distribution:')
        print("t = " + str(t))
        print("p = " + str(p))
    else:
        t, p = stats.ttest_ind(effective_ablation_vol, predicted_ablation_vol)
        print('ttest applied for samples coming from a Gaussian distribution:')
        print("t = " + str(t))
        print("p = " + str(p))

    fig, ax = plt.subplots(figsize=(12, 10))
    bplot = plt.boxplot(x=[predicted_ablation_vol, effective_ablation_vol],
                        notch=True,
                        patch_artist=True,
                        widths=0.4
                        )
    # ax.set_xlabel('')
    for element in ['medians', 'fliers', 'whiskers', 'caps']:
        plt.setp(bplot[element], color='black', linewidth=2.5)
    boxes = bplot['boxes']
    plt.setp(boxes[1], color='seagreen')
    plt.setp(boxes[0], color='sandybrown')

    if flag_subcapsular is False:
        xticklabels = ['PAV', 'EAV (Deep Tumors)']
    elif flag_subcapsular is True:
        xticklabels = ['PAV', 'EAV (Subcapsular Tumors)']
    else:
        xticklabels = ['PAV', 'EAV']
    xtickNames = plt.setp(ax, xticklabels=xticklabels)
    plt.setp(xtickNames, fontsize=10, color='black')
    plt.ylim([-2, 100])
    plt.ylabel('Ablation Volume (mL)', fontsize=24, color='k')
    plt.tick_params(labelsize=24, color='black')
    ax.tick_params(colors='black', labelsize=24, color='k')
    ax.set_ylim([-2, 100])
    # plt.title('Comparison of Ablation Volumes [ml] from MAVERRIC Dataset', fontsize=16)
    figpathHist = os.path.join("figures", "boxplot volumes EAV vs PAV Solero. Subcapsular - " + str(flag_subcapsular))
    gh.save(figpathHist, ext=['png'], close=True, tight=True)
