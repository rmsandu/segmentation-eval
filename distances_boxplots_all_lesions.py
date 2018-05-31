# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:15:03 2017

@author: Raluca Sandu
"""
import os
import numpy as np
import pandas as pd
import graphing as gh
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict


# plt.style.use('classic')
# plt.style.use('seaborn')
# %%
def plotBoxplots(data, rootdir):
    fig, axes = plt.subplots()

    # Horizontal box plot
    bplot = plt.boxplot(data,
                        vert=True,  # vertical box aligmnent
                        patch_artist=False,
                        showmeans=True)  # fill with color

    for element in ['medians', 'boxes', 'fliers', 'whiskers', 'caps']:
        plt.setp(bplot[element], color='black', linewidth=2.5)

    plt.setp(bplot['whiskers'], linestyle='--')
    plt.setp(bplot['fliers'], markersize=6)
    plt.setp(bplot['means'], marker='D', markeredgecolor='black',
             markerfacecolor='blue', label='Average')

    xlim = np.array(plt.gca().get_xlim())
    ylim = np.array(plt.gca().get_ylim())
    plt.fill_between(xlim, y1=([ylim[0], ylim[0]]), y2=([0, 0]),
                     color="#EC7063", zorder=0)
    plt.fill_between(xlim, y1=([0, 0]), y2=([5, 5]),
                     color="#FAD7A0", zorder=0)
    plt.fill_between(xlim, y1=([5, 5]), y2=(ylim[1], ylim[1]),
                     color="#ABEBC6", zorder=0)
    plt.margins(0)
    axes.set_ylim([-15, 15])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=36)

    plt.title('Surface-to-Surface Euclidean Distances for Tumor and Ablation', fontsize=36)
    plt.xlabel('Lesion', fontsize=36, color='black')
    plt.setp(axes.get_xticklabels(), color="black", fontsize=30)
    plt.setp(axes.get_yticklabels(), color="black", fontsize=30)
    plt.ylabel('[mm]', fontsize=36, color='black')
    figpath_png = os.path.join(rootdir, 'boxplots_forDistanceMaps.png')
    plt.show()
    gh.save(figpath_png)

