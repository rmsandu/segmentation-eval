# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:15:51 2017

@author: Raluca Sandu
"""
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import graphing as gh
import matplotlib.pyplot as plt
from collections import OrderedDict
# plt.style.use('ggplot')
#%%


def plotHistDistances(pat_name, pat_idx, rootdir, distanceMap, num_voxels, title):

    # PLOT THE HISTOGRAM FOR THE MAUERER EUCLIDIAN DISTANCES

    figName_hist = str(pat_name) + str(pat_idx)+ '_histogramDistances_' + title

    min_val = int(np.floor(min(distanceMap)))
    max_val = int(np.ceil(max(distanceMap)))
    
    fig, ax = plt.subplots(figsize=(24, 20))
    # col_height, bins, patches = ax.hist(distanceMap, ec='darkgrey')
    # TODO: fix column height percentage. now it's above 100% sometimes. because of calculation or display?
    col_height, bins, patches = ax.hist(distanceMap, ec='darkgrey', bins=range(min_val-1, max_val+1))
    
    voxels_nonablated = []
    voxels_insuffablated = []
    voxels_ablated = []
    
    for b, p, col_val in zip(bins, patches, col_height):
        if b < 0:
            voxels_nonablated.append(col_val)
        elif b in range(0, 5):
            voxels_insuffablated.append(col_val)
        elif b >= 5:
            voxels_ablated.append(col_val)
            
#%%
    '''calculate the total percentage of surface for ablated, non-ablated, insufficently ablated'''
    voxels_nonablated = np.asarray(voxels_nonablated)
    voxels_insuffablated = np.asarray(voxels_insuffablated)
    voxels_ablated = np.asarray(voxels_ablated)
    
    sum_perc_nonablated = ((voxels_nonablated / num_voxels) * 100).sum()
    sum_perc_insuffablated = ((voxels_insuffablated/num_voxels) * 100).sum()
    sum_perc_ablated = ((voxels_ablated/num_voxels) * 100).sum()

#%%
    '''iterate through the bins to change the colors of the patches bases on the range [mm]'''
    for b, p, col_val in zip(bins, patches, col_height):
        if b < 0:
            plt.setp(p, 'facecolor', 'darkred', label='Non-ablated Surface: ' + " %.2f" % sum_perc_nonablated + '%')
        elif b in range(0, 5):
            plt.setp(p, 'facecolor', 'orange', label='Insufficient Ablation Margin: ' + "%.2f" % sum_perc_insuffablated + '%')
        elif b >= 5:
            plt.setp(p, 'facecolor', 'darkgreen', label='Sufficient Ablation Margin: ' + " %.2f" % sum_perc_ablated + '%')

#%%                   
    '''edit the axes limits and labels'''
    plt.xlabel('[mm]', fontsize=38, color='black')
    plt.tick_params(labelsize=36, color='black')
    ax.tick_params(colors='black', labelsize=36)
    plt.grid(True)
    # ax.set_xlim([-10, 11])

    # edit the y-ticks: change to percentage of surface
    yticks, locs = plt.yticks()
    percent = (yticks/num_voxels) * 100
    percentage_surface_rounded = np.round(percent)
    yticks_percent = [str(x) + '%' for x in percentage_surface_rounded]
    new_yticks = (percentage_surface_rounded * yticks)/percent
    new_yticks[0] = 0
    plt.yticks(new_yticks, yticks_percent)
#    plt.yticks(yticks,yticks_percent)
    plt.ylabel('Percetange of surface voxels', fontsize=38, color='black')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=38, loc='best')
    
    # plt.title(title + '. Case ' + str(pat_idx), fontsize=36)
    figpathHist = os.path.join(rootdir, figName_hist + '.png')
    gh.save(figpathHist, width=24, height=20)
    figpathHist = os.path.join(rootdir, figName_hist + '.svg')
    gh.save(figpathHist, width=24, height=20)
