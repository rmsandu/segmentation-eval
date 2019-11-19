# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:58:43 2017

@author: Raluca Sandu
"""

import pandas as pd
import os
import numpy as np
import utils.graphing as gh
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.style.use('ggplot')
#%%
def plotBinsSurfacePercentage(data, rootdir, flag_all_ranges=False):

    '''plot boxplots per range of mm versus percentage of surface covered - all patients'''

    fig, axes = plt.subplots(figsize=(12, 16))    
    # Horizontal box plot
    bplot = plt.boxplot(data,
                         vert=True,   # vertical box aligmnent
                         patch_artist=True,
                         showmeans=True)   # fill with color
                        
    # set the axes ranges and the names
    plt.setp(bplot['medians'], color='black',linewidth=1.5)    
    plt.setp(bplot['means'], marker='D', markeredgecolor='darkred',
                  markerfacecolor='darkred', label='Mean')
    
    if flag_all_ranges is True:
        
        labels_neg = ['(-'+ str(x-1)+':-'+str(x)+')' for x in range(21,0,-1)]
        labels_neg[20] = '(-1:0)'
        labels_pos = ['('+ str(x-1)+':'+str(x)+')' for x in range(1,22)]
        xticklabels = labels_neg+labels_pos
        xtickNames = plt.setp(axes, xticklabels=xticklabels)
        plt.setp(xtickNames, rotation=45, fontsize=6)
        
    else:
        
        xticklabels = [r"$(\infty< x < 0$)", r"$(0 \leqslant x  \leqslant  5$)", r"$( x > 5)"]
        xtickNames = plt.setp(axes, xticklabels=xticklabels)
        plt.setp(xtickNames, fontsize=14, color='black')

    axes.tick_params(colors='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=14)
        
    plt.xlabel('Range of Distances [mm]', fontsize=14, color='black')
    plt.ylabel('Tumor Surface covered by Ablation [%]', fontsize=14, color='black')
    plt.title('Boxplots for Percentage of Tumor Surface covered by Ablation. 10 Cases (pooled)', fontsize=14)
    
    # save figure
    if flag_all_ranges is True:
        figName_hist = 'Boxplots_TheHistogramDistances_AllRanges.png' 
        
    else:
        figName_hist = 'Boxplots_TheHistogramDistances.png' 
    
    figpathHist = os.path.join(rootdir, figName_hist)    
    gh.save(figpathHist, width=12, height=10)


if __name__ == '__main__':

    flag_to_plot_subcapsular = False
    df = pd.read_excel("C:\develop\segmentation-eval\Radiomics_Radii_Chemo_LTP_Distances_ECALSS.xlsx")
    outdir = r"C:\develop\segmentation-eval\results"
    df.dropna(subset=['SurfaceDistances_Tumor2Ablation'], inplace=True)
    df['vals'] = df['SurfaceDistances_Tumor2Ablation'].apply(lambda x: x.replace('[', ''))
    df['vals'] = df['vals'].apply(lambda x: x.replace(']', ''))
    df['distances'] = df['vals'].apply(lambda x: x.split(','))

    # %%
    surface_distances = []
    for idx, row in df.iterrows():
        dists = row['distances']
        list_dists = []
        for idx, el in enumerate(dists):
            try:
                list_dists.append(float(el))
            except Exception:
                print(el)
                list_dists.append(np.nan)
                continue
        surface_distances.append(list_dists)

    print(len(surface_distances))
    df['surface_distances'] = surface_distances
    sum_perc_nonablated = []
    sum_perc_insuffablated = []
    sum_perc_ablated = []
    # %% drop subcapsular lesions
    if flag_to_plot_subcapsular is True:
        df_final = df[df['Proximity_to_surface'] == True]
    else:
        df_final = df.copy()
    print('No of lesions total:', str(len(df_final)))

    for row in df.itertuples():
        distance_map = row.distances
        num_voxels = row.number_nonzero_surface_pixels
        fig, ax = plt.subplots(figsize=(12, 16))
        col_height, bins, patches = ax.hist(distance_map, ec='darkgrey')

        voxels_nonablated = []
        voxels_insuffablated = []
        voxels_ablated = []

        for b, p, col_val in zip(bins, patches, col_height):
            if b < 0:
                voxels_nonablated.append(col_val)
            elif 0 <= b <= 5:
                voxels_insuffablated.append(col_val)
            elif b > 5:
                voxels_ablated.append(col_val)

        voxels_nonablated = np.asarray(voxels_nonablated)
        voxels_insuffablated = np.asarray(voxels_insuffablated)
        voxels_ablated = np.asarray(voxels_ablated)

        sum_perc_nonablated.append(((voxels_nonablated / num_voxels) * 100).sum())
        sum_perc_insuffablated.append(((voxels_insuffablated / num_voxels) * 100).sum())
        sum_perc_ablated.append(((voxels_ablated / num_voxels) * 100).sum())
        plt.close('all')

#%%
df['nonablated'] = sum_perc_nonablated
df['insufficient'] = sum_perc_insuffablated
df['sufficient'] = sum_perc_ablated
# data = [[1,2, 3], [8,9,10, 12], [346, 600, 900, 940]]
# data = df.loc[:, ['nonablated', 'insufficient', 'sufficient']]
fix, ax = plt.subplots(12, 16)
bp_dict = df.boxplot(column=['nonablated', 'insufficient', 'sufficient'], notch=False, ax=ax, return_type='both', patch_artist=True,
                      showfliers=True)
plt.show()