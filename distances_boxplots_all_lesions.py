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


#plt.style.use('classic')
#plt.style.use('seaborn')
#%%
def plotBoxplots(data, rootdir):
    
    
    fig, axes = plt.subplots(figsize=(12, 16))
    
    # Horizontal box plot
    bplot = plt.boxplot(data,
                         vert=True,   # vertical box aligmnent
                         patch_artist=False,
                         showmeans=True)   # fill with color
                        
                         
    for element in ['medians','boxes', 'fliers', 'whiskers', 'caps']:
        plt.setp(bplot[element], color='black',linewidth=1.5)
    
    plt.setp(bplot['whiskers'], linestyle='--')
    plt.setp(bplot['fliers'], markersize=5)
    plt.setp(bplot['means'], marker='D', markeredgecolor='black',
                      markerfacecolor='blue', label='Mean')

                         
    xlim = np.array(plt.gca().get_xlim())
    ylim = np.array(plt.gca().get_ylim())
    plt.fill_between(xlim, y1=([ylim[0],ylim[0]]) , y2=([0, 0]) ,
                     color="#EC7063", zorder=0)
    plt.fill_between(xlim, y1=([0, 0]), y2=([5, 5]) ,
                     color="#FAD7A0", zorder=0)
    plt.fill_between(xlim, y1=([5, 5]), y2=(ylim[1], ylim[1]), 
                     color="#ABEBC6", zorder=0 )
    plt.margins(0)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=16)
    
    plt.title('Boxplots for Euclidean Distances between Tumor and Ablations. 10 Cases.', fontsize=16)
    plt.xlabel('Lesion', fontsize=16,color='black')
    plt.tick_params(labelsize=16, color='black')
    plt.ylabel('[mm]', fontsize=16, color='black')
    figpath = os.path.join(rootdir,'boxplots_forDistanceMaps.png')
    plt.show()     
    gh.save(figpath, width=12, height=10)                
