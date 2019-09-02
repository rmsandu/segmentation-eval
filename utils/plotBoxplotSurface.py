# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:58:43 2017

@author: Raluca Sandu
"""
import os
import graphing as gh
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.style.use('ggplot')
#%%
def plotBinsSurfacePercentage(data, rootdir, flag_all_ranges):

    '''plot boxplots per range of mm versus percentage of surface covered - all patients'''
     # 1. plot boxplot for each discrete range (0-1), (1-0)
     # 2. plot boxplot for category (greater than -5) (-5-0mm)  (0-5mm) (5-10mm) (greater than 10mm)
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
        
        xticklabels = [ r"$(\infty< x < -10$)",r"$(-10 \leqslant x \leqslant -5$)", r"$(-5 < x \leqslant  0$)", r"$(0< x < 5$)",r"$(5  \leqslant x   \leqslant 10$)",r"$(x > 10)$"]
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