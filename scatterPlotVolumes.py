# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:43:21 2018
- plot volumes of tumor wrt to residual volumes

@author: Raluca Sandu
"""

import numpy as np
import pandas as pd
import graphing as gh
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
#import seaborn as sns
plt.style.use('ggplot')
#%%
volumes_df = pd.read_excel('volumetricResults.xlsx')
x = volumes_df[' Tumour Volume (ml)']
y = volumes_df[' Tumour residual volume (ml)']
y2 = volumes_df[' Tumor residual volume to tumor volume  ratio']
fig, ax = plt.subplots()
ax.scatter(x, y,s=200)
# use logarithmic scale
#ax.set_xscale('log')
plt.xlabel(' Tumour Volume (ml)', fontsize=14, color='black')
plt.ylabel(' Tumour residual volume (ml)', fontsize=14, color='black')
plt.tick_params(labelsize=14,color='black')
ax.tick_params(colors='black', labelsize=14)
#plt.grid(True)

#%%
'''tumor vs ablation vol'''

fig1, ax1 = plt.subplots()
ax1.scatter(x,1- y2,s=200)
# use logarithmic scale
#ax1.set_xscale('log')
plt.xlabel(' Tumour Volume (ml)', fontsize=14, color='black')
plt.ylabel(' Tumour coverage ratio', fontsize=14, color='black')
plt.tick_params(labelsize=14,color='black')
ax1.tick_params(colors='black', labelsize=14)

#%%
'''tumor vol vs residual tumor vol - rainbow scatter plot'''
fig2, ax2 = plt.subplots()
#cmap = ListedColormap(sns.color_palette())
colors = iter(cm.tab10(np.linspace(0, 1, len(y))))
#colors = iter(cmap(np.linspace(0, 2, len(y))))
labels = volumes_df['Patient number'].tolist()
labels_1 = ['Case ' + str(x) for x in labels]

for xs,ys,lab in zip(x, 1-y2,labels_1):
    ax2.scatter(xs, ys, s=200, marker='o', color=next(colors),label=lab)

#use logarithmic scale    

ax2.set_ylim([0, 1.04])
ax2.set_xlim([0.09, 40])
ax2.set_xscale('log')   
plt.xlabel(' Tumour Volume (ml)', fontsize=14, color='black')
plt.ylabel(' Tumour Volume Coverage Ratio', fontsize=14, color='black')
plt.title('Tumor Volume Coverage Ratio with respect to Tumor volume. 10 Cases')
plt.tick_params(labelsize=14,color='black')
ax2.tick_params(colors='black', labelsize=14)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
leg = plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc='lower left',frameon=True)
leg.get_frame().set_edgecolor('k')

# save figure
figPath = 'ScatterPlot_volumes.png'
#gh.save(figPath, width=12, height=10)