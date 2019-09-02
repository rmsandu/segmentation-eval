# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:43:21 2018
- plot volumes of tumor wrt to residual volumes

@author: Raluca Sandu
"""

import pandas as pd
import numpy as np
import graphing as gh
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.style.use('ggplot')
#%%
filepath_excel = r"C:\PatientDatasets_GroundTruth_Database\DistanceVolumeMetrics_Pooled_Ablation to Tumor Euclidean Distances.xlsx"
df_volumes = pd.read_excel(filepath_excel)
x = df_volumes['Tumour Volume (ml)']
y = df_volumes['Tumour residual volume (ml)']
y2 = df_volumes[' Tumour coverage ratio']
fig, ax = plt.subplots()
ax.scatter(x, y, s=200)
plt.xlabel(' Tumour Volume (ml)', fontsize=14, color='black')
plt.ylabel(' Tumour residual volume (ml)', fontsize=14, color='black')
plt.tick_params(labelsize=14, color='black')
ax.tick_params(colors='black', labelsize=14)
#plt.grid(True)

#%%
'''tumor vs ablation vol'''
fig1, ax1 = plt.subplots()
ax1.scatter(x, y2, s=200)
plt.xlabel(' Tumour Volume (ml)', fontsize=14, color='black')
plt.ylabel(' Tumour coverage ratio', fontsize=14, color='black')
plt.tick_params(labelsize=14, color='black')
ax1.tick_params(colors='black', labelsize=14)
#%%
'''tumor vol vs residual tumor vol - rainbow scatter plot'''
fig2, ax2 = plt.subplots()
colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y))))
labels = df_volumes['PatientID'].tolist()
labels_1 = ['Case ' + str(x) for x in range(1, len(labels)+1)]
i =1
for xs, ys, lab in zip(x, y2, labels_1):
    ax2.scatter(xs, ys, s=200, marker='o', color=next(colors), label=lab)
    # annotate the markers
#    plt.text(xs * (1 + 0.01), ys * (1 + 0.01) , i, fontsize=12)
#    i +=1
# use logarithmic scale because tumor volume coverage ratio is [0-1] volume is in ml
ax2.set_ylim([0, 1.04])
ax2.set_xlim([0.09, 40])
ax2.set_xscale('log')
plt.xlabel(' Tumour Volume (ml)', fontsize=14, color='black')
plt.ylabel(' Tumour Volume Coverage Ratio', fontsize=14, color='black')
plt.title('Tumor Volume Coverage Ratio with respect to Tumor volume. 21 Cases')
plt.tick_params(labelsize=14, color='black')
ax2.tick_params(colors='black', labelsize=14)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
leg = plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc='lower left', frameon=True)
leg.get_frame().set_edgecolor('k')

# save figure
#figPath = os.path.join('ScatterPlot_volumes.png')
#gh.save(figPath, width=12, height=10)
