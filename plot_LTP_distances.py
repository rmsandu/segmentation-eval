# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.graphing as gh

plt.style.use('ggplot')


def flatten(a):
    return list(itertools.chain.from_iterable(a))


# %%


input_file = r"C:\develop\segmentation-eval\Radiomics_Radii_Chemo_LTP_Distances_ECALSS.xlsx"
flag_to_plot_subcapsular = False  # default value: false, only plot non-subcapsular lesions

df_final = pd.read_excel(input_file)

df_final.dropna(subset=['LTP'], inplace=True)
df_final.dropna(subset=['SurfaceDistances_Tumor2Ablation'], inplace=True)
# 'Ablation_Volume_Brochure'].replace(0, np.nan, inplace=True)
# list(map(int, "42 0".split()))
# df_final['SurfaceDistances_Tumor2Ablation'].replace(np.nan, 'None', inplace=True)
# df_final['vals'] = df_final['SurfaceDistances_Tumor2Ablation'].apply(lambda x: str(x))
df_final['vals'] = df_final['SurfaceDistances_Tumor2Ablation'].apply(lambda x: x.replace('[', ''))
df_final['vals'] = df_final['vals'].apply(lambda x: x.replace(']', ''))
df_final['distances'] = df_final['vals'].apply(lambda x: x.split(','))

# %%
surface_distances = []
for idx, row in df_final.iterrows():
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
df_final['surface_distances'] = surface_distances

# %% drop subcapsular lesions
df_final = df_final[df_final['Proximity_to_surface'] == flag_to_plot_subcapsular]
print('No of lesions total:', str(len(df_final)))
# %%
#
# df1 = df_final.set_index('LTP')
# df1.distances.explode().rename_axis('LTP').reset_index(name='sd')
# flatten the list of lists
res = df_final.groupby('LTP')['surface_distances'].apply(flatten).reset_index()

# explode lists
res = (res['surface_distances'].apply(pd.Series)
       .stack()
       .reset_index(level=1, drop=True)
       .to_frame('distances')).reset_index()
res.rename({'index': 'LTP'}, axis=1, inplace=True)

# plot the new data
fig, ax = plt.subplots(figsize=(10, 8))
bp_dict = res.boxplot(column=['distances'], notch=False, by='LTP', ax=ax, return_type='both', patch_artist=True,
                      showfliers=True)

for row_key, (ax, row) in bp_dict.iteritems():
    ax.set_xlabel('')
    for i, box in enumerate(row['fliers']):
        box.set_marker('o')
        # box.set_edgecolor('RoyalBlue')
    for i, box in enumerate(row['boxes']):
        if i == 0:
            box.set_facecolor('ForestGreen')
            box.set_edgecolor('DarkGreen')
        else:
            box.set_facecolor('gold')
            box.set_edgecolor('DarkOrange')
    for i, box in enumerate(row['medians']):
        box.set_color(color='Black')
        box.set_linewidth(2)
    for i, box in enumerate(row['whiskers']):
        box.set_color(color='Black')

xticklabels = ['No LTP', 'LTP']
xtickNames = plt.setp(ax, xticklabels=xticklabels)
plt.setp(xtickNames, fontsize=12, color='black')
plt.ylabel('Tumor to Ablation Euclidean Surface Distances [mm]', fontsize=12, color='black')
ax.tick_params(colors='black')
# [ax_tmp.set_xlabel('aaaa') for ax_tmp in np.asarray(bp).reshape(-1)]
# fig = np.asarray(bp).reshape(-1)[0].get_figure()
plt.title('Ablation Margin by Local Tumor Progression (LTP). Number of samples: ' + str(len(df_final)) + '.')
fig.suptitle('')
# ax.set_ylim([-10, 17])
figpathHist = os.path.join("figures", "boxplot LTP ablation margin_not_subcapsular_outliers")
gh.save(figpathHist, ext=['png'], close=True)
# %%
fig, ax = plt.subplots(figsize=(10, 8))
bp_dict = df_final.boxplot(column='Ablation Volume [ml]', by='LTP', ax=ax, return_type='both', patch_artist=True)

for row_key, (ax, row) in bp_dict.iteritems():
    ax.set_xlabel('')
    for i, box in enumerate(row['boxes']):
        box.set_facecolor('CornflowerBlue')
        box.set_edgecolor('RoyalBlue')
    for i, box in enumerate(row['medians']):
        box.set_color(color='Black')
        box.set_linewidth(2)
    for i, box in enumerate(row['whiskers']):
        box.set_color(color='Black')

xticklabels = ['No LTP', 'LTP']
xtickNames = plt.setp(ax, xticklabels=xticklabels)
plt.setp(xtickNames, fontsize=12, color='black')
plt.ylabel('Ablation Volume [ml]', fontsize=12, color='black')
ax.tick_params(colors='black')
# [ax_tmp.set_xlabel('aaaa') for ax_tmp in np.asarray(bp).reshape(-1)]
# fig = np.asarray(bp).reshape(-1)[0].get_figure()
plt.title('Ablation Volumes Grouped by Local Tumor Progression (LTP). Number of samples: ' + str(len(df_final)) + '.')
fig.suptitle('')
ax.set_ylim([-1, 100])
figpathHist = os.path.join("figures", "boxplot LTP ablation volumes_not_subcapsular")
gh.save(figpathHist, ext=['png'], close=True)
# %%
fig, ax = plt.subplots(figsize=(10, 8))
# props = dict(boxes="DarkGreen", whiskers="DarkOrange", medians="DarkBlue", caps="Gray")
bp_dict = df_final.boxplot(column='Tumour Volume [ml]', by='LTP', ax=ax, return_type='both',
                           patch_artist=True, labels=['LTP', 'No LTP'])

for row_key, (ax, row) in bp_dict.iteritems():
    ax.set_xlabel('')
    for i, box in enumerate(row['boxes']):
        box.set_facecolor('DarkOrange')
        box.set_edgecolor('Firebrick')
    for i, box in enumerate(row['medians']):
        box.set_color(color='black')
        box.set_linewidth(2)
    for i, box in enumerate(row['whiskers']):
        box.set_color(color='Black')

xticklabels = ['No LTP', 'LTP']
xtickNames = plt.setp(ax, xticklabels=xticklabels)
plt.setp(xtickNames, fontsize=12, color='black')
plt.ylabel('Tumor Volume [ml]', fontsize=12, color='black')
ax.tick_params(colors='black')
# [ax_tmp.set_xlabel('aaaa') for ax_tmp in np.asarray(bp).reshape(-1)]
# fig = np.asarray(bp).reshape(-1)[0].get_figure()
plt.title('Tumor Volumes Grouped by Local Tumor Progression (LTP).Number of samples: ' + str(len(df_final)) + '.')
fig.suptitle('')
ax.set_ylim([-1, 20])
figpathHist = os.path.join("figures", "boxplot LTP tumor volumes_not_subcapsular")
gh.save(figpathHist, ext=['png'], close=True)
