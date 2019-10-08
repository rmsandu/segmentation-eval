# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

import utils.graphing as gh
plt.style.use('ggplot')


def flatten(a):
    return list(itertools.chain.from_iterable(a))

df = pd.read_excel('Radiomics_Radii_Chemo_LTP_MAVERRIC.xlsx')
rootdir = r"C:\Figures"
outdir = r"C:\develop\segmentation-eval"

frames = []
for subdir, dirs, files in os.walk(rootdir):
    for file in sorted(files):
        if file.endswith('.xlsx'):
            # check file extension is xlsx
            excel_input_file_per_lesion = os.path.join(subdir, file)
            df_single_lesion = pd.read_excel(excel_input_file_per_lesion,  sheet_name='SurfaceDistances')
            df_single_lesion.rename(columns={'lesion_id': 'Lesion_ID', 'patient_id': 'Patient_ID'}, inplace=True)
            try:
                patient_id = df_single_lesion.loc[0]['Patient_ID']
            except Exception as e:
                print(repr(e))
                print("Path to bad excel file:", excel_input_file_per_lesion)
                continue
            try:
                df_single_lesion['Lesion_ID'] = df_single_lesion['Lesion_ID'].apply(
                    lambda x: 'MAV-' + patient_id + '-L' + str(x))
            except Exception as e:
                print(repr(e))
                print("Path to bad excel file:", excel_input_file_per_lesion)
                continue
            frames.append(df_single_lesion)
            # concatenate on patient_id and lesion_id
            # rename the columns
            # concatenate the rest of the pandas dataframe based on the lesion id.
            # first edit the lesion id.

# result = pd.concat(frames, axis=1, keys=['Patient ID', 'Lesion id', 'ablation_date'], ignore_index=True)

result = pd.concat(frames, ignore_index=True)
df_final = pd.merge(df, result, how="outer", on=['Patient_ID', 'Lesion_ID'])

#%% PLOT
df_final.dropna(subset=['LTP'], inplace=True)
df_final.dropna(subset=['SurfaceDistances_Tumor2Ablation'], inplace=True)
# 'Ablation_Volume_Brochure'].replace(0, np.nan, inplace=True)
# list(map(int, "42 0".split()))
# df_final['SurfaceDistances_Tumor2Ablation'].replace(np.nan, 'None', inplace=True)
# df_final['vals'] = df_final['SurfaceDistances_Tumor2Ablation'].apply(lambda x: str(x))
df_final['vals'] = df_final['SurfaceDistances_Tumor2Ablation'].apply(lambda x: x.replace('[', ''))
df_final['vals'] = df_final['vals'].apply(lambda x: x.replace(']', ''))
df_final['distances'] = df_final['vals'].apply(lambda x: x.split(','))


#%%
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

#%%
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
res.rename({'index':'LTP'}, axis=1, inplace=True)

# plot the new data
fig, ax = plt.subplots(figsize=(10,8))
bp_dict = res.boxplot(column=['distances'],  notch=True, by='LTP', ax=ax, return_type='both', patch_artist=True,  showfliers=False)

for row_key, (ax,row) in bp_dict.iteritems():
    ax.set_xlabel('')
    for i,box in enumerate(row['boxes']):
        box.set_facecolor('CornflowerBlue')
        box.set_edgecolor('RoyalBlue')
    for i, box in enumerate(row['medians']):
        box.set_color(color='Black')
        box.set_linewidth(2)
    for i, box in enumerate(row['whiskers']):
        box.set_color(color='Black')

xticklabels = ['LTP', 'No LTP']
xtickNames = plt.setp(ax, xticklabels=xticklabels)
plt.setp(xtickNames,  fontsize=12, color='black')
plt.ylabel('Tumor to Ablation Euclidean Surface Distances [mm]', fontsize=12, color='black')
ax.tick_params(colors='black')
# [ax_tmp.set_xlabel('aaaa') for ax_tmp in np.asarray(bp).reshape(-1)]
# fig = np.asarray(bp).reshape(-1)[0].get_figure()
fig.suptitle('Ablation Margin by Local Tumor Progression (LTP)')
figpathHist = os.path.join("figures", "boxplot LTP ablation margin")
gh.save(figpathHist, ext=['png'], close=True)
#%%
fig, ax = plt.subplots(figsize=(10,8))
bp_dict = df_final.boxplot(column='Ablation Volume [ml]', by='LTP', ax=ax, return_type='both', patch_artist=True)

for row_key, (ax,row) in bp_dict.iteritems():
    ax.set_xlabel('')
    for i,box in enumerate(row['boxes']):
        box.set_facecolor('CornflowerBlue')
        box.set_edgecolor('RoyalBlue')
    for i, box in enumerate(row['medians']):
        box.set_color(color='Black')
        box.set_linewidth(2)
    for i, box in enumerate(row['whiskers']):
        box.set_color(color='Black')

xticklabels = ['LTP', 'No LTP']
xtickNames = plt.setp(ax, xticklabels=xticklabels)
plt.setp(xtickNames,  fontsize=12, color='black')
plt.ylabel('Ablation Volume [ml]', fontsize=12, color='black')
ax.tick_params(colors='black')
# [ax_tmp.set_xlabel('aaaa') for ax_tmp in np.asarray(bp).reshape(-1)]
# fig = np.asarray(bp).reshape(-1)[0].get_figure()
fig.suptitle('Ablation Volumes Grouped by Local Tumor Progression (LTP)')

ax.set_ylim([-1, 100])
figpathHist = os.path.join("figures", "boxplot LTP ablation volumes")
gh.save(figpathHist, ext=['png'], close=True)
#%%
fig, ax = plt.subplots(figsize=(10,8))
props = dict(boxes="DarkGreen", whiskers="DarkOrange", medians="DarkBlue", caps="Gray")
bp_dict = df_final.boxplot(column='Tumour Volume [ml]', by='LTP', ax=ax, return_type='both',
                         patch_artist=True, labels=['LTP', 'No LTP'])

for row_key, (ax,row) in bp_dict.iteritems():
    ax.set_xlabel('')
    for i,box in enumerate(row['boxes']):
        box.set_facecolor('CornflowerBlue')
        box.set_edgecolor('RoyalBlue')
    for i, box in enumerate(row['medians']):
        box.set_color(color='Black')
        box.set_linewidth(2)
    for i, box in enumerate(row['whiskers']):
        box.set_color(color='Black')

xticklabels = ['LTP', 'No LTP']
xtickNames = plt.setp(ax, xticklabels=xticklabels)
plt.setp(xtickNames,  fontsize=12, color='black')
plt.ylabel('Tumor Volume [ml]', fontsize=12, color='black')
ax.tick_params(colors='black')
# [ax_tmp.set_xlabel('aaaa') for ax_tmp in np.asarray(bp).reshape(-1)]
# fig = np.asarray(bp).reshape(-1)[0].get_figure()
fig.suptitle('Tumor Volumes Grouped by Local Tumor Progression (LTP)')
ax.set_ylim([-1, 20])
figpathHist = os.path.join("figures", "boxplot LTP tumor volumes")
gh.save(figpathHist, ext=['png'], close=True)

#%% TODO: write treatment id as well. the unique key must be formed out of: [patient_id, treatment_id, lesion_id]
# filepath_excel = os.path.join(outdir, "Radiomics_Radii_Chemo_LTP_Distances_MAVERRIC.xlsx")
# writer = pd.ExcelWriter(filepath_excel)
# df_final.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
# writer.save()