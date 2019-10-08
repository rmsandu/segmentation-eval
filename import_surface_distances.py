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
print(type(df_final.surface_distances.loc[0]))
print(df_final.surface_distances.loc[0])
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
res.boxplot(column=['distances'], by='LTP', return_type='axes')
# dst_ltp = []
# dst_no_ltp = []
#
# for idx, row in df_final.iterrows():
#     if row['LTP'] == 1:
#         dst_no_ltp.append(row['surface_distances'])
#         dst_ltp.append(np.nan)
#     if row['LTP'] == 0:
#         dst_ltp.append(row['surface_distances'])
#         dst_no_ltp.append(np.nan)

# new_df = pd.DataFrame(columns=['No LTP at 6m', 'LTP at 6m'], index=range(0, len(dst_ltp)))
# new_df['No LTP at 6m'] = dst_no_ltp
# new_df['LTP at 6m'] = dst_ltp
# df1 = new_df.transpose()
# fig, ax = plt.subplots(figsize=(10,8))
# df1.boxplot(column=['No LTP at 6m', 'LTP at 6m'])
# df_ltp = df_ltp.groupby('LTP')
# ls = df_ltp.surface_distances.values.tolist()
# df_ltp['dst'] = ls
# df_ltp.boxplot(column='dst', by='LTP', ax=ax, return_type='axes')
# boxplot_ltp = ax.boxplot(df_ltp.surface_distances.values.tolist())

# ax = df_ltp.boxplot(column='surface_distances', ax=ax, return_type='axes')


#
# plt.show()
# # plt.ylim([-1, 150])
# plt.tick_params(labelsize=8, color='black')
# ax.tick_params(colors='black', labelsize=8, color='k')
# ax.set_ylim([-1, 20])
# figpathHist = os.path.join("figures", "boxplot LTP ablation volumes")
#%%
# df_ltp = df_final.groupby('LTP')
fig, ax = plt.subplots(figsize=(10,8))
df_final.boxplot(column='Ablation Volume [ml]', by='LTP', ax=ax, return_type='axes', patch_artist=True)

plt.show()
# plt.ylim([-1, 150])
# plt.tick_params(labelsize=8, color='black')
# ax.tick_params(colors='black', labelsize=8, color='k')
ax.set_ylim([-1, 100])
figpathHist = os.path.join("figures", "boxplot LTP ablation volumes")

#%%
fig, ax = plt.subplots(figsize=(10,8))
props = dict(boxes="DarkGreen", whiskers="DarkOrange", medians="DarkBlue", caps="Gray")
bp_dict = df_final.boxplot(column='Tumour Volume [ml]', by='LTP', ax=ax, return_type='both',
                         patch_artist=True, labels=['LTP', 'No LTP'])
colors = ['b', 'y', 'm', 'c', 'g', 'b', 'r', 'k', ]
for row_key, (ax,row) in bp_dict.iteritems():
    ax.set_xlabel('')
    for i,box in enumerate(row['boxes']):
        box.set_facecolor('DarkOrange')


# [ax_tmp.set_xlabel('aaaa') for ax_tmp in np.asarray(bp).reshape(-1)]
# fig = np.asarray(bp).reshape(-1)[0].get_figure()
fig.suptitle('New title here')
# df.plot.box(color=props, patch_artist=True)
# plt.setp(bplot['medians'], color='black', linewidth=1.5)
# plt.setp(bplot['means'], marker='D', markeredgecolor='darkred',
#          markerfacecolor='darkred', label='Mean')
# plt.show()
# plt.ylim([-1, 150])
# ax.tick_params(labelsize=8, color='black')
# ax.xaxis.label.set_color('black')

# ax.tick_params(colors='black', labelsize=8, color='k')
ax.set_ylim([-1, 20])
figpathHist = os.path.join("figures", "boxplot LTP tumor volumes")
# gh.save(figpathHist, ext=['png'], close=True)

#%% TODO: write treatment id as well. the unique key must be formed out of: [patient_id, treatment_id, lesion_id]
# filepath_excel = os.path.join(outdir, "Radiomics_Radii_Chemo_LTP_Distances_MAVERRIC.xlsx")
# writer = pd.ExcelWriter(filepath_excel)
# df_final.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
# writer.save()