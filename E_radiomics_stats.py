# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.graphing as gh
from utils.scatter_plot import scatter_plot

sns.set(style="ticks")
plt.style.use('ggplot')

# %%
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_file", required=True, help="input file pooled radiomics ")
args = vars(ap.parse_args())

df = pd.read_excel(args["input_file"], sheet_name="radiomics")
# rmv empty rows
df['Energy [kj]'].replace('', np.nan, inplace=True)
try:
    df['MISSING'].replace('', np.nan, inplace=True)
except Exception:
    print("column MISSING is not present in the input file")
print("1. Removing RadioFrequency Devices from the input file")
df = df[df['Device_name'] != 'Boston Scientific (Boston Scientific - RF 3000)']
print("2. Droping NaNs")
df.dropna(subset=["Ablation Volume [ml]"], inplace=True)

idx_comments = df.columns.get_loc('Device_name')
df1 = df.iloc[:, idx_comments:len(df.columns)].copy()
print('3. Dropping Outliers from the Energy Column using val < quantile 0.99')
q = df1['Energy [kj]'].quantile(0.99)
df1_no_outliers = df1[df1['Energy [kj]'] < q]
df1_no_outliers.reset_index(inplace=True, drop=True)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Tumour Volume [ml]',
          'title': "Tumors Volumes for 3 MWA devices. Outliers Removed.", 'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]', 'title': "Ablation Volumes for 3 MWA devices. ",
          'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)
df1_no_outliers['Ratio_AT_vol'] = df1_no_outliers['Tumour Volume [ml]'] / df1_no_outliers['Ablation Volume [ml]']
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ratio_AT_vol',
          'title': "Tumor to Ablation Volume Ratio for 3 MWA devices.",
          'y_label': 'R(Tumor Volume: Ablation Volume)', 'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Tumour coverage ratio',
          'title': "Tumor Coverage Ratio for 3 MWA devices.",
          'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)
# %%
groups = df1.groupby('Device_name')
fig, ax = plt.subplots()
lesion_per_device = []
device_name_grp = []
for name, group in groups:
    ax.plot(group["Energy [kj]"], group["Ablation Volume [ml]"], marker='o', linestyle='', ms=14, label=name)
    lesion_per_device.append(len(group))
    device_name_grp.append(name)
L = ax.legend()
L_labels = L.get_texts()

for idx, L in enumerate(L_labels):
    L.set_text(device_name_grp[idx] + ' N=' + str(lesion_per_device[idx]))


plt.xlabel('Energy [kJ]', fontsize=20, color='black')
plt.ylabel('Ablation Volume [ml]', fontsize=20, color='black')
plt.tick_params(labelsize=20, color='black')
plt.legend(title_fontsize=20)
ax.tick_params(colors='black', labelsize=20)
figpathHist = os.path.join("figures", "Ablation Volume vs  Energy per MWA Device Category")
gh.save(figpathHist, width=18, height=16, ext=['png'], close=True)

fig, ax = plt.subplots()
lesion_per_device = []
device_name_grp = []

for name, group in groups:
    ax.plot(group["Energy [kj]"], group["Tumour Volume [ml]"], marker='o', linestyle='', ms=14, label=name)
    lesion_per_device.append(len(group))
    device_name_grp.append(name)
L = ax.legend()
L_labels = L.get_texts()
for idx, L in enumerate(L_labels):
    L.set_text(device_name_grp[idx] + ' N=' + str(lesion_per_device[idx]))
plt.ylabel('Tumor Volume [ml]', fontsize=20, color='black')
plt.xlabel('Energy [kJ]', fontsize=20, color='black')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
# plt.legend(fontsize=12)
figpathHist = os.path.join("figures", "Tumor Volume vs  Energy per MWA Device Category")
gh.save(figpathHist, width=18, height=16, ext=['png'], close=True)

# %% ANGYODINAMICS
fig, ax = plt.subplots()
df_angyodinamics = df1_no_outliers[df1_no_outliers["Device_name"] == "Angyodinamics (Acculis)"]
df_angyodinamics.dropna(subset=['Energy [kj]'], inplace=True)
df_angyodinamics.dropna(subset=['least_axis_length'], inplace=True)
title = "Minimum Ablation Diameter vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'least_axis_length_ablation', 'title': title,
          'y_label':'Least Ablation Diameter [mm]',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

title = "Maximum Ablation Diameter vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device"
ylabel = "Major Diameter (PCA-based ellipsoid approximation) [mm]"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'major_axis_length_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

title = "Minor Ablation Diameter vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device"
ylabel = "Major Diameter (PCA-based ellipsoid approximation) [mm]"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'minor_axis_length_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)


#%%
title = "Ablation Diameter Coronal Plane vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device"
ylabel = "Diameter Coronal Plane (Euclidean Distances based) [mm]"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'diameter2D_col_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
# plt.ylim([25, 50])
ylabel = "Diameter Saggital Plane (Euclidean Distances based) [mm]"
title = "Ablation Diameter Saggital Plane vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'diameter2D_row_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = "Diameter Axial Plane (Euclidean Distances based) [mm]"
title = "Ablation Diameter Coronal Plane vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'diameter2D_slice_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

#%% Time Duration and Power
df_angyodinamics["Time_Duration_Applied"] = pd.to_numeric(df_angyodinamics["Time_Duration_Applied"])
ylabel ='Least Ablation Diameter (PCA-based) [mm]'
kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'least_axis_length_ablation',
          'title': 'Time Duration Applied [s] vs Least Axis Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = 'Major Ablation Diameter (PCA-based) [mm]'
kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'major_axis_length_ablation',
          'title': 'Time Duration Applied [s] vs Major Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = 'Minor Ablation Diameter (PCA-based) [mm]'
kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'minor_axis_length',
          'title': 'Time Duration Applied [s] vs Minor Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

#%% Power
df_angyodinamics["Power"] = pd.to_numeric(df_angyodinamics["Power"])
ylabel = 'Least Ablation Diameter (PCA-based) [mm]'
kwargs = {'x_data': 'Power', 'y_data': 'least_axis_length_ablation',
          'title': 'Power Applied [W] vs Least Axis Ablation Diameter[mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = 'Major Ablation Diameter (PCA-based) [mm]'
kwargs = {'x_data': 'Power', 'y_data': 'major_axis_length_ablation',
          'title': 'Power Applied [W] vs Major Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = 'Minor Ablation Diameter (PCA-based) [mm]'
kwargs = {'x_data': 'Power', 'y_data': 'minor_axis_length_ablation',
          'title': 'Power Applied [W] vs Minor Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

#%% percentage distances  histograms

fig, ax = plt.subplots()
df1["safety_margin_distribution_0"].replace(0, np.nan, inplace=True)
df1["safety_margin_distribution_5"].replace(0, np.nan, inplace=True)
df1["safety_margin_distribution_10"].replace(0, np.nan, inplace=True)
idx_margins = df.columns.get_loc('safety_margin_distribution_0')
df_margins = df1.iloc[:, len(df1.columns) - 3: len(df1.columns)].copy()
df_margins.reset_index(drop=True, inplace=True)
df_margins_sort = pd.DataFrame(np.sort(df_margins.values, axis=0), index=df_margins.index, columns=df_margins.columns)
# df_margins_sort.hist(alpha=0.5)

labels = [{'Ablation Surface Margin ' + r'$x > 5$' + 'mm '},
          {'Ablation Surface Margin ' + r'$0 \leq  x \leq 5$' + 'mm'}, {'Ablation Surface Margin ' + r'$x < 0$' + 'mm'}]
for idx, col in enumerate(df_margins.columns):
    sns.distplot(df_margins[col], label=labels[idx],
                 bins=range(0, 101, 10),
                 kde=False, hist_kws=dict(edgecolor='black'))

plt.xlabel('Percentage of Surface Margin Covered for different ablation margins ranges', fontsize=20, color='black')
plt.ylabel('Frequency', fontsize=20, color='black')
plt.legend(fontsize=20)
plt.xticks(range(0, 101, 10))
figpathHist = os.path.join("figures", "surface margin frequency percentages overlaid")
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)
#


# %% histogram axes
plt.figure()
df_angyodinamics.hist(column=["major_axis_length_ablation"])

figpathHist = os.path.join("figures", "histogram major axis length ablation angyodinamics")
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df_angyodinamics.hist(column=["least_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram least axis length ablation angyodinamics")
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df_angyodinamics.hist(column=["minor_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram minor axis length ablation angyodinamics")
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

print('All  plots saved as *.png files in dev folder figures')
plt.close('all')
# %%

