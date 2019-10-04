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
df.dropna(subset=['Lesion_ID'], inplace=True)
df['Proximity_to_vessels'].replace(True, 'YES', inplace=True)
df['Proximity_to_vessels'].replace(False, 'NO', inplace=True)
df['Proximity_to_vessels'].replace('', 'NaN', inplace=True)
# TODO: with and without comments
# TODO: with and without vessel proximity
# TODO: volume into formula

idx_comments = df.columns.get_loc('Power')
df_corr = df.iloc[:, idx_comments:len(df.columns)].copy()
print('3. Dropping Outliers from the Energy Column using val < quantile 0.99')
q = df['Energy [kj]'].quantile(0.99)
df1_no_outliers = df[df['Energy [kj]'] < q]
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
#%% Heatmap of correlations
idx_comments = df.columns.get_loc('Power')
df_corr = df.iloc[:, idx_comments:len(df.columns)-5].copy()
sns.heatmap(df_corr.corr(), square=True, annot=True, linewidths=.2, vmin=-1, vmax=1)
figpathHist = os.path.join("figures", "HeatMap Correlations")
gh.save(figpathHist, width=24, height=24, ext=['png'], close=True)
# %% group by proximity to vessels
groups = df1_no_outliers.groupby('Proximity_to_vessels')
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
plt.title("Ablation Volume vs Energy Grouped by Proximity to Vessels",fontsize=20, color='black')
plt.tick_params(labelsize=20, color='black')
plt.legend(title_fontsize=20)
ax.tick_params(colors='black', labelsize=20)

figpathHist = os.path.join("figures", "Ablation Volume vs Energy Grouped by Proximity to Vessels")
gh.save(figpathHist, width=18, height=16, ext=['png'], close=True)
#%% group by device name
groups = df1_no_outliers.groupby('Device_name')
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
df_angyodinamics.dropna(subset=['least_axis_length_ablation'], inplace=True)

title = "Ablation Volumes for tumors treated with Angiodynamics MWA Device"

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'title': title,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

title = "Least Ablation Diameter vs. MWA Energy for tumors treated with Angiodynamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'least_axis_length_ablation', 'title': title,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

title = "Major Ablation Diameter vs. MWA Energy for tumors treated with Angiodynamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'major_axis_length_ablation',
          'title': title,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

title = "Minor Ablation Diameter vs. MWA Energy for  tumors treated with Angiodynamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'minor_axis_length_ablation', 'title': title,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)


#%%
title = "Ablation Diameter Coronal Plane vs. MWA Energy for tumors treated with Angyodinamics MWA Device"
ylabel = "Diameter Coronal Plane (Euclidean Distances based) [mm]"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'diameter2D_col_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = "Diameter Saggital Plane (Euclidean Distances based) [mm]"
title = "Ablation Diameter Saggital Plane vs. MWA Energy for tumors treated with Angyodinamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'diameter2D_row_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = "Diameter Axial Plane (Euclidean Distances based) [mm]"
title = "Ablation Diameter Coronal Plane vs. MWA Energy for  tumors treated with Angyodinamics MWA Device"
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'diameter2D_slice_ablation', 'title': title,
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

#%% Time Duration and Power
df_angyodinamics["Time_Duration_Applied"] = pd.to_numeric(df_angyodinamics["Time_Duration_Applied"])

ylabel = 'Least Ablation Diameter (PCA-based) [mm]'
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
kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'minor_axis_length_ablation',
          'title': 'Time Duration Applied [s] vs Minor Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
#
#%% Ablation vs Tumor Axis
df_angyodinamics["Time_Duration_Applied"] = pd.to_numeric(df_angyodinamics["Time_Duration_Applied"])
kwargs = {'x_data': 'least_axis_length_tumor', 'y_data': 'least_axis_length_ablation',
          'title': ' Least Axis Tumor vs Least Axis Ablation [mm]',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'major_axis_length_tumor', 'y_data': 'major_axis_length_ablation',
          'title': 'Major Axis Tumor vs Major Axis Ablation [mm]',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'minor_axis_length_tumor', 'y_data': 'minor_axis_length_ablation',
          'title': 'Minor Axis Tumor vs Minor Axis Ablation[mm]',
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
          'title': 'Power Applied [W] vs Major Axis Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

ylabel = 'Minor Ablation Diameter (PCA-based) [mm]'
kwargs = {'x_data': 'Power', 'y_data': 'minor_axis_length_ablation',
          'title': 'Power Applied [W] vs Minor Axis Ablation Diameter [mm]',
          'y_label': ylabel,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

#%% Sphericity
kwargs = {'x_data': 'sphericity_tumor', 'y_data': 'elongation_ablation',
          'title': ' Sphericity Tumor vs Elongation Ablation',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'sphericity_tumor', 'y_data': 'elongation_ablation',
          'title': ' Sphericity Tumor vs Elongation Ablation',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'y_data': 'sphericity_tumor', 'x_data': 'Energy [kj]',
          'title': ' Sphericity Tumor vs Energy Ablation ',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'elongation_ablation',
          'title': ' Elongation Ablation vs Energy Ablation [mm]',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'sphericity_ablation',
          'title': ' Sphericity Ablation vs Energy Ablation [mm]',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
#%% Gray level variance tumor vs energy
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'intensity_mean_tumor',
          'title': 'Energy Applied vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'intensity_variance_tumor',
          'title': 'Energy Applied vs Variance Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Energy Applied vs Tumor Pixel Uniformity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'gray_lvl_nonuniformity_tumor',
          'title': 'Energy Applied vs Tumor Pixel NonUniformity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
#%% tumor size vs intensities
kwargs = {'x_data': 'least_axis_length_tumor', 'y_data': 'intensity_mean_tumor',
          'title': 'Least Axis Length Tumor vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'minor_axis_length_tumor', 'y_data': 'intensity_mean_tumor',
          'title': 'Minor Axis Length Tumor vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'major_axis_length_tumor', 'y_data': 'intensity_mean_tumor',
          'title': 'Major Axis Length Tumor vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
#%% gray lvl vs ablation metrics
kwargs = {'x_data': 'least_axis_length_ablation', 'y_data': 'intensity_mean_tumor',
          'title': 'Least Ablation Axis Length vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'least_axis_length_ablation', 'y_data': 'intensity_variance_tumor',
          'title': 'Least Ablation Axis Length vs Variance Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'least_axis_length_ablation', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Least Ablation Axis Length vs Uniformity Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'intensity_variance_tumor',
          'title': 'Minor Axis Length Ablation vs VarianceTumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'intensity_mean_tumor',
          'title': 'Minor Axis Length Ablation vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Minor Axis Length Ablation vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Major Axis Length Ablation vs Tumor Pixel Uniformity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'intensity_mean_tumor',
          'title': 'Major Axis Length Ablation vs Mean Tumor Pixel',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'intensity_variance_tumor',
          'title': 'Major Axis Length Ablation vs Mean Variance Tumor Pixel',
          'lin_reg': 1}

scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'gray_lvl_nonuniformity_tumor',
          'title': 'Energy Applied vs Tumor Pixel NonUniformity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
#%% percentage distances  histograms
fig, ax = plt.subplots()
df["safety_margin_distribution_0"].replace(0, np.nan, inplace=True)
df["safety_margin_distribution_5"].replace(0, np.nan, inplace=True)
df["safety_margin_distribution_10"].replace(0, np.nan, inplace=True)
idx_margins = df.columns.get_loc('safety_margin_distribution_0')
df_margins = df.iloc[:, idx_margins: idx_margins + 3].copy()
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
plt.title('Ablation Surface Margin Coverages [%] Histogram for all MWA device models.')
plt.legend(fontsize=20)
plt.xticks(range(0, 101, 10))
figpathHist = os.path.join("figures", "surface margin frequency percentages overlaid")
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

#%% percentage distances  histograms for angyodinamics
fig, ax = plt.subplots()
df_angyodinamics["safety_margin_distribution_0"].replace(0, np.nan, inplace=True)
df_angyodinamics["safety_margin_distribution_5"].replace(0, np.nan, inplace=True)
df_angyodinamics["safety_margin_distribution_10"].replace(0, np.nan, inplace=True)
idx_margins = df_angyodinamics.columns.get_loc('safety_margin_distribution_0')
df_margins = df_angyodinamics.iloc[:, idx_margins: idx_margins + 3].copy()
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
plt.title('Ablation Surface Margin Coverages [%] Histogram for Angiodynamics MWA device model.')
plt.legend(fontsize=20)
plt.xticks(range(0, 101, 10))
figpathHist = os.path.join("figures", "surface margin frequency percentages overlaid angiodyanmics")
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)


# %% histogram axes ablation
plt.figure()
df_angyodinamics.hist(column=["major_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram major axis length ablation angyodinamics")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 30]))
plt.ylabel('mm')
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df_angyodinamics.hist(column=["least_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram least axis length ablation angyodinamics")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 30]))

gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df_angyodinamics.hist(column=["minor_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram minor axis length ablation angyodinamics")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 30]))
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

print('All  plots saved as *.png files in dev folder figures')
plt.close('all')
# %% histogram axis tumor
plt.figure()
df_angyodinamics.hist(column=["major_axis_length_tumor"])
figpathHist = os.path.join("figures", "histogram major axis length tumor angyodinamics")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 30]))

gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df_angyodinamics.hist(column=["least_axis_length_tumor"])
figpathHist = os.path.join("figures", "histogram least axis length tumor angyodinamics")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 30]))

gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df_angyodinamics.hist(column=["minor_axis_length_tumor"])
figpathHist = os.path.join("figures", "histogram minor axis length tumor angyodinamics")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 30]))
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

plt.close('all')