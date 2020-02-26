# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import argparse
import sys

import numpy as np
import pandas as pd
import seaborn as sns

from scripts.scatter_plot import scatter_plot

sns.set(style="ticks")
# plt.style.use('ggplot')

# %%
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_file", required=True, help="input file pooled radiomics ")
# e.g. Radiomics_Radii_Chemo_LTP_Distances_ECALSS
args = vars(ap.parse_args())
subcapsular_lesions_only = None
# "Proximity_to_surface"
# FALSE: non subcapsular AKA Deep Tumors
# TRUE: subcapsular AKA subcapsular TUmors
df_input = pd.read_excel(args["input_file"], sheet_name="radiomics")

# %% BOXPLOTS
# plot_boxplots_chemo(df_input)
# plot_boxplots_volumes(df_input)
# plot_boxplots_subcapsular(df_input)

#%%
if subcapsular_lesions_only is not None:
    df = df_input[df_input['Proximity_to_surface'] == subcapsular_lesions_only]
    if subcapsular_lesions_only is False:
        flag_subcapsular_title = 'Non-Subcapsular Tumors.'
    else:
        flag_subcapsular_title = 'Subcapsular Tumors.'
else:
    flag_subcapsular_title = ''
    df = df_input.copy()

# %%
# rmv empty rows
df['Energy [kj]'].replace('', np.nan, inplace=True)
try:
    df['MISSING'].replace('', np.nan, inplace=True)
except Exception:
    print("column MISSING is not present in the input file")

print("1. Removing RadioFrequency Devices from the input file")
df = df[df['Device_name'] != 'Boston Scientific (Boston Scientific - RF 3000)']
print("2. Droping NaNs")
print('2.1 Drop Nans from Ablation Volume')
df.dropna(subset=["Ablation Volume [ml]"], inplace=True)

# print('2.2 Drop Nans from Energy')
# df.dropna(subset=['Energy [kj]'], inplace=True)
# print('2.3 Drop Duplicates from Lesion')
df.dropna(subset=['Lesion_ID'], inplace=True)
df['Proximity_to_vessels'].replace(True, 'YES', inplace=True)
df['Proximity_to_vessels'].replace(False, 'NO', inplace=True)
df['Proximity_to_vessels'].replace('', 'NaN', inplace=True)
df.reset_index(inplace=True, drop=True)


# %%  Raw Data

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'x_label': 'Energy [kJ]',
          'y_label': 'Effective Ablation Volume [ml] for 3 Microwave Devices',
          'title': "Ablation Volume [ml] for 3 MWA devices. " + flag_subcapsular_title,
          'x_lim': 100,
          'y_lim': 100,
          'lin_reg': 1}
scatter_plot(df, **kwargs)

df_radiomics_amica = df[df['Device_name'] == 'Amica (Probe)']
df_radiomics_angyodinamics = df[df['Device_name'] == 'Angyodinamics (Acculis)']
df_radiomics_covidien = df[df['Device_name'] == 'Covidien (Covidien MWA)']

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'y_label': 'Effective Ablation Volume [ml]',
          'legend_title': 'Amica',
          'title': "Ablation Volume [ml] for Amica " + flag_subcapsular_title,
          'x_lim': 100,
          'y_lim': 100,
          'lin_reg': 1}
scatter_plot(df_radiomics_amica, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'y_label': 'Effective Ablation Volume [ml] ',
          'legend_title': 'Acculis',
          'title': "Ablation Volume [ml] for Angyiodinamics (Acculis) " + flag_subcapsular_title,
          'x_lim': 100,
          'y_lim': 100,
          'lin_reg': 1}
scatter_plot(df_radiomics_angyodinamics, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'y_label': 'Effective Ablation Volume [ml] for Covidien',
          'legend_title': 'Covidien',
          'title': "Ablation Volume [ml] for Covidien " + flag_subcapsular_title,
          'x_lim': 120,
          'y_lim': 120,
          'lin_reg': 1}
scatter_plot(df_radiomics_covidien, **kwargs)

sys.exit()

try:
    df['Power'] = pd.to_numeric(df['Power'], errors='coerce')
except AttributeError:
    pass
try:
    df['Time_Duration_Applied'] = pd.to_numeric(df['Time_Duration_Applied'], errors='coerce')
except AttributeError:
    pass

# df = df[pd.to_numeric(df['Power'], errors='coerce').notnull()]
kwargs = {'x_data': 'Power', 'y_data': 'Ablation Volume [ml]',
          'title': "Ablation Volume [ml] vs Power for 3 MWA devices. " + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'Ablation Volume [ml]',
          'x_label': 'Time [s]',
          'title': "Ablation Volume [ml] vs Time for 3 MWA devices. " + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Power',
          'title': "Energy [kJ] vs. Power [W] " + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Time_Duration_Applied',
          'y_label': 'Time [s]',
          'title': "Energy [kJ] vs. Time [s] " + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'Tumour Volume [ml]', 'y_data': 'Ablation Volume [ml]',
          'y_label': 'Effective Ablation Volume [ml] for 3 MWA Devices',
          'title': "Tumor Volume [ml] vs Ablation Volume [ml] for 3 MWA devices. " + flag_subcapsular_title,
          'lin_reg': 1}

scatter_plot(df, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'y_label': 'Effective Ablation Volume [ml] for 3 MWA Devices',
          'title': "Ablation Volume [ml] for 3 MWA devices by Number of Chemotherapy cycles Before Ablation. ",
          'lin_reg': 1,
          'size': 'no_chemo_cycle'}
scatter_plot(df, **kwargs)

df['Ratio_AT_vol'] = df['Tumour Volume [ml]'] / df['Ablation Volume [ml]']
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ratio_AT_vol',
          'title': "Tumor to Ablation Volume Ratio for 3 MWA devices.",
          'y_label': 'R(Tumor Volume: Ablation Volume)', 'lin_reg': 1}
scatter_plot(df, **kwargs)

# %% AXES VS ENERGY ALL
# ENERGY
title = "Least Ablation Diameter vs. MWA Energy for 3 MWA Devices."
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'least_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Least Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

title = "Maximum Ablation Diameter vs. MWA Energy for 3 MWA Devices."
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'major_axis_length_ablation',
          'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Maximum Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

title = "Minimum Ablation Diameter vs. MWA Energy for 3 MWA Devices."
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'minor_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Minimum Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

# POWER
title = "Least Ablation Diameter vs. MWA Power."
kwargs = {'x_data': 'Power', 'y_data': 'least_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Least Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

title = "Maximum Ablation Diameter vs. MWA Power."
kwargs = {'x_data': 'Power', 'y_data': 'major_axis_length_ablation',
          'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Maximum Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

title = "Minimum Ablation Diameter vs. MWA Power."
kwargs = {'x_data': 'Power', 'y_data': 'minor_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Minimum Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

#Time_Duration_Applied

title = "Least Ablation Diameter vs. MWA Time."
kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'least_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Least Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

title = "Maximum Ablation Diameter vs. MWA Time."
kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'major_axis_length_ablation',
          'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Maximum Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

title = "Minimum Ablation Diameter vs. MWA Time."
kwargs = {'x_data': 'Time_Duration_Applied', 'y_data': 'minor_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Minimum Ablation Diameter [mm]'}
scatter_plot(df, **kwargs)

# %%
print('3. Dropping Outliers from the Energy Column using val < quantile 0.98')
q = df['Energy [kj]'].quantile(0.98)
df1_no_outliers = df[df['Energy [kj]'] < q]
df1_no_outliers.reset_index(inplace=True, drop=True)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml] (parametrized_formula)',
          'title': "Energy vs Ablation Volume PCA axes for 3 MWA devices. Outliers Removed." + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml] (manufacturers)',
          'title': "Ablation Volumes from Brochure. Outliers Removed. " + flag_subcapsular_title,
          'lin_reg': 1}

scatter_plot(df1_no_outliers, **kwargs)

# kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Tumour Volume [ml]',
#           'title': "Tumors Volumes for 3 MWA devices. Outliers Removed.",
#           'lin_reg': 1}
# scatter_plot(df1_no_outliers, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'title': "Ablation Volumes for 3 MWA devices.Outliers Removed. " + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)

kwargs = {'x_data': 'Tumour Volume [ml]', 'y_data': 'Ablation Volume [ml]',
          'title': "Tumor Volume [ml] vs Ablation Volume [ml]. Outliers Removed. " + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'title': "Ablation Volume [ml] for 3 MWA devices by Number of Chemotherapy cycles Before Ablation. Outliers Removed. ",
          'lin_reg': 1,
          'colormap': 'no_chemo_cycle'}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml] (parametrized_formula)',
          'title': "Ablation Volumes based on the 3 ellipsoid axes for 3 MWA devices. Outliers Removed. ",
          'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)

df1_no_outliers['Ratio_AT_vol'] = df1_no_outliers['Tumour Volume [ml]'] / df1_no_outliers['Ablation Volume [ml]']
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ratio_AT_vol',
          'title': "Tumor to Ablation Volume Ratio for 3 MWA devices.Outliers Removed.",
          'y_label': 'R(Tumor Volume: Ablation Volume)', 'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)

title = "Major Ablation Diameter vs. Least Axis Diameter."
kwargs = {'x_data': 'least_axis_length_ablation', 'y_data': 'major_axis_length_ablation',
          'title': title,
          'lin_reg': 1}
scatter_plot(df1_no_outliers, **kwargs)

title = "Maximum vs. Minimum Ablation Diameter plotted as a function of tumor size."
kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'major_axis_length_ablation',
          'title': title,
          'lin_reg': 1,
          'size': 'Tumour Volume [ml]'}
scatter_plot(df1_no_outliers, **kwargs)

title = "Maximum vs. Minimum Ablation Diameter plotted as a function of energy."
kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'major_axis_length_ablation',
          'title': title,
          'lin_reg': 1,
          'size': 'Energy [kj]'}
scatter_plot(df1_no_outliers, **kwargs)

title = "Maximum vs. 2nd largest Ablation Diameter plotted as a function of energy."
kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'least_axis_length_ablation',
          'title': title,
          'lin_reg': 1,
          'size': 'Energy [kj]'}
scatter_plot(df1_no_outliers, **kwargs)

title = "Maximum vs. 2nd largest Ablation Diameter plotted as a function of tumor size."
kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'least_axis_length_ablation',
          'title': title,
          'lin_reg': 1,
          'size': 'Tumour Volume [ml]'}
scatter_plot(df1_no_outliers, **kwargs)

# %% group by proximity to vessels
scatter_plot_groups(df)

# %% ANGYODINAMICS
fig, ax = plt.subplots()
df_angyodinamics = df1_no_outliers[df1_no_outliers["Device_name"] == "Angyodinamics (Acculis)"]
# df_angyodinamics.dropna(subset=['Energy [kj]'], inplace=True)
# df_angyodinamics.dropna(subset=['least_axis_length_ablation'], inplace=True)


title = "Maximum vs. Minimum Ablation Diameter plotted as a function of tumor size (Angyodinamics)."
kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'major_axis_length_ablation',
          'title': title,
          'lin_reg': 1,
          'size': 'Tumour Volume [ml]'}
scatter_plot(df_angyodinamics, **kwargs)

title = "Maximum vs. Minimum Ablation Diameter plotted as a function of energy (Angyodinamics)."
kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'major_axis_length_ablation',
          'title': title,
          'lin_reg': 1,
          'size': 'Energy [kj]'}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml] (manufacturers)',
          'title': "Ablation Volumes from Brochure for Angiodynamics. " + flag_subcapsular_title, 'lin_reg': 1}

scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'Ablation Volume [ml] (manufacturers)', 'y_data': 'Ablation Volume [ml]',
          'title': "Ablation Volumes from Manufacturer's Brochure vs Resulted Measured Ablation Volume for Angiodynamics. " + flag_subcapsular_title,
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml] (parametrized_formula)',
          'title': "Ablation Volumes PCA axes for Angiodynamics. ", 'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

kwargs = {'x_data': 'Energy [kj]', 'y_data': 'Ablation Volume [ml]',
          'title': "Ablation Volumes for tumors treated with Angiodynamics.",
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)

title = "Least Ablation Diameter vs. MWA Energy for tumors treated with Angiodynamics."
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'least_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Least Ablation Diameter [mm]'}
scatter_plot(df_angyodinamics, **kwargs)

title = "Major Ablation Diameter vs. MWA Energy for tumors treated with Angiodynamics."
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'major_axis_length_ablation',
          'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Maximum Ablation Diameter [mm]'}
scatter_plot(df_angyodinamics, **kwargs)

title = "Minor Ablation Diameter vs. MWA Energy for  tumors treated with Angiodynamics."
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'minor_axis_length_ablation', 'title': title + flag_subcapsular_title,
          'lin_reg': 1,
          'y_label': 'Minimum Ablation Diameter [mm]'}
scatter_plot(df_angyodinamics, **kwargs)

# %% Gray level variance tumor vs energy
# diameter3D_tumor
# Tumour Volume [ml]
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'intensity_mean_tumor',
          'title': 'Energy Applied vs Mean Tumor Pixel Intensity',
          'colormap': 'Tumour Volume [ml]',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'intensity_variance_tumor',
          'title': 'Energy Applied vs Variance Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Energy Applied vs Tumor Pixel Uniformity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'gray_lvl_nonuniformity_tumor',
          'title': 'Energy Applied vs Tumor Pixel NonUniformity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
# %% tumor size vs intensities
kwargs = {'x_data': 'least_axis_length_tumor', 'y_data': 'intensity_mean_tumor',
          'title': 'Least Axis Length Tumor vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'minor_axis_length_tumor', 'y_data': 'intensity_mean_tumor',
          'title': 'Minor Axis Length Tumor vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'major_axis_length_tumor', 'y_data': 'intensity_mean_tumor',
          'title': 'Major Axis Length Tumor vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
# %% gray lvl vs ablation metrics
kwargs = {'x_data': 'least_axis_length_ablation', 'y_data': 'intensity_mean_tumor',
          'title': 'Least Ablation Axis Length vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'least_axis_length_ablation', 'y_data': 'intensity_variance_tumor',
          'title': 'Least Ablation Axis Length vs Variance Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'least_axis_length_ablation', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Least Ablation Axis Length vs Uniformity Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'intensity_variance_tumor',
          'title': 'Minor Axis Length Ablation vs VarianceTumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df_angyodinamics, **kwargs)
kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'intensity_mean_tumor',
          'title': 'Minor Axis Length Ablation vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'minor_axis_length_ablation', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Minor Axis Length Ablation vs Mean Tumor Pixel Intensity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)

kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'intensity_uniformity_tumor',
          'title': 'Major Axis Length Ablation vs Tumor Pixel Uniformity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'intensity_mean_tumor',
          'title': 'Major Axis Length Ablation vs Mean Tumor Pixel',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
kwargs = {'x_data': 'major_axis_length_ablation', 'y_data': 'intensity_variance_tumor',
          'title': 'Major Axis Length Ablation vs Mean Variance Tumor Pixel',
          'lin_reg': 1}

scatter_plot(df, **kwargs)
kwargs = {'x_data': 'Energy [kj]', 'y_data': 'gray_lvl_nonuniformity_tumor',
          'title': 'Energy Applied vs Tumor Pixel NonUniformity',
          'lin_reg': 1}
scatter_plot(df, **kwargs)
# %% percentage distances  histograms
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

# %% percentage distances  histograms for angyodinamics
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
df.hist(column=["major_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram major axis length ablation")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 50]))
plt.ylabel('mm')
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df.hist(column=["least_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram least axis length ablation ")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 50]))

gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df.hist(column=["minor_axis_length_ablation"])
figpathHist = os.path.join("figures", "histogram minor axis length ablation")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 50]))
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

print('All  plots saved as *.png files in dev folder figures')
plt.close('all')
# %% histogram axis tumor
plt.figure()
df.hist(column=["major_axis_length_tumor"])
figpathHist = os.path.join("figures", "histogram major axis length tumor")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 50]))

gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df.hist(column=["least_axis_length_tumor"])
figpathHist = os.path.join("figures", "histogram least axis length tumor")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 50]))

gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

df.hist(column=["minor_axis_length_tumor"])
figpathHist = os.path.join("figures", "histogram minor axis length tumor")
plt.ylabel('mm')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
ax.set_xlim([0, 100])
plt.xlim(([0, 100]))
plt.ylim(([0, 50]))
gh.save(figpathHist, ext=['png'], close=True, width=18, height=16)

plt.close('all')
