# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import graphing as gh

sns.set(style="ticks")
plt.style.use('ggplot')
# %%

df = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC.xlsx", sheet_name="radiomics")
# rmv empty rows
# df = pd.DataFrame(np.random.randn(1000, 4), columns = ['a', 'b', 'c', 'd'])
# scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
# df['Tenant'].replace('', np.nan, inplace=True)
df['Energy'].replace('', np.nan, inplace=True)
df['MISSING'].replace('', np.nan, inplace=True)
df = df[df['Device_name'] != 'Boston Scientific (Boston Scientific - RF 3000)']
df.dropna(subset=["Ablation volume (ml)"], inplace=True)
# df["Ablation Volume (ml)"].replace('', np.nan, inplace=True)
idx_comments = df.columns.get_loc('Device_name')
df1 = df.iloc[:, idx_comments:len(df.columns)].copy()

# %%
# scatter_matrix(df1, figsize = (20, 20), diagonal = 'kde')
fig, ax = plt.subplots()
df1.plot.scatter(x='Energy', y='Tumour Volume (ml)')
plt.xlabel('Energy [kJ]', fontsize=20)
plt.ylabel('Tumor Volume [ml]', fontsize=20)
plt.title("Tumors treated with 3 MWA device type", fontsize=20)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
figpathHist = "Tumors treated with 3 MWA devices. Tumor Vol vs Energy" + '.png'
gh.save(figpathHist, width=18, height=16)

fig, ax = plt.subplots()
df1.plot.scatter(x='Energy', y='Ablation volume (ml)')
plt.xlabel("Energy [kJ]", fontsize=20)
plt.ylabel("Ablation Volume [ml]", fontsize=20)
plt.title("Tumors treated with 3 MWA devices type. Ablation Volume vs  Energy Distribution", fontsize=20)
figpathHist = "Tumors treated with 3 MWA devices. Ablation Vol vs Energy" + '.png'
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, width=18, height=16)
# %%
groups = df1.groupby('Device_name')
fig, ax = plt.subplots()
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
lesion_per_device = []
device_name_grp = []
for name, group in groups:
    ax.plot(group.Energy, group["Ablation volume (ml)"], marker='x', linestyle='', ms=12, label=name)
    lesion_per_device.append(len(group))
    device_name_grp.append(name)
L = ax.legend()
L_labels = L.get_texts()

for idx, L in enumerate(L_labels):
    L.set_text(device_name_grp[idx] + ' N=' + str(lesion_per_device[idx]))

plt.xlabel('Energy [kJ]', fontsize=20, color='black')
plt.ylabel('Ablation Volume [ml]', fontsize=20, color='black')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
figpathHist = "Ablation Volume vs  Energy per MWA Device Category" + '.png'
gh.save(figpathHist, width=18, height=16)
# %%
fig, ax = plt.subplots()
lesion_per_device = []
device_name_grp = []
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.Energy, group["Tumour Volume (ml)"], marker='x', linestyle='', ms=12, label=name)
    lesion_per_device.append(len(group))
    device_name_grp.append(name)
L = ax.legend()
L_labels = L.get_texts()
for idx, L in enumerate(L_labels):
    L.set_text(device_name_grp[idx] + ' N=' + str(lesion_per_device[idx]))
plt.ylabel('Tumor Volume [ml]', fontsize=20, color='black')
plt.xlabel('Energy [kJ]', fontsize=20, color='black')
plt.legend(fontsize=20)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
figpathHist = "Tumor Volume vs  Energy per MWA Device Category" + '.png'
gh.save(figpathHist, width=18, height=16)

# %% ANGYODINAMICS
fig, ax = plt.subplots()
df_angyodinamics = df[df["Device_name"] == "Angyodinamics (Acculis)"]
df_angyodinamics.plot.scatter(x='Energy', y='least_axis_length', s=40)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.xlabel("Energy [kJ]", fontsize=20)
plt.ylabel("Minimum Diameter (PCA-based ellipsoid approximation) [mm]", fontsize=20)
plt.title("Minimum Ablation Diameter vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device", fontsize=20)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
figpathHist = "Minimum Ablation Diameter vs. MWA Energy_PCA based ellipsoid approximation" + '.png'
gh.save(figpathHist, width=18, height=16)

fig, ax = plt.subplots()
df_angyodinamics.plot.scatter(x='Energy', y='major_axis_length', s=40)
plt.xlabel('Energy [kJ]', fontsize=20)
plt.ylabel("Maximum Diameter (PCA-based ellipsoid approximation) [mm]", fontsize=20)
plt.ylim([25, 50])
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.title("Maximum Ablation Diameter vs. MWA Energy for " + str(
    len(df_angyodinamics)) + " tumors treated with Angyodinamics MWA Device", fontsize=20)

figpathHist = "Maximum Ablation Diameter vs. MWA Energy_PCA based ellipsoid approximation" + '.png'
gh.save(figpathHist, width=18, height=16)

# %% percentage distances  histograms
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
          {'Ablation Surface Margin ' + r'$0 \leq  x \leq 5$' + 'mm'},{'Ablation Surface Margin ' + r'$x < 0$' + 'mm'}]
for idx, col in enumerate(df_margins.columns):
    sns.distplot(df_margins[col], label=labels[idx],
                 bins=range(0, 101, 10),
                 kde=False, hist_kws=dict(edgecolor='black'))

plt.xlabel('Percentage of Surface Margin Covered for different ablation margins ranges', fontsize=20, color='black')
plt.ylabel('Frequency', fontsize=20, color='black')
plt.legend(fontsize=20)
plt.xticks(range(0, 101, 10))
figpathHist = "surface margin frequency percentages" + '.png'
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
gh.save(figpathHist, width=18, height=16)

# %%
df_angyodinamics["Time_Duration_Applied"] = pd.to_numeric(df_angyodinamics["Time_Duration_Applied"])
df_angyodinamics.plot.scatter(x="Time_Duration_Applied", y="least_axis_length", s=40)
plt.ylabel('Minimum Diameter (PCA-based) [mm]', fontsize=20)
plt.xlabel('Time Duration MWA [s]', fontsize=20)
figpathHist = "scatter Time Duration Applied vs least axis length" + '.png'
gh.save(figpathHist, width=18, height=16)

df_angyodinamics.plot.scatter(x="Time_Duration_Applied", y="major_axis_length", s=40)
plt.ylabel('Maximum Diameter (PCA-based) [mm]', fontsize=20)
plt.xlabel('Time Duration MWA [s]', fontsize=20)
figpathHist = "scatter Time Duration Applied vs major axis length" + '.png'
gh.save(figpathHist, width=18, height=16)



# %% histogram axes
plt.figure()
df_angyodinamics.hist(column=["major_axis_length"])
plt.xlim([0,60])
plt.show()
df_angyodinamics.hist(column=["least_axis_length"])
plt.show()
# %% tumor vol vs energy

plt.close('all')
# %%

# sns.pairplot(df1, hue="Device_name")
