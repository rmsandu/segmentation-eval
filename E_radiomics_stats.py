# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from ast import literal_eval
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
plt.style.use('ggplot')
#%%

df = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC.xlsx", sheet_name="radiomics")
# rmv empty rows
# df = pd.DataFrame(np.random.randn(1000, 4), columns = ['a', 'b', 'c', 'd'])
# scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
# df['Tenant'].replace('', np.nan, inplace=True)
df['Energy'].replace('', np.nan, inplace=True)

#%%
# df_ltp = df.dropna(subset=["safety_margin_distribution_0"])
# df_0_5 = df[["safety_margin_distribution: 0<x<=5mm [%]", "LTP"]]
# gp = df.groupby(level=('Coverage_grp'))
# gp.plot.bar()
# plt.show()
#%%
df.dropna(subset=["Ablation volume (ml)"], inplace=True)
# df["Ablation Volume (ml)"].replace('', np.nan, inplace=True)
df1 = df.iloc[:, 21:len(df.columns)].copy()

#%%
# scatter_matrix(df1, figsize = (20, 20), diagonal = 'kde')

df1.plot.scatter(x='Energy', y='Tumour Volume (ml)')
plt.xlabel('Energy (kJ)')
plt.ylabel('Tumor Volume [ml]')
plt.title("Tumors treated with 3 MWA device type")
plt.show()
df1.plot.scatter(x='Energy', y='Ablation volume (ml)')
plt.show()
plt.xlabel("Energy (kJ)")
plt.ylabel("Ablation Volume (ml)")
plt.title("Tumors treated with 3 MWA device type")
plt.show()

#%%
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
# L.get_texts()[0].set_text('make it short')
L_labels = L.get_texts()
for idx, L in enumerate(L_labels):
    # L.set_text(L + 'N=' + str(len(groups[idx])))
    L.set_text(device_name_grp[idx] + ' N='+ str(lesion_per_device[idx]))
plt.xlabel('Energy (kJ)', fontsize=20)
plt.ylabel('Ablation Volume (ml)', fontsize=20)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.show()
#%%
fig, ax = plt.subplots()
lesion_per_device = []
device_name_grp = []
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.Energy, group["Tumour Volume (ml)"], marker='x', linestyle='', ms=12, label=name)
    lesion_per_device.append(len(group))
    device_name_grp.append(name)
L = ax.legend()
# L.get_texts()[0].set_text('make it short')
L_labels = L.get_texts()
for idx, L in enumerate(L_labels):
    # L.set_text(L + 'N=' + str(len(groups[idx])))
    L.set_text(device_name_grp[idx] + ' N='+ str(lesion_per_device[idx]))
plt.xlabel('Energy (kJ)', fontsize=20)
plt.ylabel('Tumor Volume (ml)', fontsize=20)
plt.xlabel('Energy (kJ)', fontsize=20, color='black')
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.grid(True)
plt.show()
#%% ANGYODINAMICS
# plt.figure()
df_angyodinamics = df[df["Device_name"]=="Angyodinamics (Acculis)" ]
print('No of lesions for which angiodinamics was used:', print(len(df_angyodinamics)))
df_angyodinamics.plot.scatter(x='Energy', y='least_axis_length', s=40)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)

plt.xlabel("Energy (kJ)", fontsize=25)
plt.ylabel("Minimum Diameter [mm]", fontsize=25)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.show()

df_angyodinamics.plot.scatter(x='Energy', y='major_axis_length', s=40)
plt.xlabel('Energy (kJ)', fontsize=25)
plt.ylabel("Maximum Diameter [mm]", fontsize=25)
plt.ylim([25, 50])
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.show()
# %% percentage histogram

# plt.figure()
# df1["safety_margin_distribution_0"].replace(0, np.nan, inplace=True)
# df1["safety_margin_distribution_5"].replace(0, np.nan, inplace=True)
# df1["safety_margin_distribution_10"].replace(0, np.nan, inplace=True)
df_margins = df1.iloc[:, len(df1.columns)-3: len(df1.columns)].copy()
df_margins.reset_index(drop=True, inplace=True)
df_margins_sort = pd.DataFrame(np.sort(df_margins.values, axis=0), index=df_margins.index, columns=df_margins.columns)
df_margins_sort.hist( alpha=0.5)
x = df_margins_sort["safety_margin_distribution_0"].tolist()
y = df_margins_sort["safety_margin_distribution_5"].tolist()
z = df_margins_sort["safety_margin_distribution_10"].tolist()
ggg = [x,y,z]
# plt.hist(df_margins_sort["safety_margin_distribution_0"], df_margins_sort["safety_margin_distribution_0"], df_margins_sort["safety_margin_distribution_10"])
#%%
fig, ax = plt.subplots()
for a in ggg:
    sns.distplot(a, bins=range(1, 100, 10), ax=ax, kde=False)
ax.set_xlim([0, 100])
#
# plt.hist([df1['text'],df2['printed']],
#           bins=100, range=(1,100), stacked=True, color = ['r','g'])


# df1.hist(column=["safety_margin_distribution_0"])
# df1.hist(column=["safety_margin_distribution_5"])
# df_angyodinamics.hist(column=["safety_margin_distribution_10"])

#%%

#%%
# df_angyodinamics["Time_Duration_Applied"] = pd.to_numeric(df_angyodinamics["Time_Duration_Applied"])
# df_angyodinamics.plot.scatter(x="Time_Duration_Applied", y="least_axis_length", s=40)
# plt.show()
# df_angyodinamics.plot.scatter(x="Time_Duration_Applied", y="major_axis_length", s=40)
# plt.show()
#%% histogram axes
# plt.figure()
# df_angyodinamics.hist(column=["major_axis_length"])
# plt.xlim([0,60])
# plt.show()
# df_angyodinamics.hist(column=["least_axis_length"])
# plt.show()
#%% tumor vol vs energy
df_angyodinamics.plot.scatter(x='Energy', y='Tumour Volume (ml)', s=40)
plt.xlabel('Energy (kJ)')
plt.ylabel('Tumor Volume [ml]')
plt.title("Tumors treated with Angiodynamics MWA'")
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.show()
df_angyodinamics.plot.scatter(x='Energy', y='Ablation volume (ml)', s=40)
plt.show()
plt.xlabel("Energy (kJ)", fontsize=25)
plt.ylabel("Ablation Volume (ml)", fontsize=25)
plt.title("Tumors treated with Angiodynamics MWA'", fontsize=25)
plt.tick_params(labelsize=20, color='black')
ax.tick_params(colors='black', labelsize=20)
plt.show()

#%%

# sns.pairplot(df1, hue="Device_name")