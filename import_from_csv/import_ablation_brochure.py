# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import argparse
import sys
from collections import defaultdict
from math import pi
import numpy as np
import pandas as pd

# %%

file_maverric_radiomics = r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_ablation_curated_COPY.xlsx"
file_ablation_devices = r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx"
df = pd.read_excel(file_maverric_radiomics)
df_ablation_devices = pd.read_excel(file_ablation_devices)
df.drop_duplicates(subset=['Lesion_ID'], inplace=True)
print(len(df))


# df['Ablation Volume [ml]_brochure'] = (pi * df['least_axis_length_ablation'] *
#                                                      df['major_axis_length_ablation'] * df[
#                                                          'minor_axis_length_ablation']) / 6000

dd = defaultdict(list)
dict_devices = df_ablation_devices.to_dict('records', into=dd)
ablation_radii = []
for index, row in df.iterrows():
    power = row['Power']
    time = row['Time_Duration_Applied']
    device = row['Device_name']
    flag = False
    if power != np.nan and time != np.nan:
        for item in dict_devices:
            if item['Power'] == power and item['Device_name'] == device and item['Time_Duration_Applied'] == time:
                ablation_radii.append(item['Radii'])
                flag = True
        if flag is False:
            ablation_radii.append('0 0 0')

df['Ablation_Radii_Brochure'] = ablation_radii
df['major_axis_ablation_brochure'] = pd.to_numeric(df['Ablation_Radii_Brochure'].apply(lambda x: x.split()[0]))
df['minor_axis_ablation_brochure'] = pd.to_numeric(df['Ablation_Radii_Brochure'].apply(lambda x: x.split()[1]))
df['least_axis_ablation_brochure'] = pd.to_numeric(df['Ablation_Radii_Brochure'].apply(lambda x: x.split()[2]))
df['Ablation Volume [ml] (manufacturers)'] = 4 * pi * (df['major_axis_ablation_brochure'] *
                                                       df['minor_axis_ablation_brochure'] *
                                                       df['least_axis_ablation_brochure']) / 3000
df['Ablation Volume [ml] (manufacturers)'].replace(0, np.nan, inplace=True)
df['minor_axis_ablation_brochure'].replace(0, np.nan, inplace=True)
df['least_axis_ablation_brochure'].replace(0, np.nan, inplace=True)
df['major_axis_ablation_brochure'].replace(0, np.nan, inplace=True)
df_ablation_devices['Energy_brochure'] = df_ablation_devices['Power'] * df_ablation_devices['Time_Duration_Applied'] / 1000
df_ablation_devices['Predicted Ablation Volume (ml)'] = 4 * pi * (df_ablation_devices['major_axis_ablation_brochure'] *
                                                       df_ablation_devices['minor_axis_ablation_brochure'] *
                                                       df_ablation_devices['least_axis_ablation_brochure']) / 3000
#  write to Excel
writer = pd.ExcelWriter(file_ablation_devices)
df_ablation_devices.to_excel(writer,  index=False, float_format='%.4f')
writer.save()

writer = pd.ExcelWriter(file_maverric_radiomics)
df.to_excel(writer, index=False, float_format='%.4f')
writer.save()
