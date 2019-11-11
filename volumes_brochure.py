# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import pandas as pd
from collections import defaultdict
from math import pi
import numpy as np

file_ablation = r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx"
file_maverric = r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_111119.xlsx"
df = pd.read_excel(file_ablation)

dd = defaultdict(list)
dict_devices = df.to_dict('records', into=dd)
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
df['Ablation Volume [ml]_brochure'] = 4 * pi * (df['major_axis_ablation_brochure'] *
                                                       df['minor_axis_ablation_brochure'] *
                                                       df['least_axis_ablation_brochure']) / 3000
df['Ablation Volume [ml]_brochure'].replace(0, np.nan, inplace=True)
df['Energy_brochure'] = df['Power'] * df['Time_Duration_Applied'] / 1000
writer = pd.ExcelWriter(file_ablation)
df.to_excel(writer,  index=False, float_format='%.4f')
writer.save()

