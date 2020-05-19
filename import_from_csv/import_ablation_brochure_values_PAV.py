# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

from math import pi

import numpy as np
import pandas as pd

# %%

file_ablation_devices = r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx"
df = pd.read_excel(file_ablation_devices)


df['minor_axis_ablation_brochure'].replace(0, np.nan, inplace=True)
df['least_axis_ablation_brochure'].replace(0, np.nan, inplace=True)
df['major_axis_ablation_brochure'].replace(0, np.nan, inplace=True)
df['Energy_brochure(kJ)'] = df['Power'] * df['Time_Duration_Applied'] / 1000

df['Predicted_Ablation_Volume'] = 4 * pi * (df['major_axis_ablation_brochure'] *
                                                df['minor_axis_ablation_brochure'] *
                                                df['least_axis_ablation_brochure']) / 3000

#  write to Excel
writer = pd.ExcelWriter(file_ablation_devices)
df.to_excel(writer, index=False, float_format='%.4f')
writer.save()
print('Excel File Updated with Energy and PAV')

# %% OLD CODE for calculating the PAV dirrectly in the radiomics file
# dd = defaultdict(list)
# dict_devices = df.to_dict('records', into=dd)
# ablation_radii = []
# for index, row in df.iterrows():
#     power = row['Power']
#     time = row['Time_Duration_Applied']
#     device = row['Device_name']
#     flag = False
#     if power != np.nan and time != np.nan:
#         for item in dict_devices:
#             if item['Power'] == power and item['Device_name'] == device and item['Time_Duration_Applied'] == time:
#                 ablation_radii.append(item['Radii'])
#                 flag = True
#         if flag is False:
#             ablation_radii.append('0 0 0')

# df['Ablation_Radii_Brochure'] = ablation_radii
# df['major_axis_ablation_brochure'] = pd.to_numeric(df['Ablation_Radii_Brochure'].apply(lambda x: x.split()[0]))
# df['minor_axis_ablation_brochure'] = pd.to_numeric(df['Ablation_Radii_Brochure'].apply(lambda x: x.split()[1]))
# df['least_axis_ablation_brochure'] = pd.to_numeric(df['Ablation_Radii_Brochure'].apply(lambda x: x.split()[2]))
