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
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_file", required=True, help="input file pooled radiomics ")
ap.add_argument("-a", "--ablation_devices_brochure", required=False, help="input file ablation device brochure ")
args = vars(ap.parse_args())

df = pd.read_excel(args["input_file"], sheet_name="radiomics")
try:
    df_ablation_devices = pd.read_excel(args["ablation_devices_brochure"])
except Exception:
    print('file with ablation devices info brochure not provided')
    sys.exit()

df['Ablation Volume [ml] (parametrized_formula)'] = (pi * df['least_axis_length_ablation'] *
                                                     df['major_axis_length_ablation'] * df[
                                                         'minor_axis_length_ablation']) / 6000

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
#  write to Excel
writer = pd.ExcelWriter(args["input_file"])
df.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
writer.save()
