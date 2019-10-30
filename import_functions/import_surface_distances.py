# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools


df = pd.read_excel("C:\develop\segmentation-eval\Radiomics_Radii_Chemo_LTP_ECALSS.xlsx")
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


#%% TODO: write treatment id as well. the unique key must be formed out of: [patient_id, treatment_id, lesion_id]
filepath_excel = os.path.join(outdir, "Radiomics_Radii_Chemo_LTP_Distances_ECALSS.xlsx")
writer = pd.ExcelWriter(filepath_excel)
df_final.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
writer.save()