# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

file_redcap = r"C:\develop\segmentation-eval\redcap_file_all.xlsx"
file_radiomics = r"C:\develop\segmentation-eval\Radiomics_Radii_MAVERRIC.xlsx"

df_redcap = pd.read_excel(file_redcap)
df_radiomics = pd.read_excel(file_radiomics)
chemo_before_ablation = []
no_chemo_cycle = []
chemo_type = []
response_to_chemo = []
# iterate through df_redcap before ablation check the patient id

for index, row in df_radiomics.iterrows():
    patient_id = row['Patient_ID']
    idx_redcap = df_redcap.index[(df_redcap['Patient_ID'] == patient_id)
                                 & (df_redcap['Event Name'] == 'Inclusion')].tolist()
    if idx_redcap:
        chemo_before_ablation.append(df_redcap.iloc[idx_redcap]['Chemotherapy before ablation?'].tolist()[0])
        no_chemo_cycle.append(df_redcap.iloc[idx_redcap]['Number of chemo cycles'].tolist()[0])
        chemo_type.append(df_redcap.iloc[idx_redcap]['Type of chemo'])
        response_to_chemo.append(['response_to_chemo'])
    else:
        chemo_before_ablation.append(None)
        no_chemo_cycle.append(None)
        chemo_type.append(None)
        response_to_chemo.append(None)

df_radiomics['chemo_before_ablation'] = chemo_before_ablation
df_radiomics['no_chemo_cycle'] = no_chemo_cycle
df_radiomics['no_chemo_cycle'].replace(np.nan, 0, inplace=True)

writer = pd.ExcelWriter('Radiomics_Radii_Chemo_MAVERRIC.xlsx')
df_radiomics.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
writer.save()
# df_radiomics['chemo_type'] = chemo_type
# df_radiomics['response_to_chemo'] = response_to_chemo

