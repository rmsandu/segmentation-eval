# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""


import numpy as np
import pandas as pd
file_redcap = r"C:\develop\segmentation-eval\redcap_file_all.xlsx"
file_radiomics = r"C:\develop\segmentation-eval\Radiomics_Radii_Chemo_MAVERRIC.xlsx"

# Number of completely ablated lesions
# Number of incomplete ablations

df_redcap = pd.read_excel(file_redcap)
df_radiomics = pd.read_excel(file_radiomics)
patient_ids = df_radiomics.Patient_ID.unique().tolist()
LTP = []
no_lesions_total = []

for idx, patient_id in enumerate(patient_ids):
    df_patient_redcap = df_radiomics[df_radiomics['Patient_ID']==patient_id]
    df_patient_redcap.reset_index(inplace=True, drop=True)
    idx_pat_radiomics = df_radiomics.index[df_radiomics['Patient_ID'] == patient_id]
    no_of_lesions = df_radiomics.iloc[idx_pat_radiomics[0]].Nr_Lesions_Ablated
    if no_of_lesions != len(df_patient_redcap):
        print('no of rows diffferent from the number of lesions ablated for this patient:', patient_id)

# iterate through df_redcap before ablation check the patient id
for idx, patient_id in enumerate(patient_ids):
    df_patient_redcap = df_redcap[df_redcap['Patient_ID']==patient_id]
    df_patient_redcap.reset_index(inplace=True, drop=True)
    idx_pat_radiomics = df_radiomics.index[df_radiomics['Patient_ID'] == patient_id]
    no_of_lesions = df_radiomics.iloc[idx_pat_radiomics[0]].Nr_Lesions_Ablated
    no_lesions_total.append(no_of_lesions)
    for index, row in df_patient_redcap.iterrows():
        if not np.isnan(row['Number of completely ablated lesions']):
            nr_completely_ablated = row['Number of completely ablated lesions']
        if not np.isnan(row['Number of incomplete ablations']):
            nr_incompletely_ablated = row['Number of incomplete ablations']

    if nr_completely_ablated == no_of_lesions:
        for i in range(0, int(nr_completely_ablated)):
            LTP.append(True)
    elif nr_incompletely_ablated == no_of_lesions:
        for i in range(0,  int(nr_incompletely_ablated)):
            LTP.append(False)
    else:
        print('this patient must be updted for LTP:', patient_id)
        for i in range(0, int(no_of_lesions)):
            LTP.append(None)

print(np.array(no_lesions_total).sum())
df_radiomics['LTP'] = LTP

writer = pd.ExcelWriter('Radiomics_Radii_Chemo_LTP_MAVERRIC.xlsx')
df_radiomics.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
writer.save()
