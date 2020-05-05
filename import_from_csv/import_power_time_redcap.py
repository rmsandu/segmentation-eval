# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
# NOT WORKING!!!!!!!
# TODO: implement power & time reading lesion by lesion
import pandas as pd

file_redcap = r"C:\develop\segmentation-eval\SurveyOfAblationsFor_DATA_LABELS_2020-04-16_1421.xlsx"
file_radiomics = r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_153011-20200313.xlsx"

df_redcap = pd.read_excel(file_redcap)
df_radiomics = pd.read_excel(file_radiomics)
power = []
time_duration_applied = []
needle_insertions = []
# iterate through df_redcap before ablation check the patient id

for index, row in df_radiomics.iterrows():
    patient_id = row['Patient_ID']
    idx_redcap = df_redcap.index[(df_redcap['Patient_ID'] == patient_id)
                                 & (df_redcap['Event Name'] == 'Inclusion')].tolist()
    if idx_redcap:
        try:
            power.append(df_redcap.iloc[idx_redcap]['How many watts?'].tolist()[0])
        except Exception:
            power.append(None)
        try:
            time_duration_applied.append(df_redcap.iloc[idx_redcap]['How many seconds?'].tolist()[0])
        except Exception:
            time_duration_applied.append(None)
        try:
            needle_insertions.append(
                df_redcap.iloc[idx_redcap]['Number of antenna insertions for this lesion'].tolist()[0])
        except Exception:
            needle_insertions.append(None)
    else:
        power.append(None)
        time_duration_applied.append(None)
        needle_insertions.append(None)

df_radiomics['Power'] = power
df_radiomics['Time_Duration_Applied'] = time_duration_applied
df_radiomics['Antenna_Insertions'] = needle_insertions

writer = pd.ExcelWriter('Radiomics_power_time.xlsx')
df_radiomics.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
writer.save()
# df_radiomics['chemo_type'] = chemo_type
# df_radiomics['response_to_chemo'] = response_to_chemo
