# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import pandas as pd
from ast import literal_eval



lit_file = r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_153011-20200313_.xlsx"
redcap_file = r"C:\develop\segmentation-eval\SurveyOfAblationsFor_DATA_LABELS_2020-04-03_1637.xlsx"

df_lit = pd.read_excel(lit_file)
df_redcap = pd.read_excel(redcap_file)

# %%
df_redcap.drop_duplicates(subset=["Patient_ID"], inplace=True, keep='first')
df_redcap.reset_index(drop=True, inplace=True)
nr_lesions_ablated_cochlea = df_lit["Nr_Lesions_Ablated"].tolist()
nr_lesions_ablated_redcap = df_redcap["Number of ablated lesions"].tolist()
# df_lit["Ablation_IR_Date"] = df_lit["Ablation_IR_Date"].apply(literal_eval)
df_lit["Ablation_IR_Date"] = df_lit["Ablation_IR_Date"].map(lambda x: str(x))
# # datetime.datetime.strptime("2013-1-25", '%Y-%m-%d').strftime('%y%m%d')
# df_lit["Ablation_IR_Date"] = df_lit["Ablation_IR_Date"].map(lambda x: x.replace("-", ""))
# df_lit["Ablation_IR_Date"] = df_lit["Ablation_IR_Date"].map(lambda x: x.replace(" ", ""))
# df_redcap["Date of ablation"] = df_redcap["Date of ablation"].apply(lambda x: x.strftime('%Y-%m-%d'))
# df_redcap["Date of ablation"] = df_redcap["Date of ablation"].map(lambda x: x.replace("-", ""))
# df_redcap["Date of ablation"] = df_redcap["Date of ablation"].map(
#     lambda x: datetime.datetime.strptime(x, '%Y-%m-%Y').strftime('%y%m%d'))

patient_id_cochlea = df_lit["Patient_ID"].tolist()
patient_id_redcap = df_redcap["Patient_ID"].tolist()

for idx, pat in enumerate(patient_id_redcap):
    lesion_redcap = nr_lesions_ablated_redcap[idx]
    # idx_cochlea = df_lit.index[df_lit["Patient_ID"] == pat].tolist()
    df_patient = df_lit[df_lit["Patient_ID"] == pat]
    if not df_patient.empty:
        if len(df_patient) != lesion_redcap:
            # if df_lit.iloc[idx_cochlea].Nr_Lesions_Ablated.values != lesion_redcap:
            print('the number of lesions is different from redcap to cochlea database for this patient: ', pat)
            print("no of lesions in cochlea lit db:", len(df_patient))
            print("no of lesions in redcap file:", lesion_redcap)

        # date_ablation_redcap = df_redcap.iloc[idx]["Date of ablation"]
        # date_ablation_lit = df_patient.iloc[0]["Ablation_IR_Date"]
        #
        # if date_ablation_lit != date_ablation_redcap:
        #     pass
        #     print("ablation ir date is different for patient ", pat)
        #     print("ablation date cochlea lit: ", date_ablation_lit)
        #     print("ablation date redcap: ", date_ablation_redcap)

    else:
        print("patient not found in cochlea lit db: ", pat)

    # find the patient in the cochlea list
    # find the nr_ablated lesions and send error message if it doesn't match
