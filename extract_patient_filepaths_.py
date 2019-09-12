# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os
import argparse
import pandas as pd
from splitAllPaths import splitall

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True, help="input patient folder path to be processed")
ap.add_argument("-f", "--input_csv", required=True, help="input excel file patients info")

args = vars(ap.parse_args())
input_dir = args["input_dir"]
input_csv = args['input_csv']
df = pd.read_excel(input_csv)

# what do we need to extract from the CSV. 1) Patient Name  2) Patient ID 3) Date-of-birth 4) filepath
# the idea: loop through the input folder. create a filepath for each patient. add the rest of the variables.
# unique list of all the patients

patient_name = "MAV-M03"   # example
patient_id = "M03"
lesion_id = "MAV-M03-L1"
patient_dob = "19420101"

# df.drop_duplicates("Patient ID", inplace=True)
# rename patient name to patient id
# unique_df.rename(columns={'Patient Name': 'Patient ID'})
# df['a'] = df['a'].apply(lambda x: x + 1)
# map(lambda x: x.split('-')[0:1])
df["Patient Name"] = df['Lesion id']
df["Patient Name"] = df["Patient Name"].map(lambda x: x.partition("-L")[0])
df["Date-of-Birth"] = df["Date-of-Birth"].map(lambda x: str(x)+"0101")

# iterate for each patient id from  the excel and look for substring in the list of dir_paths
dir_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
patient_ids = df["Patient ID"].tolist()
path_patient_dir_col = []
for patient_id in patient_ids:
    patient_paths = []
    for idx, el in enumerate(dir_paths):
        # what if we have more than one patient folder matching the id
        if patient_id in el:
            pat_path = dir_paths[idx]
            patient_paths.append(pat_path)  # in case there are several patient folders matching the same ID
    if patient_paths:
        path_patient_dir_col.append(patient_paths)
    else:
        path_patient_dir_col.append(None)

df["Patient_Dir_Paths"] = path_patient_dir_col
df.reset_index(drop=True)
writer = pd.ExcelWriter("Batch_processing_MAVERRIC_.xlsx")
df.to_excel(writer, index=False, float_format='%.4f')
writer.save()


