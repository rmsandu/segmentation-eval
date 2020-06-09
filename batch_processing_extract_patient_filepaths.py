# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
Example of coding names for the patient data
patient_name = "MAV-M03"
patient_id = "M03"
lesion_id = "MAV-M03-L1"
patient_dob = "19420101"
the idea: loop through the input folder. create a filepath for each patient. add the rest of the variables.
Unique key: key = [Patient_ID, Treatment_ID, Lesion_ID]
TREATMENT_ID can be ablation date
"""

import argparse
import os
from datetime import datetime

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True, help="input patient folder path to be processed")
ap.add_argument("-f", "--input_excel_filename", required=True, help="input excel file patients info")
# input_excel_filename = "Batch_processing_MAVERRIC_1106.xlsx"

args = vars(ap.parse_args())
input_dir = args["input_dir"]
input_excel_filename = args['input_excel_filename']
df = pd.read_excel(input_excel_filename)

df["Patient Name"] = df['Lesion_ID']
df["Patient Name"] = df["Patient Name"].map(lambda x: x.partition("-L")[0])
# df["Date_of_Birth"] = df["Date_of_Birth"].map(lambda x: x[0:9])
# df["Ablation_IR_Date"] = df["Ablation_IR_Date"].map(lambda x: x.split(":")[2])
# df["Ablation_IR_Date"] = df["Ablation_IR_Date"].map(lambda x: x.replace("-", ""))
# df["Ablation_IR_Date"] = df["Ablation_IR_Date"].map(lambda x: x.replace(" ", ""))

# iterate for each patient id from  the excel and look for substring in the list of dir_paths
dir_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
patient_ids = df["Patient_ID"].tolist()
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
filename = os.path.splitext(input_excel_filename)[0]
timestr = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = pd.ExcelWriter(filename + '_' + timestr + '.xlsx')
# add date time to file and message that it has been printed
df.to_excel(writer, index=False, float_format='%.4f')
writer.save()
