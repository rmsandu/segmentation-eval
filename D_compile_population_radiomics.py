# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import sys
import pandas as pd
import argparse
from ast import literal_eval

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--rootdir", required=True,
                    help="path to the patient folder with Radiomics CSV to be processed")
    ap.add_argument("-o", "--output_dir", required=True, help="path to the output file with radiomics")
    ap.add_argument("-b", "--input_batch_proc_paths", required=True, help="input csv file for batch processing")

    args = vars(ap.parse_args())

    if args["rootdir"] is not None:
        print("Path to folder with Radiomics CSVs for each patient: ", args["rootdir"])
        print(args["output_dir"])
    if (args["input_batch_proc_paths"]) is not None:
        print("Path to CSV that has directory paths and subcapsular lesion info: ", args["input_batch_proc_paths"])
    print("path to output directory", args["output_dir"])

    df_download_db_all_info = pd.read_excel(args["input_batch_proc_paths"])
    frames = []  # list to store all df per lesion.

    for subdir, dirs, files in os.walk(args["rootdir"]):
        for file in sorted(files):
            if file.endswith('.xlsx'):
                # check file extension is xlsx
                excel_input_file_per_lesion = os.path.join(subdir, file)
                df_single_lesion = pd.read_excel(excel_input_file_per_lesion)
                df_single_lesion.rename(columns={'lesion_id': 'Lesion_ID', 'patient_id': 'Patient_ID'}, inplace=True)
                # MAV - B01 - L1
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
print(len(frames))
result = pd.concat(frames, ignore_index=True)
df_final = pd.merge(df_download_db_all_info, result, how="outer", on=['Patient_ID', 'Lesion_ID'])
# TODO: write treatment id as well. the unique key must be formed out of: [patient_id, treatment_id, lesion_id]
filepath_excel = os.path.join(args["output_dir"], "Radiomics_MAVERRIC_ECALSS.xlsx")
writer = pd.ExcelWriter(filepath_excel)
df_final.to_excel(writer, sheet_name='radiomics', index=False, float_format='%.4f')
writer.save()
