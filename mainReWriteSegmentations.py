# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:01:48 2018

@author: Raluca Sandu
"""
import os
import csv
import pandas as pd
import resizeDICOMSegmentations as ReaderWriterClass
#%%
# rootdir = "Z:/Public/Raluca_Radek/GroundTruth_2017/"

# rootdir = r"C:\develop\data"
rootdir = r"Z:\Public\Raluca_Radek\GroundTruth_2018\GT_23042018"
patientID = 0
# list of dictionaries containing the filepaths of the segmentations
result = {}

for path, dirs, files in os.walk(rootdir):
    tumorFilePath = ''
    ablationFilePath = ''

    for file in files:
        fileName, fileExtension = os.path.splitext(file)
        if fileExtension.endswith('.csv') and 'filepaths' in fileName.upper().lower():
            patientID +=1
            filepath_csv = os.path.normpath(os.path.join(path, file))
            reader = csv.DictReader(open(filepath_csv))
            for row in reader:
                for column, value in row.items():
                    if column != 'TrajectoryID':
                        file_value = path + value
                    else:
                        file_value = value
                    result.setdefault(column, []).append(file_value)
                # add patient ID column
                result.setdefault('PatientID', []).append(patientID)
                    
df_filepaths = pd.DataFrame(result)
resize_object = ReaderWriterClass.ResizedSegmentations(df_filepaths)
resize_object.save_images_to_disk()



