# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:01:48 2018

@author: Raluca Sandu
"""
import os
import pandas as pd
import mainDistanceVolumeMetrics

#%%
# rootdir = "Z:/Public/Raluca_Radek/GroundTruth_2017/"

rootdir = r"C:\develop\data"
counter = 0
# list of dictionaries containing the filepaths of the segmentations
FilepathsDictionary = [] 
for path, dirs, files in os.walk(rootdir):
    tumorFilePath = ''
    ablationFilePath = ''
    for subdir in dirs:
        if "Ablation" in subdir:
            ablationFilePath = os.path.join(path, subdir)
            ablationFilePath = os.path.normpath(ablationFilePath)
        if "Tumor" in subdir:
            tumorFilePath = os.path.join(path, subdir)
            tumorFilePath = os.path.normpath(tumorFilePath)
#            
    if tumorFilePath and ablationFilePath:
        counter += 1

        data = {'PatientName' : counter,
                'TumorFile' : tumorFilePath,
                'AblationFile' : ablationFilePath
                }
        FilepathsDictionary.append(data)  

# convert the filepaths to data frame 
df_patientdata = pd.DataFrame(FilepathsDictionary)
# call the function for parsing distances
mainDistanceVolumeMetrics.main_distance_metrics(df_patientdata, rootdir)
