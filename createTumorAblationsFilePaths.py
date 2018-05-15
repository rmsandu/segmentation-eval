# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:01:48 2018

@author: Raluca Sandu
"""
import os
import pandas as pd

#%%
segmentation_data = [] # list of dictionaries containing the filepaths of the segmentations
rootdir = "Z:/Public/Raluca_Radek/GroundTruth_2017/"
rootdir = "C:/develop/data/"
#%%
for subdir, dirs, files in os.walk(rootdir):
    tumorFilePath  = ''
    ablationSegm = ''
    for file in files:
        if file == "tumorSegm":
            FilePathName = os.path.join(subdir, file)
            tumorFilePath = os.path.normpath(FilePathName)
        elif file == "ablationSegm":
            FilePathName = os.path.join(subdir, file)
            ablationFilePath = os.path.normpath(FilePathName)
        else:
            print("")
        
    if (tumorFilePath) and (ablationFilePath):
        dir_name = os.path.dirname(ablationFilePath)
        dirname2 = os.path.split(dir_name)[1]
        data = {'PatientName' : dirname2,
                'TumorFile' : tumorFilePath,
                'AblationFile' : ablationFilePath
                }
        segmentation_data.append(data)  

# convert to data frame 
df_patientdata = pd.DataFrame(segmentation_data)
#%%
#path, dirs, files = os.walk(rootdir).__next__()
counter = 0
segmentation_data = [] 
for path, dirs, files in os.walk(rootdir):
    tumorFilePath = ''
    ablationFilePath = ''
    for subdir in dirs:
        if "Ablation" in subdir:
            ablationFilePath = os.path.join(path,subdir)
#            ablationFilePath = os.path.normpath(ablationFilePath)
        if "Tumor" in subdir:
            tumorFilePath = os.path.join(path,subdir)
#            tumorFilePath = os.path.normpath(tumorFilePath)
#            
    if (tumorFilePath) and (ablationFilePath):
        counter += 1

        data = {'PatientName' : counter,
                'TumorFile' : tumorFilePath,
                'AblationFile' : ablationFilePath
                }
        segmentation_data.append(data)  

# convert to data frame 
df_patientdata = pd.DataFrame(segmentation_data)    