# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:01:48 2018

@author: Raluca Sandu
"""
import os
import time
import pandas as pd
import II_Resize_Resample_Images as ReaderWriterClass
#%%
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


#%%
# for GT 2017
dictionary_filepaths = [] # list of dictionaries containing the filepaths of the segmentations
rootdir = r"C:\PatientDatasets_GroundTruth_Database\GroundTruthDB_2017"
# TODO: change rootdir to local directory
for subdir, dirs, files in os.walk(rootdir):
     tumorFilePath  = ''
     ablationSegm = ''
     for file in files:
         if file == "tumor_segmentation":
             FilePathName = os.path.join(subdir, file)
             tumorFilePath = os.path.normpath(FilePathName)
         elif file == "ablation_segmentation":
             FilePathName = os.path.join(subdir, file)
             ablationFilePath = os.path.normpath(FilePathName)
         else:
             print("")
         # TODO: create csv with filepath for tumor(s), ablations, trajectories, plan, validation , xml (?)
     if tumorFilePath and ablationFilePath:
         dir_name = os.path.dirname(ablationFilePath)
         dirname2 = os.path.split(dir_name)[1]
         data = {'PatientName' : dirname2,
                 'NeedleNr': 1,
                 ' Tumour Segmentation Path' : tumorFilePath,
                 ' Ablation Segmentation Path': ablationFilePath
                 }
         dictionary_filepaths.append(data)
#%%
# convert to data frame and output filepaths to excel
df_filepaths = pd.DataFrame(dictionary_filepaths)
timestr = time.strftime("%H%M%S-%Y%m%d")
filename = 'FilepathsGTSegmentations2017' + '_' + timestr + '.xlsx'
filepathExcel = os.path.join(rootdir, filename)
writer = pd.ExcelWriter(filepathExcel)
df_filepaths.to_excel(writer, index=False)
# TODO : add tumor type manually
print("success")
#Metrics.main_distance_volume_metrics(df_filepaths, rootdir)
