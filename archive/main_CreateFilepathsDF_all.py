# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:01:48 2018

@author: Raluca Sandu
"""
import os
import pandas as pd
import ResizeSegmentationsMain as ReaderWriterClass
import mainDistanceVolumeMetrics as Metrics
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
# for Spheres
rootdir = r"C:\develop\data\new_shift"
filepaths = []
tumorFilePath = ''
ablationFilePath = ''
typeSphere = ''
patient_name = ''

for path, dirs, files in os.walk(rootdir):
    for file in files:
        fileName, fileExtension = os.path.splitext(file)
        if 'Mask' in fileName:
            allparts = splitall(path)
            if 'ablation' in path:
                ablationFilePath = os.path.normpath((os.path.join(path, file)))
                typeSphere = allparts[5]
                patient_name = allparts[4]
            if 'lession' in path:
                tumorFilePath = os.path.normpath((os.path.join(path, file)))
                typeSphere = allparts[5]
                patient_name = allparts[4]

    if tumorFilePath and ablationFilePath:
        allparts = splitall(path)
        data = {'PatientName': patient_name,
                'Type': typeSphere,
                ' Tumour Segmentation Path': tumorFilePath,
                ' Ablation Segmentation Path': ablationFilePath
                }
        tumorFilePath = ''
        ablationFilePath = ''
        filepaths.append(data)

df_filepaths = pd.DataFrame(filepaths)
Metrics.main_distance_volume_metrics(df_filepaths, rootdir)
#%%
# for GT 2017

# dictionary_filepaths = [] # list of dictionaries containing the filepaths of the segmentations
# rootdir = "Z:/Public/Raluca&Radek/studyPatientsMasks/GroundTruthDB_ROI/"
#
# for subdir, dirs, files in os.walk(rootdir):
#     tumorFilePath  = ''
#     ablationSegm = ''
#     for file in files:
#         if file == "tumorSegm":
#             FilePathName = os.path.join(subdir, file)
#             tumorFilePath = os.path.normpath(FilePathName)
#         elif file == "ablationSegm":
#             FilePathName = os.path.join(subdir, file)
#             ablationFilePath = os.path.normpath(FilePathName)
#         else:
#             print("")
#         # TODO: create csv with filepath for tumor(s), ablations, trajectories, plan, validation , xml (?)
#     if tumorFilePath and ablationFilePath:
#         dir_name = os.path.dirname(ablationFilePath)
#         dirname2 = os.path.split(dir_name)[1]
#         data = {'PatientName' : dirname2,
#                 ' Tumour Segmentation Path' : tumorFilePath,
#                 ' Ablation Segmentation Path': ablationFilePath
#                 }
#         dictionary_filepaths.append(data)
#
# # convert to data frame
# df_filepaths = pd.DataFrame(dictionary_filepaths)
# Metrics.main_distance_volume_metrics(df_filepaths, rootdir)
# #%%
# # rootdir = r"C:\develop\data"
# rootdir = r"Z:\Public\Raluca_Radek\GroundTruth_2018\GT_23042018"
# patientID = 0
# # list of dictionaries containing the filepaths of the segmentations
# dictionary_filepaths = {}
#
# for path, dirs, files in os.walk(rootdir):
#     tumorFilePath = ''
#     ablationFilePath = ''
#
#     for file in files:
#         fileName, fileExtension = os.path.splitext(file)
#         if fileExtension.endswith('.csv') and 'filepaths' in fileName.upper().lower():
#             patientID +=1
#             filepath_csv = os.path.normpath(os.path.join(path, file))
#             reader = csv.DictReader(open(filepath_csv))
#             for row in reader:
#                 for column, value in row.items():
#                     if column != 'NeedleNr':
#                         file_value = path + value
#                     else:
#                         file_value = value
#                     dictionary_filepaths.setdefault(column, []).append(file_value)
#                 # add patient ID column
#                 dictionary_filepaths.setdefault('PatientID', []).append(patientID)
#
# df_filepaths = pd.DataFrame(dictionary_filepaths )
# resize_object = ReaderWriterClass.ResizeSegmentations(df_filepaths)
# resize_object.save_images_to_disk()
