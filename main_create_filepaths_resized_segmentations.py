import os
import time
import csv
import pandas as pd
import ResizeSegmentations as ReaderWriterClass
import mainDistanceVolumeMetrics as Metrics
pd.options.mode.chained_assignment = None
#%%
#rootdir = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_23042018"
rootdir = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\maverric_processed"
# start patient counting at 10, not to confuse with previous dataset 2017 where patients are name 1-10
patientID = 10
# # list of dictionaries containing the filepaths of the segmentations
# dictionary_filepaths = {}
# # the filepaths are already saved in a CSV
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
#                     if column != 'TrajectoryID':
#                         file_value = path + value
#                     else:
#                         file_value = value
#                     dictionary_filepaths.setdefault(column, []).append(file_value)
#                 # add patient ID column
#                 dictionary_filepaths.setdefault('PatientID', []).append(patientID)
#%%
filename_excel = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\3d_segmentation_maverric\maverric" \
                 "\MAVERRIC_Stockholm_June_all_patients_no_registration.xlsx"
df_folderpaths= pd.read_excel(filename_excel)
# df_filepaths = pd.DataFrame(dictionary_filepaths)
# Call The ReaderWriterClass to Resize Segmentations
resize_object = ReaderWriterClass.ResizeSegmentations(df_folderpaths)
folder_path_saving = r"C:\PatientDatasets_GroundTruth_Database\Stockholm\maverric_processed"
resize_object.save_images_to_disk(folder_path_saving)
df_new_filepaths = resize_object.get_new_filepaths()

#%%
timestr = time.strftime("%H%M%S-%Y%m%d")
filename = 'FilepathsResizedGTSegmentations' + '_' + timestr + '.xlsx'
filepathExcel = os.path.join(folder_path_saving, filename)
#filepathExcel = os.path.join(rootdir, filename)
writer = pd.ExcelWriter(filepathExcel)
df_new_filepaths.to_excel(writer, index=False)
print("success")

#%%

# # call distance metrics
# df_final = pd.read_excel(r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_MAVERRIC_Danderyd_June\FilepathsResizedGTSegmentations_121717-20180704.xlsx")
# df_new1 = df_final[[' Ablation Segmentation Path Resized',
#                   ' Tumour Segmentation Path Resized',
#                   'PatientID',
#                   'TrajectoryID',
#                   'Pathology']]
# df_new1.rename(columns={' Ablation Segmentation Path Resized': ' Ablation Segmentation Path',
#                         ' Tumour Segmentation Path Resized': ' Tumour Segmentation Path'}, inplace=True)
# rootdir = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_MAVERRIC_Danderyd_June"
# Metrics.main_distance_volume_metrics(df_new1, rootdir)
#df_patientdata = df_final
