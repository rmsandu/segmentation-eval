import os
import csv
import pandas as pd
import ResizeSegmentations as ReaderWriterClass

rootdir = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_23042018"
patientID = 0
# list of dictionaries containing the filepaths of the segmentations
dictionary_filepaths = {}

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
                    dictionary_filepaths.setdefault(column, []).append(file_value)
                # add patient ID column
                dictionary_filepaths.setdefault('PatientID', []).append(patientID)

df_filepaths = pd.DataFrame(dictionary_filepaths)
resize_object = ReaderWriterClass.ResizeSegmentations(df_filepaths)
folder_path_saving = r"C:\PatientDatasets_GroundTruth_Database\GroundTruth_2018\GT_23042018_anonymized"
resize_object.save_images_to_disk(folder_path_saving)
