# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os
from os import listdir
from os.path import isfile, join
import pydicom
import pandas as pd
import DicomReader as Reader
import SimpleITK as sitk
from DicomWriter import DicomWriter
from B_01_ResampleSegmentations import ResizeSegmentation
from C_mainDistanceVolumeMetrics import main_distance_volume_metrics


def create_paths(rootdir):
    list_all_ct_series = []
    for subdir, dirs, files in os.walk(rootdir):
        if not len(files) > 1:
            continue
        else:
            for file in sorted(files):
                try:
                    dcm_file = os.path.join(subdir, file)
                    ds = pydicom.read_file(dcm_file)
                except Exception:
                    # not dicom file so continue until you find one
                    continue
                path_img_folder = dcm_file
                source_series_instance_uid = ds.SeriesInstanceUID
                source_study_instance_uid = ds.StudyInstanceUID
                source_series_number = ds.SeriesNumber
                patient_id = ds.PatientID
                ablation_date = ds.StudyDate

                try:
                    path_reference_segm = ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID
                    path_reference_src = ds.SourceImageSequence[0].ReferencedSOPInstanceUID
                    lesion_number = ds.ReferencedImageSequence[0].ReferencedSegmentNumber
                    segment_label = ds.SegmentLabel
                except AttributeError:
                    path_reference_segm = None
                    path_reference_src = None
                    lesion_number = None
                    segment_label = None

                # if the ct series is not found in the dictionary, add it
                result = next((item for item in list_all_ct_series if
                               item["SeriesInstanceNumberUID"] == source_series_instance_uid), None)

                if result is None:  # that means that that the img is not yet in the dictionary
                    dict_series_folder = {
                        "PatientID": patient_id,
                        "AblationDate": ablation_date,
                        "PathSeries": path_img_folder,
                        "SegmentLabel": segment_label,
                        "LesionNumber": lesion_number,
                        "ReferenceSourceImgSeriesInstanceUID": path_reference_src,
                        "ReferenceSegmentationImgSeriesInstanceUID": path_reference_segm,
                        "SeriesNumber": source_series_number,
                        "SeriesInstanceNumberUID": source_series_instance_uid,
                        "StudyInstanceUID": source_study_instance_uid,
                    }
                    list_all_ct_series.append(dict_series_folder)

    return list_all_ct_series


if __name__ == '__main__':
    #  start with single patient folder, then load all the folders in the memory with glob
    rootdir = r"C:\tmp_patients\Pat_M03_193708128024\Study_840"
    dir_plots = r"C:\Figures"
    flag_resize_only_segmentations = 'Y'
    flag_match_with_patient_studyID = 'N'
    flag_extract_max_size = 'N'
    list_all_ct_series = create_paths(rootdir)
    df_paths_mapping = pd.DataFrame(list_all_ct_series)

    for idx, el in enumerate(df_paths_mapping.SegmentLabel):
        print(df_paths_mapping.iloc[idx].SegmentLabel)
        if df_paths_mapping.iloc[idx].SegmentLabel == 'Ablation':
            ablation_path, file = os.path.split(df_paths_mapping.iloc[idx].PathSeries)
            referenced_series_uid = df_paths_mapping.iloc[idx].ReferenceSegmentationImgSeriesInstanceUID
            if referenced_series_uid is not None:
                # the ablation segmentation has a tumor segmentation pair
                try:
                    idx_tumor_path = df_paths_mapping.index[
                        df_paths_mapping.SeriesInstanceNumberUID == referenced_series_uid].tolist()[0]
                except IndexError:
                    continue
            else:
                continue

            if df_paths_mapping.iloc[idx_tumor_path].PathSeries is not None:
                tumor_path, file = os.path.split(df_paths_mapping.iloc[idx_tumor_path].PathSeries)
                tumor_segmentation_sitk, tumor_sitk_reader = Reader.read_dcm_series(tumor_path, True)
                ablation_segmentation_sitk, ablation_sitk_reader = Reader.read_dcm_series(ablation_path, True)
                lesion_number = df_paths_mapping.iloc[idx_tumor_path].LesionNumber
                ablation_date = df_paths_mapping.iloc[idx_tumor_path].AblationDate
                patient_id = df_paths_mapping.iloc[idx_tumor_path].PatientID

                #%% RESAMPLE the tumor and the ablation
                resizer = ResizeSegmentation(ablation_segmentation_sitk, tumor_segmentation_sitk)
                tumor_segmentation_resampled = resizer.resample_segmentation()  # sitk image object
                main_distance_volume_metrics(patient_id, ablation_segmentation_sitk, tumor_segmentation_resampled,
                                             lesion_number,
                                             ablation_date, dir_plots)
                # main_distance_volume_metrics(patient_id, ablation_segmentation, tumor_segmentation, lesion_id,
                #                              ablation_date, dir_plots,
                #                              FLAG_SAVE_TO_EXCEL=True, title='Ablation to Tumor Euclidean Distances'):
                # send the tumor segm and the ablation segm to the distance & volume metrics
                # patient id, lesion_id ? that's all. maybe later print whether the patient had recurrence or not.

                #%%% bollocks that we will probably never use
                # write to DISK
                # writer = DicomWriter(ablation_segmentation_sitk, ablation_path, "", ablation_sitk_reader)
                # writer.save_image_to_file()

                # tumor_segm_array = sitk.GetArrayFromImage(tumor_segmentation_resampled)
                # only_files = [f for f in listdir(tumor_path) if isfile(join(tumor_path, f))]
                # # TODO: ensure they are in the correct order something like this slices = sorted(slices, key=lambda s: s.SliceLocation)
                # # TODO: maybe we need to replace the image from scratch
                # for z in range(0, len(tumor_segm_array)):
                #     tumor_slice = tumor_segm_array[z, :, :]
                #     dcm_file_path = os.path.join(tumor_path, dcm_file)
                #     ds = pydicom.read_file(dcm_file_path)
                #     # ds.pixel_array = tumor_slice
                #     ds.Rows, ds.Columns = tumor_slice.shape
                #     ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness = tumor_segmentation_resampled.GetSpacing()
                #     ds.PixelData = tumor_slice.tobytes()  # or tostring()
                #     ds.save_as(dcm_file_path)
