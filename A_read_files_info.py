# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os
import pydicom
import argparse
import pandas as pd
import DicomReader as Reader
from B_ResampleSegmentations import ResizeSegmentation
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
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--rootdir", required=True, help="path to the patient folder to be processed")
    ap.add_argument("-o", "--plotsdir", required=True, help="path to the output images")
    args = vars(ap.parse_args())
    #  start with single patient folder, then load all the folders in the memory with glob
    flag_resize_only_segmentations = 'Y'
    flag_match_with_patient_studyID = 'N'
    flag_extract_max_size = 'N'
    print(args["rootdir"])
    print(args["plotsdir"])
    list_all_ct_series = create_paths(args["rootdir"])
    df_paths_mapping = pd.DataFrame(list_all_ct_series)
    # print(df_paths_mapping)

    for idx, el in enumerate(df_paths_mapping.SegmentLabel):

        if df_paths_mapping.iloc[idx].SegmentLabel == 'Ablation':
            ablation_path, file = os.path.split(df_paths_mapping.iloc[idx].PathSeries)
            source_ct_ablation_series = df_paths_mapping.iloc[idx].ReferenceSourceImgSeriesInstanceUID
            referenced_series_uid = df_paths_mapping.iloc[idx].ReferenceSegmentationImgSeriesInstanceUID
            idx_source_ablation = \
                df_paths_mapping.index[df_paths_mapping.SeriesInstanceNumberUID == source_ct_ablation_series].tolist()[
                    0]
            if referenced_series_uid is not None:
                # the ablation segmentation has a tumor segmentation pair
                try:
                    idx_tumor_path = df_paths_mapping.index[
                        df_paths_mapping.SeriesInstanceNumberUID == referenced_series_uid].tolist()[0]
                    source_ct_tumor_series = df_paths_mapping.iloc[idx_tumor_path].ReferenceSourceImgSeriesInstanceUID
                    idx_source_tumor = df_paths_mapping.index[
                        df_paths_mapping.SeriesInstanceNumberUID == source_ct_tumor_series].tolist()[0]
                    print('source ct tumor series:', source_ct_tumor_series)
                    print('idx ct plan:', idx_source_tumor)
                except IndexError:
                    continue
            else:
                continue
            # if both the tumor and ablation segmentation are available
            if df_paths_mapping.iloc[idx_tumor_path].PathSeries is not None:
                tumor_path, file = os.path.split(df_paths_mapping.iloc[idx_tumor_path].PathSeries)
                source_ct_ablation_path, file = os.path.split(df_paths_mapping.iloc[idx_source_ablation].PathSeries)
                print('ct validation: ', source_ct_ablation_path)
                source_ct_tumor_path, file = os.path.split(df_paths_mapping.iloc[idx_source_tumor].PathSeries)
                print('ct plan', source_ct_tumor_path)
                tumor_segmentation_sitk, tumor_sitk_reader = Reader.read_dcm_series(tumor_path, True)
                ablation_segmentation_sitk, ablation_sitk_reader = Reader.read_dcm_series(ablation_path, True)
                source_ct_tumor_sitk, reader = Reader.read_dcm_series(source_ct_tumor_path, True)
                source_ct_ablation_sitk, reader = Reader.read_dcm_series(source_ct_ablation_path, True)

                lesion_number = df_paths_mapping.iloc[idx_tumor_path].LesionNumber
                ablation_date = df_paths_mapping.iloc[idx_tumor_path].AblationDate
                patient_id = df_paths_mapping.iloc[idx_tumor_path].PatientID

                # %% RESAMPLE the tumor and the ablation
                resizer = ResizeSegmentation(ablation_segmentation_sitk, tumor_segmentation_sitk)
                tumor_segmentation_resampled = resizer.resample_segmentation()  # sitk image object
                main_distance_volume_metrics(patient_id,
                                             source_ct_ablation_sitk, source_ct_tumor_sitk,
                                             ablation_segmentation_sitk, tumor_segmentation_resampled,
                                             lesion_number,
                                             ablation_date,
                                             args["plotsdir"])
