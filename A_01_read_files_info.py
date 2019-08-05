# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os
import pydicom
import pandas as pd
import DicomReader as Reader
import DicomWriter as DicomWriter
import B_01_ResampleSegmentations as ResizerClass


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
    rootdir = r"C:\tmp_patients\Pat_MAV_BE_B01_\Study_0"
    folder_path_new_resized_images = r" "
    flag_resize_only_segmentations = 'Y'
    flag_match_with_patient_studyID = 'N'
    flag_extract_max_size = 'N'
    list_all_ct_series = create_paths(rootdir)
    df_paths_mapping = pd.DataFrame(list_all_ct_series)
    print('Success')

    for idx, el in enumerate(df_paths_mapping.SegmentLabel):
        print(df_paths_mapping.iloc[idx].SegmentLabel)
        if df_paths_mapping.iloc[idx].SegmentLabel == 'Ablation':
            ablation_path, file = os.path.split(df_paths_mapping.iloc[idx].PathSeries)
            referenced_series_uid = df_paths_mapping.iloc[idx].ReferenceSegmentationImgSeriesInstanceUID
            idx_tumor_path = df_paths_mapping.index[
                df_paths_mapping.SeriesInstanceNumberUID == referenced_series_uid].tolist()[0]
            if df_paths_mapping.iloc[idx_tumor_path].PathSeries is not None:
                tumor_path, file = os.path.split(df_paths_mapping.iloc[idx_tumor_path].PathSeries)
                tumor_segmentation_sitk, tumor_sitk_reader = Reader.read_dcm_series(tumor_path, True)
                ablation_segmentation_sitk, ablation_sitk_path = Reader.read_dcm_series(ablation_path, True)
                lesion_number = df_paths_mapping.iloc[idx_tumor_path].LesionNumber
                #%% RESAMPLE the tumor and the ablation


            # TODO: 1) resample the segmentations
            # TODO: 2) calculate the metrics
            # TODO: 3) export the surface distances to an excel
            # TODO: 4) plots of the histogram
            # look for the matching ReferenceSeriesUIDInstance and get the path

    # result = next((item for item in list_all_ct_series if
    #                item["SeriesInstanceNumberUID"] == source_series_instance_uid), None)
    #
    # needle_idx_df_xml = df_segmentations_paths_xml.index[
    #     df_segmentations_paths_xml["NeedleIdx"] == needle_idx_val].tolist()
    # idx_referenced_segm = [el for el in needle_idx_df_xml if el != idx_segm_xml]

    # onlyfiles = [f for f in os.listdir(mypath) if os.isfile(os.join(mypath, f))
