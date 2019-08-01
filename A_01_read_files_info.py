# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import os
import pydicom
import pandas as pd
import B_01_ResampleSegmentations as ResizerClass

def create_paths(rootdir):
    list_all_ct_series = []
    for subdir, dirs, files in os.walk(rootdir):
        path, foldername = os.path.split(subdir)
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
            try:
                path_reference_segm = ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID
                path_reference_src = ds.SourceImageSequence[0].ReferencedSOPInstanceUID
                segment_label = ds.SegmentLabel
            except Exception:
                path_reference_segm = None
                path_reference_src = None

            # if the ct series is not found in the dictionary, add it
            result = next((item for item in list_all_ct_series if
                           item["SeriesInstanceNumberUID"] == source_series_instance_uid), None)

            if result is None:  # that means that that the img is not yet in the dictionary
                dict_series_folder = {
                    "PathSeries": path_img_folder,
                    "SegmentLabel": segment_label,
                    "ReferenceSourceImgSeriesInstanceUID": path_reference_src,
                    "ReferenceSegmentationImgSeriesInstanceUID": path_reference_segm,
                    "SeriesNumber": source_series_number,
                    "SeriesInstanceNumberUID": source_series_instance_uid,
                    "StudyInstanceUID": source_study_instance_uid,
                }
                list_all_ct_series.append(dict_series_folder)

        df_paths_mapping = pd.DataFrame(list_all_ct_series)
        return df_paths_mapping


if __name__ == '__main__':

    rootdir = r"C:\tmp_patients\Pat_MAV_BE_B03\Study_0"  # start with single patient folder, then load all the folders in the memory with glob
    folder_path_new_resized_images = r" "
    flag_resize_only_segmentations = 'Y'
    flag_match_with_patient_studyID = 'N'
    flag_extract_max_size =  'N'
    df_paths_mapping = create_paths(rootdir)

    # now that we have the paths, we should read the folders into simpleitk objects
    # we need a loop for all the tumors and the ablations
    # we need to identify which lesion number it is??? - should be encoded in DICOM tags the trajectory number.




