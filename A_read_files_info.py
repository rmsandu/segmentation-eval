# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

import argparse
import os
import sys
from ast import literal_eval

import pandas as pd
import pydicom

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
                try:
                    ablation_date = ds.AcquisitionDate
                except Exception:
                    try:
                        ablation_date = ds.StudyDate
                    except Exception:
                        ablation_date = None

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


def get_paths_from_metatags(df_paths_mapping):
    """

    :param df_paths_mapping:
    :param plots_dir:
    :return:
    """
    for idx_segm, el in enumerate(df_paths_mapping.SegmentLabel):
        if df_paths_mapping.iloc[idx_segm].SegmentLabel == 'Ablation':
            ablation_path, file = os.path.split(df_paths_mapping.iloc[idx_segm].PathSeries)
            source_ct_ablation_series = df_paths_mapping.iloc[idx_segm].ReferenceSourceImgSeriesInstanceUID
            referenced_series_uid = df_paths_mapping.iloc[idx_segm].ReferenceSegmentationImgSeriesInstanceUID
            idx_source_ablation = \
                df_paths_mapping.index[
                    df_paths_mapping.SeriesInstanceNumberUID == source_ct_ablation_series].tolist()[0]
            if referenced_series_uid is not None:
                # the ablation segmentation has a tumor segmentation pair
                try:
                    idx_tumor_path = df_paths_mapping.index[
                        df_paths_mapping.SeriesInstanceNumberUID == referenced_series_uid].tolist()[0]
                    source_ct_tumor_series = df_paths_mapping.iloc[
                        idx_tumor_path].ReferenceSourceImgSeriesInstanceUID
                    idx_source_tumor = df_paths_mapping.index[
                        df_paths_mapping.SeriesInstanceNumberUID == source_ct_tumor_series].tolist()[0]
                    # print('source ct tumor series:', source_ct_tumor_series)
                    # print('idx ct plan:', idx_source_tumor)
                except IndexError:
                    print(ablation_path, "some nasty error because referenced series uid was not found")
                    continue
            else:
                continue
            # if both the tumor and ablation segmentation are available
            if df_paths_mapping.iloc[idx_tumor_path].PathSeries is not None:
                tumor_path, file = os.path.split(df_paths_mapping.iloc[idx_tumor_path].PathSeries)
                source_ct_ablation_path, file = os.path.split(df_paths_mapping.iloc[idx_source_ablation].PathSeries)
                source_ct_tumor_path, file = os.path.split(df_paths_mapping.iloc[idx_source_tumor].PathSeries)
                lesion_number = df_paths_mapping.iloc[idx_tumor_path].LesionNumber
                ablation_date = df_paths_mapping.iloc[idx_tumor_path].AblationDate
                patient_id = df_paths_mapping.iloc[idx_tumor_path].PatientID
                return tumor_path, source_ct_tumor_path, ablation_path, source_ct_ablation_path, \
                       lesion_number, ablation_date, patient_id
            else:
                print("No metatags mapping found in ReferencedSequence for Tumor and Ablation!!!!")


def read_dcm_imgs(tumor_path, source_ct_tumor_path, ablation_path, source_ct_ablation_path):
    """
    reads DICOM images as SimpeleITK objects given their file paths as input
    :param tumor_path:
    :param source_ct_tumor_path:
    :param ablation_path:
    :param source_ct_ablation_path:
    :return: all DICOM files as SimpleITK objects
    """

    # READ THE SEGMENTATION MASKS AS SIMPLEITK OBJ
    tumor_segmentation_sitk, tumor_sitk_reader = Reader.read_dcm_series(tumor_path, True)
    ablation_segmentation_sitk, ablation_sitk_reader = Reader.read_dcm_series(ablation_path, True)
    # READ THE CT SOURCE IMAGES AS SIMPLEITK DICOM IMGS
    source_ct_tumor_sitk, reader = Reader.read_dcm_series(source_ct_tumor_path, True)
    source_ct_ablation_sitk, reader = Reader.read_dcm_series(source_ct_ablation_path, True)
    return tumor_segmentation_sitk, source_ct_tumor_sitk, ablation_segmentation_sitk, source_ct_ablation_sitk


def resample_tumor_ablation(tumor_segmentation_sitk, ablation_segmentation_sitk, source_ct_ablation_sitk):
    """

    :param tumor_segmentation_sitk:
    :param ablation_segmentation_sitk:
    :param source_ct_ablation_sitk:
    :return: resampled tumor and ablation segmentations to the same dimensions
    """

    # RESAMPLE THE TUMOR MASK AND ABLATION MASK TO MATCH THE SOURCE CT ABLATION VALIDATION
    # so that we can extract metrics from the two combined
    resizer = ResizeSegmentation(ablation_segmentation_sitk,
                                 tumor_segmentation_sitk,
                                 source_ct_ablation_sitk)

    tumor_segmentation_resampled, ablation_segmentation_resampled = resizer.resample_segmentation()
    return tumor_segmentation_resampled, ablation_segmentation_resampled


# %%

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--rootdir", required=False, help="path to the patient folder to be processed")
    ap.add_argument("-o", "--plots_dir", required=True, help="path to the output images")
    ap.add_argument("-b", "--input_batch_proc", required=False, help="input csv file for batch processing")
    ap.add_argument("-m", "--metatags_mapping", required=False,
                    help="Flag to specify whether the ablation tumor mapping was previously embedded in metatags")
    ap.add_argument("-t", "--tumor_path", required=False, help="path to tumor image files")
    ap.add_argument("-a", "--ablation_path", required=False, help="path to ablation image files")
    ap.add_argument("-c", "--source_ct_tumor", required=False, help="path to tumor source CT")
    ap.add_argument("-d", "--source_ct_ablation", required=False, help="path to ablation source CT")
    ap.add_argument("-p", "--patient_id", required=False, help="subject id")
    ap.add_argument("-l", "--lesion_id", required=False, help="lesion id")

    args = vars(ap.parse_args())

    if args["rootdir"] is not None:
        print("Single patient folder processing, path to folder: ", args["rootdir"])
        print(args["plots_dir"])
    elif (args["input_batch_proc"]) is not None and (args["rootdir"] is None):
        print("Batch Processing Enabled, path to Excel: ", args["input_batch_proc"])
    else:
        print("no input values provided either for single patient processing or batch processing. System Exiting")
        sys.exit()

    # one patient one lesion folder
    if args["input_batch_proc"] is None and args["rootdir"] is not None:
        if args["metatags_mapping"] is True:
            list_all_ct_series = create_paths(args["rootdir"])
            df_paths_mapping = pd.DataFrame(list_all_ct_series)
            tumor_path, source_ct_tumor_path, ablation_path, source_ct_ablation_path, \
            lesion_number, ablation_date, patient_id = get_paths_from_metatags(df_paths_mapping)
        else:
            tumor_path = args["tumor_path"]
            ablation_path = args["ablation_path"]
            source_ct_tumor_path = args["source_ct_tumor"]
            source_ct_ablation_path = args["source_ct_ablation"]
            patient_id = args["patient_id"]
            lesion_id = args["lesion_id"]
            ablation_date = None

        tumor_segmentation_sitk, source_ct_tumor_sitk, ablation_segmentation_sitk, source_ct_ablation_sitk \
            = read_dcm_imgs(tumor_path, source_ct_tumor_path, ablation_path, source_ct_ablation_path)
        tumor_segmentation_resampled, ablation_segmentation_resampled = \
            resample_tumor_ablation(tumor_segmentation_sitk, ablation_segmentation_sitk, source_ct_ablation_sitk)
        # plot the histogram of distances and extract radiomics
        main_distance_volume_metrics(patient_id,
                                     source_ct_ablation_sitk, source_ct_tumor_sitk,
                                     ablation_segmentation_resampled, tumor_segmentation_resampled,
                                     lesion_number,
                                     ablation_date,
                                     args["plots_dir"],
                                     FLAG_SAVE_TO_EXCEL=True, title='Ablation to Tumor Euclidean Distances',
                                     calculate_volume_metrics=True, calculate_radiomics=True
                                     )

        print('Extracted metrics from the patient dir: ', args["rootdir"])

        # UBELIX
        sys.stdout.flush()
        sys.exit()

    else:
        # batch processing option
        df = pd.read_excel(args["input_batch_proc"])
        df.drop_duplicates(subset=['Patient_Dir_Paths'], inplace=True)
        df['Patient_Dir_Paths'].fillna("None", inplace=True)
        df['Patient_Dir_Paths'] = df['Patient_Dir_Paths'].apply(literal_eval)

        df = df.reset_index(drop=True)
        for idx in range(len(df)):
            patient_dir_paths = df.Patient_Dir_Paths[idx]
            ablation_date_redcap = df.Ablation_IR_Date[idx]
            # reset index
            if patient_dir_paths is None:
                # no patient data image folder found for this patient
                continue
            else:
                for rootdir in patient_dir_paths:
                    rootdir = os.path.normpath(rootdir)
                    list_all_ct_series = create_paths(rootdir)
                    df_paths_mapping = pd.DataFrame(list_all_ct_series)
                    if df_paths_mapping.empty is True:
                        print('No Cas Recordings or segmentations found in this patient folder. Please investigate')
                        continue
                    else:
                        # call function to resample images and output csv for main metrics.
                        get_paths_from_metatags(df_paths_mapping, args["plots_dir"])
                        print('Extracted metrics from the patient dir: ', rootdir)
