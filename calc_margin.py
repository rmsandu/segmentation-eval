import argparse

import numpy as np
import pandas as pd

import scripts.plot_ablation_margin_hist as pm
from customradiomics.margin import compute_distances
from utils.niftireader import load_image

np.set_printoptions(suppress=True, precision=4)


def is_running_in_snakemake():
    try:
        snakemake
        return True
    except NameError:
        return False


def get_args():
    if is_running_in_snakemake():
        # noinspection PyUnresolvedReferences
        args = {
            'tumor': snakemake.input['tumor'],
            'ablation': snakemake.input['ablation'],
            'liver': snakemake.input['liver'],
            'patient_id': snakemake.params['patient_id'],
            'lesion_id': snakemake.params['lesion_id'],
            'OUTPUT': snakemake.output[0]
        }
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--tumor", required=True, help="path to the tumor segmentation")
        ap.add_argument("-a", "--ablation", required=True, help="path to the ablation segmentation")
        ap.add_argument("-l", "--liver", required=True, help="path to the liver segmentation")
        ap.add_argument("-i", "--lesion-id", required=True, help="lesion id")
        ap.add_argument("-p", "--patient-id", required=True, help="patient id from study")
        ap.add_argument("-o", "--OUTPUT", required=True, help="output file (csv)")
        args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = get_args()
    # -t
    # "data\B04\01\B04_L01_Tumor.nii.gz" - a
    # "data\B04\01\B04_L01_Ablation.nii.gz" - l
    # "data\B04\01\B04_L01_Liver.nii.gz" - i
    # 1 - p
    # "B04" - o
    # "output111.csv"
    tumor_file = args['tumor']
    ablation_file = args['ablation']
    liver_file = args['liver']
    patient_id = args['patient_id']
    lesion_id = args['lesion_id']
    output_file = args['OUTPUT']
    rootdir = r"C:\develop\segmentation-eval\figures"
    tumor, tumor_np = load_image(tumor_file)
    ablation, ablation_np = load_image(ablation_file)
    liver, liver_np = load_image(liver_file)
    affine_transform = tumor.affine

    has_liver_segmented = np.sum(liver_np.astype(np.uint8)) > 0

    pixdim = liver.header['pixdim']
    spacing = (pixdim[1], pixdim[2], pixdim[3])
    # compute_distances(mask_gt, mask_pred, exclusion_zone, spacing_mm, connectivity=1, crop=True, exclusion_distance=5)
    surface_distance = compute_distances(tumor_np, ablation_np, affine_transform,
                                         exclusion_zone=liver_np if has_liver_segmented else None,
                                         spacing_mm=spacing, connectivity=1)

    pm.plot_histogram_surface_distances(patient_id, lesion_id, rootdir, distance_map=surface_distance['distances_gt_to_pred'],
                                        num_voxels=len(surface_distance['distances_gt_to_pred']),
                                        title='SurfaceDistance_Histogram_B04_L1', ablation_date="20160101",
                                        flag_to_plot=True)

    df = pd.DataFrame(data={
        'Patient': [patient_id] * len(surface_distance['distances_gt_to_pred']),
        'Lesion': [lesion_id] * len(surface_distance['distances_gt_to_pred']),
        'Distances': surface_distance['distances_gt_to_pred']})
    df.to_csv(output_file, index=False)
