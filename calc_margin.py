import argparse

import numpy as np
import pandas as pd


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
        ap.add_argument("-o", "--OUTPUT",  required=True, help="output file (csv)")
        args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = get_args()

    tumor_file = args['tumor']
    ablation_file = args['ablation']
    liver_file = args['liver']
    patient_id = args['patient_id']
    lesion_id = args['lesion_id']
    output_file = args['OUTPUT']

    tumor, tumor_np = load_image(tumor_file)
    ablation, ablation_np = load_image(ablation_file)
    liver, liver_np = load_image(liver_file)

    has_liver_segmented = np.sum(liver_np.astype(np.uint8)) > 0

    pixdim = liver.header['pixdim']
    spacing = (pixdim[1], pixdim[2], pixdim[3])
    surface_distance = compute_distances(tumor_np, ablation_np,
                                         exclusion_zone=liver_np if has_liver_segmented else None,
                                         spacing_mm=spacing, connectivity=1)

    df = pd.DataFrame(data={
        'Patient': [patient_id] * len(surface_distance['distances_gt_to_pred']),
        'Lesion': [lesion_id] * len(surface_distance['distances_gt_to_pred']),
        'Distances': surface_distance['distances_gt_to_pred']})
    df.to_csv(output_file, index=False)
