import os
import numpy as np
import nibabel as nib

TEST_DATA_PATH = 'data'


def save_test_case(data_tumor, data_ablation, data_liver, path, sample, name):
    if not os.path.exists(os.path.join(path, sample)):
        os.mkdir(os.path.join(path, sample))
    if not os.path.exists(os.path.join(path, sample, name)):
        os.mkdir(os.path.join(path, sample, name))

    data_tumor_path = os.path.join(path, sample, name, '{0}_L{1}_Tumor.nii.gz'.format(sample, name))
    data_ablation_path = os.path.join(path, sample, name, '{0}_L{1}_Ablation.nii.gz'.format(sample, name))
    data_liver_path = os.path.join(path, sample, name, '{0}_L{1}_Liver.nii.gz'.format(sample, name))
    nib.save(nib.Nifti1Image(data_tumor, affine=np.eye(4)), data_tumor_path)
    nib.save(nib.Nifti1Image(data_ablation, affine=np.eye(4)), data_ablation_path)
    nib.save(nib.Nifti1Image(data_liver, affine=np.eye(4)), data_liver_path)


def get_empty_data():
    data_tumor = np.zeros([50, 50, 50], dtype=np.uint8)
    data_ablation = np.zeros([50, 50, 50], dtype=np.uint8)
    data_liver = np.zeros([50, 50, 50], dtype=np.uint8)
    return data_tumor, data_liver, data_ablation


if not os.path.exists(TEST_DATA_PATH):
    os.mkdir(TEST_DATA_PATH)

# perfect overlap
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:30, 20:30, 20:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', 'perfect_overlap')

# perfect overlap
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[25, 25, 25] = 1
data_ablation[25, 25, 25] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', 'perfect_overlap_1voxel')

# 1 mm positive margin
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[19:31, 19:31, 19:31] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '1mm_pos_margin')

# 1 mm negative margin
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[21:29, 21:29, 21:29] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '1mm_neg_margin')

# 5 mm positive margin
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[15:35, 15:35, 15:35] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_pos_margin')

# 4 mm neg margin
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[24:26, 24:26, 24:26] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_neg_margin')

# ablation is a rectangle of 5 x 10 x 10 millimeters covering half of a 10 x 10 x 10 cube
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:30, 20:25, 20:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_neg_margin_halfcovered')


# ablation is a cube of 10 x 10 x 10 millimeters shifted to one side by 5 mm
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[25:35, 20:30, 20:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_margin_shifted_x')

# ablation is a cube of 10 x 10 x 10 millimeters shifted to one side by 5 mm
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:30, 20:30, 25:35] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_margin_shifted_z')

# ablation is a cube of 10 x 10 x 10 millimeters shifted to one side by 5 mm
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:30, 25:35, 20:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_margin_shifted_y')

# ablation is a cube of 10 x 10 x 10 millimeters shifted to xy side by 5 mm
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[25:35, 25:35, 20:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_margin_shifted_xy')

# ablation is a cube of 10 x 10 x 10 millimeters completely outside the tumor
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[35:45, 35:45, 35:45] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_neg_outside')

# ablation doesn't cover all
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:35, 20:35, 20:35] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T01', '5mm_neg_halfcover')


## with exclusion zone
# perfect overlap
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:30, 20:30, 20:30] = 1
data_liver[0:30, 0:30, 0:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T02', 'perfect_overlap')

# perfect overlap
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:30, 20:30, 20:30] = 1
data_liver[0:35, 0:35, 0:35] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T02', 'perfect_overlap_5mm_subcapsular')

# perfect overlap
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:30, 20:30, 20:30] = 1
data_liver[0:34, 0:34, 0:34] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T02', 'perfect_overlap_4mm_subcapsular')

# neg margin at exclusion
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[20:29, 20:29, 20:29] = 1
data_liver[0:30, 0:30, 0:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T02', '1mm_neg_margin_at_surface')

# neg margin opposite of exclusion
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[21:30, 21:30, 21:30] = 1
data_liver[0:30, 0:30, 0:30] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T02', '1mm_neg_margin_op_surface')

# 5 mm margin at surface
data_tumor, data_ablation, data_liver = get_empty_data()
data_tumor[20:30, 20:30, 20:30] = 1
data_ablation[15:34, 15:34, 15:34] = 1
data_liver[0:34, 0:34, 0:34] = 1
save_test_case(data_tumor, data_ablation, data_liver, TEST_DATA_PATH, 'T02', '5mm_pos_margin')
