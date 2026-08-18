import numpy as np
import SimpleITK as sitk
from conftest import cube_mask, make_image

from B_ResampleSegmentations import ResizeSegmentation


def test_resample_matches_reference_geometry():
    reference = make_image(cube_mask(shape=(40, 40, 40)), spacing=(1.0, 1.0, 1.0))
    tumor = make_image(cube_mask(shape=(20, 20, 20), lo=5, hi=15), spacing=(2.0, 2.0, 2.0))
    ablation = make_image(cube_mask(shape=(20, 20, 20), lo=5, hi=15), spacing=(2.0, 2.0, 2.0))

    resizer = ResizeSegmentation(ablation, tumor, reference)
    resampled_tumor, resampled_ablation = resizer.resample_segmentation()

    for img in (resampled_tumor, resampled_ablation):
        assert img.GetSize() == reference.GetSize()
        assert img.GetSpacing() == reference.GetSpacing()
        assert img.GetDirection() == reference.GetDirection()


def test_resample_preserves_label_values_no_interpolation_artifacts():
    reference = make_image(cube_mask(shape=(40, 40, 40)), spacing=(1.0, 1.0, 1.0))
    tumor = make_image(cube_mask(shape=(40, 40, 40)), spacing=(1.0, 1.0, 1.0))
    ablation = make_image(cube_mask(shape=(40, 40, 40)), spacing=(1.0, 1.0, 1.0))

    resizer = ResizeSegmentation(ablation, tumor, reference)
    resampled_tumor, _ = resizer.resample_segmentation()

    values = np.unique(sitk.GetArrayFromImage(resampled_tumor))
    assert set(values.tolist()) <= {0, 1}
