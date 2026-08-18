"""Shared fixtures for synthetic tumor/ablation test cases.

Builds small binary cube segmentations directly as SimpleITK images (in
memory, no files, no patient data) with known geometric relationships, so
metric values can be asserted against hand-computed expected results.
"""
import numpy as np
import SimpleITK as sitk


def make_image(mask, spacing=(1.0, 1.0, 1.0)):
    """Wrap a numpy uint8 array (z, y, x) as a SimpleITK image with the given spacing."""
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(spacing)
    return img


def cube_mask(shape=(50, 50, 50), lo=20, hi=30):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[lo:hi, lo:hi, lo:hi] = 1
    return mask


def cube_pair(margin_mm=0, spacing=(1.0, 1.0, 1.0), shape=(50, 50, 50), lo=20, hi=30):
    """A tumor cube and an ablation cube grown/shrunk by margin_mm on every side.

    margin_mm > 0: ablation fully covers the tumor with that many mm to spare
                    (a "positive margin" -- the ablation is safely larger).
    margin_mm < 0: ablation is smaller than the tumor by that many mm on every
                    side (a "negative margin" -- residual untreated tumor).
    Assumes isotropic spacing so margin_mm can be expressed directly in voxels.
    """
    assert spacing[0] == spacing[1] == spacing[2], "cube_pair assumes isotropic spacing"
    voxel_margin = int(round(margin_mm / spacing[0]))
    tumor = cube_mask(shape, lo, hi)
    ablation = cube_mask(shape, lo - voxel_margin, hi + voxel_margin)
    return make_image(tumor, spacing), make_image(ablation, spacing)
