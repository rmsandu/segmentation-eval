import numpy as np
import pytest
from conftest import cube_mask, cube_pair, make_image

from VolumeMetrics import VolumeMetrics


def run(ablation, tumor):
    vm = VolumeMetrics()
    vm.set_image_object(ablation_segmentation=ablation, tumor_segmentation=tumor)
    vm.set_volume_metrics()
    assert vm.error_flag is False
    return vm


def test_perfect_overlap_dice_and_jaccard_are_one():
    tumor, ablation = cube_pair(margin_mm=0)
    vm = run(ablation, tumor)
    assert vm.dice == pytest.approx(1.0)
    assert vm.jaccard == pytest.approx(1.0)
    assert vm.volume_tumor == pytest.approx(vm.volume_ablation)
    assert vm.coverage_ratio == pytest.approx(1.0)
    assert vm.volume_residual == pytest.approx(0.0)


def test_dice_and_jaccard_differ_on_partial_overlap():
    # ablation 2mm larger than tumor on every side -> partial (superset) overlap
    tumor, ablation = cube_pair(margin_mm=2)
    vm = run(ablation, tumor)
    tumor_voxels = 10 ** 3
    ablation_voxels = 14 ** 3
    intersection = tumor_voxels  # ablation fully contains the tumor
    expected_dice = 2 * intersection / (tumor_voxels + ablation_voxels)
    expected_jaccard = intersection / (tumor_voxels + ablation_voxels - intersection)

    assert vm.dice == pytest.approx(expected_dice, rel=1e-3)
    assert vm.jaccard == pytest.approx(expected_jaccard, rel=1e-3)
    # regression guard for the Dice/Jaccard copy-paste bug: these must not be equal
    assert vm.dice != pytest.approx(vm.jaccard)
    assert vm.volumetric_overlap_error == pytest.approx(1 - expected_jaccard, rel=1e-3)
    assert vm.coverage_ratio == pytest.approx(1.0)


def test_disjoint_segmentations_have_zero_overlap():
    tumor = make_image(cube_mask(lo=0, hi=10))
    ablation = make_image(cube_mask(lo=20, hi=30))
    vm = run(ablation, tumor)
    assert vm.dice == pytest.approx(0.0)
    assert vm.jaccard == pytest.approx(0.0)
    assert vm.coverage_ratio == pytest.approx(0.0)


def test_volume_ml_respects_spacing():
    vm = VolumeMetrics()
    mask = cube_mask(lo=0, hi=10)  # 1000 voxels
    img = make_image(mask, spacing=(0.5, 0.5, 2.0))  # 0.5 ml per voxel
    volume_ml = vm.get_volume_ml(img)
    assert volume_ml == pytest.approx(1000 * 0.5 * 0.5 * 2.0 / 1000)


def test_empty_mask_sets_error_flag():
    vm = VolumeMetrics()
    empty = make_image(np.zeros((10, 10, 10), dtype=np.uint8))
    assert vm.get_volume_ml(empty) is None
    assert vm.error_flag is True
