import pytest
from conftest import cube_pair

from DistanceMetrics import DistanceMetrics


def test_perfect_overlap_has_zero_surface_distance():
    tumor, ablation = cube_pair(margin_mm=0)
    dm = DistanceMetrics(ablation, tumor)
    assert dm.error_flag is False
    assert max(dm.surface_distances) == pytest.approx(0.0, abs=1e-5)
    assert min(dm.surface_distances) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("margin_mm", [1, 3, 5])
def test_uniform_positive_margin_matches_known_distance(margin_mm):
    # ablation grown by margin_mm on every side of the tumor -> every tumor
    # surface point sits exactly margin_mm inside the ablation.
    tumor, ablation = cube_pair(margin_mm=margin_mm)
    dm = DistanceMetrics(ablation, tumor)
    assert dm.error_flag is False
    assert min(dm.surface_distances) == pytest.approx(margin_mm, abs=1e-2)
    assert max(dm.surface_distances) == pytest.approx(margin_mm, abs=1e-2)


def test_negative_margin_reports_negative_distance():
    # ablation smaller than the tumor by 2mm on every side -> tumor sticks out,
    # i.e. an insufficient/negative margin.
    tumor, ablation = cube_pair(margin_mm=-2)
    dm = DistanceMetrics(ablation, tumor)
    assert dm.error_flag is False
    assert max(dm.surface_distances) < 0
