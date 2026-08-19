import glob

import pandas as pd
from conftest import cube_pair

from C_mainDistanceVolumeMetrics import main_distance_volume_metrics


def test_writes_one_excel_file_with_distance_and_volume_metrics(tmp_path):
    tumor, ablation = cube_pair(margin_mm=2)

    main_distance_volume_metrics(
        patient_id="TEST01",
        source_ct_ablation=None,
        source_ct_tumor=None,
        ablation_segmentation_resampled=ablation,
        tumor_segmentation_resampled=tumor,
        lesion_id="1",
        ablation_date="20200101",
        dir_plots=str(tmp_path),
        FLAG_SAVE_TO_EXCEL=True,
        calculate_volume_metrics=True,
        calculate_radiomics=False,
    )

    excel_files = glob.glob(str(tmp_path / "*DistanceVolumeMetrics*.xlsx"))
    assert len(excel_files) == 1

    df = pd.read_excel(excel_files[0])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["patient_id"] == "TEST01"
    assert row["lesion_id"] == 1
    assert 0 < row["Dice"] < 1
    assert row["Hausdorff_AT"] >= 0


def test_no_excel_written_when_flag_is_false(tmp_path):
    tumor, ablation = cube_pair(margin_mm=0)

    main_distance_volume_metrics(
        patient_id="TEST02",
        source_ct_ablation=None,
        source_ct_tumor=None,
        ablation_segmentation_resampled=ablation,
        tumor_segmentation_resampled=tumor,
        lesion_id="1",
        ablation_date="20200101",
        dir_plots=str(tmp_path),
        FLAG_SAVE_TO_EXCEL=False,
        calculate_volume_metrics=True,
        calculate_radiomics=False,
    )

    assert glob.glob(str(tmp_path / "*.xlsx")) == []
