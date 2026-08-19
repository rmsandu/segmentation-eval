import numpy as np
import SimpleITK as sitk
from conftest import write_dcm_series

from DicomReader import read_dcm_series


def synthetic_ct(shape=(4, 6, 6), spacing=(1.0, 1.0, 2.0)):
    array = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    return image


def test_read_dcm_series_reassembles_written_series(tmp_path):
    original = synthetic_ct()
    write_dcm_series(str(tmp_path), original)

    image, reader = read_dcm_series(str(tmp_path))

    assert image is not None
    assert reader is not None
    assert image.GetSize() == original.GetSize()
    assert image.GetSpacing() == original.GetSpacing()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(original))


def test_read_dcm_series_reader_flag_false_returns_image_only(tmp_path):
    original = synthetic_ct()
    write_dcm_series(str(tmp_path), original)

    result = read_dcm_series(str(tmp_path), reader_flag=False)

    assert isinstance(result, sitk.Image)


def test_read_dcm_series_unreadable_folder_returns_none(tmp_path):
    (tmp_path / "not_a_dicom.txt").write_text("hello")

    image, reader = read_dcm_series(str(tmp_path))

    assert image is None
    assert reader is None


def test_read_dcm_series_single_file(tmp_path):
    original = synthetic_ct(shape=(1, 5, 5))
    single_file = tmp_path / "single.dcm"
    sitk.WriteImage(original, str(single_file))

    image, reader = read_dcm_series(str(single_file))

    assert image is not None
    assert reader is None
    assert image.GetSize() == original.GetSize()
