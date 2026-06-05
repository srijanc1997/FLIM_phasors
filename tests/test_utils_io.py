"""Smoke tests for helpers (no GUI)."""

from flim_phasors.export_bundle import _safe_name
from flim_phasors.io import is_supported_flim_path
from flim_phasors.utils import dataset_display_label, dataset_short_label


class _FakeData:
    sample_path = "/data/experiment_01.ptu"
    group_name = "control"


def test_is_supported_flim_path():
    assert is_supported_flim_path("a.ptu")
    assert is_supported_flim_path("b.TIFF")
    assert not is_supported_flim_path("readme.txt")


def test_safe_name():
    assert _safe_name("a/b:c") == "a_b_c"
    assert len(_safe_name("x" * 200)) <= 80


def test_dataset_labels():
    d = _FakeData()
    assert dataset_short_label(d) == "experiment_01.ptu"
    assert dataset_display_label(d) == "control · experiment_01.ptu"
