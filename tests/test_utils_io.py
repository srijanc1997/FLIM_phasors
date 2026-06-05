"""Smoke tests for I/O helpers and dataset label utilities (no GUI)."""



from flim_phasors.export_bundle import _safe_name

from flim_phasors.io import is_supported_flim_path

from flim_phasors.utils import (

    dataset_display_label,

    dataset_phasor_legend_label,

    dataset_short_label,

)





class _FakeData:

    """Minimal stand-in for :class:`PhasorData` with path and group fields."""



    sample_path = "/data/experiment_01.ptu"

    group_name = "control"





def test_is_supported_flim_path():

    """Recognize PTU, TIFF, LIF, and XLEF extensions; reject unrelated files."""

    assert is_supported_flim_path("a.ptu")

    assert is_supported_flim_path("b.TIFF")

    assert is_supported_flim_path("c.lif")

    assert is_supported_flim_path("d.xlef")

    assert not is_supported_flim_path("readme.txt")





def test_safe_name():

    """Sanitize export folder names and enforce maximum length."""

    assert _safe_name("a/b:c") == "a_b_c"

    assert len(_safe_name("x" * 200)) <= 80





def test_dataset_labels():

    """Build short and display labels from filename and group."""

    d = _FakeData()

    assert dataset_short_label(d) == "experiment_01.ptu"

    assert dataset_display_label(d) == "control · experiment_01.ptu"





def test_dataset_display_name_override():

    """User display_name overrides filename in legend helpers."""

    d = _FakeData()

    d.display_name = "Sample A"

    assert dataset_short_label(d) == "Sample A"

    assert dataset_phasor_legend_label(d, include_group=False) == "Sample A"

    assert dataset_phasor_legend_label(d, include_group=True) == "control · Sample A"


