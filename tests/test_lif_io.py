"""Tests for Leica LIF listing helpers (mocked liffile)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flim_phasors.data import PhasorData
from flim_phasors.lif_io import (
    apply_lasx_phasor_calibration,
    is_lif_path,
    list_lif_phasor_series,
    load_lif_phasor_maps,
)
from flim_phasors.utils import dataset_has_sample, dataset_short_label


def test_is_lif_path():
    """Recognize Leica .lif and .xlef extensions."""
    assert is_lif_path("scan.LIF")
    assert is_lif_path("bundle.xlef")
    assert not is_lif_path("data.ptu")


def _mock_phasor_image(name="Phasor Intensity", path="SeriesA/Phasor Intensity", shape=(64, 32)):
    """Build a liffile image mock with parent series metadata."""
    im = MagicMock()
    im.name = name
    im.path = path
    parent = MagicMock()
    parent.path = path.rsplit("/", 1)[0]
    parent.name = parent.path.split("/")[-1]
    im.parent_image = parent
    im.asarray.return_value = np.zeros(shape, dtype=np.float32)
    return im


class _MockLifImages:
    """Minimal liffile image collection for unit tests."""

    def __init__(self, items, lookup):
        """Store iterable images and regex lookup table."""
        self._items = items
        self._lookup = lookup

    def __iter__(self):
        """Iterate registered mock images."""
        return iter(self._items)

    def __getitem__(self, key):
        """Resolve a phasor channel by regex key."""
        return self._lookup[key]


@patch("liffile.LifFile")
def test_list_lif_phasor_series_dedupes_and_sorts(mock_lif_file):
    """List unique FLIM series with phasor triplets, sorted by key."""
    lif = MagicMock()
    real = MagicMock()
    imag = MagicMock()
    intensity_b = _mock_phasor_image(path="B/Phasor Intensity")
    intensity_a = _mock_phasor_image(path="A/Phasor Intensity")
    lookup = {
        ".*B.*/Phasor Intensity$": intensity_b,
        ".*B.*/Phasor Real$": real,
        ".*B.*/Phasor Imaginary$": imag,
        ".*A.*/Phasor Intensity$": intensity_a,
        ".*A.*/Phasor Real$": real,
        ".*A.*/Phasor Imaginary$": imag,
    }
    lif.images = _MockLifImages([intensity_b, intensity_a, intensity_b], lookup)
    mock_lif_file.return_value.__enter__.return_value = lif

    found = list_lif_phasor_series("test.lif")
    assert len(found) == 2
    assert found[0].image_key == "A"
    assert found[1].image_key == "B"
    assert found[0].shape_yx == (64, 32)


@patch("flim_phasors.data.load_lif_phasor_maps")
def test_load_lif_phasor_populates_maps(mock_load):
    """load_lif_phasor sets maps, frequency, and LIF metadata on PhasorData."""
    mean = np.ones((4, 5), dtype=np.float32)
    real = np.full((4, 5), 0.2, dtype=np.float32)
    imag = np.full((4, 5), 0.1, dtype=np.float32)
    mock_load.return_value = (mean, real, imag, {"frequency": 19.5, "coords": {"X": [0, 0.1]}})

    d = PhasorData()
    shape, nch = d.load_lif_phasor("file.lif", "Series1")

    assert shape == (5, 4)
    assert nch == 1
    assert d.load_source == "lif_phasor"
    assert d.lif_image_key == "Series1"
    assert d.frequency == pytest.approx(19.5)
    assert d.real_cal.shape == (4, 5)
    assert dataset_has_sample(d)


def test_apply_lasx_phasor_calibration_moves_coordinates():
    """LAS X reference phase/amplitude shifts phasor coordinates."""
    real = np.array([[0.5, -0.2]], dtype=np.float32)
    imag = np.array([[0.3, 0.8]], dtype=np.float32)
    attrs = {
        "samples": 1,
        "flim_phasor_channels": [{
            "AutomaticReferencePhase": 10.0,
            "AutomaticReferenceAmplitude": 2.0,
            "IntensityThreshold": 20,
        }],
    }
    rc, ic, info = apply_lasx_phasor_calibration(real, imag, attrs)
    assert info["applied"] is True
    assert info["intensity_threshold"] == pytest.approx(20.0)
    assert not np.allclose(rc, real)
    assert not np.allclose(ic, imag)


@patch("flim_phasors.lif_io.phasor_from_lif")
def test_load_lif_phasor_maps_applies_lasx_calibration(mock_pf):
    """load_lif_phasor_maps applies LAS X calibration metadata from attrs."""
    mean = np.ones((2, 2), dtype=np.float32)
    real = np.full((2, 2), 0.5, dtype=np.float32)
    imag = np.full((2, 2), 0.2, dtype=np.float32)
    mock_pf.return_value = (
        mean,
        real,
        imag,
        {
            "frequency": 19.5,
            "samples": 1,
            "flim_phasor_channels": [{
                "AutomaticReferencePhase": 5.0,
                "AutomaticReferenceAmplitude": 1.0,
                "IntensityThreshold": 10,
            }],
        },
    )
    _, rc, ic, attrs = load_lif_phasor_maps("test.lif")
    assert attrs["lasx_calibration"]["applied"] is True
    assert attrs["lasx_intensity_threshold"] == pytest.approx(10.0)
    assert rc.shape == (2, 2)


def test_dataset_short_label_lif():
    """Short labels include LIF filename and series key."""
    d = PhasorData()
    d.sample_path = "/data/experiment.lif"
    d.lif_image_key = "TileScan/FLIM1"
    assert dataset_short_label(d) == "experiment.lif · FLIM1"


def test_lif_phasorpy_fixture_on_semicircle():
    """Regression: calibrated LIF phasors should sit on the visible semicircle."""
    pytest.importorskip("phasorpy.datasets")
    from phasorpy.datasets import fetch

    path = fetch("FLIM_testdata.lif")
    d = PhasorData()
    d.load_lif_phasor(path)
    m = d.valid_mask()
    assert m.sum() > 100
    g = d.real_cal[m]
    s = d.imag_cal[m]
    assert float(np.nanmin(g)) >= 0.0
    assert float(np.nanmax(g)) <= 1.05
    assert float(np.nanmin(s)) >= 0.0
    assert float(np.nanmax(s)) <= 0.75
    assert d.lif_lasx_calibrated
    assert d.lif_uses_photon_intensity
    assert float(np.nanmax(d.mean_raw)) > 1.0
