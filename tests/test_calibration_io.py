"""Tests for calibration and session helpers."""

from flim_phasors.calibration import ReferenceCalibration
from flim_phasors.calibration_io import calibration_from_dict, calibration_to_dict, load_calibration, save_calibration
from flim_phasors.cursors_io import load_cursors, save_cursors
from flim_phasors.memory_est import estimate_dataset_mb


def test_calibration_roundtrip(tmp_path):
    cal = ReferenceCalibration(
        source_path="ref.ptu",
        channel=1,
        n_channels=2,
        mean_g=0.42,
        mean_s=0.21,
        use_manual=False,
    )
    path = tmp_path / "cal.json"
    save_calibration(path, cal, ui_extra={"harmonic": 1})
    loaded, ui = load_calibration(path)
    assert loaded.mean_g == 0.42
    assert loaded.channel == 1
    assert ui["harmonic"] == 1


def test_calibration_dict_manual():
    d = calibration_to_dict(ReferenceCalibration(use_manual=True, manual_g=0.1, manual_s=0.2))
    cal = calibration_from_dict(d)
    assert cal.use_manual is True


def test_cursors_io(tmp_path):
    cursors = [{
        "kind": "circle",
        "center_real": 0.5,
        "center_imag": 0.3,
        "radius": 0.05,
        "label": "A",
        "color": (1.0, 0.0, 0.0),
    }]
    path = tmp_path / "cur.json"
    save_cursors(path, cursors, sample_path="s.ptu")
    loaded, sp = load_cursors(path)
    assert len(loaded) == 1
    assert loaded[0]["center_real"] == 0.5
    assert sp == "s.ptu"


def test_memory_est_empty():
    from flim_phasors.data import PhasorData

    d = PhasorData()
    est = estimate_dataset_mb(d)
    assert est["total_mb"] == 0.0
