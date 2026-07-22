"""Tests for calibration JSON I/O, cursor persistence, and memory estimates."""

from flim_phasors.calibration import ReferenceCalibration
from flim_phasors.calibration_io import calibration_from_dict, calibration_to_dict, load_calibration, save_calibration
from flim_phasors.cursors_io import load_cursors, save_cursors
from flim_phasors.memory_est import estimate_dataset_mb


def test_calibration_roundtrip(tmp_path):
    """Save and reload a reference calibration JSON with UI metadata."""
    cal = ReferenceCalibration(
        source_path="ref.ptu",
        channel=1,
        n_channels=2,
        mean_g=0.42,
        mean_s=0.21,
        use_manual=False,
        values_ready=True,
    )
    path = tmp_path / "cal.json"
    save_calibration(path, cal, ui_extra={"harmonic": 1})
    loaded, ui = load_calibration(path)
    assert loaded.mean_g == 0.42
    assert loaded.channel == 1
    assert loaded.values_ready is True
    assert loaded.is_active is True
    assert ui["harmonic"] == 1


def test_calibration_inactive_until_values_ready():
    """Reference calibration is inactive until g/s values are marked ready."""
    cal = ReferenceCalibration(source_path="ref.ptu")
    assert cal.is_active is False
    cal.values_ready = True
    assert cal.is_active is True


def test_calibration_dict_manual():
    """Round-trip manual g/s calibration through dict serialization."""
    d = calibration_to_dict(ReferenceCalibration(use_manual=True, manual_g=0.1, manual_s=0.2))
    cal = calibration_from_dict(d)
    assert cal.use_manual is True


def test_cursors_io(tmp_path):
    """Save and load phasor cursor definitions as JSON."""
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


def test_maps_for_shape_scalar_fallback():
    """Broadcast scalar reference g/s when spatial maps are absent."""
    cal = ReferenceCalibration(
        source_path="ref.ptu",
        mean_g=0.42,
        mean_s=0.21,
        mean_intensity=100.0,
    )
    rmean, rreal, rimag = cal.maps_for_shape((2, 3))
    assert rreal.shape == (2, 3)
    assert rreal[0, 0] == 0.42
    assert rimag[0, 0] == 0.21
    assert rmean[0, 0] == 100.0


def test_maps_for_shape_dual_harmonic_pawflim():
    """PAW-FLIM needs mean (Y,X) and g/s (n_harm,Y,X) — not a 3-D mean."""
    import numpy as np
    from phasorpy.lifetime import phasor_calibrate

    cal = ReferenceCalibration(
        mean_g=0.50,
        mean_s=0.40,
        mean_intensity=100.0,
        harmonic_gs=[(0.50, 0.40), (0.20, 0.25)],
        values_ready=True,
    )
    Y, X = 16, 16
    rmean, rreal, rimag = cal.maps_for_shape((2, Y, X))
    assert rmean.shape == (Y, X)
    assert rreal.shape == (2, Y, X)
    assert rimag.shape == (2, Y, X)
    assert rreal[0, 0, 0] == 0.50
    assert rreal[1, 0, 0] == 0.20

    sample_real = np.full((2, Y, X), 0.3)
    sample_imag = np.full((2, Y, X), 0.2)
    out_r, out_i = phasor_calibrate(
        sample_real, sample_imag, rmean, rreal, rimag,
        frequency=80.0, lifetime=4.0, harmonic=[1, 2],
    )
    assert out_r.shape == (2, Y, X)


def test_memory_est_empty():
    """Empty PhasorData reports zero estimated RAM usage."""
    from flim_phasors.data import PhasorData

    d = PhasorData()
    est = estimate_dataset_mb(d)
    assert est["total_mb"] == 0.0
