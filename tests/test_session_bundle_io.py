"""Tests for .flimsession bundle save/load (no PTU histograms)."""

import io
import json
import zipfile

import numpy as np

from flim_phasors.data import PhasorData
from flim_phasors.session_bundle_io import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    MANIFEST_NAME,
    dataset_from_bundle_sample,
    load_session_bundle,
)


def _sample_maps(shape=(4, 5)):
    """Synthetic calibrated map bundle for session round-trip tests."""
    rng = np.random.default_rng(0)
    g = rng.random(shape)
    s = rng.random(shape)
    photons = rng.integers(10, 100, size=shape, dtype=np.int32).astype(float)
    return {
        "real_cal": g,
        "imag_cal": s,
        "mean_raw": photons,
        "mean_thr": photons.copy(),
        "tau_phi": g * 2.0,
        "tau_mod": s * 2.0,
        "tau_normal": (g + s) * 0.5,
    }


def _write_bundle(path, samples_meta, maps_per_sample, *, overlay=None, cursors=None):
    """Write a minimal .flimsession zip with manifest and per-sample maps."""
    manifest = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "app_version": "test",
        "segmentation_mode": "cursor",
        "active_sample_index": 0,
        "calibration": {
            "frequency_MHz": 80.0,
            "harmonic": 1,
            "reference_lifetime_ns": 4.0,
            "filter": "median",
            "min_photons": 10,
            "harmonic_mask": True,
            "reference_path": "",
            "reference_channel": 0,
            "mean_g": 0.5,
            "mean_s": 0.2,
            "manual": False,
            "manual_g": 0.0,
            "manual_s": 0.0,
        },
        "cursors": cursors or [],
        "ui": {"multi_image": len(samples_meta) > 1, "cluster_stats": []},
        "samples": samples_meta,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, maps in enumerate(maps_per_sample):
            buf = io.BytesIO()
            np.savez_compressed(buf, **maps)
            zf.writestr(f"samples/{i:03d}/maps.npz", buf.getvalue())
        if overlay is not None:
            buf = io.BytesIO()
            np.savez_compressed(buf, overlay=np.asarray(overlay, dtype=float))
            zf.writestr("overlay.npz", buf.getvalue())
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))


def test_dataset_from_bundle_sample_roundtrip():
    """Rebuild PhasorData from bundle metadata and map arrays."""
    meta = {
        "original_sample_path": "demo.ptu",
        "group": "A",
        "channel": 1,
        "n_channels": 2,
        "frequency_MHz": 80.0,
        "harmonic": 2,
        "processing_settings": {"filter_mode": "median", "intensity_min": 5},
        "intensity_stats": {"threshold": 5, "n_pixels": 20},
    }
    maps = _sample_maps()
    d = dataset_from_bundle_sample(meta, maps)
    assert d.sample_path == "demo.ptu"
    assert d.group_name == "A"
    assert d.channel == 1
    assert d.harmonic == 2
    assert d.processing_settings["filter_mode"] == "median"
    assert np.allclose(d.real_cal, maps["real_cal"])
    assert d._shape_hint == maps["real_cal"].shape


def test_load_session_bundle_multi_sample(tmp_path):
    """Load a zip bundle with two samples and optional overlay."""
    maps_a = _sample_maps((3, 4))
    maps_b = _sample_maps((3, 4))
    meta = [
        {
            "index": 0,
            "original_sample_path": "a.ptu",
            "group": "",
            "channel": 0,
            "n_channels": 1,
            "frequency_MHz": 80.0,
            "harmonic": 1,
            "maps_file": "samples/000/maps.npz",
            "processing_settings": {},
            "intensity_stats": {},
        },
        {
            "index": 1,
            "original_sample_path": "b.ptu",
            "group": "ctrl",
            "channel": 0,
            "n_channels": 1,
            "frequency_MHz": 80.0,
            "harmonic": 1,
            "maps_file": "samples/001/maps.npz",
            "processing_settings": {},
            "intensity_stats": {},
        },
    ]
    path = tmp_path / "test.flimsession"
    _write_bundle(path, meta, [maps_a, maps_b], overlay=np.ones((3, 4, 3)) * 0.5)

    loaded = load_session_bundle(path)
    assert len(loaded["datasets"]) == 2
    assert loaded["datasets"][1].group_name == "ctrl"
    assert loaded["overlay"].shape == (3, 4, 3)
    assert loaded["manifest"]["format"] == BUNDLE_FORMAT


def test_load_rejects_missing_manifest(tmp_path):
    """Raise when session zip lacks session_manifest.json."""
    path = tmp_path / "bad.flimsession"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("other.txt", "x")
    try:
        load_session_bundle(path)
        raised = False
    except ValueError as e:
        raised = True
        assert "manifest" in str(e).lower()
    assert raised
