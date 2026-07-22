"""Tests for Export all bundle artifacts (maps.npz, brightfield, clusters)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flim_phasors.data import PhasorData
from flim_phasors.export_bundle import (
    build_session_dict,
    export_analysis_bundle,
    export_sample_maps,
)
from flim_phasors.session_bundle_io import MAP_KEYS


def _make_dataset(name: str, g0: float = 0.4, s0: float = 0.3) -> PhasorData:
    d = PhasorData()
    d.sample_path = f"/data/{name}.ptu"
    d.group_name = "grp"
    d.frequency = 80.0
    d.harmonic = 1
    d.channel = 0
    rng = np.random.default_rng(0)
    h, w = 8, 8
    d.mean_raw = rng.uniform(5, 50, size=(h, w))
    d.mean_thr = d.mean_raw.copy()
    d.mean_thr[d.mean_raw < 15] = np.nan
    d.real_cal = np.full((h, w), g0, dtype=float)
    d.imag_cal = np.full((h, w), s0, dtype=float)
    d.real_cal[~np.isfinite(d.mean_thr)] = np.nan
    d.imag_cal[~np.isfinite(d.mean_thr)] = np.nan
    d.tau_phi = np.full((h, w), 2.0)
    d.tau_mod = np.full((h, w), 2.1)
    d.tau_normal = np.full((h, w), 2.05)
    d._intensity_stats = {
        "threshold": 15,
        "n_pixels": h * w,
        "n_below": int(np.sum(~np.isfinite(d.mean_thr))),
    }
    return d


def _fake_win(datasets: list[PhasorData], *, mode: str = "cursor"):
    active = datasets[0]
    cursors = [
        {
            "kind": "circle",
            "center_real": float(np.nanmean(active.real_cal)),
            "center_imag": float(np.nanmean(active.imag_cal)),
            "radius": 0.2,
            "radius_minor": None,
            "angle": 0.0,
            "label": "roi1",
            "color": (1.0, 0.0, 0.0),
        }
    ]
    fig = SimpleNamespace(
        savefig=lambda *a, **k: None,
    )
    win = SimpleNamespace(
        data=active,
        mode=mode,
        active_idx=0,
        cluster_stats=[],
        last_overlay=None,
        phasor=SimpleNamespace(cursors=cursors, fig=fig),
        ref_calibration=SimpleNamespace(
            mean_g=0.1, mean_s=0.2, use_manual=False, manual_g=0.0, manual_s=0.0,
        ),
        shared_ref_path="",
        shared_ref_channel=0,
        sp_freq=SimpleNamespace(value=lambda: 80.0),
        sp_harm=SimpleNamespace(value=lambda: 1),
        sp_reflt=SimpleNamespace(value=lambda: 4.0),
        sp_thr=SimpleNamespace(value=lambda: 15),
        cb_filter=SimpleNamespace(currentText=lambda: "median"),
        chk_detect_harm=SimpleNamespace(isChecked=lambda: True),
        chk_shared_ref=SimpleNamespace(isChecked=lambda: True),
    )
    win._all_datasets = lambda: datasets
    win._effective_ref_path = lambda d: getattr(d, "ref_path", None)
    win._rgb_hex = lambda rgb: "FF0000"
    return win


def test_export_sample_maps_writes_npz_and_brightfield(tmp_path: Path):
    d = _make_dataset("a")
    written = export_sample_maps(tmp_path, d, prefix="a")
    assert "a_maps.npz" in written
    assert "a_photons.png" in written
    assert "a_photons_colorbar.png" in written
    assert "a_brightfield.png" in written
    assert "a_brightfield_colorbar.png" in written
    assert "a_tau_mod_ns.png" in written
    assert "a_tau_mod_ns_colorbar.png" in written
    # No legacy duplicate names
    assert "photons.png" not in written
    assert "tau_mod_ns_fig.png" not in written
    npz = np.load(tmp_path / "a_maps.npz")
    for key in ("real_cal", "imag_cal", "mean_raw", "mean_thr", "tau_mod"):
        assert key in npz.files
        assert key in MAP_KEYS


def test_export_analysis_bundle_multi_sample_clusters(tmp_path: Path):
    d1 = _make_dataset("s1", g0=0.35, s0=0.28)
    d2 = _make_dataset("s2", g0=0.45, s0=0.32)
    win = _fake_win([d1, d2], mode="cursor")
    result = export_analysis_bundle(win, tmp_path)
    assert result["n_samples"] == 2
    assert (tmp_path / "clusters.csv").is_file()
    assert (tmp_path / "session.json").is_file()
    assert (tmp_path / "analysis.xlsx").is_file()
    assert not (tmp_path / "analysis_results.xlsx").exists()
    assert (tmp_path / "README_export.txt").is_file()

    text = (tmp_path / "clusters.csv").read_text(encoding="utf-8")
    assert "s1.ptu" in text
    assert "s2.ptu" in text

    session = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert "gmm_fit" in session
    assert "cluster_stats" in session
    assert session["maps_npz_keys"] == list(MAP_KEYS)

    sample_dirs = list((tmp_path / "samples").iterdir())
    assert len(sample_dirs) == 2
    for sub in sample_dirs:
        stem = "s1" if "s1" in sub.name else "s2"
        assert (sub / f"{stem}_maps.npz").is_file()
        assert (sub / f"{stem}_brightfield.png").is_file()
        assert (sub / f"{stem}_brightfield_colorbar.png").is_file()
        assert (sub / f"{stem}_clusters.csv").is_file()
        assert (sub / f"{stem}_mask_01.png").is_file()
        # No leftover unprefixed duplicates
        assert not (sub / "photons.png").exists()
        assert not (sub / "maps.npz").exists()


def test_export_rewrites_same_excel_and_clears_sample_dir(tmp_path: Path):
    d = _make_dataset("cellA")
    win = _fake_win([d])
    export_analysis_bundle(win, tmp_path)
    sub = next((tmp_path / "samples").iterdir())
    stale = sub / "photons.png"
    stale.write_bytes(b"stale")
    xlsx = tmp_path / "analysis.xlsx"
    assert xlsx.is_file()
    mtime1 = xlsx.stat().st_mtime_ns

    export_analysis_bundle(win, tmp_path)
    assert xlsx.is_file()
    assert xlsx.stat().st_mtime_ns >= mtime1
    assert not stale.exists()
    assert (sub / "cellA_brightfield.png").is_file()


def test_export_uses_per_sample_gmm_fit(tmp_path: Path):
    """Export all should use each sample's stored GMM, not only the active one."""
    d1 = _make_dataset("s1", g0=0.35, s0=0.28)
    d2 = _make_dataset("s2", g0=0.55, s0=0.40)
    fit1 = (
        np.array([0.35]), np.array([0.28]),
        np.array([0.05]), np.array([0.04]), np.array([0.0]),
    )
    fit2 = (
        np.array([0.55, 0.2]), np.array([0.40, 0.15]),
        np.array([0.05, 0.04]), np.array([0.04, 0.03]), np.array([0.0, 0.1]),
    )
    d1.gmm_fit = fit1
    d1.cluster_stats = [
        dict(idx=1, color=(1, 0, 0), label="a", tp=1.0, tm=1.1, tn=1.0,
             g=0.35, s=0.28, n=5, area=10.0),
    ]
    d2.gmm_fit = fit2
    d2.cluster_stats = [
        dict(idx=1, color=(1, 0, 0), label="b", tp=2.0, tm=2.1, tn=2.0,
             g=0.55, s=0.40, n=3, area=5.0),
        dict(idx=2, color=(0, 1, 0), label="c", tp=3.0, tm=3.1, tn=3.0,
             g=0.2, s=0.15, n=4, area=6.0),
    ]
    win = _fake_win([d1, d2], mode="gmm")
    win._gmm_fit = fit1  # active only has fit1
    export_analysis_bundle(win, tmp_path)
    text = (tmp_path / "clusters.csv").read_text(encoding="utf-8")
    assert "s1.ptu" in text and "s2.ptu" in text
    assert text.count("\n") >= 3  # header + at least 3 cluster rows
    sub2 = next(p for p in (tmp_path / "samples").iterdir() if "s2" in p.name)
    assert (sub2 / "s2_mask_01.png").is_file()
    assert (sub2 / "s2_mask_02.png").is_file()


def test_build_session_dict_includes_gmm_and_clusters():
    d = _make_dataset("x")
    win = _fake_win([d])
    win.cluster_stats = [
        dict(idx=1, color=(1, 0, 0), label="roi1", tp=1.0, tm=1.1, tn=1.05,
             g=0.4, s=0.3, n=10, area=12.5),
    ]
    win._gmm_fit = (
        np.array([0.4]), np.array([0.3]),
        np.array([0.05]), np.array([0.04]), np.array([0.0]),
    )
    session = build_session_dict(win)
    assert session["gmm_fit"] is not None
    assert session["gmm_fit"]["center_real"] == [0.4]
    assert len(session["cluster_stats"]) == 1
    assert session["cluster_stats"][0]["label"] == "roi1"
