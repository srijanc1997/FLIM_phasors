"""Rough RAM estimates for FLIM histograms and phasor maps."""

from __future__ import annotations

import numpy as np


def nbytes_array(arr) -> int:
    if arr is None:
        return 0
    try:
        return int(np.asarray(arr).nbytes)
    except Exception:
        return 0


def estimate_dataset_mb(d) -> dict:
    """Return MB estimates for one PhasorData instance."""
    hist = 0
    if getattr(d, "signal_full", None) is not None:
        try:
            hist = int(d.signal_full.nbytes)
        except Exception:
            hist = nbytes_array(d.signal_full)
    maps = sum(
        nbytes_array(getattr(d, name, None))
        for name in (
            "mean_raw", "mean_thr", "real_cal", "imag_cal",
            "tau_phi", "tau_mod", "tau_normal",
            "tau_search_phi", "tau_search_mod",
        )
    )
    return {
        "histogram_mb": hist / (1024 * 1024),
        "maps_mb": maps / (1024 * 1024),
        "total_mb": (hist + maps) / (1024 * 1024),
        "lazy": getattr(d, "signal_full", None) is None and bool(getattr(d, "sample_path", "")),
    }


def format_memory_line(d) -> str:
    est = estimate_dataset_mb(d)
    if est["lazy"]:
        return "lazy (histogram not loaded)"
    return (
        f"~{est['histogram_mb']:.1f} MB histogram + ~{est['maps_mb']:.1f} MB maps "
        f"(~{est['total_mb']:.1f} MB total)"
    )
