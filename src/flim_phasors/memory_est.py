"""Rough RAM estimates for FLIM histograms and phasor maps.

Helps the GUI report approximate memory use for loaded TCSPC stacks and
derived lifetime / phasor coordinate maps before or after processing.
"""

from __future__ import annotations

import numpy as np


def nbytes_array(arr) -> int:
    """Return the byte size of an array, or zero for missing data.

    Args:
        arr: ``numpy`` array, xarray object, or ``None``.

    Returns:
        ``nbytes`` of the array, or ``0`` when *arr* is ``None`` or invalid.
    """
    if arr is None:
        return 0
    try:
        return int(np.asarray(arr).nbytes)
    except Exception:
        return 0


def estimate_dataset_mb(d) -> dict:
    """Estimate RAM use for one :class:`~flim_phasors.data.PhasorData` instance.

    Counts the full TCSPC histogram (if loaded) plus calibrated phasor and
    lifetime maps (``real_cal``, ``imag_cal``, ``tau_phi``, etc.).

    Args:
        d: Dataset whose arrays should be measured.

    Returns:
        Dict with keys ``histogram_mb``, ``maps_mb``, ``total_mb``, and
        ``lazy`` (true when the histogram is not resident in memory).
    """
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
        )
    )
    return {
        "histogram_mb": hist / (1024 * 1024),
        "maps_mb": maps / (1024 * 1024),
        "total_mb": (hist + maps) / (1024 * 1024),
        "lazy": getattr(d, "signal_full", None) is None and bool(getattr(d, "sample_path", "")),
    }


def format_memory_line(d) -> str:
    """Format a one-line memory summary for the status bar or dataset info.

    Args:
        d: Dataset to summarize.

    Returns:
        ``"lazy (histogram not loaded)"`` or an approximate MB breakdown.
    """
    est = estimate_dataset_mb(d)
    if est["lazy"]:
        return "lazy (histogram not loaded)"
    return (
        f"~{est['histogram_mb']:.1f} MB histogram + ~{est['maps_mb']:.1f} MB maps "
        f"(~{est['total_mb']:.1f} MB total)"
    )
