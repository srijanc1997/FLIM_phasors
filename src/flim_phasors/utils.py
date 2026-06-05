"""Array and display helpers."""

import os

import numpy as np
from phasorpy.color import CATEGORICAL

from flim_phasors.constants import CATEGORICAL_NAMES


def to_2d(arr):
    """Squeeze length-1 dims; if still >2D, take leading index until 2D."""
    arr = np.squeeze(np.asarray(arr))
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def categorical_rgb(i):
    c = np.asarray(CATEGORICAL)
    return tuple(float(x) for x in c[i % len(c)])


def categorical_name(i):
    """Human-readable label for cluster/cursor index (matches ring color)."""
    return CATEGORICAL_NAMES[i % len(CATEGORICAL_NAMES)]


def reduce_signal(sig, channel):
    """Select a channel and collapse frames -> DataArray with dims (Y, X, H)."""
    if "C" in sig.dims:
        sig = sig.isel(C=int(min(channel, sig.sizes["C"] - 1)))
    if "T" in sig.dims:
        if sig.sizes["T"] > 1:
            sig = sig.sum("T")
        else:
            sig = sig.squeeze("T", drop=True)
    for d in list(sig.dims):
        if d != "H" and sig.sizes.get(d, 2) == 1:
            sig = sig.squeeze(d, drop=True)
    return sig


def photon_count_from_signal(sig):
    """Total photon count per pixel = sum of the TCSPC histogram (axis H)."""
    if hasattr(sig, "sum") and "H" in getattr(sig, "dims", ()):
        pc = sig.sum(dim="H")
        for d in list(pc.dims):
            if d != "H" and pc.sizes.get(d, 1) == 1:
                pc = pc.squeeze(d, drop=True)
        return to_2d(np.asarray(pc.values, dtype=float))
    arr = np.asarray(sig, dtype=float)
    return to_2d(np.sum(arr, axis=-1))


def dataset_short_label(d, index=0):
    return os.path.basename(d.sample_path) if d.sample_path else f"image {index + 1}"


def dataset_display_label(d, index=0):
    """Filename with optional group prefix for lists and overlay legends."""
    base = dataset_short_label(d, index)
    group = (getattr(d, "group_name", "") or "").strip()
    if group:
        return f"{group} · {base}"
    return base
