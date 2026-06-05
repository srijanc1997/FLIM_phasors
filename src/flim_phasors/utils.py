"""Array and display helpers for FLIM phasor datasets.

Utilities for reshaping TCSPC histograms, photon-count maps, and human-readable
labels used in the GUI and phasor legend.
"""

import os

import numpy as np
from phasorpy.color import CATEGORICAL

from flim_phasors.constants import CATEGORICAL_NAMES


def to_2d(arr):
    """Reduce an array to two spatial dimensions for image display.

    Squeezes length-1 axes; if still higher-dimensional, indexes the leading
    slice until a 2D array remains.

    Args:
        arr: Input array or array-like histogram / map data.

    Returns:
        A 2D ``numpy`` array.
    """
    arr = np.squeeze(np.asarray(arr))
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def categorical_rgb(i):
    """Return an RGB tuple for phasor cursor / cluster index *i*.

    Colors cycle through the phasorpy categorical palette.

    Args:
        i: Zero-based cluster or cursor index.

    Returns:
        ``(r, g, b)`` floats in ``[0, 1]``.
    """
    c = np.asarray(CATEGORICAL)
    return tuple(float(x) for x in c[i % len(c)])


def categorical_name(i):
    """Human-readable label for cluster/cursor index (matches ring color).

    Args:
        i: Zero-based cluster or cursor index.

    Returns:
        A color name string from :data:`~flim_phasors.constants.CATEGORICAL_NAMES`.
    """
    return CATEGORICAL_NAMES[i % len(CATEGORICAL_NAMES)]


def reduce_signal(sig, channel):
    """Select a channel and collapse frames to a single histogram stack.

    Args:
        sig: xarray DataArray with TCSPC histogram dimension ``H``.
        channel: Emission channel index (clamped to available channels).

    Returns:
        DataArray with dimensions ``(Y, X, H)`` after channel selection,
        temporal summation or squeeze, and singleton-dimension removal.
    """
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
    """Compute per-pixel photon counts from a TCSPC histogram.

    Args:
        sig: xarray DataArray with ``H`` dimension, or a raw histogram array
            whose last axis is time bins.

    Returns:
        2D array of total photon counts per pixel.
    """
    if hasattr(sig, "sum") and "H" in getattr(sig, "dims", ()):
        pc = sig.sum(dim="H")
        for d in list(pc.dims):
            if d != "H" and pc.sizes.get(d, 1) == 1:
                pc = pc.squeeze(d, drop=True)
        return to_2d(np.asarray(pc.values, dtype=float))
    arr = np.asarray(sig, dtype=float)
    return to_2d(np.sum(arr, axis=-1))


def dataset_has_sample(d) -> bool:
    """Return whether a dataset holds loadable FLIM or LIF phasor data.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance.

    Returns:
        True when a full histogram or precomputed LIF phasor maps are present.
    """
    if getattr(d, "signal_full", None) is not None:
        return True
    return getattr(d, "load_source", "") == "lif_phasor" and getattr(d, "_lif_base_real", None) is not None


def _dataset_file_label(d, index=0) -> str:
    """Build a default label from the sample path or LIF series key.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance.
        index: Fallback image index when no path is set.

    Returns:
        Basename, or ``basename · series`` for multi-series LIF files.
    """
    if not d.sample_path:
        return f"image {index + 1}"
    base = os.path.basename(d.sample_path)
    key = (getattr(d, "lif_image_key", "") or "").strip()
    if key:
        series = key.split("/")[-1] if "/" in key else key
        return f"{base} · {series}"
    return base


def dataset_short_label(d, index=0):
    """Return the user-facing name for a dataset.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance.
        index: Fallback image index when no path is set.

    Returns:
        Custom ``display_name`` if set, otherwise the file/series label.
    """
    custom = (getattr(d, "display_name", "") or "").strip()
    if custom:
        return custom
    return _dataset_file_label(d, index)


def dataset_phasor_legend_label(d, index=0, *, include_group: bool = True) -> str:
    """Build a label for the multi-image phasor plot legend.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance.
        index: Fallback image index when no path is set.
        include_group: When true, prefix with ``group_name`` if defined.

    Returns:
        Short dataset label, optionally prefixed by group name.
    """
    name = dataset_short_label(d, index)
    if not include_group:
        return name
    group = (getattr(d, "group_name", "") or "").strip()
    if group:
        return f"{group} · {name}"
    return name


def dataset_display_label(d, index=0):
    """Return a list/overlay label with optional group prefix.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance.
        index: Fallback image index when no path is set.

    Returns:
        Same as :func:`dataset_phasor_legend_label` with group included.
    """
    return dataset_phasor_legend_label(d, index, include_group=True)
