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

    Indexing wraps modulo the length of the phasorpy categorical palette,
    the same cyclic scheme :func:`categorical_rgb` uses, so a cursor's color
    swatch and its text label always refer to the same palette entry. Named
    colors only cover the first :data:`CATEGORICAL_NAMES` entries; indices
    beyond that fall back to a generic ``"color N"`` label rather than
    wrapping back to a name that no longer matches the swatch.

    Args:
        i: Zero-based cluster or cursor index.

    Returns:
        A color name string, or ``"color N"`` past the named entries.
    """
    idx = i % len(CATEGORICAL)
    if idx < len(CATEGORICAL_NAMES):
        return CATEGORICAL_NAMES[idx]
    return f"color {idx + 1}"


def reduce_signal(sig, channel):
    """Select a channel and collapse frames to a single histogram stack.

    Multi-channel acquisitions carry an extra ``C`` dimension that must be
    resolved to a single emission channel before phasor calculation;
    ``channel`` is clamped to the last available index rather than raising,
    so stale UI selections from a previously loaded file with fewer channels
    degrade gracefully. Multiple time frames (``T``) are summed (not
    averaged) to preserve total photon counts, matching how a single-frame
    acquisition's counts are interpreted downstream.

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

    Sums the time-bin axis (``H`` for xarray input, or the last axis for a
    plain NumPy array) to collapse the full decay curve into a single
    intensity value per pixel, in raw photon counts (not normalized or
    scaled). This is the quantity thresholded against the "Min photons"
    setting during calibration, and the basis for the raw brightfield/photon
    maps shown in the GUI and export.

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

    Three independent data sources count as "having a sample": a full raw
    TCSPC histogram (freshly loaded PTU/Imspector file), precomputed LIF
    phasor base maps (LAS X exports that skip histogram decoding entirely),
    or calibrated maps restored from a session bundle with no raw signal at
    all. Callers use this to decide whether processing/export controls
    should be enabled for a given slot in a multi-image session.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance.

    Returns:
        True when a full histogram, precomputed LIF phasor maps, or restored
        session-bundle maps (``real_cal`` without raw signal) are present.
    """
    if getattr(d, "signal_full", None) is not None:
        return True
    if getattr(d, "_lif_base_real", None) is not None:
        return True
    # Session-bundle / map-only restores have calibrated maps but no histogram.
    return getattr(d, "real_cal", None) is not None


def _dataset_file_label(d, index=0) -> str:
    """Build a default label from the sample path or LIF series key.

    Used as the fallback when a user has not set a custom ``display_name``.
    For plain FLIM files this is just the file's basename; for LIF
    containers with multiple series, the series name is appended after a
    middle-dot separator (``basename · series``) since the basename alone
    would be ambiguous across series from the same container file.

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

    This is the base label used throughout the GUI (multi-image list, combo
    boxes, table rows) before any group prefixing is applied. A user-supplied
    ``display_name`` always takes precedence over the derived file/series
    label, letting users rename samples without losing the ability to trace
    them back to a source file via :func:`_dataset_file_label`.

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

    Legend space is limited, so this builds on the short
    :func:`dataset_short_label` rather than the full file path. When
    ``include_group`` is true and the dataset has a non-empty
    ``group_name``, the group is prepended with a middle-dot separator
    (``group · name``) so compare-mode legends visually cluster related
    samples together by group.

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

    Thin wrapper around :func:`dataset_phasor_legend_label` with
    ``include_group`` always true, kept as a separate name so call sites in
    the multi-image list, compare table, and session/export metadata read
    clearly without repeating the ``include_group=True`` argument
    everywhere it is needed.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance.
        index: Fallback image index when no path is set.

    Returns:
        Same as :func:`dataset_phasor_legend_label` with group included.
    """
    return dataset_phasor_legend_label(d, index, include_group=True)
