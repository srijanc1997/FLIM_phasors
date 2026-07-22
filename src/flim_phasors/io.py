"""Load FLIM data from PicoQuant PTU, Imspector TIFF, and Leica LIF phasor maps.

Entry points for reading TCSPC histogram stacks used in phasor lifetime analysis.
LIF phasor-map loading is handled separately in :mod:`flim_phasors.lif_io`.
"""

from __future__ import annotations

import os

import numpy as np
from phasorpy.io import signal_from_imspector_tiff, signal_from_ptu

# --- unused (focused cleanup): uncomment if needed ---
# Legacy alias — phasor maps only, no full histogram cache
# def reference_phasor(ref_path: str, channel: int, harmonic):
#     """Return (mean, real, imag) reference phasor maps (histogram not kept in RAM)."""
#     from flim_phasors.calibration import get_cached_reference_phasor
#
#     cal = get_cached_reference_phasor(ref_path, channel, harmonic)
#     if cal._maps is None:
#         raise ValueError(f"No reference phasor for {ref_path}")
#     return cal._maps
#
#
# def clear_signal_caches():
#     from flim_phasors.calibration import clear_calibration_cache
#
#     clear_calibration_cache()
#
#
# def _cache_key(path: str) -> str:
#     return os.path.normcase(os.path.abspath(path))


def file_extension(path: str) -> str:
    """Return the lower-case file extension including the leading dot.

    Args:
        path: File path or name.

    Returns:
        Extension string, e.g. ``".ptu"`` or ``".tif"``.
    """
    return os.path.splitext(path)[1].lower()


def is_supported_flim_path(path: str) -> bool:
    """Return whether *path* is a supported histogram or LIF phasor file.

    Args:
        path: File path to inspect.

    Returns:
        True for PicoQuant ``.ptu``, Imspector ``.tif``/``.tiff``, or Leica
        ``.lif``/``.xlef`` paths.
    """
    from flim_phasors.lif_io import is_lif_path

    return file_extension(path) in (".ptu", ".tif", ".tiff") or is_lif_path(path)


def flim_channel_count(path: str) -> int | None:
    """Return the number of emission channels without decoding the histogram.

    Reads only the file header/metadata, which is cheap compared to decoding the
    full TCSPC stack. Used to populate the channel selector before a fast
    single-channel load.

    Args:
        path: Path to a ``.ptu`` or ``.tif``/``.tiff`` file.

    Returns:
        Channel count, or ``None`` when it cannot be determined cheaply
        (e.g. Imspector TIFF, which must be opened to learn its shape).
    """
    ext = file_extension(path)
    if ext == ".ptu":
        try:
            import ptufile

            with ptufile.PtuFile(path) as ptu:
                return max(1, int(ptu.number_channels))
        except Exception:
            return None
    return None


def flim_frame_count(path: str) -> int | None:
    """Return the number of time frames without fully decoding the histogram.

    Args:
        path: Path to a ``.ptu`` or ``.tif``/``.tiff`` file.

    Returns:
        Frame count (≥1), or ``None`` when unknown.
    """
    ext = file_extension(path)
    if ext == ".ptu":
        try:
            import ptufile

            with ptufile.PtuFile(path) as ptu:
                n = int(getattr(ptu, "number_images", 0) or 0)
                return max(1, n) if n else 1
        except Exception:
            return None
    return None


def load_flim_signal(path: str, *, channel=None, frame=-1, dtype=np.uint32):
    """Load a TCSPC histogram stack from PicoQuant or Imspector files.

    Args:
        path: Path to a ``.ptu`` or ``.tif``/``.tiff`` file.
        channel: Emission channel to decode (``None`` keeps all channels). For
            PTU files a single channel is decoded directly, which lowers peak
            memory and binning cost; for TIFF the ``C`` axis is sliced after read.
        frame: Frame index for multi-frame stacks; ``-1`` sums all frames.
        dtype: Target integer dtype for histogram counts.

    Returns:
        xarray DataArray with dimensions including ``H`` (time bins) and
        spatial axes ``Y``, ``X`` (and optionally ``C``, ``T``).

    Raises:
        ValueError: If the file extension is not ``.ptu``, ``.tif``, or
            ``.tiff``.
    """
    ext = file_extension(path)
    if ext == ".ptu":
        return signal_from_ptu(path, channel=channel, frame=frame, dtype=dtype)
    if ext in (".tif", ".tiff"):
        sig = signal_from_imspector_tiff(path)
        if dtype is not None:
            sig = sig.astype(dtype)
        if "C" in sig.dims:
            # Record true channel count before optional single-channel slice
            # (fast-load still reads the whole TIFF, but keeps the combo accurate).
            sig.attrs = dict(sig.attrs or {})
            sig.attrs["n_channels"] = int(sig.sizes["C"])
            if channel is not None:
                sig = sig.isel(C=int(min(channel, sig.sizes["C"] - 1)))
        if "T" in sig.dims:
            if frame == -1 and sig.sizes.get("T", 1) > 1:
                n_t = int(sig.sizes["T"])
                sig.attrs = dict(sig.attrs or {})
                sig.attrs["n_frames"] = n_t
                sig = sig.sum("T")
            elif frame is not None and frame >= 0:
                n_t = int(sig.sizes["T"])
                sig.attrs = dict(sig.attrs or {})
                sig.attrs["n_frames"] = n_t
                sig = sig.isel(T=int(min(frame, sig.sizes["T"] - 1)))
            else:
                sig = sig.squeeze("T", drop=True)
        return sig
    raise ValueError(f"Unsupported FLIM file type: {ext!r} ({path})")
