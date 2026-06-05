"""Load FLIM data from PicoQuant PTU and Imspector TIFF."""

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
    return os.path.splitext(path)[1].lower()


def is_supported_flim_path(path: str) -> bool:
    return file_extension(path) in (".ptu", ".tif", ".tiff")


def load_flim_signal(path: str, *, channel=None, frame=-1, dtype=np.uint32):
    """Load a TCSPC histogram stack from PicoQuant .ptu or Imspector .tif(f)."""
    ext = file_extension(path)
    if ext == ".ptu":
        return signal_from_ptu(path, channel=channel, frame=frame, dtype=dtype)
    if ext in (".tif", ".tiff"):
        sig = signal_from_imspector_tiff(path)
        if dtype is not None:
            sig = sig.astype(dtype)
        if "T" in sig.dims:
            if frame == -1 and sig.sizes.get("T", 1) > 1:
                sig = sig.sum("T")
            elif frame is not None and frame >= 0:
                sig = sig.isel(T=int(min(frame, sig.sizes["T"] - 1)))
            else:
                sig = sig.squeeze("T", drop=True)
        return sig
    raise ValueError(f"Unsupported FLIM file type: {ext!r} ({path})")
