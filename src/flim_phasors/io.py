"""Load FLIM data from PicoQuant PTU and Imspector TIFF; decode caches."""

from __future__ import annotations

import os

import numpy as np
from phasorpy.io import signal_from_imspector_tiff, signal_from_ptu
from phasorpy.phasor import phasor_from_signal

from flim_phasors.utils import reduce_signal, to_2d

_REF_SIGNAL_CACHE: dict[str, object] = {}
_REF_PHASOR_CACHE: dict[tuple, tuple] = {}


def _cache_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


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


def load_reference_signal(path: str):
    """Decode a reference file once; reuse the in-memory histogram for later Apply calls."""
    key = _cache_key(path)
    if key not in _REF_SIGNAL_CACHE:
        _REF_SIGNAL_CACHE[key] = load_flim_signal(path, channel=None, frame=-1, dtype=np.uint32)
    return _REF_SIGNAL_CACHE[key]


def reference_phasor(ref_path: str, channel: int, harmonic):
    """Cached (mean, real, imag) for a reference file at the given channel and harmonic(s)."""
    harm_key = tuple(harmonic) if isinstance(harmonic, (list, tuple)) else (int(harmonic),)
    key = (_cache_key(ref_path), int(channel), harm_key)
    if key not in _REF_PHASOR_CACHE:
        rsig = reduce_signal(load_reference_signal(ref_path), channel)
        rmean, rreal, rimag = phasor_from_signal(rsig, axis="H", harmonic=harmonic)
        _REF_PHASOR_CACHE[key] = (rmean, rreal, rimag)
    return _REF_PHASOR_CACHE[key]


def clear_signal_caches():
    _REF_SIGNAL_CACHE.clear()
    _REF_PHASOR_CACHE.clear()
