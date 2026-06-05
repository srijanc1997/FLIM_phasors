"""Leica LIF/XLEF I/O for pre-computed phasor image triplets.

Leica LAS X can export FLIM results as separate ``Phasor Intensity``,
``Phasor Real``, and ``Phasor Imaginary`` images inside a container file.
This module discovers those series, applies LAS X automatic reference
calibration (phase rotation and modulation scaling), resolves photon-count
vs. normalized intensity for thresholding, and returns maps ready for
``PhasorData.load_lif_phasor``.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from phasorpy.io import phasor_from_lif
from phasorpy.phasor import phasor_transform

from flim_phasors.utils import to_2d

LIF_EXTENSIONS = (".lif", ".xlef", ".xlif", ".lof")


@dataclass(frozen=True)
class LifPhasorSeries:
    """Metadata for one FLIM measurement that includes LAS X phasor exports.

    Attributes:
        lif_path: Absolute path to the container file.
        image_key: Internal liffile path/key for the parent FLIM series.
        display_name: Human-readable series name for UI lists.
        shape_yx: Optional ``(height, width)`` from Phasor Intensity preview.
        frequency_mhz: Modulation frequency if known (often None at list time).
    """

    lif_path: str
    image_key: str
    display_name: str
    shape_yx: tuple[int, int] | None = None
    frequency_mhz: float | None = None


def is_lif_path(path: str) -> bool:
    """Return whether ``path`` has a recognized Leica container extension.

    Args:
        path: File path to test.

    Returns:
        True for ``.lif``, ``.xlef``, ``.xlif``, or ``.lof`` (case-insensitive).
    """
    return os.path.splitext(path)[1].lower() in LIF_EXTENSIONS


def _phasor_parent_key(im) -> str | None:
    """Return the parent FLIM series key for a ``Phasor Intensity`` child image.

    LAS X nests phasor exports under the original FLIM stack. Only images
    named ``Phasor Intensity`` are considered; their ``parent_image.path``
    identifies the series root.

    Args:
        im: A ``liffile`` image object.

    Returns:
        Parent series path string, or None if ``im`` is not Phasor Intensity.
    """
    if getattr(im, "name", "") != "Phasor Intensity":
        return None
    parent = im.parent_image
    if parent is None:
        return None
    return str(parent.path)


def _has_phasor_triplet(lif, image_key: str) -> bool:
    """Return whether all three LAS X phasor companion images exist for a series.

    Args:
        lif: Open ``liffile.LifFile`` instance.
        image_key: Parent series path/key under which phasor images are sought.

    Returns:
        True when Phasor Intensity, Real, and Imaginary are all present.
    """
    prefix = f".*{re.escape(image_key)}.*/"
    try:
        lif.images[prefix + "Phasor Intensity$"]
        lif.images[prefix + "Phasor Real$"]
        lif.images[prefix + "Phasor Imaginary$"]
    except Exception:
        return False
    return True


def list_lif_phasor_series(path: str) -> list[LifPhasorSeries]:
    """List FLIM series in a Leica file that include exported phasor triplets.

    Scans the container for ``Phasor Intensity`` images, verifies that Real and
    Imaginary companions exist, and returns sorted ``LifPhasorSeries`` entries.

    Args:
        path: Path to a Leica ``.lif`` / ``.xlef`` / similar container.

    Returns:
        Sorted list of series metadata; empty when no phasor exports are found.
    """
    import liffile

    norm = os.path.abspath(path)
    found: list[LifPhasorSeries] = []
    seen: set[str] = set()

    with liffile.LifFile(norm) as lif:
        for im in lif.images:
            key = _phasor_parent_key(im)
            if not key or key in seen:
                continue
            if not _has_phasor_triplet(lif, key):
                continue
            seen.add(key)

            shape_yx = None
            freq = None
            try:
                arr = im.asarray()
                if arr.ndim >= 2:
                    shape_yx = (int(arr.shape[-2]), int(arr.shape[-1]))
            except Exception:
                pass

            parent = im.parent_image
            display = getattr(parent, "name", None) or key
            if "/" in key and display == key.split("/")[-1]:
                display = key

            found.append(
                LifPhasorSeries(
                    lif_path=norm,
                    image_key=key,
                    display_name=str(display),
                    shape_yx=shape_yx,
                    frequency_mhz=freq,
                )
            )

    found.sort(key=lambda s: (os.path.basename(s.lif_path).lower(), s.image_key.lower()))
    return found


def lasx_channel_meta(attrs: dict[str, Any], channel: int = 0) -> dict[str, Any] | None:
    """Return per-channel LAS X phasor metadata from phasorpy attribute dict.

    Args:
        attrs: Attribute dictionary returned by ``phasor_from_lif``.
        channel: Channel index (clamped to available entries).

    Returns:
        Channel metadata dict, or None when ``flim_phasor_channels`` is absent.
    """
    channels = attrs.get("flim_phasor_channels") or []
    if not channels:
        return None
    return channels[min(max(0, int(channel)), len(channels) - 1)]


def lasx_intensity_threshold(
    attrs: dict[str, Any],
    channel: int = 0,
    *,
    photon_image: bool = False,
) -> float:
    """Convert LAS X ``IntensityThreshold`` to the scale used for map masking.

    On the photon-count ``Intensity`` image the threshold is used directly.
    On normalized ``Phasor Intensity`` it is divided by the acquisition
    ``samples`` count to match phasorpy's mean-intensity convention.

    Args:
        attrs: Metadata from ``phasor_from_lif``.
        channel: FLIM channel index for threshold lookup.
        photon_image: True when threshold applies to raw photon counts.

    Returns:
        Threshold value on the appropriate intensity scale; 0.0 when unknown.
    """
    ch = lasx_channel_meta(attrs, channel)
    if not ch:
        return 0.0
    raw_thr = float(ch.get("IntensityThreshold", 0))
    if photon_image:
        return raw_thr
    samples = float(attrs.get("samples", 1) or 1)
    if samples <= 0:
        samples = 1.0
    return raw_thr / samples


def load_lif_photon_image(path: str, image_key: str | None = None) -> np.ndarray | None:
    """Load the LAS X photon-count ``Intensity`` image (not Phasor Intensity).

    Some exports include a separate cumulative photon map used for display and
    intensity thresholding instead of normalized phasor mean intensity.

    Args:
        path: Leica container file path.
        image_key: Optional parent series key; empty matches any series.

    Returns:
        2-D float32 photon map, or None if the image is missing or unreadable.
    """
    import liffile

    prefix = "" if not image_key else f".*{re.escape(image_key)}.*/"
    try:
        with liffile.LifFile(path) as lif:
            im = lif.images[prefix + "Intensity$"]
            return to_2d(np.asarray(im.asarray(), dtype=np.float32))
    except Exception:
        return None


def apply_lasx_phasor_calibration(
    real: np.ndarray,
    imag: np.ndarray,
    attrs: dict[str, Any],
    *,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply LAS X automatic reference calibration to exported phasor maps.

    Uncalibrated Real/Imaginary exports are rotated by
    ``-AutomaticReferencePhase`` and scaled in modulation by the inverse of
    ``AutomaticReferenceAmplitude`` using ``phasor_transform``, matching LAS X
    instrument reference correction.

    Args:
        real: Uncalibrated g map from the file.
        imag: Uncalibrated s map from the file.
        attrs: Metadata dict containing ``flim_phasor_channels``.
        channel: Channel whose automatic reference values to apply.

    Returns:
        Tuple ``(real_cal, imag_cal, info)`` where ``info`` records whether
        calibration was applied and the reference phase/modulation used.
    """
    ch = lasx_channel_meta(attrs, channel)
    if not ch:
        return real, imag, {"applied": False}

    phase_deg = float(ch["AutomaticReferencePhase"])
    modulation = float(ch["AutomaticReferenceAmplitude"])
    if modulation == 0.0:
        modulation = 1.0

    real, imag = phasor_transform(
        real,
        imag,
        -math.radians(phase_deg),
        1.0 / modulation,
    )
    return (
        to_2d(np.asarray(real, dtype=np.float32)),
        to_2d(np.asarray(imag, dtype=np.float32)),
        {
            "applied": True,
            "channel": int(ch.get("Channel", channel)),
            "reference_phase_deg": phase_deg,
            "reference_modulation": modulation,
            "intensity_threshold": lasx_intensity_threshold(attrs, channel),
        },
    )


def lif_intensity_for_maps(
    path: str,
    image_key: str | None,
    phasor_mean: np.ndarray,
    attrs: dict[str, Any],
    *,
    channel: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose intensity map and threshold metadata for filtering and display.

    Prefers the separate photon ``Intensity`` image when it matches the phasor
    map shape; otherwise falls back to ``Phasor Intensity`` mean. Updates
    ``attrs`` with ``uses_photon_intensity`` and ``lasx_intensity_threshold``.

    Args:
        path: Leica container path (for photon image lookup).
        image_key: Parent series key passed to ``load_lif_photon_image``.
        phasor_mean: Mean intensity from phasor export (fallback).
        attrs: Mutable metadata dict to annotate.
        channel: Channel for LAS X threshold metadata.

    Returns:
        Tuple ``(intensity_map, attrs)`` with the selected 2-D intensity array.
    """
    photon = load_lif_photon_image(path, image_key)
    if photon is not None and photon.shape == phasor_mean.shape:
        attrs["uses_photon_intensity"] = True
        attrs["photon_image"] = photon
        attrs["lasx_intensity_threshold"] = lasx_intensity_threshold(
            attrs, channel, photon_image=True)
        return np.asarray(photon, dtype=np.float32), attrs
    attrs["uses_photon_intensity"] = False
    attrs["lasx_intensity_threshold"] = lasx_intensity_threshold(
        attrs, channel, photon_image=False)
    return phasor_mean, attrs


def load_lif_phasor_maps(
    path: str,
    image_key: str | None = None,
    *,
    channel: int = 0,
    apply_lasx_calibration: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load mean, g, and s phasor maps and metadata from a Leica container.

    Uses ``phasorpy.io.phasor_from_lif``, optionally applies LAS X automatic
    reference calibration, resolves photon vs. phasor intensity for masking,
    and annotates channel count and calibration info in ``attrs``.

    Args:
        path: Leica container file path.
        image_key: Optional internal series key; None loads the default series.
        channel: Emission channel for LAS X calibration metadata.
        apply_lasx_calibration: When True, rotate/scale Real and Imaginary
            using ``AutomaticReferencePhase`` / ``AutomaticReferenceAmplitude``.

    Returns:
        Tuple ``(mean, real, imag, attrs)`` of 2-D float32 maps and a metadata
        dict (frequency, LAS X thresholds, calibration record, etc.).
    """
    mean, real, imag, attrs = phasor_from_lif(path, image=image_key)
    mean = to_2d(np.asarray(mean, dtype=np.float32))
    real = to_2d(np.asarray(real, dtype=np.float32))
    imag = to_2d(np.asarray(imag, dtype=np.float32))
    attrs = dict(attrs)

    cal_info = {"applied": False}
    if apply_lasx_calibration:
        real, imag, cal_info = apply_lasx_phasor_calibration(
            real, imag, attrs, channel=channel)
    attrs["lasx_calibration"] = cal_info
    mean, attrs = lif_intensity_for_maps(path, image_key, mean, attrs, channel=channel)
    channels = attrs.get("flim_phasor_channels") or []
    attrs["n_phasor_channels"] = len(channels)

    return mean, real, imag, attrs
