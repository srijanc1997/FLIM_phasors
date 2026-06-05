"""Reference phasor calibration for instrument response correction.

Computes reference (g, s) maps from a calibration measurement (e.g. uniform
fluorophore of known lifetime) and stores spatial mean/intensity maps rather
than raw TCSPC histograms. Sample phasor maps are corrected via
``phasorpy.lifetime.phasor_calibrate`` using either full spatial reference
maps or uniform scalar g/s values from manual entry or saved session metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from phasorpy.phasor import phasor_from_signal

from flim_phasors.io import load_flim_signal
from flim_phasors.utils import reduce_signal, to_2d


@dataclass
class ReferenceCalibration:
    """Instrument reference state used to calibrate sample phasor maps.

    Holds either spatial reference maps (mean intensity, g, s per pixel) from
    a decoded reference file, or scalar manual g/s values. Calibration is
    considered active when ``use_manual`` is set or ``values_ready`` after
    loading or setting maps.

    Attributes:
        source_path: Path to the reference measurement file, if any.
        channel: Emission channel index used when building reference maps.
        n_channels: Number of channels in the reference file.
        harmonic: Harmonic index used for phasor transform (1 = fundamental).
        mean_g: Spatial mean of reference g after ``set_maps``.
        mean_s: Spatial mean of reference s after ``set_maps``.
        mean_intensity: Spatial mean of reference photon counts.
        use_manual: When True, ``manual_g`` / ``manual_s`` override file maps.
        manual_g: User-entered reference g for uniform calibration.
        manual_s: User-entered reference s for uniform calibration.
        manual_mean: Reference intensity scale for manual mode.
        values_ready: True after maps or means have been populated.
    """

    source_path: str = ""
    channel: int = 0
    n_channels: int = 1
    harmonic: int = 1
    mean_g: float = 0.0
    mean_s: float = 0.0
    mean_intensity: float = 1.0
    use_manual: bool = False
    manual_g: float = 0.0
    manual_s: float = 0.0
    manual_mean: float = 1.0
    values_ready: bool = False
    _maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, repr=False)

    @property
    def is_active(self) -> bool:
        """Return whether calibration values are available for sample preprocessing.

        Returns:
            True when manual g/s are enabled or reference maps/means are ready.
        """
        return self.use_manual or self.values_ready

    @property
    def has_spatial_maps(self) -> bool:
        """Return whether per-pixel reference maps are stored (not scalar-only).

        Returns:
            True if ``set_maps`` populated ``_maps``; False for manual or JSON-only g/s.
        """
        return self._maps is not None

    def set_maps(self, rmean, rreal, rimag):
        """Store reference intensity and phasor maps and update scalar means.

        Args:
            rmean: Reference mean photon count or intensity map.
            rreal: Reference real (g) phasor map.
            rimag: Reference imaginary (s) phasor map.
        """
        rmean = to_2d(np.asarray(rmean, dtype=float))
        rreal = to_2d(np.asarray(rreal, dtype=float))
        rimag = to_2d(np.asarray(rimag, dtype=float))
        self._maps = (rmean, rreal, rimag)
        finite = np.isfinite(rreal) & np.isfinite(rimag)
        if np.any(finite):
            self.mean_g = float(np.nanmean(rreal[finite]))
            self.mean_s = float(np.nanmean(rimag[finite]))
            self.mean_intensity = float(np.nanmean(rmean[finite]))
        else:
            self.mean_g = self.mean_s = 0.0
            self.mean_intensity = 1.0
        self.values_ready = True

    def maps_for_shape(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return reference maps sized for ``phasor_calibrate`` on a sample.

        Manual mode fills ``shape`` with uniform manual g/s and intensity.
        When only scalar means exist (e.g. loaded calibration JSON), uniform
        fields are built from ``mean_g``, ``mean_s``, and ``mean_intensity``.
        Stored spatial maps are returned unchanged when their shape matches.

        Args:
            shape: Target spatial shape ``(height, width)`` of the sample maps.

        Returns:
            Tuple ``(rmean, rreal, rimag)`` of float arrays with shape ``shape``.
        """
        if self.use_manual:
            g = float(self.manual_g)
            s = float(self.manual_s)
            m = float(self.manual_mean)
            return (
                np.full(shape, m, dtype=float),
                np.full(shape, g, dtype=float),
                np.full(shape, s, dtype=float),
            )
        if self._maps is None:
            # Saved g/s only (Load cal JSON, session metadata) — uniform reference field.
            return (
                np.full(shape, float(self.mean_intensity), dtype=float),
                np.full(shape, float(self.mean_g), dtype=float),
                np.full(shape, float(self.mean_s), dtype=float),
            )
        rmean, rreal, rimag = self._maps
        if rreal.shape == shape:
            return rmean, rreal, rimag
        # Broadcast scalar means if stored maps differ (e.g. after resize)
        return (
            np.full(shape, self.mean_intensity, dtype=float),
            np.full(shape, self.mean_g, dtype=float),
            np.full(shape, self.mean_s, dtype=float),
        )

    def clear(self):
        """Reset all calibration fields to defaults and drop stored maps."""
        self.source_path = ""
        self.channel = 0
        self._maps = None
        self.mean_g = self.mean_s = 0.0
        self.mean_intensity = 1.0
        self.use_manual = False
        self.values_ready = False


def compute_reference_phasor(
    ref_path: str,
    channel: int = 0,
    harmonic: int | list = 1,
) -> ReferenceCalibration:
    """Build reference phasor maps from a FLIM file and discard the histogram.

    Loads the reference TCSPC stack, reduces to one channel, transforms along
    the time axis (``axis="H"``) with ``phasor_from_signal``, and stores 2-D
    mean/g/s maps in a ``ReferenceCalibration``. The raw signal is released
    after transform to limit memory use.

    Args:
        ref_path: Path to the reference FLIM container (e.g. .lif, .ptu).
        channel: Emission channel index to use after multi-channel reduction.
        harmonic: Single harmonic index or list (e.g. ``[1, 2]`` for dual-harmonic);
            only the first harmonic's g/s maps are stored when a list is given.

    Returns:
        Populated ``ReferenceCalibration`` with ``values_ready`` True.
    """
    harm = harmonic
    if isinstance(harmonic, int):
        harm = int(harmonic)
    else:
        harm = list(harmonic)
    sig = load_flim_signal(ref_path, channel=None, frame=-1, dtype=np.uint32)
    n_channels = int(sig.sizes["C"]) if "C" in sig.dims else 1
    rsig = reduce_signal(sig, int(channel))
    del sig
    rmean, rreal, rimag = phasor_from_signal(rsig, axis="H", harmonic=harm)
    del rsig
    cal = ReferenceCalibration(
        source_path=ref_path,
        channel=int(channel),
        n_channels=n_channels,
        harmonic=harm if isinstance(harm, int) else harm[0],
    )
    if isinstance(harm, list):
        cal.set_maps(to_2d(rmean), to_2d(rreal[0]), to_2d(rimag[0]))
    else:
        cal.set_maps(rmean, rreal, rimag)
    return cal


# Small LRU-style cache of ReferenceCalibration by (path, channel, harmonic)
_CAL_CACHE: dict[tuple, ReferenceCalibration] = {}


# --- unused (focused cleanup): uncomment if needed ---
# def get_cached_reference_phasor(ref_path: str, channel: int, harmonic) -> ReferenceCalibration:
#     harm_key = tuple(harmonic) if isinstance(harmonic, (list, tuple)) else (int(harmonic),)
#     import os
#
#     key = (os.path.normcase(os.path.abspath(ref_path)), int(channel), harm_key)
#     if key not in _CAL_CACHE:
#         _CAL_CACHE[key] = compute_reference_phasor(ref_path, channel, harmonic)
#     return _CAL_CACHE[key]


def clear_calibration_cache():
    """Clear the in-memory cache of computed ``ReferenceCalibration`` objects."""
    _CAL_CACHE.clear()
