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

from flim_phasors.io import flim_channel_count, load_flim_signal
from flim_phasors.utils import reduce_signal, to_2d


def _weighted_gs(rmean, rreal, rimag) -> tuple[float, float, float]:
    """Compute an intensity-weighted mean reference phasor coordinate.

    Pixels with more photons contribute more to the average, matching
    phasorpy's ``phasor_center`` mean method. This gives a single
    representative (g, s) for the whole reference measurement, which is used
    to build uniform reference maps in :meth:`ReferenceCalibration.maps_for_shape`
    when full spatial reference maps are unavailable or mismatched in shape.

    Args:
        rmean: Reference mean photon count or intensity map.
        rreal: Reference real (g) phasor map, same shape as ``rmean``.
        rimag: Reference imaginary (s) phasor map, same shape as ``rmean``.

    Returns:
        Tuple ``(g, s, mean_intensity)``. Falls back to an unweighted mean
        over finite pixels when every intensity is zero, and to
        ``(0.0, 0.0, 1.0)`` when no pixel has finite g, s, and intensity.
    """
    rmean = np.asarray(rmean, dtype=float)
    rreal = np.asarray(rreal, dtype=float)
    rimag = np.asarray(rimag, dtype=float)
    finite = np.isfinite(rreal) & np.isfinite(rimag) & np.isfinite(rmean)
    if not np.any(finite):
        return 0.0, 0.0, 1.0
    weights = np.clip(rmean[finite], 0.0, None)
    wsum = float(weights.sum())
    if wsum > 0:
        return (
            float(np.average(rreal[finite], weights=weights)),
            float(np.average(rimag[finite], weights=weights)),
            float(np.mean(rmean[finite])),
        )
    # Unweighted fallback when all intensities are zero (still finite g/s).
    return (
        float(np.mean(rreal[finite])),
        float(np.mean(rimag[finite])),
        1.0,
    )


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
        mean_g: Spatial mean of reference g after ``set_maps`` (first harmonic).
        mean_s: Spatial mean of reference s after ``set_maps`` (first harmonic).
        mean_intensity: Spatial mean of reference photon counts.
        harmonic_gs: Optional per-harmonic ``(g, s)`` pairs for PAW-FLIM.
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
    harmonic_gs: list[tuple[float, float]] | None = None
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

        This is the gate checked before running ``phasor_calibrate`` on a
        sample: calibration is considered active either because the user
        has entered manual reference g/s values (``use_manual``), or because
        a reference measurement was successfully loaded/computed
        (``values_ready``). When neither is true, samples are processed
        uncalibrated.

        Returns:
            True when manual g/s are enabled or reference maps/means are ready.
        """
        return self.use_manual or self.values_ready

    @property
    def has_spatial_maps(self) -> bool:
        """Return whether per-pixel reference maps are stored (not scalar-only).

        Distinguishes a reference that was freshly computed from a decoded
        file (spatial mean/g/s maps kept in ``_maps``) from one restored from
        manual entry or a saved calibration JSON, which only carries scalar
        means. Spatial maps allow per-pixel calibration when their shape
        matches the sample; otherwise :meth:`maps_for_shape` falls back to
        uniform scalar fields regardless of this flag.

        Returns:
            True if ``set_maps`` populated ``_maps``; False for manual or JSON-only g/s.
        """
        return self._maps is not None

    def set_maps(self, rmean, rreal, rimag):
        """Store reference intensity and phasor maps and update scalar means.

        Called after computing a single-harmonic reference phasor from a
        decoded FLIM file. Maps are reduced to 2-D via :func:`to_2d` and kept
        for per-pixel calibration when their shape matches a sample; the
        scalar ``mean_g``/``mean_s``/``mean_intensity`` are simultaneously
        derived via an intensity-weighted average (see :func:`_weighted_gs`)
        so calibration also works when spatial maps cannot be used directly
        (mismatched shape) or are not persisted (e.g. saved-JSON reload).

        Args:
            rmean: Reference mean photon count or intensity map.
            rreal: Reference real (g) phasor map.
            rimag: Reference imaginary (s) phasor map.
        """
        rmean = to_2d(np.asarray(rmean, dtype=float))
        rreal = to_2d(np.asarray(rreal, dtype=float))
        rimag = to_2d(np.asarray(rimag, dtype=float))
        self._maps = (rmean, rreal, rimag)
        self.mean_g, self.mean_s, self.mean_intensity = _weighted_gs(rmean, rreal, rimag)
        self.harmonic_gs = [(self.mean_g, self.mean_s)]
        self.values_ready = True

    def set_harmonic_means(self, rmean, rreal, rimag):
        """Store per-harmonic weighted mean g/s (PAW-FLIM dual-harmonic path).

        Does not keep full spatial maps in RAM. ``mean_g`` / ``mean_s`` are set
        from the first harmonic.

        Args:
            rmean: Reference intensity map ``(Y, X)``.
            rreal: Reference g with leading harmonic axis ``(n_harm, Y, X)``.
            rimag: Reference s with leading harmonic axis ``(n_harm, Y, X)``.
        """
        rmean = to_2d(np.asarray(rmean, dtype=float))
        rreal = np.asarray(rreal, dtype=float)
        rimag = np.asarray(rimag, dtype=float)
        if rreal.ndim != 3 or rimag.ndim != 3:
            raise ValueError(
                f"Expected harmonic g/s with shape (n, Y, X); got {rreal.shape}, {rimag.shape}")
        if rreal.shape != rimag.shape:
            raise ValueError(f"g/s harmonic shapes differ: {rreal.shape} vs {rimag.shape}")
        self._maps = None  # PAW path keeps scalars only; spatial maps would be wrong shape.
        self.harmonic_gs = []
        for i in range(rreal.shape[0]):
            g, s, inten = _weighted_gs(rmean, rreal[i], rimag[i])
            self.harmonic_gs.append((g, s))
            if i == 0:
                self.mean_g, self.mean_s, self.mean_intensity = g, s, inten
        self.values_ready = True

    def maps_for_shape(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return reference maps sized for ``phasor_calibrate`` on a sample.

        Manual mode fills ``shape`` with uniform manual g/s and intensity.
        When only scalar means exist (e.g. loaded calibration JSON), uniform
        fields are built from ``mean_g``, ``mean_s``, and ``mean_intensity``.
        Stored spatial maps are returned unchanged when their shape matches.

        For dual-harmonic sample arrays ``shape=(n_harm, Y, X)``, returns
        ``rmean`` of shape ``(Y, X)`` and ``rreal``/``rimag`` of shape
        ``(n_harm, Y, X)`` — the layout ``phasor_calibrate`` requires.

        Args:
            shape: Sample ``real`` array shape: ``(Y, X)`` or ``(n_harm, Y, X)``.

        Returns:
            Tuple ``(rmean, rreal, rimag)`` shaped for ``phasor_calibrate``.
        """
        # Dual-harmonic PAW-FLIM: real/imag are (n_harm, Y, X); mean must stay (Y, X).
        if len(shape) == 3:
            n_harm, *spatial = shape
            spatial = tuple(spatial)
            rmean, rreal_2d, rimag_2d = self._scalar_or_spatial_2d(spatial)
            rreal, rimag = self._expand_gs_to_harmonics(
                n_harm, spatial, rreal_2d, rimag_2d)
            return rmean, rreal, rimag

        return self._scalar_or_spatial_2d(shape)

    def _scalar_or_spatial_2d(self, shape: tuple[int, ...]):
        """Build or return 2-D reference maps for a spatial ``shape``.

        Chooses among three sources in priority order: manual g/s/mean when
        ``use_manual`` is set, stored spatial maps from ``set_maps`` when
        their shape matches ``shape``, or uniform fields broadcast from the
        scalar ``mean_g``/``mean_s``/``mean_intensity`` otherwise (covers
        loaded calibration JSON and shape mismatches from a different ROI or
        resolution).

        Args:
            shape: Target 2-D spatial shape ``(Y, X)`` matching the sample map.

        Returns:
            Tuple ``(rmean, rreal, rimag)`` of arrays shaped like ``shape``.
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
            return (
                np.full(shape, float(self.mean_intensity), dtype=float),
                np.full(shape, float(self.mean_g), dtype=float),
                np.full(shape, float(self.mean_s), dtype=float),
            )
        rmean, rreal, rimag = self._maps
        if rreal.shape == shape:
            return rmean, rreal, rimag
        # Shape mismatch (different ROI/resolution): fall back to uniform scalar g/s.
        return (
            np.full(shape, self.mean_intensity, dtype=float),
            np.full(shape, self.mean_g, dtype=float),
            np.full(shape, self.mean_s, dtype=float),
        )

    def _expand_gs_to_harmonics(self, n_harm, spatial, rreal_2d, rimag_2d):
        """Stack per-harmonic g/s planes for multi-harmonic ``phasor_calibrate``.

        PAW-FLIM calibrates on two harmonics (H and 2H) simultaneously, so the
        reference g/s must have a leading harmonic axis matching the sample's
        ``(n_harm, Y, X)`` real/imag arrays. Prefers exact per-harmonic scalars
        from ``harmonic_gs`` (set by ``set_harmonic_means``); falls back to
        broadcasting a single 2-D map to every harmonic plane, or to the
        scalar ``mean_g``/``mean_s`` when neither is available.

        Args:
            n_harm: Number of harmonic planes required by the sample array.
            spatial: Target 2-D spatial shape ``(Y, X)`` for each plane.
            rreal_2d: Fallback 2-D real (g) map used when no per-harmonic data exists.
            rimag_2d: Fallback 2-D imaginary (s) map used when no per-harmonic data exists.

        Returns:
            Tuple ``(rreal, rimag)`` each shaped ``(n_harm, *spatial)``.
        """
        if self.use_manual:
            g = float(self.manual_g)
            s = float(self.manual_s)
            return (
                np.stack([np.full(spatial, g, dtype=float) for _ in range(n_harm)]),
                np.stack([np.full(spatial, s, dtype=float) for _ in range(n_harm)]),
            )
        gs = self.harmonic_gs
        if gs and len(gs) >= n_harm:
            return (
                np.stack([np.full(spatial, g, dtype=float) for g, _ in gs[:n_harm]]),
                np.stack([np.full(spatial, s, dtype=float) for _, s in gs[:n_harm]]),
            )
        # Fallback: broadcast primary (or 2-D map) to every harmonic plane.
        if rreal_2d.shape == spatial:
            return (
                np.broadcast_to(rreal_2d, (n_harm, *spatial)).copy(),
                np.broadcast_to(rimag_2d, (n_harm, *spatial)).copy(),
            )
        return (
            np.stack([np.full(spatial, float(self.mean_g)) for _ in range(n_harm)]),
            np.stack([np.full(spatial, float(self.mean_s)) for _ in range(n_harm)]),
        )

    def clear(self):
        """Reset all calibration fields to defaults and drop stored maps.

        Used when the user removes the reference file or switches to a fresh
        session; after this call ``is_active`` is False (both ``use_manual``
        and ``values_ready`` are cleared), so samples processed afterward run
        uncalibrated until a new reference is set.
        """
        self.source_path = ""
        self.channel = 0
        self._maps = None
        self.mean_g = self.mean_s = 0.0
        self.mean_intensity = 1.0
        self.harmonic_gs = None
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
    mean/g/s maps (or per-harmonic scalars for PAW-FLIM) in a
    ``ReferenceCalibration``. The raw signal is released after transform to
    limit memory use.

    Args:
        ref_path: Path to the reference FLIM container (e.g. .lif, .ptu).
        channel: Emission channel index to use after multi-channel reduction.
        harmonic: Single harmonic index or list (e.g. ``[1, 2]`` for dual-harmonic).

    Returns:
        Populated ``ReferenceCalibration`` with ``values_ready`` True.
    """
    harm = harmonic
    if isinstance(harmonic, int):
        harm = int(harmonic)
    else:
        harm = list(harmonic)
    # The reference only ever uses one channel, so decode just that one to keep
    # peak memory and binning cost low (the reference histogram is discarded below).
    n_channels = flim_channel_count(ref_path)
    ch = int(channel)
    if n_channels:
        ch = min(ch, n_channels - 1)
    sig = load_flim_signal(ref_path, channel=ch, frame=-1, dtype=np.uint32)
    if not n_channels:
        n_channels = int(sig.sizes["C"]) if "C" in getattr(sig, "dims", ()) else 1
    rsig = reduce_signal(sig, 0)  # channel axis already collapsed by load_flim_signal
    del sig
    rmean, rreal, rimag = phasor_from_signal(rsig, axis="H", harmonic=harm)
    del rsig  # histogram released; only (mean, g, s) maps are kept
    cal = ReferenceCalibration(
        source_path=ref_path,
        channel=int(ch),
        n_channels=n_channels,
        harmonic=harm if isinstance(harm, int) else harm[0],
    )
    if isinstance(harm, list):
        # Keep per-harmonic scalars (needed for correct PAW-FLIM calibration).
        cal.set_harmonic_means(rmean, rreal, rimag)
    else:
        cal.set_maps(rmean, rreal, rimag)
    return cal
