"""In-memory FLIM sample state: load, calibrate, filter, and lifetime maps.

``PhasorData`` holds either raw TCSPC histograms or pre-exported Leica phasor
maps. Processing applies reference calibration (``phasor_calibrate``),
spatial or signal-domain filtering (median, Gaussian, PAW-FLIM), photon
thresholding, and computes per-pixel apparent lifetimes τ_φ, τ_m, and τ_n
at ``frequency * harmonic``.
"""

from __future__ import annotations

import numpy as np
from phasorpy.filter import (
    phasor_filter_gaussian,
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
    signal_filter_gaussian,
    signal_filter_median,
)
from phasorpy.lifetime import (
    phasor_calibrate,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_from_signal

from flim_phasors.io import flim_channel_count, flim_frame_count, load_flim_signal
from flim_phasors.lif_io import load_lif_phasor_maps
from flim_phasors.utils import reduce_signal, to_2d


class PhasorData:
    """Mutable container for one FLIM sample's histogram or phasor maps.

    Supports two load paths: decode TCSPC and transform with
    ``phasor_from_signal``, or import LAS X phasor triplets via
    ``load_lif_phasor``. After ``apply_processing``, exposes calibrated g/s
    maps, thresholded intensity, and lifetime images for GUI and export.
    """

    def __init__(self):
        """Initialize an empty dataset with default acquisition parameters.

        Takes no arguments; all state (histogram, phasor maps, calibration
        flags, and per-sample segmentation) is populated later via
        ``load_sample``/``load_lif_phasor`` and ``apply_processing``. Delegates
        to :meth:`reset` so the initial state matches a manually cleared one.
        """
        self.reset()

    def reset(self):
        """Clear all loaded data, maps, paths, and processing flags.

        Restores every attribute to its default value (no histogram, no
        phasor/lifetime maps, no calibration or LIF metadata, empty
        segmentation state). Used both by ``__init__`` and when the GUI
        discards a sample to load a fresh one in its place.
        """
        self.signal_full = None
        self.n_channels = 1
        self.channel = 0
        self.fast_loaded_channel = None  # channel index when only one was decoded
        self.n_frames = 1
        self.frequency = 80.0
        self.harmonic = 1
        self.mean_raw = None
        self.mean_thr = None
        self.real_cal = None
        self.imag_cal = None
        self.tau_phi = None
        self.tau_mod = None
        self.tau_normal = None
        self.frame_index = -1
        self.sample_path = ""
        self.display_name = ""
        self.group_name = ""
        self.ref_path = ""
        self.ref_n_channels = 1
        self.ref_channel = 0
        self.pixel_size_um = 0.0
        self._shape_hint = None
        self.processing_settings = None  # per-sample filter/harmonic stash (dict)
        self.maps_calibrated = False  # True after Apply with reference calibration
        self.load_source = ""  # "" | "lif_phasor"
        self.lif_image_key = ""
        self.lif_lasx_calibrated = False
        self.lif_lasx_intensity_threshold = 0.0
        self.lif_uses_photon_intensity = False
        self._lif_base_mean = None
        self._lif_base_real = None
        self._lif_base_imag = None
        # Per-sample segmentation (remembered when switching multi-image samples)
        self.gmm_fit = None
        self.cluster_stats = []
        self.last_overlay = None

    def ensure_loaded(self, frame=None):
        """Load sample data lazily if not already in memory.

        For histogram samples, decodes the file on first access. For LIF phasor
        imports, loads maps when the base arrays are missing.

        Args:
            frame: Optional frame index override for multi-frame files.

        Returns:
            Tuple ``(spatial_shape, n_channels)`` where ``spatial_shape`` is
            ``(width, height)`` or ``(height, width)`` per ``_shape_hint``.

        Raises:
            ValueError: If no ``sample_path`` is set and nothing is loaded.
        """
        if self.signal_full is not None:
            return self._shape_hint or (256, 256), self.n_channels
        if self.load_source == "lif_phasor" and self._lif_base_real is not None:
            return self._shape_hint or (256, 256), self.n_channels
        if not self.sample_path:
            raise ValueError("No sample path to load.")
        if self.load_source == "lif_phasor":
            return self.load_lif_phasor(self.sample_path, self.lif_image_key or None)
        return self.load_sample(
            self.sample_path, frame=frame, load_channel=self.fast_loaded_channel)

    @property
    def has_loaded_maps(self) -> bool:
        """Return whether calibrated or base phasor maps are present.

        Covers two distinct sources: a regular sample that has been through
        Apply/calibration (``real_cal`` set), or a LIF phasor import whose
        maps were decoded directly from LAS X exports and never needed a raw
        histogram (``_lif_base_real`` set). Used to decide whether
        map-dependent UI (phasor plot, export, cluster stats) can be enabled
        for this dataset.

        Returns:
            True if ``real_cal`` or LIF base real component is set.
        """
        return self.real_cal is not None or self._lif_base_real is not None

    def _pixel_size_from_lif_attrs(self, attrs):
        """Set ``pixel_size_um`` from LIF coordinate steps or raw metadata.

        LIF/XLEF coordinate arrays are stored in micrometers, so the pixel
        size is simply the spacing between consecutive X coordinate samples
        (falling back to Y if X is unavailable, since pixels are assumed
        square). If neither axis yields a usable step, this falls back to
        the ``VoxelSizeX``/``VoxelSizeY`` fields embedded in the raw
        ``flim_rawdata`` block. Mutates ``self.pixel_size_um`` in place and
        leaves it untouched (default) if no source yields a positive value.

        Args:
            attrs: Attribute dict from ``load_lif_phasor_maps`` (coords or
                ``flim_rawdata`` VoxelSize fields).
        """
        coords = attrs.get("coords") or {}
        # LIF coords are in µm; X step alone is enough when pixels are square.
        for ax in ("X", "Y"):
            c = coords.get(ax)
            if c is not None and len(c) > 1:
                try:
                    step = float(c[1] - c[0])
                    if step > 0:
                        self.pixel_size_um = step
                        return  # stop after first positive axis step
                except (TypeError, ValueError):
                    pass
        raw = attrs.get("flim_rawdata") or {}
        for key in ("VoxelSizeX", "VoxelSizeY"):
            if key in raw:
                try:
                    self.pixel_size_um = float(raw[key])
                    return
                except (TypeError, ValueError):
                    pass

    def load_lif_phasor(self, path, image_key=None, *, channel=0):
        """Load pre-computed phasor maps from a Leica LIF/XLEF file.

        Stores base mean/g/s arrays, applies LAS X intensity threshold on first
        finalize, and records modulation frequency and calibration flags from
        file metadata.

        Args:
            path: Leica container path.
            image_key: Internal series key from ``list_lif_phasor_series``.
            channel: Emission channel index for multi-channel phasor exports.

        Returns:
            Tuple ``(spatial_shape, n_channels)``.
        """
        mean, real, imag, attrs = load_lif_phasor_maps(path, image_key, channel=channel)
        self.signal_full = None
        self.sample_path = path
        self.load_source = "lif_phasor"
        self.lif_image_key = str(image_key or "")
        self.frame_index = -1
        self.n_channels = max(1, int(attrs.get("n_phasor_channels", 1)))
        self.channel = min(max(0, int(channel)), self.n_channels - 1)
        self.frequency = float(attrs.get("frequency", 80.0))
        cal = attrs.get("lasx_calibration") or {}
        self.lif_lasx_calibrated = bool(cal.get("applied", False))
        self.lif_lasx_intensity_threshold = float(attrs.get("lasx_intensity_threshold", 0.0))
        self.lif_uses_photon_intensity = bool(attrs.get("uses_photon_intensity", False))
        # Store (width, height) = (X, Y) for GUI layout; arrays remain (Y, X).
        self._shape_hint = (int(mean.shape[1]), int(mean.shape[0]))
        self._pixel_size_from_lif_attrs(attrs)
        # Base maps keep LAS X calibration/threshold state; Apply re-runs from here.
        self._lif_base_mean = np.asarray(mean, dtype=float)
        self._lif_base_real = np.asarray(real, dtype=float)
        self._lif_base_imag = np.asarray(imag, dtype=float)
        self._finalize_phasor_maps(
            self._lif_base_mean,
            self._lif_base_real,
            self._lif_base_imag,
            intensity_min=self.lif_lasx_intensity_threshold,
            detect_harmonics=True,
        )
        return self._shape_hint, self.n_channels

    # --- unused (focused cleanup): uncomment if needed ---
    # def register_lazy(self, path: str, *, frame: int = -1, n_channels: int = 1, frequency: float = 80.0):
    #     """Register path without decoding histogram (multi-image lazy mode)."""
    #     self.sample_path = path
    #     self.frame_index = int(frame)
    #     self.n_channels = max(1, int(n_channels))
    #     self.frequency = float(frequency)
    def load_sample(self, path, frame=None, load_channel=None):
        """Decode a FLIM file into an in-memory TCSPC histogram (no phasor yet).

        Sets ``signal_full``, channel count, modulation frequency, and spatial
        shape hint from the reduced first channel. Phasor maps are produced by
        ``apply_processing``.

        Args:
            path: Path to the sample FLIM file.
            frame: Time frame index (-1 = last/single frame).
            load_channel: When ``None`` (default) all channels are decoded and
                kept so channel switching is instant. When an int, only that
                channel is decoded ("fast load"), lowering memory and decode
                cost; switching channels then requires a re-decode.

        Returns:
            Tuple ``(spatial_shape, n_channels)``.
        """
        if frame is None:
            frame = int(getattr(self, "frame_index", -1))
        else:
            self.frame_index = int(frame)
        # Probe frame count before decode (T is often summed away during load).
        n_frames = flim_frame_count(path)
        if load_channel is None:
            sig = load_flim_signal(path, channel=None, frame=self.frame_index, dtype=np.uint32)
            self.signal_full = sig
            self.n_channels = int(sig.sizes["C"]) if "C" in sig.dims else 1
            self.channel = 0
            self.fast_loaded_channel = None
        else:
            true_nch = flim_channel_count(path)
            lc = max(0, int(load_channel))
            if true_nch:
                lc = min(lc, true_nch - 1)
            sig = load_flim_signal(path, channel=lc, frame=self.frame_index, dtype=np.uint32)
            if "C" in getattr(sig, "dims", ()):
                # TIFF path may retain a length-1 C axis after slicing.
                true_nch = true_nch or int(sig.sizes["C"])
                if sig.sizes["C"] > 1:
                    sig = sig.isel(C=min(lc, sig.sizes["C"] - 1))
            self.signal_full = sig
            self.n_channels = max(
                1,
                int(true_nch or 1),
                int((getattr(sig, "attrs", {}) or {}).get("n_channels", 0) or 0),
                lc + 1,
            )
            self.channel = lc
            self.fast_loaded_channel = lc
        if n_frames is None and "T" in getattr(sig, "dims", ()):
            n_frames = int(sig.sizes.get("T", 1))
        if n_frames is None:
            n_frames = (getattr(sig, "attrs", {}) or {}).get("n_frames")
        self.n_frames = max(1, int(n_frames or 1))
        self.sample_path = path
        self.frequency = float(sig.attrs.get("frequency", 80.0))
        red = reduce_signal(sig, self.channel)
        # Spatial shape excludes time axis H; order follows xarray dim order (often Y, X).
        yx = [s for d, s in zip(red.dims, red.shape) if d != "H"]
        shape = tuple(yx) if len(yx) == 2 else (red.shape[0], red.shape[1])
        self._shape_hint = shape
        attrs = getattr(sig, "attrs", {}) or {}
        ps = attrs.get("pixel_size") or attrs.get("PixelSize")
        if ps is not None:
            try:
                self.pixel_size_um = float(ps)
            except (TypeError, ValueError):
                pass
        return shape, self.n_channels

    def _sample_channel_signal(self):
        """Return the TCSPC stack reduced to the active emission channel.

        Thin wrapper around :func:`~flim_phasors.utils.reduce_signal` bound
        to this dataset's current ``channel`` selection, used as the common
        entry point before phasor computation so channel/frame reduction
        logic lives in one place rather than being duplicated at each
        call site.

        Returns:
            Reduced signal array with time axis ``H`` and spatial dimensions.
        """
        return reduce_signal(self.signal_full, self.channel)

    def _apply_reference_calibration(self, real, imag, mean, ref_cal, *, frequency, lifetime, harmonic):
        """Calibrate sample g/s maps against reference maps or scalar g/s.

        Wraps ``phasor_calibrate`` with reference mean/g/s fields sized to the
        sample map shape from ``ref_cal.maps_for_shape``.

        Args:
            real: Sample real (g) phasor map before calibration.
            imag: Sample imaginary (s) phasor map.
            mean: Sample mean intensity (photon counts).
            ref_cal: Active ``ReferenceCalibration`` instance.
            frequency: Laser modulation frequency in MHz.
            lifetime: Known reference fluorophore lifetime in ns for calibration.
            harmonic: Harmonic index or list for multi-harmonic PAW-FLIM.

        Returns:
            Tuple ``(real_cal, imag_cal)`` of calibrated phasor components.
        """
        shape = real.shape
        rmean, rreal, rimag = ref_cal.maps_for_shape(shape)
        # phasor_calibrate rotates/scales sample (real, imag) ≡ (g, s) using ref maps
        # and the known reference fluorophore lifetime (ns) at frequency (MHz).
        return phasor_calibrate(
            real, imag, rmean, rreal, rimag,
            frequency=frequency, lifetime=float(lifetime), harmonic=harmonic)

    def _finalize_phasor_maps(
        self,
        photon_count,
        real,
        imag,
        *,
        intensity_min=0.0,
        detect_harmonics=True,
    ):
        """Apply photon threshold, store maps, and compute lifetime images.

        NaN-invalid pixels are masked in ``mean_thr``. Apparent lifetimes use
        ``work_frequency`` = ``frequency * harmonic``.

        Args:
            photon_count: Mean photon count or intensity per pixel.
            real: Real (g) phasor component (calibrated if applicable).
            imag: Imaginary (s) phasor component.
            intensity_min: Minimum photons for finite phasor values.
            detect_harmonics: Passed to ``phasor_threshold`` for harmonic stacks.
        """
        real = to_2d(real)
        imag = to_2d(imag)
        photon_count = to_2d(photon_count)
        self.mean_raw = np.asarray(photon_count, dtype=float)

        thr = float(intensity_min)
        if thr > 0:
            mean_thr, real, imag = phasor_threshold(
                photon_count,
                real,
                imag,
                mean_min=thr,
                detect_harmonics=bool(detect_harmonics),
            )
        else:
            # thr==0: skip phasor_threshold but still NaN-mask invalid g/s pixels.
            mean_thr = self.mean_raw.copy()
            bad = ~np.isfinite(real) | ~np.isfinite(imag)
            if np.any(bad):
                mean_thr = mean_thr.copy()
                mean_thr[bad] = np.nan
                real = np.where(bad, np.nan, real)
                imag = np.where(bad, np.nan, imag)

        self.mean_thr = to_2d(mean_thr)
        self.real_cal = to_2d(real)
        self.imag_cal = to_2d(imag)
        self._intensity_stats = self._compute_intensity_stats(thr)

        # Lifetimes use ω at the analysis harmonic, not the laser fundamental alone.
        work_freq = self.frequency * self.harmonic
        with np.errstate(invalid="ignore", divide="ignore"):
            tau_phi, tau_mod = phasor_to_apparent_lifetime(real, imag, work_freq)
            tau_normal = phasor_to_normal_lifetime(real, imag, work_freq)
        self.tau_phi = np.asarray(tau_phi, dtype=float)
        self.tau_mod = np.asarray(tau_mod, dtype=float)
        self.tau_normal = np.asarray(tau_normal, dtype=float)

    def _apply_processing_from_lif_maps(
        self,
        ref_calibration=None,
        ref_path=None,
        ref_lifetime=4.0,
        filter_mode="median",
        median_size=3,
        median_repeat=1,
        intensity_min=0.0,
        detect_harmonics=True,
    ):
        """Reprocess loaded LIF phasor base maps (calibrate, filter, threshold).

        Signal-domain filters are not available without histograms; ``pawflim``
        and ``signal *`` modes fall back to phasor-domain median filtering.

        Args:
            ref_calibration: Optional ``ReferenceCalibration`` for g/s correction.
            ref_path: Path string stored on the dataset when calibration runs.
            ref_lifetime: Reference fluorophore lifetime in ns.
            filter_mode: ``"median"``, ``"gaussian"``, or aliases mapped to median.
            median_size: Kernel size for spatial phasor filters.
            median_repeat: Number of filter passes.
            intensity_min: Photon threshold applied in ``_finalize_phasor_maps``.
            detect_harmonics: Harmonic-aware thresholding flag.

        Raises:
            ValueError: If LIF base phasor arrays were never loaded.
        """
        if self._lif_base_real is None:
            raise ValueError("No LIF phasor maps loaded.")
        H = int(self.harmonic)
        freq = float(self.frequency)
        mean = np.asarray(self._lif_base_mean, dtype=float)
        real = np.asarray(self._lif_base_real, dtype=float)
        imag = np.asarray(self._lif_base_imag, dtype=float)

        # LIF exports have no TCSPC stack — histogram/signal/PAW filters unavailable.
        if filter_mode in ("pawflim", "signal median", "signal gaussian"):
            filter_mode = "median"

        if ref_calibration is not None and ref_calibration.is_active:
            real, imag = self._apply_reference_calibration(
                real, imag, mean, ref_calibration,
                frequency=freq, lifetime=ref_lifetime, harmonic=H)
            if ref_path:
                self.ref_path = ref_path

        if filter_mode == "median" and median_size >= 1 and median_repeat >= 1:
            mean, real, imag = phasor_filter_median(
                mean, real, imag, size=int(median_size), repeat=int(median_repeat))
        elif filter_mode == "gaussian" and median_size >= 1 and median_repeat >= 1:
            mean, real, imag = phasor_filter_gaussian(
                mean, real, imag, size=int(median_size), repeat=int(median_repeat))

        self._finalize_phasor_maps(
            mean, real, imag,
            intensity_min=float(intensity_min),
            detect_harmonics=bool(detect_harmonics),
        )

    def apply_processing(
        self,
        ref_calibration=None,
        ref_path=None,
        ref_lifetime=4.0,
        filter_mode="median",
        median_size=3,
        median_repeat=1,
        paw_sigma=2.0,
        paw_levels=1,
        intensity_min=0.0,
        detect_harmonics=True,
    ):
        """Build calibrated phasor and lifetime maps from histogram or LIF data.

        Histogram path: optional signal filtering, ``phasor_from_signal``,
        reference calibration, phasor-domain filtering, or PAW-FLIM dual-harmonic
        workflow. LIF path delegates to ``_apply_processing_from_lif_maps``.

        Args:
            ref_calibration: Optional instrument reference for ``phasor_calibrate``.
            ref_path: Reference file path recorded on the dataset.
            ref_lifetime: Known reference lifetime in ns (default 4.0).
            filter_mode: ``"median"``, ``"gaussian"``, ``"pawflim"``,
                ``"signal median"``, or ``"signal gaussian"``.
            median_size: Spatial kernel size (phasor or signal filters).
            median_repeat: Repeat count for median/Gaussian filters.
            paw_sigma: Gaussian sigma for PAW-FLIM phasor filtering.
            paw_levels: Wavelet levels for PAW-FLIM.
            intensity_min: Minimum photons per pixel after processing.
            detect_harmonics: Use harmonic-aware thresholding when applicable.
        """
        if self.load_source == "lif_phasor" and self._lif_base_real is not None:
            return self._apply_processing_from_lif_maps(
                ref_calibration=ref_calibration,
                ref_path=ref_path,
                ref_lifetime=ref_lifetime,
                filter_mode=filter_mode,
                median_size=median_size,
                median_repeat=median_repeat,
                intensity_min=intensity_min,
                detect_harmonics=detect_harmonics,
            )

        H = int(self.harmonic)
        freq = float(self.frequency)

        if filter_mode == "pawflim":
            harmonics = [H, 2 * H]
            sig = self._sample_channel_signal()
            mean, real, imag = phasor_from_signal(sig, axis="H", harmonic=harmonics)
            # Derive photon counts from phasor mean (avoids a second full H-sum).
            photon_count = self._photon_count_from_mean(mean, sig)
            if ref_calibration is not None and ref_calibration.is_active:
                real, imag = self._apply_reference_calibration(
                    real, imag, mean, ref_calibration,
                    frequency=freq, lifetime=ref_lifetime, harmonic=harmonics)
                if ref_path:
                    self.ref_path = ref_path
            mean, real, imag = phasor_filter_pawflim(
                mean, real, imag, sigma=float(paw_sigma),
                levels=int(paw_levels), harmonic=harmonics)
            # PAW-FLIM filters both harmonics; GUI maps use the primary harmonic only.
            real = to_2d(real[0])
            imag = to_2d(imag[0])
            mean = to_2d(mean)
        else:
            sig = self._sample_channel_signal()
            if filter_mode == "signal median" and median_size >= 1:
                sig = signal_filter_median(
                    sig, size=int(median_size), repeat=int(median_repeat))
            elif filter_mode == "signal gaussian" and median_size >= 1:
                sig = signal_filter_gaussian(
                    sig, size=int(median_size), repeat=int(median_repeat))
            mean, real, imag = phasor_from_signal(sig, axis="H", harmonic=H)
            mean, real, imag = to_2d(mean), to_2d(real), to_2d(imag)
            photon_count = self._photon_count_from_mean(mean, sig)
            if ref_calibration is not None and ref_calibration.is_active:
                real, imag = self._apply_reference_calibration(
                    real, imag, mean, ref_calibration,
                    frequency=freq, lifetime=ref_lifetime, harmonic=H)
                if ref_path:
                    self.ref_path = ref_path
            if filter_mode == "median" and median_size >= 1 and median_repeat >= 1:
                mean, real, imag = phasor_filter_median(
                    mean, real, imag, size=int(median_size), repeat=int(median_repeat))
            elif filter_mode == "gaussian" and median_size >= 1 and median_repeat >= 1:
                mean, real, imag = phasor_filter_gaussian(
                    mean, real, imag, size=int(median_size), repeat=int(median_repeat))

        self._finalize_phasor_maps(
            photon_count, real, imag,
            intensity_min=float(intensity_min),
            detect_harmonics=bool(detect_harmonics),
        )

    @staticmethod
    def _photon_count_from_mean(mean, sig):
        """Convert phasorpy mean intensity to total photons per pixel (mean × n_bins).

        ``phasor_from_signal`` returns mean = (sum over H) / n_bins; thresholding
        and export expect cumulative photon counts, hence the scale factor.
        """
        mean2 = to_2d(np.asarray(mean, dtype=float))
        if hasattr(sig, "sizes") and "H" in getattr(sig, "dims", ()):
            n_h = int(sig.sizes["H"])
        else:
            arr = np.asarray(sig)
            n_h = int(arr.shape[-1]) if arr.ndim >= 1 else 1
        return mean2 * float(max(n_h, 1))

    def intensity_brightfield(self):
        """Full intensity image including pixels below the Min N threshold.

        Prefers the original LIF / photon intensity map when available; otherwise
        uses ``mean_raw`` (photon counts before threshold masking). Does not
        apply the phasor valid-pixel mask.

        Returns:
            2-D float array, or ``None`` if no intensity has been computed.
        """
        if self._lif_base_mean is not None:
            return np.asarray(self._lif_base_mean, dtype=float)
        if self.mean_raw is not None:
            return np.asarray(self.mean_raw, dtype=float)
        return None

    @property
    def work_frequency(self):
        """Effective modulation frequency for lifetime conversion (MHz).

        FLIM lifetime formulas assume the phasor was computed at the
        harmonic frequency actually used, not the fundamental laser
        repetition rate; multiplying by ``harmonic`` (1 for the fundamental,
        2+ for higher harmonics) gives the frequency to pass into
        :func:`~flim_phasors.analysis.lifetimes_at_phasor` and related
        conversions so τφ/τm/τn come out in nanoseconds correctly.

        Returns:
            ``frequency * harmonic`` — harmonic-scaled frequency used for τ maps.
        """
        return self.frequency * self.harmonic

    def valid_mask(self):
        """Return pixels with finite calibrated phasor coordinates.

        Reflects threshold/NaN masking in ``mean_thr`` and g/s maps, not raw
        photon counts alone.

        Returns:
            Boolean array matching ``real_cal`` shape.
        """
        return np.isfinite(self.real_cal) & np.isfinite(self.imag_cal)

    def _compute_intensity_stats(self, threshold):
        """Summarize raw photon counts and masking fraction for the status bar.

        Computed from ``mean_raw`` (total photon counts per pixel, before
        thresholding) rather than the calibrated/masked maps, so the summary
        reflects the intensity distribution the user is choosing a threshold
        against. ``masked_pct`` is the fraction of finite pixels that fall
        below ``threshold`` and would therefore be excluded (NaN) from the
        calibrated g/s maps once Apply runs.

        Args:
            threshold: Intensity cutoff used during finalize (0 = no cutoff).

        Returns:
            Dict with threshold, min/median/max counts, pixel counts, and
            ``masked_pct`` below threshold; empty dict if no raw mean loaded.
        """
        if self.mean_raw is None:
            return {}
        raw = np.asarray(self.mean_raw, dtype=float)
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            return {
                "threshold": threshold,
                "min": 0.0,
                "median": 0.0,
                "max": 0.0,
                "n_pixels": 0,
                "n_below": 0,
                "masked_pct": 100.0,
            }
        n_pixels = int(finite.size)
        n_below = int(np.sum(finite < threshold)) if threshold > 0 else 0
        masked_pct = 100.0 * n_below / n_pixels if n_pixels else 0.0
        return {
            "threshold": threshold,
            "min": float(np.min(finite)),
            "median": float(np.median(finite)),
            "max": float(np.max(finite)),
            "n_pixels": n_pixels,
            "n_below": n_below,
            "masked_pct": masked_pct,
        }
