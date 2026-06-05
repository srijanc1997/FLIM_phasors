"""FLIM sample container: load, calibrate, filter, lifetime maps."""

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
from phasorpy.lifetime import phasor_to_lifetime_search
from phasorpy.lifetime import (
    phasor_calibrate,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_from_signal

from flim_phasors.io import load_flim_signal
from flim_phasors.utils import photon_count_from_signal, reduce_signal, to_2d


class PhasorData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.signal_full = None
        self.n_channels = 1
        self.channel = 0
        self.frequency = 80.0
        self.harmonic = 1
        self.mean_raw = None
        self.mean_thr = None
        self.real_cal = None
        self.imag_cal = None
        self.tau_phi = None
        self.tau_mod = None
        self.tau_normal = None
        self.tau_search_phi = None
        self.tau_search_mod = None
        self.frame_index = -1
        self.sample_path = ""
        self.group_name = ""
        self.ref_path = ""
        self.ref_n_channels = 1
        self.ref_channel = 0
        self.pixel_size_um = 0.0
        self._shape_hint = None
        self.processing_settings = None  # per-sample filter/harmonic stash (dict)

    # --- unused (focused cleanup): uncomment if needed ---
    # def unload_histogram(self):
    #     """Drop TCSPC histogram from RAM; keep computed phasor maps."""
    #     self.signal_full = None

    def ensure_loaded(self, frame=None):
        if self.signal_full is not None:
            return self._shape_hint or (256, 256), self.n_channels
        if not self.sample_path:
            raise ValueError("No sample path to load.")
        return self.load_sample(self.sample_path, frame=frame)

    # --- unused (focused cleanup): uncomment if needed ---
    # def register_lazy(self, path: str, *, frame: int = -1, n_channels: int = 1, frequency: float = 80.0):
    #     """Register path without decoding histogram (multi-image lazy mode)."""
    #     self.sample_path = path
    #     self.frame_index = int(frame)
    #     self.n_channels = max(1, int(n_channels))
    #     self.frequency = float(frequency)
    def load_sample(self, path, frame=None):
        if frame is None:
            frame = int(getattr(self, "frame_index", -1))
        else:
            self.frame_index = int(frame)
        sig = load_flim_signal(path, channel=None, frame=self.frame_index, dtype=np.uint32)
        self.signal_full = sig
        self.sample_path = path
        self.n_channels = int(sig.sizes["C"]) if "C" in sig.dims else 1
        self.channel = 0
        self.frequency = float(sig.attrs.get("frequency", 80.0))
        red = reduce_signal(sig, 0)
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
        return reduce_signal(self.signal_full, self.channel)

    def _apply_reference_calibration(self, real, imag, mean, ref_cal, *, frequency, lifetime, harmonic):
        shape = real.shape
        rmean, rreal, rimag = ref_cal.maps_for_shape(shape)
        if isinstance(harmonic, (list, tuple)):
            return phasor_calibrate(
                real, imag, rmean, rreal, rimag,
                frequency=frequency, lifetime=float(lifetime), harmonic=harmonic)
        return phasor_calibrate(
            real, imag, rmean, rreal, rimag,
            frequency=frequency, lifetime=float(lifetime), harmonic=harmonic)

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
        H = int(self.harmonic)
        freq = float(self.frequency)

        if filter_mode == "pawflim":
            harmonics = [H, 2 * H]
            sig = self._sample_channel_signal()
            photon_count = photon_count_from_signal(sig)
            mean, real, imag = phasor_from_signal(sig, axis="H", harmonic=harmonics)
            if ref_calibration is not None and ref_calibration.is_active:
                real, imag = self._apply_reference_calibration(
                    real, imag, mean, ref_calibration,
                    frequency=freq, lifetime=ref_lifetime, harmonic=harmonics)
                if ref_path:
                    self.ref_path = ref_path
            mean, real, imag = phasor_filter_pawflim(
                mean, real, imag, sigma=float(paw_sigma),
                levels=int(paw_levels), harmonic=harmonics)
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
            photon_count = photon_count_from_signal(sig)
            mean, real, imag = phasor_from_signal(sig, axis="H", harmonic=H)
            mean, real, imag = to_2d(mean), to_2d(real), to_2d(imag)
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

        work_freq = freq * H
        with np.errstate(invalid="ignore", divide="ignore"):
            tau_phi, tau_mod = phasor_to_apparent_lifetime(real, imag, work_freq)
            tau_normal = phasor_to_normal_lifetime(real, imag, work_freq)
        self.tau_phi = np.asarray(tau_phi, dtype=float)
        self.tau_mod = np.asarray(tau_mod, dtype=float)
        self.tau_normal = np.asarray(tau_normal, dtype=float)

        with np.errstate(invalid="ignore"):
            ts_phi, ts_mod = phasor_to_lifetime_search(real, imag, work_freq)
        self.tau_search_phi = np.asarray(ts_phi, dtype=float)
        self.tau_search_mod = np.asarray(ts_mod, dtype=float)

    @property
    def work_frequency(self):
        return self.frequency * self.harmonic

    def valid_mask(self):
        return np.isfinite(self.real_cal) & np.isfinite(self.imag_cal)

    def _compute_intensity_stats(self, threshold):
        if self.mean_raw is None:
            return {}
        raw = np.asarray(self.mean_raw, dtype=float)
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            return {"threshold": threshold, "masked_pct": 100.0}
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
