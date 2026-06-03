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
    phasor_from_lifetime,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_from_signal

from flim_phasors.analysis import global_phasor_center
from flim_phasors.io import load_flim_signal, reference_phasor
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
        self.is_synthetic = False
        self._syn = None
        self.mean_raw = None
        self.mean_thr = None
        self.real_cal = None
        self.imag_cal = None
        self.tau_phi = None
        self.tau_mod = None
        self.tau_normal = None
        self.tau_search_phi = None
        self.tau_search_mod = None
        self.phasor_center_g = None
        self.phasor_center_s = None
        self.sample_path = ""
        self.group_name = ""
        self.ref_path = ""
        self.ref_n_channels = 1
        self.ref_channel = 0

    def load_sample(self, path):
        sig = load_flim_signal(path, channel=None, frame=-1, dtype=np.uint32)
        self.signal_full = sig
        self.is_synthetic = False
        self.sample_path = path
        self.n_channels = int(sig.sizes["C"]) if "C" in sig.dims else 1
        self.channel = 0
        self.frequency = float(sig.attrs.get("frequency", 80.0))
        red = reduce_signal(sig, 0)
        yx = [s for d, s in zip(red.dims, red.shape) if d != "H"]
        shape = tuple(yx) if len(yx) == 2 else (red.shape[0], red.shape[1])
        return shape, self.n_channels

    def load_synthetic(self, shape=(256, 256)):
        h, w = shape
        rng = np.random.default_rng(0)
        freq = 80.0
        self.frequency = freq
        self.harmonic = 1
        self.is_synthetic = True
        self.signal_full = None
        self.n_channels = 1
        self.channel = 0
        g = np.zeros(shape)
        s = np.zeros(shape)
        mean = np.zeros(shape)
        bg_g, bg_s = phasor_from_lifetime(freq, 0.4)
        g[:] = bg_g
        s[:] = bg_s
        mean[:] = rng.uniform(20, 80, shape)
        yy, xx = np.mgrid[0:h, 0:w]
        col_g, col_s = phasor_from_lifetime(freq, 0.25)
        for _ in range(6):
            cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(15, 40)
            m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            g[m] = col_g
            s[m] = col_s
            mean[m] = rng.uniform(300, 900, m.sum())
        ves_g, ves_s = phasor_from_lifetime(freq, 2.0)
        for _ in range(5):
            y0 = rng.integers(0, h)
            x = np.arange(w)
            y = (y0 + 25 * np.sin(x / 30.0 + rng.uniform(0, 6))).astype(int) % h
            for dy in range(-3, 4):
                yi = (y + dy) % h
                g[yi, x] = ves_g
                s[yi, x] = ves_s
                mean[yi, x] = rng.uniform(400, 1000, w)
        g += rng.normal(0, 0.012, shape)
        s += rng.normal(0, 0.012, shape)
        self._syn = (mean, g, s)
        self.sample_path = "<synthetic demo>"
        return shape, 1

    def _sample_channel_signal(self):
        return reduce_signal(self.signal_full, self.channel)

    def _reference_phasor(self, ref_path, harmonic):
        rmean, rreal, rimag = reference_phasor(ref_path, self.ref_channel, harmonic)
        if isinstance(harmonic, (list, tuple)):
            return rmean, rreal, rimag
        return to_2d(rmean), to_2d(rreal), to_2d(rimag)

    def apply_processing(
        self,
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

        if filter_mode == "pawflim" and not self.is_synthetic:
            harmonics = [H, 2 * H]
            sig = self._sample_channel_signal()
            photon_count = photon_count_from_signal(sig)
            mean, real, imag = phasor_from_signal(sig, axis="H", harmonic=harmonics)
            if ref_path:
                rmean, rreal, rimag = self._reference_phasor(ref_path, harmonics)
                real, imag = phasor_calibrate(
                    real, imag, rmean, rreal, rimag,
                    frequency=freq, lifetime=float(ref_lifetime), harmonic=harmonics)
                self.ref_path = ref_path
            mean, real, imag = phasor_filter_pawflim(
                mean, real, imag, sigma=float(paw_sigma),
                levels=int(paw_levels), harmonic=harmonics)
            real = to_2d(real[0])
            imag = to_2d(imag[0])
            mean = to_2d(mean)
        else:
            if self.is_synthetic:
                mean, real, imag = (a.astype(float).copy() for a in self._syn)
                mean, real, imag = to_2d(mean), to_2d(real), to_2d(imag)
                photon_count = mean.copy()
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
            if ref_path and not self.is_synthetic:
                rmean, rreal, rimag = self._reference_phasor(ref_path, H)
                real, imag = phasor_calibrate(
                    real, imag, rmean, rreal, rimag,
                    frequency=freq, lifetime=float(ref_lifetime), harmonic=H)
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

        try:
            cg, cs, _ = global_phasor_center(mean_thr, real, imag)
            self.phasor_center_g = cg
            self.phasor_center_s = cs
        except Exception:
            self.phasor_center_g = None
            self.phasor_center_s = None

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
