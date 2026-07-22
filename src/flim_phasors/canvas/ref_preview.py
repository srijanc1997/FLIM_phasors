"""Small reference phasor preview plot.

Shows the universal semicircle and the calibrated reference phasor position
(g, s) used for lifetime calibration.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from phasorpy.lifetime import phasor_from_lifetime, phasor_semicircle


class RefPreviewCanvas(FigureCanvas):
    """Compact phasor semicircle preview for reference calibration status.

    A small (default ~2.4x2.1 inch) matplotlib widget intended to sit next to
    calibration controls and give the user quick visual feedback on whether a
    reference dataset has been loaded and calibrated correctly. Two states are
    rendered: an "empty" placeholder (:meth:`_draw_empty`) shown before a
    reference is available, and the calibration view
    (:meth:`show_calibration`) once a
    :class:`~flim_phasors.calibration.ReferenceCalibration` object is active.
    In the calibration view, the measured (raw) reference phasor is plotted
    as a red "+", the theoretical single-exponential phasor for the known
    reference dye's lifetime is plotted as a green "o" on the universal
    semicircle, and a dashed arrow connects them to visualize the correction
    that ``phasor_calibrate`` applies to every pixel. Unlike the main
    :class:`~flim_phasors.canvas.phasor.PhasorCanvas`, this widget is
    non-interactive (no mouse handlers) and always performs a full
    ``ax.clear()`` redraw since it is small and infrequently updated.
    """

    def __init__(self, parent=None):
        """Create the preview widget with a placeholder empty state.

        Builds the small figure/axes and immediately calls
        :meth:`_draw_empty` so the widget shows a "Load Reference... then
        Calibrate" prompt before any reference dataset is available,
        rather than an ambiguous blank plot.

        Args:
            parent: Optional Qt parent widget.
        """
        self.fig = Figure(figsize=(2.4, 2.1), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(120)
        self.ax = self.fig.add_subplot(111)
        self._draw_empty()

    def _draw_empty(self):
        """Render the semicircle and a prompt to load and calibrate reference.

        This is the initial state shown on widget construction and whenever
        :meth:`show_calibration` is called with an inactive or ``None``
        calibration object (see the ``cal is not None and cal.is_active``
        guard there — when that condition is false, ``show_calibration``
        still draws the semicircle and styling but skips the reference/target
        markers, leaving a similar "nothing calibrated yet" look). The prompt
        text is anchored in axes-fraction coordinates so it stays centered
        regardless of the dynamic x/y limits set by :meth:`_style_axes`.
        """
        self.ax.clear()
        g, s = phasor_semicircle(101)
        self.ax.plot(g, s, "k-", lw=0.8, alpha=0.5)
        self.ax.text(
            0.5, 0.38, "Load Reference…\nthen Calibrate",
            ha="center", va="center", fontsize=7, color="gray",
            transform=self.ax.transAxes,
        )
        self._style_axes()
        self.fig.tight_layout()
        self.draw()

    def _effective_gs(self, cal) -> tuple[float, float]:
        """Return the (g, s) pair used for display from a calibration object.

        Mirrors the same manual-vs-computed precedence used by the actual
        calibration math: when the user has opted to type in a manual
        reference phasor (``use_manual``), that value is shown instead of
        the mean measured from the reference image, so the preview always
        reflects what calibration will actually use.

        Args:
            cal: :class:`~flim_phasors.calibration.ReferenceCalibration` instance.

        Returns:
            Tuple ``(g, s)`` from manual override or computed means.
        """
        if cal.use_manual:
            return float(cal.manual_g), float(cal.manual_s)
        return float(cal.mean_g), float(cal.mean_s)

    def _target_gs(self, ref_lifetime_ns: float, frequency_mhz: float, harmonic: int):
        """Return the theoretical single-exponential phasor for the reference dye.

        This is where the measured reference is moved to by ``phasor_calibrate``.

        Args:
            ref_lifetime_ns: Known reference lifetime in ns.
            frequency_mhz: Laser modulation frequency in MHz.
            harmonic: Harmonic index the reference g/s were computed at.

        Returns:
            Tuple ``(g, s)`` of the target phasor, or ``(nan, nan)`` if invalid.
        """
        try:
            if ref_lifetime_ns <= 0 or frequency_mhz <= 0:
                return float("nan"), float("nan")
            # phasor_from_lifetime expects ω at the harmonic used for the reference maps.
            tg, ts = phasor_from_lifetime(
                float(frequency_mhz) * max(1, int(harmonic)), float(ref_lifetime_ns))
            return float(tg), float(ts)
        except Exception:
            return float("nan"), float("nan")

    def show_calibration(
        self, cal, *, ref_lifetime_ns: float = 4.0, frequency_mhz: float = 80.0,
        harmonic: int = 1,
    ):
        """Draw the measured reference phasor and its calibration target.

        The red ``+`` is the raw (uncalibrated) reference position; the green ``o``
        is where a single-exponential dye of lifetime ``ref_lifetime_ns`` sits on
        the semicircle — i.e. where calibration moves the reference to. A dashed
        arrow between them visualises the correction.

        Args:
            cal: Calibration object, or ``None`` when inactive.
            ref_lifetime_ns: Known reference lifetime in ns (target marker).
            frequency_mhz: Laser modulation frequency in MHz (target marker).
            harmonic: Harmonic index the reference g/s were computed at.
        """
        self.ax.clear()
        g, s = phasor_semicircle(101)
        self.ax.plot(g, s, "k-", lw=0.8, alpha=0.5)
        rg = rs = 0.0
        if cal is not None and cal.is_active:
            rg, rs = self._effective_gs(cal)
            if cal._maps is not None and not cal.use_manual:
                _, rreal, rimag = cal._maps
                rreal = np.asarray(rreal, dtype=float)
                rimag = np.asarray(rimag, dtype=float)
                finite = np.isfinite(rreal) & np.isfinite(rimag)
                if np.any(finite):
                    gr = rreal[finite].ravel()
                    sr = rimag[finite].ravel()
                    step = max(1, gr.size // 1200)
                    # Subsample scatter — full ref cloud can be 100k+ pixels in a tiny preview.
                    self.ax.scatter(
                        gr[::step], sr[::step],
                        s=2, alpha=0.3, c="steelblue", rasterized=True, zorder=2)
            tg, ts = self._target_gs(ref_lifetime_ns, frequency_mhz, harmonic)
            measured_ok = np.isfinite(rg) and np.isfinite(rs)
            target_ok = np.isfinite(tg) and np.isfinite(ts)
            if measured_ok and target_ok:
                self.ax.annotate(
                    "", xy=(tg, ts), xytext=(rg, rs),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8,
                                    linestyle="--", alpha=0.8), zorder=4)
            if target_ok:
                self.ax.plot(tg, ts, "o", ms=7, mfc="none", mec="green",
                             mew=1.5, zorder=5,
                             label=f"target τ={ref_lifetime_ns:.2f} ns")
            if measured_ok:
                self.ax.plot(rg, rs, "r+", ms=12, mew=2, zorder=6)
                self.ax.annotate(
                    f"g={rg:.3f}\ns={rs:.3f}",
                    (rg, rs),
                    fontsize=7,
                    xytext=(5, 5),
                    textcoords="offset points",
                    zorder=7,
                )
            if target_ok:
                self.ax.legend(loc="upper right", fontsize=6, framealpha=0.6)
        self._style_axes(rg, rs)
        self.fig.tight_layout()
        self.draw()

    def _style_axes(self, rg: float = 0.0, rs: float = 0.0):
        """Apply consistent limits, aspect, and title to the preview axes.

        The universal semicircle spans roughly ``g in [0, 1]``, ``s in [0,
        0.5]``, so default limits (``1.05`` / ``0.75``) give a small margin
        around it for the empty-state view. Long-lifetime reference dyes
        (small g at high frequency, or vice versa) can measure well outside
        the semicircle bounds due to noise, so when a reference position is
        supplied and lies beyond the default limits, the corresponding axis
        is expanded just enough to keep that point comfortably in view
        (``+0.08`` on g, ``+0.1`` on s) rather than clipping it silently.

        Args:
            rg: Measured reference g coordinate; expands the x-limit beyond
                the default when positive and larger than it.
            rs: Measured reference s coordinate; expands the y-limit beyond
                the default when positive and larger than it.
        """
        xmax = max(1.05, float(rg) + 0.08) if rg > 0 else 1.05
        ymax = max(0.55, float(rs) + 0.1) if rs > 0 else 0.75
        self.ax.set_xlim(0, xmax)
        self.ax.set_ylim(0, ymax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Ref preview", fontsize=8)
        self.ax.tick_params(labelsize=6)
