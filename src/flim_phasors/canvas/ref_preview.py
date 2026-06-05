"""Small reference phasor preview plot."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from phasorpy.lifetime import phasor_semicircle


class RefPreviewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(2.4, 2.1), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(120)
        self.ax = self.fig.add_subplot(111)
        self._draw_empty()

    def _draw_empty(self):
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
        if cal.use_manual:
            return float(cal.manual_g), float(cal.manual_s)
        return float(cal.mean_g), float(cal.mean_s)

    def show_calibration(self, cal, *, ref_lifetime_ns: float = 4.0, frequency_mhz: float = 80.0):
        del ref_lifetime_ns, frequency_mhz  # reserved for future τ marker on semicircle
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
                    self.ax.scatter(
                        gr[::step], sr[::step],
                        s=2, alpha=0.3, c="steelblue", rasterized=True, zorder=2)
            if np.isfinite(rg) and np.isfinite(rs):
                self.ax.plot(rg, rs, "r+", ms=12, mew=2, zorder=5)
                self.ax.annotate(
                    f"g={rg:.3f}\ns={rs:.3f}",
                    (rg, rs),
                    fontsize=7,
                    xytext=(5, 5),
                    textcoords="offset points",
                    zorder=6,
                )
        self._style_axes(rg, rs)
        self.fig.tight_layout()
        self.draw()

    def _style_axes(self, rg: float = 0.0, rs: float = 0.0):
        xmax = max(1.05, float(rg) + 0.08) if rg > 0 else 1.05
        ymax = max(0.55, float(rs) + 0.1) if rs > 0 else 0.75
        self.ax.set_xlim(0, xmax)
        self.ax.set_ylim(0, ymax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Ref preview", fontsize=8)
        self.ax.tick_params(labelsize=6)
