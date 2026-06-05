"""FLIM image / lifetime map display canvas."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self._im = None
        self._cbar = None

    def _reset_ax(self):
        """Rebuild the axes from scratch so any previous colorbar is dropped."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self._cbar = None

    def show_intensity(
        self,
        mean,
        vmin=None,
        vmax=None,
        *,
        log_scale: bool = False,
        auto_contrast: bool = True,
        title: str | None = None,
    ):
        """Show photon image; NaN = masked (shown dark), scaling from finite pixels only."""
        self._reset_ax()
        arr = np.squeeze(np.asarray(mean, dtype=float))
        while arr.ndim > 2:
            arr = arr[0]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            arr = np.zeros((2, 2))
            finite = arr.ravel()
        display = arr.copy()
        if log_scale:
            display = np.where(np.isfinite(display) & (display > 0), display, np.nan)
            pos = display[np.isfinite(display) & (display > 0)]
            if pos.size:
                display = np.log10(np.maximum(display, np.nanmin(pos) * 0.1))
        if vmin is None or vmax is None:
            src = display[np.isfinite(display)]
            if src.size == 0:
                lo, hi = 0.0, 1.0
            elif auto_contrast:
                lo, hi = np.percentile(src, [2, 98])
            else:
                lo, hi = float(np.min(src)), float(np.max(src))
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        if vmax <= vmin:
            vmax = vmin + 1.0
        masked = np.ma.masked_invalid(display)
        self._im = self.ax.imshow(masked, cmap="gray", vmin=vmin, vmax=vmax)
        self._last_intensity = arr
        if title:
            self.ax.set_title(title, fontsize=9)
        self.draw_idle()

    def show_map(self, arr, title, cmap="viridis", label="ns", vmin=None, vmax=None):
        """Display a per-pixel scalar map (e.g. apparent lifetime) with a colorbar."""
        self._reset_ax()
        self._im = self.ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        self._cbar = self.fig.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.04)
        self._cbar.set_label(label)
        self.draw_idle()

    def show_overlay(self, overlay, title=None):
        self._reset_ax()
        self._im = self.ax.imshow(overlay)
        self.draw_idle()

    def draw_scale_bar(self, bar_pixels: float, *, label: str = "10 µm"):
        if self._im is None:
            return
        from matplotlib.patches import Rectangle
        h, w = self._im.get_array().shape[:2]
        x0 = w * 0.05
        y0 = h * 0.92
        rect = Rectangle((x0, y0), bar_pixels, max(2, h * 0.01), fc="white", ec="black", lw=0.8)
        self.ax.add_patch(rect)
        self.ax.text(x0 + bar_pixels / 2, y0 - h * 0.03, label, color="white",
                     ha="center", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        self.draw_idle()

    # --- unused (focused cleanup): uncomment if needed ---
    # def show_click_marker(self, shape, y: int, x: int, *, title: str = ""):
    #     """Overlay a crosshair at (y,x) on the current intensity view."""
    #     disp = getattr(self, "_last_intensity", None)
    #     if disp is None:
    #         return
    #     self.show_intensity(disp, title=title or "Phasor click")
    #     self.ax.plot(x, y, "c+", ms=14, mew=2)
    #     self.draw_idle()

    def update_overlay(self, overlay, title=None):
        """Fast path for live updates: reuse the AxesImage via set_data."""
        arr = None if self._im is None else self._im.get_array()
        if (self._cbar is None and self._im is not None and arr is not None
                and tuple(arr.shape) == tuple(overlay.shape)):
            self._im.set_data(overlay)
        else:
            self._reset_ax()
            self._im = self.ax.imshow(overlay)
        self.draw_idle()
