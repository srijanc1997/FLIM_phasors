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

    def show_intensity(self, mean, vmin=None, vmax=None):
        """Show photon image; NaN = masked (shown dark), scaling from finite pixels only."""
        self._reset_ax()
        arr = np.squeeze(np.asarray(mean, dtype=float))
        while arr.ndim > 2:
            arr = arr[0]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            arr = np.zeros((2, 2))
            finite = arr.ravel()
        if vmin is None or vmax is None:
            lo, hi = np.percentile(finite, [2, 98])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        if vmax <= vmin:
            vmax = vmin + 1.0
        masked = np.ma.masked_invalid(arr)
        self._im = self.ax.imshow(masked, cmap="gray", vmin=vmin, vmax=vmax)
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
