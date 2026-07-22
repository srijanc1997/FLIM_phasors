"""FLIM image and lifetime-map display canvas.

Provides :class:`ImageCanvas`, a Qt-embedded matplotlib figure for intensity
images, scalar maps, overlays, and scale bars. Reuses the existing
``AxesImage`` (and colorbar) when shape and view kind match, avoiding a full
``fig.clear()`` on every refresh.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from PySide6.QtCore import Signal


class ImageCanvas(FigureCanvas):
    """Matplotlib canvas for FLIM intensity images and derived scalar maps."""

    imageClicked = Signal(int, int)  # row y, column x in image pixel coordinates

    def __init__(self, parent=None):
        """Initialize an empty image canvas with axes hidden.

        Args:
            parent: Optional Qt parent widget.
        """
        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self._im = None
        self._cbar = None
        self._view_kind = None  # "intensity" | "map" | "overlay"
        self._scale_artists: list = []
        self._click_marker = None
        self._click_marker_artist = None
        self.mpl_connect("button_press_event", self._on_press)

    def _reset_ax(self):
        """Rebuild the axes from scratch so any previous colorbar is dropped."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self._im = None
        self._cbar = None
        self._view_kind = None
        self._scale_artists = []
        self._click_marker_artist = None

    def _clear_scale_bar(self):
        """Remove any previously drawn scale-bar artists."""
        for art in self._scale_artists:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._scale_artists = []

    def _array_shape2d(self, arr) -> tuple[int, int]:
        """Return ``(H, W)`` for a display array (ignoring channel axis)."""
        a = np.asarray(arr)
        if a.ndim >= 2:
            return int(a.shape[0]), int(a.shape[1])
        return 0, 0

    def _can_reuse(self, kind: str, arr, *, with_cbar: bool) -> bool:
        """Return whether the current AxesImage can be updated in place."""
        if self._im is None or self._view_kind != kind:
            return False
        cur = self._im.get_array()
        if cur is None:
            return False
        if self._array_shape2d(cur) != self._array_shape2d(arr):
            return False
        has_cbar = self._cbar is not None
        return has_cbar == with_cbar

    def _set_or_create_image(
        self,
        arr,
        *,
        kind: str,
        cmap,
        vmin,
        vmax,
        with_cbar: bool,
        cbar_label: str | None = None,
        title: str | None = None,
    ):
        """Update existing image artist or rebuild axes when needed."""
        self._clear_scale_bar()
        if self._can_reuse(kind, arr, with_cbar=with_cbar):
            self._im.set_data(arr)
            if vmin is not None and vmax is not None:
                self._im.set_clim(vmin, vmax)
            if cmap is not None:
                self._im.set_cmap(cmap)
            if self._cbar is not None and cbar_label is not None:
                self._cbar.update_normal(self._im)
                self._cbar.set_label(cbar_label)
            if title:
                self.ax.set_title(title, fontsize=9)
            else:
                self.ax.set_title("")
            self._view_kind = kind
            return

        self._reset_ax()
        self._im = self.ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        self._view_kind = kind
        if with_cbar:
            self._cbar = self.fig.colorbar(
                self._im, ax=self.ax, fraction=0.046, pad=0.04)
            if cbar_label:
                self._cbar.set_label(cbar_label)
        if title:
            self.ax.set_title(title, fontsize=9)

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
        """Show a photon-count intensity image.

        NaN values are masked and shown dark. Contrast limits are derived from
        finite pixels only. Reuses the existing image artist when possible.

        Args:
            mean: 2-D (or squeezable) intensity array.
            vmin: Optional lower display limit; auto-computed when ``None``.
            vmax: Optional upper display limit; auto-computed when ``None``.
            log_scale: When ``True``, apply log10 scaling to positive values.
            auto_contrast: When ``True``, use 2nd–98th percentile for limits.
            title: Optional axes title.
        """
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
        self._set_or_create_image(
            masked, kind="intensity", cmap="gray", vmin=vmin, vmax=vmax,
            with_cbar=False, title=title,
        )
        self._last_intensity = arr
        self._redraw_click_marker()
        self.draw_idle()

    def set_click_marker(self, y: int | None, x: int | None):
        """Show or clear a crosshair at image pixel ``(y, x)``.

        Args:
            y: Row index, or ``None`` to clear the marker.
            x: Column index, or ``None`` to clear the marker.
        """
        if y is None or x is None:
            self._click_marker = None
        else:
            self._click_marker = (int(y), int(x))
        self._redraw_click_marker()

    def _redraw_click_marker(self):
        """Draw the stored click crosshair on the current image axes."""
        if self._click_marker_artist is not None:
            try:
                self._click_marker_artist.remove()
            except (ValueError, AttributeError):
                pass
            self._click_marker_artist = None
        if self._click_marker is not None and self._im is not None:
            y, x = self._click_marker
            (self._click_marker_artist,) = self.ax.plot(
                x, y, "c+", ms=14, mew=2, zorder=20)
        self.draw_idle()

    def _on_press(self, event):
        """Emit :attr:`imageClicked` for left-clicks inside the image axes."""
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        y = int(round(event.ydata))
        x = int(round(event.xdata))
        self.imageClicked.emit(y, x)

    def show_map(self, arr, title, cmap="viridis", label="ns", vmin=None, vmax=None):
        """Display a per-pixel scalar map with a colorbar.

        Typical use: apparent lifetime (τ) or other derived quantity maps.
        Reuses the existing image + colorbar when shape matches.

        Args:
            arr: 2-D scalar array to display.
            title: Axes title.
            cmap: Matplotlib colormap name.
            label: Colorbar axis label.
            vmin: Optional lower color scale limit.
            vmax: Optional upper color scale limit.
        """
        data = np.asarray(arr, dtype=float)
        self._set_or_create_image(
            data, kind="map", cmap=cmap, vmin=vmin, vmax=vmax,
            with_cbar=True, cbar_label=label, title=title,
        )
        self._redraw_click_marker()
        self.draw_idle()

    def show_overlay(self, overlay, title=None):
        """Display an RGB or RGBA overlay image.

        Args:
            overlay: Image array accepted by ``Axes.imshow``.
            title: Optional axes title.
        """
        self._set_or_create_image(
            overlay, kind="overlay", cmap=None, vmin=None, vmax=None,
            with_cbar=False, title=title,
        )
        self._redraw_click_marker()
        self.draw_idle()

    def draw_scale_bar(self, bar_pixels: float, *, label: str = "10 µm"):
        """Draw a horizontal scale bar on the current image.

        Previous scale-bar artists are removed first so reuse-based redraws
        do not stack bars.

        Args:
            bar_pixels: Bar width in image pixel units.
            label: Text shown below the bar.
        """
        if self._im is None:
            return
        self._clear_scale_bar()
        h, w = self._im.get_array().shape[:2]
        x0 = w * 0.05
        y0 = h * 0.92
        rect = Rectangle(
            (x0, y0), bar_pixels, max(2, h * 0.01),
            fc="white", ec="black", lw=0.8)
        self.ax.add_patch(rect)
        txt = self.ax.text(
            x0 + bar_pixels / 2, y0 - h * 0.03, label, color="white",
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        self._scale_artists = [rect, txt]
        self.draw_idle()

    def update_overlay(self, overlay, title=None):
        """Fast path for live overlay updates via ``AxesImage.set_data``.

        Reuses the existing axes when shape matches; otherwise rebuilds the view.

        Args:
            overlay: New overlay array.
            title: Optional title.
        """
        if self._can_reuse("overlay", overlay, with_cbar=False):
            self._clear_scale_bar()
            self._im.set_data(overlay)
            if title:
                self.ax.set_title(title, fontsize=9)
        else:
            self.show_overlay(overlay, title=title)
            return
        self._redraw_click_marker()
        self.draw_idle()
