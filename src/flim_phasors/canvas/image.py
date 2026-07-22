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


def _adjust_window(lo: float, hi: float, brightness: float, contrast: float) -> tuple[float, float]:
    """Shift/scale a display window by brightness and contrast amounts.

    Shared by :meth:`ImageCanvas.show_intensity` (applied to the grayscale
    ``vmin``/``vmax`` clim) and :meth:`ImageCanvas.show_overlay`/
    :meth:`~ImageCanvas.update_overlay` (applied to the normalized ``[0, 1]``
    RGB range of a pseudo-color segmentation overlay), so both views respond
    to the same Brightness/Contrast sliders identically.

    Args:
        lo: Lower bound of the window before adjustment.
        hi: Upper bound of the window before adjustment.
        brightness: Shifts the window's center, as a fraction of its own
            (post-contrast) half-width. ``0`` leaves it unchanged.
        contrast: Scales the window's width by ``10 ** -contrast``. ``0``
            leaves it unchanged; positive narrows (more contrast), negative
            widens (less contrast).

    Returns:
        Adjusted ``(lo, hi)``; unchanged if both ``brightness`` and
        ``contrast`` are falsy.
    """
    if not brightness and not contrast:
        return lo, hi
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * (10.0 ** (-contrast))
    center += brightness * half
    return center - half, center + half


def _adjust_rgb(overlay: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Apply a brightness/contrast window adjustment to an RGB(A) overlay array.

    ``imshow`` ignores ``vmin``/``vmax`` for RGB(A) arrays (unlike the
    grayscale intensity view), so a pseudo-color segmentation overlay needs
    its own per-channel remap to respond to the same Brightness/Contrast
    sliders as :meth:`ImageCanvas.show_intensity`. Reuses :func:`_adjust_window`
    over the normalized ``[0, 1]`` color range to get an adjusted window,
    then linearly stretches every RGB channel through it (clipped to
    ``[0, 1]``). An alpha channel, if present, passes through unchanged —
    only color, not transparency, should respond to these sliders.

    Args:
        overlay: RGB or RGBA array with values in ``[0, 1]``.
        brightness: See :func:`_adjust_window`.
        contrast: See :func:`_adjust_window`.

    Returns:
        Adjusted array, or ``overlay`` unchanged if both ``brightness`` and
        ``contrast`` are falsy.
    """
    if not brightness and not contrast:
        return overlay
    lo, hi = _adjust_window(0.0, 1.0, brightness, contrast)
    span = (hi - lo) or 1.0
    out = np.array(overlay, dtype=float, copy=True)
    n_color = min(3, out.shape[-1]) if out.ndim else 0
    out[..., :n_color] = np.clip((out[..., :n_color] - lo) / span, 0.0, 1.0)
    return out


class ImageCanvas(FigureCanvas):
    """Matplotlib canvas for FLIM intensity images and derived scalar maps.

    Displays exactly one "view" at a time — a grayscale intensity image, a
    colorbar-annotated scalar map (e.g. apparent lifetime τ), or an RGB/RGBA
    overlay — tracked via :attr:`_view_kind`. Rather than clearing and
    rebuilding the whole figure on every update (which is comparatively slow
    and causes visible flicker), the canvas reuses the existing
    ``AxesImage``/colorbar artists via :meth:`_set_or_create_image` whenever
    the incoming array's shape and view kind match what is already on
    screen, falling back to a full ``fig.clear()`` rebuild only when the view
    kind, array shape, or colorbar presence changes. This reuse path matters
    most for :meth:`update_overlay`, which is called on every frame of a live
    cursor drag. A crosshair click marker and an optional scale bar are drawn
    as separate, independently managed artist groups so they survive
    in-place image updates without being redrawn from scratch each time.
    """

    imageClicked = Signal(int, int)  # row y, column x in image pixel coordinates

    def __init__(self, parent=None):
        """Initialize an empty image canvas with axes hidden.

        Sets up the matplotlib figure/axes with decorations off (no ticks
        or frame, since images fill the canvas edge-to-edge), clears the
        cached image/colorbar/scale-bar/click-marker artist references to
        ``None``/empty so the first real draw always takes the "create"
        path in :meth:`_set_or_create_image`, and wires up the
        ``button_press_event`` handler that emits :attr:`imageClicked`.

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
        """Rebuild the axes from scratch so any previous colorbar is dropped.

        A colorbar created by ``fig.colorbar()`` lives on its own auxiliary
        ``Axes`` alongside the main image axes. Simply removing the
        ``AxesImage`` (e.g. via ``self._im.remove()``) leaves that colorbar
        axes orphaned on the figure, so switching from a colorbar view (e.g.
        a scalar map) to a non-colorbar view (e.g. intensity or overlay)
        requires clearing the whole figure and re-adding a fresh subplot.
        This is the "slow path" fallback used by :meth:`_set_or_create_image`
        when :meth:`_can_reuse` returns ``False`` — it resets all cached
        artist references (``_im``, ``_cbar``, click marker, scale bar) so
        the next draw call starts from a clean, single-axes figure with axes
        decorations hidden.
        """
        # fig.clear() is required — removing only the AxesImage leaves orphaned colorbar axes.
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self._im = None
        self._cbar = None
        self._view_kind = None
        self._scale_artists = []
        self._click_marker_artist = None

    def _clear_scale_bar(self):
        """Remove any previously drawn scale-bar artists, if present.

        The scale bar is composed of two artists (a ``Rectangle`` and a
        ``Text`` label) tracked together in :attr:`_scale_artists` so both
        can be torn down as a unit before drawing a new bar or switching
        views; without this, repeated calls to :meth:`draw_scale_bar` (or an
        image update that reuses the axes in place) would stack duplicate
        bars on top of each other. Removal errors are swallowed since the
        artists may already have been detached by a prior ``fig.clear()`` in
        :meth:`_reset_ax`.
        """
        for art in self._scale_artists:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._scale_artists = []

    def _array_shape2d(self, arr) -> tuple[int, int]:
        """Return the row/column extent of a display array, ignoring channels.

        Used by :meth:`_can_reuse` to decide whether an incoming array is
        shape-compatible with the currently displayed ``AxesImage`` without
        being tripped up by a trailing RGB/RGBA channel axis (overlays are
        ``(H, W, 3)`` or ``(H, W, 4)`` while intensity/map views are plain
        ``(H, W)``); only the first two axes are compared.

        Args:
            arr: Any array-like accepted by ``Axes.imshow`` (2-D scalar image
                or 3-D RGB/RGBA image).

        Returns:
            Tuple ``(H, W)`` of the first two dimensions, or ``(0, 0)`` if
            ``arr`` has fewer than 2 dimensions.
        """
        a = np.asarray(arr)
        if a.ndim >= 2:
            return int(a.shape[0]), int(a.shape[1])
        return 0, 0

    def _can_reuse(self, kind: str, arr, *, with_cbar: bool) -> bool:
        """Return whether the current AxesImage can be updated in place.

        In-place reuse (via ``AxesImage.set_data`` / ``set_clim`` / etc. in
        :meth:`_set_or_create_image`) avoids the cost and flicker of a full
        ``fig.clear()`` rebuild, but is only safe when all of the following
        hold: an image artist already exists, the requested view ``kind``
        matches what is currently displayed (an intensity image should never
        silently reuse an overlay's axes state), the new array's spatial
        shape matches the current one (mismatched shapes would leave stale
        axis limits/aspect), and colorbar presence matches the request
        (``with_cbar``) — since intensity and overlay views have no
        colorbar while map views do, toggling between them always requires a
        colorbar to be added or removed, which ``set_data`` cannot do.

        Args:
            kind: Requested view kind (``"intensity"``, ``"map"``, or
                ``"overlay"``).
            arr: The array that would be displayed if reuse succeeds.
            with_cbar: Whether the requested view needs a colorbar.

        Returns:
            ``True`` when the existing ``AxesImage`` can be updated in place
            for this request; ``False`` if a full axes rebuild
            (:meth:`_reset_ax`) is required.
        """
        if self._im is None or self._view_kind != kind:
            return False
        cur = self._im.get_array()
        if cur is None:
            return False
        if self._array_shape2d(cur) != self._array_shape2d(arr):
            return False
        has_cbar = self._cbar is not None
        # intensity/overlay have no colorbar; map view does — mismatch forces _reset_ax.
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
        """Update existing image artist or rebuild axes when needed.

        Central dispatcher used by :meth:`show_intensity`, :meth:`show_map`,
        and :meth:`show_overlay` (and the fast :meth:`update_overlay` path).
        First checks :meth:`_can_reuse`; if reuse is possible, updates the
        existing ``AxesImage`` data, color limits, and colormap in place,
        refreshes the colorbar's normalization and label if one is present,
        and sets or clears the title — all without touching the figure
        layout. Otherwise, calls :meth:`_reset_ax` to drop any prior
        colorbar/axes, creates a brand-new ``AxesImage`` via ``ax.imshow``,
        and optionally attaches a fresh colorbar. The scale bar is always
        cleared first since it is drawn in data coordinates that may no
        longer be valid for the new image content.

        Args:
            arr: 2-D scalar array or image array to display.
            kind: View kind (``"intensity"``, ``"map"``, or ``"overlay"``),
                stored on the canvas so later calls can detect a view switch.
            cmap: Matplotlib colormap name or object, or ``None`` to leave the
                colormap unset (used for RGB/RGBA overlays).
            vmin: Lower color-scale limit, or ``None`` to leave unset.
            vmax: Upper color-scale limit, or ``None`` to leave unset.
            with_cbar: Whether this view should have an attached colorbar.
            cbar_label: Optional colorbar axis label; only applied when
                ``with_cbar`` is ``True``.
            title: Optional axes title; cleared (set to empty string) when
                reusing an existing image if no title is supplied, so a
                previous view's title does not linger.
        """
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
        brightness: float = 0.0,
        contrast: float = 0.0,
        title: str | None = None,
    ):
        """Show a photon-count intensity image.

        NaN values are masked and shown dark. Contrast limits are derived from
        finite pixels only, then adjusted by ``brightness``/``contrast`` (see
        below) before being applied. Reuses the existing image artist when
        possible.

        Args:
            mean: 2-D (or squeezable) intensity array.
            vmin: Optional lower display limit; auto-computed when ``None``.
            vmax: Optional upper display limit; auto-computed when ``None``.
            log_scale: When ``True``, apply log10 scaling to positive values.
            auto_contrast: When ``True``, use 2nd–98th percentile for limits.
            brightness: Shifts the display window's center, as a fraction of
                its own (post-contrast) half-width. ``0`` leaves it
                unchanged; ``+1``/``-1`` shift by a full half-width toward
                brighter/darker.
            contrast: Scales the display window's width by ``10 ** -contrast``.
                ``0`` leaves it unchanged; positive values narrow the window
                (more contrast, more clipping), negative values widen it
                (less contrast).
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
        vmin, vmax = _adjust_window(vmin, vmax, brightness, contrast)
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

        Stores the pixel position (or ``None`` to clear) and delegates to
        :meth:`_redraw_click_marker`, which handles the actual artist
        creation/removal; this split lets other code paths (e.g. an image
        view switch) trigger a marker redraw without going through this
        setter.

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
        """Draw the stored click crosshair on the current image axes.

        Removes any previously drawn marker artist first (both to move it
        and because a prior :meth:`_reset_ax` may have invalidated the old
        reference), then, if :attr:`_click_marker` is set and an image is
        currently displayed, plots a cyan "+" at that pixel. Note the
        coordinate convention: :attr:`_click_marker` stores ``(y, x)`` — row,
        column, matching :attr:`imageClicked` and :meth:`set_click_marker` —
        but ``Axes.plot`` takes ``(x, y)`` in that order, so the tuple is
        unpacked and passed to ``plot`` as ``(x, y)`` here. No marker is
        drawn when ``_im`` is ``None`` (nothing to overlay it on). Always
        requests a repaint via ``draw_idle()``, even when the marker is being
        cleared.
        """
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
        """Emit :attr:`imageClicked` for left-clicks inside the image axes.

        Ignores clicks outside the image axes, non-left mouse buttons, and
        clicks where matplotlib could not resolve data coordinates (e.g.
        right at the figure edge). Converts the continuous ``event.xdata``/
        ``event.ydata`` (column, row in float pixel units, matplotlib's
        image coordinate convention) into rounded integer pixel indices and
        emits them as ``(y, x)`` — row, column — matching the convention used
        by :meth:`set_click_marker` and :attr:`_click_marker`, which is the
        reverse order from matplotlib's own ``(x, y)`` event attributes.

        Args:
            event: Matplotlib ``MouseEvent`` from the ``button_press_event``
                connection.
        """
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

    def show_overlay(self, overlay, title=None, *, brightness: float = 0.0, contrast: float = 0.0):
        """Display an RGB or RGBA overlay image.

        Used for segmentation-mask and lifetime-colored overlays drawn on
        top of the intensity image. ``brightness``/``contrast`` (see
        :func:`_adjust_window`) are applied to the RGB channels via
        :func:`_adjust_rgb` before display, since ``imshow`` has no
        vmin/vmax-style adjustment for RGB(A) arrays the way it does for the
        grayscale intensity view. Delegates to :meth:`_set_or_create_image`
        with ``kind="overlay"`` and no colormap or color limits, then
        refreshes the click marker so it remains visible above the new
        overlay.

        Args:
            overlay: Image array accepted by ``Axes.imshow``.
            title: Optional axes title.
            brightness: Passed to :func:`_adjust_rgb`.
            contrast: Passed to :func:`_adjust_rgb`.
        """
        overlay = _adjust_rgb(overlay, brightness, contrast)
        self._set_or_create_image(
            overlay, kind="overlay", cmap=None, vmin=None, vmax=None,
            with_cbar=False, title=title,
        )
        self._redraw_click_marker()
        self.draw_idle()

    def draw_scale_bar(
        self, bar_pixels: float, *, label: str = "10 µm", location: str = "bottom left",
    ):
        """Draw a horizontal scale bar on the current image.

        Previous scale-bar artists are removed first so reuse-based redraws
        do not stack bars.

        Args:
            bar_pixels: Bar width in image pixel units.
            label: Text shown below (or above, near the top) the bar.
            location: One of ``"bottom left"``, ``"bottom right"``,
                ``"top left"``, ``"top right"`` — corner of the image the bar
                is anchored to. The label sits on the side facing away from
                the corresponding edge (above the bar when anchored to the
                bottom, below it when anchored to the top) so it never runs
                off the image.
        """
        if self._im is None:
            return
        self._clear_scale_bar()
        h, w = self._im.get_array().shape[:2]
        margin_x, margin_y, gap = w * 0.05, h * 0.08, h * 0.03
        bar_h = max(2, h * 0.01)
        x0 = (w - margin_x - bar_pixels) if "right" in location else margin_x
        if "top" in location:
            y0 = margin_y
            text_y, va = y0 + bar_h + gap, "bottom"
        else:
            y0 = h - margin_y
            text_y, va = y0 - gap, "top"
        rect = Rectangle(
            (x0, y0), bar_pixels, bar_h,
            fc="white", ec="black", lw=0.8)
        self.ax.add_patch(rect)
        txt = self.ax.text(
            x0 + bar_pixels / 2, text_y, label, color="white",
            ha="center", va=va, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        self._scale_artists = [rect, txt]
        self.draw_idle()

    def update_overlay(self, overlay, title=None, *, brightness: float = 0.0, contrast: float = 0.0):
        """Fast path for live overlay updates via ``AxesImage.set_data``.

        Reuses the existing axes when shape matches; otherwise rebuilds the
        view. ``brightness``/``contrast`` are applied the same way as in
        :meth:`show_overlay` (see :func:`_adjust_rgb`); the reuse check runs
        against the raw, pre-adjustment array since the adjustment never
        changes its shape.

        Args:
            overlay: New overlay array.
            title: Optional title.
            brightness: Passed to :func:`_adjust_rgb`.
            contrast: Passed to :func:`_adjust_rgb`.
        """
        # Called on every cursor drag frame — set_data avoids fig.clear when shape is stable.
        if self._can_reuse("overlay", overlay, with_cbar=False):
            self._clear_scale_bar()
            self._im.set_data(_adjust_rgb(overlay, brightness, contrast))
            if title:
                self.ax.set_title(title, fontsize=9)
        else:
            self.show_overlay(overlay, title=title, brightness=brightness, contrast=contrast)
            return
        self._redraw_click_marker()
        self.draw_idle()
