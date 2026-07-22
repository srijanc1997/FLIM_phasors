"""Interactive phasor plot with circular segmentation cursors.

Provides :class:`PhasorCanvas`, a Qt-embedded matplotlib figure for phasor
density histograms, multi-dataset comparison overlays, GMM ellipses, and
user-drawn circle/ellipse segmentation cursors.
"""
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle, Ellipse
from PySide6.QtCore import Signal
from phasorpy.lifetime import phasor_from_lifetime, phasor_semicircle
from flim_phasors.constants import (
    COMPARE_CMAPS,
    COMPARE_SCATTER_MAX,
    PHASOR_HIST_BINS,
    PHASOR_HIST_CACHE_MAX,
    PHASOR_HIST_MAX_POINTS,
)
from flim_phasors.data import PhasorData
from flim_phasors.utils import (
    categorical_name,
    categorical_rgb,
    dataset_short_label,
)


def freq_ok(data):
    """Return whether a dataset has a valid modulation frequency for tick marks.

    Used to gate :meth:`PhasorCanvas._draw_lifetime_ticks`, since placing
    lifetime annotations on the universal semicircle requires converting a
    lifetime to a phasor coordinate via the laser frequency; without a
    positive ``work_frequency`` that conversion is undefined and the ticks
    are simply skipped.

    Args:
        data: :class:`~flim_phasors.data.PhasorData` instance or ``None``.

    Returns:
        ``True`` when ``data`` is non-null and has positive ``work_frequency``.
    """
    return data is not None and data.frequency and data.work_frequency > 0


def cmap_mid_color(name):
    """Sample a colormap at normalized position 0.65 for legend swatches.

    Compare-mode density clouds are drawn with partial alpha near the low
    end of their colormap, so a swatch taken from the very start of the
    colormap would look washed out in the legend; 0.65 was chosen as a
    representative, clearly visible color for that cloud's legend marker.

    Args:
        name: Matplotlib colormap name.

    Returns:
        RGBA color tuple.
    """
    return matplotlib.cm.get_cmap(name)(0.65)


def _subsample_phasor_points(g, s, max_points=COMPARE_SCATTER_MAX):
    """Randomly subsample phasor coordinates for scatter compare mode.

    Uses a fixed RNG seed for reproducible plots.

    Args:
        g: Real (g) coordinate array.
        s: Imaginary (s) coordinate array.
        max_points: Maximum number of points to retain.

    Returns:
        Tuple ``(g_sub, s_sub)`` of subsampled arrays.
    """
    n = int(g.size)
    if n <= max_points:
        return g, s
    idx = np.random.default_rng(0).choice(n, max_points, replace=False)
    return g[idx], s[idx]


def _phasor_map_key(d) -> tuple:
    """Build a stable cache key for a dataset's calibrated phasor maps.

    Used to key :attr:`PhasorCanvas._hist_cache` so repeated redraws (cursor
    drags, click markers, legend updates) can reuse a previously binned 2-D
    histogram instead of recomputing it. The key mixes object identity
    (``id()``) of the dataset and its ``real_cal``/``imag_cal`` arrays with
    the array shape rather than array contents, since hashing or comparing
    full phasor maps on every redraw would be far more expensive than the
    identity check. This means the cache is only valid as long as calibrated
    arrays are replaced (not mutated in place) whenever calibration changes —
    which matches how :class:`~flim_phasors.data.PhasorData` is updated
    elsewhere in the codebase.

    Args:
        d: :class:`~flim_phasors.data.PhasorData` instance, or ``None``.

    Returns:
        A hashable tuple suitable for use as a dict key. When ``d`` is
        ``None`` or lacks calibrated maps, returns ``(id(d), None, None,
        None)`` so all "empty" datasets collapse onto distinct-but-stable
        keys per object identity.
    """
    if d is None or d.real_cal is None or d.imag_cal is None:
        return (id(d), None, None, None)
    return (
        id(d),
        id(d.real_cal),
        id(d.imag_cal),
        tuple(d.real_cal.shape),
    )


class PhasorCanvas(FigureCanvas):
    """Matplotlib canvas for interactive phasor histograms and segmentation cursors.

    Renders one or more datasets as 2-D density histograms (or scatter/summary
    overlays in compare mode) on the universal semicircle, and lets the user
    draw, move, resize, and delete circular or elliptical segmentation cursors
    directly on the plot. Phasor coordinates follow the convention ``g`` (real
    part, x-axis) and ``s`` (imaginary part, y-axis). Cursors are stored as
    plain dicts in :attr:`cursors` (keys: ``center_real``, ``center_imag``,
    ``radius``, ``kind``, ``radius_minor``, ``angle``, ``color``, ``label``,
    ``patch``) so they can be serialized independently of their matplotlib
    ``patch`` artist, which is rebuilt on every redraw. Two-dimensional
    histograms are memoized in :attr:`_hist_cache` keyed by dataset identity
    and array shape so that cursor drags and click markers can redraw quickly
    without recomputing the binning.

    Emits Qt signals when cursors are added, moved, resized, or when the user
    clicks on the phasor plane outside existing cursors:

    * :attr:`cursorChanged` — a "committed" edit (add/remove/drag end) that
      warrants a full downstream recompute (masks, lifetimes, etc.).
    * :attr:`cursorMoving` — a fast, interactive update (drag/scroll/slider)
      meant for cheap overlay-only redraws.
    * :attr:`cursorSelectionChanged` — the active cursor index changed.
    * :attr:`phasorClicked` — the user clicked the phasor plane outside any
      cursor, reporting the ``(g, s)`` coordinate.
    """

    cursorChanged = Signal()   # committed: gesture end / add / remove (full recompute)
    cursorMoving = Signal()    # interactive: drag / scroll / slider (fast overlay only)
    cursorSelectionChanged = Signal(int)  # active circle index (-1 if none)
    phasorClicked = Signal(float, float)  # g, s when clicking outside cursors

    def __init__(self, parent=None):
        """Initialize axes, mouse handlers, and empty cursor state.

        Sets up the matplotlib figure/axes, the compare-mode defaults
        (single dataset, cloud style, no layers), the empty segmentation
        cursor list, and an empty 2-D histogram cache, then wires up the
        button-press/release, motion, and scroll Qt/matplotlib event
        callbacks that drive cursor selection, dragging, and resizing.

        Args:
            parent: Optional Qt parent widget.
        """
        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.data = None
        self.compare_enabled = False
        self.compare_style = "cloud"   # cloud | scatter | summary
        self.legend_loc = "upper right"
        self.legend_fontsize = 11.0
        self.compare_layers = []       # [{data, label, color, visible, index}, ...]
        self.cursors = []
        self.selected = -1
        self._dragging = False
        self._gmm_artists = []
        self._click_marker = None
        self._click_marker_artist = None
        self._image_highlight = None
        # Cached 2-D histograms: key -> (counts, xedges, yedges)
        self._hist_cache: dict = {}
        self._init_axes()
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("scroll_event", self.on_scroll)

    def _init_axes(self):
        """Clear and configure the phasor axes with the universal semicircle.

        Called at the start of every :meth:`redraw_hist` (and once during
        ``__init__``) to reset the axes to a known baseline before layering
        histograms, GMM ellipses, and cursors back on top. Also drops the
        cached click-marker artist reference, since ``ax.clear()`` destroys
        all artists including it; callers are responsible for re-adding the
        marker afterward (see :meth:`_redraw_click_marker`). The universal
        semicircle (the locus of single-exponential lifetimes in phasor
        space) is drawn once here so every subsequent redraw shows it, and
        the ``(g, s)`` axis limits and 1:1 aspect ratio are fixed so distances
        on screen match distances in phasor units.
        """
        self.ax.clear()
        self._click_marker_artist = None
        self.ax.set_xlabel("g")
        self.ax.set_ylabel("s")
        g_uni, s_uni = phasor_semicircle(201)
        self.ax.plot(g_uni, s_uni, "k-", lw=1, alpha=0.6)
        self.ax.set_xlim(0, 1.05)
        self.ax.set_ylim(0, 0.75)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.margins(0)
        self.ax.grid(alpha=0.2)

    # --- unused (focused cleanup): uncomment if needed ---
    # def set_data(self, data):
    #     self.data = data

    def set_compare(self, enabled, style, layers, *, legend_loc=None, legend_fontsize=None):
        """Configure multi-dataset compare overlay mode.

        Stores the compare flag, style, and layer list on the canvas so
        the next :meth:`redraw_hist` renders clouds, scatter points, or
        mean/std summaries for each layer instead of a single dataset.
        Does not redraw by itself; callers typically follow with
        :meth:`update_display` or :meth:`redraw_hist`.

        Args:
            enabled: When ``True``, render multiple compare layers.
            style: One of ``"cloud"``, ``"scatter"``, or ``"summary"``.
            layers: List of layer dicts with ``data``, ``label``, ``color``, etc.
            legend_loc: Optional matplotlib legend location string.
            legend_fontsize: Optional legend font size (minimum 6 pt).
        """
        self.compare_enabled = bool(enabled)
        self.compare_style = style if style in ("cloud", "scatter", "summary") else "cloud"
        self.compare_layers = list(layers or [])
        if legend_loc:
            self.legend_loc = str(legend_loc)
        if legend_fontsize is not None:
            self.legend_fontsize = max(6.0, float(legend_fontsize))

    def update_display(
        self,
        data,
        compare_enabled=False,
        compare_style="cloud",
        compare_layers=None,
        *,
        legend_loc=None,
        legend_fontsize=None,
    ):
        """Replace the active dataset and redraw the phasor histogram.

        Convenience wrapper that combines :attr:`data` assignment with
        :meth:`set_compare` and a full :meth:`redraw_hist`, so callers
        updating both the primary dataset and compare configuration in
        response to a UI change don't need to sequence the calls
        themselves.

        Args:
            data: Primary :class:`~flim_phasors.data.PhasorData` to display.
            compare_enabled: Enable multi-layer compare rendering.
            compare_style: Compare visualization style.
            compare_layers: Optional list of compare layer dicts.
            legend_loc: Optional legend location override.
            legend_fontsize: Optional legend font size override.
        """
        self.data = data
        self.set_compare(
            compare_enabled, compare_style, compare_layers,
            legend_loc=legend_loc, legend_fontsize=legend_fontsize,
        )
        self.redraw_hist()

    def _draw_lifetime_ticks(self, data):
        """Annotate standard lifetime values on the universal semicircle.

        Marks the phasor positions corresponding to a fixed set of
        reference lifetimes (0.5-8 ns) so users can visually calibrate
        cursor placement against known single-exponential decays. Silently
        does nothing when ``data`` lacks a usable modulation frequency
        (see :func:`freq_ok`), since the lifetime-to-phasor conversion is
        frequency-dependent.

        Args:
            data: Dataset supplying ``work_frequency`` for tick placement.
        """
        if not freq_ok(data):
            return
        for tau in (0.5, 1, 2, 3, 4, 8):
            gg, ss = phasor_from_lifetime(data.work_frequency, tau)
            self.ax.plot(gg, ss, "k.", ms=4)
            self.ax.annotate(f"{tau}ns", (gg, ss), fontsize=7, alpha=0.7)

    def _store_hist_cache(self, key, value):
        """Insert a histogram into the bounded, insertion-ordered cache.

        Implements a simple LRU-ish eviction policy: inserting (or
        re-inserting, which callers do by popping then storing) a key moves
        it to the "most recently used" end of the cache because Python dicts
        preserve insertion order. Once the cache exceeds
        :data:`~flim_phasors.constants.PHASOR_HIST_CACHE_MAX` entries, the
        oldest key (least recently touched) is evicted. This keeps memory
        bounded when many datasets or bin-count/key-prefix combinations are
        cycled through during a session (e.g. switching between single,
        compare-cloud, and per-layer histograms).

        Args:
            key: Cache key, typically produced alongside a bins/key_prefix
                pair and a :func:`_phasor_map_key` for the dataset.
            value: The ``(counts, xedges, yedges)`` tuple to store.
        """
        self._hist_cache[key] = value
        while len(self._hist_cache) > PHASOR_HIST_CACHE_MAX:
            # dict preserves insertion order (Py3.7+)
            self._hist_cache.pop(next(iter(self._hist_cache)))

    def _histogram2d_cached(self, d, *, bins: int, key_prefix: str):
        """Return ``(counts, xedges, yedges)`` for valid phasor pixels.

        Results are cached by map identity so repeated redraws (cursor moves,
        click markers) skip re-binning. Large clouds are subsampled before
        binning to keep hist2d cheap. Bins with fewer than 1 count are set to
        ``NaN`` (matching matplotlib's ``hist2d(cmin=1)`` behavior) so empty
        regions of phasor space are rendered transparent instead of as a
        solid low-value color. ``key_prefix`` lets callers keep separate cache
        entries for the same dataset rendered at different resolutions or in
        different compare-layer roles (e.g. ``"single"`` vs ``"cmp0"``,
        ``"cmp1"``, ...), since bin count and role both affect the result.

        Args:
            d: :class:`~flim_phasors.data.PhasorData` with calibrated
                ``real_cal``/``imag_cal`` maps and a ``valid_mask()`` method.
            bins: Number of bins per axis passed to ``np.histogram2d``.
            key_prefix: Cache-key namespace distinguishing single-view vs
                per-compare-layer histograms at potentially different bin
                counts.

        Returns:
            Tuple ``(counts, xedges, yedges)`` where ``counts`` has shape
            ``(bins, bins)`` with ``NaN`` in empty bins, and ``xedges``/
            ``yedges`` are the bin edge arrays along g and s respectively.
            When the dataset has no valid pixels, returns an all-``NaN``
            histogram over the default ``[0, 1.05] x [0, 0.75]`` phasor
            range.
        """
        key = (key_prefix, bins, _phasor_map_key(d))
        cached = self._hist_cache.get(key)
        if cached is not None:
            # Refresh LRU order
            self._hist_cache.pop(key)
            self._store_hist_cache(key, cached)
            return cached

        g, s = d.real_cal, d.imag_cal
        m = d.valid_mask()
        if m.sum() == 0:
            empty = (
                np.zeros((bins, bins), dtype=float),
                np.linspace(0, 1.05, bins + 1),
                np.linspace(0, 0.75, bins + 1),
            )
            self._store_hist_cache(key, empty)
            return empty

        gv = g[m].ravel()
        sv = s[m].ravel()
        gv, sv = _subsample_phasor_points(gv, sv, max_points=PHASOR_HIST_MAX_POINTS)
        counts, xedges, yedges = np.histogram2d(
            gv, sv, bins=bins, range=[[0, 1.05], [0, 0.75]])
        # Match hist2d(cmin=1): hide empty bins
        counts = counts.astype(float)
        counts[counts < 1] = np.nan
        value = (counts, xedges, yedges)
        self._store_hist_cache(key, value)
        return value

    def _draw_single_cloud(self, d):
        """Render a single-dataset phasor density histogram (turbo colormap).

        Used when compare mode is off (or only one layer is visible).
        Pulls a cached, pre-binned 2-D histogram via
        :meth:`_histogram2d_cached` and paints it with ``pcolormesh``; if
        the dataset has no finite bins (e.g. no valid pixels), nothing is
        drawn.

        Args:
            d: :class:`~flim_phasors.data.PhasorData` with calibrated maps.
        """
        counts, xedges, yedges = self._histogram2d_cached(
            d, bins=PHASOR_HIST_BINS, key_prefix="single")
        if not np.any(np.isfinite(counts)):
            return
        # .T: histogram2d is (x,y) but pcolormesh expects rows = y bins.
        self.ax.pcolormesh(
            xedges, yedges, counts.T, cmap="turbo", shading="auto",
            rasterized=True)

    def _draw_layer_cloud(self, d, color_idx):
        """Render one compare-layer density cloud with a distinct colormap.

        Used in multi-layer compare mode with ``style="cloud"``: each layer
        gets its own colormap (cycled from
        :data:`~flim_phasors.constants.COMPARE_CMAPS`) and is drawn with
        partial transparency and a fixed ``zorder`` so overlapping layers
        remain individually distinguishable rather than one fully
        occluding another.

        Args:
            d: Dataset for this layer.
            color_idx: Index into :data:`~flim_phasors.constants.COMPARE_CMAPS`.
        """
        counts, xedges, yedges = self._histogram2d_cached(
            d, bins=200, key_prefix=f"cmp{color_idx}")
        if not np.any(np.isfinite(counts)):
            return
        im = self.ax.pcolormesh(
            xedges, yedges, counts.T,
            cmap=COMPARE_CMAPS[color_idx % len(COMPARE_CMAPS)],
            shading="auto", rasterized=True, alpha=0.38)
        im.set_zorder(1)


    def _draw_layer_scatter(self, d, color, label):
        """Render subsampled scatter points for one compare layer.

        Used in multi-layer compare mode with ``style="scatter"``. Points
        are subsampled via :func:`_subsample_phasor_points` and drawn with
        very small markers and low alpha so overlapping layers remain
        legible; the ``label`` is attached to the scatter artist so
        :meth:`_draw_compare_legend` can build a legend directly from the
        axes' handles instead of constructing proxy artists.

        Args:
            d: Dataset for this layer.
            color: Matplotlib color for scatter markers.
            label: Legend label for this layer.
        """
        g, s = d.real_cal, d.imag_cal
        m = d.valid_mask()
        if m.sum() == 0:
            return
        gv, sv = _subsample_phasor_points(g[m].ravel(), s[m].ravel())
        self.ax.scatter(gv, sv, s=0.6, alpha=0.14, c=[color], label=label, rasterized=True)

    def _draw_layer_summary(self, d, color, label):
        """Render mean ± std error bars for one compare layer.

        Used in multi-layer compare mode with ``style="summary"``, this
        collapses an entire dataset's valid phasor pixels down to a single
        marker at the mean ``(g, s)`` with error bars at the standard
        deviation along each axis, which is useful for comparing many
        samples at a glance without the visual clutter of full clouds or
        scatter points. Error bars have a small floor (``1e-6``) so a
        dataset with zero spread still renders a visible bar.

        Args:
            d: Dataset for this layer.
            color: Marker and error bar color.
            label: Legend label for this layer.
        """
        g, s = d.real_cal, d.imag_cal
        m = d.valid_mask()
        if m.sum() == 0:
            return
        gv, sv = g[m].ravel(), s[m].ravel()
        mg, ms = float(np.nanmean(gv)), float(np.nanmean(sv))
        sg, ss = float(np.nanstd(gv)), float(np.nanstd(sv))
        self.ax.errorbar(
            mg, ms, xerr=max(sg, 1e-6), yerr=max(ss, 1e-6),
            fmt="o", color=color, label=label, capsize=3, ms=8, mew=1.5, alpha=0.95)

    def redraw_hist(self):
        """Rebuild the full phasor plot: semicircle, clouds, legend, and cursors.

        This is the "full" redraw path (as opposed to the cheaper
        :meth:`_redraw_cursors`/:meth:`_redraw_click_marker` paths used during
        drags) and is called whenever the underlying dataset, compare
        configuration, or lifetime-tick frequency changes. It re-derives the
        set of visible layers from :attr:`data` or :attr:`compare_layers`
        (skipping layers without calibrated maps or marked invisible),
        chooses a rendering style per layer — density cloud, scatter, or
        mean/std "summary" — based on :attr:`compare_style`, draws a legend
        only when more than one layer is visible in compare mode, and finally
        re-adds cursor patches and the click-marker crosshair, which
        :meth:`_init_axes` would otherwise have discarded via ``ax.clear()``.
        The 2-D histogram cache is untouched by this call, so repeated
        redraws of an unchanged dataset are cheap.
        """
        # Clears axes (and cursor patch refs) but _hist_cache survives — keyed by numpy array id.
        self._init_axes()
        visible_layers = []
        if self.compare_enabled and self.compare_layers:
            visible_layers = [
                L for L in self.compare_layers
                if L.get("visible", True) and L["data"].real_cal is not None
            ]
        elif self.data is not None and self.data.real_cal is not None:
            visible_layers = [{
                "data": self.data,
                "label": dataset_short_label(self.data, 0),
                "color": categorical_rgb(0),
                "index": 0,
            }]

        tick_data = self.data
        if tick_data is None or tick_data.real_cal is None:
            tick_data = visible_layers[0]["data"] if visible_layers else None
        self._draw_lifetime_ticks(tick_data)

        multi_compare = self.compare_enabled and len(visible_layers) > 1
        style = self.compare_style if self.compare_enabled else "cloud"

        for i, layer in enumerate(visible_layers):
            d = layer["data"]
            label = layer.get("label", dataset_short_label(d, i))
            color = layer.get("color", categorical_rgb(i))
            idx = layer.get("index", i)
            if style == "summary":
                self._draw_layer_summary(d, color, label)
            elif style == "scatter":
                self._draw_layer_scatter(d, color, label)
            elif multi_compare:
                self._draw_layer_cloud(d, idx)
            else:
                self._draw_single_cloud(d)

        if multi_compare:
            self._draw_compare_legend(visible_layers, style)

        self._redraw_cursors()
        self._redraw_click_marker(draw=False)
        self.ax.set_xlim(0, 1.05)
        self.ax.set_ylim(0, 0.75)
        self.draw_idle()

    def _redraw_click_marker(self, *, draw: bool = True):
        """Draw or clear the phasor click crosshair without rebuilding the hist.

        The crosshair is a lightweight, separately tracked artist
        (``_click_marker_artist``) so it can be shown or moved via
        :meth:`set_click_marker` without paying the cost of a full
        :meth:`redraw_hist`. The previous marker artist (if any) is always
        removed first — both to move it and because ``_init_axes`` clears the
        axes and invalidates any prior reference. When ``_click_marker`` is
        ``None`` the crosshair simply stays absent after removal.

        Args:
            draw: When ``True`` (default), request a canvas repaint via
                ``draw_idle()`` after updating the artist. Callers that are
                about to redraw the whole figure anyway (e.g. the end of
                :meth:`redraw_hist`) pass ``False`` to avoid a redundant
                paint.
        """
        if self._click_marker_artist is not None:
            try:
                self._click_marker_artist.remove()
            except (ValueError, AttributeError):
                pass
            self._click_marker_artist = None
        if self._click_marker is not None:
            (self._click_marker_artist,) = self.ax.plot(
                self._click_marker[0], self._click_marker[1],
                "wx", ms=10, mew=2, zorder=20)
        if draw:
            self.draw_idle()

    def _remove_legend(self):
        """Remove the current axes legend if one exists.

        Matplotlib keeps at most one legend per axes, but calling
        ``ax.legend(...)`` again does not automatically discard a
        differently-styled previous legend artist in every version, so
        compare-mode redraws explicitly remove the old legend before
        deciding whether (and how) to draw a new one. Safe to call when no
        legend is present.
        """
        leg = self.ax.get_legend()
        if leg is not None:
            leg.remove()

    def _draw_compare_legend(self, visible_layers, style):
        """Build or refresh the compare-mode legend.

        Removes any existing legend first, then either constructs proxy
        square-marker handles colored from each layer's colormap (for
        ``style="cloud"``, since pcolormesh artists don't make good legend
        handles) or reuses the axes' own scatter/errorbar handles (for
        ``"scatter"``/``"summary"``). No legend is drawn when compare mode
        is off or fewer than two layers are visible, since a legend adds
        no information for a single dataset.

        Args:
            visible_layers: Layers currently drawn on the axes.
            style: Active compare style (``"cloud"``, ``"scatter"``, or ``"summary"``).
        """
        if not (self.compare_enabled and len(visible_layers) > 1):
            self._remove_legend()
            return
        self._remove_legend()
        fs = max(6.0, float(self.legend_fontsize))
        marker_pt = fs * 1.15
        handles, labels = [], []
        if style == "cloud":
            for layer in visible_layers:
                idx = layer.get("index", 0)
                handles.append(Line2D(
                    [0], [0], marker="s", linestyle="",
                    markerfacecolor=cmap_mid_color(COMPARE_CMAPS[idx % len(COMPARE_CMAPS)]),
                    markersize=marker_pt,
                    markeredgewidth=0.8))
                labels.append(layer.get("label", ""))
        else:
            handles, _ = self.ax.get_legend_handles_labels()
            labels = [layer.get("label", "") for layer in visible_layers]
        if labels:
            loc = self.legend_loc if self.legend_loc in (
                "upper right", "upper left", "lower right", "lower left", "best",
            ) else "upper right"
            self.ax.legend(
                handles, labels, loc=loc,
                fontsize=fs,
                markerscale=max(0.75, fs / 7.0),
                handlelength=max(1.2, fs / 5.5),
                handletextpad=0.6,
                labelspacing=0.35,
                framealpha=0.88,
            )

    def update_compare_legend(
        self,
        layers,
        *,
        compare_style=None,
        legend_loc=None,
        legend_fontsize=None,
    ):
        """Refresh legend text and style without rebuilding phasor density plots.

        Cheaper than :meth:`update_display`/:meth:`redraw_hist` for cases
        where only labels, colors, style, or legend placement changed
        (e.g. a sample was renamed) and the underlying density clouds,
        scatter points, or summary markers don't need to be recomputed.
        No-ops entirely when compare mode is disabled.

        Args:
            layers: Updated compare layer list.
            compare_style: Optional new compare style.
            legend_loc: Optional new legend location.
            legend_fontsize: Optional new legend font size.
        """
        if not self.compare_enabled:
            return
        self.compare_layers = list(layers or [])
        if compare_style in ("cloud", "scatter", "summary"):
            self.compare_style = compare_style
        if legend_loc:
            self.legend_loc = str(legend_loc)
        if legend_fontsize is not None:
            self.legend_fontsize = max(6.0, float(legend_fontsize))
        visible_layers = [
            L for L in self.compare_layers
            if L.get("visible", True) and L["data"].real_cal is not None
        ]
        self._draw_compare_legend(visible_layers, self.compare_style)
        self.draw_idle()

    def set_click_marker(self, g: float | None, s: float | None):
        """Show or clear a crosshair at a phasor click position.

        Updates only the marker artist (does not rebuild the density histogram).

        Args:
            g: Real coordinate, or ``None`` to clear the marker.
            s: Imaginary coordinate, or ``None`` to clear the marker.
        """
        if g is None or s is None:
            self._click_marker = None
        else:
            self._click_marker = (float(g), float(s))
        self._redraw_click_marker(draw=True)

    def add_cursor(self, radius=0.05, *, kind="circle", radius_minor=None, angle=0.0,
                   emit_changed=True):
        """Add a new segmentation cursor centered on the data median (or default).

        Centering on the median of the currently loaded dataset's valid
        pixels gives a sensible starting position over the densest region
        of the phasor cloud; when no dataset (or no valid pixels) is
        available, falls back to a fixed default position. The new cursor
        is appended to :attr:`cursors`, immediately selected via
        :meth:`select_cursor`, and — unless ``emit_changed`` is
        ``False`` — triggers :attr:`cursorChanged` so downstream masks and
        lifetime computations refresh.

        Args:
            radius: Major radius in phasor units.
            kind: ``"circle"`` or ``"ellipse"``.
            radius_minor: Ellipse minor radius; defaults to ``0.65 * radius``.
            angle: Ellipse rotation angle in radians.
            emit_changed: When ``True``, emit :attr:`cursorChanged` after creation.
        """
        if self.data is not None and self.data.valid_mask().sum() > 0:
            m = self.data.valid_mask()
            cr = float(np.nanmedian(self.data.real_cal[m]))
            ci = float(np.nanmedian(self.data.imag_cal[m]))
        else:
            cr, ci = 0.5, 0.3
        idx = len(self.cursors)
        r = float(radius)
        rm = float(radius_minor) if radius_minor is not None else r * 0.65
        self.cursors.append(dict(
            center_real=cr, center_imag=ci, radius=r,
            kind=kind if kind in ("circle", "ellipse") else "circle",
            radius_minor=rm, angle=float(angle),
            color=categorical_rgb(idx),
            label=categorical_name(idx), patch=None))
        self.select_cursor(idx, emit=emit_changed)
        if emit_changed:
            self.cursorChanged.emit()

    def select_cursor(self, index, emit=True):
        """Select which cursor is active for move, resize, or delete.

        Clamps ``index`` to ``-1`` when it is out of range or when there
        are no cursors, then redraws cursor patches so the newly selected
        one is highlighted with a thicker outline (see
        :meth:`_redraw_cursors`). :attr:`cursorSelectionChanged` is only
        emitted when the selection actually changes and ``emit`` is
        ``True``, avoiding redundant signal traffic when re-selecting the
        same cursor.

        Args:
            index: Cursor index, or ``-1`` for no selection.
            emit: When ``True``, emit :attr:`cursorSelectionChanged` on change.
        """
        if not self.cursors:
            index = -1
        elif index < 0 or index >= len(self.cursors):
            index = -1
        changed = index != self.selected
        self.selected = index
        self._redraw_cursors()
        self.draw_idle()
        if emit and changed:
            self.cursorSelectionChanged.emit(index)

    def remove_selected(self):
        """Delete the currently selected cursor and emit :attr:`cursorChanged`.

        No-ops when :attr:`selected` is out of range (``-1`` or stale),
        which happens whenever no cursor is currently active. On removal, the
        cursor's matplotlib patch artist is detached first via
        :meth:`_remove_cursor_artists`, then the cursor dict is popped from
        :attr:`cursors`. Selection falls back to the new last cursor in the
        list (or ``-1`` if the list becomes empty) so a subsequent delete
        continues to operate on a sensible target rather than an index that
        no longer exists.
        """
        if 0 <= self.selected < len(self.cursors):
            self._remove_cursor_artists(self.cursors[self.selected])
            self.cursors.pop(self.selected)
            new_sel = len(self.cursors) - 1 if self.cursors else -1
            self.select_cursor(new_sel, emit=True)
            self.draw()
            self.cursorChanged.emit()

    def clear_cursors(self):
        """Remove all segmentation cursors and reset selection.

        Detaches every cursor's matplotlib patch artist before clearing
        :attr:`cursors` to an empty list, then calls
        :meth:`_purge_stray_cursor_artists` as a defensive sweep for any
        ``Circle``/``Ellipse`` patches that ended up on the axes outside the
        tracked cursor dicts (e.g. from a code path that added a patch
        directly). Selection is reset to ``-1`` and both
        :attr:`cursorSelectionChanged` and :attr:`cursorChanged` are emitted
        so listeners (segmentation panels, mask overlays) drop all
        cursor-derived state.
        """
        for c in self.cursors:
            self._remove_cursor_artists(c)
        self.cursors = []
        self.select_cursor(-1, emit=True)
        self._purge_stray_cursor_artists()
        self.draw(); self.cursorChanged.emit()

    @property
    def is_dragging_cursor(self) -> bool:
        """Return whether the user is actively dragging a cursor.

        Reflects the internal ``_dragging`` flag, which is set ``True`` in
        :meth:`on_press` when a left-click lands with a cursor selected and
        cleared in :meth:`on_release`. Note that scroll-wheel resizing
        (:meth:`on_scroll`) never sets this flag — only click-and-drag moves
        do — so callers using this property to suppress other UI updates
        during interaction should also consider scroll events separately if
        needed.

        Returns:
            ``True`` while a mouse-button drag of the selected cursor is in
            progress, ``False`` otherwise.
        """
        return self._dragging

    def set_selected_radius(self, r):
        """Set the major radius of the selected cursor (slider control).

        Intended for continuous slider-driven resizing: updates the radius
        in place, redraws the cursor patch, and emits the lightweight
        :attr:`cursorMoving` signal (rather than :attr:`cursorChanged`) so
        listeners perform cheap overlay-only updates during interactive
        dragging instead of a full recompute on every slider tick. No-ops
        when no cursor is selected.

        Args:
            r: New radius in phasor units.
        """
        if 0 <= self.selected < len(self.cursors):
            self.cursors[self.selected]["radius"] = float(r)
            self._redraw_cursors(); self.draw_idle(); self.cursorMoving.emit()

    def _remove_cursor_artists(self, c):
        """Remove matplotlib patch artists attached to one cursor dict.

        Detaches the cursor's ``"patch"`` artist from the axes (if any)
        and clears the dict entry back to ``None``. Errors from an
        already-removed or stale artist are swallowed, since axes-clearing
        redraws (:meth:`_init_axes`) can invalidate patch references
        before this is called.

        Args:
            c: Cursor state dict with optional ``"patch"`` key.
        """
        art = c.get("patch")
        if art is not None:
            try:
                art.remove()
            except Exception:
                pass
            c["patch"] = None

    def _purge_stray_cursor_artists(self):
        """Remove circle/ellipse patches left behind after cursor deletion.

        Defensive cleanup complementing :meth:`_remove_cursor_artists`: it
        scans ``ax.patches`` for any ``Circle`` or ``Ellipse`` artist that is
        not currently referenced by a live cursor dict's ``"patch"`` entry
        and removes it. This guards against orphaned patches that can
        accumulate if a cursor's artist reference is lost or overwritten
        without being explicitly removed first (e.g. due to an exception
        mid-update), which would otherwise silently pile up as invisible or
        stale shapes on the phasor plot across repeated redraws.
        """
        keep_patches = {c["patch"] for c in self.cursors if c.get("patch")}
        for p in list(self.ax.patches):
            if isinstance(p, (Circle, Ellipse)) and p not in keep_patches:
                try:
                    p.remove()
                except Exception:
                    pass

    def _redraw_cursors(self):
        """Rebuild the matplotlib patch for every cursor from scratch.

        Called after any cursor add/move/resize/select/delete so drawn
        shapes stay in sync with the `cursors` state dicts. Detaches every
        existing patch artist first (via `_remove_cursor_artists`) and
        sweeps for orphaned patches (`_purge_stray_cursor_artists`) before
        creating fresh `Circle`/`Ellipse` artists, since matplotlib patches
        cannot have their center/radius/angle updated in place as cheaply as
        just recreating them. The selected cursor is drawn with a thicker
        outline (`lw=2.5` vs `1.2`) so it is visually distinguishable.
        """
        for c in self.cursors:
            self._remove_cursor_artists(c)
        self._purge_stray_cursor_artists()
        for i, c in enumerate(self.cursors):
            lw = 2.5 if i == self.selected else 1.2
            if c.get("kind") == "ellipse":
                patch = Ellipse(
                    (c["center_real"], c["center_imag"]),
                    2 * c["radius"], 2 * c.get("radius_minor", c["radius"] * 0.65),
                    angle=np.degrees(c.get("angle", 0.0)),
                    fill=False, edgecolor=c["color"], lw=lw)
            else:
                patch = Circle(
                    (c["center_real"], c["center_imag"]), c["radius"],
                    fill=False, edgecolor=c["color"], lw=lw)
            self.ax.add_patch(patch)
            c["patch"] = patch

    def show_gmm_ellipses(self, center_real, center_imag, radius_major, radius_minor, angle, colors):
        """Draw phasorpy GMM component ellipses and center crosses.

        Radii are expected to include the 95% confidence scaling (sigma baked in).

        Args:
            center_real: Array of g centers per component.
            center_imag: Array of s centers per component.
            radius_major: Major-axis half-widths per component.
            radius_minor: Minor-axis half-widths per component.
            angle: Rotation angles in radians per component.
            colors: Edge/marker color per component.
        """
        self.clear_gmm()
        # GMM ellipses live in _gmm_artists, separate from user-drawn cursor patches.
        for k in range(len(center_real)):
            e = Ellipse(
                (center_real[k], center_imag[k]),
                2 * radius_major[k], 2 * radius_minor[k],
                angle=np.degrees(angle[k]),
                fill=False, edgecolor=colors[k], lw=1.5, alpha=0.9)
            self.ax.add_patch(e)
            self._gmm_artists.append(e)
            pt = self.ax.plot(center_real[k], center_imag[k], "x", color=colors[k], ms=8, mew=2)[0]
            self._gmm_artists.append(pt)
        self.draw_idle()

    # --- unused (focused cleanup): uncomment if needed; GUI uses show_gmm_ellipses ---
    # def show_gmm(self, means, covs, colors):
    #     self.clear_gmm()
    #     for k in range(len(means)):
    #         mg, ms = means[k]; cov = covs[k]
    #         vals, vecs = np.linalg.eigh(cov)
    #         order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    #         angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    #         for n in (1, 2):
    #             w, h = 2 * n * np.sqrt(np.maximum(vals, 1e-9))
    #             e = Ellipse((mg, ms), w, h, angle=angle, fill=False,
    #                         edgecolor=colors[k], lw=1.5, alpha=0.9 / n)
    #             self.ax.add_patch(e); self._gmm_artists.append(e)
    #         pt = self.ax.plot(mg, ms, "x", color=colors[k], ms=8, mew=2)[0]
    #         self._gmm_artists.append(pt)
    #     self.draw_idle()

    def clear_gmm(self):
        """Remove all GMM ellipse and center marker artists.

        ``_gmm_artists`` holds a flat list of matplotlib artists (each
        ellipse plus its paired "x" center marker) added by
        :meth:`show_gmm_ellipses`, kept separate from the user-drawn
        segmentation cursor patches so the two overlays can be cleared and
        redrawn independently. Removal errors are swallowed since an artist
        may already have been detached by an axes-clearing redraw (e.g.
        :meth:`_init_axes`) before this is called.
        """
        for a in self._gmm_artists:
            try: a.remove()
            except Exception: pass
        self._gmm_artists = []
        self.draw_idle()

    def _cursor_hit(self, c, x, y):
        """Return whether phasor coordinates ``(x, y)`` lie inside cursor ``c``.

        Hit region is 1.3× the drawn radius for easier picking.

        Args:
            c: Cursor state dict.
            x: g coordinate.
            y: s coordinate.

        Returns:
            ``True`` when the point is inside the expanded hit region.
        """
        dx = x - c["center_real"]
        dy = y - c["center_imag"]
        if c.get("kind") == "ellipse":
            ang = float(c.get("angle", 0.0))
            cos_a, sin_a = np.cos(-ang), np.sin(-ang)
            lx = cos_a * dx - sin_a * dy
            ly = sin_a * dx + cos_a * dy
            a = max(c["radius"] * 1.3, 1e-6)
            b = max(c.get("radius_minor", c["radius"] * 0.65) * 1.3, 1e-6)
            return (lx / a) ** 2 + (ly / b) ** 2 <= 1.0
        return np.hypot(dx, dy) <= c["radius"] * 1.3

    def _hit(self, x, y):
        """Return the best cursor index under ``(x, y)``, or ``-1`` if none.

        Prefers the already-selected cursor when multiple overlap; otherwise
        returns the topmost (highest index) hit.

        Args:
            x: g coordinate.
            y: s coordinate.

        Returns:
            Cursor index, or ``-1``.
        """
        hits = []
        for i, c in enumerate(self.cursors):
            if self._cursor_hit(c, x, y):
                hits.append(i)
        if not hits:
            return -1
        if self.selected in hits:
            return self.selected
        return max(hits)

    def on_press(self, event):
        """Handle mouse press: select, drag, delete, or phasor click.

        Right-click deletes a hit cursor; left-click starts drag or emits
        :attr:`phasorClicked` when no cursor is active.

        Args:
            event: Matplotlib mouse event.
        """
        if event.inaxes != self.ax or event.xdata is None:
            return
        i = self._hit(event.xdata, event.ydata)
        if event.button == 3 and i >= 0:
            self._remove_cursor_artists(self.cursors[i])
            self.cursors.pop(i)
            new_sel = len(self.cursors) - 1 if self.cursors else -1
            self.select_cursor(new_sel, emit=True)
            self.draw(); self.cursorChanged.emit()
            return
        if event.button != 1:
            return
        if i >= 0:
            self.select_cursor(i, emit=True)
        if 0 <= self.selected < len(self.cursors):
            self._dragging = True
            self.draw_idle()
            # Click outside all cursors but with one selected: drag moves that cursor here.
            if i < 0:
                self.cursorMoving.emit()
        else:
            self.select_cursor(-1, emit=True)
            self.phasorClicked.emit(float(event.xdata), float(event.ydata))

    def on_motion(self, event):
        """Handle mouse drag to move the selected cursor center.

        Connected to matplotlib's ``motion_notify_event``. Only acts while
        :attr:`_dragging` is set (by :meth:`on_press`) and the pointer is
        within the axes with valid data coordinates; otherwise it is a
        no-op so ordinary mouse movement outside a drag gesture is cheap.
        Updates the selected cursor's center, redraws cursor patches, and
        emits the lightweight :attr:`cursorMoving` signal on every move so
        listeners can do fast overlay-only updates during the drag.

        Args:
            event: Matplotlib mouse event.
        """
        if not self._dragging or event.inaxes != self.ax or event.xdata is None:
            return
        if 0 <= self.selected < len(self.cursors):
            self.cursors[self.selected]["center_real"] = float(event.xdata)
            self.cursors[self.selected]["center_imag"] = float(event.ydata)
            self._redraw_cursors(); self.draw_idle(); self.cursorMoving.emit()

    def on_release(self, event):
        """End cursor drag and emit :attr:`cursorChanged` when appropriate.

        Connected to matplotlib's ``button_release_event``. Clears the
        :attr:`_dragging` flag set in :meth:`on_press` and, only if a drag
        was actually in progress, emits the "committed"
        :attr:`cursorChanged` signal so downstream masks and lifetimes are
        recomputed once at the end of the gesture rather than on every
        intermediate :attr:`cursorMoving` tick.

        Args:
            event: Matplotlib mouse event.
        """
        # Scroll-wheel resize never sets _dragging — those commits use debounced cursorMoving instead.
        if self._dragging:
            self._dragging = False; self.cursorChanged.emit()

    def on_scroll(self, event):
        """Resize the selected cursor with the mouse scroll wheel.

        Ellipse minor radius scales proportionally with the major radius.

        Args:
            event: Matplotlib scroll event.
        """
        if event.inaxes != self.ax or not (0 <= self.selected < len(self.cursors)):
            return
        step = 0.005 if event.button == "up" else -0.005
        c = self.cursors[self.selected]
        r = max(0.005, c["radius"] + step)
        c["radius"] = r
        if c.get("kind") == "ellipse":
            ratio = c.get("radius_minor", r * 0.65) / max(c["radius"] - step, 1e-6)
            c["radius_minor"] = max(0.003, r * ratio)
        self._redraw_cursors(); self.draw_idle(); self.cursorMoving.emit()
