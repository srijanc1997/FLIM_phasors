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
from flim_phasors.constants import COMPARE_CMAPS, COMPARE_SCATTER_MAX
from flim_phasors.data import PhasorData
from flim_phasors.utils import (
    categorical_name,
    categorical_rgb,
    dataset_short_label,
)


def freq_ok(data):
    """Return whether a dataset has a valid modulation frequency for tick marks.

    Args:
        data: :class:`~flim_phasors.data.PhasorData` instance or ``None``.

    Returns:
        ``True`` when ``data`` is non-null and has positive ``work_frequency``.
    """
    return data is not None and data.frequency and data.work_frequency > 0


def cmap_mid_color(name):
    """Sample a colormap at normalized position 0.65 for legend swatches.

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


class PhasorCanvas(FigureCanvas):
    """Matplotlib canvas for interactive phasor histograms and segmentation cursors.

    Emits Qt signals when cursors are added, moved, resized, or when the user
    clicks on the phasor plane outside existing cursors.
    """

    cursorChanged = Signal()   # committed: gesture end / add / remove (full recompute)
    cursorMoving = Signal()    # interactive: drag / scroll / slider (fast overlay only)
    cursorSelectionChanged = Signal(int)  # active circle index (-1 if none)
    phasorClicked = Signal(float, float)  # g, s when clicking outside cursors

    def __init__(self, parent=None):
        """Initialize axes, mouse handlers, and empty cursor state.

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
        self._image_highlight = None
        self._init_axes()
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("scroll_event", self.on_scroll)

    def _init_axes(self):
        """Clear and configure the phasor axes with the universal semicircle."""
        self.ax.clear()
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

        Args:
            data: Dataset supplying ``work_frequency`` for tick placement.
        """
        if not freq_ok(data):
            return
        for tau in (0.5, 1, 2, 3, 4, 8):
            gg, ss = phasor_from_lifetime(data.work_frequency, tau)
            self.ax.plot(gg, ss, "k.", ms=4)
            self.ax.annotate(f"{tau}ns", (gg, ss), fontsize=7, alpha=0.7)

    def _draw_single_cloud(self, d):
        """Render a single-dataset phasor density histogram (turbo colormap).

        Args:
            d: :class:`~flim_phasors.data.PhasorData` with calibrated maps.
        """
        g, s = d.real_cal, d.imag_cal
        m = d.valid_mask()
        if m.sum() > 0:
            self.ax.hist2d(g[m].ravel(), s[m].ravel(), bins=256,
                           range=[[0, 1.05], [0, 0.75]], cmap="turbo", cmin=1)

    def _draw_layer_cloud(self, d, color_idx):
        """Render one compare-layer density cloud with a distinct colormap.

        Args:
            d: Dataset for this layer.
            color_idx: Index into :data:`~flim_phasors.constants.COMPARE_CMAPS`.
        """
        g, s = d.real_cal, d.imag_cal
        m = d.valid_mask()
        if m.sum() == 0:
            return
        _, _, _, im = self.ax.hist2d(
            g[m].ravel(), s[m].ravel(), bins=200,
            range=[[0, 1.05], [0, 0.75]],
            cmap=COMPARE_CMAPS[color_idx % len(COMPARE_CMAPS)], cmin=1)
        im.set_alpha(0.38)

    def _draw_layer_scatter(self, d, color, label):
        """Render subsampled scatter points for one compare layer.

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
        """Rebuild the full phasor plot: semicircle, clouds, legend, and cursors."""
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
        if self._click_marker is not None:
            self.ax.plot(
                self._click_marker[0], self._click_marker[1],
                "wx", ms=10, mew=2, zorder=20)
        self.ax.set_xlim(0, 1.05)
        self.ax.set_ylim(0, 0.75)
        self.draw_idle()

    def _remove_legend(self):
        """Remove the current axes legend if one exists."""
        leg = self.ax.get_legend()
        if leg is not None:
            leg.remove()

    def _draw_compare_legend(self, visible_layers, style):
        """Build or refresh the compare-mode legend.

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

        Args:
            g: Real coordinate, or ``None`` to clear the marker.
            s: Imaginary coordinate, or ``None`` to clear the marker.
        """
        if g is None or s is None:
            self._click_marker = None
        else:
            self._click_marker = (float(g), float(s))
        self.redraw_hist()

    def add_cursor(self, radius=0.05, *, kind="circle", radius_minor=None, angle=0.0,
                   emit_changed=True):
        """Add a new segmentation cursor centered on the data median (or default).

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
        """Delete the currently selected cursor and emit :attr:`cursorChanged`."""
        if 0 <= self.selected < len(self.cursors):
            self._remove_cursor_artists(self.cursors[self.selected])
            self.cursors.pop(self.selected)
            new_sel = len(self.cursors) - 1 if self.cursors else -1
            self.select_cursor(new_sel, emit=True)
            self.draw()
            self.cursorChanged.emit()

    def clear_cursors(self):
        """Remove all segmentation cursors and reset selection."""
        for c in self.cursors:
            self._remove_cursor_artists(c)
        self.cursors = []
        self.select_cursor(-1, emit=True)
        self._purge_stray_cursor_artists()
        self.draw(); self.cursorChanged.emit()

    @property
    def is_dragging_cursor(self) -> bool:
        """Return whether the user is actively dragging a cursor."""
        return self._dragging

    def set_selected_radius(self, r):
        """Set the major radius of the selected cursor (slider control).

        Args:
            r: New radius in phasor units.
        """
        if 0 <= self.selected < len(self.cursors):
            self.cursors[self.selected]["radius"] = float(r)
            self._redraw_cursors(); self.draw_idle(); self.cursorMoving.emit()

    def _remove_cursor_artists(self, c):
        """Remove matplotlib patch artists attached to one cursor dict.

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
        """Remove circle/ellipse patches left behind after cursor deletion."""
        keep_patches = {c["patch"] for c in self.cursors if c.get("patch")}
        for p in list(self.ax.patches):
            if isinstance(p, (Circle, Ellipse)) and p not in keep_patches:
                try:
                    p.remove()
                except Exception:
                    pass

    def _redraw_cursors(self):
        """Recreate matplotlib patch artists for all cursors."""
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
        """Remove all GMM ellipse and center marker artists."""
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
            if i < 0:
                self.cursorMoving.emit()
        else:
            self.select_cursor(-1, emit=True)
            self.phasorClicked.emit(float(event.xdata), float(event.ydata))

    def on_motion(self, event):
        """Handle mouse drag to move the selected cursor center.

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

        Args:
            event: Matplotlib mouse event.
        """
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
