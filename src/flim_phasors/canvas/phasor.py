"""Interactive phasor plot with circular segmentation cursors."""
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle, Ellipse
from PySide6.QtCore import Signal
from phasorpy.lifetime import phasor_from_lifetime, phasor_semicircle
from phasorpy.phasor import phasor_from_signal

from flim_phasors.constants import COMPARE_CMAPS, COMPARE_SCATTER_MAX
from flim_phasors.data import PhasorData
from flim_phasors.utils import (
    categorical_name,
    categorical_rgb,
    dataset_short_label,
)


def freq_ok(data):
    return data is not None and data.frequency and data.work_frequency > 0


def cmap_mid_color(name):
    return matplotlib.cm.get_cmap(name)(0.65)


def _subsample_phasor_points(g, s, max_points=COMPARE_SCATTER_MAX):
    n = int(g.size)
    if n <= max_points:
        return g, s
    idx = np.random.default_rng(0).choice(n, max_points, replace=False)
    return g[idx], s[idx]


class PhasorCanvas(FigureCanvas):
    cursorChanged = Signal()   # committed: gesture end / add / remove (full recompute)
    cursorMoving = Signal()    # interactive: drag / scroll / slider (fast overlay only)
    cursorSelectionChanged = Signal(int)  # active circle index (-1 if none)
    phasorClicked = Signal(float, float)  # g, s when clicking outside cursors

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.data = None
        self.compare_enabled = False
        self.compare_style = "cloud"   # cloud | scatter | summary
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

    def set_compare(self, enabled, style, layers):
        self.compare_enabled = bool(enabled)
        self.compare_style = style if style in ("cloud", "scatter", "summary") else "cloud"
        self.compare_layers = list(layers or [])

    def update_display(self, data, compare_enabled=False, compare_style="cloud", compare_layers=None):
        self.data = data
        self.set_compare(compare_enabled, compare_style, compare_layers)
        self.redraw_hist()

    def _draw_lifetime_ticks(self, data):
        if not freq_ok(data):
            return
        for tau in (0.5, 1, 2, 3, 4, 8):
            gg, ss = phasor_from_lifetime(data.work_frequency, tau)
            self.ax.plot(gg, ss, "k.", ms=4)
            self.ax.annotate(f"{tau}ns", (gg, ss), fontsize=7, alpha=0.7)

    def _draw_single_cloud(self, d):
        g, s = d.real_cal, d.imag_cal
        m = d.valid_mask()
        if m.sum() > 0:
            self.ax.hist2d(g[m].ravel(), s[m].ravel(), bins=256,
                           range=[[0, 1.05], [0, 0.75]], cmap="turbo", cmin=1)

    def _draw_layer_cloud(self, d, color_idx):
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
        g, s = d.real_cal, d.imag_cal
        m = d.valid_mask()
        if m.sum() == 0:
            return
        gv, sv = _subsample_phasor_points(g[m].ravel(), s[m].ravel())
        self.ax.scatter(gv, sv, s=0.6, alpha=0.14, c=[color], label=label, rasterized=True)

    def _draw_layer_summary(self, d, color, label):
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
            handles, labels = self.ax.get_legend_handles_labels()
            if style == "cloud" and not labels:
                from matplotlib.lines import Line2D
                handles, labels = [], []
                for layer in visible_layers:
                    idx = layer.get("index", 0)
                    handles.append(Line2D(
                        [0], [0], marker="s", linestyle="",
                        markerfacecolor=cmap_mid_color(COMPARE_CMAPS[idx % len(COMPARE_CMAPS)]),
                        markersize=8))
                    labels.append(layer.get("label", ""))
            if labels:
                self.ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.88)

        self._redraw_cursors()
        if self._click_marker is not None:
            self.ax.plot(
                self._click_marker[0], self._click_marker[1],
                "wx", ms=10, mew=2, zorder=20)
        self.ax.set_xlim(0, 1.05)
        self.ax.set_ylim(0, 0.75)
        self.draw_idle()

    def set_click_marker(self, g: float | None, s: float | None):
        if g is None or s is None:
            self._click_marker = None
        else:
            self._click_marker = (float(g), float(s))
        self.redraw_hist()

    def add_cursor(self, radius=0.05, *, kind="circle", radius_minor=None, angle=0.0):
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
        self.select_cursor(idx, emit=True)
        self.cursorChanged.emit()

    def select_cursor(self, index, emit=True):
        """Select which circle is active for move / resize / delete."""
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
        if 0 <= self.selected < len(self.cursors):
            self._remove_cursor_artists(self.cursors[self.selected])
            self.cursors.pop(self.selected)
            new_sel = len(self.cursors) - 1 if self.cursors else -1
            self.select_cursor(new_sel, emit=True)
            self.draw()
            self.cursorChanged.emit()

    def clear_cursors(self):
        for c in self.cursors:
            self._remove_cursor_artists(c)
        self.cursors = []
        self.select_cursor(-1, emit=True)
        self._purge_stray_cursor_artists()
        self.draw(); self.cursorChanged.emit()

    def set_selected_radius(self, r):
        if 0 <= self.selected < len(self.cursors):
            self.cursors[self.selected]["radius"] = float(r)
            self._redraw_cursors(); self.draw_idle(); self.cursorMoving.emit()

    def _remove_cursor_artists(self, c):
        art = c.get("patch")
        if art is not None:
            try:
                art.remove()
            except Exception:
                pass
            c["patch"] = None

    def _purge_stray_cursor_artists(self):
        """Remove circle patches left behind after a cursor was deleted."""
        keep_patches = {c["patch"] for c in self.cursors if c.get("patch")}
        for p in list(self.ax.patches):
            if isinstance(p, (Circle, Ellipse)) and p not in keep_patches:
                try:
                    p.remove()
                except Exception:
                    pass

    def _redraw_cursors(self):
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
        """Draw phasorpy GMM ellipses (95% confidence, sigma baked into radii)."""
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
        for a in self._gmm_artists:
            try: a.remove()
            except Exception: pass
        self._gmm_artists = []
        self.draw_idle()

    def _cursor_hit(self, c, x, y):
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
        """Return indices of cursors under (x, y), ordered for picking (topmost first)."""
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
        if not self._dragging or event.inaxes != self.ax or event.xdata is None:
            return
        if 0 <= self.selected < len(self.cursors):
            self.cursors[self.selected]["center_real"] = float(event.xdata)
            self.cursors[self.selected]["center_imag"] = float(event.ydata)
            self._redraw_cursors(); self.draw_idle(); self.cursorMoving.emit()

    def on_release(self, event):
        if self._dragging:
            self._dragging = False; self.cursorChanged.emit()

    def on_scroll(self, event):
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
