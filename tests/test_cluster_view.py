"""Cluster brightness multiplies the painted pixels of that color."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from phasorpy.cursor import pseudo_color

from flim_phasors.cluster_view import (
    cluster_color_image,
    compose_cluster_overlay,
)


def test_gain_one_matches_phasorpy_overlay():
    intensity = np.array([[0.05, 0.4], [0.8, 0.0]], dtype=float)
    mask = np.array([[True, False], [True, True]])
    color = (0.8, 0.1, 0.15)
    got = compose_cluster_overlay(intensity, mask[None, ...], [color], [1.0])
    ref = pseudo_color(mask, intensity=intensity, colors=np.array([color]))
    assert np.allclose(got, ref, atol=1e-5)


def test_brightness_multiplies_the_painted_cloud_and_keeps_its_hue():
    intensity = np.array([[10.0, 0.0], [0.0, 0.0]], dtype=float)
    # Peak is 10, so this pixel normalizes to 1 and the overlay is the color
    # itself only at full scale. Use a dim pixel against a brighter neighbor.
    intensity = np.array([[0.5, 10.0], [0.0, 0.0]], dtype=float)
    mask = np.array([[True, False], [False, False]])
    color = (0.8, 0.1, 0.15)
    dim = compose_cluster_overlay(intensity, mask[None, ...], [color], [1.0])
    bright = compose_cluster_overlay(intensity, mask[None, ...], [color], [4.0])
    assert np.allclose(bright[0, 0], np.clip(dim[0, 0] * 4.0, 0, 1), atol=1e-5)
    assert bright[0, 0, 0] > dim[0, 0, 0]
    # Red stays the strongest channel, so the cloud does not turn white or gray.
    assert bright[0, 0, 0] > bright[0, 0, 1]
    assert bright[0, 0, 0] > bright[0, 0, 2]
    # The other pixel is outside the cluster and is not scaled.
    assert np.allclose(bright[0, 1], dim[0, 1])


def test_one_cluster_image_paints_only_that_mask():
    intensity = np.array([[4.0, 9.0], [1.0, 2.0]], dtype=float)
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    alone = cluster_color_image(intensity, mask, (0.2, 0.8, 0.2), gain=1.0)
    assert alone[0, 1].sum() == 0.0
    assert alone[0, 0].sum() > 0.0


def test_separate_cluster_export_writes_png_phasor_and_tiff(tmp_path):
    import tifffile

    from flim_phasors.data import PhasorData
    from flim_phasors.export_bundle import (
        _export_separate_clusters,
        _write_cluster_color_tiff,
    )

    d = PhasorData()
    d.real_cal = np.full((4, 4), 0.4)
    d.imag_cal = np.full((4, 4), 0.3)
    # Cluster pixels are dim next to a bright neighbor, which is the case
    # the brightness multiplier is meant to lift.
    d.mean_raw = np.full((4, 4), 1.0)
    d.mean_raw[0, 2] = 20.0
    d.mean_thr = d.mean_raw.copy()
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 0, 1] = True
    stats = [
        {"color": (0.8, 0.1, 0.1), "label": "A", "brightness": 3.0},
        {"color": (0.1, 0.2, 0.9), "label": "B", "brightness": 1.0},
    ]
    written = []
    _export_separate_clusters(tmp_path, d, masks, stats, written, prefix="s")
    assert _write_cluster_color_tiff(
        tmp_path, d, masks, stats, written, prefix="s") is None
    assert (tmp_path / "s_cluster_01.png").is_file()
    assert (tmp_path / "s_phasor_cluster_01.png").is_file()
    pages = tifffile.imread(tmp_path / "s_colors.tif")
    assert pages.shape == (2, 4, 4, 3)
    # Page 0 is the red cluster picture. The bright neighbor is not that color.
    assert pages[0, 0, 0, 0] > pages[0, 0, 0, 1]
    assert int(pages[0, 0, 2].sum()) == 0
    assert int(pages[1, 0, 0].sum()) == 0
    assert pages[1, 0, 1, 2] > pages[1, 0, 1, 0]
    # Brightness 3 lifts the painted PNG; the TIFF page uses that same picture.
    import matplotlib.image as mpimg

    painted = mpimg.imread(tmp_path / "s_cluster_01.png")
    assert painted[0, 0, 0] > painted[0, 0, 1]


def test_missing_tifffile_warns_with_the_install_command(monkeypatch):
    import builtins

    from flim_phasors.export_bundle import TIFF_INSTALL_COMMAND, _write_cluster_color_tiff

    real_import = builtins.__import__

    def _block_tifffile(name, *args, **kwargs):
        if name == "tifffile":
            raise ImportError("no tifffile")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_tifffile)
    from flim_phasors.data import PhasorData

    d = PhasorData()
    d.mean_raw = np.ones((2, 2))
    masks = np.zeros((1, 2, 2), dtype=bool)
    masks[0, 0, 0] = True
    note = _write_cluster_color_tiff(
        None, d, masks, [{"label": "red", "color": (1, 0, 0)}], [], prefix="s")
    assert TIFF_INSTALL_COMMAND in note
    assert "pip install tifffile" in note


def test_saved_masks_follow_the_colors_on_the_image():
    from flim_phasors.cluster_view import masks_shown_on_image

    intensity = np.array([[1.0, 8.0, 0.2], [0.0, 3.0, 6.0]], dtype=float)
    masks = np.zeros((2, 2, 3), dtype=bool)
    masks[0, 0, 0] = True
    masks[0, 1, 1] = True
    masks[1, 0, 1] = True  # overlaps red; green is painted last
    masks[1, 0, 2] = True
    red, green = (0.85, 0.15, 0.12), (0.15, 0.75, 0.2)
    overlay = compose_cluster_overlay(intensity, masks, [red, green], [1.0, 2.0])
    shown = masks_shown_on_image(overlay, [red, green], masks)
    # The bright green pixel and the dim red pixel keep their image colors.
    assert shown[0, 0, 0]
    assert not shown[1, 0, 0]
    assert shown[1, 0, 2]
    assert not shown[0, 0, 2]
    # Overlap shows the later color, which is the color on the image.
    assert shown[1, 0, 1]
    assert not shown[0, 0, 1]
    # Background is neither color.
    assert not shown[:, 1, 0].any()
    assert not shown[:, 1, 2].any()


def test_cluster_phasor_outline_follows_the_saved_pixels():
    from flim_phasors.export_bundle import _masked_phasor_ellipse

    rng = np.random.default_rng(0)
    g = 0.2 + rng.normal(0, 0.01, 40)
    s = 0.4 + rng.normal(0, 0.008, 40)
    center, ellipse = _masked_phasor_ellipse(g, s)
    assert abs(center[0] - 0.2) < 0.02
    assert abs(center[1] - 0.4) < 0.02
    width, height, _angle = ellipse
    # 2 standard deviations of this tight cloud, not a plot-sized GMM ellipse.
    assert width < 0.2
    assert height < 0.2


def test_pixel_size_reads_micrometers_from_ptu_metadata():
    from flim_phasors.data import PhasorData

    d = PhasorData()
    d._pixel_size_from_attrs({"ptu_tags": {"ImgHdr_PixResol": 0.2}})
    assert d.pixel_size_um == 0.2

    d = PhasorData()
    # PTU coordinates are meters: 0.2 µm is 2e-7 m between samples.
    d._pixel_size_from_attrs({"coords": {"X": [0.0, 2e-7, 4e-7]}})
    assert abs(d.pixel_size_um - 0.2) < 1e-9

    d = PhasorData()
    d._pixel_size_from_attrs({"coords": {"X": [0.0, 0.25, 0.5]}})
    assert d.pixel_size_um == 0.25


def test_scale_bar_length_shrinks_when_it_would_fill_the_image():
    from flim_phasors.gui.main_window import _fitted_scale_bar_um

    # 256 px at 0.2 µm/px is a 51 µm field. 50 µm would cover almost all of it.
    fitted = _fitted_scale_bar_um(256, 0.2, 50.0)
    assert fitted < 20
    assert _fitted_scale_bar_um(256, 0.2, 10.0) == 10.0


def test_results_table_brightness_lifts_that_cluster_only():
    from PySide6 import QtWidgets

    from flim_phasors.data import PhasorData
    from flim_phasors.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    intensity = np.zeros((8, 8), dtype=float)
    intensity[0, 0] = 0.5
    intensity[0, 1] = 10.0
    d = PhasorData()
    d.mean_raw = intensity
    d.mean_thr = intensity.copy()
    d.real_cal = np.full((8, 8), 0.5)
    d.imag_cal = np.full((8, 8), 0.2)
    win.data = d
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    win.cluster_stats = [dict(
        idx=1, color=(0.8, 0.1, 0.15), label="A",
        g=0.5, s=0.2, tp=1.0, tm=1.0, tn=1.0, n=1, area=10.0, brightness=1.0,
    )]
    win._seg_masks = mask[None, ...]
    win._fill_table()
    assert win.table.columnCount() == 11
    assert win.table.item(0, 3).text() == "A"
    spin = win.table.cellWidget(0, 2)
    assert isinstance(spin, QtWidgets.QDoubleSpinBox)
    assert spin.value() == 1.0
    win._recompose_overlay()
    dim = np.array(win.last_overlay)
    spin.setValue(4.0)
    app.processEvents()
    bright = np.array(win.last_overlay)
    assert np.allclose(bright[0, 0], np.clip(dim[0, 0] * 4.0, 0, 1), atol=1e-4)
    assert bright[0, 0, 0] > dim[0, 0, 0]
    assert np.allclose(bright[0, 1], dim[0, 1])
    win.close()


def test_gmm_image_list_follows_the_setup_sample_list():
    """Analyze Image and Setup Active sample select the same loaded image."""
    from PySide6 import QtWidgets

    from flim_phasors.data import PhasorData
    from flim_phasors.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()

    def sample(name):
        d = PhasorData()
        d.display_name = name
        d.sample_path = f"{name}.ptu"
        d.mean_raw = np.ones((8, 8))
        d.mean_thr = d.mean_raw.copy()
        d.real_cal = np.full((8, 8), 0.4)
        d.imag_cal = np.full((8, 8), 0.2)
        return d

    first, second = sample("alpha"), sample("beta")
    win.datasets = [first]
    win.data = first
    win.active_idx = 0
    win.chk_multi.setChecked(True)
    app.processEvents()
    assert win._gmm_sample_row.isHidden()

    win.datasets = [first, second]
    win._refresh_sample_combo()
    app.processEvents()
    assert not win._gmm_sample_row.isHidden()
    assert win.cb_sample.count() == 2
    assert [win.cb_gmm_sample.itemText(i) for i in range(2)] == [
        win.cb_sample.itemText(i) for i in range(2)
    ]

    win.rb_gmm.setChecked(True)
    win.cb_gmm_sample.setCurrentIndex(1)
    app.processEvents()
    assert win.active_idx == 1
    assert win.data is second
    assert win.cb_sample.currentIndex() == 1

    win.cb_sample.setCurrentIndex(0)
    app.processEvents()
    assert win.active_idx == 0
    assert win.data is first
    assert win.cb_gmm_sample.currentIndex() == 0
    win._dirty = False
    win.close()
