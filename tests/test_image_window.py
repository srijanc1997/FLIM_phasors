"""Brightness and contrast move the display window in the expected direction."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from flim_phasors.canvas.image import _adjust_rgb, _adjust_window


def test_positive_brightness_brightens():
    lo, hi = _adjust_window(0.0, 1.0, brightness=0.5, contrast=0.0)
    assert (lo + hi) / 2.0 < 0.5
    mid = np.full((1, 1, 3), 0.5)
    out = _adjust_rgb(mid, brightness=0.5, contrast=0.0)
    assert out[0, 0, 0] > 0.5


def test_show_scale_bar_tick_adds_and_removes_the_bar():
    from PySide6 import QtWidgets

    from flim_phasors.data import PhasorData
    from flim_phasors.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    d = PhasorData()
    d.mean_raw = np.ones((32, 40))
    d.mean_thr = d.mean_raw.copy()
    d.real_cal = np.full((32, 40), 0.4)
    d.imag_cal = np.full((32, 40), 0.2)
    d.pixel_size_um = 0.2
    win.data = d
    win.sp_pixel_um.setValue(0.2)
    app.processEvents()
    assert win.chk_show_scale_bar.isChecked()
    assert len(win.image._scale_artists) == 3

    win.chk_show_scale_bar.setChecked(False)
    app.processEvents()
    assert win.image._scale_artists == []

    win.chk_show_scale_bar.setChecked(True)
    app.processEvents()
    assert len(win.image._scale_artists) == 3

    # A lifetime view with no map used to skip the redraw, so the tick
    # could not take the bar off the image that was still on screen.
    win.cb_image_view.setCurrentIndex(2)
    app.processEvents()
    win.chk_show_scale_bar.setChecked(False)
    app.processEvents()
    assert win.image._scale_artists == []
    win.chk_show_scale_bar.setChecked(True)
    app.processEvents()
    assert len(win.image._scale_artists) == 3

    win.sp_pixel_um.setValue(0.0)
    app.processEvents()
    assert win.image._scale_artists == []
    win.close()


def test_positive_contrast_spreads_midtones():
    lo, hi = _adjust_window(0.0, 1.0, brightness=0.0, contrast=0.5)
    assert (hi - lo) < 1.0
    img = np.zeros((1, 2, 3))
    img[0, 0] = 0.4
    img[0, 1] = 0.6
    out = _adjust_rgb(img, brightness=0.0, contrast=0.5)
    assert out[0, 1, 0] - out[0, 0, 0] > 0.2
