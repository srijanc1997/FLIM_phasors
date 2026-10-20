"""Cross-platform form controls stay readable and editable."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from flim_phasors.gui.theme import (
    THEME_PHASOR_LAB,
    THEME_PHASOR_LAB_LIGHT,
    palette_for,
    stylesheet_for,
)


def _app():
    QtCore.QLocale.setDefault(QtCore.QLocale.c())
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    return app


def _contrast(color_a: QtGui.QColor, color_b: QtGui.QColor) -> int:
    return abs(color_a.red() - color_b.red()) + abs(color_a.green() - color_b.green()) + abs(
        color_a.blue() - color_b.blue()
    )


def test_theme_palette_text_contrasts_with_the_field():
    for theme in (THEME_PHASOR_LAB, THEME_PHASOR_LAB_LIGHT):
        pal = palette_for(theme)
        text = pal.color(QtGui.QPalette.ColorRole.Text)
        field = pal.color(QtGui.QPalette.ColorRole.Base)
        assert _contrast(text, field) > 200
        sheet = stylesheet_for(theme)
        assert 'QSpinBox[textEntry="true"]::up-button' in sheet
        assert "QSpinBox QLineEdit" in sheet
        assert "QSplitter::handle" in sheet


def _line_edit_keeps_the_field(box, fraction: float):
    """The typed value must own the spin box, not a 1 px leftover gutter."""
    editor = box.lineEdit()
    assert editor is not None
    assert editor.width() >= int(box.width() * fraction)


def test_text_entry_spinbox_keeps_the_value_visible():
    app = _app()
    app.setPalette(palette_for(THEME_PHASOR_LAB))
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(1, 1000)
    box.setDecimals(3)
    box.setSuffix(" MHz")
    box.setValue(80)
    box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
    box.setProperty("textEntry", True)
    box.setLocale(QtCore.QLocale.c())
    from flim_phasors.gui.main_window import ensure_spin_line_edit_fills_field

    box.setStyleSheet(stylesheet_for(THEME_PHASOR_LAB))
    ensure_spin_line_edit_fills_field(box)
    box.setMinimumWidth(160)
    box.resize(180, 32)
    box.show()
    app.processEvents()
    style = box.style()
    style.unpolish(box)
    style.polish(box)
    app.processEvents()

    editor = box.lineEdit()
    assert editor is not None
    assert "80.000" in editor.text()
    _line_edit_keeps_the_field(box, 0.75)

    image = box.grab().toImage()
    assert not image.isNull() and image.width() > 1 and image.height() > 1
    colors = {image.pixel(x, y) for y in range(image.height()) for x in range(image.width())}
    paints = [QtGui.QColor.fromRgba(pixel) for pixel in colors]
    # Digits must contrast with the field. A same-color value leaves one shade.
    assert max(_contrast(a, b) for a in paints for b in paints) > 200


def test_main_window_processing_fields_are_editable():
    app = _app()
    from flim_phasors.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()

    freq = win.sp_freq
    assert freq.property("textEntry") is True
    assert "80.000" in freq.lineEdit().text()
    _line_edit_keeps_the_field(freq, 0.7)
    assert win.cb_filter.currentText() == "median"
    assert win.cb_filter.minimumWidth() > 40
    assert 400 <= win.panel_wrap.minimumWidth() <= 520
    assert win.panel_wrap.maximumWidth() <= 560
    assert freq.focusPolicy() == QtCore.Qt.FocusPolicy.StrongFocus
    assert freq.locale().decimalPoint() == "."
    assert win.edit_ref_g.locale().decimalPoint() == "."
    assert win.cb_scalebar_loc.minimumWidth() > 70
    assert "Bottom left" in win.cb_scalebar_loc.itemText(0)
    assert (
        win.sp_scalebar_um.sizePolicy().horizontalPolicy()
        == QtWidgets.QSizePolicy.Policy.Maximum
    )
    win.close()
