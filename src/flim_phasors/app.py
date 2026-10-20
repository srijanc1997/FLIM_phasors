"""Application entry point and Qt/matplotlib bootstrap.

Configures PySide6 and the QtAgg matplotlib backend, then launches the main
FLIM phasor analysis window.
"""

from __future__ import annotations

import os
import sys


def _configure_backends():
    """Select PySide6 for Qt and QtAgg for matplotlib before any GUI imports."""
    os.environ.setdefault("QT_API", "pyside6")
    try:
        from PySide6 import QtWidgets  # noqa: F401
    except ImportError:
        sys.exit("PySide6 is required:  pip install PySide6")
    import matplotlib

    matplotlib.use("QtAgg")


def main():
    """Create the Qt application and show the main FLIM phasor window."""
    _configure_backends()
    from PySide6 import QtCore, QtGui, QtWidgets

    from flim_phasors.gui.main_window import MainWindow

    # Pass fractional scaling through unchanged. Rounding 125% to 100% or 200%
    # makes stylesheet pixels and the OS font disagree, which clips spin-box text.
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # Dot decimals on every machine. A locale that uses "," makes the
    # processing spin boxes reject typed values such as "80.0".
    QtCore.QLocale.setDefault(QtCore.QLocale.c())
    app = QtWidgets.QApplication(sys.argv)
    # Fusion draws spin boxes, combo boxes, and checkboxes the same way on
    # Windows, macOS, and Linux. The native Windows style discards arrows and
    # can hide the value once a stylesheet sets the field color.
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
