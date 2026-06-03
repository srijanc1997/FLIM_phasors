"""Application entry point and Qt/matplotlib bootstrap."""

from __future__ import annotations

import os
import sys


def _configure_backends():
    os.environ.setdefault("QT_API", "pyside6")
    try:
        from PySide6 import QtWidgets  # noqa: F401
    except ImportError:
        sys.exit("PySide6 is required:  pip install PySide6")
    import matplotlib

    matplotlib.use("QtAgg")


def main():
    _configure_backends()
    from PySide6 import QtWidgets

    from flim_phasors.gui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
