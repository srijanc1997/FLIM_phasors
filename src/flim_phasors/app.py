"""Application entry point and Qt/matplotlib bootstrap.

Configures PySide6 and the QtAgg matplotlib backend, then launches the main
FLIM phasor analysis window.
"""

from __future__ import annotations

import os
import sys


def _configure_backends():
    """Select PySide6 for Qt and QtAgg for matplotlib before any GUI imports.

    Must run before importing ``MainWindow`` or creating matplotlib figures so
    Qt bindings and the canvas backend agree. Sets ``QT_API`` to ``pyside6``
    when unset, verifies PySide6 can be imported, then switches matplotlib to
    the QtAgg backend.

    Exits the process with an install hint if PySide6 is not available. Takes
    no arguments and returns ``None``.
    """
    os.environ.setdefault("QT_API", "pyside6")
    try:
        from PySide6 import QtWidgets  # noqa: F401
    except ImportError:
        sys.exit("PySide6 is required:  pip install PySide6")
    import matplotlib

    matplotlib.use("QtAgg")


def main():
    """Create the Qt application and show the main FLIM phasor window.

    Configures the Qt/matplotlib backends first (so the correct binding is
    selected before any Qt or matplotlib symbols are imported), then builds
    and displays :class:`~flim_phasors.gui.main_window.MainWindow` and hands
    control to the Qt event loop for the remainder of the process lifetime.

    Takes no arguments; reads ``sys.argv`` for Qt's own command-line options.
    This call does not return — it exits the process via ``sys.exit`` once
    the Qt event loop ends.
    """
    _configure_backends()
    from PySide6 import QtWidgets

    from flim_phasors.gui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
