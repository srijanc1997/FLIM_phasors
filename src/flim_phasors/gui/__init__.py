"""GUI package — avoid importing MainWindow here (pulls in Qt/matplotlib).

Use ``from flim_phasors.gui.main_window import MainWindow`` instead.
"""

__all__ = ["MainWindow"]


def __getattr__(name: str):
    """Lazy-load ``MainWindow`` on first attribute access (PEP 562)."""
    if name == "MainWindow":
        from flim_phasors.gui.main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
