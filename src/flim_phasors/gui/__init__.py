"""GUI package — avoid importing MainWindow here (pulls in Qt/matplotlib).

Use ``from flim_phasors.gui.main_window import MainWindow`` instead.
"""

__all__ = ["MainWindow"]


def __getattr__(name: str):
    """Lazy-load heavy GUI modules on first attribute access.

    Implements :pep:`562` module-level ``__getattr__`` so that
    ``from flim_phasors.gui import MainWindow`` only imports
    :mod:`flim_phasors.gui.main_window` (and, transitively, Qt and
    matplotlib) the first time ``MainWindow`` is actually accessed,
    keeping a plain ``import flim_phasors.gui`` cheap for code that only
    needs other package contents.

    Args:
        name: Attribute name requested on this package.

    Returns:
        The requested attribute (currently only ``MainWindow``).

    Raises:
        AttributeError: If ``name`` is not a known lazy export.
    """
    if name == "MainWindow":
        from flim_phasors.gui.main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
