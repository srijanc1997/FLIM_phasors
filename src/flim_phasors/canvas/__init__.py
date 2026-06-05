"""Matplotlib/Qt canvas widgets for FLIM phasor and image display.

Re-exports :class:`ImageCanvas` and :class:`PhasorCanvas` for convenient imports.
"""

from flim_phasors.canvas.image import ImageCanvas
from flim_phasors.canvas.phasor import PhasorCanvas

__all__ = ["ImageCanvas", "PhasorCanvas"]
