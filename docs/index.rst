flim-phasors documentation
============================

Interactive **phasor analysis and segmentation** for fluorescence lifetime imaging (FLIM).
Works with PicoQuant ``.ptu``, Imspector ``.tif`` stacks, and Leica LAS X ``.lif`` phasor exports.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api/index

Overview
--------

The package is organized into layers:

* **Core** — ``PhasorData``, phasor math, calibration, and analysis helpers
* **I/O** — file loaders, session/cursor/calibration persistence, export bundles
* **GUI** — PySide6 main window, processing controls, and UI enhancements
* **Canvas** — matplotlib widgets for phasor, image, and reference preview plots
* **CLI** — desktop entry point and batch folder processor

Install with development and documentation extras::

   pip install -e ".[all,dev]"

Run the GUI::

   python -m flim_phasors

Build these API docs locally::

   cd docs
   sphinx-build -b html . _build/html

Open ``docs/_build/html/index.html`` in a browser.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
