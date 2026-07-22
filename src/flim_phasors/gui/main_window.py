"""Main Qt window for the FLIM phasor analyzer GUI.

Hosts sample loading, reference calibration, phasor preprocessing, multi-image
comparison, cursor/GMM segmentation, and export workflows.
"""
import os
import sys
import time

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from flim_phasors import __version__
from flim_phasors.analysis import (
    fit_phasor_gmm,
    label_pixels_by_gmm,
    lifetimes_at_phasor,
)
from flim_phasors.export_bundle import export_analysis_bundle
from flim_phasors.constants import (
    CHANNEL_PRESELECT_MAX,
    COMPARE_STYLE_MAP,
    CURSOR_SHAPES,
    FILTER_MODES,
    FLIM_FILE_FILTER,
    IMAGE_VIEW_ITEMS,
    LEGEND_FORMAT_ITEMS,
    LEGEND_LOC_ITEMS,
    LEGEND_SIZE_DEFAULT,
    LEGEND_SIZE_MAX,
    LEGEND_SIZE_MIN,
)
from flim_phasors.calibration import ReferenceCalibration, compute_reference_phasor
from flim_phasors.calibration import clear_calibration_cache
from flim_phasors.data import PhasorData
from flim_phasors.io import flim_channel_count, is_supported_flim_path
from flim_phasors.lif_io import LifPhasorSeries, is_lif_path, list_lif_phasor_series
from flim_phasors.gui.lif_dialog import LifSeriesDialog
from flim_phasors.utils import dataset_has_sample
from flim_phasors.busy import CancelledError, run_busy_qt
from flim_phasors.canvas.image import ImageCanvas
from flim_phasors.canvas.phasor import PhasorCanvas
from flim_phasors.gui.enhancements import EnhancementsMixin
from flim_phasors.gui.processing import (
    apply_processing_settings_to_ui,
    capture_processing_from_ui,
    filter_label_for_dataset,
    per_sample_processing,
    processing_params_for_dataset,
    run_processing_on_dataset,
)
from flim_phasors.memory_est import format_memory_line
from flim_phasors.utils import (
    categorical_name,
    categorical_rgb,
    dataset_display_label,
    dataset_phasor_legend_label,
    dataset_short_label,
    _dataset_file_label,
)

try:
    import sklearn.mixture  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

from phasorpy.cursor import mask_from_circular_cursor, mask_from_elliptic_cursor, pseudo_color


class MainWindow(EnhancementsMixin, QtWidgets.QMainWindow):
    """Primary FLIM phasor analysis window: load, calibrate, segment, and export.

    Owns the active :class:`~flim_phasors.data.PhasorData` sample (and optional
    multi-image list), the shared/per-sample :class:`~flim_phasors.calibration.ReferenceCalibration`,
    and the matplotlib canvases for the phasor plot and intensity/lifetime image.
    Users load PTU/TIFF/LIF files, apply frequency/harmonic/filter/threshold
    settings, place cursors or fit a GMM in phasor space, paint segmentation on
    the image, and export maps/tables/sessions via :meth:`export_all`.

    Inherits menu/session/theme helpers from
    :class:`~flim_phasors.gui.enhancements.EnhancementsMixin`. Constructed once
    by :func:`flim_phasors.app.main`.
    """

    def __init__(self):
        """Construct the main window: session state, timers, UI, and enhancements.

        Sets up the single active :class:`PhasorData` (``self.data``) plus the
        multi-image dataset list, shared-reference bookkeeping, and the current
        segmentation mode ("cursor" or "gmm"). Creates three ``QTimer`` instances
        that debounce expensive redraws while the user drags cursors, scrolls the
        radius slider, or edits per-sample processing settings, then delegates to
        :meth:`_build_ui` to lay out widgets and to
        :meth:`~flim_phasors.gui.enhancements.EnhancementsMixin._init_enhancements`
        to wire drag-and-drop, recents, autosave, and other mixin features. Called
        exactly once, when the window is created.
        """
        super().__init__()
        self.setWindowTitle(f"FLIM Phasor Analyzer v{__version__}")
        self.resize(1500, 980)
        self._settings = QtCore.QSettings("FLIMPhasors", "FLIMPhasorAnalyzer")
        self.data = PhasorData()
        self.datasets = []        # multi-image mode: list of PhasorData
        self._loading_proc_ui = False
        self.active_idx = -1
        self.shared_ref_path = ""
        self.shared_ref_n_channels = 1
        pref_rch = int(self._settings.value("preferred_ref_channel", 0))
        self.shared_ref_channel = min(max(0, pref_rch), CHANNEL_PRESELECT_MAX)
        self.ref_calibration = ReferenceCalibration()
        self.last_overlay = None
        self.cluster_stats = []
        self.mode = "cursor"
        # Live timer (~40 ms): throttled overlay-only redraw while dragging/resizing.
        # Debounce timer (~300 ms): full cluster stats after scroll-wheel or slider settle.
        self._cursor_debounce_timer = QtCore.QTimer(self)
        self._cursor_debounce_timer.setSingleShot(True)
        self._cursor_debounce_timer.setInterval(300)
        self._cursor_debounce_timer.timeout.connect(self._deferred_cursor_compute)
        self._cursor_live_timer = QtCore.QTimer(self)
        self._cursor_live_timer.setSingleShot(True)
        self._cursor_live_timer.setInterval(40)
        self._cursor_live_timer.timeout.connect(self._deferred_cursor_live_overlay)
        self._proc_debounce_timer = QtCore.QTimer(self)
        self._proc_debounce_timer.setSingleShot(True)
        self._proc_debounce_timer.setInterval(250)
        self._proc_debounce_timer.timeout.connect(self._deferred_refresh_compare_list)
        self._filling_table = False
        self._build_ui()
        self._init_enhancements()

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self):
        """Build every widget in the main window and wire their signals.

        Assembles the left-hand plot/table area (phasor canvas, image canvas,
        and results table inside splitters) and the right-hand tabbed control
        panel (Setup, Multi-phasor, Analyze), then connects each control's
        signal to its handler (loading, processing, multi-image, cursor/GMM
        segmentation, and export actions). This is a one-time layout pass
        called from :meth:`__init__`; nothing here depends on a loaded sample,
        so most widgets start disabled or hidden until data is present.
        """
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)

        _small = "font-size: 10px;"
        _lbl_file = f"color: gray; {_small}"

        def _tab_page():
            """Create a scrollable tab page with a top-aligned vertical layout.

            Used to build each of the Setup / Multi-phasor / Analyze tabs so
            their contents scroll independently instead of forcing the whole
            control panel to grow when a group box is expanded.

            Returns:
                Tuple ``(scroll, layout)``: the ``QScrollArea`` to add as a tab
                page, and the ``QVBoxLayout`` of its inner widget that callers
                populate with group boxes.
            """
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            inner = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(inner)
            lay.setAlignment(Qt.AlignmentFlag.AlignTop)
            lay.setSpacing(5)
            lay.setContentsMargins(4, 4, 4, 4)
            scroll.setWidget(inner)
            return scroll, lay

        setup_scroll, setup_l = _tab_page()
        compare_scroll, compare_l = _tab_page()
        analyze_scroll, analyze_l = _tab_page()

        self.lbl_panel_status = QtWidgets.QLabel("No sample loaded")
        self.lbl_panel_status.setStyleSheet(_lbl_file)
        self.lbl_panel_status.setWordWrap(True)

        # ---- files (sample | reference) + log ----
        gb_io = QtWidgets.QGroupBox("Files")
        io_main = QtWidgets.QVBoxLayout(gb_io)
        io_main.setSpacing(4)
        io_row = QtWidgets.QHBoxLayout()
        io_row.setSpacing(6)

        sample_col = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(sample_col); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(2)
        row_s = QtWidgets.QHBoxLayout(); row_s.setSpacing(4)
        btn_sample = QtWidgets.QPushButton("Sample…")
        btn_sample.setMinimumWidth(72)
        btn_sample.setToolTip(
            "Load one or more FLIM files (.ptu, .tif). "
            "In the file dialog, Ctrl+click or Shift+click to select multiple.")
        btn_sample.clicked.connect(self.choose_sample)
        self.cb_channel = QtWidgets.QComboBox()
        self.cb_channel.addItems([str(i) for i in range(CHANNEL_PRESELECT_MAX + 1)])
        self.cb_channel.setMinimumWidth(48)
        pref_ch = 0
        if hasattr(self, "_settings"):
            pref_ch = int(self._settings.value("preferred_sample_channel", 0))
        self.cb_channel.setCurrentIndex(min(max(0, pref_ch), CHANNEL_PRESELECT_MAX))
        self.cb_channel.setToolTip(
            "Emission channel to use. Set this before Sample… "
            "(especially with Fast load). After load, the list is limited to "
            "channels present in the file.")
        self.cb_channel.currentIndexChanged.connect(self.on_channel_change)
        row_s.addWidget(btn_sample, 1)
        ch_lbl = QtWidgets.QLabel("Ch")
        ch_lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        row_s.addWidget(ch_lbl)
        row_s.addWidget(self.cb_channel)
        sl.addLayout(row_s)
        self.chk_fast_load = QtWidgets.QCheckBox("Fast load (single channel)")
        self.chk_fast_load.setToolTip(
            "Decode only the Ch selected above (big RAM win for multi-channel PTU).\n"
            "Pick Ch first, then Sample…. TIFF still reads the whole file, then keeps one channel.\n"
            "Switching channel after load re-decodes.")
        self.chk_fast_load.setChecked(
            self._settings.value("fast_load", False, type=bool)
            if hasattr(self, "_settings") else False)
        self.chk_fast_load.toggled.connect(self._on_fast_load_toggled)
        sl.addWidget(self.chk_fast_load)
        self.lbl_sample = QtWidgets.QLabel("(no sample)")
        self.lbl_sample.setStyleSheet(_lbl_file)
        self.lbl_sample.setWordWrap(True)
        sl.addWidget(self.lbl_sample)

        ref_col = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(ref_col); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(2)
        self.chk_shared_ref = QtWidgets.QCheckBox("Shared ref")
        self.chk_shared_ref.setChecked(True)
        self.chk_shared_ref.setToolTip(
            "One reference measurement calibrates all samples (only phasor maps are kept in memory).")
        self.chk_shared_ref.toggled.connect(self.on_shared_ref_toggle)
        rl.addWidget(self.chk_shared_ref)
        row_r = QtWidgets.QHBoxLayout(); row_r.setSpacing(4)
        btn_ref = QtWidgets.QPushButton("Reference…")
        btn_ref.setMinimumWidth(72)
        btn_ref.clicked.connect(self.choose_ref)
        self.cb_ref_channel = QtWidgets.QComboBox()
        self.cb_ref_channel.addItems([str(i) for i in range(CHANNEL_PRESELECT_MAX + 1)])
        self.cb_ref_channel.setMinimumWidth(48)
        pref_rch = 0
        if hasattr(self, "_settings"):
            pref_rch = int(self._settings.value("preferred_ref_channel", 0))
        self.cb_ref_channel.setCurrentIndex(min(max(0, pref_rch), CHANNEL_PRESELECT_MAX))
        self.shared_ref_channel = self.cb_ref_channel.currentIndex()
        self.cb_ref_channel.setEnabled(True)
        self.cb_ref_channel.setToolTip(
            "Reference emission channel. Set this before Reference… — "
            "auto-calibrate uses the selected channel.")
        self.cb_ref_channel.currentIndexChanged.connect(self.on_ref_channel_change)
        row_r.addWidget(btn_ref, 1)
        ref_ch_lbl = QtWidgets.QLabel("Ch")
        ref_ch_lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        row_r.addWidget(ref_ch_lbl)
        row_r.addWidget(self.cb_ref_channel)
        rl.addLayout(row_r)
        self.lbl_ref = QtWidgets.QLabel("(none)")
        self.lbl_ref.setStyleSheet(_lbl_file)
        self.lbl_ref.setWordWrap(True)
        rl.addWidget(self.lbl_ref)

        io_row.addWidget(sample_col, 1)
        io_row.addWidget(ref_col, 1)
        io_main.addLayout(io_row)
        self.btn_calibrate = QtWidgets.QPushButton("Recalibrate")
        self.btn_calibrate.setToolTip(
            "Optional: reference g/s are computed automatically when you pick a "
            "reference and when you Apply. Use this to force a re-decode / refresh "
            "the reference preview without processing samples.")
        self.btn_calibrate.clicked.connect(self.calibrate_reference)
        row_cal_io = QtWidgets.QHBoxLayout()
        row_cal_io.addWidget(self.btn_calibrate)
        row_cal_io.addStretch(1)
        io_main.addLayout(row_cal_io)
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setObjectName("activity_log")
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(400)
        self.txt_log.setMinimumHeight(72)
        self.txt_log.setMaximumHeight(100)
        self.txt_log.setPlaceholderText("Activity log…")
        io_main.addWidget(self.txt_log)
        setup_l.addWidget(gb_io)

        # ---- samples & processing ----
        self.gb_proc = QtWidgets.QGroupBox("Processing")
        proc_vl = QtWidgets.QVBoxLayout(self.gb_proc)
        self._proc_active_row = QtWidgets.QWidget()
        proc_active_l = QtWidgets.QHBoxLayout(self._proc_active_row)
        proc_active_l.setContentsMargins(0, 0, 0, 0)
        proc_active_l.addWidget(QtWidgets.QLabel("Active sample"))
        self.cb_sample = QtWidgets.QComboBox()
        self.cb_sample.setToolTip(
            "Sample shown in the plots and used for Apply below.")
        self.cb_sample.currentIndexChanged.connect(self._on_sample_combo_change)
        proc_active_l.addWidget(self.cb_sample, 1)
        self.lbl_proc_active = QtWidgets.QLabel("")
        self.lbl_proc_active.setStyleSheet(_lbl_file)
        self.lbl_proc_active.setWordWrap(True)
        proc_active_l.addWidget(self.lbl_proc_active, 1)
        self._proc_active_row.setVisible(False)
        proc_vl.addWidget(self._proc_active_row)
        self.proc_inner = QtWidgets.QWidget()
        prg = QtWidgets.QGridLayout(self.proc_inner)
        self.proc_grid = prg
        prg.setHorizontalSpacing(6)
        prg.setVerticalSpacing(3)
        self.sp_harm = QtWidgets.QSpinBox(); self.sp_harm.setRange(1, 8); self.sp_harm.setValue(1)
        self.sp_freq = QtWidgets.QDoubleSpinBox(); self.sp_freq.setRange(1, 1000)
        self.sp_freq.setDecimals(3); self.sp_freq.setValue(80.0); self.sp_freq.setSuffix(" MHz")
        self.sp_reflt = QtWidgets.QDoubleSpinBox(); self.sp_reflt.setRange(0.0, 100.0)
        self.sp_reflt.setDecimals(3); self.sp_reflt.setValue(4.0); self.sp_reflt.setSuffix(" ns")
        self.sp_reflt.setToolTip(
            "Known lifetime of the reference dye (ns), e.g. ~4 for fluorescein. "
            "Wrong value shifts the phasor cloud vertically on the plot.")
        self.cb_filter = QtWidgets.QComboBox()
        self.cb_filter.addItems(list(FILTER_MODES))
        self.cb_filter.setCurrentText("median")
        self.cb_filter.currentTextChanged.connect(self.on_filter_change)
        self.chk_detect_harm = QtWidgets.QCheckBox("Harmonic mask")
        self.chk_detect_harm.setChecked(True)
        self.chk_detect_harm.setToolTip(
            "With Min N > 0, also mask harmonic-overtone pixels (phasorpy phasor_threshold).")
        self.sp_msize = QtWidgets.QSpinBox(); self.sp_msize.setRange(3, 11); self.sp_msize.setSingleStep(2); self.sp_msize.setValue(3)
        self.sp_mrep = QtWidgets.QSpinBox(); self.sp_mrep.setRange(1, 10); self.sp_mrep.setValue(1)
        self.sp_psigma = QtWidgets.QDoubleSpinBox(); self.sp_psigma.setRange(0.5, 6.0)
        self.sp_psigma.setSingleStep(0.5); self.sp_psigma.setValue(2.0)
        self.sp_plevels = QtWidgets.QSpinBox(); self.sp_plevels.setRange(1, 6); self.sp_plevels.setValue(1)
        self.sp_thr = QtWidgets.QSpinBox()
        self.sp_thr.setRange(0, 2_000_000_000)
        self.sp_thr.setSingleStep(100)
        self.sp_thr.setValue(0)
        self.sp_thr.setToolTip(
            "Minimum total photon count per pixel (sum of the TCSPC histogram). "
            "Pixels below this count are excluded from the phasor plot and segmentation only; "
            "the intensity image always shows all photons. 0 = off.")
        self.lbl_photon_range = QtWidgets.QLabel("(apply for photon range)")
        self.lbl_photon_range.setStyleSheet(_lbl_file)

        prg.addWidget(QtWidgets.QLabel("Harm."), 0, 0)
        prg.addWidget(self.sp_harm, 0, 1)
        prg.addWidget(QtWidgets.QLabel("Laser"), 0, 2)
        prg.addWidget(self.sp_freq, 0, 3)
        prg.addWidget(QtWidgets.QLabel("Ref τ"), 1, 0)
        prg.addWidget(self.sp_reflt, 1, 1)
        prg.addWidget(QtWidgets.QLabel("Filter"), 1, 2)
        prg.addWidget(self.cb_filter, 1, 3)
        self.lbl_msize = QtWidgets.QLabel("Kernel")
        prg.addWidget(self.lbl_msize, 2, 0)
        prg.addWidget(self.sp_msize, 2, 1)
        self.lbl_mrep = QtWidgets.QLabel("Repeat")
        prg.addWidget(self.lbl_mrep, 2, 2)
        prg.addWidget(self.sp_mrep, 2, 3)
        self.lbl_psigma = QtWidgets.QLabel("paw σ")
        prg.addWidget(self.lbl_psigma, 3, 0)
        prg.addWidget(self.sp_psigma, 3, 1)
        self.lbl_plevels = QtWidgets.QLabel("paw lvl")
        prg.addWidget(self.lbl_plevels, 3, 2)
        prg.addWidget(self.sp_plevels, 3, 3)
        self.row_msize = (self.lbl_msize, self.sp_msize, self.lbl_mrep, self.sp_mrep)
        self.row_psigma = (self.lbl_psigma, self.sp_psigma, self.lbl_plevels, self.sp_plevels)
        prg.addWidget(QtWidgets.QLabel("Min N"), 4, 0)
        prg.addWidget(self.sp_thr, 4, 1)
        prg.addWidget(self.chk_detect_harm, 4, 2, 1, 2)
        prg.addWidget(QtWidgets.QLabel("Frame"), 5, 0)
        self.sp_frame = QtWidgets.QSpinBox()
        self.sp_frame.setRange(-1, 0)
        self.sp_frame.setValue(-1)
        self.sp_frame.setSpecialValueText("all")
        self.sp_frame.setToolTip(
            "Time index for stacks with a T dimension (-1 = sum all frames). Reloads the active sample.")
        self.sp_frame.valueChanged.connect(self.on_frame_change)
        prg.addWidget(self.sp_frame, 5, 1)
        for sp in (
            self.sp_harm, self.sp_msize, self.sp_mrep, self.sp_plevels,
            self.sp_thr, self.sp_frame,
        ):
            self._spin_for_typing(sp, min_width=76)
        for sp in (self.sp_freq, self.sp_reflt, self.sp_psigma):
            self._spin_for_typing(sp, min_width=88)
        self.sp_thr.setMinimumWidth(96)
        prg.addWidget(self.lbl_photon_range, 6, 0, 1, 4)
        prg.addWidget(QtWidgets.QLabel("Ref g"), 7, 0)
        self.edit_ref_g = QtWidgets.QLineEdit("0")
        self.edit_ref_g.setValidator(QtGui.QDoubleValidator(-0.05, 1.05, 5, self))
        self.edit_ref_g.setFixedWidth(56)
        prg.addWidget(self.edit_ref_g, 7, 1)
        prg.addWidget(QtWidgets.QLabel("s"), 7, 2)
        self.edit_ref_s = QtWidgets.QLineEdit("0")
        self.edit_ref_s.setValidator(QtGui.QDoubleValidator(-0.05, 1.05, 5, self))
        self.edit_ref_s.setFixedWidth(56)
        prg.addWidget(self.edit_ref_s, 7, 3)
        self.chk_manual_cal = QtWidgets.QCheckBox("Manual ref phasor")
        self.chk_manual_cal.setToolTip(
            "Type g/s above, click Set g/s, then Apply on samples. "
            "Shows a small ref preview (measured g/s vs Ref τ target).")
        self.chk_manual_cal.toggled.connect(self._on_manual_cal_toggled)
        prg.addWidget(self.chk_manual_cal, 8, 0, 1, 2)
        self.btn_set_manual_gs = QtWidgets.QPushButton("Set g/s")
        self.btn_set_manual_gs.setFixedWidth(56)
        self.btn_set_manual_gs.setToolTip(
            "Apply the manual g and s values to calibration (updates ref preview). "
            "Then click Apply to preprocess samples.")
        self.btn_set_manual_gs.clicked.connect(self.apply_manual_gs)
        self.btn_set_manual_gs.setEnabled(False)
        prg.addWidget(self.btn_set_manual_gs, 8, 2)
        btn_clear_cal = QtWidgets.QPushButton("Clear cal")
        btn_clear_cal.setFixedWidth(64)
        btn_clear_cal.clicked.connect(self._clear_calibration)
        prg.addWidget(btn_clear_cal, 8, 3)
        self.lbl_cal_display = QtWidgets.QLabel("(uncalibrated — load a reference file)")
        self.lbl_cal_display.setStyleSheet(_lbl_file)
        self.lbl_cal_display.setWordWrap(True)
        prg.addWidget(self.lbl_cal_display, 9, 0, 1, 4)
        self.edit_ref_g.setEnabled(False)
        self.edit_ref_s.setEnabled(False)
        row_cal_apply = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(lambda: self.apply_processing(scope="active"))
        self.btn_apply_all = QtWidgets.QPushButton("Apply all")
        self.btn_apply_all.setToolTip(
            "Preprocess every loaded sample using the filter and threshold "
            "settings currently shown on the Setup tab.")
        self.btn_apply_all.clicked.connect(lambda: self.apply_processing(scope="all"))
        self.btn_apply_all.setVisible(False)
        row_cal_apply.addWidget(self.btn_apply, 1)
        row_cal_apply.addWidget(self.btn_apply_all, 1)
        prg.addLayout(row_cal_apply, 10, 0, 1, 4)
        self.sp_harm.valueChanged.connect(self._on_harm_or_ref_setting_changed)
        self.cb_filter.currentTextChanged.connect(self._on_harm_or_ref_setting_changed)
        self.sp_reflt.valueChanged.connect(self._on_ref_lifetime_or_freq_changed)
        self.sp_freq.valueChanged.connect(self._on_ref_lifetime_or_freq_changed)
        proc_vl.addWidget(self.proc_inner)
        setup_l.addWidget(self.gb_proc)
        setup_l.addStretch(1)
        self.on_filter_change("median")
        self._connect_per_sample_proc_autosave()
        self._table_sel_lock = False

        # ---- multi-image (sample list + overlay) ----
        gb_multi = QtWidgets.QGroupBox("Multi-image")
        mbl = QtWidgets.QVBoxLayout(gb_multi)
        mbl.setSpacing(3)
        self.chk_multi = QtWidgets.QCheckBox("Multi-image mode")
        self.chk_multi.setToolTip(
            "Load and compare several samples. Use the table below to switch images.")
        self.chk_multi.toggled.connect(self.on_multi_toggle)
        mbl.addWidget(self.chk_multi)
        self._multi_strip = QtWidgets.QWidget()
        multi_l = QtWidgets.QVBoxLayout(self._multi_strip)
        multi_l.setContentsMargins(0, 0, 0, 0)
        row_tbl = QtWidgets.QHBoxLayout()
        self.table_compare = QtWidgets.QTableWidget(0, 8)
        self.table_compare.setHorizontalHeaderLabels(
            ["Show", "#", "Sample", "Group", "Filter", "Min N", "Ref", "Status"])
        self.table_compare.verticalHeader().setVisible(False)
        self.table_compare.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_compare.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table_compare.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table_compare.setMinimumHeight(120)
        self.table_compare.setMaximumHeight(220)
        self.table_compare.setToolTip(
            "Click a row to activate that sample. Double-click Sample or Group to rename. "
            "Tick Show for phasor overlay.")
        hdr = self.table_compare.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_compare.cellChanged.connect(self._on_compare_table_changed)
        self.table_compare.itemSelectionChanged.connect(self._on_sample_table_selection)
        self.btn_rmimg = QtWidgets.QPushButton("−")
        self.btn_rmimg.setFixedWidth(28)
        self.btn_rmimg.setToolTip("Remove selected sample")
        self.btn_rmimg.clicked.connect(self.remove_image)
        row_tbl.addWidget(self.table_compare, 1)
        row_tbl.addWidget(self.btn_rmimg)
        multi_l.addLayout(row_tbl)
        self.lbl_editing = QtWidgets.QLabel("Active: —")
        self.lbl_editing.setStyleSheet(_lbl_file)
        self.lbl_editing.setWordWrap(True)
        multi_l.addWidget(self.lbl_editing)
        row_apply_all = QtWidgets.QHBoxLayout()
        self.btn_apply_settings_all = QtWidgets.QPushButton("Apply settings to all")
        self.btn_apply_settings_all.setToolTip(
            "Copy the current filter settings from the Setup tab to every sample, "
            "then preprocess all.")
        self.btn_apply_settings_all.clicked.connect(self.apply_settings_to_all)
        row_apply_all.addWidget(self.btn_apply_settings_all, 1)
        multi_l.addLayout(row_apply_all)
        self._multi_strip.setVisible(False)
        mbl.addWidget(self._multi_strip)
        row_name = QtWidgets.QHBoxLayout()
        row_name.addWidget(QtWidgets.QLabel("Name"))
        self.edit_display_name = QtWidgets.QLineEdit()
        self.edit_display_name.setPlaceholderText("Display name on phasor legend")
        self.edit_display_name.setToolTip(
            "Rename the active sample for the phasor legend and sample table "
            "(does not change the file on disk).")
        self.edit_display_name.editingFinished.connect(self._apply_display_name_from_field)
        btn_name = QtWidgets.QPushButton("Set")
        btn_name.setFixedWidth(40)
        btn_name.clicked.connect(self._apply_display_name_from_field)
        row_name.addWidget(self.edit_display_name, 1)
        row_name.addWidget(btn_name)
        mbl.addLayout(row_name)
        row_grp = QtWidgets.QHBoxLayout()
        row_grp.addWidget(QtWidgets.QLabel("Group"))
        self.edit_group = QtWidgets.QLineEdit()
        self.edit_group.setPlaceholderText("e.g. condition A")
        self.edit_group.setToolTip(
            "Group label for the active sample (overlay legend and table).")
        self.edit_group.editingFinished.connect(self._apply_group_from_field)
        btn_grp = QtWidgets.QPushButton("Set")
        btn_grp.setFixedWidth(40)
        btn_grp.clicked.connect(self._apply_group_from_field)
        row_grp.addWidget(self.edit_group, 1)
        row_grp.addWidget(btn_grp)
        mbl.addLayout(row_grp)
        self.chk_compare = QtWidgets.QCheckBox("Multi-image phasor view")
        self.chk_compare.setToolTip(
            "Overlay preprocessed samples on the phasor plot (enable manually when needed).")
        self.chk_compare.toggled.connect(self._on_compare_ui_changed)
        mbl.addWidget(self.chk_compare)
        row_cmp = QtWidgets.QHBoxLayout()
        self.cb_compare_style = QtWidgets.QComboBox()
        self.cb_compare_style.addItems(list(COMPARE_STYLE_MAP.keys()))
        self.cb_compare_style.currentIndexChanged.connect(self._on_compare_ui_changed)
        btn_cmp_all = QtWidgets.QPushButton("All")
        btn_cmp_all.setFixedWidth(32)
        btn_cmp_all.setToolTip("Show every preprocessed sample on the phasor plot.")
        btn_cmp_all.clicked.connect(self._compare_select_all)
        btn_cmp_none = QtWidgets.QPushButton("None")
        btn_cmp_none.setFixedWidth(36)
        btn_cmp_none.clicked.connect(self._compare_select_none)
        row_cmp.addWidget(self.cb_compare_style, 1)
        row_cmp.addWidget(btn_cmp_all)
        row_cmp.addWidget(btn_cmp_none)
        mbl.addLayout(row_cmp)
        row_grp_filt = QtWidgets.QHBoxLayout()
        row_grp_filt.addWidget(QtWidgets.QLabel("Overlay group"))
        self.cb_compare_group = QtWidgets.QComboBox()
        self.cb_compare_group.addItem("All groups")
        self.cb_compare_group.setToolTip("Limit phasor overlay to one group name.")
        self.cb_compare_group.currentIndexChanged.connect(self._on_compare_ui_changed)
        row_grp_filt.addWidget(self.cb_compare_group, 1)
        mbl.addLayout(row_grp_filt)
        row_legend = QtWidgets.QHBoxLayout()
        row_legend.addWidget(QtWidgets.QLabel("Legend"))
        self.cb_legend_format = QtWidgets.QComboBox()
        self.cb_legend_format.addItems(list(LEGEND_FORMAT_ITEMS))
        self.cb_legend_format.setToolTip("How sample names appear in the phasor legend.")
        self.cb_legend_format.currentIndexChanged.connect(self._on_legend_ui_changed)
        self.cb_legend_loc = QtWidgets.QComboBox()
        self.cb_legend_loc.addItems(list(LEGEND_LOC_ITEMS))
        self.cb_legend_loc.setToolTip("Legend position on the phasor plot.")
        self.cb_legend_loc.currentIndexChanged.connect(self._on_legend_ui_changed)
        row_legend.addWidget(self.cb_legend_format, 1)
        row_legend.addWidget(self.cb_legend_loc, 1)
        row_legend.addWidget(QtWidgets.QLabel("Size"))
        self.sp_legend_size = QtWidgets.QSpinBox()
        self.sp_legend_size.setRange(LEGEND_SIZE_MIN, LEGEND_SIZE_MAX)
        self.sp_legend_size.setValue(LEGEND_SIZE_DEFAULT)
        self.sp_legend_size.setToolTip(
            "Legend text and colour-marker size on the phasor plot.")
        self.sp_legend_size.valueChanged.connect(self._on_legend_ui_changed)
        row_legend.addWidget(self.sp_legend_size)
        mbl.addLayout(row_legend)
        self._compare_sel_buttons = (btn_cmp_all, btn_cmp_none)
        self._sample_label_widgets = (
            self.edit_display_name, btn_name, self.edit_group, btn_grp,
        )
        self._multi_detail_widgets = (
            self._proc_active_row, self.cb_sample,
            self._multi_strip, self.table_compare, self.btn_rmimg,
            self.lbl_editing, self.btn_apply_settings_all,
            self.chk_compare, self.cb_compare_style, self.cb_compare_group,
            self.cb_legend_format, self.cb_legend_loc, self.sp_legend_size,
            btn_cmp_all, btn_cmp_none,
        )
        self._set_multi_detail_enabled(False)
        self._set_compare_controls_enabled(False)
        compare_l.addWidget(gb_multi)
        compare_l.addStretch(1)
        self.gb_multi = gb_multi
        self._update_apply_buttons()

        # ---- mode ----
        gb_mode = QtWidgets.QGroupBox("Segmentation")
        ml = QtWidgets.QVBoxLayout(gb_mode)
        ml.setSpacing(3)
        mode_row = QtWidgets.QHBoxLayout()
        self.rb_cursor = QtWidgets.QRadioButton("Cursors")
        self.rb_gmm = QtWidgets.QRadioButton("GMM")
        self.rb_cursor.setChecked(True); self.rb_cursor.toggled.connect(self.on_mode_change)
        mode_row.addWidget(self.rb_cursor); mode_row.addWidget(self.rb_gmm)
        ml.addLayout(mode_row)

        self.cursor_box = QtWidgets.QWidget()
        cbl = QtWidgets.QGridLayout(self.cursor_box); cbl.setContentsMargins(0, 0, 0, 0)
        b_add = QtWidgets.QPushButton("+"); b_add.setFixedWidth(28); b_add.clicked.connect(self.add_cursor)
        b_del = QtWidgets.QPushButton("Del"); b_del.clicked.connect(self.remove_cursor)
        b_clr = QtWidgets.QPushButton("Clear"); b_clr.clicked.connect(self.clear_cursors)
        cbl.addWidget(b_add, 0, 0); cbl.addWidget(b_del, 0, 1); cbl.addWidget(b_clr, 0, 2)
        row_cur = QtWidgets.QHBoxLayout()
        row_cur.addWidget(QtWidgets.QLabel("Move"))
        self.cb_active_cursor = QtWidgets.QComboBox()
        self.cb_active_cursor.setToolTip(
            "Choose which circle to move or resize. Drag anywhere on the phasor plot.")
        self.cb_active_cursor.currentIndexChanged.connect(self.on_active_cursor_change)
        row_cur.addWidget(self.cb_active_cursor, 1)
        cbl.addLayout(row_cur, 1, 0, 1, 3)
        row_shape = QtWidgets.QHBoxLayout()
        row_shape.addWidget(QtWidgets.QLabel("Shape"))
        self.cb_cursor_shape = QtWidgets.QComboBox()
        self.cb_cursor_shape.addItems(list(CURSOR_SHAPES))
        self.cb_cursor_shape.setToolTip("Circle or ellipse ROI on the phasor plot (phasorpy cursors).")
        row_shape.addWidget(self.cb_cursor_shape, 1)
        row_shape.addWidget(QtWidgets.QLabel("Aspect"))
        self.sld_aspect = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_aspect.setRange(30, 100)
        self.sld_aspect.setValue(65)
        self.sld_aspect.setToolTip("Ellipse minor/major axis ratio (%).")
        self.lbl_aspect = QtWidgets.QLabel("65%")
        self.sld_aspect.valueChanged.connect(
            lambda v: self.lbl_aspect.setText(f"{v}%"))
        row_shape.addWidget(self.sld_aspect, 1)
        row_shape.addWidget(self.lbl_aspect)
        cbl.addLayout(row_shape, 2, 0, 1, 3)
        self.sld_radius = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_radius.setRange(5, 400)
        self.sld_radius.setValue(50)
        self.sld_radius.valueChanged.connect(self.on_radius_slider)
        self.sld_radius.sliderReleased.connect(self._on_radius_slider_released)
        self.sp_radius = QtWidgets.QDoubleSpinBox()
        self.sp_radius.setRange(0.005, 0.400)
        self.sp_radius.setDecimals(3)
        self.sp_radius.setSingleStep(0.005)
        self.sp_radius.setValue(0.050)
        self.sp_radius.setToolTip("Exact circle radius on the phasor plot (g, s units).")
        self._spin_for_typing(self.sp_radius, min_width=56)
        self.sp_radius.valueChanged.connect(self.on_radius_spin)
        self.sp_radius.editingFinished.connect(self._on_radius_spin_committed)
        radius_entry = QtWidgets.QHBoxLayout()
        radius_entry.setContentsMargins(0, 0, 0, 0)
        radius_entry.addWidget(QtWidgets.QLabel("r"))
        radius_entry.addWidget(self.sp_radius)
        radius_entry_w = QtWidgets.QWidget()
        radius_entry_w.setLayout(radius_entry)
        cbl.addWidget(self.sld_radius, 3, 0, 1, 2)
        cbl.addWidget(radius_entry_w, 3, 2)
        ml.addWidget(self.cursor_box)

        self.gmm_box = QtWidgets.QWidget()
        gbl = QtWidgets.QGridLayout(self.gmm_box); gbl.setContentsMargins(0, 0, 0, 0)
        gbl.addWidget(QtWidgets.QLabel("k"), 0, 0)
        self.edit_ncomp = QtWidgets.QLineEdit("3")
        self.edit_ncomp.setValidator(QtGui.QIntValidator(1, 12, self))
        self.edit_ncomp.setFixedWidth(32)
        self.edit_ncomp.setToolTip("Number of GMM components (1–12), or max k when BIC auto-k is on.")
        gbl.addWidget(self.edit_ncomp, 0, 1)
        gbl.addWidget(QtWidgets.QLabel("Cov"), 0, 2)
        self.cb_cov = QtWidgets.QComboBox(); self.cb_cov.addItems(["full", "tied", "diag", "spherical"])
        gbl.addWidget(self.cb_cov, 0, 3)
        self.chk_bic = QtWidgets.QCheckBox("BIC auto-k")
        self.chk_bic.setToolTip("Search k = 1 … max k using BIC on valid phasor pixels.")
        self.chk_bic.toggled.connect(self._on_bic_toggled)
        gbl.addWidget(self.chk_bic, 1, 0, 1, 2)
        gbl.addWidget(QtWidgets.QLabel("σ"), 2, 0)
        self.edit_gmm_sigma = QtWidgets.QLineEdit("2.0")
        self.edit_gmm_sigma.setValidator(QtGui.QDoubleValidator(0.5, 6.0, 2, self))
        self.edit_gmm_sigma.setFixedWidth(40)
        self.edit_gmm_sigma.setToolTip(
            "Ellipse scale for phasorpy phasor_cluster_gmm (95% contour at 2.0).")
        gbl.addWidget(self.edit_gmm_sigma, 2, 1)
        b_fit = QtWidgets.QPushButton("Fit GMM"); b_fit.clicked.connect(self.fit_gmm)
        b_fit.setToolTip("Fit GMM on the active sample and paint (remembered per image).")
        b_fit_all = QtWidgets.QPushButton("Fit all")
        b_fit_all.setToolTip(
            "Fit GMM on every loaded sample with the current k/σ, keep results per image, "
            "then Export all to save them together.")
        b_fit_all.clicked.connect(self.fit_gmm_all)
        b_clr_gmm = QtWidgets.QPushButton("Clear"); b_clr_gmm.clicked.connect(self.clear_gmm)
        gbl.addWidget(b_fit, 2, 2)
        gbl.addWidget(b_fit_all, 3, 2)
        gbl.addWidget(b_clr_gmm, 2, 3)
        self.gmm_box.setVisible(False); ml.addWidget(self.gmm_box)
        analyze_l.addWidget(gb_mode)

        # ---- actions ----
        self.gb_act = QtWidgets.QGroupBox("Results")
        al = QtWidgets.QVBoxLayout(self.gb_act)
        al.setSpacing(3)
        row_paint = QtWidgets.QHBoxLayout()
        self.btn_paint = QtWidgets.QPushButton("Paint")
        self.btn_paint.clicked.connect(self.compute_and_paint)
        self.chk_live = QtWidgets.QCheckBox("Live"); self.chk_live.setChecked(True)
        self.chk_overlay = QtWidgets.QCheckBox("Overlay"); self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(self.refresh_image)
        row_paint.addWidget(self.btn_paint, 1)
        row_paint.addWidget(self.chk_live)
        row_paint.addWidget(self.chk_overlay)
        al.addLayout(row_paint)
        self.btn_export = QtWidgets.QPushButton("Export all…")
        self.btn_export.setToolTip(
            "Save one folder for the whole session: each sample's maps, remembered GMM/paint "
            "results, tables, analysis.xlsx, and session.json.")
        self.btn_export.clicked.connect(self.export_all)
        al.addWidget(self.btn_export)
        analyze_l.addWidget(self.gb_act)
        analyze_l.addStretch(1)

        self.panel_tabs = QtWidgets.QTabWidget()
        self.panel_tabs.setDocumentMode(True)
        self._tab_setup_idx = self.panel_tabs.addTab(setup_scroll, "Setup")
        self._tab_compare_idx = self.panel_tabs.addTab(compare_scroll, "Multi-phasor")
        self._tab_analyze_idx = self.panel_tabs.addTab(analyze_scroll, "Analyze")
        self.panel_tabs.setTabToolTip(
            self._tab_setup_idx,
            "Load samples, reference, calibrate, and preprocess.")
        self.panel_tabs.setTabToolTip(
            self._tab_compare_idx,
            "Multi-image table, groups, and phasor overlay.")
        self.panel_tabs.setTabToolTip(
            self._tab_analyze_idx,
            "Cursors or GMM segmentation, paint, and export.")

        panel_wrap = QtWidgets.QWidget()
        panel_wrap.setFixedWidth(420)
        pwl = QtWidgets.QVBoxLayout(panel_wrap)
        pwl.setContentsMargins(0, 0, 0, 0)
        pwl.setSpacing(4)
        pwl.addWidget(self.lbl_panel_status)
        pwl.addWidget(self.panel_tabs, 1)

        # ---- plots ----
        plots = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        lw = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(lw)
        self.phasor = PhasorCanvas(self)
        # cursorMoving: cheap overlay during drag/scroll; cursorChanged: commit on mouseup/add/remove.
        self.phasor.cursorChanged.connect(self.on_cursor_changed)
        self.phasor.cursorMoving.connect(self.on_cursor_moving)
        self.phasor.cursorSelectionChanged.connect(self._on_phasor_cursor_selected)
        self.phasor.mpl_connect("key_press_event", self._on_phasor_key_press)
        self.phasor.phasorClicked.connect(self._on_phasor_click)
        self.phasor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.phasor_toolbar = NavigationToolbar(self.phasor, self)
        self.phasor_toolbar.setObjectName("mpl_toolbar")
        lv.addWidget(self.phasor_toolbar)
        lv.addWidget(self.phasor)
        rw = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(rw)
        row_img_view = QtWidgets.QHBoxLayout()
        row_img_view.addWidget(QtWidgets.QLabel("View"))
        self.cb_image_view = QtWidgets.QComboBox()
        self.cb_image_view.addItems(list(IMAGE_VIEW_ITEMS))
        self.cb_image_view.setToolTip(
            "Pixel maps from Apply settings. "
            "Photons (masked) hides pixels below Min N / invalid phasors; "
            "Brightfield (all photons) shows the full intensity including filtered-out pixels. "
            "τ maps need Apply. Selecting a view turns Overlay off so the map is visible.")
        self.cb_image_view.currentIndexChanged.connect(self._on_image_view_changed)
        row_img_view.addWidget(self.cb_image_view, 1)
        rv.addLayout(row_img_view)
        self.image = ImageCanvas(self)
        self.image.imageClicked.connect(self._on_image_click)
        self.image_toolbar = NavigationToolbar(self.image, self)
        self.image_toolbar.setObjectName("mpl_toolbar")
        rv.addWidget(self.image_toolbar)
        rv.addWidget(self.image)
        plots.addWidget(lw); plots.addWidget(rw); plots.setSizes([700, 700])

        self.table = QtWidgets.QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["#", "Color", "Label (what you see)", "g", "s",
             "tau_phi (ns)", "tau_mod (ns)", "tau_normal (ns)",
             "Pixels", "Area %"])
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(220)
        self.table.itemChanged.connect(self._label_edited)

        rightside = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        rightside.addWidget(plots); rightside.addWidget(self.table); rightside.setSizes([720, 220])
        main.addWidget(rightside, 1)
        main.addWidget(panel_wrap)

        self.status = self.statusBar()
        self._log(
            "Ready — load sample(s), choose Reference, then Apply (calibration is automatic).")

    def _log(self, message, update_status=True):
        """Append a timestamped line to the Files log and optionally the status bar.

        The primary way the rest of the window reports progress and errors to
        the user. Text is appended to ``self.txt_log`` (auto-scrolled to the
        bottom, capped at 400 blocks) and, unless suppressed, also shown in the
        status bar so transient messages are visible without opening the log.

        Args:
            message: Human-readable text to record (no leading timestamp).
            update_status: When False, only the log gets the message (used for
                high-frequency progress updates that would spam the status bar).
        """
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        if hasattr(self, "txt_log"):
            self.txt_log.blockSignals(True)
            self.txt_log.appendPlainText(line)
            self.txt_log.blockSignals(False)
            sb = self.txt_log.verticalScrollBar()
            sb.setValue(sb.maximum())
        if update_status and hasattr(self, "status"):
            self.status.showMessage(message)

    def _form_row(self, form, label, widget):
        """Add a labelled row and return a (label_widget, field_widget) tuple for show/hide.

        Small helper for building ``QFormLayout`` sections whose rows need to
        be toggled together later (e.g. showing/hiding a group of options based
        on filter mode); returning both widgets lets the caller store them in a
        tuple and call ``setVisible`` on each.

        Args:
            form: ``QFormLayout`` to append the row to.
            label: Text for the row's label.
            widget: Field widget placed next to the label.

        Returns:
            ``(label_widget, widget)`` pair for later visibility toggling.
        """
        lbl = QtWidgets.QLabel(label)
        form.addRow(lbl, widget)
        return (lbl, widget)

    def _run_busy(self, message: str, fn, *, cancellable: bool = True):
        """Run heavy work off the GUI thread while showing a busy indicator.

        Thin wrapper around :func:`flim_phasors.busy.run_busy_qt` used by every
        decode/processing/export call in this window so the UI stays responsive
        and progress is echoed to the activity log. On cancellation, logs
        "Cancelled." and re-raises so the caller's own cleanup/early-return runs.

        Args:
            message: Busy-dialog and log text shown while ``fn`` runs.
            fn: Zero-argument callable executed off the GUI thread.
            cancellable: Whether the busy dialog offers a Cancel button.

        Returns:
            Whatever ``fn`` returns (typically wrapped with elapsed time by the
            underlying ``run_busy_qt`` helper).

        Raises:
            CancelledError: If the user cancels; logged before re-raising.
        """
        try:
            return run_busy_qt(
                self, message, fn,
                log_fn=lambda m: self._log(m, update_status=False),
                cancellable=cancellable,
            )
        except CancelledError:
            self._log("Cancelled.")
            raise

    @staticmethod
    def _spin_for_typing(sp, *, min_width: int = 76):
        """Hide stepper arrows so typed values are fully visible.

        Applied to the numeric processing spinboxes (harmonic, frequency,
        reference lifetime, kernel size, threshold, frame index, etc.) so users
        can type exact values without the up/down buttons truncating the field.

        Args:
            sp: ``QAbstractSpinBox`` (or subclass) to reconfigure.
            min_width: Minimum pixel width to reserve for the widened field.
        """
        sp.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        sp.setMinimumWidth(min_width)
        sp.setMaximumWidth(16777215)

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        """Format a duration for log messages (milliseconds or seconds).

        Used throughout loading/processing/export logging so short operations
        read as e.g. "320 ms" instead of "0.32 s", while longer ones stay in
        seconds with two decimal places.

        Args:
            seconds: Elapsed wall-clock time in seconds.

        Returns:
            Human-readable duration string.
        """
        if seconds < 1.0:
            return f"{seconds * 1000:.0f} ms"
        return f"{seconds:.2f} s"

    # ---- show/hide filter params ------------------------------------------
    def on_filter_change(self, mode):
        """Show kernel or pawflim controls and block unsupported LIF filter modes.

        Connected to ``cb_filter.currentTextChanged``. LIF phasor-map samples
        only carry precomputed (g, s) maps (no TCSPC histogram), so pawflim and
        the "signal" kernel filters — which operate on the raw histogram — are
        silently downgraded to "median" with a log message. Otherwise just
        toggles which of the kernel-size/repeat or pawflim sigma/levels rows are
        visible for the selected mode.

        Args:
            mode: Filter mode name from ``FILTER_MODES`` (e.g. "median",
                "gaussian", "pawflim", "signal median", "signal gaussian").
        """
        if (
            dataset_has_sample(self.data)
            and self.data.load_source == "lif_phasor"
            and mode in ("pawflim", "signal median", "signal gaussian")
        ):
            self.cb_filter.blockSignals(True)
            self.cb_filter.setCurrentText("median")
            self.cb_filter.blockSignals(False)
            mode = "median"
            self._log(
                "LIF phasor maps: use phasor median/gaussian filters (no TCSPC histogram).")
        is_kernel = mode in ("median", "gaussian", "signal median", "signal gaussian")
        is_paw = mode == "pawflim"
        for w in self.row_msize:
            w.setVisible(is_kernel)
        for w in self.row_psigma:
            w.setVisible(is_paw)

    def _connect_per_sample_proc_autosave(self):
        """Autosave processing spinboxes to the active dataset in multi-image mode.

        Connects every processing-related control (harmonic, frequency,
        reference lifetime, kernel size/repeat, pawflim sigma/levels,
        threshold, filter mode, harmonic-mask checkbox, channel) to
        :meth:`_on_per_sample_proc_changed` so that, when per-sample processing
        is enabled, edits are written back into the active dataset's stored
        ``processing_settings`` instead of only living in the shared UI state.
        Called once from :meth:`_build_ui`.
        """
        for w in (
            self.sp_harm, self.sp_freq, self.sp_reflt, self.sp_msize, self.sp_mrep,
            self.sp_psigma, self.sp_plevels, self.sp_thr,
        ):
            w.valueChanged.connect(self._on_per_sample_proc_changed)
        self.cb_filter.currentTextChanged.connect(self._on_per_sample_proc_changed)
        self.chk_detect_harm.toggled.connect(self._on_per_sample_proc_changed)
        self.cb_channel.currentIndexChanged.connect(self._on_per_sample_proc_changed)

    def _on_per_sample_proc_changed(self, *_args):
        """Persist UI processing settings and debounce compare-table refresh.

        Fired by any processing control's change signal. Ignored while
        :meth:`_load_proc_to_ui` is programmatically repopulating the UI
        (``self._loading_proc_ui``) or when per-sample processing is off, to
        avoid feedback loops and unnecessary writes. Otherwise saves the
        current UI values onto the active dataset and (re)starts
        ``_proc_debounce_timer`` so the multi-image compare table refreshes
        once editing settles, rather than on every keystroke/spin.

        Args:
            *_args: Ignored; accepts the varying signal payloads (index, text,
                value) of the connected widgets.
        """
        if self._loading_proc_ui or not per_sample_processing(self):
            return
        self._save_proc_from_ui(self.data)
        self._proc_debounce_timer.start()

    def _deferred_refresh_compare_list(self):
        """Refresh the multi-image table after per-sample settings change.

        Timeout slot for ``_proc_debounce_timer`` (started by
        :meth:`_on_per_sample_proc_changed`). Re-checks that per-sample
        processing is still enabled before rebuilding the compare table, since
        the mode may have been toggled off while the timer was pending.
        """
        if per_sample_processing(self):
            self._refresh_compare_list()

    def _on_bic_toggled(self, checked: bool):
        """Update GMM component spinbox tooltip for fixed-k vs BIC auto-k mode.

        Connected to ``chk_bic.toggled``. The ``edit_ncomp`` field means two
        different things depending on the checkbox: an exact component count
        when BIC search is off, or an upper bound (max k) to search over when
        it is on. Purely cosmetic — does not change any stored value.

        Args:
            checked: New state of the BIC auto-k checkbox.
        """
        if checked:
            self.edit_ncomp.setToolTip("Maximum k for BIC search (1–12).")
        else:
            self.edit_ncomp.setToolTip("Number of GMM components (1–12).")

    def _init_dataset_proc_settings(self, d: PhasorData):
        """Seed per-sample processing settings from the current UI if unset.

        Called right after a dataset is created/loaded so every dataset has a
        ``processing_settings`` dict to read from and write to, even before the
        user ever touches the Setup tab controls for it. Does nothing if the
        dataset already has settings (e.g. restored from a session bundle).

        Args:
            d: Dataset to initialize.
        """
        if d.processing_settings is None:
            d.processing_settings = capture_processing_from_ui(self)

    def _save_proc_from_ui(self, d: PhasorData):
        """Copy Setup-tab processing controls into a dataset's stored settings.

        No-op when per-sample processing is disabled (all samples then share
        the single set of UI values instead of individual stored settings), or
        when ``d`` is ``None`` (e.g. no active dataset yet).

        Args:
            d: Dataset whose ``processing_settings`` should be overwritten with
                the current UI state, or ``None`` to skip.
        """
        if d is None:
            return
        if per_sample_processing(self):
            d.processing_settings = capture_processing_from_ui(self)

    def _load_proc_to_ui(self, d: PhasorData):
        """Populate Setup-tab controls from a dataset's stored processing settings.

        Called when switching the active dataset (per-sample processing mode)
        so the Setup tab always reflects the sample currently being edited.
        Initializes ``d.processing_settings`` first if missing. Sets
        ``self._loading_proc_ui`` around the UI update so
        :meth:`_on_per_sample_proc_changed` does not re-save these values back
        onto the dataset as if the user had edited them.

        Args:
            d: Dataset whose stored settings should populate the controls.
        """
        if not per_sample_processing(self):
            return
        stash = getattr(d, "processing_settings", None)
        if not stash:
            self._init_dataset_proc_settings(d)
            stash = d.processing_settings
        self._loading_proc_ui = True
        try:
            apply_processing_settings_to_ui(self, stash)
        finally:
            self._loading_proc_ui = False

    def _update_multi_strip(self):
        """Show or hide multi-image table and active-sample labels.

        Called after almost every dataset-list mutation (load, remove, mode
        toggle, activation) to keep the Multi-phasor tab's compare table and
        the Setup tab's "Active sample" row visible only when multi-image mode
        is on and there is enough data to make them meaningful (table needs 2+
        samples; the active-sample row needs at least 1). Also refreshes the
        "Active: …" label text and the compact active-sample label.
        """
        multi_on = hasattr(self, "chk_multi") and self.chk_multi.isChecked()
        show_strip = multi_on and len(self.datasets) > 1
        show_active = multi_on and len(self.datasets) >= 1
        if hasattr(self, "_multi_strip"):
            self._multi_strip.setVisible(show_strip)
        if hasattr(self, "_proc_active_row"):
            self._proc_active_row.setVisible(show_active)
            if hasattr(self, "lbl_proc_active"):
                self.lbl_proc_active.setVisible(not show_active)
        if hasattr(self, "lbl_editing"):
            if show_strip and 0 <= self.active_idx < len(self.datasets):
                self.lbl_editing.setText(
                    f"Active: {dataset_display_label(self.data, self.active_idx)}")
            elif multi_on:
                self.lbl_editing.setText("Load another sample to compare.")
            else:
                self.lbl_editing.setText("")
        self._update_proc_active_label()

    def _refresh_sample_combo(self):
        """Rebuild the active-sample dropdown from loaded datasets.

        Clears and repopulates ``cb_sample`` with a display label per dataset
        (blocking its signal so this does not itself trigger an activation),
        restores the selection to ``active_idx``, then updates the multi-image
        strip visibility. Called whenever the dataset list or a display
        name/group changes.
        """
        if not hasattr(self, "cb_sample"):
            return
        cb = self.cb_sample
        cb.blockSignals(True)
        cb.clear()
        for i, d in enumerate(self.datasets):
            cb.addItem(dataset_display_label(d, i))
        if 0 <= self.active_idx < len(self.datasets):
            cb.setCurrentIndex(self.active_idx)
        cb.blockSignals(False)
        self._update_multi_strip()

    def _on_sample_combo_change(self, idx: int):
        """Handle active-sample selection from the Setup-tab combo box.

        Connected to ``cb_sample.currentIndexChanged``; delegates to the shared
        :meth:`_on_sample_picker_change` so combo-box and compare-table
        selection use identical activation logic.

        Args:
            idx: New selected row index in ``cb_sample``.
        """
        self._on_sample_picker_change(idx)

    def _on_sample_picker_change(self, idx: int):
        """Switch the active dataset when the user picks a sample index.

        Shared handler for both the Setup-tab sample combo and the Multi-phasor
        compare table selection. Ignored while ``_table_sel_lock`` is set (i.e.
        while :meth:`_sync_sample_table_selection` is itself updating these
        widgets, to avoid re-entrant activation) or when the index is already
        active or out of range.

        Args:
            idx: Requested dataset index to activate.
        """
        if self._table_sel_lock or not (0 <= idx < len(self.datasets)):
            return
        if idx == self.active_idx:
            return
        self._activate_dataset(idx)

    def _sync_sample_table_selection(self):
        """Keep compare table row and sample combo aligned with active_idx.

        Sets ``_table_sel_lock`` and blocks both widgets' signals while
        selecting/scrolling to the active row and setting the combo index, so
        this programmatic sync does not re-trigger
        :meth:`_on_sample_picker_change`. Called after any activation, dataset
        list change, or rename.
        """
        if not hasattr(self, "table_compare"):
            return
        self._table_sel_lock = True
        self.table_compare.blockSignals(True)
        if hasattr(self, "cb_sample"):
            self.cb_sample.blockSignals(True)
        if 0 <= self.active_idx < self.table_compare.rowCount():
            self.table_compare.selectRow(self.active_idx)
            self.table_compare.scrollToItem(
                self.table_compare.item(self.active_idx, 0),
                QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        if hasattr(self, "cb_sample") and 0 <= self.active_idx < self.cb_sample.count():
            self.cb_sample.setCurrentIndex(self.active_idx)
        if hasattr(self, "cb_sample"):
            self.cb_sample.blockSignals(False)
        self.table_compare.blockSignals(False)
        self._table_sel_lock = False
        self._update_multi_strip()

    def _update_apply_buttons(self):
        """Relabel Apply buttons and show Apply all when multiple samples are loaded.

        With a single sample, "Apply" preprocesses the only dataset; once a
        second sample is loaded the label changes to "Apply selected" to
        clarify it only affects the active one, and "Apply all" becomes visible
        and enabled as the way to preprocess every loaded sample at once.
        """
        if not hasattr(self, "chk_multi"):
            return
        multi = len(self.datasets) > 1
        if multi:
            self.btn_apply.setText("Apply selected")
            self.btn_apply.setToolTip(
                "Preprocess the active sample with the settings on the Setup tab.")
        else:
            self.btn_apply.setText("Apply")
            self.btn_apply.setToolTip(
                "Preprocess the loaded sample: phasor maps, filters, and calibration if set.")
        if hasattr(self, "btn_apply_all"):
            self.btn_apply_all.setVisible(multi)
            self.btn_apply_all.setEnabled(multi)

    # ---- mode switching ----------------------------------------------------
    def on_mode_change(self):
        """Toggle between cursor ROI and GMM segmentation modes.

        Connected to ``rb_cursor.toggled``. Shows the matching options box and
        clears state belonging to the other mode: switching to cursors clears
        any GMM fit/labels/overlay; switching to GMM clears phasor cursors and
        the results table (cursor and GMM segmentation share the image overlay
        but keep independent phasor-plot artists and cluster stats).
        """
        self.mode = "cursor" if self.rb_cursor.isChecked() else "gmm"
        self.cursor_box.setVisible(self.mode == "cursor")
        self.gmm_box.setVisible(self.mode == "gmm")
        # Cursor and GMM share the image overlay but not phasor artists — switch clears the other mode.
        if self.mode == "cursor":
            self.clear_gmm()
        else:
            self.phasor.clear_cursors()
            self.last_overlay = None
            self.cluster_stats = []
            self._fill_table()

    # ---- file actions ------------------------------------------------------
    def _dialog_dir(self, key: str, fallback: str = "") -> str:
        """Return the last-used directory for a QFileDialog category.

        Reads a path from ``QSettings`` so file dialogs (sample, reference,
        export) reopen where the user last browsed instead of always starting
        at the OS default location.

        Args:
            key: Settings key such as ``"sample_dir"`` or ``"reference_dir"``.
            fallback: Directory to use when no value has been stored yet.

        Returns:
            Stored directory path, ``fallback``, or ``""``.
        """
        return self._settings.value(key, fallback) or ""

    def choose_sample(self):
        """Open a file dialog and load one or more FLIM sample files.

        Lets the user Ctrl/Shift-click multiple files. Filters the selection
        to supported FLIM formats (``.ptu``, ``.tif``/``.tiff``, ``.lif``/
        ``.xlef``), logging and skipping anything else; shows a warning if none
        of the picked files are supported. Remaining paths are expanded into
        load jobs via :meth:`_expand_sample_load_jobs` (prompting for LIF
        series selection when needed), confirmed against the current session
        state via :meth:`_prepare_sample_load`, and finally decoded by
        :meth:`_load_sample_paths`. Remembers the chosen directory and up to
        five recent paths for next time.
        """
        start = self._dialog_dir("sample_dir")
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choose sample FLIM file(s)", start, FLIM_FILE_FILTER)
        if paths:
            self._settings.setValue("sample_dir", os.path.dirname(paths[0]))
            for p in paths[:5]:
                if hasattr(self, "_remember_recent"):
                    self._remember_recent("recent_samples", p)
        if not paths:
            return
        supported = []
        skipped = 0
        for path in paths:
            if is_supported_flim_path(path):
                supported.append(path)
            else:
                skipped += 1
                self._log(f"Skipped unsupported file: {os.path.basename(path)}")
        if not supported:
            QtWidgets.QMessageBox.warning(
                self, "Unsupported file",
                "Use PicoQuant .ptu, Imspector .tif / .tiff, or Leica .lif / .xlef files.",
            )
            return
        if skipped:
            self._log(f"Skipped {skipped} unsupported file(s).")
        try:
            jobs = self._expand_sample_load_jobs(supported)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "LIF file", str(e))
            return
        if not jobs:
            return
        if not self._prepare_sample_load(jobs):
            return
        self._load_sample_paths(jobs)

    def _expand_sample_load_jobs(self, paths):
        """Turn file paths into load jobs; prompt when a LIF holds multiple FLIM series.

        Plain histogram files (``.ptu``/``.tif``) become one job each. LIF/XLEF
        files are inspected for embedded phasor-map series: a LIF with exactly
        one series is queued automatically; files with multiple series are
        collected and shown in a single :class:`LifSeriesDialog` so the user
        picks which series to load (possibly across several LIF files at once).
        Turning multiple jobs on also switches the window into multi-image
        mode.

        Args:
            paths: File paths already confirmed to be supported FLIM files.

        Returns:
            List of ``(path, image_key)`` job tuples (``image_key`` is ``None``
            for histogram files); empty list if the user cancels the LIF series
            dialog or picks nothing.

        Raises:
            ValueError: If a LIF file cannot be read or contains no phasor
                images at all.
        """
        histogram_paths = []
        lif_pending: dict[str, list[LifPhasorSeries]] = {}

        for path in paths:
            if is_lif_path(path):
                try:
                    series = list_lif_phasor_series(path)
                except Exception as e:
                    raise ValueError(
                        f"{os.path.basename(path)}: cannot read LIF ({e})") from e
                if not series:
                    raise ValueError(
                        f"{os.path.basename(path)}: no phasor images found "
                        "(export phasor maps from LAS X or use .ptu).")
                lif_pending[path] = series
            else:
                histogram_paths.append(path)

        lif_jobs: list[tuple[str, str | None]] = []
        need_dialog: dict[str, list[LifPhasorSeries]] = {}
        for path, series in lif_pending.items():
            if len(series) == 1:
                lif_jobs.append((path, series[0].image_key))
            else:
                need_dialog[path] = series

        if need_dialog:
            dlg = LifSeriesDialog(need_dialog, self)
            if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return []
            picked = dlg.selected_series()
            if not picked:
                return []
            lif_jobs.extend((s.lif_path, s.image_key) for s in picked)

        jobs = [(p, None) for p in histogram_paths] + lif_jobs
        if len(jobs) > 1:
            self.chk_multi.setChecked(True)
        return jobs

    def _prepare_sample_load(self, jobs):
        """Confirm how new load jobs merge with the current session. Returns False if cancelled.

        Asks the user, via message boxes, how to reconcile newly chosen files
        with whatever is already loaded:

        - Batch load (2+ jobs) with an existing sample: offers "Replace all"
          (clears ``self.datasets``) or "Add to list" (keeps the current
          sample and appends), or Cancel.
        - Single job with an existing sample and multi-image mode off: asks
          Yes (turn on multi-image mode and keep the current sample), No
          (replace it), or Cancel.
        - Single job with multi-image mode already on: silently ensures the
          current sample is registered in ``self.datasets`` before the new one
          is appended by the caller.

        Args:
            jobs: Pending load jobs from :meth:`_expand_sample_load_jobs`.

        Returns:
            True if loading should proceed; False if the user cancelled.
        """
        has_current = dataset_has_sample(self.data)
        batch = len(jobs) > 1

        if batch:
            self.chk_multi.setChecked(True)
            if has_current:
                mbox = QtWidgets.QMessageBox(self)
                mbox.setWindowTitle("Load multiple samples")
                mbox.setText(f"Load {len(jobs)} sample(s).")
                mbox.setInformativeText(
                    "Replace all currently loaded samples, or add the new files to the list?")
                btn_replace = mbox.addButton(
                    "Replace all", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                btn_add = mbox.addButton(
                    "Add to list", QtWidgets.QMessageBox.ButtonRole.ActionRole)
                mbox.addButton(
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                mbox.setDefaultButton(btn_add)
                mbox.exec()
                clicked = mbox.clickedButton()
                if clicked is None or clicked == mbox.button(QtWidgets.QMessageBox.StandardButton.Cancel):
                    return False
                if clicked == btn_replace:
                    self.datasets.clear()
                    self.active_idx = -1
                elif self.data not in self.datasets:
                    self.datasets.append(self.data)
            return True

        if has_current and not self.chk_multi.isChecked():
            btn = QtWidgets.QMessageBox.question(
                self, "Multi-image mode",
                "Keep the current sample and load another? This turns on multi-image mode "
                "so you can use the overlay table and shared reference.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if btn == QtWidgets.QMessageBox.StandardButton.Cancel:
                return False
            if btn == QtWidgets.QMessageBox.StandardButton.Yes:
                self.chk_multi.setChecked(True)
                if self.data not in self.datasets:
                    self.datasets.append(self.data)
            elif btn == QtWidgets.QMessageBox.StandardButton.No:
                self.datasets.clear()
                self.active_idx = -1
        elif has_current and self.chk_multi.isChecked() and self.data not in self.datasets:
            self.datasets.append(self.data)
        return True

    def _load_sample_paths(self, jobs):
        """Decode FLIM files and show uncalibrated phasor / image preview.

        For each ``(path, lif_key)`` job, creates a fresh :class:`PhasorData`
        and decodes it off the GUI thread (via :meth:`_run_busy`), either as a
        LIF phasor series or a raw TCSPC/TIFF histogram at the current frame
        index and (if Fast load is on) a single channel. Decode errors are
        logged and shown in a message box; if at least one sample already
        loaded successfully, remaining jobs are simply skipped rather than
        aborting the whole batch. Every successfully decoded dataset is
        activated via :meth:`_activate_new_dataset` and logged with its shape,
        channel count, frequency, and memory estimate.

        After decoding, runs :meth:`_process_uncalibrated_preview` on all newly
        loaded datasets so the phasor plot and image are visible immediately,
        even before a reference is chosen, then refreshes the UI for the active
        sample and warns if the resulting preview is empty (wrong channel or
        no photons).

        Args:
            jobs: List of ``(path, image_key)`` tuples from
                :meth:`_expand_sample_load_jobs`.
        """
        n = len(jobs)
        loaded = []
        t_decode = 0.0
        frame = int(self.sp_frame.value()) if hasattr(self, "sp_frame") else -1
        pref_ch = max(0, self.cb_channel.currentIndex())
        self._settings.setValue("preferred_sample_channel", pref_ch)
        load_ch = pref_ch if self.chk_fast_load.isChecked() else None
        for i, (path, lif_key) in enumerate(jobs):
            d = PhasorData()
            label = os.path.basename(path)
            if lif_key:
                series = lif_key.split("/")[-1] if "/" in lif_key else lif_key
                label = f"{label} · {series}"
            try:
                if lif_key is not None:
                    (shape, nch), elapsed = self._run_busy(
                        f"Reading LIF {i + 1}/{n}: {label}…",
                        lambda p=path, k=lif_key, ds=d: ds.load_lif_phasor(p, k),
                    )
                else:
                    (shape, nch), elapsed = self._run_busy(
                        f"Decoding {i + 1}/{n}: {label}…",
                        lambda p=path, ds=d, fr=frame, lc=load_ch: ds.load_sample(
                            p, frame=fr, load_channel=lc),
                    )
                t_decode += elapsed
                loaded.append((d, path, shape, nch, lif_key))
            except CancelledError:
                if not loaded:
                    return
                break
            except Exception as e:
                self._log(f"Load error ({label}): {e}")
                QtWidgets.QMessageBox.critical(self, "Load error", f"{label}:\n{e}")
                if not loaded:
                    return
                break

        for d, path, shape, nch, lif_key in loaded:
            # Full load defaults channel to 0; honor the Ch combo chosen before Sample….
            if lif_key is None and getattr(d, "fast_loaded_channel", None) is None:
                d.channel = min(pref_ch, max(0, int(nch) - 1))
            self._activate_new_dataset(d, refresh_table=False)
            name = os.path.basename(path)
            if lif_key:
                series = lif_key.split("/")[-1] if "/" in lif_key else lif_key
                name = f"{name} · {series}"
            lasx_note = ""
            if d.lif_lasx_calibrated:
                lasx_note = " · LAS X phasor calibration"
            photon_note = ""
            if getattr(d, "lif_uses_photon_intensity", False):
                photon_note = " · photon Intensity image"
            self._log(
                f"Loaded {name} — {shape[1]}×{shape[0]}, {nch} ch, "
                f"{d.frequency:.2f} MHz{lasx_note}{photon_note} · {format_memory_line(d)}")

        preview_targets = [d for d, _, _, _, _ in loaded]
        t_preview = 0.0
        try:
            _, t_preview = self._run_busy(
                "Uncalibrated preview…",
                lambda: self._process_uncalibrated_preview(preview_targets),
            )
        except CancelledError:
            pass
        except Exception as e:
            self._log(f"Preview error: {e}")
            QtWidgets.QMessageBox.warning(self, "Preview", str(e))

        self._restore_ui_for_active()
        self._refresh_image_combo()
        self._update_apply_buttons()
        self._ensure_compare_overlay_off()
        self.refresh_image()
        if preview_targets and preview_targets[0].real_cal is not None:
            n_valid = int(self.data.valid_mask().sum())
            self._log(
                f"{len(loaded)} sample(s) loaded ({self._fmt_elapsed(t_decode + t_preview)}) — "
                f"uncalibrated preview ({n_valid} valid px). "
                "Choose a Reference and Apply for reference-corrected maps.")
            st = getattr(self.data, "_intensity_stats", {}) or {}
            self._warn_if_empty_phasor(n_valid, st, after_load=True)
        else:
            self._log(
                f"{len(loaded)} sample(s) decoded ({self._fmt_elapsed(t_decode)}) — "
                "choose Reference, then Apply (calibration is automatic).")

    def _warn_if_empty_phasor(self, n_valid: int, stats: dict | None = None, *, after_load: bool = False):
        """Alert when the image/phasor is empty (wrong channel or too-high Min N).

        Distinguishes three empty-result causes from ``stats`` (produced by
        :class:`PhasorData`'s intensity statistics) to give an actionable
        message: the selected channel has zero photons everywhere, the Min N
        threshold masked out every pixel, or some other calibration/filter
        issue left no valid phasor pixels. Shown as a warning dialog and also
        logged; does nothing when ``n_valid`` is positive and the max photon
        count is nonzero.

        Args:
            n_valid: Count of valid (finite, above-threshold) phasor pixels.
            stats: Intensity statistics dict with optional "max" and
                "threshold" keys; treated as empty if ``None``.
            after_load: True right after decoding a sample (titles the dialog
                "Empty image"); False after Apply/preprocessing (titles it
                "Empty phasor").
        """
        stats = stats or {}
        max_ph = float(stats.get("max", -1))
        if n_valid > 0 and max_ph != 0:
            return
        ch = getattr(self.data, "channel", 0)
        nch = getattr(self.data, "n_channels", 1)
        thr = float(stats.get("threshold", 0) or 0)
        if max_ph == 0:
            msg = (
                f"Channel {ch} has no photons (counts are all zero).\n\n"
                "Try another sample channel — with Fast load, switching channel "
                "re-decodes that channel only."
            )
            if nch > 1:
                msg += f"\nThis file reports {nch} channels."
        elif thr > 0:
            msg = (
                f"No valid pixels after Apply (Min N = {thr:.0f}).\n\n"
                "Lower Min N, or check that the selected channel has enough photons."
            )
        else:
            msg = (
                "No valid phasor pixels.\n\n"
                "Check sample channel, reference calibration, and filter settings."
            )
        title = "Empty image" if after_load else "Empty phasor"
        QtWidgets.QMessageBox.warning(self, title, msg)
        self._log(f"Warning: {msg.splitlines()[0]}")

    def _effective_ref_path(self, d=None):
        """Return the reference file path for a dataset (shared or per-sample).

        In shared-reference mode, every dataset uses ``self.shared_ref_path``
        regardless of its own ``ref_path``. Otherwise falls back to the
        dataset's own reference path.

        Args:
            d: Dataset to look up; defaults to the active dataset.

        Returns:
            Reference file path, or ``None`` if no reference is set.
        """
        d = d or self.data
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return self.shared_ref_path
        return d.ref_path or None

    def _ref_channel_for_dataset(self, d):
        """Return the reference channel index used when calibrating a dataset.

        Mirrors :meth:`_effective_ref_path`'s shared-vs-per-sample choice, but
        for the reference detector channel, and clamps the index to the known
        channel count so a stale value from a previous reference file (with
        fewer channels) can't go out of range.

        Args:
            d: Dataset to look up.

        Returns:
            Zero-based reference channel index.
        """
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return min(self.shared_ref_channel, max(0, self.shared_ref_n_channels - 1))
        return min(d.ref_channel, max(0, d.ref_n_channels - 1)) if d.ref_path else 0

    def _propagate_shared_reference(self):
        """Copy shared reference settings onto every loaded sample when shared mode is on.

        Writes ``shared_ref_path``, ``shared_ref_n_channels``, and a clamped
        ``shared_ref_channel`` onto every dataset returned by
        :meth:`_all_datasets`, so per-dataset code paths (export, session
        save, compare-table status) see a consistent reference even though the
        UI only exposes one set of shared-reference controls. No-op when
        shared-reference mode is off or no shared path has been chosen yet.
        """
        if not self.chk_shared_ref.isChecked():
            return
        path = self.shared_ref_path
        if not path:
            return
        for d in self._all_datasets():
            d.ref_path = path
            d.ref_n_channels = self.shared_ref_n_channels
            d.ref_channel = min(self.shared_ref_channel, d.ref_n_channels - 1)

    def on_shared_ref_toggle(self, checked):
        """Enable or disable one reference file for all loaded samples.

        Connected to ``chk_shared_ref.toggled``. Turning shared mode on
        immediately propagates the existing shared reference to every loaded
        dataset via :meth:`_propagate_shared_reference`; turning it off leaves
        each dataset's own ``ref_path`` in charge going forward. Either way,
        refreshes the UI for the active dataset and logs the new mode.

        Args:
            checked: New state of the "Shared ref" checkbox.
        """
        if checked and self.shared_ref_path:
            self._propagate_shared_reference()
        self._restore_ui_for_active()
        self._log(
            "Shared reference on — one reference file calibrates all samples."
            if checked else
            "Shared reference off — each sample can use its own reference (active sample).")

    def _all_datasets(self):
        """Return active and listed datasets without duplicates.

        The active dataset (``self.data``) is often the same object as an
        entry in ``self.datasets``, but not always (e.g. right after loading,
        before it is appended); this merges both sources so callers that need
        to touch "every dataset" (reference propagation, export, session save)
        never process the active one twice or miss it.

        Returns:
            List of unique :class:`PhasorData` instances, active dataset first.
        """
        seen = []
        for d in [self.data] + list(self.datasets):
            if d is not None and d not in seen:
                seen.append(d)
        return seen

    def _reference_harmonic_for_cal(self):
        """Return harmonic index (or list for pawflim) for reference calibration.

        pawflim's frequency-domain filter needs both the fundamental and its
        first overtone to calibrate correctly, so when that filter is selected
        this returns ``[h, 2 * h]`` instead of a single harmonic index.

        Returns:
            Single harmonic integer, or a two-element list for pawflim.
        """
        h = int(self.sp_harm.value())
        if self.cb_filter.currentText() == "pawflim":
            return [h, 2 * h]
        return h

    def _effective_ref_file_path(self):
        """Return the reference file path selected for Calibrate/Apply.

        Preferred order: the shared reference path (if shared mode is on),
        then the active dataset's own ``ref_path``. If no sample is loaded
        yet, falls back to the last picked shared path so choosing Reference…
        before any sample still primes calibration; once a sample exists, its
        own (possibly empty) ``ref_path`` is authoritative and no fallback is
        used, so per-sample mode does not leak the shared path onto a sample
        that intentionally has no reference.

        Returns:
            Reference file path, or ``""`` if none is applicable.
        """
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return self.shared_ref_path
        if self.data.ref_path:
            return self.data.ref_path
        # Per-sample mode: only fall back to the last picked path before any
        # sample is loaded (Reference… can be chosen first). Once a sample is
        # loaded its own (possibly empty) ref_path is authoritative.
        if not dataset_has_sample(self.data):
            return self.shared_ref_path or ""
        return ""

    def _calibration_ready_for_apply(self) -> bool:
        """True when Apply can run without decoding the reference file again.

        With manual calibration, readiness only depends on the manual g/s
        being active. With a file-based reference, Apply can proceed either
        because calibration is already current or because no reference file
        is selected at all (samples are then processed uncalibrated).

        Returns:
            True if Apply does not need to trigger a fresh reference decode.
        """
        if self.chk_manual_cal.isChecked():
            return self.ref_calibration.is_active
        if not self._effective_ref_file_path():
            return True
        return self.ref_calibration.is_active

    def _recompute_reference_calibration(self):
        """Decode reference once; store scalar g/s only (reference histogram is released).

        Runs :func:`~flim_phasors.calibration.compute_reference_phasor` off the
        GUI thread for the effective reference file, channel, and harmonic(s),
        then discards the decoded reference maps (``cal._maps = None``) so only
        the mean (g, s) — and per-harmonic g/s for pawflim — stay resident in
        memory. Also syncs the shared reference channel count, propagates it to
        all datasets, refreshes the reference-channel combo and manual g/s
        fields, updates the calibration display and preview plot, and marks the
        calibration as current for the active dataset. Errors are logged and
        shown in a message box rather than raised.

        Returns:
            True on success; False if no reference path is set or decoding
            failed.
        """
        path = self._effective_ref_file_path()
        if not path:
            return False
        ch = self._ref_channel_for_dataset(self.data)
        harm = self._reference_harmonic_for_cal()

        def work():
            """Decode reference file and compute mean g/s (runs off GUI thread).

            Closure over ``path``, ``ch``, and ``harm`` captured from the
            enclosing :meth:`_recompute_reference_calibration` call; passed to
            :meth:`_run_busy` so the decode does not block the UI.

            Returns:
                :class:`~flim_phasors.calibration.ReferenceCalibration` with
                mean g/s (and harmonic g/s for multi-harmonic requests).
            """
            return compute_reference_phasor(path, ch, harm)

        try:
            cal, elapsed = self._run_busy(
                f"Reference phasor ({os.path.basename(path)})…", work)
        except Exception as e:
            self._log(f"Reference calibration error: {e}")
            QtWidgets.QMessageBox.critical(self, "Reference error", str(e))
            return False
        cal._maps = None
        self.ref_calibration = cal
        self.ref_calibration.use_manual = self.chk_manual_cal.isChecked()
        # Keep ref-channel combo in sync (auto-calibrate path used to skip this).
        self.shared_ref_n_channels = max(1, int(cal.n_channels))
        self.shared_ref_channel = min(self.shared_ref_channel, self.shared_ref_n_channels - 1)
        self.data.ref_n_channels = self.shared_ref_n_channels
        self._propagate_shared_reference()
        self._update_ref_channel_combo()
        if self.chk_manual_cal.isChecked():
            self._apply_manual_calibration_fields()
        else:
            self._sync_manual_fields_from_calibration()
        self._update_calibration_display()
        self._update_ref_preview()
        self._mark_calibration_current()
        self._ensure_compare_overlay_off()
        gs_note = ""
        if cal.harmonic_gs and len(cal.harmonic_gs) > 1:
            parts = [f"H{i + 1}=({g:.4f},{s:.4f})" for i, (g, s) in enumerate(cal.harmonic_gs)]
            gs_note = " [" + "; ".join(parts) + "]"
        self._log(
            f"Reference g/s stored (g={cal.mean_g:.4f}, s={cal.mean_s:.4f}){gs_note} "
            f"— scalar calibration; reference file not kept in RAM "
            f"({self._fmt_elapsed(elapsed)}).")
        return True

    def _sync_manual_fields_from_calibration(self):
        """Copy stored calibration g/s into the manual entry fields.

        Lets the user switch on manual calibration and see the currently
        active (file-derived) g/s as a starting point instead of an empty or
        stale field. Blocks signals while writing so this does not itself
        trigger a manual-calibration re-apply.
        """
        self.edit_ref_g.blockSignals(True)
        self.edit_ref_s.blockSignals(True)
        self.edit_ref_g.setText(f"{self.ref_calibration.mean_g:.5f}")
        self.edit_ref_s.setText(f"{self.ref_calibration.mean_s:.5f}")
        self.edit_ref_g.blockSignals(False)
        self.edit_ref_s.blockSignals(False)

    def _apply_manual_calibration_fields(self):
        """Read manual g/s fields and update the in-memory calibration object.

        Parses ``edit_ref_g``/``edit_ref_s`` and, if both are valid floats,
        stores them as both the manual and the effective mean g/s on
        ``self.ref_calibration`` and marks it as manual. Silently does nothing
        on a parse error (e.g. field temporarily empty while typing) — the
        caller is expected to have already gated this on the Manual ref phasor
        checkbox being checked.
        """
        try:
            self.ref_calibration.manual_g = float(self.edit_ref_g.text().strip())
            self.ref_calibration.manual_s = float(self.edit_ref_s.text().strip())
        except ValueError:
            return
        self.ref_calibration.use_manual = True
        self.ref_calibration.mean_g = self.ref_calibration.manual_g
        self.ref_calibration.mean_s = self.ref_calibration.manual_s

    def _update_calibration_display(self):
        """Refresh calibration summary label and panel status line.

        Rebuilds ``lbl_cal_display`` text from ``self.ref_calibration``: shows
        an "uncalibrated" placeholder when inactive, otherwise the source
        (manual or reference filename plus channel), mean g/s, an optional
        second-harmonic g/s for pawflim, and the current reference lifetime.
        Also refreshes the top panel status line, which reflects the same
        calibration state in condensed form.
        """
        if hasattr(self, "_update_panel_status"):
            self._update_panel_status()
        cal = self.ref_calibration
        if not cal.is_active:
            self.lbl_cal_display.setText("(uncalibrated — load a reference file)")
            return
        if cal.use_manual:
            src = "Manual g/s"
            chan = ""
        else:
            src = os.path.basename(cal.source_path or "") or "reference"
            chan = f"  |  ch {cal.channel}"
        gs = f"g={cal.mean_g:.4f}, s={cal.mean_s:.4f}"
        if cal.harmonic_gs and len(cal.harmonic_gs) > 1:
            g2, s2 = cal.harmonic_gs[1]
            gs += f" · H2 g={g2:.4f}, s={s2:.4f}"
        self.lbl_cal_display.setText(
            f"Calibrated · {src}  |  {gs}  |  "
            f"τ_ref={self.sp_reflt.value():.2f} ns{chan}")

    def _on_manual_cal_toggled(self, checked):
        """Enable manual g/s entry and recompute file-based calibration when off.

        Connected to ``chk_manual_cal.toggled``. Turning manual mode on enables
        the g/s fields and Set button and seeds them from the current
        calibration; turning it off disables them and, if a reference file was
        previously selected, re-decodes it via
        :meth:`_recompute_reference_calibration` so file-based g/s replaces the
        manual values. Refreshes the calibration display and reference preview
        either way.

        Args:
            checked: New state of the "Manual ref phasor" checkbox.
        """
        self.edit_ref_g.setEnabled(checked)
        self.edit_ref_s.setEnabled(checked)
        self.btn_set_manual_gs.setEnabled(checked)
        self.ref_calibration.use_manual = checked
        if checked:
            self._sync_manual_fields_from_calibration()
        elif self.ref_calibration.source_path:
            self._recompute_reference_calibration()
        self._update_calibration_display()
        self._update_ref_preview()

    def apply_manual_gs(self):
        """Apply typed g/s to calibration (does not preprocess samples).

        Connected to the "Set g/s" button. Requires Manual ref phasor to be
        enabled first (otherwise shows an informational message). Validates
        that both fields parse as floats and lie roughly on/near the phasor
        semicircle (``-0.05`` to ``1.05`` for both g and s) before committing
        them as the active calibration, updating the display and preview, and
        marking calibration current. The user still needs to click Apply to
        actually reprocess sample(s) with the new g/s.
        """
        if not self.chk_manual_cal.isChecked():
            QtWidgets.QMessageBox.information(
                self, "Manual calibration", "Enable Manual ref phasor first.")
            return
        try:
            g = float(self.edit_ref_g.text().strip())
            s = float(self.edit_ref_s.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Manual calibration", "Enter valid numeric g and s values.")
            return
        if not (-0.05 <= g <= 1.05 and -0.05 <= s <= 1.05):
            QtWidgets.QMessageBox.warning(
                self, "Manual calibration",
                "g and s should be on the phasor semicircle (roughly g: 0–1, s: 0–0.7).")
            return
        self.ref_calibration.use_manual = True
        self.ref_calibration.manual_g = g
        self.ref_calibration.manual_s = s
        self.ref_calibration.mean_g = g
        self.ref_calibration.mean_s = s
        self.ref_calibration.values_ready = True
        self._update_calibration_display()
        self._update_ref_preview()
        self._mark_calibration_current()
        self._log(
            f"Manual g/s set — g={g:.4f}, s={s:.4f}. "
            "Click Apply to preprocess sample(s) with this calibration.")

    def _on_ref_lifetime_or_freq_changed(self, *_args):
        """Refresh calibration label/preview after Ref τ or laser frequency changes.

        Connected to ``sp_reflt.valueChanged`` and ``sp_freq.valueChanged``.
        Neither value affects the stored reference g/s itself (that depends
        only on the decoded reference file), but both change where the
        reference *should* sit on the universal semicircle, so the calibration
        summary text and the reference preview plot's target marker need to be
        redrawn. Does nothing while calibration is inactive.

        Args:
            *_args: Ignored spinbox value-changed payload.
        """
        if self.ref_calibration.is_active:
            self._update_calibration_display()
            self._update_ref_preview()

    def _on_harm_or_ref_setting_changed(self, *_args):
        """Harmonic/filter changes need a fresh Calibrate before Apply.

        Connected to ``sp_harm.valueChanged`` and ``cb_filter.currentTextChanged``.
        Changing the harmonic (or switching to/from pawflim, which needs a
        second harmonic) invalidates any previously computed file-based g/s,
        since that was decoded at the old harmonic; this clears
        ``values_ready`` and the cached reference maps so the next
        Calibrate/Apply re-decodes. Manual calibration is unaffected since it
        does not depend on harmonic. Updates the calibration display and
        "stale" styling whenever there is an active calibration or a reference
        file selected.

        Args:
            *_args: Ignored signal payload.
        """
        if getattr(self, "_loading_proc_ui", False):
            return
        if self.ref_calibration.values_ready and not self.chk_manual_cal.isChecked():
            self.ref_calibration.values_ready = False
            self.ref_calibration._maps = None
        if self.ref_calibration.is_active or self._effective_ref_file_path():
            self._update_calibration_display()
            self._update_calibration_stale_style()

    def _clear_calibration(self):
        """Reset reference calibration, paths, and per-sample ref assignments.

        Connected to the "Clear cal" button. Clears ``self.ref_calibration``
        and the module-level reference decode cache, resets the shared
        reference path/channel count, turns off manual calibration, and clears
        ``ref_path`` on every dataset via :meth:`_all_datasets` — a full reset
        so the next Reference… pick starts from a clean slate rather than
        merging with stale state.
        """
        self.ref_calibration.clear()
        clear_calibration_cache()
        self.shared_ref_path = ""
        self.shared_ref_n_channels = 1
        pref_rch = int(self._settings.value("preferred_ref_channel", self.shared_ref_channel))
        self.shared_ref_channel = min(max(0, pref_rch), CHANNEL_PRESELECT_MAX)
        self.lbl_ref.setText("(none)")
        self.chk_manual_cal.setChecked(False)
        self._sync_manual_fields_from_calibration()
        self._update_calibration_display()
        self._update_ref_preview()
        for d in self._all_datasets():
            d.ref_path = ""
            d.ref_n_channels = 1
            d.ref_channel = self.shared_ref_channel
        self._update_ref_channel_combo()
        self._log("Calibration cleared.")

    def choose_ref(self):
        """Pick the calibration reference file (decode happens on Calibrate).

        Opens a file dialog restricted to supported histogram FLIM files;
        rejects LIF files outright since Leica LIF phasor maps have no TCSPC
        histogram and cannot serve as a reference measurement. On a valid
        pick, delegates to :meth:`_set_reference_path`, which (by default)
        triggers automatic calibration. Remembers the chosen directory and
        path for next time.
        """
        start = self._dialog_dir("reference_dir", self._dialog_dir("sample_dir"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose calibration reference file", start, FLIM_FILE_FILTER)
        if path:
            self._settings.setValue("reference_dir", os.path.dirname(path))
            if hasattr(self, "_remember_recent"):
                self._remember_recent("recent_refs", path)
        if not path:
            return
        if not is_supported_flim_path(path):
            QtWidgets.QMessageBox.warning(
                self, "Unsupported file",
                "Use a PicoQuant .ptu or Imspector .tif / .tiff FLIM stack.",
            )
            return
        if is_lif_path(path):
            QtWidgets.QMessageBox.warning(
                self, "Reference file",
                "Leica LIF phasor maps cannot be used as a TCSPC reference.\n"
                "Use a .ptu or .tif reference measurement instead.",
            )
            return
        self._set_reference_path(path)

    def _set_reference_path(self, path: str, *, auto: bool = True):
        """Store the reference path and (by default) calibrate automatically.

        Invalidates any previously computed reference maps when the path
        actually changes and manual calibration is not checked, updates
        the reference label, propagates the shared reference to other
        datasets, and refreshes the reference-channel combo box. When
        ``auto`` is true this also triggers an immediate decode and g/s
        computation so the calibration preview updates without a separate
        button click.

        Args:
            path: Reference file path.
            auto: When True, decode the reference and compute g/s immediately so
                no separate Calibrate click is needed. Set False from the manual
                Calibrate button, which performs its own decode.
        """
        norm_new = os.path.normcase(path or "")
        norm_old = os.path.normcase(self.ref_calibration.source_path or "")
        if norm_new and norm_old and norm_new != norm_old and not self.chk_manual_cal.isChecked():
            self.ref_calibration._maps = None
            self.ref_calibration.values_ready = False
        self.shared_ref_path = path
        self.lbl_ref.setText(os.path.basename(path))
        self.data.ref_path = path
        # Probe channel count and keep the Ch combo choice made before Reference….
        nch = flim_channel_count(path) if path else None
        if nch:
            self.shared_ref_n_channels = max(1, int(nch))
        else:
            self.shared_ref_n_channels = max(1, self.shared_ref_n_channels)
        pref = max(0, self.cb_ref_channel.currentIndex())
        self.shared_ref_channel = min(pref, self.shared_ref_n_channels - 1)
        self._settings.setValue("preferred_ref_channel", self.shared_ref_channel)
        self.data.ref_n_channels = self.shared_ref_n_channels
        self.data.ref_channel = self.shared_ref_channel
        self._propagate_shared_reference()
        self._update_ref_channel_combo()
        self._log(
            f"Reference file selected: {os.path.basename(path)} "
            f"(channel {self.shared_ref_channel}).")
        if auto and path and not self.chk_manual_cal.isChecked():
            self._auto_calibrate_reference()
        else:
            self._update_calibration_display()

    def _auto_calibrate_reference(self) -> bool:
        """Compute reference g/s on demand, without an explicit Calibrate click.

        Decodes the reference once when a file is selected and g/s are not
        current (first use, or after a harmonic/channel change invalidated them).

        Returns:
            True when calibration is ready to Apply (or intentionally
            uncalibrated because no reference is selected); False if a reference
            is selected but could not be decoded.
        """
        if self.chk_manual_cal.isChecked():
            self._apply_manual_calibration_fields()
            return self.ref_calibration.is_active
        if not self._effective_ref_file_path():
            return True
        if self.ref_calibration.values_ready:
            return True
        return self._recompute_reference_calibration()

    def calibrate_reference(self):
        """Decode reference and compute g/s (does not preprocess samples).

        Connected to the "Recalibrate" button — an explicit way to force a
        re-decode / refresh the reference preview outside the automatic
        calibration that normally happens when a reference is picked or Apply
        runs. If no reference is currently set, prompts with a file dialog
        first. Under manual calibration, just reapplies the typed g/s instead
        of decoding a file. Otherwise calls
        :meth:`_recompute_reference_calibration` and, on success, syncs the
        reference channel count/combo, refreshes the preview, and marks the
        calibration current. Does not preprocess any sample — that still
        requires clicking Apply.
        """
        path = self._effective_ref_file_path()
        if not path:
            start = self._dialog_dir("reference_dir", self._dialog_dir("sample_dir"))
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose calibration reference file", start, FLIM_FILE_FILTER)
            if not path:
                QtWidgets.QMessageBox.information(
                    self, "Calibrate",
                    "Choose a reference file under Reference… or here first.")
                return
            if not is_supported_flim_path(path):
                return
            self._set_reference_path(path, auto=False)
        if self.chk_manual_cal.isChecked():
            self._apply_manual_calibration_fields()
            self.ref_calibration.use_manual = True
            self.ref_calibration.source_path = path or ""
            self.ref_calibration.values_ready = True
            self._update_calibration_display()
            self._update_ref_preview()
            self._mark_calibration_current()
            self._ensure_compare_overlay_off()
            self._log("Manual calibration values applied.")
            return
        if not self._recompute_reference_calibration():
            return
        ref_nch = max(1, int(self.ref_calibration.n_channels))
        self.shared_ref_n_channels = ref_nch
        # Keep the reference channel the user selected; only clamp to valid range.
        # (Do not force it to the sample channel — the calibration was computed
        # with the selected reference channel and the UI must stay consistent.)
        self.shared_ref_channel = min(self.shared_ref_channel, ref_nch - 1)
        self.data.ref_n_channels = ref_nch
        self._propagate_shared_reference()
        self._update_ref_channel_combo()
        self._update_ref_preview()
        self._mark_calibration_current()
        self._ensure_compare_overlay_off()
        self._log(
            f"Calibration ready (g={self.ref_calibration.mean_g:.4f}, "
            f"s={self.ref_calibration.mean_s:.4f}) — click Apply to preprocess samples.")

    # ---- multi-image management -------------------------------------------
    def _apply_lif_dataset_defaults(self, d: PhasorData):
        """Seed per-sample settings from LAS X metadata after LIF load.

        LIF phasor-map series arrive already calibrated by Leica LAS X at a
        known modulation frequency, and sometimes with an intensity threshold
        LAS X itself applied. This copies that frequency into the dataset's
        stored processing settings and, if LAS X reported an intensity
        threshold, seeds ``intensity_min`` with it so the Setup tab's Min N
        reflects the same cutoff. No-op for non-LIF datasets.

        Args:
            d: Newly loaded dataset to seed defaults for.
        """
        if d.load_source != "lif_phasor":
            return
        if d.processing_settings is None:
            d.processing_settings = capture_processing_from_ui(self)
        d.processing_settings["frequency"] = float(d.frequency)
        thr = float(getattr(d, "lif_lasx_intensity_threshold", 0) or 0)
        if thr > 0:
            d.processing_settings["intensity_min"] = thr

    def _activate_new_dataset(self, d, *, refresh_table: bool = True):
        """Make d the active dataset; append to the set if multi-image mode is on.

        Called once per freshly loaded dataset from :meth:`_load_sample_paths`.
        Assigns a reference (shared reference if enabled, otherwise the
        existing shared path as a fallback if the dataset has none of its
        own), appends to ``self.datasets`` and updates ``active_idx`` when
        multi-image mode is on, seeds per-sample processing settings and LIF
        defaults, and finally makes ``d`` the active dataset and refreshes the
        UI for it.

        Args:
            d: Newly created and decoded dataset.
            refresh_table: When True (default) and multi-image mode is on,
                also rebuilds the compare table; callers loading a whole batch
                pass False and refresh once at the end instead.
        """
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            d.ref_path = self.shared_ref_path
            d.ref_n_channels = self.shared_ref_n_channels
            d.ref_channel = min(self.shared_ref_channel, d.ref_n_channels - 1)
        elif self.shared_ref_path and not d.ref_path:
            d.ref_path = self.shared_ref_path
            d.ref_n_channels = max(1, self.shared_ref_n_channels)
            d.ref_channel = min(self.shared_ref_channel, d.ref_n_channels - 1)
        if self.chk_multi.isChecked():
            self.datasets.append(d)
            self.active_idx = len(self.datasets) - 1
        self._init_dataset_proc_settings(d)
        self._apply_lif_dataset_defaults(d)
        self.data = d
        self._restore_ui_for_active()
        if self.chk_multi.isChecked() and refresh_table:
            self._refresh_compare_list()

    @staticmethod
    def _compact_filename(path, fallback="(none)", max_len=30):
        """Truncate a basename for compact panel and status labels.

        Keeps sample/reference filenames from overflowing the narrow labels in
        the Files group box and panel status line by ellipsizing anything
        longer than ``max_len`` characters.

        Args:
            path: File path to display, or falsy to use ``fallback`` directly.
            fallback: Text shown when ``path`` is empty/``None``.
            max_len: Maximum characters before truncating with an ellipsis.

        Returns:
            Basename of ``path`` (or ``fallback``), truncated if needed.
        """
        name = os.path.basename(path) if path else fallback
        if len(name) > max_len:
            return name[: max_len - 1] + "…"
        return name

    def _update_proc_active_label(self):
        """Show the active sample filename beside the sample combo when solo.

        The "Active sample" row on the Setup tab shows either the sample combo
        (when 2+ samples are loaded in multi-image mode) or a plain filename
        label (when there is 0 or 1 sample, so a dropdown would be
        redundant). This keeps that label's text in sync; visibility of the
        row itself is handled by :meth:`_update_multi_strip`.
        """
        if not hasattr(self, "lbl_proc_active"):
            return
        multi = hasattr(self, "chk_multi") and self.chk_multi.isChecked() and len(self.datasets) > 1
        if multi:
            self.lbl_proc_active.setText("")
            return
        if dataset_has_sample(self.data):
            self.lbl_proc_active.setText(
                self._compact_filename(dataset_short_label(self.data), "(no sample)"))
        else:
            self.lbl_proc_active.setText("(no sample)")

    def _update_panel_status(self):
        """Update the top panel status with active sample and calibration state.

        Builds the always-visible status line above the tabbed control panel:
        the active sample's compact filename plus one of "uncalibrated",
        "preview (uncalibrated)" (real_cal computed but not yet
        reference-corrected), "ref g=… s (not applied)" (calibration ready but
        Apply not yet run), or the actual applied g/s. Appends a sample count
        when more than one dataset is loaded.
        """
        if not hasattr(self, "lbl_panel_status"):
            return
        if not dataset_has_sample(self.data):
            self.lbl_panel_status.setText("No sample loaded")
            return
        name = self._compact_filename(dataset_short_label(self.data), "(no sample)")
        if getattr(self.data, "maps_calibrated", False):
            cal = f"g={self.ref_calibration.mean_g:.3f} s={self.ref_calibration.mean_s:.3f}"
        elif self.data.real_cal is not None:
            cal = "preview (uncalibrated)"
        elif self.ref_calibration.is_active:
            cal = f"ref g={self.ref_calibration.mean_g:.3f} s (not applied)"
        else:
            cal = "uncalibrated"
        n = len(self.datasets)
        if n > 1:
            self.lbl_panel_status.setText(
                f"Active: {name}  ·  {cal}  ·  {n} samples")
        else:
            self.lbl_panel_status.setText(f"Active: {name}  ·  {cal}")

    def _restore_ui_for_active(self):
        """Sync all file, channel, processing, and label controls to the active dataset.

        The central "make the UI match ``self.data``" routine, called after
        every activation, load, or dataset-list change. Updates the sample and
        reference filename labels, rebuilds the channel combo for the
        dataset's channel count, syncs the reference-channel combo, restores
        frequency/harmonic spinboxes, applies LIF-specific defaults (intensity
        threshold, forcing the image view to the first entry), loads
        per-sample processing settings into the Setup tab controls, updates
        the frame spinbox range/value, and populates the display-name/group
        fields and their placeholders. Finishes by syncing table/combo
        selection to match.
        """
        d = self.data
        self.lbl_sample.setText(self._compact_filename(d.sample_path, "(no sample)"))
        self._update_proc_active_label()
        self._update_panel_status()
        if self.chk_shared_ref.isChecked():
            ref = self.shared_ref_path
        else:
            ref = d.ref_path
        self.lbl_ref.setText(self._compact_filename(ref) if ref else "(none)")
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        if dataset_has_sample(d):
            nch = max(1, d.n_channels)
            self.cb_channel.addItems([str(i) for i in range(nch)])
            self.cb_channel.setCurrentIndex(min(d.channel, nch - 1))
        else:
            # No sample yet: keep a preselect list so Ch can be chosen before Sample….
            self.cb_channel.addItems([str(i) for i in range(CHANNEL_PRESELECT_MAX + 1)])
            pref = int(self._settings.value("preferred_sample_channel", 0))
            self.cb_channel.setCurrentIndex(min(max(0, pref), CHANNEL_PRESELECT_MAX))
        self.cb_channel.blockSignals(False)
        self._update_ref_channel_combo()
        # Block harmonic/filter signals: restoring another sample's settings must
        # not invalidate the shared reference calibration as a side effect.
        self.sp_freq.blockSignals(True)
        self.sp_harm.blockSignals(True)
        self.cb_filter.blockSignals(True)
        try:
            self.sp_freq.setValue(d.frequency)
            self.sp_harm.setValue(d.harmonic)
            if d.load_source == "lif_phasor":
                thr = float(getattr(d, "lif_lasx_intensity_threshold", 0) or 0)
                if thr > 0 and hasattr(self, "sp_thr"):
                    self.sp_thr.setValue(int(thr))
                if hasattr(self, "cb_image_view"):
                    self.cb_image_view.blockSignals(True)
                    self.cb_image_view.setCurrentIndex(0)
                    self.cb_image_view.blockSignals(False)
            self._load_proc_to_ui(d)
        finally:
            self.sp_freq.blockSignals(False)
            self.sp_harm.blockSignals(False)
            self.cb_filter.blockSignals(False)
        self._update_frame_control()
        if hasattr(self, "edit_display_name"):
            self.edit_display_name.blockSignals(True)
            self.edit_display_name.setText((d.display_name or "").strip())
            file_hint = _dataset_file_label(d, self.active_idx)
            self.edit_display_name.setPlaceholderText(file_hint)
            self.edit_display_name.blockSignals(False)
        if hasattr(self, "edit_group"):
            self.edit_group.blockSignals(True)
            self.edit_group.setText((d.group_name or "").strip())
            self.edit_group.blockSignals(False)
        self._update_sample_label_controls()
        self._sync_sample_table_selection()

    def _update_ref_channel_combo(self):
        """Populate reference channel combo from shared or per-sample ref metadata.

        Rebuilds ``cb_ref_channel`` with one entry per known reference channel
        (shared or per-sample). When no reference is loaded yet, keeps a
        preselect list (0…``CHANNEL_PRESELECT_MAX``) so the channel can be
        chosen before Reference…. The combo stays enabled either way.
        """
        ref_path = self._effective_ref_path(self.data)
        has_ref = bool(ref_path)
        if has_ref:
            if self.chk_shared_ref.isChecked() and self.shared_ref_path:
                nch = max(1, self.shared_ref_n_channels)
                ch = self.shared_ref_channel
            else:
                nch = max(1, self.data.ref_n_channels)
                ch = self.data.ref_channel
        else:
            nch = CHANNEL_PRESELECT_MAX + 1
            ch = self.shared_ref_channel
        self.cb_ref_channel.blockSignals(True)
        self.cb_ref_channel.clear()
        self.cb_ref_channel.addItems([str(i) for i in range(nch)])
        self.cb_ref_channel.setCurrentIndex(min(max(0, ch), nch - 1))
        self.cb_ref_channel.setEnabled(True)
        self.cb_ref_channel.blockSignals(False)

    def _set_multi_detail_enabled(self, enabled):
        """Enable or disable multi-image tab widgets as a group.

        Toggles every widget in ``self._multi_detail_widgets`` (sample combo,
        compare table, overlay controls, legend controls, etc.) together, then
        re-applies the separate sample-label enable rule since name/group
        fields depend on whether a sample is loaded, not just multi-image mode.

        Args:
            enabled: New enabled state for the widget group.
        """
        for w in self._multi_detail_widgets:
            w.setEnabled(enabled)
        self._update_sample_label_controls()

    def _update_sample_label_controls(self):
        """Enable display-name and group fields only when a sample is loaded.

        Prevents the user from typing a display name or group label before any
        data exists (there would be nothing to attach it to). Called from
        :meth:`_restore_ui_for_active` and :meth:`_set_multi_detail_enabled`.
        """
        has_sample = dataset_has_sample(self.data)
        for w in getattr(self, "_sample_label_widgets", ()):
            w.setEnabled(has_sample)

    def _process_all_loaded_datasets(self, *, use_ui_settings=True):
        """Re-run phasor pipeline on every loaded sample (caller may wrap in _run_busy).

        Iterates either all datasets (multi-image mode with a non-empty list)
        or just the single active dataset, skipping any without a loaded
        sample, and calls :meth:`_run_processing_on_dataset` for each. Saves
        and restores ``self.data``/``self.active_idx`` around the loop so the
        originally active dataset remains active afterward regardless of
        iteration order. Does not itself show a busy dialog — callers such as
        "Apply all" wrap this in :meth:`_run_busy`.

        Args:
            use_ui_settings: Forwarded to
                :meth:`_run_processing_on_dataset` — whether to pull settings
                from the shared UI controls versus each dataset's stored
                per-sample settings.
        """
        if self.chk_multi.isChecked() and self.datasets:
            targets = list(self.datasets)
        else:
            targets = [self.data]
        saved = self.data
        saved_idx = self.active_idx
        for d in targets:
            if not dataset_has_sample(d):
                continue
            self._run_processing_on_dataset(d, use_ui_settings=use_ui_settings)
        if 0 <= saved_idx < len(self.datasets):
            self.data = self.datasets[saved_idx]
            self.active_idx = saved_idx
        else:
            self.data = saved

    def _refresh_views_after_processing(self):
        """Update UI, phasor plot, and image view after Apply completes.

        Called once Apply/Apply all finishes reprocessing. Resyncs the Setup
        tab to the active dataset, rebuilds the compare table and overlay
        group filter, redraws the phasor plot, turns the image Overlay
        checkbox off (any existing segmentation mask is now stale relative to
        the freshly processed maps, so showing the raw map is less misleading
        than an outdated overlay), redraws the image, and refreshes the
        metadata panel if present.
        """
        self._restore_ui_for_active()
        self._refresh_compare_list()
        self._refresh_compare_group_filter()
        self._update_phasor_display()
        # Segmentation masks are stale until Paint; show the fresh base map instead.
        self.chk_overlay.blockSignals(True)
        self.chk_overlay.setChecked(False)
        self.chk_overlay.blockSignals(False)
        self.refresh_image()
        if hasattr(self, "_update_metadata_panel"):
            self._update_metadata_panel()

    def _refresh_image_combo(self):
        """Refresh sample table, dropdown, and overlay filters.

        Convenience bundle called after loading new samples: rebuilds the
        compare table, the overlay group filter combo, the active-sample
        combo, and re-syncs table/combo selection — everything that lists
        datasets by name, in one call.
        """
        self._refresh_compare_list()
        self._refresh_compare_group_filter()
        self._refresh_sample_combo()
        self._sync_sample_table_selection()

    def _refresh_compare_group_filter(self):
        """Rebuild overlay group filter combo from dataset group names.

        Collects the distinct, non-empty ``group_name`` values across all
        datasets, sorts them, and repopulates ``cb_compare_group`` with an
        "All groups" entry followed by each group name, preserving the
        previous selection when it still exists (falling back to "All
        groups" otherwise). Called whenever datasets are loaded or a group
        name changes.
        """
        if not hasattr(self, "cb_compare_group"):
            return
        current = self.cb_compare_group.currentText()
        groups = sorted({
            (d.group_name or "").strip()
            for d in self.datasets
            if (d.group_name or "").strip()
        })
        self.cb_compare_group.blockSignals(True)
        self.cb_compare_group.clear()
        self.cb_compare_group.addItem("All groups")
        for g in groups:
            self.cb_compare_group.addItem(g)
        idx = self.cb_compare_group.findText(current)
        self.cb_compare_group.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_compare_group.blockSignals(False)

    def _apply_group_from_field(self):
        """Save the active sample's group name and refresh overlay legend.

        Connected to ``edit_group.editingFinished`` and the adjacent Set
        button. Writes the trimmed field text as ``group_name`` on the active
        dataset (or the corresponding entry in ``self.datasets`` when in
        multi-image mode), then rebuilds the compare table and the group
        filter combo so the new/changed group is reflected everywhere, and
        refreshes just the phasor legend (not a full redraw) if the sample is
        already calibrated.
        """
        text = self.edit_group.text().strip() if hasattr(self, "edit_group") else ""
        if self.chk_multi.isChecked() and 0 <= self.active_idx < len(self.datasets):
            self.datasets[self.active_idx].group_name = text
        else:
            self.data.group_name = text
        self._refresh_compare_list()
        self._refresh_compare_group_filter()
        if self.data.real_cal is not None:
            self._update_phasor_legend_only()
        if text:
            self._log(f"Group set to “{text}” for active sample.")

    def _apply_display_name_from_field(self):
        """Save the active sample's legend display name and update the compare table.

        Connected to ``edit_display_name.editingFinished`` and its Set button.
        Writes the trimmed field text as ``display_name`` on the active
        dataset, patches just that row's Sample cell in the compare table
        (rather than a full rebuild), refreshes the sample combo label, the
        "Active: …" label, and the phasor legend. Clearing the field back to
        empty restores the placeholder to a filename-derived hint via
        :func:`~flim_phasors.utils._dataset_file_label` so the field shows
        what name would be used by default.
        """
        text = self.edit_display_name.text().strip() if hasattr(self, "edit_display_name") else ""
        d = self.data
        row = self.active_idx
        if self.chk_multi.isChecked() and 0 <= self.active_idx < len(self.datasets):
            d = self.datasets[self.active_idx]
            row = self.active_idx
        d.display_name = text
        self._patch_sample_table_name(row)
        self._refresh_sample_combo()
        self._update_active_sample_label()
        self._update_phasor_legend_only()
        if text:
            self._log(f"Display name set to “{text}”.")
        elif hasattr(self, "edit_display_name"):
            file_hint = _dataset_file_label(d, row)
            self.edit_display_name.setPlaceholderText(file_hint)

    def _patch_sample_table_name(self, row: int):
        """Update one compare-table Sample cell after a display-name change.

        Cheaper than :meth:`_refresh_compare_list` for the common case of
        renaming just the active sample: updates or creates the row's Sample
        item text and refreshes its tooltip (showing the underlying file path
        when a custom display name is set, or a "double-click to rename" hint
        otherwise). Does nothing if ``row`` is out of range for either the
        dataset list or the table.

        Args:
            row: Dataset/table row index whose Sample cell should be patched.
        """
        if not (0 <= row < len(self.datasets) and row < self.table_compare.rowCount()):
            return
        d = self.datasets[row]
        self.table_compare.blockSignals(True)
        item = self.table_compare.item(row, 2)
        if item is None:
            item = self._editable_sample(dataset_short_label(d, row))
            self.table_compare.setItem(row, 2, item)
        else:
            item.setText(dataset_short_label(d, row))
        file_hint = _dataset_file_label(d, row)
        if (d.display_name or "").strip():
            item.setToolTip(f"File: {file_hint}")
        else:
            item.setToolTip("Double-click to rename for the phasor legend")
        self.table_compare.blockSignals(False)

    def _update_active_sample_label(self):
        """Refresh the Active label on the Multi-phasor tab.

        Updates ``lbl_editing`` text to the active dataset's display label,
        but only when the compare table/strip is actually visible (multi-image
        mode on with 2+ samples) — otherwise :meth:`_update_multi_strip` owns
        that label's text (e.g. showing a "load another sample" hint).
        """
        if not hasattr(self, "lbl_editing"):
            return
        if self.chk_multi.isChecked() and len(self.datasets) > 1 and 0 <= self.active_idx < len(self.datasets):
            self.lbl_editing.setText(
                f"Active: {dataset_display_label(self.data, self.active_idx)}")

    def _legend_include_group(self) -> bool:
        """Return whether phasor legend labels should include group names.

        Reflects the "Legend" format combo on the Multi-phasor tab; any option
        other than plain "Sample name" is treated as wanting the group
        prefixed/appended in the legend. Defaults to True (include group) if
        the combo does not exist yet.

        Returns:
            True if group names should be part of legend labels.
        """
        if not hasattr(self, "cb_legend_format"):
            return True
        return self.cb_legend_format.currentText() != "Sample name"

    def _legend_loc(self) -> str:
        """Return the matplotlib legend location key for the phasor plot.

        Reads ``cb_legend_loc`` and validates its text against
        ``LEGEND_LOC_ITEMS`` so a stale or unexpected combo value can never
        produce an invalid matplotlib ``loc`` argument.

        Returns:
            A valid matplotlib legend location string, defaulting to
            ``"upper right"``.
        """
        if not hasattr(self, "cb_legend_loc"):
            return "upper right"
        loc = self.cb_legend_loc.currentText().strip()
        return loc if loc in LEGEND_LOC_ITEMS else "upper right"

    def _legend_fontsize(self) -> float:
        """Return legend font size for multi-image phasor overlay.

        Reads ``sp_legend_size``, falling back to ``LEGEND_SIZE_DEFAULT``
        before that control exists.

        Returns:
            Legend text and marker size, in points.
        """
        if not hasattr(self, "sp_legend_size"):
            return float(LEGEND_SIZE_DEFAULT)
        return float(self.sp_legend_size.value())

    def _update_frame_control(self):
        """Enable frame spinbox when the active sample has a time stack.

        Determines the number of time frames from ``d.n_frames`` (metadata
        recorded at load, since the T dimension is usually summed away during
        decode) or, if a raw signal is still resident, from its "T" dimension
        size. When there is more than one frame, enables ``sp_frame`` with
        range ``[-1, n - 1]`` (``-1`` meaning "sum all frames") and restores
        the dataset's stored frame index; otherwise disables it and resets to
        the "all" special value.
        """
        d = self.data
        self.sp_frame.blockSignals(True)
        n = int(getattr(d, "n_frames", 1) or 1)
        # Prefer metadata stored at load — T is usually summed away during decode.
        if d.signal_full is not None and "T" in d.signal_full.dims:
            n = max(n, int(d.signal_full.sizes.get("T", 1)))
        if dataset_has_sample(d) and n > 1:
            self.sp_frame.setEnabled(True)
            self.sp_frame.setRange(-1, n - 1)
            self.sp_frame.setValue(int(getattr(d, "frame_index", -1)))
        else:
            self.sp_frame.setEnabled(False)
            self.sp_frame.setRange(-1, 0)
            self.sp_frame.setValue(-1)
        self.sp_frame.blockSignals(False)

    def on_frame_change(self, value):
        """Reload and reprocess the active sample for a new time-frame index.

        Connected to ``sp_frame.valueChanged``. A frame change requires
        re-decoding the raw file (the previously loaded histogram was already
        summed/selected to a specific frame), so this stores the new frame
        index on the dataset and delegates to :meth:`_reload_active_sample`.
        No-op if no sample is loaded or the value did not actually change.

        Args:
            value: New frame index (``-1`` means sum all frames).
        """
        if self.data.sample_path:
            if int(value) == int(getattr(self.data, "frame_index", -1)):
                return
            self.data.frame_index = int(value)
            self._reload_active_sample()

    def _reload_active_sample(self):
        """Decode the active sample at the current frame and rerun Apply.

        Only applicable to histogram-based samples (LIF phasor series have no
        frame concept and are skipped). Re-decodes the file at the dataset's
        stored ``frame_index`` and channel (honoring Fast load), off the GUI
        thread via :meth:`_run_busy`; on failure logs and shows an error
        dialog and leaves the previous decode untouched. On success, clamps
        the channel to the (possibly changed) channel count and calls
        :meth:`apply_processing` to rebuild phasor maps for the new frame.
        """
        if self.data.load_source == "lif_phasor":
            return
        path = self.data.sample_path
        if not path or not os.path.isfile(path):
            return
        ch = self.data.channel
        frame = self.data.frame_index
        lc = ch if self.chk_fast_load.isChecked() else None
        try:
            self._run_busy(
                f"Reloading frame {frame} ({os.path.basename(path)})…",
                lambda: self.data.load_sample(path, frame=frame, load_channel=lc),
            )
        except Exception as e:
            self._log(f"Reload error: {e}")
            QtWidgets.QMessageBox.critical(self, "Reload error", str(e))
            return
        self.data.channel = min(ch, max(0, self.data.n_channels - 1))
        self.apply_processing()

    def _compare_show_item(self, row):
        """Return the Show checkbox item for a compare-table row.

        The Show column (column 0) holds a checkable ``QTableWidgetItem``
        whose ``UserRole`` data is the dataset index; other compare-table
        helpers use this accessor instead of indexing the table directly.

        Args:
            row: Row index in ``table_compare``.

        Returns:
            The ``QTableWidgetItem`` for that row's Show checkbox, or
            ``None`` if the row does not exist.
        """
        return self.table_compare.item(row, 0)

    def _compare_dataset_index(self, row):
        """Return dataset index stored in a compare-table row's Show item.

        Each row's Show checkbox item carries the index into
        ``self.datasets`` as its ``UserRole`` data, set when the table is
        rebuilt in ``_refresh_compare_list``. Used to map table rows back to
        datasets when checkboxes are toggled or rows are selected.

        Args:
            row: Row index in ``table_compare``.

        Returns:
            The dataset index for that row, or -1 if the row or Show item is
            missing.
        """
        it = self._compare_show_item(row)
        if it is None:
            return -1
        return int(it.data(Qt.ItemDataRole.UserRole))

    def _refresh_compare_list(self):
        """Rebuild the multi-image compare table from loaded datasets.

        Repopulates every row of ``table_compare`` (Show, #, Sample, Group,
        Filter, Min N, Reference, Status) from ``self.datasets``, preserving
        the user's Show checkbox state for samples that were already
        calibrated and applied (newly pending rows always start unchecked).
        Table signals are blocked during the rebuild to avoid re-entrant
        ``itemChanged`` callbacks. Called whenever samples are added,
        removed, applied, or renamed while multi-image mode is active.
        """
        checked = {}
        was_ready = {}
        for row in range(self.table_compare.rowCount()):
            idx = self._compare_dataset_index(row)
            it = self._compare_show_item(row)
            if idx >= 0 and it is not None:
                checked[idx] = it.checkState() == Qt.CheckState.Checked
            status_it = self.table_compare.item(row, 7)
            if idx >= 0:
                was_ready[idx] = status_it is not None and status_it.text() != "pending"

        self.table_compare.blockSignals(True)
        self.table_compare.setRowCount(len(self.datasets))
        for i, d in enumerate(self.datasets):
            ready = d.real_cal is not None
            show = QtWidgets.QTableWidgetItem()
            show.setData(Qt.ItemDataRole.UserRole, i)
            show.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if ready:
                show.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                # Preserve Show ticks only when the row was already applied (not newly "pending").
                if i in checked and was_ready.get(i):
                    show_checked = checked[i]
                else:
                    show_checked = False
                show.setCheckState(
                    Qt.CheckState.Checked if show_checked else Qt.CheckState.Unchecked)
            else:
                show.setFlags(Qt.ItemFlag.ItemIsEnabled)
                show.setCheckState(Qt.CheckState.Unchecked)
                show.setToolTip("Run Apply on this image first")

            num = self._ro(str(i + 1))
            num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            sample = self._editable_sample(dataset_short_label(d, i))
            if i == self.active_idx:
                sample.setFont(QtGui.QFont(sample.font().family(), sample.font().pointSize(),
                                           QtGui.QFont.Weight.Bold))
                sample.setToolTip("Selected — settings below apply to this sample")
            else:
                file_hint = _dataset_file_label(d, i)
                if (d.display_name or "").strip():
                    sample.setToolTip(f"File: {file_hint}")
                else:
                    sample.setToolTip("Double-click to rename for the phasor legend")
            group = self._editable_group((d.group_name or "").strip())
            stash = getattr(d, "processing_settings", None) or {}
            filt = self._ro(
                str(stash.get("filter_mode", filter_label_for_dataset(self, d))))
            min_n = self._ro(str(int(stash.get("intensity_min", 0))))
            ref_lbl = self._ro(
                self._effective_ref_label(d) if hasattr(self, "_effective_ref_label")
                else "—")
            if not ready:
                status = self._ro("pending")
                status.setForeground(QtGui.QBrush(Qt.GlobalColor.gray))
            elif getattr(d, "maps_calibrated", False):
                status = self._ro("calibrated")
            else:
                status = self._ro("preview")
                status.setToolTip("Uncalibrated preview — Calibrate + Apply to correct")

            self.table_compare.setItem(i, 0, show)
            self.table_compare.setItem(i, 1, num)
            self.table_compare.setItem(i, 2, sample)
            self.table_compare.setItem(i, 3, group)
            self.table_compare.setItem(i, 4, filt)
            self.table_compare.setItem(i, 5, min_n)
            self.table_compare.setItem(i, 6, ref_lbl)
            self.table_compare.setItem(i, 7, status)
        self.table_compare.blockSignals(False)
        self._set_compare_controls_enabled(
            self.chk_multi.isChecked() and len(self.datasets) >= 2)
        if hasattr(self, "_update_apply_buttons"):
            self._update_apply_buttons()

    def _ensure_compare_overlay_off(self, *, update_display: bool = True):
        """Keep multi-image phasor overlay off until the user enables it.

        Called after Apply/calibration changes the active sample's phasor
        map, since a freshly recomputed map should not be silently stacked
        into a stale multi-image overlay. If the overlay checkbox is
        currently checked, it is unchecked (with signals blocked to avoid a
        recursive callback) and the compare controls' enabled state is
        refreshed.

        Args:
            update_display: If True, redraw the phasor plot after turning
                the overlay off.
        """
        # After Apply/calibration the active sample's map changed — don't auto-stack overlays.
        if not hasattr(self, "chk_compare"):
            return
        if self.chk_compare.isChecked():
            self.chk_compare.blockSignals(True)
            self.chk_compare.setChecked(False)
            self.chk_compare.blockSignals(False)
        self._set_compare_controls_enabled(
            self.chk_multi.isChecked() and len(self.datasets) >= 2)
        if update_display:
            self._update_phasor_display()

    def _set_compare_controls_enabled(self, compare_available):
        """Enable overlay style, legend, and table controls when compare is available.

        Enables/disables the "Show compare" checkbox and, when compare is
        both available and currently checked, the overlay style combo,
        legend format/position/size controls, and the per-row selection
        buttons. The compare table itself stays enabled whenever more than
        one sample is loaded, independent of whether the overlay is shown.
        Also refreshes the multi-image thumbnail strip.

        Args:
            compare_available: Whether multi-image mode is on and at least
                two samples are loaded.
        """
        cmp_on = compare_available and self.chk_compare.isChecked()
        self.chk_compare.setEnabled(compare_available)
        self.cb_compare_style.setEnabled(cmp_on)
        if hasattr(self, "cb_legend_format"):
            self.cb_legend_format.setEnabled(cmp_on)
        if hasattr(self, "cb_legend_loc"):
            self.cb_legend_loc.setEnabled(cmp_on)
        if hasattr(self, "sp_legend_size"):
            self.sp_legend_size.setEnabled(cmp_on)
        if hasattr(self, "table_compare"):
            self.table_compare.setEnabled(len(self.datasets) > 1)
        for btn in getattr(self, "_compare_sel_buttons", ()):
            btn.setEnabled(cmp_on)
        self._update_multi_strip()

    def _compare_set_all_checks(self, checked):
        """Check or uncheck all Show boxes in the compare table.

        Skips rows whose Show item is not user-checkable (samples that have
        not been applied yet). Table signals are blocked while the
        checkboxes are updated so each change does not trigger its own
        ``itemChanged`` handler, then the phasor overlay is redrawn once
        with the final selection.

        Args:
            checked: True to check every eligible row, False to uncheck all.
        """
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.table_compare.blockSignals(True)
        for row in range(self.table_compare.rowCount()):
            it = self._compare_show_item(row)
            if it is None or not (it.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            it.setCheckState(state)
        self.table_compare.blockSignals(False)
        self._update_phasor_display()

    def _compare_select_all(self):
        """Show every preprocessed sample on the phasor overlay.

        Convenience wrapper around ``_compare_set_all_checks(True)`` bound to
        the compare tab's "Select all" button.
        """
        self._compare_set_all_checks(True)

    def _compare_select_none(self):
        """Hide all samples from the phasor overlay.

        Convenience wrapper around ``_compare_set_all_checks(False)`` bound
        to the compare tab's "Select none" button.
        """
        self._compare_set_all_checks(False)

    def _on_compare_table_changed(self, row, column):
        """Handle inline edits to sample name, group, or Show checkbox.

        Dispatches based on which compare-table column changed: editing the
        Sample column (2) updates the dataset's display name (and the
        Setup tab's name field, if the row is active), editing the Group
        column (3) updates the group name and the group-filter combo, and
        toggling the Show column (0) delegates to ``_on_compare_ui_changed``
        to redraw the overlay. Other columns are ignored.

        Args:
            row: Row index of the edited cell.
            column: Column index of the edited cell.
        """
        if column == 2:
            if 0 <= row < len(self.datasets):
                item = self.table_compare.item(row, 2)
                text = item.text().strip() if item else ""
                self.datasets[row].display_name = text
                if row == self.active_idx and hasattr(self, "edit_display_name"):
                    self.edit_display_name.blockSignals(True)
                    self.edit_display_name.setText(text)
                    self.edit_display_name.setPlaceholderText(
                        _dataset_file_label(self.datasets[row], row))
                    self.edit_display_name.blockSignals(False)
                self._refresh_sample_combo()
                self._update_active_sample_label()
                self._update_phasor_legend_only()
            return
        if column == 3:
            if 0 <= row < len(self.datasets):
                item = self.table_compare.item(row, 3)
                text = item.text().strip() if item else ""
                self.datasets[row].group_name = text
                if row == self.active_idx and hasattr(self, "edit_group"):
                    self.edit_group.blockSignals(True)
                    self.edit_group.setText(text)
                    self.edit_group.blockSignals(False)
                self._refresh_compare_group_filter()
                self._update_phasor_legend_only()
            return
        if column != 0:
            return
        self._on_compare_ui_changed()

    def _on_sample_table_selection(self):
        """Activate the dataset for the selected compare-table row.

        Connected to the compare table's selection-changed signal. Ignored
        while ``_table_sel_lock`` is set (e.g. during a programmatic
        rebuild) or when there is at most one sample, and is a no-op if the
        selection already matches the active dataset.
        """
        if self._table_sel_lock or len(self.datasets) <= 1:
            return
        sel = self.table_compare.selectionModel().selectedRows()
        if not sel:
            return
        row = sel[0].row()
        if row == self.active_idx:
            return
        self._activate_dataset(row)

    def _build_compare_layers(self):
        """Build phasor overlay layer descriptors from checked compare-table rows.

        Iterates every compare-table row, resolves it to a dataset, and
        (optionally) filters by the group selected in the group-filter
        combo. Each surviving row becomes a layer dict with the dataset,
        legend label, categorical color, visibility (from the Show
        checkbox), and dataset index, consumed by the phasor canvas overlay
        renderer.

        Returns:
            List of layer dicts with keys ``data``, ``label``, ``color``,
            ``visible``, and ``index``.
        """
        group_sel = (
            self.cb_compare_group.currentText()
            if hasattr(self, "cb_compare_group") else "All groups"
        )
        layers = []
        for row in range(self.table_compare.rowCount()):
            idx = self._compare_dataset_index(row)
            if idx < 0 or idx >= len(self.datasets):
                continue
            show = self._compare_show_item(row)
            d = self.datasets[idx]
            gname = (d.group_name or "").strip()
            if group_sel != "All groups" and gname != group_sel:
                continue
            layers.append({
                "data": d,
                "label": dataset_phasor_legend_label(
                    d, idx, include_group=self._legend_include_group()),
                "color": categorical_rgb(idx),
                "visible": show is not None and show.checkState() == Qt.CheckState.Checked,
                "index": idx,
            })
        return layers

    def _compare_style_key(self):
        """Return the phasor canvas style key for the selected overlay mode.

        Translates the user-facing overlay style text (e.g. "Point cloud")
        into the internal style key (e.g. ``"cloud"``) expected by
        ``PhasorCanvas.update_display``, defaulting to ``"cloud"`` if the
        combo text is unrecognized.

        Returns:
            The internal compare-style key string.
        """
        return COMPARE_STYLE_MAP.get(self.cb_compare_style.currentText(), "cloud")

    def _compare_overlay_active(self):
        """Return whether multi-image overlay is on and its layer list.

        Builds the full layer list via ``_build_compare_layers`` and
        computes which layers are both checked visible and backed by a
        calibrated dataset. The overlay is considered active only when
        multi-image mode and the compare checkbox are both on and at least
        one such visible layer exists.

        Returns:
            Tuple ``(compare_on, layers, visible)`` where ``compare_on`` is a
            bool, ``layers`` is the full layer list, and ``visible`` is the
            subset of layers actually shown on the plot.
        """
        layers = self._build_compare_layers()
        visible = [L for L in layers if L.get("visible") and L["data"].real_cal is not None]
        compare_on = (
            self.chk_multi.isChecked()
            and self.chk_compare.isChecked()
            and len(visible) >= 1
        )
        return compare_on, layers, visible

    def _update_phasor_display(self, status_note=""):
        """Redraw phasor plot with optional multi-image overlay and legend.

        Recomputes the active overlay state and delegates the full redraw to
        ``PhasorCanvas.update_display``. When two or more overlay layers are
        visible, checks whether their working frequency or harmonic differ
        (positions would then not be directly comparable) and logs a
        warning in that case instead of the caller's status note. This is
        the main entry point for refreshing the phasor plot after Apply,
        calibration, cursor edits, or overlay UI changes.

        Args:
            status_note: Optional message to log when no frequency/harmonic
                mismatch warning is needed.
        """
        compare_on, layers, visible = self._compare_overlay_active()
        self.phasor.update_display(
            self.data,
            compare_enabled=compare_on,
            compare_style=self._compare_style_key(),
            compare_layers=layers if compare_on else None,
            legend_loc=self._legend_loc(),
            legend_fontsize=self._legend_fontsize(),
        )
        if compare_on and len(visible) >= 2:
            freqs = {round(L["data"].work_frequency, 4) for L in visible}
            harms = {L["data"].harmonic for L in visible}
            if len(freqs) > 1 or len(harms) > 1:
                self._log(
                    "Compare overlay: working frequency or harmonic differs between images — "
                    "positions may not be directly comparable.")
            elif status_note:
                self._log(status_note)

    def _update_phasor_legend_only(self):
        """Refresh overlay legend without redrawing phasor point clouds.

        Cheaper than ``_update_phasor_display`` for cases where only labels
        changed (e.g. renaming a sample or group) — skips replotting the
        scatter/density layers and just calls
        ``PhasorCanvas.update_compare_legend``. No-op unless the overlay is
        active with at least two visible layers.
        """
        compare_on, layers, visible = self._compare_overlay_active()
        if not compare_on or len(visible) < 2:
            return
        self.phasor.update_compare_legend(
            layers,
            compare_style=self._compare_style_key(),
            legend_loc=self._legend_loc(),
            legend_fontsize=self._legend_fontsize(),
        )

    def _on_legend_ui_changed(self, *_args):
        """React to legend format, position, or size changes on the overlay.

        Connected to the legend format combo, position combo, and font-size
        spinbox. Refreshes which compare controls are enabled and repaints
        just the legend (not the full overlay) to reflect the new settings.

        Args:
            *_args: Ignored Qt signal arguments from the triggering widget.
        """
        self._set_compare_controls_enabled(
            self.chk_multi.isChecked() and len(self.datasets) >= 2)
        self._update_phasor_legend_only()

    def _on_compare_ui_changed(self, *_args):
        """React to overlay toggle, style, or group-filter changes.

        Connected to the "Show compare" checkbox, overlay style combo, and
        group-filter combo. Refreshes which compare controls are enabled and
        triggers a full phasor redraw when there is already a phasor to show
        or the compare checkbox is checked.

        Args:
            *_args: Ignored Qt signal arguments from the triggering widget.
        """
        self._set_compare_controls_enabled(
            self.chk_multi.isChecked() and len(self.datasets) >= 2)
        if self.data.real_cal is not None or self.chk_compare.isChecked():
            self._update_phasor_display()

    def on_multi_toggle(self, checked):
        """Enter or leave multi-image mode and refresh compare UI.

        On enabling, adopts the currently loaded sample as the first slot in
        ``self.datasets`` (if not already present), ensures every dataset
        has initialized processing settings, and refreshes the sample combo
        and thumbnail strip. On disabling, switches back to the Setup tab,
        forces the compare overlay off, and redraws the single-image phasor
        plot. Either way, updates which Apply buttons are shown and logs a
        status message.

        Args:
            checked: True when multi-image mode was just enabled, False when
                disabled.
        """
        self._set_multi_detail_enabled(checked)
        if checked:
            # adopt the currently loaded image as the first slot (non-destructive)
            has_image = dataset_has_sample(self.data)
            if has_image and self.data not in self.datasets:
                self.datasets.append(self.data)
                self.active_idx = len(self.datasets) - 1
            for d in self.datasets:
                self._init_dataset_proc_settings(d)
            self._refresh_image_combo()
            self._update_multi_strip()
            self._log(
                "Multi-image mode — sample table on Multi-phasor tab; "
                "edit filters on Setup, then Apply selected or Apply settings to all.")
        else:
            self._update_multi_strip()
            if hasattr(self, "panel_tabs"):
                self.panel_tabs.setCurrentIndex(self._tab_setup_idx)
            self.chk_compare.blockSignals(True)
            self.chk_compare.setChecked(False)
            self.chk_compare.blockSignals(False)
            self._set_compare_controls_enabled(False)
            self._update_phasor_display()
            self._log("Multi-image mode off (current image stays active).")
        self._update_apply_buttons()

    def _activate_dataset(self, idx: int):
        """Switch active sample, restore its UI settings, and refresh plots.

        Saves the outgoing sample's per-sample processing settings and
        segmentation (cursors/GMM fit/overlay) before swapping, since each
        dataset keeps its own state. After making ``idx`` active, restores
        its UI controls, compare table, thumbnail strip, phasor plot, and
        stashed segmentation. If cursor live-update is on and the new sample
        has cursors but no stored cluster stats yet, recomputes the cursor
        segmentation immediately. Called from compare-table row selection
        and the sample combo box.

        Args:
            idx: Index into ``self.datasets`` to activate.
        """
        if not (0 <= idx < len(self.datasets)):
            return
        if per_sample_processing(self):
            self._save_proc_from_ui(self.data)
        # Each sample keeps its own cursors/GMM fit/overlay — stash before swapping active_idx.
        if 0 <= self.active_idx < len(self.datasets):
            self._stash_segmentation_to_dataset(self.datasets[self.active_idx])
        self.active_idx = idx
        self.data = self.datasets[idx]
        self._restore_ui_for_active()
        self._refresh_compare_list()
        self._update_multi_strip()
        self._update_phasor_display()
        self._restore_segmentation_from_dataset(self.data)
        # Live cursor refresh only when this sample has no stored segmentation yet
        if (
            self.chk_live.isChecked()
            and self.mode == "cursor"
            and self.phasor.cursors
            and self.data.real_cal is not None
            and not getattr(self.data, "cluster_stats", None)
        ):
            self._compute_cursor()
        self._log(f"Selected sample {idx + 1}: {dataset_display_label(self.data, idx)}")
        self.activateWindow()
        self.raise_()

    def _stash_segmentation_to_dataset(self, d=None):
        """Remember GMM fit, cluster table, and overlay on a sample dataset.

        Copies the currently displayed segmentation state — GMM ellipse
        parameters, cluster statistics rows, and the last painted
        pseudo-color overlay — onto the dataset object so it survives
        switching to another sample. Called before deactivating a sample
        (e.g. in ``_activate_dataset`` or before batch operations that
        temporarily swap ``self.data``).

        Args:
            d: Dataset to stash onto; defaults to the currently active
                ``self.data``.
        """
        d = self.data if d is None else d
        if d is None:
            return
        d.gmm_fit = getattr(self, "_gmm_fit", None)
        d.cluster_stats = [dict(st) for st in (self.cluster_stats or [])]
        if self.last_overlay is None:
            d.last_overlay = None
        else:
            d.last_overlay = np.asarray(self.last_overlay).copy()

    def _clear_segmentation_on_dataset(self, d):
        """Drop stored GMM / paint results for one sample (e.g. after Apply).

        Resets ``gmm_fit``, ``cluster_stats``, and ``last_overlay`` on the
        given dataset to their empty defaults, since recomputing the phasor
        maps invalidates any previously fitted clusters or overlays for that
        sample.

        Args:
            d: Dataset to clear; no-op if ``None``.
        """
        if d is None:
            return
        d.gmm_fit = None
        d.cluster_stats = []
        d.last_overlay = None

    def _restore_segmentation_from_dataset(self, d=None):
        """Restore GMM ellipses, table, and overlay from a sample's stash.

        Counterpart to ``_stash_segmentation_to_dataset``: reinstates the
        GMM fit (redrawing ellipses on the phasor canvas, or clearing them if
        the sample has none), the cluster statistics table, and the last
        pseudo-color overlay (re-enabling the overlay checkbox if one
        exists). Called when activating a dataset so its previous
        segmentation reappears instead of the previous sample's.

        Args:
            d: Dataset to restore from; defaults to the currently active
                ``self.data``.
        """
        d = self.data if d is None else d
        fit = getattr(d, "gmm_fit", None) if d is not None else None
        if fit is not None:
            self._gmm_fit = fit
            n = len(fit[0])
            colors = [categorical_rgb(k) for k in range(n)]
            self.phasor.show_gmm_ellipses(*fit, colors)
        else:
            self.phasor.clear_gmm()
            if hasattr(self, "_gmm_fit"):
                del self._gmm_fit
        self.cluster_stats = [dict(st) for st in (getattr(d, "cluster_stats", None) or [])]
        overlay = getattr(d, "last_overlay", None) if d is not None else None
        self.last_overlay = None if overlay is None else np.asarray(overlay).copy()
        self._fill_table()
        has_overlay = self.last_overlay is not None
        self.chk_overlay.blockSignals(True)
        self.chk_overlay.setChecked(has_overlay)
        self.chk_overlay.blockSignals(False)
        self.refresh_image()

    def apply_settings_to_all(self):
        """Copy active sample's processing settings to all datasets and Apply all.

        Requires at least two loaded samples (shows an info dialog and
        returns otherwise). Saves the active sample's current UI settings,
        snapshots them via ``capture_processing_from_ui``, and stamps that
        snapshot onto every dataset's ``processing_settings`` before
        preprocessing all of them via ``apply_processing(scope="all")``.
        This lets the user tune filters on one sample and broadcast them to
        the rest of the batch.
        """
        if len(self.datasets) <= 1:
            QtWidgets.QMessageBox.information(
                self, "Need multiple samples",
                "Load at least two samples in multi-image mode first.")
            return
        self._save_proc_from_ui(self.data)
        snap = capture_processing_from_ui(self)
        for d in self.datasets:
            d.processing_settings = dict(snap)
        self._refresh_compare_list()
        self._log(
            f"Copied settings from {dataset_short_label(self.data, self.active_idx)} "
            f"to all {len(self.datasets)} samples.")
        self.apply_processing(scope="all")

    def remove_image(self):
        """Remove the active sample from the multi-image list.

        Pops the active dataset out of ``self.datasets`` and logs which
        sample was removed. If the list becomes empty, clears the active
        index and refreshes the (now empty) sample combo. Otherwise clamps
        the active index into bounds and activates the resulting sample so
        the UI always has a valid selection.
        """
        if not (0 <= self.active_idx < len(self.datasets)):
            return
        name = dataset_short_label(self.data, self.active_idx)
        self.datasets.pop(self.active_idx)
        self._log(f"Removed sample: {name}")
        if not self.datasets:
            self.active_idx = -1
            self._refresh_image_combo()
            return
        self.active_idx = min(self.active_idx, len(self.datasets) - 1)
        self._refresh_image_combo()
        self._activate_dataset(self.active_idx)

    def on_channel_change(self, idx):
        """Switch sample channel and reprocess if maps already exist.

        Channel can be chosen before a sample is loaded (for Fast load). That
        preference is persisted. Once data is loaded, changing channel either
        reprocesses or (in fast-load mode) re-decodes the selected channel.
        """
        new_ch = max(0, idx)
        self._settings.setValue("preferred_sample_channel", new_ch)
        if self.data.signal_full is None:
            return
        if new_ch == self.data.channel:
            return
        self.data.channel = new_ch
        loaded = getattr(self.data, "fast_loaded_channel", None)
        if loaded is not None and new_ch != loaded:
            self._reload_sample_channel(new_ch)
            return
        if self.data.real_cal is not None:
            self.apply_processing()
        else:
            self._log(f"Sample channel {self.data.channel} — click Apply to preprocess.")

    def _reload_sample_channel(self, new_ch):
        """Re-decode only ``new_ch`` for the active sample (fast-load mode).

        Used when fast-load is enabled and the user switches to a channel
        that was not already decoded. Reloads the sample file at the current
        frame with ``load_channel=new_ch``; on success, reapplies processing
        if maps already existed, otherwise recomputes the uncalibrated
        preview so the image view has something to show before Apply.

        Args:
            new_ch: Channel index to decode and switch to.
        """
        path = self.data.sample_path
        if not path or not os.path.isfile(path):
            return
        frame = self.data.frame_index
        had_maps = self.data.real_cal is not None
        try:
            self._run_busy(
                f"Loading channel {new_ch} ({os.path.basename(path)})…",
                lambda: self.data.load_sample(path, frame=frame, load_channel=new_ch),
            )
        except Exception as e:
            self._log(f"Channel load error: {e}")
            QtWidgets.QMessageBox.critical(self, "Channel load error", str(e))
            return
        if had_maps:
            self.apply_processing()
        else:
            try:
                self._process_uncalibrated_preview([self.data])
            except Exception:
                pass
            self.refresh_image()
            self._log(f"Channel {new_ch} decoded — click Apply to preprocess.")

    def _on_fast_load_toggled(self, checked):
        """Persist the fast-load preference and explain the trade-off.

        Saves the checkbox state to ``QSettings`` so it survives restarts,
        and logs a short explanation of the current mode: fast load decodes
        only the active channel (cheaper memory, but switching channels
        re-decodes the file), while full load keeps every channel in memory
        for instant switching.

        Args:
            checked: New state of the "Fast load" checkbox.
        """
        self._settings.setValue("fast_load", bool(checked))
        if checked:
            self._log(
                "Fast load on — only the selected channel is decoded "
                "(switching channel re-decodes the file).")
        else:
            self._log(
                "Fast load off — all channels are decoded and kept "
                "for instant channel switching.")

    def on_ref_channel_change(self, idx):
        """Change reference channel and invalidate stored g/s until Calibrate.

        Channel can be chosen before a reference file is loaded; that
        preference is persisted and used by the next Reference… auto-calibrate.
        When a shared reference is in use, updates the shared reference channel
        and propagates it to every dataset via ``_propagate_shared_reference``;
        otherwise updates only the active dataset's reference channel. If
        automatic (non-manual) calibration is in effect, marks the stored
        g/s as stale so it is recomputed on the next Apply.

        Args:
            idx: New reference channel index selected in the combo box.
        """
        ch = max(0, idx)
        self._settings.setValue("preferred_ref_channel", ch)
        if self.chk_shared_ref.isChecked():
            if self.shared_ref_path:
                ch = min(ch, max(0, self.shared_ref_n_channels - 1))
            self.shared_ref_channel = ch
            if self.shared_ref_path:
                self._propagate_shared_reference()
        else:
            if self.data.ref_path:
                ch = min(ch, max(0, self.data.ref_n_channels - 1))
            self.data.ref_channel = ch
        if not self._effective_ref_path(self.data):
            return
        if self._effective_ref_file_path() and not self.chk_manual_cal.isChecked():
            self.ref_calibration.values_ready = False
            self.ref_calibration._maps = None
            self._update_calibration_display()
            self._update_calibration_stale_style()
            self._log("Reference channel changed — g/s will be recomputed automatically on Apply.")
            return

    # ---- processing --------------------------------------------------------
    def _active_calibration(self):
        """Return the calibration object to apply, syncing manual fields first.

        If manual calibration is enabled, first pushes the manual g/s spin
        box values into ``ref_calibration`` via
        ``_apply_manual_calibration_fields`` so the returned object reflects
        the latest UI entry.

        Returns:
            ``self.ref_calibration`` if it is active (has usable g/s
            reference values), otherwise ``None``.
        """
        if self.chk_manual_cal.isChecked():
            self._apply_manual_calibration_fields()
        return self.ref_calibration if self.ref_calibration.is_active else None

    def _process_uncalibrated_preview(self, datasets):
        """Compute quick uncalibrated phasor maps after sample load.

        Runs preprocessing with ``calibrate=False`` on every dataset that
        has a loaded sample, skipping ``lif_phasor`` sources (which already
        carry precomputed phasor data). Used right after loading a file so
        the phasor plot and image view show a preview before the user sets
        up a reference and clicks Apply.

        Args:
            datasets: Iterable of dataset objects to preview.
        """
        for d in datasets:
            if not dataset_has_sample(d):
                continue
            if d.load_source == "lif_phasor":
                continue
            run_processing_on_dataset(self, d, use_ui_settings=True, calibrate=False)

    def _run_processing_on_dataset(self, d, *, use_ui_settings=False, calibrate=True):
        """Delegate phasor preprocessing for one dataset to processing module.

        Thin wrapper around the module-level ``run_processing_on_dataset``
        helper, passing ``self`` so it can read filter/threshold widgets and
        the active calibration.

        Args:
            d: Dataset to preprocess.
            use_ui_settings: If True, read filter/threshold settings from the
                Setup tab widgets instead of the dataset's stashed settings.
            calibrate: If True, apply the active reference calibration to the
                computed phasor coordinates.
        """
        run_processing_on_dataset(
            self, d, use_ui_settings=use_ui_settings, calibrate=calibrate)

    def apply_processing(self, scope: str = "auto"):
        """Preprocess active or all samples with current calibration and filters.

        The main "Apply" action. Ensures a sample is loaded, saves the
        active sample's UI settings (and, for ``scope="all"``, broadcasts
        them to every dataset) when per-sample processing is enabled, then
        auto-calibrates the reference if it is not already ready (decoding
        the reference file and computing g/s without a separate Calibrate
        click). Runs the actual phasor computation inside a busy dialog,
        clears any now-stale GMM/cursor segmentation for the affected
        dataset(s), refreshes the phasor and image views, and logs a summary
        line with calibration, filter, harmonic, and valid-pixel counts.
        Processing and calibration errors are caught and shown as message
        boxes rather than propagated to the caller.

        Args:
            scope: ``"auto"`` (defaults to the active sample), ``"active"``
                to process only the active sample, or ``"all"`` to process
                every loaded sample in multi-image mode.
        """
        if not dataset_has_sample(self.data):
            QtWidgets.QMessageBox.information(self, "No data", "Load a sample first.")
            return
        multi = len(self.datasets) > 1
        if scope == "auto":
            scope = "active"
        if per_sample_processing(self):
            self._save_proc_from_ui(self.data)
            if scope == "all":
                snap = capture_processing_from_ui(self)
                for d in self.datasets:
                    d.processing_settings = dict(snap)
        self.data.pixel_size_um = self.sp_pixel_um.value() if hasattr(self, "sp_pixel_um") else 0.0
        if not self._calibration_ready_for_apply():
            # Auto-calibrate: decode the reference once and compute g/s, then
            # continue — no separate Calibrate click required.
            if not self._auto_calibrate_reference() or not self._calibration_ready_for_apply():
                QtWidgets.QMessageBox.information(
                    self,
                    "Calibration",
                    "Set a reference first: choose a reference file, or enable "
                    "Manual ref phasor and click Set g/s.",
                )
                return
        try:
            if scope == "all" and multi:
                _, elapsed = self._run_busy(
                    "Preprocessing all samples…",
                    lambda: self._process_all_loaded_datasets(use_ui_settings=True),
                )
            else:
                _, elapsed = self._run_busy(
                    "Preprocessing…",
                    lambda: self._run_processing_on_dataset(self.data, use_ui_settings=True),
                )
        except CancelledError:
            return
        except Exception as e:
            self._log(f"Processing error: {e}")
            QtWidgets.QMessageBox.critical(self, "Processing error", repr(e))
            return
        # Maps changed — drop stale per-sample GMM / paint results
        if scope == "all" and multi:
            for d in self.datasets:
                self._clear_segmentation_on_dataset(d)
            if hasattr(self, "_gmm_fit"):
                del self._gmm_fit
            self.last_overlay = None
            self.cluster_stats = []
            self.phasor.clear_gmm()
        else:
            self._clear_segmentation_on_dataset(self.data)
            if hasattr(self, "_gmm_fit"):
                del self._gmm_fit
            self.last_overlay = None
            self.cluster_stats = []
            self.phasor.clear_gmm()
        self._refresh_views_after_processing()
        if self.ref_calibration.is_active:
            ch = self._ref_channel_for_dataset(self.data)
            mode = "manual" if self.ref_calibration.use_manual else "file"
            ref_note = f", cal ({mode}) ch {ch} g={self.ref_calibration.mean_g:.3f} s={self.ref_calibration.mean_s:.3f}"
        else:
            ref_note = ""
        st = getattr(self.data, "_intensity_stats", {}) or {}
        if st and "min" in st and "max" in st:
            self.lbl_photon_range.setText(
                f"Image photon counts: {st['min']:.0f} – {st['max']:.0f} "
                f"(median {st.get('median', 0):.0f})")
            n_below = int(round(st.get("masked_pct", 0) * st.get("n_pixels", 0) / 100.0))
            int_msg = (f" | min photons ≥ {st.get('threshold', 0):.0f} "
                       f"({n_below} px removed)")
        else:
            int_msg = ""
        n_valid = int(self.data.valid_mask().sum()) if self.data.real_cal is not None else 0
        filt = filter_label_for_dataset(self, self.data)
        scope_note = " (all samples)" if scope == "all" and multi else ""
        self._log(
            f"Phasor recomputed{scope_note} — sample ch {self.data.channel}{ref_note}, "
            f"filter={filt}, H={self.data.harmonic}{int_msg}, "
            f"{n_valid} valid px ({self._fmt_elapsed(elapsed)})")
        self._warn_if_empty_phasor(n_valid, st)

    # ---- cursor actions ----------------------------------------------------
    def _refresh_active_cursor_combo(self, select_idx=None):
        """Rebuild Move combo from phasor cursors and select the given index.

        Clears and repopulates ``cb_active_cursor`` with one entry per cursor
        on the phasor canvas (using its label or a categorical fallback
        name), disabling the combo when there are no cursors. Signals are
        blocked while rebuilding to avoid re-entrant selection callbacks.
        Called whenever cursors are added, removed, or the selection
        changes.

        Args:
            select_idx: Cursor index to select after rebuilding; defaults to
                the phasor canvas's currently selected cursor.
        """
        if not hasattr(self, "cb_active_cursor"):
            return
        idx = self.phasor.selected if select_idx is None else select_idx
        self.cb_active_cursor.blockSignals(True)
        self.cb_active_cursor.clear()
        for i, c in enumerate(self.phasor.cursors):
            label = c.get("label") or categorical_name(i)
            self.cb_active_cursor.addItem(label)
        enabled = len(self.phasor.cursors) > 0
        self.cb_active_cursor.setEnabled(enabled)
        if enabled and 0 <= idx < len(self.phasor.cursors):
            self.cb_active_cursor.setCurrentIndex(idx)
        self.cb_active_cursor.blockSignals(False)

    def _on_phasor_cursor_selected(self, idx):
        """Sync Move combo and radius controls when a cursor is clicked.

        Connected to the phasor canvas's cursor-selection signal. Keeps the
        Move combo box and the radius slider/spinbox in sync with whichever
        cursor the user just clicked directly on the plot.

        Args:
            idx: Index of the cursor that was clicked/selected.
        """
        self._refresh_active_cursor_combo(select_idx=idx)
        self._sync_radius_slider()

    def on_active_cursor_change(self, combo_idx):
        """Select the cursor index chosen in the Move combo box.

        Ignored outside cursor mode or for an invalid combo index. Tells the
        phasor canvas to select the corresponding cursor (without
        re-emitting a selection signal, since the combo itself triggered
        this) and syncs the radius slider/spinbox to match.

        Args:
            combo_idx: Index selected in the Move combo box.
        """
        if self.mode != "cursor" or combo_idx < 0:
            return
        self.phasor.select_cursor(combo_idx, emit=False)
        self._sync_radius_slider()

    def add_cursor(self):
        """Add a circle or ellipse ROI at the current radius on the phasor plot.

        Pushes an undo snapshot first (if undo support is present), then
        adds a new cursor to the phasor canvas using the radius spinbox
        value, the shape selected in the cursor-shape combo (circle or
        ellipse), and — for ellipses — a minor radius derived from the
        aspect-ratio slider. Refreshes the Move combo so the new cursor is
        selectable.
        """
        if hasattr(self, "_push_cursor_undo"):
            self._push_cursor_undo()
        r = self.sp_radius.value()
        kind = "ellipse" if self.cb_cursor_shape.currentText() == "Ellipse" else "circle"
        aspect = self.sld_aspect.value() / 100.0
        self.phasor.add_cursor(
            radius=r, kind=kind,
            radius_minor=r * aspect if kind == "ellipse" else None)
        self._refresh_active_cursor_combo()

    def remove_cursor(self):
        """Delete the selected phasor cursor and refresh segmentation.

        No-op outside cursor mode. Pushes an undo snapshot, then removes the
        currently selected cursor from the phasor canvas; if no cursor is
        selected, logs a hint instead of doing nothing silently. After
        removal, refreshes the Move combo and recomputes segmentation
        (overlay + cluster table) for the remaining cursors.
        """
        if self.mode != "cursor":
            return
        if hasattr(self, "_push_cursor_undo"):
            self._push_cursor_undo()
        if self.phasor.selected < 0:
            self._log("Choose a circle in Move, or click one on the phasor plot.")
            return
        self.phasor.remove_selected()
        self._refresh_active_cursor_combo()
        self._refresh_after_cursor_edit()
        self._log("Circle removed.")

    def clear_cursors(self):
        """Remove all phasor cursors and clear segmentation results.

        Clears every cursor from the phasor canvas, refreshes the (now
        empty) Move combo, and updates the segmentation overlay/table to
        reflect that no cursors remain.
        """
        self.phasor.clear_cursors()
        self._refresh_active_cursor_combo()
        self._refresh_after_cursor_edit()
        self._log("All circles cleared.")

    def _refresh_after_cursor_edit(self):
        """Update segmentation overlay and table whenever circles are added/removed/moved.

        Stops the pending cursor-stats debounce timer (this call supersedes
        it), then either recomputes cluster statistics via
        ``_compute_cursor`` when cursors exist, or clears the overlay,
        cluster table, and overlay checkbox when none remain. No-op outside
        cursor mode or before phasor maps exist.
        """
        self._cursor_debounce_timer.stop()
        if self.mode != "cursor" or self.data.real_cal is None:
            return
        if self.phasor.cursors:
            self._compute_cursor()
        else:
            self.last_overlay = None
            self.cluster_stats = []
            self._fill_table()
            self.chk_overlay.blockSignals(True)
            self.chk_overlay.setChecked(False)
            self.chk_overlay.blockSignals(False)
            self.refresh_image()

    def _on_phasor_key_press(self, event):
        """Handle Delete/Backspace on the phasor canvas to remove cursors.

        Connected to the phasor canvas's key-press signal. Only acts in
        cursor mode; Delete or Backspace removes the currently selected
        cursor via ``remove_cursor``.

        Args:
            event: Key event object with a ``key`` attribute naming the key
                pressed (e.g. ``"delete"``, ``"backspace"``).
        """
        if self.mode != "cursor":
            return
        if event.key in ("delete", "backspace"):
            self.remove_cursor()

    def clear_gmm(self):
        """Remove GMM ellipses, fit state, and painted segmentation.

        Clears the GMM ellipses from the phasor canvas, drops the cached
        ``gmm``/``_gmm_fit`` attributes, resets the overlay and cluster
        stats, clears the stored segmentation on the active dataset, and
        refreshes the cluster table, overlay checkbox, and image view to
        match the now-empty state.
        """
        self.phasor.clear_gmm()
        if hasattr(self, "gmm"):
            del self.gmm
        if hasattr(self, "_gmm_fit"):
            del self._gmm_fit
        self.last_overlay = None
        self.cluster_stats = []
        self._clear_segmentation_on_dataset(self.data)
        self._fill_table()
        self.chk_overlay.blockSignals(True)
        self.chk_overlay.setChecked(False)
        self.chk_overlay.blockSignals(False)
        self.refresh_image()
        self._log("GMM cleared.")

    def _sync_radius_controls(self, r: float):
        """Set slider and spinbox to the selected cursor's radius without signals.

        Clamps ``r`` into the valid range, then writes it to both the radius
        slider (scaled by 1000 for integer slider units) and the radius
        spinbox with signals blocked, so the update does not itself trigger
        ``on_radius_slider``/``on_radius_spin`` and resize the cursor again.

        Args:
            r: Radius value to display, in phasor-plot units.
        """
        r = max(0.005, min(0.400, float(r)))
        self.sld_radius.blockSignals(True)
        self.sld_radius.setValue(int(round(r * 1000)))
        self.sld_radius.blockSignals(False)
        self.sp_radius.blockSignals(True)
        self.sp_radius.setValue(r)
        self.sp_radius.blockSignals(False)

    def _apply_radius(self, r: float, *, from_slider: bool = False, from_spin: bool = False):
        """Update radius UI widgets and resize the selected phasor cursor.

        Clamps ``r`` into the valid range and writes it to whichever of the
        slider/spinbox did not originate the change (keeping both widgets in
        sync without feedback loops), then resizes the currently selected
        cursor on the phasor canvas.

        Args:
            r: New radius value, in phasor-plot units.
            from_slider: True if the slider triggered this call (skip
                updating the slider itself).
            from_spin: True if the spinbox triggered this call (skip updating
                the spinbox itself).
        """
        r = max(0.005, min(0.400, float(r)))
        if not from_slider:
            self.sld_radius.blockSignals(True)
            self.sld_radius.setValue(int(round(r * 1000)))
            self.sld_radius.blockSignals(False)
        if not from_spin:
            self.sp_radius.blockSignals(True)
            self.sp_radius.setValue(r)
            self.sp_radius.blockSignals(False)
        self.phasor.set_selected_radius(r)

    def on_radius_slider(self, v):
        """Resize selected cursor while the radius slider moves.

        Connected to the radius slider's ``valueChanged`` signal. Converts
        the integer slider units (thousandths) back to phasor-plot radius
        units and applies it via ``_apply_radius``.

        Args:
            v: Raw slider value (radius * 1000, as an int).
        """
        self._apply_radius(v * 0.001, from_slider=True)

    def on_radius_spin(self, r):
        """Resize selected cursor while the radius spinbox changes.

        Connected to the radius spinbox's ``valueChanged`` signal. Applies
        the new radius via ``_apply_radius``, keeping the slider in sync.

        Args:
            r: New radius value from the spinbox, in phasor-plot units.
        """
        self._apply_radius(r, from_spin=True)

    def _sync_radius_slider(self):
        """Mirror the selected cursor radius to slider and spinbox.

        No-op if no cursor is currently selected. Otherwise reads the
        selected cursor's radius from the phasor canvas and pushes it into
        the slider/spinbox via ``_sync_radius_controls``.
        """
        i = self.phasor.selected
        if 0 <= i < len(self.phasor.cursors):
            self._sync_radius_controls(self.phasor.cursors[i]["radius"])

    def _on_radius_spin_committed(self):
        """Recompute segmentation after the user finishes typing a radius.

        Connected to the radius spinbox's ``editingFinished`` signal (as
        opposed to every keystroke). Stops the debounce timer and forces an
        immediate ``_refresh_after_cursor_edit`` when in cursor mode with
        cursors present and phasor maps available.
        """
        if self.mode == "cursor" and self.phasor.cursors and self.data.real_cal is not None:
            self._cursor_debounce_timer.stop()
            self._refresh_after_cursor_edit()

    def _live_active(self):
        """Return whether live cursor overlay updates are enabled.

        Combines the "Live" checkbox state, cursor mode, and the presence of
        computed phasor maps — all three must hold for drag/scroll updates
        to paint a throttled overlay in real time.

        Returns:
            True if live overlay updates should run, False otherwise.
        """
        return self.chk_live.isChecked() and self.mode == "cursor" and self.data.real_cal is not None

    def _update_cursor_live_overlay(self):
        """Paint a throttled pseudo-color overlay while cursors move.

        No-op unless live mode is active and cursors exist. Computes the
        current cursor masks and colors, builds a pseudo-color overlay
        scaled by the segmentation intensity image, and either updates the
        existing ``AxesImage`` in place (when the array shape matches, to
        avoid an expensive full figure clear/colorbar rebuild during
        dragging) or falls back to ``show_overlay``. Also auto-enables the
        overlay checkbox if it was off.
        """
        if not self._live_active() or not self.phasor.cursors:
            return
        masks, colors = self._cursor_masks_colors()
        if masks is None:
            return
        intensity = self._segmentation_intensity()
        overlay = pseudo_color(*[masks[k] for k in range(len(masks))],
                               intensity=intensity, colors=np.array(colors))
        self.last_overlay = np.clip(np.asarray(overlay), 0, 1)
        if not self.chk_overlay.isChecked():
            self.chk_overlay.blockSignals(True)
            self.chk_overlay.setChecked(True)
            self.chk_overlay.blockSignals(False)
        title = f"Segmentation ({self.mode})"
        # Reuse AxesImage when shape matches — avoids fig.clear and colorbar churn during drag.
        arr = None if self.image._im is None else self.image._im.get_array()
        if arr is not None and tuple(arr.shape) == tuple(self.last_overlay.shape):
            self.image.update_overlay(self.last_overlay, title=title)
        else:
            self.image.show_overlay(self.last_overlay, title=title)

    def _deferred_cursor_live_overlay(self):
        """Timer callback for throttled live cursor overlay refresh.

        Fired by ``_cursor_live_timer`` while a cursor is being dragged or
        scrolled, so the overlay repaints at a bounded rate instead of on
        every mouse-move event. Simply delegates to
        ``_update_cursor_live_overlay``.
        """
        self._update_cursor_live_overlay()

    def on_cursor_moving(self):
        """Interactive drag/scroll: throttled overlay; table stats wait for release.

        Connected to the phasor canvas's cursor-moving signal, fired
        continuously while a cursor is dragged or its radius is scrolled.
        Keeps the radius controls in sync on every call. When live mode is
        on, paints an immediate overlay update on the first move and then
        (re)starts the live-overlay throttle timer for subsequent moves. For
        scroll-wheel radius changes that are not an active drag, restarts
        the cursor-stats debounce timer so the full cluster table
        recomputes only once scrolling settles (drag releases instead
        trigger ``on_cursor_changed``).
        """
        self._sync_radius_slider()
        if self._live_active() and self.phasor.cursors:
            if not self._cursor_live_timer.isActive():
                self._update_cursor_live_overlay()
            self._cursor_live_timer.start()
        # Scroll wheel: debounce stats until scrolling stops (drag uses cursorChanged on release).
        if (
            not self.phasor.is_dragging_cursor
            and self.mode == "cursor"
            and self.phasor.cursors
            and self.data.real_cal is not None
        ):
            self._cursor_debounce_timer.start()

    def on_cursor_changed(self):
        """Committed change (mouseup / add / remove): full table + overlay update.

        Connected to the phasor canvas's cursor-changed signal, fired once a
        drag ends or a cursor is added/removed. Stops any pending debounce
        timer, refreshes the Move combo and radius controls, and runs a
        full ``_refresh_after_cursor_edit`` to recompute cluster statistics
        and repaint the segmentation overlay.
        """
        self._cursor_debounce_timer.stop()
        self._refresh_active_cursor_combo()
        self._sync_radius_slider()
        self._refresh_after_cursor_edit()

    def _on_radius_slider_released(self):
        """Recompute segmentation after the radius slider is released.

        Connected to the radius slider's ``sliderReleased`` signal. Stops
        the debounce timer and forces an immediate
        ``_refresh_after_cursor_edit`` when in cursor mode with cursors
        present and phasor maps available, so dragging the slider only
        recomputes full statistics once at the end rather than on every
        intermediate value.
        """
        if self.mode == "cursor" and self.phasor.cursors and self.data.real_cal is not None:
            self._cursor_debounce_timer.stop()
            self._refresh_after_cursor_edit()

    def _deferred_cursor_compute(self):
        """Debounced timer callback to recompute cursor cluster stats.

        Fired by ``_cursor_debounce_timer`` after scroll-wheel radius edits
        settle. Runs the full ``_compute_cursor`` pass only if still in
        cursor mode with cursors present and phasor maps available, in case
        state changed while the timer was pending.
        """
        if self.mode == "cursor" and self.phasor.cursors and self.data.real_cal is not None:
            self._compute_cursor()

    def _gmm_k_max(self) -> int:
        """Read the component-count field and clamp it to a valid GMM range.

        Used both as the fixed cluster count when BIC auto-selection is
        disabled and as the upper bound ``k_max`` scanned by
        ``select_gmm_clusters_bic`` when it is enabled. Falls back to 3 if
        the field is missing or does not contain a valid integer.

        Returns:
            Integer clamped to ``[1, 12]``.
        """
        text = self.edit_ncomp.text().strip() if hasattr(self, "edit_ncomp") else "3"
        try:
            return max(1, min(12, int(text)))
        except ValueError:
            return 3

    def _gmm_sigma(self) -> float:
        """Read the ellipse contour scale field and clamp it to a sane range.

        Controls how many standard deviations the drawn GMM ellipses (and
        the derived cluster masks) extend from each fitted component's
        center; passed straight through to `fit_phasor_gmm`. Falls back to
        2.0 if the field is missing or unparsable.

        Returns:
            Float clamped to ``[0.5, 6.0]``.
        """
        text = self.edit_gmm_sigma.text().strip() if hasattr(self, "edit_gmm_sigma") else "2.0"
        try:
            return max(0.5, min(6.0, float(text)))
        except ValueError:
            return 2.0

    # ---- GMM ---------------------------------------------------------------
    def _fit_gmm_worker(self):
        """Fit the phasor GMM off the UI thread for use inside `_run_busy`.

        Reads valid pixels from the active dataset, optionally runs BIC-based
        component-count selection when "Auto (BIC)" is checked, then fits
        `fit_phasor_gmm` with the chosen covariance type and sigma. Called
        from `fit_gmm` and `fit_gmm_all` via `_run_busy` so the GUI stays
        responsive during the (potentially slow) sklearn fit.

        Returns:
            Tuple ``(fit, n_clusters, sigma)`` where ``fit`` is the ellipse
            parameter tuple returned by `fit_phasor_gmm`.
        """
        from flim_phasors.analysis import select_gmm_clusters_bic

        m = self.data.valid_mask()
        g, s = self.data.real_cal, self.data.imag_cal
        cov = self.cb_cov.currentText()
        sigma = self._gmm_sigma()
        if self.chk_bic.isChecked():
            X = np.column_stack([g[m], s[m]])
            n_clusters, best_bic = select_gmm_clusters_bic(
                X, k_max=self._gmm_k_max(), covariance_type=cov,
            )
            self._log(f"GMM auto-selected {n_clusters} components (BIC={best_bic:.0f}).")
        else:
            n_clusters = self._gmm_k_max()
        fit = fit_phasor_gmm(
            g, s, clusters=n_clusters, sigma=sigma, covariance_type=cov)
        return fit, n_clusters, sigma

    def fit_gmm(self):
        """Fit a GMM on the active sample's phasor pixels and draw the ellipses.

        Validates that scikit-learn is available, phasor data has been
        computed, and enough valid pixels exist, then runs `_fit_gmm_worker`
        in a busy-cursor worker thread. On success, stores the fit on
        `_gmm_fit`, draws cluster ellipses on the phasor plot, immediately
        computes and paints the segmentation via `_compute_gmm`, and stashes
        the result on the active dataset so it survives sample switches.
        Bound to the "Fit GMM" button; switches the mode radio button to GMM
        if it was not already selected.
        """
        if not HAVE_SKLEARN:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "pip install scikit-learn"); return
        if not self.rb_gmm.isChecked():
            self.rb_gmm.setChecked(True)
        if self.data.real_cal is None:
            QtWidgets.QMessageBox.information(self, "GMM", "Load data and click Apply first."); return
        m = self.data.valid_mask()
        if m.sum() < 10:
            QtWidgets.QMessageBox.information(self, "GMM", "Not enough valid pixels."); return
        try:
            (fit, n_clusters, sigma), _elapsed = self._run_busy(
                "Fitting GMM…", self._fit_gmm_worker)
        except CancelledError:
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "GMM fit failed", str(e)); return
        self._gmm_fit = fit
        self._log(f"phasorpy GMM: {n_clusters} cluster(s), σ={sigma:.1f}.")
        n = len(self._gmm_fit[0])
        colors = [categorical_rgb(k) for k in range(n)]
        self.phasor.show_gmm_ellipses(*self._gmm_fit, colors)
        self._compute_gmm()
        self._stash_segmentation_to_dataset(self.data)
        return

    def fit_gmm_all(self):
        """Fit a GMM on every loaded sample and cache results for batch export.

        Iterates all datasets with valid phasor data (>=10 valid pixels),
        temporarily swapping `self.data`/`self.active_idx` to reuse the
        single-sample `_fit_gmm_worker` + `_compute_gmm` pipeline for each
        one, and stashes each fit's segmentation on its dataset via
        `_stash_segmentation_to_dataset`. Restores the originally active
        sample and its UI state afterward via `_restore_ui_for_active` and
        `_restore_segmentation_from_dataset`. Samples that fail to fit are
        skipped and reported in a summary dialog; a `CancelledError` from the
        busy dialog stops the loop early. Bound to "Fit GMM (all)"; intended
        to be followed by "Export all…" to save every sample's results
        together.
        """
        if not HAVE_SKLEARN:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "pip install scikit-learn")
            return
        if not self.rb_gmm.isChecked():
            self.rb_gmm.setChecked(True)
        datasets = self._all_datasets() if hasattr(self, "_all_datasets") else [self.data]
        targets = [
            (i, d) for i, d in enumerate(datasets)
            if d.real_cal is not None and int(d.valid_mask().sum()) >= 10
        ]
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "GMM", "Load data and click Apply on samples first.")
            return
        self._stash_segmentation_to_dataset(self.data)
        saved_data = self.data
        saved_idx = self.active_idx
        ok = 0
        errors = []
        for i, d in targets:
            self.data = d
            self.active_idx = i
            try:
                (fit, n_clusters, sigma), _elapsed = self._run_busy(
                    f"Fitting GMM ({i + 1}/{len(datasets)})…", self._fit_gmm_worker)
            except CancelledError:
                break
            except Exception as e:
                errors.append(f"{dataset_short_label(d, i)}: {e}")
                continue
            self._gmm_fit = fit
            self._compute_gmm()
            self._stash_segmentation_to_dataset(d)
            ok += 1
            self._log(
                f"GMM on {dataset_short_label(d, i)}: {n_clusters} cluster(s), σ={sigma:.1f}.")
        self.data = saved_data
        self.active_idx = saved_idx
        self._restore_ui_for_active()
        self._update_phasor_display()
        self._restore_segmentation_from_dataset(self.data)
        self._log(f"GMM fitted on {ok} sample(s). Use Export all… to save them together.")
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "GMM (partial)",
                f"Fitted {ok} sample(s). Failures:\n" + "\n".join(errors[:8]))
        return

    def _on_phasor_click(self, g, s):
        """Report lifetimes for a phasor-plot click and mark the matching pixel.

        Connected to the phasor canvas's click signal. Requires phasor data
        and a positive excitation frequency; computes apparent lifetimes
        (phase, modulation, normal) at the clicked (g, s) via
        `lifetimes_at_phasor`, places a marker on the phasor plot, highlights
        the nearest matching pixel on the image view via
        `_highlight_phasor_on_image`, and logs the readout.

        Args:
            g: Real (g) phasor coordinate of the click.
            s: Imaginary (s) phasor coordinate of the click.
        """
        if self.data.real_cal is None or self.data.work_frequency <= 0:
            return
        self.phasor.set_click_marker(g, s)
        try:
            tp, tm, tn = lifetimes_at_phasor(g, s, self.data.work_frequency)
        except Exception as e:
            self._log(f"Phasor readout failed: {e}")
            return
        self._highlight_phasor_on_image(g, s)
        self._log(
            f"Phasor click ({g:.3f}, {s:.3f}) → τφ={tp:.3f} ns, τmod={tm:.3f} ns, τn={tn:.3f} ns")

    def _on_image_click(self, y, x):
        """Report the phasor coordinate and lifetimes for a clicked image pixel.

        Connected to the image canvas's click signal. Guards against missing
        phasor data, out-of-bounds coordinates, and pixels excluded by the
        valid mask (below threshold) or with a non-finite phasor value,
        logging a status message in each such case instead of raising. On a
        valid pixel, marks both the image and phasor-plot click positions
        and logs the (g, s) coordinate with its apparent lifetimes.

        Args:
            y: Row index of the clicked pixel.
            x: Column index of the clicked pixel.
        """
        if self.data.real_cal is None or self.data.work_frequency <= 0:
            self._log("Image click — run Apply first to compute phasors.")
            return
        valid = self.data.valid_mask()
        h, w = valid.shape
        if y < 0 or x < 0 or y >= h or x >= w:
            return
        if not valid[y, x]:
            self._log(f"Image click ({x}, {y}) — pixel masked (below threshold).")
            return
        g = float(self.data.real_cal[y, x])
        s = float(self.data.imag_cal[y, x])
        if not np.isfinite(g) or not np.isfinite(s):
            self._log(f"Image click ({x}, {y}) — no valid phasor at this pixel.")
            return
        self.phasor.set_click_marker(g, s)
        self.image.set_click_marker(y, x)
        try:
            tp, tm, tn = lifetimes_at_phasor(g, s, self.data.work_frequency)
        except Exception as e:
            self._log(f"Image readout failed: {e}")
            return
        self._log(
            f"Image click ({x}, {y}) → phasor ({g:.3f}, {s:.3f}), "
            f"τφ={tp:.3f} ns, τmod={tm:.3f} ns, τn={tn:.3f} ns")

    def _highlight_phasor_on_image(self, g, s):
        """Mark the image pixel whose phasor value is nearest to (g, s).

        Used by `_on_phasor_click` to reflect a phasor-plot click back onto
        the image view. Searches only valid pixels (finite, above threshold)
        for the minimum squared Euclidean distance in (g, s) space and sets
        the image click marker at that pixel's location. No-op if phasor
        data has not been computed or no pixels are valid.

        Args:
            g: Real (g) phasor coordinate to match.
            s: Imaginary (s) phasor coordinate to match.
        """
        if self.data.real_cal is None:
            return
        valid = self.data.valid_mask()
        if not np.any(valid):
            return
        gr = self.data.real_cal[valid]
        gi = self.data.imag_cal[valid]
        idx = int(np.argmin((gr - g) ** 2 + (gi - s) ** 2))
        ys, xs = np.where(valid)
        y, x = int(ys[idx]), int(xs[idx])
        self.image.set_click_marker(y, x)

    # ---- compute lifetimes + paint ----------------------------------------
    def compute_and_paint(self):
        """Recompute cluster statistics and repaint the segmentation overlay.

        Dispatches to `_compute_cursor` or `_compute_gmm` depending on the
        current `self.mode` ("cursor" vs "gmm"), each of which fills
        `cluster_stats`, builds the pseudo-color overlay, refreshes the
        results table, and stashes segmentation state on the active dataset.
        Bound to the "Compute & Paint" button; requires phasor data to
        already be computed (logs and returns otherwise).
        """
        if self.data.real_cal is None:
            self._log("Paint skipped — run Apply first.")
            return
        self._log(f"Paint ({self.mode})…")
        if self.mode == "cursor":
            self._compute_cursor()
        else:
            self._compute_gmm()
        self._log("Paint complete.")

    @staticmethod
    def _fmt_table_ns(value):
        """Format a lifetime (ns) value for display in the results table.

        Used when populating cluster/cursor rows so that empty or
        degenerate segments (which produce NaN or infinite apparent
        lifetimes) render as a clear placeholder rather than a confusing
        "nan" or "inf" string.

        Args:
            value: Lifetime in nanoseconds, possibly NaN/inf for an empty
                or degenerate cluster.

        Returns:
            An em dash ("—") when ``value`` is not finite, otherwise the
            value formatted to 3 decimal places.
        """
        return "—" if not np.isfinite(value) else f"{value:.3f}"

    def _cursor_masks_colors(self):
        """Build boolean pixel masks and colors for the current phasor cursors.

        For each cursor circle/ellipse in `self.phasor.cursors`, computes a
        per-pixel inclusion mask via `mask_from_elliptic_cursor` (ellipse
        cursors) or `mask_from_circular_cursor` (circle cursors) over the
        active dataset's (g, s) maps, intersected with the valid-pixel mask.
        Used by `_compute_cursor` to aggregate per-cluster statistics and by
        `_paint` to build the pseudo-color overlay.

        Returns:
            Tuple ``(masks, colors)`` where ``masks`` is a stacked boolean
            array of shape ``(n_cursors, H, W)`` and ``colors`` is the list
            of each cursor's RGB color. Returns ``(None, None)`` if there are
            no cursors or no phasor data.
        """
        cur = self.phasor.cursors
        if not cur or self.data.real_cal is None:
            return None, None
        g, s = self.data.real_cal, self.data.imag_cal
        valid = self.data.valid_mask()
        masks = []
        for c in cur:
            if c.get("kind") == "ellipse":
                mk = mask_from_elliptic_cursor(
                    g, s, [c["center_real"]], [c["center_imag"]],
                    radius=[c["radius"]],
                    radius_minor=[c.get("radius_minor", c["radius"] * 0.65)],
                    angle=[c.get("angle", 0.0)],
                )
            else:
                mk = mask_from_circular_cursor(
                    g, s, [c["center_real"]], [c["center_imag"]], radius=[c["radius"]])
            if mk.ndim == 3:
                mk = mk[0]
            masks.append(mk & valid)
        masks = np.stack(masks)
        return masks, [c["color"] for c in cur]

    def _compute_cursor(self):
        """Aggregate lifetimes and pixel counts within each cursor ROI and paint.

        For each cursor mask from `_cursor_masks_colors`, averages (g, s)
        over included pixels, derives apparent lifetimes via
        `lifetimes_at_phasor`, and records the cluster's color, label, pixel
        count, and percent area (relative to all valid pixels) into
        `self.cluster_stats`. Empty clusters get NaN statistics rather than
        being dropped, so the results table stays aligned with the cursor
        list. Finishes by repainting the segmentation overlay (`_paint`),
        refreshing the results table (`_fill_table`), and stashing the
        segmentation on the active dataset. Shows a message box and returns
        early if no cursors are defined.
        """
        cur = self.phasor.cursors
        if not cur:
            QtWidgets.QMessageBox.information(self, "Cursors", "Add at least one circle."); return
        g, s = self.data.real_cal, self.data.imag_cal
        masks, colors = self._cursor_masks_colors()
        valid = self.data.valid_mask()
        total_valid = max(int(valid.sum()), 1)
        self.cluster_stats = []
        for k, c in enumerate(cur):
            mk = masks[k]; n = int(mk.sum())
            if n > 0:
                cg = float(np.nanmean(g[mk])); cs = float(np.nanmean(s[mk]))
                tp, tm, tn = lifetimes_at_phasor(cg, cs, self.data.work_frequency)
            else:
                cg = cs = tp = tm = tn = float("nan")
            self.cluster_stats.append(dict(idx=k + 1, color=c["color"], label=c["label"],
                                           tp=tp, tm=tm, tn=tn,
                                           g=cg, s=cs, n=n,
                                           area=100.0 * n / total_valid))
        self._paint(masks, colors)
        self._fill_table()
        self._stash_segmentation_to_dataset(self.data)

    def _compute_gmm(self):
        """Label pixels by nearest GMM cluster and aggregate per-cluster lifetimes.

        Requires a prior `_gmm_fit` (set by `fit_gmm`/`fit_gmm_all`). Assigns
        each valid pixel to its nearest GMM cluster center via
        `label_pixels_by_gmm`, then for every component computes apparent
        lifetimes directly from that component's fitted (g, s) center
        (rather than averaging assigned pixels, since the GMM center is
        already the cluster's phasor location), and records pixel count and
        percent area into `self.cluster_stats` using categorical colors and
        names. Finishes by repainting the segmentation overlay (`_paint`),
        refreshing the results table (`_fill_table`), and stashing the
        segmentation on the active dataset. Shows a message box and returns
        early if no GMM fit exists or too few valid pixels remain.
        """
        if not hasattr(self, "_gmm_fit"):
            QtWidgets.QMessageBox.information(self, "GMM", "Fit a GMM first."); return
        cr, ci, rm, ri, ang = self._gmm_fit
        g, s = self.data.real_cal, self.data.imag_cal
        valid = self.data.valid_mask()
        if valid.sum() < 10:
            QtWidgets.QMessageBox.information(self, "GMM", "Not enough valid pixels."); return
        n_comp = len(cr)
        labelmap = label_pixels_by_gmm(g, s, cr, ci, rm)
        masks = np.stack([(labelmap == k) & valid for k in range(n_comp)])
        colors = [categorical_rgb(k) for k in range(n_comp)]
        total_valid = max(int(valid.sum()), 1)
        self.cluster_stats = []
        for k in range(n_comp):
            cg, cs = float(cr[k]), float(ci[k])
            mk = masks[k]
            tp, tm, tn = lifetimes_at_phasor(cg, cs, self.data.work_frequency)
            n = int(mk.sum())
            self.cluster_stats.append(dict(idx=k + 1, color=colors[k], label=categorical_name(k),
                                           tp=tp, tm=tm, tn=tn,
                                           g=cg, s=cs, n=n,
                                           area=100.0 * n / total_valid))
        self._paint(masks, colors)
        self._fill_table()
        self._stash_segmentation_to_dataset(self.data)

    def _phasor_valid_mask(self):
        """Return the boolean mask of pixels included in the phasor histogram.

        Thin wrapper around `self.data.valid_mask()` used so image-view
        helpers (`_mask_like_phasor`, `_photon_image_filtered`) share exactly
        the same pixel inclusion rule as the phasor plot.

        Returns:
            Boolean array matching the dataset's spatial shape, or ``None``
            if no phasor data has been computed yet.
        """
        if self.data.real_cal is None:
            return None
        return self.data.valid_mask()

    def _mask_like_phasor(self, arr):
        """Set pixels excluded from the phasor plot to NaN in a display array.

        Used to keep intensity/tau map displays visually consistent with the
        phasor histogram by hiding pixels below threshold or with invalid
        phasor coordinates.

        Args:
            arr: Array-like map with the same spatial shape as the dataset
                (e.g. a tau map), or ``None``.

        Returns:
            A float copy of ``arr`` with excluded pixels set to NaN, or
            ``None`` if ``arr`` is ``None``. Returns ``arr`` unmasked (as a
            float copy) if the valid mask shape does not match.
        """
        if arr is None:
            return None
        out = np.asarray(arr, dtype=float).copy()
        valid = self._phasor_valid_mask()
        if valid is not None and valid.shape == out.shape:
            out[~valid] = np.nan
        return out

    def _photon_image_filtered(self):
        """Photon-count image with the phasor exclusion mask applied.

        Prefers the threshold-cropped mean image (`data.mean_thr`) and falls
        back to the raw mean image (`data.mean_raw`) if thresholding has not
        been applied. Used for the "Photons (masked)" image view and as the
        intensity background for segmentation overlays.

        Returns:
            Float array with NaN at pixels excluded from the phasor plot, or
            ``None`` if no intensity image is available.
        """
        base = self.data.mean_thr if self.data.mean_thr is not None else self.data.mean_raw
        return self._mask_like_phasor(base)

    def _segmentation_intensity(self):
        """Build the intensity background used behind pseudo-color overlays.

        Wraps `_photon_image_filtered`, replacing NaN/masked pixels with 0 so
        `pseudo_color` (which expects a plain intensity array) renders
        excluded regions as black rather than propagating NaNs into the
        overlay.

        Returns:
            Non-negative float array matching the dataset's spatial shape.
            Returns a trivial 2x2 zero array if no intensity image exists
            (avoids errors in `pseudo_color` when called with no data).
        """
        raw = self._photon_image_filtered()
        if raw is None:
            return np.zeros((2, 2), dtype=np.float64)
        out = np.zeros(np.shape(raw), dtype=np.float64)
        finite = np.isfinite(raw)
        out[finite] = np.nan_to_num(raw[finite])
        return out

    def _paint(self, masks, colors):
        """Composite cluster masks into a pseudo-color overlay and display it.

        Combines the per-cluster boolean masks with `_segmentation_intensity`
        via `pseudo_color`, clips the result to [0, 1], stores it as
        `self.last_overlay`, checks the "Overlay" toggle, and calls
        `refresh_image` to draw it. Called by `_compute_cursor` and
        `_compute_gmm` after they finish aggregating cluster statistics.

        Args:
            masks: Boolean array of shape ``(n_clusters, H, W)`` selecting
                the pixels belonging to each cluster.
            colors: Sequence of RGB colors, one per cluster, matching
                ``masks``.
        """
        intensity = self._segmentation_intensity()
        overlay = pseudo_color(*[masks[k] for k in range(len(masks))],
                               intensity=intensity, colors=np.array(colors))
        self.last_overlay = np.clip(np.asarray(overlay), 0, 1)
        self.chk_overlay.setChecked(True); self.refresh_image()

    def _on_image_view_changed(self):
        """Switch the image panel to the selected base view, hiding the overlay.

        Connected to the image-view combo box's change signal. Since the
        tau/photon maps and the segmentation overlay share a single
        `AxesImage` (`self.image`), only one can be visible at a time; this
        unchecks "Overlay" (without re-triggering its own handler, via
        `blockSignals`) before calling `refresh_image` to draw the newly
        selected view.
        """
        # τ/photon maps and RGB overlay share one AxesImage — only one can be visible.
        if hasattr(self, "chk_overlay") and self.chk_overlay.isChecked():
            self.chk_overlay.blockSignals(True)
            self.chk_overlay.setChecked(False)
            self.chk_overlay.blockSignals(False)
        self.refresh_image()

    def refresh_image(self):
        """Redraw the image panel with either the overlay or the base view.

        Central image-refresh entry point called after painting a
        segmentation, toggling "Overlay", or changing the view combo box.
        When "Overlay" is checked and `last_overlay` exists, reuses the
        existing `AxesImage` via `image.update_overlay` when the array shape
        is unchanged (avoids a full `fig.clear`/colorbar rebuild, which is
        expensive during interactive dragging), falling back to
        `image.show_overlay` otherwise. When unchecked, delegates to
        `_show_base_image`.
        """
        if self.chk_overlay.isChecked() and self.last_overlay is not None:
            title = f"Segmentation ({self.mode})"
            arr = None if self.image._im is None else self.image._im.get_array()
            if arr is not None and tuple(arr.shape) == tuple(self.last_overlay.shape):
                self.image.update_overlay(self.last_overlay, title=title)
            else:
                self.image.show_overlay(self.last_overlay, title=title)
        else:
            self._show_base_image()

    def _show_base_image(self):
        """Render the currently selected non-overlay image view.

        Reads the image-view combo box (defaulting to the first item if the
        widget does not exist yet) and dispatches to the appropriate
        renderer: photon counts (masked or raw brightfield) via
        `image.show_intensity` with the log-scale/auto-contrast toggles
        applied, or one of the tau maps (phase, modulation, normal) via
        `image.show_map` with a robust 2nd-98th percentile color range from
        `_robust_range`. Draws the scale bar afterward when a map is shown.
        No-op if the selected tau source has no data yet.
        """
        choice = (
            self.cb_image_view.currentText()
            if hasattr(self, "cb_image_view")
            else IMAGE_VIEW_ITEMS[0]
        )
        if choice in (IMAGE_VIEW_ITEMS[0], IMAGE_VIEW_ITEMS[1]):
            if choice == IMAGE_VIEW_ITEMS[0]:
                disp = self._photon_image_filtered()
                title = "Photons (masked)"
            else:
                disp = self.data.intensity_brightfield()
                title = "Brightfield (all photons)"
            if disp is not None:
                log_scale = getattr(self, "chk_log_display", None) and self.chk_log_display.isChecked()
                auto_c = not getattr(self, "chk_auto_contrast", None) or self.chk_auto_contrast.isChecked()
                self.image.show_intensity(
                    disp, log_scale=log_scale, auto_contrast=auto_c, title=title)
                self._draw_scale_bar()
            return
        tau_sources = {
            IMAGE_VIEW_ITEMS[2]: (self.data.tau_phi, "τφ phase (ns)"),
            IMAGE_VIEW_ITEMS[3]: (self.data.tau_mod, "τmod (ns)"),
            IMAGE_VIEW_ITEMS[4]: (self.data.tau_normal, "τ normal (ns)"),
        }
        src = tau_sources.get(choice)
        if src is None:
            return
        tau_map, cbar_label = src
        disp = self._mask_like_phasor(tau_map)
        if disp is None:
            return
        vmin, vmax = self._robust_range(disp)
        self.image.show_map(disp, title=choice, cmap="turbo", label=cbar_label,
                            vmin=vmin, vmax=vmax)
        self._draw_scale_bar()

    def _draw_scale_bar(self):
        """Draw a 10 µm scale bar on the image view if pixel size is known.

        Reads the pixel-size spin box, falling back to the dataset's stored
        `pixel_size_um` if the spin box is zero/absent. Skips drawing if no
        pixel size is available, no phasor data exists, or the resulting bar
        would exceed 40% of the image width (avoids an overly dominant bar
        on very high-magnification or very narrow images).
        """
        um = self.sp_pixel_um.value() if hasattr(self, "sp_pixel_um") else 0.0
        if um <= 0 and getattr(self.data, "pixel_size_um", 0) > 0:
            um = self.data.pixel_size_um
        if um > 0 and self.data.real_cal is not None:
            h, w = self.data.real_cal.shape
            bar_um = 10.0
            bar_px = bar_um / um
            if bar_px < w * 0.4:
                self.image.draw_scale_bar(bar_px, label=f"{bar_um:g} µm")

    @staticmethod
    def _robust_range(arr):
        """Compute a percentile-based color range robust to outlier pixels.

        Used to set `vmin`/`vmax` for tau map displays so a handful of
        extreme (typically noise-driven) lifetime values do not wash out the
        color scale for the rest of the image.

        Args:
            arr: Array-like map; non-finite values are ignored.

        Returns:
            Tuple ``(vmin, vmax)`` at the 2nd and 98th percentiles of the
            finite values, or ``(None, None)`` if no finite values exist.
        """
        finite = np.asarray(arr)[np.isfinite(arr)]
        if finite.size == 0:
            return None, None
        return float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))

    # ---- results table -----------------------------------------------------
    def _fill_table(self):
        """Repopulate the results table from `self.cluster_stats`.

        Rebuilds every row: index, color swatch, editable label, mean
        (g, s), the three apparent lifetimes (formatted via
        `_fmt_table_ns`), pixel count, and percent area. Sets
        `_filling_table` around the rebuild so `_label_edited` (triggered by
        `setItem` on the editable label column) ignores these programmatic
        changes and only reacts to user edits. Called after
        `_compute_cursor`/`_compute_gmm` whenever cluster statistics change.
        """
        self._filling_table = True
        self.table.setRowCount(len(self.cluster_stats))
        for r, st in enumerate(self.cluster_stats):
            self.table.setItem(r, 0, self._ro(str(st["idx"])))
            sw = QtWidgets.QTableWidgetItem(); sw.setBackground(QtGui.QColor.fromRgbF(*st["color"]))
            sw.setFlags(Qt.ItemFlag.ItemIsEnabled); self.table.setItem(r, 1, sw)
            lab = QtWidgets.QTableWidgetItem(st["label"])
            lab.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, 2, lab)
            self.table.setItem(r, 3, self._ro(f"{st['g']:.4f}"))
            self.table.setItem(r, 4, self._ro(f"{st['s']:.4f}"))
            self.table.setItem(r, 5, self._ro(self._fmt_table_ns(st["tp"])))
            self.table.setItem(r, 6, self._ro(self._fmt_table_ns(st["tm"])))
            self.table.setItem(r, 7, self._ro(self._fmt_table_ns(st["tn"])))
            self.table.setItem(r, 8, self._ro(str(st["n"])))
            self.table.setItem(r, 9, self._ro(f"{st['area']:.2f}"))
        self._filling_table = False

    def _label_edited(self, item):
        """Sync a user-edited results-table label back into app state.

        Connected to the results table's `itemChanged` signal. Ignored while
        `_fill_table` is programmatically rebuilding the table
        (`_filling_table` guard) to avoid feedback loops. For a real edit to
        the label column, updates the corresponding `cluster_stats` entry
        and, in cursor mode, the matching `phasor.cursors` entry and its
        combo-box entry via `_refresh_active_cursor_combo`.

        Args:
            item: The `QTableWidgetItem` that changed.
        """
        if self._filling_table:
            return
        if item.column() == 2:
            r = item.row()
            if 0 <= r < len(self.cluster_stats):
                self.cluster_stats[r]["label"] = item.text()
                if self.mode == "cursor" and r < len(self.phasor.cursors):
                    self.phasor.cursors[r]["label"] = item.text()
                    self._refresh_active_cursor_combo(select_idx=r)

    @staticmethod
    def _ro(text):
        """Create a read-only, selectable results-table cell.

        Most results-table columns (lifetime values, pixel counts, etc.)
        should be selectable and copyable but not user-editable; only the
        cluster label column uses an editable item instead (see
        :meth:`_editable_group` for the analogous compare-table case).

        Args:
            text: Cell text to display.

        Returns:
            A `QTableWidgetItem` with the editable flag stripped, used for
            every results-table column except the cluster label.
        """
        it = QtWidgets.QTableWidgetItem(text); it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable); return it

    @staticmethod
    def _editable_group(text):
        """Create an editable compare-table cell for a sample's group name.

        Used in the multi-sample comparison table so users can tag each
        loaded file with a biological group (e.g. "Tumor", "Control") for
        downstream grouped statistics/plots.

        Args:
            text: Initial group name text.

        Returns:
            An editable, selectable `QTableWidgetItem` with a tooltip
            explaining the field's purpose.
        """
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable)
        it.setToolTip("Group name for this file (e.g. Tumor, Control)")
        return it

    @staticmethod
    def _editable_sample(text):
        """Create an editable compare-table cell for a sample's display name.

        Lets users rename how a loaded file is labeled in the comparison
        table and exported reports, independent of its filename.

        Args:
            text: Initial sample display name.

        Returns:
            An editable, selectable `QTableWidgetItem`.
        """
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable)
        return it

    @staticmethod
    def _rgb_hex(c):
        """Convert an RGB float triplet in [0, 1] to an ARGB hex color string.

        Used when writing cluster colors into Excel exports (openpyxl fill
        colors use ARGB hex strings). Values outside [0, 1] are clamped
        before conversion.

        Args:
            c: Sequence of at least 3 floats ``(r, g, b)`` in ``[0, 1]``.

        Returns:
            8-character ARGB hex string with a fully opaque ("FF") alpha
            channel, e.g. ``"FFRRGGBB"``.
        """
        return "FF" + "".join(f"{int(round(255 * max(0.0, min(1.0, x)))):02X}" for x in c[:3])

    # ---- export ------------------------------------------------------------
    def export_all(self):
        """Prompt for a folder and export the full analysis bundle there.

        Requires the active dataset to have a loaded sample. Remembers and
        reuses the last export directory (falling back to the last sample
        directory) via `QSettings`. Delegates the actual file writing to
        `export_analysis_bundle`, which saves plots, maps, tables, and
        session data for every loaded sample; reports failures via a message
        box and logs/summarizes success (file count, sample count,
        destination folder) on completion. Bound to the "Export all…"
        button.
        """
        if not dataset_has_sample(self.data):
            QtWidgets.QMessageBox.information(
                self, "Export", "Load a sample and click Apply before exporting.")
            return
        default = self._dialog_dir("export_dir", self._dialog_dir("sample_dir"))
        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose export folder", default)
        if not out:
            return
        self._settings.setValue("export_dir", out)
        try:
            result = export_analysis_bundle(self, out)
        except Exception as e:
            self._log(f"Export failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return
        n = result["n_samples"]
        nf = len(result["files"])
        self._log(f"Exported {nf} file(s) for {n} sample(s) → {result['directory']}")
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {nf} files for {n} sample(s) to:\n{result['directory']}",
        )

