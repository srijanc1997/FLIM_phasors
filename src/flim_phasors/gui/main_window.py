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
from flim_phasors.io import is_supported_flim_path
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
    """Primary FLIM phasor analysis window with plots, controls, and results table."""

    def __init__(self):
        """Initialize state, build the UI, and wire enhancement features."""
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
        self.shared_ref_channel = 0
        self.ref_calibration = ReferenceCalibration()
        self.last_overlay = None
        self.cluster_stats = []
        self.mode = "cursor"
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
        """Construct the control panel, phasor/image plots, and results table."""
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)

        _small = "font-size: 10px;"
        _lbl_file = f"color: gray; {_small}"

        def _tab_page():
            """Create a scrollable tab page with a top-aligned vertical layout."""
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
        self.cb_channel.addItem("0")
        self.cb_channel.setMinimumWidth(48)
        self.cb_channel.currentIndexChanged.connect(self.on_channel_change)
        row_s.addWidget(btn_sample, 1)
        ch_lbl = QtWidgets.QLabel("Ch")
        ch_lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        row_s.addWidget(ch_lbl)
        row_s.addWidget(self.cb_channel)
        sl.addLayout(row_s)
        self.chk_fast_load = QtWidgets.QCheckBox("Fast load (single channel)")
        self.chk_fast_load.setToolTip(
            "Decode only the selected channel to save memory and time.\n"
            "Best for multi-channel files. Switching channel re-decodes the file.")
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
        self.cb_ref_channel.addItem("0")
        self.cb_ref_channel.setMinimumWidth(48)
        self.cb_ref_channel.setEnabled(False)
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
            "Type g/s above, click Set g/s, then Apply on samples.")
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
        """Append a timestamped line to the Files log and optionally the status bar."""
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
        """Add a labelled row and return a (label_widget, field_widget) tuple for show/hide."""
        lbl = QtWidgets.QLabel(label)
        form.addRow(lbl, widget)
        return (lbl, widget)

    def _run_busy(self, message: str, fn, *, cancellable: bool = True):
        """Run heavy work off the GUI thread; optional Cancel."""
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
        """Hide stepper arrows so typed values are fully visible."""
        sp.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        sp.setMinimumWidth(min_width)
        sp.setMaximumWidth(16777215)

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        """Format a duration for log messages (milliseconds or seconds)."""
        if seconds < 1.0:
            return f"{seconds * 1000:.0f} ms"
        return f"{seconds:.2f} s"

    # ---- show/hide filter params ------------------------------------------
    def on_filter_change(self, mode):
        """Show kernel or pawflim controls and block unsupported LIF filter modes."""
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
        """Autosave processing spinboxes to the active dataset in multi-image mode."""
        for w in (
            self.sp_harm, self.sp_freq, self.sp_reflt, self.sp_msize, self.sp_mrep,
            self.sp_psigma, self.sp_plevels, self.sp_thr,
        ):
            w.valueChanged.connect(self._on_per_sample_proc_changed)
        self.cb_filter.currentTextChanged.connect(self._on_per_sample_proc_changed)
        self.chk_detect_harm.toggled.connect(self._on_per_sample_proc_changed)
        self.cb_channel.currentIndexChanged.connect(self._on_per_sample_proc_changed)

    def _on_per_sample_proc_changed(self, *_args):
        """Persist UI processing settings and debounce compare-table refresh."""
        if self._loading_proc_ui or not per_sample_processing(self):
            return
        self._save_proc_from_ui(self.data)
        self._proc_debounce_timer.start()

    def _deferred_refresh_compare_list(self):
        """Refresh the multi-image table after per-sample settings change."""
        if per_sample_processing(self):
            self._refresh_compare_list()

    def _on_bic_toggled(self, checked: bool):
        """Update GMM component spinbox tooltip for fixed-k vs BIC auto-k mode."""
        if checked:
            self.edit_ncomp.setToolTip("Maximum k for BIC search (1–12).")
        else:
            self.edit_ncomp.setToolTip("Number of GMM components (1–12).")

    def _init_dataset_proc_settings(self, d: PhasorData):
        """Seed per-sample processing settings from the current UI if unset."""
        if d.processing_settings is None:
            d.processing_settings = capture_processing_from_ui(self)

    def _save_proc_from_ui(self, d: PhasorData):
        """Copy Setup-tab processing controls into a dataset's stored settings."""
        if d is None:
            return
        if per_sample_processing(self):
            d.processing_settings = capture_processing_from_ui(self)

    def _load_proc_to_ui(self, d: PhasorData):
        """Populate Setup-tab controls from a dataset's stored processing settings."""
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
        """Show or hide multi-image table and active-sample labels."""
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
        """Rebuild the active-sample dropdown from loaded datasets."""
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
        """Handle active-sample selection from the Setup-tab combo box."""
        self._on_sample_picker_change(idx)

    def _on_sample_picker_change(self, idx: int):
        """Switch the active dataset when the user picks a sample index."""
        if self._table_sel_lock or not (0 <= idx < len(self.datasets)):
            return
        if idx == self.active_idx:
            return
        self._activate_dataset(idx)

    def _sync_sample_table_selection(self):
        """Keep compare table row and sample combo aligned with active_idx."""
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
        """Relabel Apply buttons and show Apply all when multiple samples are loaded."""
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
        """Toggle between cursor ROI and GMM segmentation modes."""
        self.mode = "cursor" if self.rb_cursor.isChecked() else "gmm"
        self.cursor_box.setVisible(self.mode == "cursor")
        self.gmm_box.setVisible(self.mode == "gmm")
        if self.mode == "cursor":
            self.clear_gmm()
        else:
            self.phasor.clear_cursors()
            self.last_overlay = None
            self.cluster_stats = []
            self._fill_table()

    # ---- file actions ------------------------------------------------------
    def _dialog_dir(self, key: str, fallback: str = "") -> str:
        """Return the last-used directory for a QFileDialog category."""
        return self._settings.value(key, fallback) or ""

    def choose_sample(self):
        """Open a file dialog and load one or more FLIM sample files."""
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
        """Turn file paths into load jobs; prompt when a LIF holds multiple FLIM series."""
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
        """Confirm how new load jobs merge with the current session. Returns False if cancelled."""
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
        """Decode FLIM files and show uncalibrated phasor / image preview."""
        n = len(jobs)
        loaded = []
        t_decode = 0.0
        frame = int(self.sp_frame.value()) if hasattr(self, "sp_frame") else -1
        load_ch = self.cb_channel.currentIndex() if self.chk_fast_load.isChecked() else None
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
        else:
            self._log(
                f"{len(loaded)} sample(s) decoded ({self._fmt_elapsed(t_decode)}) — "
                "choose Reference, then Apply (calibration is automatic).")

    def _effective_ref_path(self, d=None):
        """Return the reference file path for a dataset (shared or per-sample)."""
        d = d or self.data
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return self.shared_ref_path
        return d.ref_path or None

    def _ref_channel_for_dataset(self, d):
        """Return the reference channel index used when calibrating a dataset."""
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return min(self.shared_ref_channel, max(0, self.shared_ref_n_channels - 1))
        return min(d.ref_channel, max(0, d.ref_n_channels - 1)) if d.ref_path else 0

    def _propagate_shared_reference(self):
        """Copy shared reference settings onto every loaded sample when shared mode is on."""
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
        """Enable or disable one reference file for all loaded samples."""
        if checked and self.shared_ref_path:
            self._propagate_shared_reference()
        self._restore_ui_for_active()
        self._log(
            "Shared reference on — one reference file calibrates all samples."
            if checked else
            "Shared reference off — each sample can use its own reference (active sample).")

    def _all_datasets(self):
        """Return active and listed datasets without duplicates."""
        seen = []
        for d in [self.data] + list(self.datasets):
            if d is not None and d not in seen:
                seen.append(d)
        return seen

    def _reference_harmonic_for_cal(self):
        """Return harmonic index (or list for pawflim) for reference calibration."""
        h = int(self.sp_harm.value())
        if self.cb_filter.currentText() == "pawflim":
            return [h, 2 * h]
        return h

    def _effective_ref_file_path(self):
        """Return the reference file path selected for Calibrate/Apply."""
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
        """True when Apply can run without decoding the reference file again."""
        if self.chk_manual_cal.isChecked():
            return self.ref_calibration.is_active
        if not self._effective_ref_file_path():
            return True
        return self.ref_calibration.is_active

    def _recompute_reference_calibration(self):
        """Decode reference once; store scalar g/s only (reference histogram is released)."""
        path = self._effective_ref_file_path()
        if not path:
            return False
        ch = self._ref_channel_for_dataset(self.data)
        harm = self._reference_harmonic_for_cal()

        def work():
            """Decode reference file and compute mean g/s (runs off GUI thread)."""
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
        """Copy stored calibration g/s into the manual entry fields."""
        self.edit_ref_g.blockSignals(True)
        self.edit_ref_s.blockSignals(True)
        self.edit_ref_g.setText(f"{self.ref_calibration.mean_g:.5f}")
        self.edit_ref_s.setText(f"{self.ref_calibration.mean_s:.5f}")
        self.edit_ref_g.blockSignals(False)
        self.edit_ref_s.blockSignals(False)

    def _apply_manual_calibration_fields(self):
        """Read manual g/s fields and update the in-memory calibration object."""
        try:
            self.ref_calibration.manual_g = float(self.edit_ref_g.text().strip())
            self.ref_calibration.manual_s = float(self.edit_ref_s.text().strip())
        except ValueError:
            return
        self.ref_calibration.use_manual = True
        self.ref_calibration.mean_g = self.ref_calibration.manual_g
        self.ref_calibration.mean_s = self.ref_calibration.manual_s

    def _update_calibration_display(self):
        """Refresh calibration summary label and panel status line."""
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
        """Enable manual g/s entry and recompute file-based calibration when off."""
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
        """Apply typed g/s to calibration (does not preprocess samples)."""
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
        """Refresh calibration label/preview: Ref τ and laser frequency change
        the target phasor position but not the stored reference g/s."""
        if self.ref_calibration.is_active:
            self._update_calibration_display()
            self._update_ref_preview()

    def _on_harm_or_ref_setting_changed(self, *_args):
        """Harmonic/filter changes need a fresh Calibrate before Apply."""
        if self.ref_calibration.values_ready and not self.chk_manual_cal.isChecked():
            self.ref_calibration.values_ready = False
            self.ref_calibration._maps = None
        if self.ref_calibration.is_active or self._effective_ref_file_path():
            self._update_calibration_display()
            self._update_calibration_stale_style()

    def _clear_calibration(self):
        """Reset reference calibration, paths, and per-sample ref assignments."""
        self.ref_calibration.clear()
        clear_calibration_cache()
        self.shared_ref_path = ""
        self.shared_ref_n_channels = 1
        self.lbl_ref.setText("(none)")
        self.chk_manual_cal.setChecked(False)
        self._sync_manual_fields_from_calibration()
        self._update_calibration_display()
        self._update_ref_preview()
        for d in self._all_datasets():
            d.ref_path = ""
        self._log("Calibration cleared.")

    def choose_ref(self):
        """Pick the calibration reference file (decode happens on Calibrate)."""
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
        self._propagate_shared_reference()
        self.shared_ref_n_channels = max(1, self.shared_ref_n_channels)
        self._update_ref_channel_combo()
        self._log(f"Reference file selected: {os.path.basename(path)}.")
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
        """Decode reference and compute g/s (does not preprocess samples)."""
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
        """Seed per-sample settings from LAS X metadata after LIF load."""
        if d.load_source != "lif_phasor":
            return
        if d.processing_settings is None:
            d.processing_settings = capture_processing_from_ui(self)
        d.processing_settings["frequency"] = float(d.frequency)
        thr = float(getattr(d, "lif_lasx_intensity_threshold", 0) or 0)
        if thr > 0:
            d.processing_settings["intensity_min"] = thr

    def _activate_new_dataset(self, d, *, refresh_table: bool = True):
        """Make d the active dataset; append to the set if multi-image mode is on."""
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
        """Truncate a basename for compact panel and status labels."""
        name = os.path.basename(path) if path else fallback
        if len(name) > max_len:
            return name[: max_len - 1] + "…"
        return name

    def _update_proc_active_label(self):
        """Show the active sample filename beside the sample combo when solo."""
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
        """Update the top panel status with active sample and calibration state."""
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
        """Sync all file, channel, processing, and label controls to the active dataset."""
        d = self.data
        self.lbl_sample.setText(self._compact_filename(d.sample_path, "(no sample)"))
        self._update_proc_active_label()
        self._update_panel_status()
        if self.chk_shared_ref.isChecked():
            ref = self.shared_ref_path
        else:
            ref = d.ref_path
        self.lbl_ref.setText(self._compact_filename(ref) if ref else "(none)")
        nch = max(1, d.n_channels)
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        self.cb_channel.addItems([str(i) for i in range(nch)])
        self.cb_channel.setCurrentIndex(min(d.channel, nch - 1))
        self.cb_channel.blockSignals(False)
        self._update_ref_channel_combo()
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
        """Populate reference channel combo from shared or per-sample ref metadata."""
        ref_path = self._effective_ref_path(self.data)
        has_ref = bool(ref_path)
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            nch = max(1, self.shared_ref_n_channels)
            ch = self.shared_ref_channel
        else:
            nch = max(1, self.data.ref_n_channels)
            ch = self.data.ref_channel
        self.cb_ref_channel.blockSignals(True)
        self.cb_ref_channel.clear()
        self.cb_ref_channel.addItems([str(i) for i in range(nch)])
        if has_ref:
            self.cb_ref_channel.setCurrentIndex(min(ch, nch - 1))
        self.cb_ref_channel.setEnabled(has_ref)
        self.cb_ref_channel.blockSignals(False)

    def _set_multi_detail_enabled(self, enabled):
        """Enable or disable multi-image tab widgets as a group."""
        for w in self._multi_detail_widgets:
            w.setEnabled(enabled)
        self._update_sample_label_controls()

    def _update_sample_label_controls(self):
        """Enable display-name and group fields only when a sample is loaded."""
        has_sample = dataset_has_sample(self.data)
        for w in getattr(self, "_sample_label_widgets", ()):
            w.setEnabled(has_sample)

    def _process_all_loaded_datasets(self, *, use_ui_settings=True):
        """Re-run phasor pipeline on every loaded sample (caller may wrap in _run_busy)."""
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
        """Update UI, phasor plot, and image view after Apply completes."""
        self._restore_ui_for_active()
        self._refresh_compare_list()
        self._refresh_compare_group_filter()
        self._update_phasor_display()
        self.chk_overlay.blockSignals(True)
        self.chk_overlay.setChecked(False)
        self.chk_overlay.blockSignals(False)
        self.refresh_image()
        if hasattr(self, "_update_metadata_panel"):
            self._update_metadata_panel()

    def _refresh_image_combo(self):
        """Refresh sample table, dropdown, and overlay filters."""
        self._refresh_compare_list()
        self._refresh_compare_group_filter()
        self._refresh_sample_combo()
        self._sync_sample_table_selection()

    def _refresh_compare_group_filter(self):
        """Rebuild overlay group filter combo from dataset group names."""
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
        """Save the active sample's group name and refresh overlay legend."""
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
        """Save the active sample's legend display name and update the compare table."""
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
        """Update one compare-table Sample cell after a display-name change."""
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
        """Refresh the Active label on the Multi-phasor tab."""
        if not hasattr(self, "lbl_editing"):
            return
        if self.chk_multi.isChecked() and len(self.datasets) > 1 and 0 <= self.active_idx < len(self.datasets):
            self.lbl_editing.setText(
                f"Active: {dataset_display_label(self.data, self.active_idx)}")

    def _legend_include_group(self) -> bool:
        """Return whether phasor legend labels should include group names."""
        if not hasattr(self, "cb_legend_format"):
            return True
        return self.cb_legend_format.currentText() != "Sample name"

    def _legend_loc(self) -> str:
        """Return the matplotlib legend location key for the phasor plot."""
        if not hasattr(self, "cb_legend_loc"):
            return "upper right"
        loc = self.cb_legend_loc.currentText().strip()
        return loc if loc in LEGEND_LOC_ITEMS else "upper right"

    def _legend_fontsize(self) -> float:
        """Return legend font size for multi-image phasor overlay."""
        if not hasattr(self, "sp_legend_size"):
            return float(LEGEND_SIZE_DEFAULT)
        return float(self.sp_legend_size.value())

    def _update_frame_control(self):
        """Enable frame spinbox when the active sample has a time stack."""
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
        """Reload and reprocess the active sample for a new time-frame index."""
        if self.data.sample_path:
            if int(value) == int(getattr(self.data, "frame_index", -1)):
                return
            self.data.frame_index = int(value)
            self._reload_active_sample()

    def _reload_active_sample(self):
        """Decode the active sample at the current frame and rerun Apply."""
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
        """Return the Show checkbox item for a compare-table row."""
        return self.table_compare.item(row, 0)

    def _compare_dataset_index(self, row):
        """Return dataset index stored in a compare-table row's Show item."""
        it = self._compare_show_item(row)
        if it is None:
            return -1
        return int(it.data(Qt.ItemDataRole.UserRole))

    def _refresh_compare_list(self):
        """Rebuild the multi-image compare table from loaded datasets."""
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
        """Keep multi-image phasor overlay off until the user enables it."""
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
        """Enable overlay style, legend, and table controls when compare is available."""
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
        """Check or uncheck all Show boxes in the compare table."""
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
        """Show every preprocessed sample on the phasor overlay."""
        self._compare_set_all_checks(True)

    def _compare_select_none(self):
        """Hide all samples from the phasor overlay."""
        self._compare_set_all_checks(False)

    def _on_compare_table_changed(self, row, column):
        """Handle inline edits to sample name, group, or Show checkbox."""
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
        """Activate the dataset for the selected compare-table row."""
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
        """Build phasor overlay layer descriptors from checked compare-table rows."""
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
        """Return the phasor canvas style key for the selected overlay mode."""
        return COMPARE_STYLE_MAP.get(self.cb_compare_style.currentText(), "cloud")

    def _compare_overlay_active(self):
        """Return whether multi-image overlay is on and its layer list."""
        layers = self._build_compare_layers()
        visible = [L for L in layers if L.get("visible") and L["data"].real_cal is not None]
        compare_on = (
            self.chk_multi.isChecked()
            and self.chk_compare.isChecked()
            and len(visible) >= 1
        )
        return compare_on, layers, visible

    def _update_phasor_display(self, status_note=""):
        """Redraw phasor plot with optional multi-image overlay and legend."""
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
        """Refresh overlay legend without redrawing phasor point clouds."""
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
        """React to legend format, position, or size changes on the overlay."""
        self._set_compare_controls_enabled(
            self.chk_multi.isChecked() and len(self.datasets) >= 2)
        self._update_phasor_legend_only()

    def _on_compare_ui_changed(self, *_args):
        """React to overlay toggle, style, or group-filter changes."""
        self._set_compare_controls_enabled(
            self.chk_multi.isChecked() and len(self.datasets) >= 2)
        if self.data.real_cal is not None or self.chk_compare.isChecked():
            self._update_phasor_display()

    def on_multi_toggle(self, checked):
        """Enter or leave multi-image mode and refresh compare UI."""
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
        """Switch active sample, restore its UI settings, and refresh plots."""
        if not (0 <= idx < len(self.datasets)):
            return
        if per_sample_processing(self):
            self._save_proc_from_ui(self.data)
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
        """Remember GMM fit, cluster table, and overlay on a sample dataset."""
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
        """Drop stored GMM / paint results for one sample (e.g. after Apply)."""
        if d is None:
            return
        d.gmm_fit = None
        d.cluster_stats = []
        d.last_overlay = None

    def _restore_segmentation_from_dataset(self, d=None):
        """Restore GMM ellipses, table, and overlay from a sample's stash."""
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
        """Copy active sample's processing settings to all datasets and Apply all."""
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
        """Remove the active sample from the multi-image list."""
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

        In fast-load mode only one channel is in memory, so selecting a
        different channel re-decodes just that channel.
        """
        if self.data.signal_full is None:
            return
        new_ch = max(0, idx)
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
        """Re-decode only ``new_ch`` for the active sample (fast-load mode)."""
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
        """Persist the fast-load preference and explain the trade-off."""
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
        """Change reference channel and invalidate stored g/s until Calibrate."""
        if not self._effective_ref_path(self.data):
            return
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            ch = max(0, min(idx, self.shared_ref_n_channels - 1))
            self.shared_ref_channel = ch
            self._propagate_shared_reference()
        else:
            self.data.ref_channel = max(0, min(idx, self.data.ref_n_channels - 1))
        if self._effective_ref_file_path() and not self.chk_manual_cal.isChecked():
            self.ref_calibration.values_ready = False
            self.ref_calibration._maps = None
            self._update_calibration_display()
            self._update_calibration_stale_style()
            self._log("Reference channel changed — g/s will be recomputed automatically on Apply.")
            return

    # ---- processing --------------------------------------------------------
    def _active_calibration(self):
        """Return the calibration object to apply, syncing manual fields first."""
        if self.chk_manual_cal.isChecked():
            self._apply_manual_calibration_fields()
        return self.ref_calibration if self.ref_calibration.is_active else None

    def _process_uncalibrated_preview(self, datasets):
        """Compute quick uncalibrated phasor maps after sample load."""
        for d in datasets:
            if not dataset_has_sample(d):
                continue
            if d.load_source == "lif_phasor":
                continue
            run_processing_on_dataset(self, d, use_ui_settings=True, calibrate=False)

    def _run_processing_on_dataset(self, d, *, use_ui_settings=False, calibrate=True):
        """Delegate phasor preprocessing for one dataset to processing module."""
        run_processing_on_dataset(
            self, d, use_ui_settings=use_ui_settings, calibrate=calibrate)

    def apply_processing(self, scope: str = "auto"):
        """Preprocess active or all samples with current calibration and filters."""
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
        st = getattr(self.data, "_intensity_stats", {})
        if st:
            self.lbl_photon_range.setText(
                f"Image photon counts: {st['min']:.0f} – {st['max']:.0f} "
                f"(median {st['median']:.0f})")
            n_below = int(round(st["masked_pct"] * st.get("n_pixels", 0) / 100.0))
            int_msg = (f" | min photons ≥ {st['threshold']:.0f} "
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

    # ---- cursor actions ----------------------------------------------------
    def _refresh_active_cursor_combo(self, select_idx=None):
        """Rebuild Move combo from phasor cursors and select the given index."""
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
        """Sync Move combo and radius controls when a cursor is clicked."""
        self._refresh_active_cursor_combo(select_idx=idx)
        self._sync_radius_slider()

    def on_active_cursor_change(self, combo_idx):
        """Select the cursor index chosen in the Move combo box."""
        if self.mode != "cursor" or combo_idx < 0:
            return
        self.phasor.select_cursor(combo_idx, emit=False)
        self._sync_radius_slider()

    def add_cursor(self):
        """Add a circle or ellipse ROI at the current radius on the phasor plot."""
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
        """Delete the selected phasor cursor and refresh segmentation."""
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
        """Remove all phasor cursors and clear segmentation results."""
        self.phasor.clear_cursors()
        self._refresh_active_cursor_combo()
        self._refresh_after_cursor_edit()
        self._log("All circles cleared.")

    def _refresh_after_cursor_edit(self):
        """Update segmentation overlay and table whenever circles are added/removed/moved."""
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
        """Handle Delete/Backspace on the phasor canvas to remove cursors."""
        if self.mode != "cursor":
            return
        if event.key in ("delete", "backspace"):
            self.remove_cursor()

    def clear_gmm(self):
        """Remove GMM ellipses, fit state, and painted segmentation."""
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
        """Set slider and spinbox to the selected cursor's radius without signals."""
        r = max(0.005, min(0.400, float(r)))
        self.sld_radius.blockSignals(True)
        self.sld_radius.setValue(int(round(r * 1000)))
        self.sld_radius.blockSignals(False)
        self.sp_radius.blockSignals(True)
        self.sp_radius.setValue(r)
        self.sp_radius.blockSignals(False)

    def _apply_radius(self, r: float, *, from_slider: bool = False, from_spin: bool = False):
        """Update radius UI widgets and resize the selected phasor cursor."""
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
        """Resize selected cursor while the radius slider moves."""
        self._apply_radius(v * 0.001, from_slider=True)

    def on_radius_spin(self, r):
        """Resize selected cursor while the radius spinbox changes."""
        self._apply_radius(r, from_spin=True)

    def _sync_radius_slider(self):
        """Mirror the selected cursor radius to slider and spinbox."""
        i = self.phasor.selected
        if 0 <= i < len(self.phasor.cursors):
            self._sync_radius_controls(self.phasor.cursors[i]["radius"])

    def _on_radius_spin_committed(self):
        """Recompute segmentation after the user finishes typing a radius."""
        if self.mode == "cursor" and self.phasor.cursors and self.data.real_cal is not None:
            self._cursor_debounce_timer.stop()
            self._refresh_after_cursor_edit()

    def _live_active(self):
        """Return whether live cursor overlay updates are enabled."""
        return self.chk_live.isChecked() and self.mode == "cursor" and self.data.real_cal is not None

    def _update_cursor_live_overlay(self):
        """Paint a throttled pseudo-color overlay while cursors move."""
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
        arr = None if self.image._im is None else self.image._im.get_array()
        if arr is not None and tuple(arr.shape) == tuple(self.last_overlay.shape):
            self.image.update_overlay(self.last_overlay, title=title)
        else:
            self.image.show_overlay(self.last_overlay, title=title)

    def _deferred_cursor_live_overlay(self):
        """Timer callback for throttled live cursor overlay refresh."""
        self._update_cursor_live_overlay()

    def on_cursor_moving(self):
        """Interactive drag/scroll: throttled overlay; table stats wait for release."""
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
        """Committed change (mouseup / add / remove): full table + overlay update."""
        self._cursor_debounce_timer.stop()
        self._refresh_active_cursor_combo()
        self._sync_radius_slider()
        self._refresh_after_cursor_edit()

    def _on_radius_slider_released(self):
        """Recompute segmentation after the radius slider is released."""
        if self.mode == "cursor" and self.phasor.cursors and self.data.real_cal is not None:
            self._cursor_debounce_timer.stop()
            self._refresh_after_cursor_edit()

    def _deferred_cursor_compute(self):
        """Debounced timer callback to recompute cursor cluster stats."""
        if self.mode == "cursor" and self.phasor.cursors and self.data.real_cal is not None:
            self._compute_cursor()

    def _gmm_k_max(self) -> int:
        """Parse fixed k or BIC max-k from the GMM component field."""
        text = self.edit_ncomp.text().strip() if hasattr(self, "edit_ncomp") else "3"
        try:
            return max(1, min(12, int(text)))
        except ValueError:
            return 3

    def _gmm_sigma(self) -> float:
        """Parse GMM ellipse contour scale (sigma) from the UI field."""
        text = self.edit_gmm_sigma.text().strip() if hasattr(self, "edit_gmm_sigma") else "2.0"
        try:
            return max(0.5, min(6.0, float(text)))
        except ValueError:
            return 2.0

    # ---- GMM ---------------------------------------------------------------
    def _fit_gmm_worker(self):
        """Fit phasor GMM on valid pixels (optionally with BIC auto-k)."""
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
        """Fit GMM on phasor pixels and draw cluster ellipses on the plot."""
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
        """Fit GMM on every loaded sample and keep results per image for Export all."""
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
        """Show lifetimes at a clicked phasor coordinate and mark nearest pixel."""
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
        """Mark the phasor point for a clicked image pixel and log lifetimes."""
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
        """Mark nearest valid pixel to the clicked phasor on the image view."""
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
        """Compute cluster lifetimes and paint pseudo-color segmentation on the image."""
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
        """Format a lifetime value for the results table (dash if non-finite)."""
        return "—" if not np.isfinite(value) else f"{value:.3f}"

    def _cursor_masks_colors(self):
        """Boolean masks (within valid pixels) and colors for the current circles."""
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
        """Aggregate lifetimes inside cursor ROIs and paint segmentation masks."""
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
        """Label pixels by GMM components and paint cluster segmentation."""
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
        """Return the boolean mask of pixels included in the phasor plot."""
        if self.data.real_cal is None:
            return None
        return self.data.valid_mask()

    def _mask_like_phasor(self, arr):
        """Apply the same finite-pixel mask used for the phasor histogram."""
        if arr is None:
            return None
        out = np.asarray(arr, dtype=float).copy()
        valid = self._phasor_valid_mask()
        if valid is not None and valid.shape == out.shape:
            out[~valid] = np.nan
        return out

    def _photon_image_filtered(self):
        """Photon counts with NaN where excluded from phasor (threshold + invalid phasor)."""
        base = self.data.mean_thr if self.data.mean_thr is not None else self.data.mean_raw
        return self._mask_like_phasor(base)

    def _segmentation_intensity(self):
        """Background for pseudo_color — photons masked like the phasor plot."""
        raw = self._photon_image_filtered()
        if raw is None:
            return np.zeros((2, 2), dtype=np.float64)
        out = np.zeros(np.shape(raw), dtype=np.float64)
        finite = np.isfinite(raw)
        out[finite] = np.nan_to_num(raw[finite])
        return out

    def _paint(self, masks, colors):
        """Build pseudo-color overlay from cluster masks and refresh the image."""
        intensity = self._segmentation_intensity()
        overlay = pseudo_color(*[masks[k] for k in range(len(masks))],
                               intensity=intensity, colors=np.array(colors))
        self.last_overlay = np.clip(np.asarray(overlay), 0, 1)
        self.chk_overlay.setChecked(True); self.refresh_image()

    def _on_image_view_changed(self):
        """Show the selected View map; turn Overlay off so it is not hidden."""
        if hasattr(self, "chk_overlay") and self.chk_overlay.isChecked():
            self.chk_overlay.blockSignals(True)
            self.chk_overlay.setChecked(False)
            self.chk_overlay.blockSignals(False)
        self.refresh_image()

    def refresh_image(self):
        """Show segmentation overlay or the selected base image view."""
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
        """Render the view selected above the image."""
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
        """Draw a µm scale bar on the image when pixel size is known."""
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
        """Return 2nd–98th percentile range for tau map color scaling."""
        finite = np.asarray(arr)[np.isfinite(arr)]
        if finite.size == 0:
            return None, None
        return float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))

    # ---- results table -----------------------------------------------------
    def _fill_table(self):
        """Populate the results table from current cluster_stats."""
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
        """Sync an edited table label back to cluster_stats and phasor cursors."""
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
        """Create a read-only table cell for numeric or status columns."""
        it = QtWidgets.QTableWidgetItem(text); it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable); return it

    @staticmethod
    def _editable_group(text):
        """Create an editable compare-table cell for sample group names."""
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable)
        it.setToolTip("Group name for this file (e.g. Tumor, Control)")
        return it

    @staticmethod
    def _editable_sample(text):
        """Create an editable compare-table cell for sample display names."""
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable)
        return it

    @staticmethod
    def _rgb_hex(c):
        """Convert an RGB float triplet to an ARGB hex string for Excel export."""
        return "FF" + "".join(f"{int(round(255 * max(0.0, min(1.0, x)))):02X}" for x in c[:3])

    # ---- export ------------------------------------------------------------
    def export_all(self):
        """Export plots, maps, tables, and session data to a chosen folder."""
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

