"""Main application window."""
import os
import sys
import time

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from flim_phasors import __version__
from flim_phasors.analysis import fit_phasor_gmm, label_pixels_by_gmm, lifetimes_at_phasor
from flim_phasors.export_bundle import export_analysis_bundle
from flim_phasors.constants import (
    COMPARE_STYLE_MAP,
    CURSOR_SHAPES,
    FILTER_MODES,
    FLIM_FILE_FILTER,
    IMAGE_VIEW_ITEMS,
)
from flim_phasors.calibration import ReferenceCalibration, compute_reference_phasor
from flim_phasors.calibration import clear_calibration_cache
from flim_phasors.data import PhasorData
from flim_phasors.io import is_supported_flim_path
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
    dataset_short_label,
)

try:
    from sklearn.mixture import GaussianMixture
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

from phasorpy.cursor import mask_from_circular_cursor, mask_from_elliptic_cursor, pseudo_color
from phasorpy.lifetime import phasor_to_apparent_lifetime, phasor_to_normal_lifetime


class MainWindow(EnhancementsMixin, QtWidgets.QMainWindow):
    def __init__(self):
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
        self._label_signal_connected = False
        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.setSingleShot(True)
        self._paint_timer.setInterval(250)   # ms after interaction stops -> full recompute
        self._paint_timer.timeout.connect(self._deferred_full_compute)
        self._build_ui()
        self._init_enhancements()
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Delete), self, self.remove_cursor)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Backspace), self, self.remove_cursor)

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)

        _small = "font-size: 10px;"
        _lbl_file = f"color: gray; {_small}"

        def _tab_page():
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
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(400)
        self.txt_log.setMinimumHeight(72)
        self.txt_log.setMaximumHeight(100)
        self.txt_log.setPlaceholderText("Activity log…")
        self.txt_log.setStyleSheet(
            f"font-family: Consolas, monospace; {_small} background: palette(base);")
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
            "Sample shown in the plots and used for Calibrate / Apply below.")
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
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate")
        self.btn_calibrate.setToolTip(
            "Compute reference g/s from the file chosen under Reference (no sample processing).")
        self.btn_calibrate.clicked.connect(self.calibrate_reference)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(lambda: self.apply_processing(scope="active"))
        self.btn_apply_all = QtWidgets.QPushButton("Apply all")
        self.btn_apply_all.setToolTip(
            "Preprocess every loaded sample. In multi-image mode each sample uses "
            "its own saved filter settings.")
        self.btn_apply_all.clicked.connect(lambda: self.apply_processing(scope="all"))
        self.btn_apply_all.setVisible(False)
        row_cal_apply.addWidget(self.btn_calibrate)
        row_cal_apply.addWidget(self.btn_apply, 1)
        row_cal_apply.addWidget(self.btn_apply_all, 1)
        prg.addLayout(row_cal_apply, 10, 0, 1, 4)
        self.sp_harm.valueChanged.connect(self._on_harm_or_ref_setting_changed)
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
            "Click a row to activate that sample. Double-click Group to rename. "
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
        row_grp = QtWidgets.QHBoxLayout()
        row_grp.addWidget(QtWidgets.QLabel("Group"))
        self.edit_group = QtWidgets.QLineEdit()
        self.edit_group.setPlaceholderText("e.g. condition A")
        self.edit_group.setToolTip(
            "Label for the active sample (overlay legend and table).")
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
        self._compare_sel_buttons = (btn_cmp_all, btn_cmp_none)
        self._multi_detail_widgets = (
            self._proc_active_row, self.cb_sample,
            self._multi_strip, self.table_compare, self.btn_rmimg,
            self.lbl_editing, self.btn_apply_settings_all,
            self.edit_group, btn_grp,
            self.chk_compare, self.cb_compare_style, self.cb_compare_group,
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
        self.sld_radius = QtWidgets.QSlider(Qt.Orientation.Horizontal); self.sld_radius.setRange(5, 400); self.sld_radius.setValue(50)
        self.sld_radius.valueChanged.connect(self.on_radius_slider)
        self.lbl_radius = QtWidgets.QLabel("r=0.05")
        cbl.addWidget(self.sld_radius, 3, 0, 1, 2); cbl.addWidget(self.lbl_radius, 3, 2)
        ml.addWidget(self.cursor_box)

        self.gmm_box = QtWidgets.QWidget()
        gbl = QtWidgets.QGridLayout(self.gmm_box); gbl.setContentsMargins(0, 0, 0, 0)
        gbl.addWidget(QtWidgets.QLabel("k"), 0, 0)
        self.edit_ncomp = QtWidgets.QLineEdit("3")
        self.edit_ncomp.setValidator(QtGui.QIntValidator(1, 12, self))
        self.edit_ncomp.setFixedWidth(32)
        self.edit_ncomp.setToolTip("Number of GMM components (1–12).")
        gbl.addWidget(self.edit_ncomp, 0, 1)
        gbl.addWidget(QtWidgets.QLabel("Cov"), 0, 2)
        self.cb_cov = QtWidgets.QComboBox(); self.cb_cov.addItems(["full", "tied", "diag", "spherical"])
        gbl.addWidget(self.cb_cov, 0, 3)
        self.chk_bic = QtWidgets.QCheckBox("BIC auto-k"); gbl.addWidget(self.chk_bic, 1, 0, 1, 2)
        gbl.addWidget(QtWidgets.QLabel("σ"), 2, 0)
        self.edit_gmm_sigma = QtWidgets.QLineEdit("2.0")
        self.edit_gmm_sigma.setValidator(QtGui.QDoubleValidator(0.5, 6.0, 2, self))
        self.edit_gmm_sigma.setFixedWidth(40)
        self.edit_gmm_sigma.setToolTip(
            "Ellipse scale for phasorpy phasor_cluster_gmm (95% contour at 2.0).")
        gbl.addWidget(self.edit_gmm_sigma, 2, 1)
        b_fit = QtWidgets.QPushButton("Fit GMM"); b_fit.clicked.connect(self.fit_gmm)
        b_clr_gmm = QtWidgets.QPushButton("Clear"); b_clr_gmm.clicked.connect(self.clear_gmm)
        gbl.addWidget(b_fit, 2, 2); gbl.addWidget(b_clr_gmm, 2, 3)
        self.gmm_box.setVisible(False); ml.addWidget(self.gmm_box)
        analyze_l.addWidget(gb_mode)

        # ---- actions ----
        self.gb_act = QtWidgets.QGroupBox("Results")
        al = QtWidgets.QVBoxLayout(self.gb_act)
        al.setSpacing(3)
        row_paint = QtWidgets.QHBoxLayout()
        b_paint = QtWidgets.QPushButton("Paint"); b_paint.clicked.connect(self.compute_and_paint)
        self.chk_live = QtWidgets.QCheckBox("Live"); self.chk_live.setChecked(True)
        self.chk_overlay = QtWidgets.QCheckBox("Overlay"); self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(self.refresh_image)
        row_paint.addWidget(b_paint, 1); row_paint.addWidget(self.chk_live); row_paint.addWidget(self.chk_overlay)
        al.addLayout(row_paint)
        btn_export = QtWidgets.QPushButton("Export all…")
        btn_export.setToolTip(
            "Save a folder with phasor plot, maps for every sample, tables, session JSON, and Excel (if openpyxl).")
        btn_export.clicked.connect(self.export_all)
        al.addWidget(btn_export)
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
        lv.addWidget(NavigationToolbar(self.phasor, self)); lv.addWidget(self.phasor)
        rw = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(rw)
        row_img_view = QtWidgets.QHBoxLayout()
        row_img_view.addWidget(QtWidgets.QLabel("View"))
        self.cb_image_view = QtWidgets.QComboBox()
        self.cb_image_view.addItems(list(IMAGE_VIEW_ITEMS))
        self.cb_image_view.setToolTip(
            "Pixel maps from the current Apply settings (same mask as the phasor plot).")
        self.cb_image_view.currentIndexChanged.connect(self.refresh_image)
        row_img_view.addWidget(self.cb_image_view, 1)
        rv.addLayout(row_img_view)
        self.image = ImageCanvas(self)
        rv.addWidget(NavigationToolbar(self.image, self)); rv.addWidget(self.image)
        plots.addWidget(lw); plots.addWidget(rw); plots.setSizes([700, 700])

        self.table = QtWidgets.QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["#", "Color", "Label (what you see)", "g", "s",
             "tau_phi (ns)", "tau_mod (ns)", "tau_normal (ns)",
             "Pixels", "Area %"])
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(220)

        rightside = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        rightside.addWidget(plots); rightside.addWidget(self.table); rightside.setSizes([720, 220])
        main.addWidget(rightside, 1)
        main.addWidget(panel_wrap)

        self.status = self.statusBar()
        self._log(
            "Ready — load sample(s), choose Reference, Calibrate, then Apply to preprocess.")

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
        if seconds < 1.0:
            return f"{seconds * 1000:.0f} ms"
        return f"{seconds:.2f} s"

    # ---- show/hide filter params ------------------------------------------
    def on_filter_change(self, mode):
        is_kernel = mode in ("median", "gaussian", "signal median", "signal gaussian")
        is_paw = mode == "pawflim"
        for w in self.row_msize:
            w.setVisible(is_kernel)
        for w in self.row_psigma:
            w.setVisible(is_paw)

    def _connect_per_sample_proc_autosave(self):
        for w in (
            self.sp_harm, self.sp_freq, self.sp_reflt, self.sp_msize, self.sp_mrep,
            self.sp_psigma, self.sp_plevels, self.sp_thr,
        ):
            w.valueChanged.connect(self._on_per_sample_proc_changed)
        self.cb_filter.currentTextChanged.connect(self._on_per_sample_proc_changed)
        self.chk_detect_harm.toggled.connect(self._on_per_sample_proc_changed)
        self.cb_channel.currentIndexChanged.connect(self._on_per_sample_proc_changed)

    def _on_per_sample_proc_changed(self, *_args):
        if self._loading_proc_ui or not per_sample_processing(self):
            return
        self._save_proc_from_ui(self.data)
        self._refresh_compare_list()

    def _init_dataset_proc_settings(self, d: PhasorData):
        if d.processing_settings is None:
            d.processing_settings = capture_processing_from_ui(self)

    def _save_proc_from_ui(self, d: PhasorData):
        if d is None:
            return
        if per_sample_processing(self):
            d.processing_settings = capture_processing_from_ui(self)

    def _load_proc_to_ui(self, d: PhasorData):
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
        self._on_sample_picker_change(idx)

    def _on_sample_picker_change(self, idx: int):
        if self._table_sel_lock or not (0 <= idx < len(self.datasets)):
            return
        if idx == self.active_idx:
            return
        self._activate_dataset(idx)

    def _sync_sample_table_selection(self):
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
        return self._settings.value(key, fallback) or ""

    def choose_sample(self):
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
                "Use PicoQuant .ptu or Imspector .tif / .tiff FLIM stacks.",
            )
            return
        if skipped:
            self._log(f"Skipped {skipped} unsupported file(s).")
        if not self._prepare_sample_load(supported):
            return
        self._load_sample_paths(supported)

    def _prepare_sample_load(self, paths):
        """Confirm how new paths merge with the current session. Returns False if cancelled."""
        has_current = self.data.signal_full is not None
        batch = len(paths) > 1

        if batch:
            self.chk_multi.setChecked(True)
            if has_current:
                mbox = QtWidgets.QMessageBox(self)
                mbox.setWindowTitle("Load multiple samples")
                mbox.setText(f"Load {len(paths)} file(s).")
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

    def _load_sample_paths(self, paths):
        """Decode FLIM files into memory (no phasor preprocessing until Apply)."""
        n = len(paths)
        loaded = []
        t_decode = 0.0
        frame = int(self.sp_frame.value()) if hasattr(self, "sp_frame") else -1
        for i, path in enumerate(paths):
            d = PhasorData()
            try:
                (shape, nch), elapsed = self._run_busy(
                    f"Decoding {i + 1}/{n}: {os.path.basename(path)}…",
                    lambda p=path, ds=d, fr=frame: ds.load_sample(p, frame=fr),
                )
                t_decode += elapsed
                loaded.append((d, path, shape, nch))
            except CancelledError:
                if not loaded:
                    return
                break
            except Exception as e:
                self._log(f"Load error ({os.path.basename(path)}): {e}")
                QtWidgets.QMessageBox.critical(
                    self, "Load error", f"{os.path.basename(path)}:\n{e}")
                if not loaded:
                    return
                break

        for d, path, shape, nch in loaded:
            self._activate_new_dataset(d)
            self._log(
                f"Loaded {os.path.basename(path)} — {shape[1]}×{shape[0]}, {nch} ch, "
                f"{d.frequency:.2f} MHz · {format_memory_line(d)}")

        self._restore_ui_for_active()
        self._refresh_image_combo()
        self._update_apply_buttons()
        self._update_phasor_display()
        self.refresh_image()
        self._log(
            f"{len(loaded)} sample(s) decoded ({self._fmt_elapsed(t_decode)}) — "
            "choose Reference, Calibrate, then Apply.")

    def _effective_ref_path(self, d=None):
        d = d or self.data
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return self.shared_ref_path
        return d.ref_path or None

    def _ref_channel_for_dataset(self, d):
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
        if checked and self.shared_ref_path:
            self._propagate_shared_reference()
        self._restore_ui_for_active()
        self._log(
            "Shared reference on — one reference file calibrates all samples."
            if checked else
            "Shared reference off — each sample can use its own reference (active sample).")

    def _all_datasets(self):
        seen = []
        for d in [self.data] + list(self.datasets):
            if d is not None and d not in seen:
                seen.append(d)
        return seen

    def _reference_harmonic_for_cal(self):
        h = int(self.sp_harm.value())
        if self.cb_filter.currentText() == "pawflim":
            return [h, 2 * h]
        return h

    def _effective_ref_file_path(self):
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return self.shared_ref_path
        if self.data.ref_path:
            return self.data.ref_path
        # Reference… may run before samples load — keep the last picked path.
        return self.shared_ref_path or ""

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
        if self.chk_manual_cal.isChecked():
            self._apply_manual_calibration_fields()
        else:
            self._sync_manual_fields_from_calibration()
        self._update_calibration_display()
        self._update_ref_preview()
        self._mark_calibration_current()
        self._ensure_compare_overlay_off()
        self._log(
            f"Reference g/s stored (g={cal.mean_g:.4f}, s={cal.mean_s:.4f}) "
            f"— scalar calibration; reference file not kept in RAM "
            f"({self._fmt_elapsed(elapsed)}).")
        return True

    def _sync_manual_fields_from_calibration(self):
        self.edit_ref_g.blockSignals(True)
        self.edit_ref_s.blockSignals(True)
        self.edit_ref_g.setText(f"{self.ref_calibration.mean_g:.5f}")
        self.edit_ref_s.setText(f"{self.ref_calibration.mean_s:.5f}")
        self.edit_ref_g.blockSignals(False)
        self.edit_ref_s.blockSignals(False)

    def _apply_manual_calibration_fields(self):
        try:
            self.ref_calibration.manual_g = float(self.edit_ref_g.text().strip())
            self.ref_calibration.manual_s = float(self.edit_ref_s.text().strip())
        except ValueError:
            return
        self.ref_calibration.use_manual = True
        self.ref_calibration.mean_g = self.ref_calibration.manual_g
        self.ref_calibration.mean_s = self.ref_calibration.manual_s

    def _update_calibration_display(self):
        if hasattr(self, "_update_panel_status"):
            self._update_panel_status()
        cal = self.ref_calibration
        if not cal.is_active:
            self.lbl_cal_display.setText("(uncalibrated — load a reference file)")
            return
        src = "manual g,s" if cal.use_manual else os.path.basename(cal.source_path or "")
        self.lbl_cal_display.setText(
            f"{src}  |  g={cal.mean_g:.4f}, s={cal.mean_s:.4f}  |  "
            f"τ_ref={self.sp_reflt.value():.2f} ns  |  ch {cal.channel}")

    def _on_manual_cal_toggled(self, checked):
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

    def _on_harm_or_ref_setting_changed(self, *_args):
        """Harmonic/filter changes need a fresh Calibrate before Apply."""
        if self.ref_calibration.values_ready and not self.chk_manual_cal.isChecked():
            self.ref_calibration.values_ready = False
            self.ref_calibration._maps = None
        if self.ref_calibration.is_active or self._effective_ref_file_path():
            self._update_calibration_display()
            self._update_calibration_stale_style()

    def _clear_calibration(self):
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
        self._set_reference_path(path)

    def _set_reference_path(self, path: str):
        """Store reference path only; decode on Calibrate or automatically on Apply."""
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
        if self.ref_calibration.values_ready:
            self._update_calibration_display()
        else:
            self.lbl_cal_display.setText(
                f"{os.path.basename(path)} selected — click Calibrate to compute g/s.")
        self._log(f"Reference file selected: {os.path.basename(path)} (not decoded yet).")

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
            self._set_reference_path(path)
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
        self.shared_ref_channel = min(self.shared_ref_channel, ref_nch - 1)
        if self.data.signal_full is not None:
            self.shared_ref_channel = min(self.data.channel, ref_nch - 1)
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
    def _activate_new_dataset(self, d):
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
        self.data = d
        self._restore_ui_for_active()
        if self.chk_multi.isChecked():
            self._refresh_compare_list()

    @staticmethod
    def _compact_filename(path, fallback="(none)", max_len=30):
        name = os.path.basename(path) if path else fallback
        if len(name) > max_len:
            return name[: max_len - 1] + "…"
        return name

    def _update_proc_active_label(self):
        if not hasattr(self, "lbl_proc_active"):
            return
        multi = hasattr(self, "chk_multi") and self.chk_multi.isChecked() and len(self.datasets) > 1
        if multi:
            self.lbl_proc_active.setText("")
            return
        if self.data.signal_full is not None:
            self.lbl_proc_active.setText(
                self._compact_filename(self.data.sample_path, "(no sample)"))
        else:
            self.lbl_proc_active.setText("(no sample)")

    def _update_panel_status(self):
        if not hasattr(self, "lbl_panel_status"):
            return
        if self.data.signal_full is None:
            self.lbl_panel_status.setText("No sample loaded")
            return
        name = self._compact_filename(self.data.sample_path, "(no sample)")
        if self.ref_calibration.is_active:
            cal = f"g={self.ref_calibration.mean_g:.3f} s={self.ref_calibration.mean_s:.3f}"
        else:
            cal = "uncalibrated"
        n = len(self.datasets)
        if n > 1:
            self.lbl_panel_status.setText(
                f"Active: {name}  ·  {cal}  ·  {n} samples")
        else:
            self.lbl_panel_status.setText(f"Active: {name}  ·  {cal}")

    def _restore_ui_for_active(self):
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
        self._load_proc_to_ui(d)
        self._update_frame_control()
        if hasattr(self, "edit_group"):
            self.edit_group.blockSignals(True)
            self.edit_group.setText((d.group_name or "").strip())
            self.edit_group.blockSignals(False)
        self._sync_sample_table_selection()

    def _update_ref_channel_combo(self):
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
        for w in self._multi_detail_widgets:
            w.setEnabled(enabled)

    def _process_all_loaded_datasets(self, *, use_ui_settings=True):
        """Re-run phasor pipeline on every loaded sample (caller may wrap in _run_busy)."""
        if self.chk_multi.isChecked() and self.datasets:
            targets = list(self.datasets)
        else:
            targets = [self.data]
        saved = self.data
        saved_idx = self.active_idx
        for d in targets:
            if d.signal_full is None:
                continue
            self._run_processing_on_dataset(d, use_ui_settings=use_ui_settings)
        if 0 <= saved_idx < len(self.datasets):
            self.data = self.datasets[saved_idx]
            self.active_idx = saved_idx
        else:
            self.data = saved

    def _refresh_views_after_processing(self):
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

    def _recompute_all_datasets(self):
        """Re-run phasor pipeline on every loaded sample (e.g. after shared reference change)."""
        try:
            self._run_busy(
                "Recomputing phasor maps…",
                lambda: self._process_all_loaded_datasets(use_ui_settings=True),
            )
        except Exception as e:
            self._log(f"Processing error: {e}")
            QtWidgets.QMessageBox.critical(self, "Processing error", str(e))
            return
        self._refresh_views_after_processing()

    def _refresh_image_combo(self):
        """Refresh sample table, dropdown, and overlay filters."""
        self._refresh_compare_list()
        self._refresh_compare_group_filter()
        self._refresh_sample_combo()
        self._sync_sample_table_selection()

    def _refresh_compare_group_filter(self):
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
        text = self.edit_group.text().strip() if hasattr(self, "edit_group") else ""
        if self.chk_multi.isChecked() and 0 <= self.active_idx < len(self.datasets):
            self.datasets[self.active_idx].group_name = text
        else:
            self.data.group_name = text
        self._refresh_compare_list()
        self._refresh_compare_group_filter()
        if self.data.real_cal is not None:
            self._update_phasor_display()
        if text:
            self._log(f"Group set to “{text}” for active sample.")

    def _update_frame_control(self):
        d = self.data
        self.sp_frame.blockSignals(True)
        if d.signal_full is None:
            self.sp_frame.setEnabled(False)
            self.sp_frame.setRange(-1, 0)
            self.sp_frame.setValue(-1)
        elif "T" in d.signal_full.dims and int(d.signal_full.sizes.get("T", 1)) > 1:
            n = int(d.signal_full.sizes["T"])
            self.sp_frame.setEnabled(True)
            self.sp_frame.setRange(-1, n - 1)
            self.sp_frame.setValue(int(getattr(d, "frame_index", -1)))
        else:
            self.sp_frame.setEnabled(False)
            self.sp_frame.setValue(-1)
        self.sp_frame.blockSignals(False)

    def on_frame_change(self, value):
        if self.data.sample_path:
            if int(value) == int(getattr(self.data, "frame_index", -1)):
                return
            self.data.frame_index = int(value)
            self._reload_active_sample()

    def _reload_active_sample(self):
        path = self.data.sample_path
        if not path or not os.path.isfile(path):
            return
        ch = self.data.channel
        frame = self.data.frame_index
        try:
            self._run_busy(
                f"Reloading frame {frame} ({os.path.basename(path)})…",
                lambda: self.data.load_sample(path, frame=frame),
            )
        except Exception as e:
            self._log(f"Reload error: {e}")
            QtWidgets.QMessageBox.critical(self, "Reload error", str(e))
            return
        self.data.channel = min(ch, max(0, self.data.n_channels - 1))
        self.apply_processing()

    def _compare_show_item(self, row):
        return self.table_compare.item(row, 0)

    def _compare_dataset_index(self, row):
        it = self._compare_show_item(row)
        if it is None:
            return -1
        return int(it.data(Qt.ItemDataRole.UserRole))

    def _refresh_compare_list(self):
        checked = {}
        was_ready = {}
        for row in range(self.table_compare.rowCount()):
            idx = self._compare_dataset_index(row)
            it = self._compare_show_item(row)
            if idx >= 0 and it is not None:
                checked[idx] = it.checkState() == Qt.CheckState.Checked
            status_it = self.table_compare.item(row, 7)
            if idx >= 0:
                was_ready[idx] = status_it is not None and status_it.text() == "ready"

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
            sample = self._ro(dataset_short_label(d, i))
            if i == self.active_idx:
                sample.setFont(QtGui.QFont(sample.font().family(), sample.font().pointSize(),
                                           QtGui.QFont.Weight.Bold))
                sample.setToolTip("Selected — settings below apply to this sample")
            group = self._editable_group((d.group_name or "").strip())
            stash = getattr(d, "processing_settings", None) or {}
            filt = self._ro(
                str(stash.get("filter_mode", filter_label_for_dataset(self, d))))
            min_n = self._ro(str(int(stash.get("intensity_min", 0))))
            ref_lbl = self._ro(
                self._effective_ref_label(d) if hasattr(self, "_effective_ref_label")
                else "—")
            status = self._ro("ready" if ready else "pending")
            if not ready:
                status.setForeground(QtGui.QBrush(Qt.GlobalColor.gray))

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
        cmp_on = compare_available and self.chk_compare.isChecked()
        self.chk_compare.setEnabled(compare_available)
        self.cb_compare_style.setEnabled(cmp_on)
        if hasattr(self, "table_compare"):
            self.table_compare.setEnabled(len(self.datasets) > 1)
        for btn in getattr(self, "_compare_sel_buttons", ()):
            btn.setEnabled(cmp_on)
        self._update_multi_strip()

    def _compare_set_all_checks(self, checked):
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
        self._compare_set_all_checks(True)

    def _compare_select_none(self):
        self._compare_set_all_checks(False)

    def _on_compare_table_changed(self, row, column):
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
                self._update_phasor_display()
            return
        if column != 0:
            return
        self._on_compare_ui_changed()

    def _on_sample_table_selection(self):
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
                "label": dataset_display_label(d, idx),
                "color": categorical_rgb(idx),
                "visible": show is not None and show.checkState() == Qt.CheckState.Checked,
                "index": idx,
            })
        return layers

    def _compare_style_key(self):
        return COMPARE_STYLE_MAP.get(self.cb_compare_style.currentText(), "cloud")

    def _update_phasor_display(self, status_note=""):
        layers = self._build_compare_layers()
        visible = [L for L in layers if L.get("visible") and L["data"].real_cal is not None]
        compare_on = (
            self.chk_multi.isChecked()
            and self.chk_compare.isChecked()
            and len(visible) >= 1
        )
        self.phasor.update_display(
            self.data,
            compare_enabled=compare_on,
            compare_style=self._compare_style_key(),
            compare_layers=layers if compare_on else None,
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

    def _on_compare_ui_changed(self, *_args):
        self._set_compare_controls_enabled(
            self.chk_multi.isChecked() and len(self.datasets) >= 2)
        if self.data.real_cal is not None or self.chk_compare.isChecked():
            self._update_phasor_display()

    def on_multi_toggle(self, checked):
        self._set_multi_detail_enabled(checked)
        if checked:
            # adopt the currently loaded image as the first slot (non-destructive)
            has_image = self.data.signal_full is not None
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
        if not (0 <= idx < len(self.datasets)):
            return
        if per_sample_processing(self):
            self._save_proc_from_ui(self.data)
        self.active_idx = idx
        self.data = self.datasets[idx]
        self._restore_ui_for_active()
        self._refresh_compare_list()
        self._update_multi_strip()
        self._update_phasor_display()
        self.last_overlay = None
        self.cluster_stats = []
        if self.chk_live.isChecked() and self.mode == "cursor" and self.phasor.cursors \
                and self.data.real_cal is not None:
            self._compute_cursor()
        else:
            self._fill_table()
            self.chk_overlay.blockSignals(True)
            self.chk_overlay.setChecked(False)
            self.chk_overlay.blockSignals(False)
            self.refresh_image()
        self._sync_sample_table_selection()
        self._log(f"Selected sample {idx + 1}: {dataset_display_label(self.data, idx)}")
        self.activateWindow()
        self.raise_()

    def apply_settings_to_all(self):
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
        if self.data.signal_full is None:
            return
        self.data.channel = max(0, idx)
        if self.data.real_cal is not None:
            self.apply_processing()
        else:
            self._log(f"Sample channel {self.data.channel} — click Apply to preprocess.")

    def on_ref_channel_change(self, idx):
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
            self._log("Reference channel changed — click Calibrate to update g/s, then Apply.")
            return

    # ---- processing --------------------------------------------------------
    def _active_calibration(self):
        if self.chk_manual_cal.isChecked():
            self._apply_manual_calibration_fields()
        return self.ref_calibration if self.ref_calibration.is_active else None

    def _run_processing_on_dataset(self, d, *, use_ui_settings=False):
        run_processing_on_dataset(self, d, use_ui_settings=use_ui_settings)

    def apply_processing(self, scope: str = "auto"):
        if self.data.signal_full is None:
            QtWidgets.QMessageBox.information(self, "No data", "Load a sample first.")
            return
        multi = len(self.datasets) > 1
        if scope == "auto":
            scope = "active"
        if per_sample_processing(self):
            self._save_proc_from_ui(self.data)
        self.data.pixel_size_um = self.sp_pixel_um.value() if hasattr(self, "sp_pixel_um") else 0.0
        if not self._calibration_ready_for_apply():
            QtWidgets.QMessageBox.information(
                self,
                "Calibration",
                "A reference file is selected but g/s are not set yet.\n"
                "Click Calibrate (decodes the reference once) or Load cal…, then Apply.",
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
        self._refresh_active_cursor_combo(select_idx=idx)
        self._sync_radius_slider()

    def on_active_cursor_change(self, combo_idx):
        if self.mode != "cursor" or combo_idx < 0:
            return
        self.phasor.select_cursor(combo_idx, emit=False)
        self._sync_radius_slider()

    def add_cursor(self):
        if hasattr(self, "_push_cursor_undo"):
            self._push_cursor_undo()
        r = self.sld_radius.value() * 0.001
        kind = "ellipse" if self.cb_cursor_shape.currentText() == "Ellipse" else "circle"
        aspect = self.sld_aspect.value() / 100.0
        self.phasor.add_cursor(
            radius=r, kind=kind,
            radius_minor=r * aspect if kind == "ellipse" else None)
        self._refresh_active_cursor_combo()

    def remove_cursor(self):
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
        self.phasor.clear_cursors()
        self._refresh_active_cursor_combo()
        self._refresh_after_cursor_edit()
        self._log("All circles cleared.")

    def _refresh_after_cursor_edit(self):
        """Update segmentation overlay and table whenever circles are added/removed/moved."""
        self._paint_timer.stop()
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
        if self.mode != "cursor":
            return
        if event.key in ("delete", "backspace"):
            self.remove_cursor()

    def clear_gmm(self):
        self.phasor.clear_gmm()
        if hasattr(self, "gmm"):
            del self.gmm
        if hasattr(self, "_gmm_fit"):
            del self._gmm_fit
        self.last_overlay = None
        self.cluster_stats = []
        self._fill_table()
        self.refresh_image()
        self._log("GMM fit cleared.")

    def on_radius_slider(self, v):
        r = v * 0.001
        self.lbl_radius.setText(f"r={r:.2f}")
        self.phasor.set_selected_radius(r)

    def _sync_radius_slider(self):
        i = self.phasor.selected
        if 0 <= i < len(self.phasor.cursors):
            r = self.phasor.cursors[i]["radius"]
            self.sld_radius.blockSignals(True); self.sld_radius.setValue(int(round(r * 1000))); self.sld_radius.blockSignals(False)
            self.lbl_radius.setText(f"{r:.3f}")

    def _live_active(self):
        return self.chk_live.isChecked() and self.mode == "cursor" and self.data.real_cal is not None

    def on_cursor_moving(self):
        """Interactive drag/scroll/slider: repaint overlay only (fast), defer the rest."""
        self._sync_radius_slider()
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
            self.chk_overlay.blockSignals(True); self.chk_overlay.setChecked(True); self.chk_overlay.blockSignals(False)
        self.image.update_overlay(self.last_overlay, title=f"Segmentation ({self.mode})")
        self._paint_timer.start()   # full recompute (lifetimes + table) once interaction settles

    def on_cursor_changed(self):
        """Committed change: sync slider and refresh segmentation (always, not only live-paint)."""
        self._refresh_active_cursor_combo()
        self._sync_radius_slider()
        self._refresh_after_cursor_edit()

    def _deferred_full_compute(self):
        if self._live_active() and self.phasor.cursors:
            self._compute_cursor()

    def _gmm_k_max(self) -> int:
        text = self.edit_ncomp.text().strip() if hasattr(self, "edit_ncomp") else "3"
        try:
            return max(1, min(12, int(text)))
        except ValueError:
            return 3

    def _gmm_sigma(self) -> float:
        text = self.edit_gmm_sigma.text().strip() if hasattr(self, "edit_gmm_sigma") else "2.0"
        try:
            return max(0.5, min(6.0, float(text)))
        except ValueError:
            return 2.0

    # ---- GMM ---------------------------------------------------------------
    def fit_gmm(self):
        if not HAVE_SKLEARN:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "pip install scikit-learn"); return
        if not self.rb_gmm.isChecked():
            self.rb_gmm.setChecked(True)
        if self.data.real_cal is None:
            QtWidgets.QMessageBox.information(self, "GMM", "Load data and click Apply first."); return
        m = self.data.valid_mask()
        if m.sum() < 10:
            QtWidgets.QMessageBox.information(self, "GMM", "Not enough valid pixels."); return
        g, s = self.data.real_cal, self.data.imag_cal
        cov = self.cb_cov.currentText()
        sigma = self._gmm_sigma()
        try:
            if self.chk_bic.isChecked():
                X = np.column_stack([g[m], s[m]])
                best_n, best_bic = 1, np.inf
                for n in range(1, self._gmm_k_max() + 1):
                    gm = GaussianMixture(n, covariance_type=cov, random_state=0).fit(X)
                    b = gm.bic(X)
                    if b < best_bic:
                        best_bic, best_n = b, n
                n_clusters = best_n
                self._log(f"GMM auto-selected {best_n} components (BIC={best_bic:.0f}).")
            else:
                n_clusters = self._gmm_k_max()
            self._gmm_fit = fit_phasor_gmm(
                g, s, clusters=n_clusters, sigma=sigma, covariance_type=cov)
            self._log(f"phasorpy GMM: {n_clusters} cluster(s), σ={sigma:.1f}.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "GMM fit failed", str(e)); return
        n = len(self._gmm_fit[0])
        colors = [categorical_rgb(k) for k in range(n)]
        self.phasor.show_gmm_ellipses(*self._gmm_fit, colors)
        self._compute_gmm()
        return

    def _on_phasor_click(self, g, s):
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
        disp = self._photon_image_filtered()
        if disp is not None:
            log_scale = getattr(self, "chk_log_display", None) and self.chk_log_display.isChecked()
            auto_c = not getattr(self, "chk_auto_contrast", None) or self.chk_auto_contrast.isChecked()
            self.image.show_intensity(
                disp, log_scale=log_scale, auto_contrast=auto_c,
                title="Nearest pixel to phasor click")
            self.image.ax.plot(x, y, "c+", ms=14, mew=2)
            self.image.draw_idle()
            self._draw_scale_bar()

    # ---- compute lifetimes + paint ----------------------------------------
    def compute_and_paint(self):
        if self.data.real_cal is None:
            self._log("Paint skipped — run Apply first.")
            return
        self._log(f"Paint ({self.mode})…")
        if self.mode == "cursor":
            self._compute_cursor()
        else:
            self._compute_gmm()
        self._log("Paint complete.")

    def _lifetimes(self, g, s):
        freq = self.data.work_frequency
        tp, tm = phasor_to_apparent_lifetime(g, s, freq)
        tn = phasor_to_normal_lifetime(g, s, freq)
        return float(tp), float(tm), float(tn)

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
                tp, tm, tn = self._lifetimes(cg, cs)
            else:
                cg = cs = tp = tm = tn = float("nan")
            self.cluster_stats.append(dict(idx=k + 1, color=c["color"], label=c["label"],
                                           tp=tp, tm=tm, tn=tn, g=cg, s=cs, n=n,
                                           area=100.0 * n / total_valid))
        self._paint(masks, colors); self._fill_table()

    def _compute_gmm(self):
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
            tp, tm, tn = self._lifetimes(cg, cs)
            n = int(masks[k].sum())
            self.cluster_stats.append(dict(idx=k + 1, color=colors[k], label=categorical_name(k),
                                           tp=tp, tm=tm, tn=tn, g=cg, s=cs, n=n,
                                           area=100.0 * n / total_valid))
        self._paint(masks, colors); self._fill_table()

    def _phasor_valid_mask(self):
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
        intensity = self._segmentation_intensity()
        overlay = pseudo_color(*[masks[k] for k in range(len(masks))],
                               intensity=intensity, colors=np.array(colors))
        self.last_overlay = np.clip(np.asarray(overlay), 0, 1)
        self.chk_overlay.setChecked(True); self.refresh_image()

    def refresh_image(self):
        if self.chk_overlay.isChecked() and self.last_overlay is not None:
            self.image.show_overlay(self.last_overlay, title=f"Segmentation ({self.mode})")
        else:
            self._show_base_image()

    def _show_base_image(self):
        """Render the view selected above the image (masked like the phasor plot)."""
        choice = (
            self.cb_image_view.currentText()
            if hasattr(self, "cb_image_view")
            else IMAGE_VIEW_ITEMS[0]
        )
        if choice == IMAGE_VIEW_ITEMS[0]:
            disp = self._photon_image_filtered()
            if disp is not None:
                log_scale = getattr(self, "chk_log_display", None) and self.chk_log_display.isChecked()
                auto_c = not getattr(self, "chk_auto_contrast", None) or self.chk_auto_contrast.isChecked()
                title = "Photons (masked)"
                self.image.show_intensity(
                    disp, log_scale=log_scale, auto_contrast=auto_c, title=title)
                self._draw_scale_bar()
            return
        tau_sources = {
            IMAGE_VIEW_ITEMS[1]: (self.data.tau_phi, "τφ phase (ns)"),
            IMAGE_VIEW_ITEMS[2]: (self.data.tau_mod, "τmod (ns)"),
            IMAGE_VIEW_ITEMS[3]: (self.data.tau_normal, "τ normal (ns)"),
            IMAGE_VIEW_ITEMS[4]: (self.data.tau_search_phi, "τ search phase (ns)"),
            IMAGE_VIEW_ITEMS[5]: (self.data.tau_search_mod, "τ search mod (ns)"),
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
        finite = np.asarray(arr)[np.isfinite(arr)]
        if finite.size == 0:
            return None, None
        return float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))

    # ---- results table -----------------------------------------------------
    def _fill_table(self):
        if getattr(self, "_label_signal_connected", False):
            self.table.itemChanged.disconnect(self._label_edited)
            self._label_signal_connected = False
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
            self.table.setItem(r, 5, self._ro(f"{st['tp']:.3f}"))
            self.table.setItem(r, 6, self._ro(f"{st['tm']:.3f}"))
            self.table.setItem(r, 7, self._ro(f"{st['tn']:.3f}"))
            self.table.setItem(r, 8, self._ro(str(st["n"])))
            self.table.setItem(r, 9, self._ro(f"{st['area']:.2f}"))
        self.table.itemChanged.connect(self._label_edited)
        self._label_signal_connected = True

    def _label_edited(self, item):
        if item.column() == 2:
            r = item.row()
            if 0 <= r < len(self.cluster_stats):
                self.cluster_stats[r]["label"] = item.text()
                if self.mode == "cursor" and r < len(self.phasor.cursors):
                    self.phasor.cursors[r]["label"] = item.text()
                    self._refresh_active_cursor_combo(select_idx=r)

    @staticmethod
    def _ro(text):
        it = QtWidgets.QTableWidgetItem(text); it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable); return it

    @staticmethod
    def _editable_group(text):
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable)
        it.setToolTip("Group name for this file (e.g. Tumor, Control)")
        return it

    @staticmethod
    def _rgb_hex(c):
        return "FF" + "".join(f"{int(round(255 * max(0.0, min(1.0, x)))):02X}" for x in c[:3])

    # ---- export ------------------------------------------------------------
    def export_all(self):
        if self.data.signal_full is None:
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

