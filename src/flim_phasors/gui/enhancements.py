"""Optional features mixin for MainWindow (menus, session, lazy load, etc.).

Groups all of MainWindow's secondary UI concerns — the menu bar, keyboard
shortcuts, drag-and-drop file loading, recent-files tracking, calibration
save/load, session bundle/JSON save/load, batch export, Phasor Lab theme
switching, cursor undo/save/load, and table export — into
:class:`EnhancementsMixin` so ``main_window.py`` can stay focused on core
widget layout and processing logic. ``MainWindow`` mixes this class in and
calls ``_init_enhancements()`` once ``_build_ui`` has run.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from flim_phasors import __version__
from flim_phasors.calibration_io import load_calibration, save_calibration
from flim_phasors.canvas.ref_preview import RefPreviewCanvas
from flim_phasors.constants import FLIM_FILE_FILTER
from flim_phasors.cursors_io import load_cursors, save_cursors
from flim_phasors.io import is_supported_flim_path
from flim_phasors.memory_est import format_memory_line
from flim_phasors.session_bundle_io import (
    BUNDLE_EXTENSION,
    apply_session_bundle_to_window,
    is_session_bundle,
    load_session_bundle,
    save_session_bundle,
)
from flim_phasors.session_io import (
    apply_calibration_from_session,
    load_session_json,
    missing_paths_message,
    register_sample_from_session_row,
    restore_cursors_to_phasor,
)
from flim_phasors.gui.theme import (
    DEFAULT_THEME,
    PRIMARY_BUTTON_ATTRS,
    THEME_MENU_LABELS,
    THEME_PHASOR_LAB,
    THEME_PHASOR_LAB_LIGHT,
    is_dark_theme,
    log_style_for,
    normalize_theme_id,
    stylesheet_for,
    toolbar_colors_for,
    toolbar_style_for,
)

def _signed_or_zero(value: int) -> str:
    """Format a slider value as "0" at zero, or with an explicit +/- sign otherwise.
    Used for the Brightness/Contrast readout labels so the neutral position
    reads as a plain "0" instead of the "+0" that ``f"{0:+d}"`` would give.
    """
    return "0" if value == 0 else f"{value:+d}"

class EnhancementsMixin:
    """Mixin supplying MainWindow's "optional" UI features.
    Bundles everything that is layered on top of the core processing UI built by
    ``MainWindow._build_ui``: extra widgets (calibration I/O buttons, reference
    preview, metadata label, display toggles, cursor undo/save/load buttons, table
    export buttons), the menu bar, keyboard shortcuts, drag-and-drop of FLIM files,
    theme handling, session bundle/JSON load-save, recent-files menus, and cursor
    undo history. ``MainWindow`` must call ``_init_enhancements()`` once, after
    ``_build_ui`` has created the widgets this mixin extends (``proc_grid``,
    ``cursor_box``, ``gb_act``, etc.), otherwise attribute lookups here will fail.
    """

    def _init_enhancements(self):
        """Initialize all mixin-provided UI and behavior on the main window.
        Must be called by ``MainWindow.__init__`` after ``_build_ui`` so that the
        widgets extended here already exist. Initializes the calibration
        staleness hash and the cursor undo stack, restores the persisted theme
        id/dark-mode flag from ``QSettings``, then in order: extends the
        processing/cursor panels with extra widgets (``_extend_ui``), builds the
        menu bar (``_build_menus``), binds keyboard shortcuts
        (``_setup_shortcuts``), enables drag-and-drop of FLIM files
        (``_setup_drag_drop``), tags primary action buttons for accent styling
        (``_tag_primary_buttons``), and finally applies the restored theme
        (``_apply_ui_theme``) so the stylesheet and widget palettes match on
        first paint.
        """
        self._cal_settings_hash = ""
        self._cursor_undo_stack: list[list] = []
        self._ui_theme = self._load_ui_theme_setting()
        self._dark_theme = is_dark_theme(self._ui_theme)
        self._extend_ui()
        self._build_menus()
        self._setup_shortcuts()
        self._setup_drag_drop()
        self._tag_primary_buttons()
        self._apply_ui_theme(self._ui_theme)

    def _extend_ui(self):
        """Add calibration I/O, reference preview, metadata, and export widgets.
        Called once from ``_init_enhancements`` to graft additional rows onto
        widgets created by ``MainWindow._build_ui``. Adds, in the processing
        grid (``self.proc_grid``): a Save/Load calibration button row wired to
        ``save_calibration_file``/``load_calibration_file``; the reference
        phasor preview canvas (``self.ref_preview``, a
        :class:`RefPreviewCanvas`); a metadata label (``self.lbl_metadata``)
        that later shows memory/pixel-size/version info; a pixel-size spin box
        (``self.sp_pixel_um``, wired to ``_on_pixel_size_changed`` so edits
        redraw immediately) used for the image scale bar, alongside a "Show
        scale bar" checkbox (``self.chk_show_scale_bar``, default on) that
        lets the bar be hidden even when a pixel size is known; a scale-bar
        row with a length spin box (``self.sp_scalebar_um``, default 10 µm)
        and a corner combo box (``self.cb_scalebar_loc``: bottom/top ×
        left/right), both wired to ``refresh_image``; and a display-options
        row with "Log photons" and "Auto contrast" checkboxes
        (``self.chk_log_display``, ``self.chk_auto_contrast``) both wired to
        ``refresh_image``; and Brightness/Contrast sliders (``self.sl_brightness``,
        ``self.sl_contrast``, range -100..100, centered on 0 = no change) that
        adjust the intensity display window on top of whatever Auto
        contrast/Log photons compute, plus a Reset button that zeros both.
        Also appends an Undo/Save cursors/Load cursors button row to
        ``self.cursor_box`` and an Export table CSV/Copy table button row to
        ``self.gb_act``. Creates new instance attributes on ``self`` for every
        widget it adds; does not remove or reflow any pre-existing widgets.
        """
        _small = "font-size: 10px;"

        row_cal_io = QtWidgets.QHBoxLayout()
        btn_save_cal = QtWidgets.QPushButton("Save cal…")
        btn_save_cal.clicked.connect(self.save_calibration_file)
        btn_load_cal = QtWidgets.QPushButton("Load cal…")
        btn_load_cal.clicked.connect(self.load_calibration_file)
        row_cal_io.addWidget(btn_save_cal)
        row_cal_io.addWidget(btn_load_cal)
        self.proc_grid.addLayout(row_cal_io, 11, 0, 1, 4)

        self.ref_preview = RefPreviewCanvas(self)
        self.ref_preview.setMinimumHeight(120)
        self.ref_preview.setMaximumHeight(180)
        # Only useful when editing manual g/s — keep hidden for file-based cal.
        self.ref_preview.setVisible(False)
        self.proc_grid.addWidget(self.ref_preview, 12, 0, 1, 4)

        self.lbl_metadata = QtWidgets.QLabel("")
        self.lbl_metadata.setStyleSheet(f"color: gray; {_small}")
        self.lbl_metadata.setWordWrap(True)
        self.proc_grid.addWidget(self.lbl_metadata, 13, 0, 1, 4)

        self.sp_pixel_um = QtWidgets.QDoubleSpinBox()
        self.sp_pixel_um.setRange(0, 100)
        self.sp_pixel_um.setDecimals(3)
        self.sp_pixel_um.setSuffix(" µm/px")
        self.sp_pixel_um.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.sp_pixel_um.setMinimumWidth(88)
        self.sp_pixel_um.setToolTip(
            "Pixel size in µm — auto-filled from file metadata when available, "
            "or enter manually (0 = off). Used for the image scale bar.")
        self.sp_pixel_um.valueChanged.connect(self._on_pixel_size_changed)
        self.chk_show_scale_bar = QtWidgets.QCheckBox("Show scale bar")
        self.chk_show_scale_bar.setChecked(True)
        self.chk_show_scale_bar.setToolTip(
            "Draw a scale bar on the image when pixel size is known "
            "(length/corner set below).")
        self.chk_show_scale_bar.stateChanged.connect(self.refresh_image)
        row_px = QtWidgets.QHBoxLayout()
        row_px.addWidget(QtWidgets.QLabel("Pixel"))
        row_px.addWidget(self.sp_pixel_um)
        row_px.addWidget(self.chk_show_scale_bar)
        self.proc_grid.addLayout(row_px, 14, 0, 1, 4)

        self.sp_scalebar_um = QtWidgets.QDoubleSpinBox()
        self.sp_scalebar_um.setRange(0.1, 1000)
        self.sp_scalebar_um.setDecimals(1)
        self.sp_scalebar_um.setValue(10.0)
        self.sp_scalebar_um.setSuffix(" µm")
        self.sp_scalebar_um.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.sp_scalebar_um.setMinimumWidth(70)
        self.sp_scalebar_um.setToolTip("Scale bar length.")
        self.sp_scalebar_um.valueChanged.connect(self.refresh_image)
        self.cb_scalebar_loc = QtWidgets.QComboBox()
        self.cb_scalebar_loc.addItems(
            ["Bottom left", "Bottom right", "Top left", "Top right"])
        self.cb_scalebar_loc.setToolTip("Scale bar corner.")
        self.cb_scalebar_loc.currentIndexChanged.connect(self.refresh_image)
        row_scalebar = QtWidgets.QHBoxLayout()
        row_scalebar.addWidget(QtWidgets.QLabel("Scale bar"))
        row_scalebar.addWidget(self.sp_scalebar_um)
        row_scalebar.addWidget(self.cb_scalebar_loc)
        self.proc_grid.addLayout(row_scalebar, 15, 0, 1, 4)

        self.chk_log_display = QtWidgets.QCheckBox("Log photons")
        self.chk_log_display.stateChanged.connect(self.refresh_image)
        self.chk_auto_contrast = QtWidgets.QCheckBox("Auto contrast")
        self.chk_auto_contrast.setChecked(True)
        self.chk_auto_contrast.stateChanged.connect(self.refresh_image)
        row_disp = QtWidgets.QHBoxLayout()
        row_disp.addWidget(self.chk_log_display)
        row_disp.addWidget(self.chk_auto_contrast)
        self.proc_grid.addLayout(row_disp, 16, 0, 1, 4)

        _bc_label_width = 62
        lbl_bright = QtWidgets.QLabel("Brightness")
        lbl_bright.setFixedWidth(_bc_label_width)
        self.sl_brightness = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sl_brightness.setRange(-100, 100)
        self.sl_brightness.setValue(0)
        self.sl_brightness.setToolTip("Shift the display window brighter/darker.")
        self.lbl_brightness_val = QtWidgets.QLabel("0")
        self.lbl_brightness_val.setFixedWidth(28)
        self.lbl_brightness_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sl_brightness.valueChanged.connect(self._on_brightness_changed)
        row_bright = QtWidgets.QHBoxLayout()
        row_bright.addWidget(lbl_bright)
        row_bright.addWidget(self.sl_brightness)
        row_bright.addWidget(self.lbl_brightness_val)
        self.proc_grid.addLayout(row_bright, 17, 0, 1, 4)

        lbl_contrast = QtWidgets.QLabel("Contrast")
        lbl_contrast.setFixedWidth(_bc_label_width)
        self.sl_contrast = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sl_contrast.setRange(-100, 100)
        self.sl_contrast.setValue(0)
        self.sl_contrast.setToolTip("Narrow/widen the display window (more/less contrast).")
        self.lbl_contrast_val = QtWidgets.QLabel("0")
        self.lbl_contrast_val.setFixedWidth(28)
        self.lbl_contrast_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sl_contrast.valueChanged.connect(self._on_contrast_changed)
        row_contrast = QtWidgets.QHBoxLayout()
        row_contrast.addWidget(lbl_contrast)
        row_contrast.addWidget(self.sl_contrast)
        row_contrast.addWidget(self.lbl_contrast_val)
        self.proc_grid.addLayout(row_contrast, 18, 0, 1, 4)

        btn_reset_bc = QtWidgets.QPushButton("Reset brightness/contrast")
        btn_reset_bc.setToolTip("Reset brightness and contrast to 0.")
        btn_reset_bc.clicked.connect(self._reset_brightness_contrast)
        row_reset = QtWidgets.QHBoxLayout()
        row_reset.addStretch(1)
        row_reset.addWidget(btn_reset_bc)
        self.proc_grid.addLayout(row_reset, 19, 0, 1, 4)

        row_cur_io = QtWidgets.QHBoxLayout()
        btn_undo = QtWidgets.QPushButton("Undo")
        btn_undo.setFixedWidth(48)
        btn_undo.clicked.connect(self.undo_cursor)
        btn_save_cur = QtWidgets.QPushButton("Save cursors…")
        btn_save_cur.clicked.connect(self.save_cursors_file)
        btn_load_cur = QtWidgets.QPushButton("Load cursors…")
        btn_load_cur.clicked.connect(self.load_cursors_file)
        row_cur_io.addWidget(btn_undo)
        row_cur_io.addWidget(btn_save_cur)
        row_cur_io.addWidget(btn_load_cur)
        self.cursor_box.layout().addLayout(row_cur_io, 4, 0, 1, 3)

        btn_exp_table = QtWidgets.QPushButton("Export table CSV…")
        btn_exp_table.clicked.connect(self.export_table_csv)
        btn_copy = QtWidgets.QPushButton("Copy table")
        btn_copy.clicked.connect(self.copy_table_to_clipboard)
        row_exp = QtWidgets.QHBoxLayout()
        row_exp.addWidget(btn_exp_table)
        row_exp.addWidget(btn_copy)
        self.gb_act.layout().addLayout(row_exp)
    def _on_pixel_size_changed(self, _value: float):
        """Redraw the image and metadata line after the Pixel spin box changes.
        Connected to ``sp_pixel_um.valueChanged``. The spin box has no other
        listener, so without this the scale bar and the "pixel X µm
        (manual)" metadata line would only catch up the next time something
        else happened to trigger a repaint (e.g. Apply) rather than
        immediately as the user types.
        Args:
            _value: New spin box value (unused; both refreshed methods read
                the widget directly).
        """
        self.refresh_image()
        if hasattr(self, "_update_metadata_panel"):
            self._update_metadata_panel()
    def _on_brightness_changed(self, value: int):
        """Update the brightness readout label and repaint the image.
        Connected to ``sl_brightness.valueChanged``. The slider's raw -100..100
        integer is what :meth:`~flim_phasors.gui.main_window.MainWindow._show_base_image`
        reads (normalized to -1.0..1.0) when building the intensity display
        window; this handler only updates the adjacent numeric label and
        triggers a repaint via ``refresh_image``.
        Args:
            value: New slider value.
        """
        self.lbl_brightness_val.setText(_signed_or_zero(value))
        self.refresh_image()
    def _on_contrast_changed(self, value: int):
        """Update the contrast readout label and repaint the image.
        Connected to ``sl_contrast.valueChanged``. See
        :meth:`_on_brightness_changed` for how the slider value is consumed.
        Args:
            value: New slider value.
        """
        self.lbl_contrast_val.setText(_signed_or_zero(value))
        self.refresh_image()
    def _reset_brightness_contrast(self):
        """Zero both brightness and contrast sliders, bound to the Reset button.
        Setting ``sl_brightness``/``sl_contrast`` back to 0 fires their
        ``valueChanged`` signals, which update the readout labels and repaint
        the image via :meth:`_on_brightness_changed`/:meth:`_on_contrast_changed`,
        so no separate repaint call is needed here.
        """
        self.sl_brightness.setValue(0)
        self.sl_contrast.setValue(0)

    def _build_menus(self):
        """Build the File, View, and Help menus on the window's menu bar.
        Called once from ``_init_enhancements``. Populates **File** with sample
        and reference file pickers, calibration, the "Recent samples"/"Recent
        references" submenus (stored as ``self._recent_samples_menu`` and
        ``self._recent_refs_menu`` and filled by ``_refresh_recent_menus``),
        session open/save, calibration save/load, batch export, export-all, and
        quit — each bound to its handler method and, where noted, a keyboard
        shortcut. Populates **View → Theme** with a checkable, mutually
        exclusive action per available theme (grouped via
        ``self._theme_group``, tracked in ``self._theme_actions``); triggering
        an action calls ``_apply_ui_theme`` with that theme id, and the action
        matching the currently restored theme is pre-checked. Populates **Help**
        with the shortcuts reference and About dialogs. Finishes by calling
        ``_refresh_recent_menus`` to populate the recent-file submenus from
        persisted settings.
        """
        mb = self.menuBar()
        file_m = mb.addMenu("&File")
        file_m.addAction("Sample…", self.choose_sample, "Ctrl+O")
        file_m.addAction("Reference…", self.choose_ref, "Ctrl+R")
        file_m.addAction("Calibrate", self.calibrate_reference, "F6")
        file_m.addSeparator()
        self._recent_samples_menu = file_m.addMenu("Recent samples")
        self._recent_refs_menu = file_m.addMenu("Recent references")
        file_m.addSeparator()
        file_m.addAction("Open session…", self.open_session, "Ctrl+Shift+O")
        file_m.addAction("Save session…", self.save_session, "Ctrl+Shift+S")
        file_m.addAction("Save calibration…", self.save_calibration_file, "Ctrl+Shift+B")
        file_m.addAction("Load calibration…", self.load_calibration_file, "Ctrl+Shift+K")
        file_m.addSeparator()
        file_m.addAction("Batch export folder…", self.batch_export_folder)
        file_m.addAction("Export all…", self.export_all, "Ctrl+E")
        file_m.addSeparator()
        file_m.addAction("E&xit", self.close, "Ctrl+Q")

        view_m = mb.addMenu("&View")
        theme_m = view_m.addMenu("Theme")
        self._theme_group = QtGui.QActionGroup(self)
        self._theme_actions = {}
        for theme_id in (THEME_PHASOR_LAB, THEME_PHASOR_LAB_LIGHT):
            act = theme_m.addAction(THEME_MENU_LABELS[theme_id])
            act.setCheckable(True)
            self._theme_group.addAction(act)
            self._theme_actions[theme_id] = act
            act.triggered.connect(
                # Default arg captures theme_id — bare `theme_id` in the loop would bind late.
                lambda _checked=False, tid=theme_id: self._apply_ui_theme(tid))
        checked = self._theme_actions.get(self._ui_theme)
        if checked is not None:
            checked.setChecked(True)

        help_m = mb.addMenu("&Help")
        help_m.addAction("Keyboard shortcuts…", self.show_shortcuts)
        help_m.addAction("About…", self.show_about)

        self._refresh_recent_menus()

    def _setup_shortcuts(self):
        """Register application-wide keyboard shortcuts for common actions.
        Called once from ``_init_enhancements``. Creates one ``QShortcut`` per
        entry in a local table mapping key sequences to handlers, covering
        processing (F5 Apply, F6 Calibrate, F7 Paint), cursor/segmentation
        editing (Ctrl+Shift+N add, Ctrl+Z undo, Ctrl+Shift+X clear, Delete/
        Backspace remove, Ctrl+G fit GMM, Ctrl+M toggle Cursors/GMM mode),
        right-panel tab navigation (Ctrl+1/2/3), and cursor/calibration file
        I/O (Ctrl+Shift+U/Y/B/K). Each ``QShortcut`` is parented to ``self`` so
        it fires regardless of which child widget has focus, and is torn down
        automatically when the window is destroyed. See ``show_shortcuts`` for
        the user-facing reference list kept in sync with this table.
        """
        bind = QtGui.QShortcut
        seq = QtGui.QKeySequence
        shortcuts = (
            (seq("F5"), lambda: self.apply_processing()),
            (seq("F6"), self.calibrate_reference),
            (seq("F7"), self.compute_and_paint),
            (seq("Ctrl+Shift+N"), self.add_cursor),
            (seq("Ctrl+Z"), self.undo_cursor),
            (seq("Ctrl+Shift+X"), self.clear_cursors),
            (seq(Qt.Key.Key_Delete), self.remove_cursor),
            (seq(Qt.Key.Key_Backspace), self.remove_cursor),
            (seq("Ctrl+G"), self.fit_gmm),
            (seq("Ctrl+M"), self._shortcut_toggle_segmentation_mode),
            (seq("Ctrl+1"), lambda: self._shortcut_goto_tab("setup")),
            (seq("Ctrl+2"), lambda: self._shortcut_goto_tab("compare")),
            (seq("Ctrl+3"), lambda: self._shortcut_goto_tab("analyze")),
            (seq("Ctrl+Shift+U"), self.save_cursors_file),
            (seq("Ctrl+Shift+Y"), self.load_cursors_file),
            (seq("Ctrl+Shift+B"), self.save_calibration_file),
            (seq("Ctrl+Shift+K"), self.load_calibration_file),
        )
        for key, handler in shortcuts:
            bind(key, self, handler)

    def _shortcut_goto_tab(self, which: str):
        """Switch the right-hand panel tabs in response to a Ctrl+1/2/3 shortcut.
        Looks up the target tab index from ``_tab_setup_idx``,
        ``_tab_compare_idx``, or ``_tab_analyze_idx`` attributes on ``self``
        (falling back to 0/1/2 respectively if MainWindow has not set them),
        then calls ``self.panel_tabs.setCurrentIndex`` to switch the visible
        tab. Silently does nothing if ``self`` has no ``panel_tabs`` attribute
        yet (e.g. shortcut fired before ``_build_ui`` finished) or if ``which``
        is not a recognized key, in which case it defaults to index 0.
        Args:
            which: One of ``"setup"``, ``"compare"``, or ``"analyze"``,
                identifying which right-panel tab to activate.
        """
        if not hasattr(self, "panel_tabs"):
            return
        idx = {
            "setup": getattr(self, "_tab_setup_idx", 0),
            "compare": getattr(self, "_tab_compare_idx", 1),
            "analyze": getattr(self, "_tab_analyze_idx", 2),
        }.get(which, 0)
        self.panel_tabs.setCurrentIndex(idx)

    def _shortcut_toggle_segmentation_mode(self):
        """Flip the segmentation-mode radio buttons on the Ctrl+M shortcut.
        Triggered by the Ctrl+M keyboard shortcut. If the GMM radio button
        (``self.rb_gmm``) is currently checked, checks the cursor ROI radio
        button (``self.rb_cursor``) instead, and vice versa; checking a
        ``QRadioButton`` in a mutually exclusive group automatically unchecks
        its sibling and fires that button's own ``toggled``/state-change
        handlers, which drive the rest of the segmentation-mode UI update. Does
        nothing if ``self`` has no ``rb_cursor`` attribute yet.
        """
        if not hasattr(self, "rb_cursor"):
            return
        if self.rb_gmm.isChecked():
            self.rb_cursor.setChecked(True)
        else:
            self.rb_gmm.setChecked(True)

    def show_shortcuts(self):
        """Display a modal reference list of all keyboard shortcuts.
        Bound to Help → "Keyboard shortcuts…". Builds a static HTML string
        grouping shortcuts by category (files & session, processing,
        segmentation, navigation) matching the bindings registered in
        ``_setup_shortcuts``, and shows it in a blocking
        ``QMessageBox.information`` dialog. Purely informational; does not
        change any application state.
        """
        text = (
            "<b>Files & session</b><br>"
            "Ctrl+O — Sample…<br>"
            "Ctrl+R — Reference…<br>"
            "F6 — Calibrate<br>"
            "Ctrl+Shift+B — Save calibration…<br>"
            "Ctrl+Shift+K — Load calibration…<br>"
            "Ctrl+Shift+O — Open session…<br>"
            "Ctrl+Shift+S — Save session…<br>"
            "Ctrl+E — Export all…<br>"
            "Ctrl+Q — Quit<br><br>"
            "<b>Processing</b><br>"
            "F5 — Apply<br>"
            "F7 — Paint<br><br>"
            "<b>Segmentation</b><br>"
            "Ctrl+Shift+N — Add cursor<br>"
            "Delete / Backspace — Remove cursor<br>"
            "Ctrl+Z — Undo cursor<br>"
            "Ctrl+Shift+X — Clear all cursors<br>"
            "Ctrl+G — Fit GMM<br>"
            "Ctrl+M — Toggle Cursors / GMM<br>"
            "Ctrl+Shift+U — Save cursors…<br>"
            "Ctrl+Shift+Y — Load cursors…<br><br>"
            "<b>Navigation</b><br>"
            "Ctrl+1 — Setup tab<br>"
            "Ctrl+2 — Multi-phasor tab<br>"
            "Ctrl+3 — Analyze tab"
        )
        QtWidgets.QMessageBox.information(self, "Keyboard shortcuts", text)

    def _setup_drag_drop(self):
        """Enable drag-and-drop of FLIM files onto the main window.
        Called once from ``_init_enhancements``. Simply calls
        ``self.setAcceptDrops(True)`` so Qt starts delivering
        ``dragEnterEvent``/``dropEvent`` to this window; the actual filtering
        and loading logic lives in those two event handlers below.
        """
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Qt event handler that decides whether a drag can be dropped here.
        Fired continuously by Qt while the user drags an item over the window
        (requires ``setAcceptDrops(True)``, set in ``_setup_drag_drop``).
        Inspects the drag's mime data for file URLs and, on the first URL whose
        local path ``is_supported_flim_path`` (a supported PTU/TIF/LIF file),
        calls ``event.acceptProposedAction()`` to show the "drop allowed"
        cursor and enable ``dropEvent`` to fire; otherwise leaves the event
        unaccepted so Qt shows the "drop rejected" cursor. Does not modify any
        window state — this only affects drag-cursor feedback.
        Args:
            event: The ``QDragEnterEvent`` describing the pending drag.
        """
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if is_supported_flim_path(url.toLocalFile()):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        """Qt event handler that loads FLIM files dropped onto the window.
        Fired by Qt when the user releases the mouse to complete a drop that
        was accepted by ``dragEnterEvent``. Filters the dropped URLs down to
        local paths accepted by ``is_supported_flim_path`` and, if any remain,
        routes them through the same job pipeline used by the Sample… file
        dialog: ``_expand_sample_load_jobs`` (which raises ``ValueError`` — shown
        via a warning dialog — if a dropped LIF file needs series selection and
        that selection is cancelled or invalid), then ``_prepare_sample_load``
        (which may prompt the user and can abort the load), and finally
        ``_load_sample_paths`` to actually read the files into new dataset
        entries and refresh the UI. Does nothing if the drop contains no
        supported files.
        Args:
            event: The ``QDropEvent`` carrying the dropped mime data (file
                URLs).
        """
        paths = [
            url.toLocalFile() for url in event.mimeData().urls()
            if is_supported_flim_path(url.toLocalFile())
        ]
        if not paths:
            return
        try:
            jobs = self._expand_sample_load_jobs(paths)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "LIF file", str(e))
            return
        if not jobs:
            return
        if not self._prepare_sample_load(jobs):
            return
        self._load_sample_paths(jobs)

    def _remember_recent(self, key: str, path: str, max_items: int = 8):
        """Record a file path in a persisted recent-files list and refresh the menu.
        Reads the existing list stored under ``key`` in ``self._settings``
        (normalizing a legacy single-string value into a one-item list),
        removes any existing entry equal to the absolute form of ``path``, then
        inserts the absolute path at the front (most-recent-first) and
        truncates to ``max_items`` before writing the list back to
        ``QSettings``. Finishes by calling ``_refresh_recent_menus`` so the
        File menu's "Recent samples"/"Recent references" submenus immediately
        reflect the change.
        Args:
            key: QSettings key for the list, e.g. ``"recent_samples"`` or
                ``"recent_refs"``.
            path: File path to move to the front of the recent list; stored in
                absolute form.
            max_items: Maximum number of entries to retain (oldest entries
                beyond this are dropped). Defaults to 8.
        """
        items = list(self._settings.value(key, []) or [])
        if isinstance(items, str):
            items = [items]
        path = os.path.abspath(path)
        items = [p for p in items if p != path]
        items.insert(0, path)
        self._settings.setValue(key, items[:max_items])
        self._refresh_recent_menus()

    def _refresh_recent_menus(self):
        """Rebuild the "Recent samples" and "Recent references" submenus.
        Called after any change to the recent-files lists (from
        ``_remember_recent``) or once at startup (from ``_build_menus``). For
        each of the two submenus, clears all existing actions, reads the
        corresponding list from ``self._settings`` (normalizing a legacy
        single-string value into a one-item list), and if the list is empty
        adds a single disabled "(empty)" placeholder action. Otherwise adds one
        action per path that still exists on disk (stale entries are silently
        skipped rather than removed from settings), labeled with the file's
        base name, wired via a closure that binds both the path and the
        open-handler by default argument so each action opens the correct file
        with the correct handler (sample vs reference) regardless of loop
        iteration order.
        """
        for menu, key, handler in (
            (self._recent_samples_menu, "recent_samples", self._open_recent_sample),
            (self._recent_refs_menu, "recent_refs", self._open_recent_ref),
        ):
            menu.clear()
            items = self._settings.value(key, []) or []
            if isinstance(items, str):
                items = [items]
            if not items:
                a = menu.addAction("(empty)")
                a.setEnabled(False)
                continue
            for p in items:
                if os.path.isfile(p):
                    # Bind both path and handler — a bare `handler` in the lambda
                    # would close over the for-loop variable and always call the
                    # last handler (_open_recent_ref), sending samples to Reference.
                    menu.addAction(
                        os.path.basename(p),
                        lambda checked=False, path=p, h=handler: h(path),
                    )

    def _open_recent_sample(self, path):
        """Load a sample chosen from the "Recent samples" menu.
        Triggered when the user clicks an entry in the recent-samples submenu.
        Mirrors ``dropEvent``'s pipeline for a single path: expands it into
        load jobs via ``_expand_sample_load_jobs`` (showing a warning dialog
        and aborting if the path is an invalid/ambiguous LIF file), then, if
        any jobs resulted, confirms the load via ``_prepare_sample_load`` and
        finally loads via ``_load_sample_paths``, which appends a new dataset
        and refreshes the sample list/image views.
        Args:
            path: Absolute path to the previously-opened sample file.
        """
        try:
            jobs = self._expand_sample_load_jobs([path])
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "LIF file", str(e))
            return
        if not jobs:
            return
        if not self._prepare_sample_load(jobs):
            return
        self._load_sample_paths(jobs)

    def _open_recent_ref(self, path):
        """Select a reference file chosen from the "Recent references" menu.
        Triggered when the user clicks an entry in the recent-references
        submenu. Simply delegates to ``choose_ref_with_path``, which validates
        the path and updates the active calibration reference without opening
        a file dialog.
        Args:
            path: Absolute path to the previously-used reference file.
        """
        self.choose_ref_with_path(path)

    def choose_ref_with_path(self, path: str):
        """Set the active calibration reference file without a file dialog.
        Used by ``_open_recent_ref`` (and available for programmatic/session
        restore use) as an alternative to the "Reference…" file-picker action.
        Validates ``path`` with ``is_supported_flim_path`` and silently does
        nothing if it is empty or unsupported; otherwise delegates to
        ``_set_reference_path``, which updates ``self.shared_ref_path``/the
        active dataset's reference and refreshes dependent UI (reference label,
        calibration display).
        Args:
            path: Candidate reference file path.
        """
        if not path or not is_supported_flim_path(path):
            return
        self._set_reference_path(path)

    def _calibration_settings_tuple(self):
        """Snapshot the calibration-relevant UI controls as a hashable tuple.
        Reads the current harmonic spin box, filter combo box text, reference
        channel combo box index, and the manual-calibration checkbox state. Used
        by ``_mark_calibration_current``/``_update_calibration_stale_style`` to
        detect, by string comparison against a previously stored snapshot,
        whether the user has changed calibration-affecting settings since the
        last time Calibrate was run — used purely for the "stale" warning
        styling on the calibration label, not for the actual processing math.
        Returns:
            Tuple of ``(harmonic, filter_text, ref_channel_index,
            manual_cal_checked)``.
        """
        return (
            int(self.sp_harm.value()),
            self.cb_filter.currentText(),
            int(self.cb_ref_channel.currentIndex()),
            bool(self.chk_manual_cal.isChecked()),
        )

    def _mark_calibration_current(self):
        """Snapshot calibration settings as the new "not stale" baseline.
        Called right after a successful Calibrate action (and after loading a
        calibration file) to record the current harmonic/filter/channel/manual
        settings, via ``_calibration_settings_tuple``, into
        ``self._cal_settings_hash``. Immediately calls
        ``_update_calibration_stale_style`` afterward so the calibration label
        clears any stale-orange styling right away, since by definition the
        settings now match the just-recorded snapshot.
        """
        self._cal_settings_hash = str(self._calibration_settings_tuple())
        self._update_calibration_stale_style()

    def _update_calibration_stale_style(self):
        """Recolor the calibration display label to warn about stale calibration.
        Computes whether the reference calibration is "stale": it is active
        (``self.ref_calibration.is_active``), manual calibration override is
        off, and the current ``_calibration_settings_tuple`` no longer matches
        the snapshot recorded by ``_mark_calibration_current``
        (``self._cal_settings_hash``) — meaning the user changed
        harmonic/filter/channel/manual-cal after the last Calibrate run without
        re-running it. Sets ``self.lbl_cal_display``'s stylesheet to an amber
        color (``#b45309``) when stale, otherwise gray. Purely cosmetic; does
        not alter calibration data or block processing.
        """
        stale = (
            self.ref_calibration.is_active
            and not self.chk_manual_cal.isChecked()
            and str(self._calibration_settings_tuple()) != self._cal_settings_hash
        )
        color = "#b45309" if stale else "gray"
        self.lbl_cal_display.setStyleSheet(f"color: {color}; font-size: 10px;")

    def _update_ref_preview(self):
        """Show/redraw the reference preview only in Manual ref phasor mode.
        File-based calibration already shows g/s in the status label; the mini
        plot is mainly useful when typing manual g/s so you can see the point
        vs the Ref τ target. Hidden otherwise to avoid clutter.
        """
        if not hasattr(self, "ref_preview"):
            return
        manual = bool(getattr(self, "chk_manual_cal", None) and self.chk_manual_cal.isChecked())
        self.ref_preview.setVisible(manual)
        if not manual:
            return
        self.ref_preview.show_calibration(
            self.ref_calibration,
            ref_lifetime_ns=self.sp_reflt.value(),
            frequency_mhz=self.sp_freq.value(),
            harmonic=int(self.sp_harm.value()),
        )

    def _update_metadata_panel(self):
        """Rebuild the small metadata line shown below the processing controls.
        Assembles a "·"-separated string from whichever of these apply: a
        memory-usage estimate for the active dataset (via
        ``format_memory_line``, only if a sample is loaded); the dataset's
        detected pixel size in µm if known, or otherwise the user-entered
        manual pixel size from ``self.sp_pixel_um`` if non-zero (labeled
        "(manual)"); and the installed ``phasorpy`` package version, if
        importable. Writes the resulting string (or an empty string if nothing
        applies) into ``self.lbl_metadata``. Called whenever a sample is
        loaded/changed or the manual pixel size is edited.
        """
        d = self.data
        parts = []
        if d.sample_path:
            parts.append(format_memory_line(d))
        if getattr(d, "pixel_size_um", 0) > 0:
            parts.append(f"pixel {d.pixel_size_um:.3f} µm")
        elif self.sp_pixel_um.value() > 0:
            parts.append(f"pixel {self.sp_pixel_um.value():.3f} µm (manual)")
        try:
            import phasorpy
            parts.append(f"phasorpy {getattr(phasorpy, '__version__', '?')}")
        except ImportError:
            pass
        self.lbl_metadata.setText(" · ".join(parts) if parts else "")

    def save_calibration_file(self):
        """Save the active reference calibration and related UI settings to JSON.
        Bound to the "Save cal…" button, File → "Save calibration…", and the
        Ctrl+Shift+B shortcut. If no calibration is active
        (``self.ref_calibration.is_active`` is falsy), shows an informational
        dialog and returns without prompting for a file. Otherwise prompts for
        a save path via a file dialog (defaulting to the remembered
        ``"cal_dir"``), appends a ``.json`` extension if missing, and calls
        ``save_calibration`` with the reference calibration object plus a
        dict of UI extras (reference lifetime, frequency, harmonic, filter
        mode) so the file can later restore both the numeric g/s values and the
        matching processing settings. Remembers the chosen directory in
        ``self._settings`` and appends a confirmation line to the activity log.
        Does not save the reference image data itself — only derived g/s
        values and settings.
        """
        if not self.ref_calibration.is_active:
            QtWidgets.QMessageBox.information(self, "Calibration", "No calibration to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration", self._dialog_dir("cal_dir"), "JSON (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        ui = {
            "reference_lifetime_ns": self.sp_reflt.value(),
            "frequency_MHz": self.sp_freq.value(),
            "harmonic": self.sp_harm.value(),
            "filter": self.cb_filter.currentText(),
        }
        save_calibration(path, self.ref_calibration, ui_extra=ui)
        self._settings.setValue("cal_dir", os.path.dirname(path))
        self._log(f"Calibration saved → {path}")

    def load_calibration_file(self):
        """Load a previously saved calibration JSON and apply it to the UI.
        Bound to the "Load cal…" button, File → "Load calibration…", and the
        Ctrl+Shift+K shortcut. Prompts for a file via a file dialog (defaulting
        to the remembered ``"cal_dir"``); returns immediately if cancelled.
        Parses the file with ``load_calibration`` into a calibration object and
        a dict of UI extras, then replaces ``self.ref_calibration``. If the
        file recorded the original reference file's path, updates
        ``self.shared_ref_path``, the reference label, the active dataset's
        ``ref_path``, and propagates the shared reference to other datasets via
        ``_propagate_shared_reference`` — note the reference *image* itself is
        not stored in the JSON, only its path, so Apply will still need that
        file to exist on disk. Restores reference lifetime, frequency,
        harmonic, and filter-mode spin/combo boxes from the UI extras when
        present, sets the manual-calibration checkbox to match the loaded
        calibration, and syncs manual g/s fields
        (``_sync_manual_fields_from_calibration``). Refreshes the calibration
        display label, the reference preview canvas, and marks the calibration
        as current (clearing stale styling). Turns off any active compare
        overlay if supported, and logs a summary line including the loaded g/s
        values.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration", self._dialog_dir("cal_dir"), "JSON (*.json)")
        if not path:
            return
        cal, ui = load_calibration(path)
        self.ref_calibration = cal
        if cal.source_path:
            self.shared_ref_path = cal.source_path
            self.lbl_ref.setText(os.path.basename(cal.source_path))
            self.data.ref_path = cal.source_path
            self._propagate_shared_reference()
        if ui.get("reference_lifetime_ns") is not None:
            self.sp_reflt.setValue(float(ui["reference_lifetime_ns"]))
        if ui.get("frequency_MHz") is not None:
            self.sp_freq.setValue(float(ui["frequency_MHz"]))
        if ui.get("harmonic") is not None:
            self.sp_harm.setValue(int(ui["harmonic"]))
        if ui.get("filter"):
            self.cb_filter.setCurrentText(str(ui["filter"]))
        self.chk_manual_cal.setChecked(cal.use_manual)
        self._sync_manual_fields_from_calibration()
        self._update_calibration_display()
        self._update_ref_preview()
        self._mark_calibration_current()
        if hasattr(self, "_ensure_compare_overlay_off"):
            self._ensure_compare_overlay_off()
        self._log(
            f"Calibration loaded from {os.path.basename(path)} "
            f"(g={cal.mean_g:.4f}, s={cal.mean_s:.4f}; reference file not decoded).")
        # JSON stores g/s only — user must still have the reference PTU/TIF on disk for Apply.

    def open_session(self):
        """Prompt for and open a session file, dispatching by session type.
        Bound to File → "Open session…" and the Ctrl+Shift+O shortcut. Shows a
        file dialog (defaulting to the remembered ``"session_dir"``) accepting
        session bundle files, plain JSON, or any file. Returns immediately if
        the user cancels. Uses ``is_session_bundle`` to detect whether the
        chosen file is a self-contained bundle (processed maps embedded) versus
        a lightweight path-only JSON session, and delegates to
        ``_open_session_bundle`` or ``_open_session_json`` respectively — the
        two have different restore semantics and error handling, documented on
        each.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open session",
            self._dialog_dir("session_dir"),
            f"Session bundle (*{BUNDLE_EXTENSION});;JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        if is_session_bundle(path):
            self._open_session_bundle(path)
            return
        self._open_session_json(path)

    def _open_session_bundle(self, path: str):
        """Restore full application state from a self-contained session bundle.
        Called by ``open_session`` when the chosen file is detected as a
        bundle (embeds processed phasor maps rather than just paths, so no
        original PTU/TIF files are required to view results). Loads the bundle
        via ``load_session_bundle`` and applies it to the window via
        ``apply_session_bundle_to_window``, which is responsible for
        repopulating ``self.datasets``, calibration state, cursors, and other
        UI state; any exception during either step is shown in a critical
        dialog and aborts without further changes. On success, logs a summary
        line with the number of restored samples and the bundle file's size on
        disk, and remembers the containing directory as ``"session_dir"`` in
        ``self._settings``.
        Args:
            path: Path to the session bundle file to load.
        """
        try:
            loaded = load_session_bundle(path)
            apply_session_bundle_to_window(self, loaded)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Session", str(e))
            return
        n = len(loaded["datasets"])
        size_mb = os.path.getsize(path) / (1024 * 1024)
        self._settings.setValue("session_dir", os.path.dirname(path))
        self._mark_clean()
        self._log(
            f"Session bundle loaded ({n} sample{'s' if n != 1 else ''}, "
            f"{size_mb:.2f} MB) — no PTU/TIF required."
        )

    def _open_session_json(self, path: str):
        """Restore the sample list and settings from a lightweight, path-based session.
        Called by ``open_session`` when the chosen file is a plain JSON session
        (stores file paths and settings only, not processed image data — unlike
        a bundle). Parses the file via ``load_session_json``, showing a
        critical dialog and aborting on failure. If any referenced sample/
        reference files are missing on disk, warns the user via
        ``missing_paths_message`` and a dialog listing up to 8 missing paths
        (truncated with "…" if more), reminding them that a session bundle
        would have avoided this dependency. Clears ``self.datasets`` and
        rebuilds it by calling ``register_sample_from_session_row`` for each
        saved sample row; if any datasets resulted, enables multi-image mode
        (``self.chk_multi``) and restores the previously active sample index
        (clamped to the valid range), making it ``self.data``. Applies saved
        calibration via ``apply_calibration_from_session`` and, if present,
        restores phasor cursors via ``restore_cursors_to_phasor``. Finishes by
        calling ``_restore_ui_for_active`` and ``_refresh_image_combo`` to sync
        the UI to the newly active dataset, remembers the directory as
        ``"session_dir"``, and logs a reminder that samples still need to be
        (re)loaded/decoded and Applied since raw data was not embedded.
        Args:
            path: Path to the JSON session file to load.
        """
        try:
            session = load_session_json(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Session", str(e))
            return
        missing = missing_paths_message(session)
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing files",
                "This JSON only stores paths — original files are still required:\n"
                + "\n".join(missing[:8])
                + ("\n…" if len(missing) > 8 else "")
                + f"\n\nUse {BUNDLE_EXTENSION} (File → Save session…) to archive processed maps.",
            )
        self.datasets.clear()
        self.active_idx = -1
        for row in session.get("samples", []):
            d = register_sample_from_session_row(row)
            self.datasets.append(d)
        if self.datasets:
            self.chk_multi.setChecked(True)
            self.active_idx = int(session.get("active_sample_index", 0))
            self.active_idx = max(0, min(self.active_idx, len(self.datasets) - 1))
            self.data = self.datasets[self.active_idx]
        apply_calibration_from_session(self, session)
        if session.get("cursors"):
            restore_cursors_to_phasor(self, session["cursors"])
        self._restore_ui_for_active()
        self._refresh_image_combo()
        self._settings.setValue("session_dir", os.path.dirname(path))
        self._mark_clean()
        self._log(f"Session JSON loaded from {os.path.basename(path)} — load/decode samples and Apply.")

    def save_session(self):
        """Save all processed results and UI state to a portable session bundle.
        Bound to File → "Save session…" and the Ctrl+Shift+S shortcut. Requires
        that Apply has run on the active dataset (checks ``self.data.real_cal
        is not None``); if not, shows an informational dialog explaining that
        the bundle stores processed maps rather than raw PTU data, and returns
        without prompting. Otherwise prompts for a save path via a file dialog
        (defaulting to the remembered ``"session_dir"`` or, failing that,
        ``"export_dir"``), appending the bundle extension if missing. Calls
        ``save_session_bundle`` with ``self`` and the chosen path to actually
        serialize datasets, calibration, cursors, and settings; any exception
        is shown in a critical dialog and aborts without further changes. On
        success, remembers the containing directory as ``"session_dir"`` and
        logs a summary line with the resulting file name, sample count, and
        size on disk.
        """
        if self.data.real_cal is None:
            QtWidgets.QMessageBox.information(
                self,
                "Save session",
                "Run Apply on at least one image first — the bundle stores processed maps, not raw PTU data.",
            )
            return
        default = self._dialog_dir("session_dir", self._dialog_dir("export_dir"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save session bundle",
            default,
            f"Session bundle (*{BUNDLE_EXTENSION})",
        )
        if not path:
            return
        if not path.lower().endswith(BUNDLE_EXTENSION):
            path += BUNDLE_EXTENSION
        try:
            result = save_session_bundle(self, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save session", str(e))
            return
        self._settings.setValue("session_dir", os.path.dirname(result["path"]))
        self._mark_clean()
        self._log(
            f"Session saved → {os.path.basename(result['path'])} "
            f"({result['n_samples']} sample{'s' if result['n_samples'] != 1 else ''}, "
            f"{result['size_mb']:.2f} MB)"
        )

    def batch_export_folder(self):
        """Batch-process every supported FLIM file in a folder using current settings.
        Bound to File → "Batch export folder…". Prompts the user for an input
        folder (containing FLIM files to process) and then an output folder for
        results; returns without doing anything if either dialog is cancelled.
        Builds a command-line argument list for
        ``flim_phasors.batch_cli.main`` mirroring the current UI's harmonic,
        frequency, filter mode, and intensity threshold, and, if a shared
        reference file is set, also passes the reference path, reference
        channel, and reference lifetime for calibration. Imports
        ``batch_cli.main`` lazily to avoid a module-level dependency, and runs
        it inside ``_run_busy`` so the UI shows a "Batch processing…" busy
        indicator while every file in the folder is processed and exported.
        Logs a completion message with the output folder on success, or shows
        a critical error dialog if the batch run raises.
        """
        inp = QtWidgets.QFileDialog.getExistingDirectory(self, "Input folder of FLIM files")
        if not inp:
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if not out:
            return
        from flim_phasors.batch_cli import main as batch_main

        ref = self.shared_ref_path or ""
        argv = [inp, "-o", out, "--harmonic", str(self.sp_harm.value()),
                "--frequency", str(self.sp_freq.value()),
                "--filter", self.cb_filter.currentText(),
                "--min-photons", str(self.sp_thr.value())]
        if ref:
            argv.extend(["-r", ref, "--ref-channel", str(self.shared_ref_channel),
                         "--ref-lifetime", str(self.sp_reflt.value())])
        try:
            self._run_busy("Batch processing…", lambda: batch_main(argv))
            self._log(f"Batch export complete → {out}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Batch", str(e))

    def show_about(self):
        """Display the About dialog with application and dependency versions.
        Bound to Help → "About…". Attempts to import ``phasorpy`` to read its
        ``__version__`` (falling back to ``"not installed"`` if the import
        fails), then shows a standard ``QMessageBox.about`` dialog listing the
        FLIM Phasor Analyzer version (from ``flim_phasors.__version__``), the
        phasorpy version, and the running Python version. Purely informational;
        does not change any application state.
        """
        try:
            import phasorpy
            pp_ver = getattr(phasorpy, "__version__", "unknown")
        except ImportError:
            pp_ver = "not installed"
        QtWidgets.QMessageBox.about(
            self,
            "About FLIM Phasors",
            f"FLIM Phasor Analyzer v{__version__}\n"
            f"phasorpy {pp_ver}\nPython {sys.version.split()[0]}",
        )

    def _load_ui_theme_setting(self) -> str:
        """Determine which Phasor Lab theme to start the window with.
        Called once from ``_init_enhancements`` before any theme is applied.
        Prefers the current ``"ui_theme"`` QSettings key if present, normalized
        via ``normalize_theme_id`` to tolerate legacy/typo'd values. Falls back
        to the legacy boolean ``"dark_theme"`` key (from before named themes
        existed) if present, mapping ``True`` to ``THEME_PHASOR_LAB`` and
        ``False`` to ``THEME_PHASOR_LAB_LIGHT``. If neither key exists (first
        run), returns ``DEFAULT_THEME``. Does not itself apply the theme or
        write any settings.
        Returns:
            The theme id string to use as the initial theme.
        """
        if self._settings.contains("ui_theme"):
            return normalize_theme_id(str(self._settings.value("ui_theme", DEFAULT_THEME)))
        if self._settings.contains("dark_theme"):
            legacy = bool(self._settings.value("dark_theme", True))
            return THEME_PHASOR_LAB if legacy else THEME_PHASOR_LAB_LIGHT
        return DEFAULT_THEME

    def _tag_primary_buttons(self):
        """Flag the key workflow buttons so the theme stylesheet accents them.
        Called once from ``_init_enhancements``, after ``_build_ui`` has
        created the buttons named in ``PRIMARY_BUTTON_ATTRS`` (Calibrate,
        Apply, Apply all, Export, Paint). For each attribute name in that
        tuple, if the corresponding button exists on ``self``, sets its
        ``"primary"`` Qt dynamic property to ``True``; the stylesheets in
        ``theme.py`` key off ``QPushButton[primary="true"]`` to render these
        buttons with an accent color, distinguishing the primary workflow
        actions from ordinary buttons. Does not itself repaint the buttons —
        see ``_repolish_primary_buttons`` for forcing Qt to re-evaluate the
        selector after a stylesheet change.
        """
        for attr in PRIMARY_BUTTON_ATTRS:
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setProperty("primary", True)

    def _repolish_primary_buttons(self):
        """Force Qt to re-evaluate primary-button styling after a stylesheet swap.
        Called at the end of ``_apply_ui_theme``. Setting a new application
        stylesheet does not automatically re-run dynamic-property selectors
        (like ``QPushButton[primary="true"]``) on widgets that already exist,
        so for each button attribute named in ``PRIMARY_BUTTON_ATTRS`` that
        exists on ``self``, calls ``style().unpolish(btn)`` followed by
        ``style().polish(btn)`` to force Qt to recompute and reapply the
        widget's style from the new stylesheet, ensuring the accent color
        actually updates when switching between Phasor Lab themes.
        """
        # unpolish/polish forces Qt to re-run QPushButton[primary="true"] selectors.
        for attr in PRIMARY_BUTTON_ATTRS:
            btn = getattr(self, attr, None)
            if btn is not None:
                style = btn.style()
                style.unpolish(btn)
                style.polish(btn)

    def _apply_ui_theme(self, theme: str):
        """Switch the entire application to a Phasor Lab theme and persist it.
        Called from ``_init_enhancements`` at startup and whenever the user
        picks a theme from the View → Theme menu (each theme action's
        ``triggered`` signal is wired directly to this method with its theme
        id). Normalizes ``theme`` via ``normalize_theme_id``, updates
        ``self._ui_theme`` and the derived ``self._dark_theme`` boolean, and
        writes both the theme id and dark-mode flag to ``self._settings`` so
        the choice persists across restarts (the boolean is kept for backward
        compatibility with older settings readers). Applies the theme's global
        stylesheet via ``self.setStyleSheet(stylesheet_for(theme))``, then
        calls ``_apply_theme_widgets`` to restyle the activity log and
        matplotlib toolbars (which use manual palettes, not just the
        stylesheet) and ``_repolish_primary_buttons`` to force existing accent
        buttons to pick up the new colors. Finally, updates the checked state
        of every action in ``self._theme_actions`` so the menu's radio-style
        checkmark matches the newly active theme.
        Args:
            theme: Theme id to activate, e.g. ``THEME_PHASOR_LAB`` or
                ``THEME_PHASOR_LAB_LIGHT`` (tolerant of legacy/typo'd values
                via ``normalize_theme_id``).
        """
        theme = normalize_theme_id(theme)
        self._ui_theme = theme
        self._dark_theme = is_dark_theme(theme)
        self._settings.setValue("ui_theme", theme)
        self._settings.setValue("dark_theme", self._dark_theme)
        self.setStyleSheet(stylesheet_for(theme))
        self._apply_theme_widgets(theme)
        # Stylesheet swap does not re-evaluate dynamic [primary="true"] rules on existing buttons.
        self._repolish_primary_buttons()
        for theme_id, act in getattr(self, "_theme_actions", {}).items():
            act.setChecked(theme_id == theme)

    def _apply_theme_widgets(self, theme: str):
        """Restyle widgets that need manual palette updates beyond the stylesheet.
        Called from ``_apply_ui_theme`` after the global stylesheet is set.
        The activity log (``self.txt_log``, if present) gets its stylesheet
        replaced via ``log_style_for(theme)``. Each matplotlib navigation
        toolbar attribute (``phasor_toolbar``, ``image_toolbar``, if present on
        ``self``) gets its Qt stylesheet set via ``toolbar_style_for(theme)``
        and, because matplotlib toolbars render some backgrounds via the
        widget palette rather than the stylesheet alone, also has its
        ``QPalette`` window/button/text colors explicitly set from
        ``toolbar_colors_for(theme)`` and ``setAutoFillBackground(True)``
        enabled so the palette colors actually show through.
        Args:
            theme: Theme id whose colors should be applied (any legacy/typo'd
                value is normalized inside the ``*_for`` helper functions).
        """
        if hasattr(self, "txt_log"):
            self.txt_log.setStyleSheet(log_style_for(theme))
        tb_style = toolbar_style_for(theme)
        tb_bg_hex, tb_fg_hex = toolbar_colors_for(theme)
        tb_bg = QtGui.QColor(tb_bg_hex)
        tb_fg = QtGui.QColor(tb_fg_hex)
        for attr in ("phasor_toolbar", "image_toolbar"):
            tb = getattr(self, attr, None)
            if tb is not None:
                tb.setStyleSheet(tb_style)
                pal = tb.palette()
                pal.setColor(QtGui.QPalette.ColorRole.Window, tb_bg)
                pal.setColor(QtGui.QPalette.ColorRole.Button, tb_bg)
                pal.setColor(QtGui.QPalette.ColorRole.WindowText, tb_fg)
                pal.setColor(QtGui.QPalette.ColorRole.ButtonText, tb_fg)
                tb.setPalette(pal)
                tb.setAutoFillBackground(True)

    def _push_cursor_undo(self):
        """Push a deep copy of current cursors onto the undo stack (max 30)."""
        import copy
        self._cursor_undo_stack.append(copy.deepcopy(self.phasor.cursors))
        if len(self._cursor_undo_stack) > 30:
            self._cursor_undo_stack.pop(0)

    def undo_cursor(self):
        """Revert the most recent cursor edit, bound to the Ctrl+Z shortcut.
        Triggered by the "Undo" button or the Ctrl+Z shortcut. If
        ``self._cursor_undo_stack`` is empty, logs "Nothing to undo." and
        returns without changing anything. Otherwise pops the most recent
        snapshot pushed by ``_push_cursor_undo`` and assigns it to
        ``self.phasor.cursors``, replacing the current cursor set with the
        prior one. Calls ``self.phasor.redraw_hist()`` to repaint the phasor
        plot with the restored cursors, ``_refresh_active_cursor_combo`` to
        update the active-cursor selector to match, and
        ``_refresh_after_cursor_edit`` to recompute any dependent
        segmentation/statistics, then logs "Cursor undo."
        """
        if not self._cursor_undo_stack:
            self._log("Nothing to undo.")
            return
        self.phasor.cursors = self._cursor_undo_stack.pop()
        self.phasor.redraw_hist()
        self._refresh_active_cursor_combo()
        self._refresh_after_cursor_edit()
        self._log("Cursor undo.")

    def save_cursors_file(self):
        """Save the current phasor cursor ROIs to a JSON file.
        Bound to the "Save cursors…" button, and the Ctrl+Shift+U shortcut. If
        there are no cursors defined (``self.phasor.cursors`` is empty), shows
        an informational dialog and returns without prompting. Otherwise
        prompts for a save path via a file dialog (defaulting to the
        remembered ``"sample_dir"``); returns if cancelled. Calls
        ``save_cursors`` with the cursor list and the active sample's path
        (stored as a hint for later matching on load), then logs a
        confirmation line with the saved path.
        """
        if not self.phasor.cursors:
            QtWidgets.QMessageBox.information(self, "Cursors", "No cursors to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save cursors", self._dialog_dir("sample_dir"), "JSON (*.json)")
        if not path:
            return
        save_cursors(path, self.phasor.cursors, sample_path=self.data.sample_path)
        self._log(f"Cursors saved → {path}")

    def load_cursors_file(self):
        """Load phasor cursor ROIs from a JSON file and apply them to the plot.
        Bound to the "Load cursors…" button and the Ctrl+Shift+Y shortcut.
        Prompts for a file via a file dialog (defaulting to the remembered
        ``"sample_dir"``); returns if cancelled. Parses the file with
        ``load_cursors`` into a list of cursor dicts plus a ``sample_hint``
        (the original sample path they were saved against, currently unused
        for matching but reserved for future validation). Restores the cursors
        onto the phasor plot via ``restore_cursors_to_phasor``, refreshes the
        active-cursor combo box (``_refresh_active_cursor_combo``) and
        dependent segmentation/statistics (``_refresh_after_cursor_edit``), and
        logs the count of cursors loaded and the source file name. Note this
        does not push an undo snapshot first, so loading replaces the current
        cursors without an undo path back to them.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load cursors", self._dialog_dir("sample_dir"), "JSON (*.json)")
        if not path:
            return
        cursors, sample_hint = load_cursors(path)
        restore_cursors_to_phasor(self, cursors)
        self._refresh_active_cursor_combo()
        self._refresh_after_cursor_edit()
        self._log(f"Loaded {len(cursors)} cursor(s) from {os.path.basename(path)}.")

    def export_table_csv(self):
        """Export the current cluster/lifetime statistics table to a CSV file.
        Bound to the "Export table CSV…" button. If ``self.cluster_stats`` is
        empty (Paint has not been run yet), shows an informational dialog and
        returns without prompting. Otherwise prompts for a save path via a
        file dialog (defaulting to the remembered ``"export_dir"``); returns if
        cancelled. Writes one row per cluster with the fixed column set
        ``idx, label, g, s, tp, tm, tn, n, area`` using ``csv.DictWriter`` with
        ``extrasaction="ignore"`` (any extra keys present on the stat dicts,
        e.g. internal bookkeeping fields, are silently dropped), then logs a
        confirmation line with the saved path.
        """
        if not self.cluster_stats:
            QtWidgets.QMessageBox.information(self, "Export", "No cluster data — run Paint first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export table", self._dialog_dir("export_dir"), "CSV (*.csv)")
        if not path:
            return
        fields = [
            "idx", "label", "g", "s", "tp", "tm", "tn", "n", "area",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.cluster_stats)
        self._log(f"Table exported → {path}")

    def copy_table_to_clipboard(self):
        """Copy the cluster/lifetime statistics table as tab-separated text.
        Bound to the "Copy table" button. Does nothing if
        ``self.cluster_stats`` is empty (Paint has not been run yet). Otherwise
        builds a header line (``#, Label, g, s, τφ, τmod, τn, Pixels, Area%``)
        followed by one tab-separated line per cluster with values formatted to
        a fixed number of decimal places, joins them with newlines, and sets
        the result as the system clipboard's text via
        ``QApplication.clipboard().setText`` — suitable for pasting directly
        into a spreadsheet. Logs a confirmation line afterward.
        """
        if not self.cluster_stats:
            return
        lines = ["#\tLabel\tg\ts\tτφ\tτmod\tτn\tPixels\tArea%"]
        for st in self.cluster_stats:
            lines.append(
                f"{st['idx']}\t{st['label']}\t{st['g']:.4f}\t{st['s']:.4f}\t"
                f"{st['tp']:.3f}\t{st['tm']:.3f}\t{st['tn']:.3f}\t"
                f"{st['n']}\t{st['area']:.2f}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self._log("Table copied to clipboard.")

    def _effective_ref_label(self, d) -> str:
        """Return a short display label for a dataset's effective reference file.
        Used when populating the compare/multi-sample table, where each row
        needs a compact indicator of which reference file it was (or will be)
        calibrated against. If shared-reference mode is enabled
        (``self.chk_shared_ref`` checked) and a shared reference path is set,
        returns that shared file's base name — since in that mode every
        dataset uses the same reference regardless of its own ``ref_path``.
        Otherwise returns the dataset's own ``ref_path`` base name, or an
        em dash (``"—"``) placeholder if it has no reference assigned.
        Args:
            d: Dataset (:class:`PhasorData`-like) whose reference label to
                compute.
        Returns:
            Base file name of the effective reference, or ``"—"`` if none.
        """
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return os.path.basename(self.shared_ref_path)
        ref = d.ref_path or ""
        return os.path.basename(ref) if ref else "—"
