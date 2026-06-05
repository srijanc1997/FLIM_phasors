"""Optional features mixin for MainWindow (menus, session, lazy load, etc.)."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from flim_phasors import __version__
# from flim_phasors.busy import CancelledError, run_busy_qt  # unused (focused cleanup)
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


class EnhancementsMixin:
    """Mixin methods — call ``_init_enhancements()`` from MainWindow after ``_build_ui``."""

    def _init_enhancements(self):
        self._cal_settings_hash = ""
        self._cursor_undo_stack: list[list] = []
        self._dark_theme = bool(self._settings.value("dark_theme", False))
        self._extend_ui()
        self._build_menus()
        self._setup_shortcuts()
        self._setup_drag_drop()
        if self._dark_theme:
            self._apply_dark_theme(True)

    def _extend_ui(self):
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
        self.sp_pixel_um.setToolTip("Optional pixel size for scale bar on images (0 = off).")
        row_px = QtWidgets.QHBoxLayout()
        row_px.addWidget(QtWidgets.QLabel("Pixel"))
        row_px.addWidget(self.sp_pixel_um)
        self.proc_grid.addLayout(row_px, 14, 0, 1, 4)

        self.chk_log_display = QtWidgets.QCheckBox("Log photons")
        self.chk_log_display.stateChanged.connect(self.refresh_image)
        self.chk_auto_contrast = QtWidgets.QCheckBox("Auto contrast")
        self.chk_auto_contrast.setChecked(True)
        self.chk_auto_contrast.stateChanged.connect(self.refresh_image)
        row_disp = QtWidgets.QHBoxLayout()
        row_disp.addWidget(self.chk_log_display)
        row_disp.addWidget(self.chk_auto_contrast)
        self.proc_grid.addLayout(row_disp, 15, 0, 1, 4)

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


    def _build_menus(self):
        mb = self.menuBar()
        file_m = mb.addMenu("&File")
        file_m.addAction("Sample…", self.choose_sample, "Ctrl+O")
        file_m.addAction("Reference…", self.choose_ref, "Ctrl+R")
        file_m.addAction("Calibrate", self.calibrate_reference)
        file_m.addSeparator()
        self._recent_samples_menu = file_m.addMenu("Recent samples")
        self._recent_refs_menu = file_m.addMenu("Recent references")
        file_m.addSeparator()
        file_m.addAction("Open session…", self.open_session, "Ctrl+Shift+O")
        file_m.addAction("Save session…", self.save_session, "Ctrl+Shift+S")
        file_m.addAction("Save calibration…", self.save_calibration_file)
        file_m.addAction("Load calibration…", self.load_calibration_file)
        file_m.addSeparator()
        file_m.addAction("Batch export folder…", self.batch_export_folder)
        file_m.addAction("Export all…", self.export_all, "Ctrl+E")
        file_m.addSeparator()
        file_m.addAction("E&xit", self.close, "Ctrl+Q")

        view_m = mb.addMenu("&View")
        self._act_dark = view_m.addAction("Dark theme")
        self._act_dark.setCheckable(True)
        self._act_dark.setChecked(self._dark_theme)
        self._act_dark.toggled.connect(self._apply_dark_theme)

        help_m = mb.addMenu("&Help")
        help_m.addAction("About…", self.show_about)

        self._refresh_recent_menus()

    def _setup_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, self.apply_processing)

    def _setup_drag_drop(self):
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if is_supported_flim_path(url.toLocalFile()):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        paths = [
            url.toLocalFile() for url in event.mimeData().urls()
            if is_supported_flim_path(url.toLocalFile())
        ]
        if not paths:
            return
        if not self._prepare_sample_load(paths):
            return
        self._load_sample_paths(paths)

    def _remember_recent(self, key: str, path: str, max_items: int = 8):
        items = list(self._settings.value(key, []) or [])
        if isinstance(items, str):
            items = [items]
        path = os.path.abspath(path)
        items = [p for p in items if p != path]
        items.insert(0, path)
        self._settings.setValue(key, items[:max_items])
        self._refresh_recent_menus()

    def _refresh_recent_menus(self):
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
                    menu.addAction(os.path.basename(p), lambda checked=False, path=p: handler(path))

    def _open_recent_sample(self, path):
        if not self._prepare_sample_load([path]):
            return
        self._load_sample_paths([path])

    def _open_recent_ref(self, path):
        self.choose_ref_with_path(path)

    def choose_ref_with_path(self, path: str):
        if not path or not is_supported_flim_path(path):
            return
        self._set_reference_path(path)

    def _calibration_settings_tuple(self):
        return (
            int(self.sp_harm.value()),
            self.cb_filter.currentText(),
            int(self.cb_ref_channel.currentIndex()) if self.cb_ref_channel.isEnabled() else 0,
            bool(self.chk_manual_cal.isChecked()),
        )

    def _mark_calibration_current(self):
        self._cal_settings_hash = str(self._calibration_settings_tuple())
        self._update_calibration_stale_style()

    def _update_calibration_stale_style(self):
        stale = (
            self.ref_calibration.is_active
            and not self.chk_manual_cal.isChecked()
            and str(self._calibration_settings_tuple()) != self._cal_settings_hash
        )
        color = "#b45309" if stale else "gray"
        self.lbl_cal_display.setStyleSheet(f"color: {color}; font-size: 10px;")

    def _update_ref_preview(self):
        if hasattr(self, "ref_preview"):
            self.ref_preview.show_calibration(
                self.ref_calibration,
                ref_lifetime_ns=self.sp_reflt.value(),
                frequency_mhz=self.sp_freq.value(),
            )

    def _update_metadata_panel(self):
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

    def open_session(self):
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
        try:
            loaded = load_session_bundle(path)
            apply_session_bundle_to_window(self, loaded)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Session", str(e))
            return
        n = len(loaded["datasets"])
        size_mb = os.path.getsize(path) / (1024 * 1024)
        self._settings.setValue("session_dir", os.path.dirname(path))
        self._log(
            f"Session bundle loaded ({n} sample{'s' if n != 1 else ''}, "
            f"{size_mb:.2f} MB) — no PTU/TIF required."
        )

    def _open_session_json(self, path: str):
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
        self._log(f"Session JSON loaded from {os.path.basename(path)} — load/decode samples and Apply.")

    def save_session(self):
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
        self._log(
            f"Session saved → {os.path.basename(result['path'])} "
            f"({result['n_samples']} sample{'s' if result['n_samples'] != 1 else ''}, "
            f"{result['size_mb']:.2f} MB)"
        )

    def batch_export_folder(self):
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

    def _apply_dark_theme(self, on: bool):
        self._dark_theme = on
        self._settings.setValue("dark_theme", on)
        if on:
            self.setStyleSheet(
                "QWidget { background-color: #2b2b2b; color: #e0e0e0; }"
                "QGroupBox { border: 1px solid #555; margin-top: 6px; padding-top: 6px; }"
                "QPlainTextEdit { background-color: #1e1e1e; }")
        else:
            self.setStyleSheet("")

    def _push_cursor_undo(self):
        import copy
        self._cursor_undo_stack.append(copy.deepcopy(self.phasor.cursors))
        if len(self._cursor_undo_stack) > 30:
            self._cursor_undo_stack.pop(0)

    def undo_cursor(self):
        if not self._cursor_undo_stack:
            self._log("Nothing to undo.")
            return
        self.phasor.cursors = self._cursor_undo_stack.pop()
        self.phasor.redraw_hist()
        self._refresh_active_cursor_combo()
        self._refresh_after_cursor_edit()
        self._log("Cursor undo.")

    def save_cursors_file(self):
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
        if not self.cluster_stats:
            QtWidgets.QMessageBox.information(self, "Export", "No cluster data — run Paint first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export table", self._dialog_dir("export_dir"), "CSV (*.csv)")
        if not path:
            return
        fields = ["idx", "label", "g", "s", "tp", "tm", "tn", "n", "area"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.cluster_stats)
        self._log(f"Table exported → {path}")

    def copy_table_to_clipboard(self):
        if not self.cluster_stats:
            return
        lines = ["#\tLabel\tg\ts\tτφ\tτmod\tτn\tPixels\tArea%"]
        for st in self.cluster_stats:
            lines.append(
                f"{st['idx']}\t{st['label']}\t{st['g']:.4f}\t{st['s']:.4f}\t"
                f"{st['tp']:.3f}\t{st['tm']:.3f}\t{st['tn']:.3f}\t{st['n']}\t{st['area']:.2f}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self._log("Table copied to clipboard.")

    def _effective_ref_label(self, d) -> str:
        if self.chk_shared_ref.isChecked() and self.shared_ref_path:
            return os.path.basename(self.shared_ref_path)
        ref = d.ref_path or ""
        return os.path.basename(ref) if ref else "—"
