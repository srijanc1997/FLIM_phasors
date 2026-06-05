"""Restore analysis sessions from exported session.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from flim_phasors.calibration import ReferenceCalibration
# from flim_phasors.calibration_io import calibration_from_dict  # unused (focused cleanup)
# from flim_phasors.cursors_io import load_cursors  # unused (focused cleanup)
from flim_phasors.data import PhasorData


def load_session_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_calibration_from_session(win, session: dict):
    cal_block = session.get("calibration") or {}
    cal = ReferenceCalibration(
        source_path=str(cal_block.get("reference_path", "")),
        channel=int(cal_block.get("reference_channel", 0)),
        mean_g=float(cal_block.get("mean_g", 0.0)),
        mean_s=float(cal_block.get("mean_s", 0.0)),
        use_manual=bool(cal_block.get("manual", False)),
        manual_g=float(cal_block.get("manual_g", 0.0)),
        manual_s=float(cal_block.get("manual_s", 0.0)),
    )
    if cal.use_manual:
        cal.mean_g = cal.manual_g
        cal.mean_s = cal.manual_s
    win.ref_calibration = cal
    ref_path = cal_block.get("reference_path") or session.get("shared_reference_path") or ""
    if ref_path and os.path.isfile(ref_path):
        win.shared_ref_path = ref_path
        win.data.ref_path = ref_path
    if hasattr(win, "sp_freq"):
        win.sp_freq.setValue(float(cal_block.get("frequency_MHz", win.sp_freq.value())))
    if hasattr(win, "sp_harm"):
        win.sp_harm.setValue(int(cal_block.get("harmonic", win.sp_harm.value())))
    if hasattr(win, "sp_reflt"):
        win.sp_reflt.setValue(float(cal_block.get("reference_lifetime_ns", win.sp_reflt.value())))
    if hasattr(win, "cb_filter") and cal_block.get("filter"):
        win.cb_filter.setCurrentText(str(cal_block["filter"]))
    if hasattr(win, "sp_thr"):
        win.sp_thr.setValue(int(cal_block.get("min_photons", 0)))
    if hasattr(win, "chk_detect_harm"):
        win.chk_detect_harm.setChecked(bool(cal_block.get("harmonic_mask", True)))
    if hasattr(win, "chk_manual_cal"):
        win.chk_manual_cal.setChecked(cal.use_manual)
    if hasattr(win, "_sync_manual_fields_from_calibration"):
        win._sync_manual_fields_from_calibration()
    if hasattr(win, "_update_calibration_display"):
        win._update_calibration_display()


def restore_cursors_to_phasor(win, cursors: list[dict]):
    win.phasor.clear_cursors()
    for c in cursors:
        kind = c.get("kind", "circle")
        rm = c.get("radius_minor")
        win.phasor.add_cursor(
            radius=float(c["radius"]),
            kind=kind,
            radius_minor=float(rm) if rm is not None else None,
            angle=float(c.get("angle", 0.0)),
        )
        win.phasor.cursors[-1]["center_real"] = float(c["center_real"])
        win.phasor.cursors[-1]["center_imag"] = float(c["center_imag"])
        win.phasor.cursors[-1]["label"] = c.get("label", "")
        if "color" in c:
            win.phasor.cursors[-1]["color"] = tuple(c["color"][:3])
    win.phasor.redraw_hist()
    if hasattr(win, "_refresh_active_cursor_combo"):
        win._refresh_active_cursor_combo()


def register_sample_from_session_row(row: dict) -> PhasorData:
    d = PhasorData()
    path = row.get("sample_path", "")
    d.sample_path = path
    d.group_name = (row.get("group") or "").strip()
    d.channel = int(row.get("channel", 0))
    d.harmonic = int(row.get("harmonic", 1))
    d.frequency = float(row.get("frequency_MHz", 80.0))
    d.ref_path = row.get("reference_path") or ""
    if d.ref_path:
        d.ref_channel = int(row.get("reference_channel", 0))
    return d


def missing_paths_message(session: dict) -> list[str]:
    missing = []
    for row in session.get("samples", []):
        p = row.get("sample_path", "")
        if p and not os.path.isfile(p):
            missing.append(p)
    ref = (session.get("calibration") or {}).get("reference_path") or session.get("shared_reference_path")
    if ref and not os.path.isfile(ref):
        missing.append(ref)
    return missing
