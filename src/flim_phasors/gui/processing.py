"""Processing helpers (extracted from MainWindow for maintainability)."""

from __future__ import annotations

from flim_phasors.data import PhasorData

# Keys stored per sample when "Per-sample filters" is enabled (plus runtime ref/cal).
PROC_SETTING_KEYS = (
    "harmonic",
    "frequency",
    "channel",
    "ref_lifetime",
    "filter_mode",
    "median_size",
    "median_repeat",
    "paw_sigma",
    "paw_levels",
    "intensity_min",
    "detect_harmonics",
)


def capture_processing_from_ui(win) -> dict:
    """Snapshot filter / threshold / harmonic controls from the main window."""
    mode = win.cb_filter.currentText()
    return {
        "harmonic": int(win.sp_harm.value()),
        "frequency": float(win.sp_freq.value()),
        "channel": max(0, win.cb_channel.currentIndex()),
        "ref_lifetime": float(win.sp_reflt.value()),
        "filter_mode": mode,
        "median_size": int(win.sp_msize.value()),
        "median_repeat": int(win.sp_mrep.value()),
        "paw_sigma": float(win.sp_psigma.value()),
        "paw_levels": int(win.sp_plevels.value()),
        "intensity_min": float(win.sp_thr.value()),
        "detect_harmonics": bool(win.chk_detect_harm.isChecked()),
    }


def per_sample_processing(win) -> bool:
    """True when multiple samples are loaded — each keeps its own filter settings."""
    return len(getattr(win, "datasets", [])) > 1


def filter_label_for_dataset(win, d: PhasorData) -> str:
    """Filter name for exports/logs — per-sample stash or current UI."""
    stash = getattr(d, "processing_settings", None)
    if stash and stash.get("filter_mode"):
        return str(stash["filter_mode"])
    if hasattr(win, "cb_filter"):
        return win.cb_filter.currentText()
    return "median"


def apply_processing_settings_to_ui(win, settings: dict) -> None:
    """Load stored settings into widgets (signals blocked by caller)."""
    if not settings:
        return
    if "harmonic" in settings:
        win.sp_harm.setValue(int(settings["harmonic"]))
    if "frequency" in settings:
        win.sp_freq.setValue(float(settings["frequency"]))
    if "channel" in settings:
        n = max(1, win.cb_channel.count())
        win.cb_channel.setCurrentIndex(min(int(settings["channel"]), n - 1))
    if "ref_lifetime" in settings:
        win.sp_reflt.setValue(float(settings["ref_lifetime"]))
    mode = settings.get("filter_mode", "median")
    if mode in [win.cb_filter.itemText(i) for i in range(win.cb_filter.count())]:
        win.cb_filter.setCurrentText(mode)
    win.on_filter_change(win.cb_filter.currentText())
    if "median_size" in settings:
        win.sp_msize.setValue(int(settings["median_size"]))
    if "median_repeat" in settings:
        win.sp_mrep.setValue(int(settings["median_repeat"]))
    if "paw_sigma" in settings:
        win.sp_psigma.setValue(float(settings["paw_sigma"]))
    if "paw_levels" in settings:
        win.sp_plevels.setValue(int(settings["paw_levels"]))
    if "intensity_min" in settings:
        win.sp_thr.setValue(int(settings["intensity_min"]))
    if "detect_harmonics" in settings:
        win.chk_detect_harm.setChecked(bool(settings["detect_harmonics"]))


def processing_params_from_ui(win, d: PhasorData) -> dict:
    """Build kwargs for PhasorData.apply_processing from window widgets."""
    mode = win.cb_filter.currentText()
    ref_path = win._effective_ref_path(d)
    return {
        "ref_calibration": win._active_calibration(),
        "ref_path": ref_path,
        "ref_lifetime": win.sp_reflt.value(),
        "filter_mode": mode,
        "median_size": win.sp_msize.value(),
        "median_repeat": win.sp_mrep.value(),
        "paw_sigma": win.sp_psigma.value(),
        "paw_levels": win.sp_plevels.value(),
        "intensity_min": float(win.sp_thr.value()),
        "detect_harmonics": win.chk_detect_harm.isChecked(),
    }


def processing_params_for_dataset(win, d: PhasorData) -> dict:
    """Per-sample stash when multiple samples loaded, otherwise current UI."""
    if per_sample_processing(win):
        stash = getattr(d, "processing_settings", None) or {}
        if stash:
            mode = stash.get("filter_mode", "median")
            ref_path = win._effective_ref_path(d)
            return {
                "ref_calibration": win._active_calibration(),
                "ref_path": ref_path,
                "ref_lifetime": float(stash.get("ref_lifetime", win.sp_reflt.value())),
                "filter_mode": mode,
                "median_size": int(stash.get("median_size", win.sp_msize.value())),
                "median_repeat": int(stash.get("median_repeat", win.sp_mrep.value())),
                "paw_sigma": float(stash.get("paw_sigma", win.sp_psigma.value())),
                "paw_levels": int(stash.get("paw_levels", win.sp_plevels.value())),
                "intensity_min": float(stash.get("intensity_min", win.sp_thr.value())),
                "detect_harmonics": bool(
                    stash.get("detect_harmonics", win.chk_detect_harm.isChecked())),
            }
    return processing_params_from_ui(win, d)


def apply_dataset_harmonic_channel(win, d: PhasorData, *, use_ui_settings: bool) -> None:
    """Set harmonic / frequency / channel on d before apply_processing."""
    if not use_ui_settings:
        return
    if per_sample_processing(win):
        stash = getattr(d, "processing_settings", None) or {}
        if stash:
            d.harmonic = int(stash.get("harmonic", win.sp_harm.value()))
            d.frequency = float(stash.get("frequency", win.sp_freq.value()))
            d.channel = int(stash.get("channel", max(0, win.cb_channel.currentIndex())))
            return
    d.harmonic = win.sp_harm.value()
    d.frequency = win.sp_freq.value()
    d.channel = max(0, win.cb_channel.currentIndex())


def run_processing_on_dataset(win, d: PhasorData, *, use_ui_settings: bool = False):
    apply_dataset_harmonic_channel(win, d, use_ui_settings=use_ui_settings)
    if win._effective_ref_path(d):
        d.ref_channel = win._ref_channel_for_dataset(d)
    d.apply_processing(**processing_params_for_dataset(win, d))
