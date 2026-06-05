"""Processing helpers (extracted from MainWindow for maintainability).

Centralizes capture and application of per-sample filter settings, harmonic/channel
selection, and kwargs assembly for :meth:`PhasorData.apply_processing`.
"""

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
    """Snapshot filter, threshold, and harmonic controls from the main window.

    Args:
        win: Main window with processing widgets.

    Returns:
        Dict of processing settings suitable for stashing on a :class:`PhasorData`.
    """
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
    """Return whether multiple samples are loaded with independent filter settings.

    Args:
        win: Main window instance.

    Returns:
        ``True`` when ``win.datasets`` contains more than one sample.
    """
    return len(getattr(win, "datasets", [])) > 1


def filter_label_for_dataset(win, d: PhasorData) -> str:
    """Return the filter name for exports and logs.

    Prefers per-sample stashed settings; falls back to the current UI selection.

    Args:
        win: Main window instance.
        d: Dataset whose stashed ``processing_settings`` may override the UI.

    Returns:
        Human-readable filter mode string (e.g. ``"median"``).
    """
    stash = getattr(d, "processing_settings", None)
    if stash and stash.get("filter_mode"):
        return str(stash["filter_mode"])
    if hasattr(win, "cb_filter"):
        return win.cb_filter.currentText()
    return "median"


def apply_processing_settings_to_ui(win, settings: dict) -> None:
    """Load stored processing settings into main-window widgets.

    Caller should block widget signals before invoking this function.

    Args:
        win: Main window whose controls are updated.
        settings: Dict produced by :func:`capture_processing_from_ui` or loaded
            from a session bundle.
    """
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
    """Build keyword arguments for :meth:`PhasorData.apply_processing` from the UI.

    Args:
        win: Main window with current control values.
        d: Target dataset (used for effective reference path resolution).

    Returns:
        Dict of kwargs accepted by :meth:`PhasorData.apply_processing`.
    """
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
    """Return processing kwargs for a dataset, honoring per-sample stash when active.

    Args:
        win: Main window instance.
        d: Dataset to process.

    Returns:
        Dict of kwargs for :meth:`PhasorData.apply_processing`.
    """
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
    """Set harmonic, frequency, and channel on a dataset before processing.

    Args:
        win: Main window instance.
        d: Dataset whose acquisition parameters are updated in place.
        use_ui_settings: When ``False``, leaves ``d`` unchanged.
    """
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


def run_processing_on_dataset(
    win,
    d: PhasorData,
    *,
    use_ui_settings: bool = False,
    calibrate: bool = True,
):
    """Run the full apply-processing pipeline on one dataset.

    Updates harmonic/channel from UI or stash, resolves reference channel when
    calibrating, and sets ``d.maps_calibrated`` from calibration activity.

    Args:
        win: Main window providing calibration and processing parameters.
        d: Dataset to process in place.
        use_ui_settings: When ``True``, copy harmonic/frequency/channel from UI
            or per-sample stash onto ``d``.
        calibrate: When ``False``, skip reference calibration and clear ref kwargs.
    """
    apply_dataset_harmonic_channel(win, d, use_ui_settings=use_ui_settings)
    if calibrate and win._effective_ref_path(d):
        d.ref_channel = win._ref_channel_for_dataset(d)
    params = processing_params_for_dataset(win, d)
    ref_cal = params.get("ref_calibration") if calibrate else None
    if not calibrate:
        params["ref_calibration"] = None
        params["ref_path"] = None
    d.apply_processing(**params)
    d.maps_calibrated = bool(
        calibrate and ref_cal is not None and getattr(ref_cal, "is_active", False)
    )
