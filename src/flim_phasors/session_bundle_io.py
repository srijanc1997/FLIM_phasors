"""Save and load self-contained imaging session bundles.

A ``.flimsession`` zip stores processed phasor maps, calibration, cursors, UI
state, and optional overlay data without raw PTU histograms.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from flim_phasors import __version__
from flim_phasors.cursors_io import cursors_to_list
from flim_phasors.data import PhasorData
from flim_phasors.gui.processing import PROC_SETTING_KEYS
from flim_phasors.utils import effective_reference_path, sample_core_metadata

BUNDLE_FORMAT = "flim_phasors_session_bundle"
BUNDLE_VERSION = 1
BUNDLE_EXTENSION = ".flimsession"
MANIFEST_NAME = "manifest.json"

MAP_KEYS = (
    "real_cal",  # g after calibration/threshold
    "imag_cal",  # s after calibration/threshold
    "mean_raw",
    "mean_thr",  # NaN where masked; same scale as mean_raw (photon counts)
    "tau_phi",
    "tau_mod",
    "tau_normal",
)


def is_session_bundle(path: str | Path) -> bool:
    """Return whether a path looks like a session bundle file.

    The check is purely name-based (suffix comparison); it does not open the
    file or verify that it is a valid zip archive with a manifest. Callers
    that need certainty should attempt :func:`load_session_bundle` and handle
    the resulting ``ValueError`` instead.

    Args:
        path: File path to inspect.

    Returns:
        ``True`` when the suffix is ``.flimsession``.
    """
    p = Path(path)
    return p.suffix.lower() == BUNDLE_EXTENSION or p.name.lower().endswith(BUNDLE_EXTENSION)


def _json_default(obj):
    """JSON serializer hook for NumPy scalars and arrays.

    Passed as the ``default`` callback to :func:`json.dumps` when writing the
    bundle manifest, since the standard encoder does not know how to handle
    NumPy types that can appear in metadata dicts (e.g. ``np.float64`` means
    or ``np.ndarray`` harmonic tables). It only needs to cover types that
    slip through despite the explicit ``float``/``int``/``.tolist()`` casts
    used elsewhere in this module.

    Args:
        obj: Value to serialize.

    Returns:
        JSON-compatible Python scalar or nested list.

    Raises:
        TypeError: If ``obj`` is not a supported NumPy type.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _maps_from_dataset(d: PhasorData) -> dict[str, np.ndarray]:
    """Collect computed map arrays present on a dataset.

    Only attributes that are actually set (i.e. not ``None``) are included,
    so a dataset that has not been through calibration/thresholding yet
    contributes an empty dict. Arrays are cast to float64 for a stable
    on-disk representation; ``last_overlay``, when present, is additionally
    clipped to the ``[0, 1]`` range expected by the RGBA overlay renderer.
    The returned dict is written directly into a per-sample ``maps.npz``
    archive by :func:`save_session_bundle`.

    Args:
        d: Processed :class:`~flim_phasors.data.PhasorData` instance.

    Returns:
        Dict mapping :data:`MAP_KEYS` names to float64 arrays, plus optional
        ``last_overlay`` when segmentation was painted for this sample.
    """
    out: dict[str, np.ndarray] = {}
    for key in MAP_KEYS:
        arr = getattr(d, key, None)
        if arr is not None:
            out[key] = np.asarray(arr, dtype=np.float64)
    overlay = getattr(d, "last_overlay", None)
    if overlay is not None:
        out["last_overlay"] = np.clip(np.asarray(overlay, dtype=np.float64), 0, 1)
    return out


def _apply_maps_to_dataset(d: PhasorData, maps: dict[str, np.ndarray]) -> None:
    """Attach bundled map arrays onto a :class:`PhasorData` instance.

    This is the inverse of :func:`_maps_from_dataset`, used when rehydrating
    a dataset from a loaded bundle. Only keys present in ``maps`` are set, so
    a sample that was saved before it had e.g. a computed ``tau_normal`` map
    simply leaves that attribute unset. After attaching ``real_cal``, the
    dataset's internal ``_shape_hint`` is refreshed so downstream UI code can
    size widgets without needing the original raw histogram.

    Args:
        d: Dataset updated in place.
        maps: Dict of array names to NumPy arrays from an ``maps.npz`` archive.
    """
    for key in MAP_KEYS:
        if key in maps:
            setattr(d, key, np.asarray(maps[key], dtype=np.float64))
    if "last_overlay" in maps:
        d.last_overlay = np.asarray(maps["last_overlay"], dtype=np.float64)
    if d.real_cal is not None:
        d._shape_hint = tuple(int(x) for x in d.real_cal.shape)


def _sample_meta_row(win, d: PhasorData, index: int, maps_file: str) -> dict:
    """Build one manifest ``samples`` entry for a dataset."""
    ref = effective_reference_path(win, d)
    st = getattr(d, "_intensity_stats", {}) or {}
    stash = getattr(d, "processing_settings", None) or {}
    core = sample_core_metadata(d, index, reference_path=ref)
    return {
        "index": index,
        "label": core["label"],
        "original_sample_path": d.sample_path or "",
        "display_name": core["display_name"],
        "group": core["group"],
        "channel": int(core["channel"]),
        "n_channels": max(1, int(getattr(d, "n_channels", 1))),
        "frame_index": int(getattr(d, "frame_index", -1)),
        "frequency_MHz": float(core["frequency_MHz"]),
        "harmonic": int(core["harmonic"]),
        "work_frequency_MHz": float(core["work_frequency_MHz"]),
        "pixel_size_um": float(getattr(d, "pixel_size_um", 0.0) or 0.0),
        "reference_path": core["reference_path"],
        "reference_channel": int(d.ref_channel) if ref else 0,
        "reference_n_channels": max(1, int(getattr(d, "ref_n_channels", 1))),
        "processing_settings": {k: stash[k] for k in PROC_SETTING_KEYS if k in stash},
        "intensity_stats": dict(st),
        "maps_file": maps_file,
        "computed": core["computed"],
        "gmm_fit": _serialize_gmm_fit(getattr(d, "gmm_fit", None)),
        "cluster_stats": _serialize_cluster_stats(getattr(d, "cluster_stats", None) or []),
    }


def _serialize_gmm_fit(fit) -> dict | None:
    """Convert a GMM ellipse fit tuple to a JSON-friendly dict.

    The in-memory fit is a 5-tuple of parallel arrays (one entry per Gaussian
    cluster) as returned by :func:`~flim_phasors.analysis.fit_phasor_gmm`.
    JSON has no native array/tuple distinction, so each component is
    converted to a plain Python list via ``.tolist()`` for storage in the
    bundle manifest; angles remain in radians, matching the GMM fit
    convention used elsewhere in the app.

    Args:
        fit: Tuple of center, radii, and angle arrays, or ``None``.

    Returns:
        Serialized fit dict, or ``None`` when ``fit`` is ``None``.
    """
    if fit is None:
        return None
    cr, ci, rm, ri, ang = fit
    return {
        "center_real": np.asarray(cr, dtype=float).tolist(),
        "center_imag": np.asarray(ci, dtype=float).tolist(),
        "radius_major": np.asarray(rm, dtype=float).tolist(),
        "radius_minor": np.asarray(ri, dtype=float).tolist(),
        "angle": np.asarray(ang, dtype=float).tolist(),
    }


def _deserialize_gmm_fit(block: dict | None):
    """Restore a GMM ellipse fit tuple from manifest JSON.

    This is the inverse of :func:`_serialize_gmm_fit`: each JSON list is
    converted back to a float64 NumPy array so the tuple can be passed
    directly to plotting helpers (e.g. ``phasor.show_gmm_ellipses``) that
    expect the same array layout produced by the original GMM fit.

    Args:
        block: Serialized GMM dict from the manifest, or ``None``.

    Returns:
        Tuple of NumPy arrays ``(center_real, center_imag, radius_major,
        radius_minor, angle)``, or ``None``.
    """
    if not block:
        return None
    return (
        np.asarray(block["center_real"], dtype=float),
        np.asarray(block["center_imag"], dtype=float),
        np.asarray(block["radius_major"], dtype=float),
        np.asarray(block["radius_minor"], dtype=float),
        np.asarray(block["angle"], dtype=float),
    )


def _serialize_cluster_stats(stats: list[dict]) -> list[dict]:
    """Prepare cluster statistics rows for JSON export.

    Cluster stat rows carry an RGB ``color`` field used for table swatches
    and legend entries; NumPy or Qt color types are not directly
    JSON-serializable, so this makes a shallow copy of each row and coerces
    ``color`` (when present) to a plain 3-element list of floats. All other
    fields (label, mean lifetimes, pixel counts, etc.) are passed through
    unchanged.

    Args:
        stats: List of cluster stat dicts from the main window.

    Returns:
        Copy of each row with RGB ``color`` tuples converted to float lists.
    """
    rows = []
    for st in stats or []:
        row = dict(st)
        color = row.get("color")
        if color is not None:
            row["color"] = [float(x) for x in color[:3]]
        rows.append(row)
    return rows


def _deserialize_cluster_stats(rows: list[dict]) -> list[dict]:
    """Restore cluster statistics rows after loading a bundle.

    Inverse of :func:`_serialize_cluster_stats`: each row's ``color`` field,
    if it is a list of at least three numbers, is converted back to a tuple
    of floats matching the format the cluster/legend rendering code expects.
    Rows without a usable ``color`` (e.g. older bundles) are passed through
    unchanged rather than raising.

    Args:
        rows: Serialized cluster stat list from the manifest.

    Returns:
        Rows with ``color`` fields converted back to RGB tuples when present.
    """
    out = []
    for st in rows or []:
        row = dict(st)
        color = row.get("color")
        if isinstance(color, list) and len(color) >= 3:
            row["color"] = tuple(float(x) for x in color[:3])
        out.append(row)
    return out


def build_bundle_manifest(win) -> dict:
    """Assemble the full manifest dict for the current window state.

    Includes calibration, cursors, per-sample metadata, GMM/UI settings, and
    version stamps. Only samples with computed maps are listed.

    Args:
        win: Main window whose state is serialized.

    Returns:
        Manifest dictionary written to ``manifest.json`` inside the bundle.
    """
    datasets = [
        d for d in (win._all_datasets() if hasattr(win, "_all_datasets") else [win.data])
        if d.real_cal is not None  # raw histograms are not stored in the zip
    ]
    cursors = cursors_to_list(list(win.phasor.cursors)) if hasattr(win, "phasor") else []
    try:
        import phasorpy
        pp_ver = getattr(phasorpy, "__version__", "")
    except ImportError:
        pp_ver = ""

    ui: dict[str, Any] = {
        "multi_image": bool(getattr(win, "chk_multi", None) and win.chk_multi.isChecked()),
        "compare_overlay": bool(getattr(win, "chk_compare", None) and win.chk_compare.isChecked()),
        "overlay_checked": bool(getattr(win, "chk_overlay", None) and win.chk_overlay.isChecked()),
        "manual_pixel_um": float(win.sp_pixel_um.value()) if hasattr(win, "sp_pixel_um") else 0.0,
        "compare_style": (
            win.cb_compare_style.currentText()
            if hasattr(win, "cb_compare_style") else ""
        ),
        "compare_group_filter": (
            win.cb_compare_group.currentText()
            if hasattr(win, "cb_compare_group") else ""
        ),
        "legend_format": (
            win.cb_legend_format.currentText()
            if hasattr(win, "cb_legend_format") else ""
        ),
        "legend_loc": (
            win.cb_legend_loc.currentText()
            if hasattr(win, "cb_legend_loc") else ""
        ),
        "legend_size": (
            int(win.sp_legend_size.value())
            if hasattr(win, "sp_legend_size") else 11
        ),
        "gmm_covariance": win.cb_cov.currentText() if hasattr(win, "cb_cov") else "",
        "gmm_sigma": float(win._gmm_sigma()) if hasattr(win, "_gmm_sigma") else 2.0,
        "gmm_use_bic": bool(getattr(win, "chk_bic", None) and win.chk_bic.isChecked()),
        "gmm_n_comp": win.edit_ncomp.text().strip() if hasattr(win, "edit_ncomp") else "",
        "gmm_fit": _serialize_gmm_fit(getattr(win, "_gmm_fit", None)),
        "cluster_stats": _serialize_cluster_stats(getattr(win, "cluster_stats", [])),
        "compare_checked_indices": _compare_checked_indices(win),
    }

    active = getattr(win, "active_idx", -1)
    if active < 0 and datasets:
        active = 0

    return {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "app_version": __version__,
        "phasorpy_version": pp_ver,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "segmentation_mode": getattr(win, "mode", "cursor"),
        "shared_reference": bool(win.chk_shared_ref.isChecked()) if hasattr(win, "chk_shared_ref") else False,
        "shared_reference_path": getattr(win, "shared_ref_path", ""),
        "shared_reference_channel": int(getattr(win, "shared_ref_channel", 0)),
        "shared_reference_n_channels": max(1, int(getattr(win, "shared_ref_n_channels", 1))),
        "calibration": {
            "frequency_MHz": float(win.sp_freq.value()),
            "harmonic": int(win.sp_harm.value()),
            "reference_lifetime_ns": float(win.sp_reflt.value()),
            "filter": win.cb_filter.currentText() if hasattr(win, "cb_filter") else "median",
            "min_photons": int(win.sp_thr.value()),
            "harmonic_mask": bool(win.chk_detect_harm.isChecked()) if hasattr(win, "chk_detect_harm") else True,
            "reference_path": getattr(win, "shared_ref_path", "") or getattr(win.data, "ref_path", ""),
            "reference_channel": int(getattr(win, "shared_ref_channel", 0)),
            "mean_g": float(getattr(win.ref_calibration, "mean_g", 0.0)) if hasattr(win, "ref_calibration") else 0.0,
            "mean_s": float(getattr(win.ref_calibration, "mean_s", 0.0)) if hasattr(win, "ref_calibration") else 0.0,
            "harmonic_gs": (
                [[float(g), float(s)] for g, s in (win.ref_calibration.harmonic_gs or [])]
                if hasattr(win, "ref_calibration") and getattr(win.ref_calibration, "harmonic_gs", None)
                else None
            ),
            "manual": bool(getattr(win.ref_calibration, "use_manual", False)) if hasattr(win, "ref_calibration") else False,
            "manual_g": float(getattr(win.ref_calibration, "manual_g", 0.0)) if hasattr(win, "ref_calibration") else 0.0,
            "manual_s": float(getattr(win.ref_calibration, "manual_s", 0.0)) if hasattr(win, "ref_calibration") else 0.0,
        },
        "active_sample_index": active,
        "cursors": cursors,
        "ui": ui,
        "samples": [
            _sample_meta_row(win, d, i, f"samples/{i:03d}/maps.npz")
            for i, d in enumerate(datasets)
        ],
    }


def _compare_checked_indices(win) -> list[int]:
    """Read checked compare-table row indices from the main window.

    The compare table's checkbox column drives which datasets are drawn as
    overlay layers in compare mode; this reads the Qt check state of column 0
    for every row and returns the dataset index stored in that item's
    ``UserRole`` data, so the selection can be persisted in the bundle
    manifest and reapplied later by :func:`_restore_compare_checks`.

    Args:
        win: Main window with an optional ``table_compare`` widget.

    Returns:
        List of dataset indices currently checked for overlay comparison.
    """
    from PySide6.QtCore import Qt

    if not hasattr(win, "table_compare"):
        return []
    out = []
    for row in range(win.table_compare.rowCount()):
        it = win.table_compare.item(row, 0)
        if it is None:
            continue
        idx = int(it.data(Qt.ItemDataRole.UserRole))
        if it.checkState() == Qt.CheckState.Checked:
            out.append(idx)
    return out


def dataset_from_bundle_sample(meta: dict, maps: dict[str, np.ndarray]) -> PhasorData:
    """Reconstruct a :class:`PhasorData` from manifest metadata and map arrays.

    Because bundles intentionally omit raw TCSPC histograms, this rebuilds a
    dataset purely from already-computed maps and metadata rather than
    re-running acquisition/calibration. The resulting dataset is marked
    ``maps_calibrated = True`` so downstream code treats its ``real_cal``/
    ``imag_cal`` maps as post-processing output, not raw phasor coordinates
    requiring another calibration pass.

    Args:
        meta: One ``samples`` row from the bundle manifest.
        maps: Decompressed map arrays from the sample's ``maps.npz``.

    Returns:
        Fully populated dataset with maps attached (no raw histogram reload).
    """
    d = PhasorData()
    d.sample_path = str(meta.get("original_sample_path", "") or "")
    d.display_name = str(meta.get("display_name", "") or "").strip()
    d.group_name = str(meta.get("group", "") or "")
    d.channel = int(meta.get("channel", 0))
    d.n_channels = max(1, int(meta.get("n_channels", 1)))
    d.frame_index = int(meta.get("frame_index", -1))
    d.frequency = float(meta.get("frequency_MHz", 80.0))
    d.harmonic = int(meta.get("harmonic", 1))
    d.pixel_size_um = float(meta.get("pixel_size_um", 0.0) or 0.0)
    ref = meta.get("reference_path") or ""
    d.ref_path = str(ref)
    d.ref_channel = int(meta.get("reference_channel", 0))
    d.ref_n_channels = max(1, int(meta.get("reference_n_channels", 1)))
    d.processing_settings = dict(meta.get("processing_settings") or {})
    d._intensity_stats = dict(meta.get("intensity_stats") or {})
    d.gmm_fit = _deserialize_gmm_fit(meta.get("gmm_fit"))
    d.cluster_stats = _deserialize_cluster_stats(meta.get("cluster_stats") or [])
    _apply_maps_to_dataset(d, maps)
    # Bundles store post-Apply maps; no histogram to re-run calibration on load.
    d.maps_calibrated = bool(meta.get("maps_calibrated", True))
    return d


def save_session_bundle(win, path: str | Path) -> dict:
    """Write one ``.flimsession`` zip with manifest and per-sample map arrays.

    Only datasets that have been through Apply (i.e. have a non-``None``
    ``real_cal``) are included; raw histograms are never written, which keeps
    bundles small and avoids re-embedding potentially large PTU/LIF source
    data. Before serializing, this flushes the active window's in-progress UI
    edits (processing settings, segmentation) onto its dataset so the saved
    state matches what is currently displayed. Each sample's maps are stored
    as a compressed ``.npz`` under ``samples/{i:03d}/maps.npz``, and a single
    ``manifest.json`` ties everything together.

    Args:
        win: Main window whose computed datasets and UI state are saved.
        path: Destination path; ``.flimsession`` suffix is enforced.

    Returns:
        Summary dict with ``path``, ``n_samples``, and ``size_mb``.

    Raises:
        ValueError: When no computed samples are available to save.
    """
    path = Path(path)
    if path.suffix.lower() != BUNDLE_EXTENSION:
        path = path.with_suffix(BUNDLE_EXTENSION)

    datasets = [
        d for d in (win._all_datasets() if hasattr(win, "_all_datasets") else [win.data])
        if d.real_cal is not None  # raw histograms are not stored in the zip
    ]
    if not datasets:
        raise ValueError("No computed samples — run Apply on at least one image first.")

    if hasattr(win, "_save_proc_from_ui"):
        win._save_proc_from_ui(win.data)
    if hasattr(win, "_stash_segmentation_to_dataset"):
        win._stash_segmentation_to_dataset(win.data)
        # Also stash any other loaded samples that share window state only for active;
        # each dataset already holds its own gmm_fit/cluster_stats/overlay.

    manifest = build_bundle_manifest(win)
    overlay = getattr(win, "last_overlay", None)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, d in enumerate(datasets):
            maps = _maps_from_dataset(d)
            buf = io.BytesIO()
            np.savez_compressed(buf, **maps)
            zf.writestr(f"samples/{i:03d}/maps.npz", buf.getvalue())
        if overlay is not None:
            buf = io.BytesIO()
            np.savez_compressed(buf, overlay=np.clip(np.asarray(overlay, dtype=float), 0, 1))
            zf.writestr("overlay.npz", buf.getvalue())
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, default=_json_default))

    size_mb = path.stat().st_size / (1024 * 1024)
    return {
        "path": str(path),
        "n_samples": len(datasets),
        "size_mb": size_mb,
    }


def load_session_bundle(path: str | Path) -> dict:
    """Read a ``.flimsession`` zip archive.

    Validates the manifest format tag and version before touching any map
    data, then decompresses each sample's ``maps.npz`` and rebuilds a
    :class:`PhasorData` per sample via :func:`dataset_from_bundle_sample`.
    This function only parses the archive and returns in-memory objects; it
    does not mutate any GUI state — see :func:`apply_session_bundle_to_window`
    for wiring the result into a main window.

    Args:
        path: Path to an existing bundle file.

    Returns:
        Dict with keys ``manifest``, ``datasets``, ``overlay``, ``gmm_fit``, and
        ``cluster_stats``.

    Raises:
        ValueError: When the file is missing a manifest, has an unknown format,
            or references missing map archives.
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        if MANIFEST_NAME not in zf.namelist():
            raise ValueError(f"Not a session bundle (missing {MANIFEST_NAME}).")
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        if manifest.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"Unsupported bundle format: {manifest.get('format')!r}")
        version = int(manifest.get("format_version", 0))
        if version > BUNDLE_VERSION:
            raise ValueError(
                f"Bundle version {version} is newer than supported ({BUNDLE_VERSION})."
            )

        datasets: list[PhasorData] = []
        for row in manifest.get("samples", []):
            maps_file = row.get("maps_file", "")
            if maps_file not in zf.namelist():
                raise ValueError(f"Missing maps in bundle: {maps_file}")
            with zf.open(maps_file) as f:
                npz = np.load(f)
                maps = {k: npz[k] for k in npz.files}
            datasets.append(dataset_from_bundle_sample(row, maps))

        overlay = None
        if "overlay.npz" in zf.namelist():
            with zf.open("overlay.npz") as f:
                npz = np.load(f)
                if "overlay" in npz.files:
                    overlay = np.asarray(npz["overlay"], dtype=float)

    return {
        "manifest": manifest,
        "datasets": datasets,
        "overlay": overlay,
        "gmm_fit": _deserialize_gmm_fit((manifest.get("ui") or {}).get("gmm_fit")),
        "cluster_stats": _deserialize_cluster_stats((manifest.get("ui") or {}).get("cluster_stats")),
    }


def apply_session_bundle_to_window(win, loaded: dict) -> None:
    """Restore GUI state from :func:`load_session_bundle` output.

    Rebuilds datasets, calibration, cursors, GMM overlays, compare UI, and
    refreshes all dependent views on the main window.

    Args:
        win: Main window updated in place.
        loaded: Dict returned by :func:`load_session_bundle`.

    Raises:
        ValueError: When the bundle contains no samples.
    """
    from flim_phasors.session_io import apply_calibration_from_session, restore_cursors_to_phasor
    from flim_phasors.utils import categorical_rgb

    manifest = loaded["manifest"]
    datasets = loaded["datasets"]
    if not datasets:
        raise ValueError("Bundle contains no samples.")

    ui = manifest.get("ui") or {}
    active = int(manifest.get("active_sample_index", 0))
    active = max(0, min(active, len(datasets) - 1))
    multi = bool(ui.get("multi_image")) or len(datasets) > 1
    if multi:
        win.datasets = list(datasets)
        win.active_idx = active
        win.data = win.datasets[active]
    else:
        win.datasets = []
        win.active_idx = -1
        win.data = datasets[active]

    win.shared_ref_path = str(manifest.get("shared_reference_path", "") or "")
    win.shared_ref_channel = int(manifest.get("shared_reference_channel", 0))
    win.shared_ref_n_channels = max(1, int(manifest.get("shared_reference_n_channels", 1)))

    if hasattr(win, "chk_multi"):
        win.chk_multi.blockSignals(True)
        win.chk_multi.setChecked(bool(ui.get("multi_image")) or len(win.datasets) > 1)
        win.chk_multi.blockSignals(False)
        if hasattr(win, "_set_multi_detail_enabled"):
            win._set_multi_detail_enabled(win.chk_multi.isChecked())

    apply_calibration_from_session(win, manifest)

    mode = manifest.get("segmentation_mode", "cursor")
    if hasattr(win, "rb_cursor"):
        win.rb_cursor.blockSignals(True)
        win.rb_gmm.blockSignals(True)
        if mode == "gmm":
            win.rb_gmm.setChecked(True)
        else:
            win.rb_cursor.setChecked(True)
        win.rb_cursor.blockSignals(False)
        win.rb_gmm.blockSignals(False)
        if hasattr(win, "on_mode_change"):
            win.on_mode_change()

    if manifest.get("cursors"):
        restore_cursors_to_phasor(win, manifest["cursors"])

    gmm_fit = loaded.get("gmm_fit")
    active_d = win.data
    # Active sample's per-image GMM overrides manifest-level ui.gmm_fit.
    if getattr(active_d, "gmm_fit", None) is not None:
        gmm_fit = active_d.gmm_fit
    if gmm_fit is not None:
        win._gmm_fit = gmm_fit
        n = len(gmm_fit[0])
        colors = [categorical_rgb(k) for k in range(n)]
        win.phasor.show_gmm_ellipses(*gmm_fit, colors)

    if hasattr(win, "cb_cov") and ui.get("gmm_covariance"):
        win.cb_cov.setCurrentText(str(ui["gmm_covariance"]))
    if hasattr(win, "edit_gmm_sigma"):
        win.edit_gmm_sigma.setText(str(ui.get("gmm_sigma", 2.0)))
    if hasattr(win, "chk_bic"):
        win.chk_bic.setChecked(bool(ui.get("gmm_use_bic")))
    if hasattr(win, "edit_ncomp") and ui.get("gmm_n_comp"):
        win.edit_ncomp.setText(str(ui["gmm_n_comp"]))

    if hasattr(win, "sp_pixel_um"):
        win.sp_pixel_um.setValue(float(ui.get("manual_pixel_um", 0.0)))

    stored_stats = list(getattr(active_d, "cluster_stats", None) or [])
    win.cluster_stats = stored_stats if stored_stats else list(loaded.get("cluster_stats") or [])
    win.last_overlay = getattr(active_d, "last_overlay", None)
    if win.last_overlay is None:
        win.last_overlay = loaded.get("overlay")  # legacy root overlay.npz fallback

    if hasattr(win, "chk_compare"):
        win.chk_compare.blockSignals(True)
        win.chk_compare.setChecked(bool(ui.get("compare_overlay")))
        win.chk_compare.blockSignals(False)
    if hasattr(win, "cb_compare_style") and ui.get("compare_style"):
        win.cb_compare_style.setCurrentText(str(ui["compare_style"]))
    if hasattr(win, "cb_compare_group") and ui.get("compare_group_filter"):
        win.cb_compare_group.setCurrentText(str(ui["compare_group_filter"]))
    if hasattr(win, "cb_legend_format") and ui.get("legend_format"):
        win.cb_legend_format.setCurrentText(str(ui["legend_format"]))
    if hasattr(win, "cb_legend_loc") and ui.get("legend_loc"):
        win.cb_legend_loc.setCurrentText(str(ui["legend_loc"]))
    if hasattr(win, "sp_legend_size") and ui.get("legend_size") is not None:
        win.sp_legend_size.setValue(int(ui["legend_size"]))

    win._restore_ui_for_active()
    if hasattr(win, "_refresh_image_combo"):
        win._refresh_image_combo()
        _restore_compare_checks(win, ui.get("compare_checked_indices") or [])
    elif hasattr(win, "_refresh_compare_list"):
        win._refresh_compare_list()
    if hasattr(win, "_refresh_compare_group_filter"):
        win._refresh_compare_group_filter()
    if hasattr(win, "_update_apply_buttons"):
        win._update_apply_buttons()
    if hasattr(win, "_update_multi_strip"):
        win._update_multi_strip()
    if hasattr(win, "_refresh_image_combo"):
        win._refresh_image_combo()
    if hasattr(win, "_update_phasor_display"):
        win._update_phasor_display()
    if hasattr(win, "_fill_table"):
        win._fill_table()

    if hasattr(win, "chk_overlay"):
        win.chk_overlay.blockSignals(True)
        overlay_on = bool(ui.get("overlay_checked")) and win.last_overlay is not None
        win.chk_overlay.setChecked(overlay_on)
        win.chk_overlay.blockSignals(False)
    if hasattr(win, "refresh_image"):
        win.refresh_image()
    if hasattr(win, "_update_metadata_panel"):
        win._update_metadata_panel()
    if hasattr(win, "_mark_calibration_current"):
        win._mark_calibration_current()


def _restore_compare_checks(win, indices: list[int]) -> None:
    """Restore compare-table checkbox states from saved indices.

    Inverse of :func:`_compare_checked_indices`. Signals are blocked while
    updating check states so re-applying a saved selection does not trigger
    the table's ``itemChanged`` handlers (and the resulting redraw cascade)
    once per row; only rows whose column-0 item is user-checkable are
    touched, matching how the table is populated elsewhere.

    Args:
        win: Main window with a ``table_compare`` widget.
        indices: Dataset indices that should be checked.
    """
    from PySide6.QtCore import Qt

    if not hasattr(win, "table_compare"):
        return
    want = set(int(i) for i in indices)
    win.table_compare.blockSignals(True)
    for row in range(win.table_compare.rowCount()):
        it = win.table_compare.item(row, 0)
        if it is None or not (it.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            continue
        idx = int(it.data(Qt.ItemDataRole.UserRole))
        it.setCheckState(
            Qt.CheckState.Checked if idx in want else Qt.CheckState.Unchecked
        )
    win.table_compare.blockSignals(False)
