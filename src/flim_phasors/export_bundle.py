"""Export a complete FLIM phasor analysis folder for sharing and archiving.

Writes phasor plots, per-sample lifetime and photon maps, GMM cluster masks,
CSV/Excel tables (g, s, τ_φ, τ_m, τ_n, cluster areas), and ``session.json``
with calibration settings, cursor geometry, and sample metadata. Expects a
GUI ``MainWindow`` instance for live plot state and processing labels.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flim_phasors import __version__
from flim_phasors.utils import dataset_display_label, dataset_short_label


def _filter_for_export(win, d) -> str:
    """Resolve the human-readable filter label for one dataset in the export.

    Args:
        win: Main window with processing UI state.
        d: ``PhasorData`` instance for the sample.

    Returns:
        Filter name string (e.g. ``"median"``); falls back to combo box text.
    """
    try:
        from flim_phasors.gui.processing import filter_label_for_dataset
        return filter_label_for_dataset(win, d)
    except Exception:
        return getattr(win, "cb_filter", None) and win.cb_filter.currentText() or "median"


def _safe_name(text: str, max_len: int = 80) -> str:
    """Sanitize a string for use as a filesystem folder or file name.

    Args:
        text: Raw label or path fragment.
        max_len: Maximum returned length.

    Returns:
        Underscore-separated name safe for Windows paths.
    """
    text = (text or "sample").strip()
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:max_len] or "sample"


def _sample_folder_name(d, index: int) -> str:
    """Build a unique subfolder name for one exported sample.

    Args:
        d: ``PhasorData`` with optional ``group_name`` and path metadata.
        index: Zero-based sample index in the session.

    Returns:
        Sanitized folder name, optionally prefixed with group name.
    """
    base = dataset_short_label(d, index)
    group = (getattr(d, "group_name", "") or "").strip()
    if group:
        return _safe_name(f"{group}__{base}")
    return _safe_name(f"{index + 1:02d}_{base}")


def _write_map_png(path: Path, arr, *, cmap="viridis", vmin=None, vmax=None):
    """Write a false-color map PNG with robust percentile scaling.

    Args:
        path: Output ``.png`` path.
        arr: 2-D numeric array (NaN masked).
        cmap: Matplotlib colormap name.
        vmin: Optional lower display bound; 2nd percentile if None.
        vmax: Optional upper display bound; 98th percentile if None.

    Returns:
        False when no finite values exist; True after successful write.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(arr, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    if vmin is None or vmax is None:
        lo, hi = np.percentile(finite, [2, 98])
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1.0
    masked = np.ma.masked_invalid(data)
    plt.imsave(path, masked, cmap=cmap, vmin=vmin, vmax=vmax)
    return True


def _write_photon_png(path: Path, arr):
    """Write a grayscale photon-count map PNG with percentile contrast.

    Args:
        path: Output ``.png`` path.
        arr: 2-D photon or intensity array.

    Returns:
        False when no finite values exist; True after successful write.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(arr, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    masked = np.ma.masked_invalid(data)
    plt.imsave(path, masked, cmap="gray", vmin=lo, vmax=hi)
    return True


def cluster_rows_for_dataset(win, d, stats):
    """Build CSV row dicts for GMM or cursor cluster statistics on one sample.

    Each row records phasor center (g, s), apparent lifetimes, pixel count,
    area fraction, harmonic, reference path, and active filter settings.

    Args:
        win: Main window for reference path and filter resolution.
        d: ``PhasorData`` for the clustered sample.
        stats: Iterable of cluster stat dicts (``idx``, ``label``, ``g``, ``s``,
            ``tp``, ``tm``, ``tn``, ``n``, ``area``).

    Returns:
        List of row dictionaries keyed for ``clusters.csv`` columns.
    """
    rows = []
    ref = win._effective_ref_path(d) if hasattr(win, "_effective_ref_path") else d.ref_path
    ref_base = os.path.basename(ref) if ref else ""
    sample_base = os.path.basename(d.sample_path) if d.sample_path else ""
    group = (getattr(d, "group_name", "") or "").strip()
    for st in stats:
        rows.append({
            "sample": sample_base,
            "group": group,
            "cluster": st["idx"],
            "label": st["label"],
            "g": st["g"],
            "s": st["s"],
            "tau_phi_ns": st["tp"],
            "tau_mod_ns": st["tm"],
            "tau_normal_ns": st["tn"],
            "pixels": st["n"],
            "area_percent": st["area"],
            "frequency_MHz": d.work_frequency,
            "harmonic": d.harmonic,
            "sample_channel": d.channel,
            "ref_channel": d.ref_channel if d.ref_path else "",
            "filter": _filter_for_export(win, d),
            "reference": ref_base,
        })
    return rows


def sample_metadata_row(win, d, index: int) -> dict:
    """Build one row of sample-level metadata for summary CSV and session JSON.

    Args:
        win: Main window for shared reference and filter state.
        d: ``PhasorData`` instance.
        index: Zero-based sample index.

    Returns:
        Dict with paths, channels, frequencies, threshold stats, and flags.
    """
    ref = win._effective_ref_path(d) if hasattr(win, "_effective_ref_path") else d.ref_path
    st = getattr(d, "_intensity_stats", {}) or {}
    return {
        "index": index + 1,
        "label": dataset_display_label(d, index),
        "sample_path": d.sample_path,
        "display_name": (getattr(d, "display_name", "") or "").strip(),
        "group": (getattr(d, "group_name", "") or "").strip(),
        "channel": d.channel,
        "frequency_MHz": d.frequency,
        "harmonic": d.harmonic,
        "work_frequency_MHz": d.work_frequency,
        "reference_path": ref or "",
        "reference_channel": d.ref_channel if ref else "",
        "filter": _filter_for_export(win, d),
        "min_photons": st.get("threshold", 0),
        "pixels_total": st.get("n_pixels", ""),
        "pixels_masked": st.get("n_below", ""),
        "computed": d.real_cal is not None,
    }


def build_session_dict(win) -> dict:
    """Serialize GUI session state for ``session.json`` in the export bundle.

    Captures app and phasorpy versions, segmentation mode, shared reference
    settings, manual/scalar calibration g/s, per-sample metadata, active index,
    and phasor cursor geometry (circles and ellipses).

    Args:
        win: Main window instance (``MainWindow``).

    Returns:
        JSON-serializable dictionary describing the exported session.
    """
    datasets = win._all_datasets() if hasattr(win, "_all_datasets") else [win.data]
    cursors = []
    if hasattr(win, "phasor"):
        for c in win.phasor.cursors:
            cursors.append({
                "kind": c.get("kind", "circle"),
                "center_real": c["center_real"],
                "center_imag": c["center_imag"],
                "radius": c["radius"],
                "radius_minor": c.get("radius_minor"),
                "angle": c.get("angle", 0.0),
                "label": c.get("label", ""),
            })
    try:
        import phasorpy
        pp_ver = getattr(phasorpy, "__version__", "")
    except ImportError:
        pp_ver = ""
    return {
        "app_version": __version__,
        "phasorpy_version": pp_ver,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "segmentation_mode": getattr(win, "mode", ""),
        "shared_reference": bool(win.chk_shared_ref.isChecked()) if hasattr(win, "chk_shared_ref") else False,
        "shared_reference_path": getattr(win, "shared_ref_path", ""),
        "calibration": {
            "frequency_MHz": win.sp_freq.value(),
            "harmonic": win.sp_harm.value(),
            "reference_lifetime_ns": win.sp_reflt.value(),
            "filter": _filter_for_export(win, d),
            "min_photons": win.sp_thr.value(),
            "harmonic_mask": win.chk_detect_harm.isChecked() if hasattr(win, "chk_detect_harm") else True,
            "reference_path": getattr(win, "shared_ref_path", "") or getattr(win.data, "ref_path", ""),
            "reference_channel": getattr(win, "shared_ref_channel", 0),
            "mean_g": getattr(win.ref_calibration, "mean_g", 0.0) if hasattr(win, "ref_calibration") else 0.0,
            "mean_s": getattr(win.ref_calibration, "mean_s", 0.0) if hasattr(win, "ref_calibration") else 0.0,
            "manual": bool(getattr(win.ref_calibration, "use_manual", False)) if hasattr(win, "ref_calibration") else False,
            "manual_g": getattr(win.ref_calibration, "manual_g", 0.0) if hasattr(win, "ref_calibration") else 0.0,
            "manual_s": getattr(win.ref_calibration, "manual_s", 0.0) if hasattr(win, "ref_calibration") else 0.0,
        },
        "samples": [sample_metadata_row(win, d, i) for i, d in enumerate(datasets)],
        "active_sample_index": getattr(win, "active_idx", -1),
        "cursors": cursors,
    }


def write_clusters_csv(path: Path, rows: list[dict]):
    """Write cluster statistics rows to a UTF-8 CSV file.

    Args:
        path: Destination ``.csv`` path.
        rows: Non-empty list of homogeneous row dicts from ``cluster_rows_for_dataset``.
    """
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_samples_summary(path: Path, rows: list[dict]):
    """Write per-sample metadata rows to a UTF-8 CSV summary file.

    Args:
        path: Destination ``samples_summary.csv`` path.
        rows: Non-empty list of dicts from ``sample_metadata_row``.
    """
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_excel_bundle(path: Path, win, all_cluster_rows: list[dict], sample_rows: list[dict]):
    """Write a multi-sheet Excel workbook with summary, samples, and clusters.

    Cluster label cells are filled with GUI cluster colors when available.
    Requires ``openpyxl``; callers should catch ``ImportError``.

    Args:
        path: Destination ``.xlsx`` path.
        win: Main window for export metadata and color lookup.
        all_cluster_rows: Combined cluster CSV rows (may be empty).
        sample_rows: Per-sample metadata rows for the Samples sheet.
    """
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(["Parameter", "Value"])
    for c in summary[1]:
        c.font = Font(bold=True)
    meta = [
        ("Software", f"FLIM Phasors {__version__}"),
        ("Exported (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
        ("Samples", len(sample_rows)),
        ("Segmentation mode", getattr(win, "mode", "")),
        ("Filter", _filter_for_export(win, win.data)),
    ]
    for k, v in meta:
        summary.append([k, v])
    summary.column_dimensions["A"].width = 28
    summary.column_dimensions["B"].width = 48

    ws_samples = wb.create_sheet("Samples")
    if sample_rows:
        headers = list(sample_rows[0].keys())
        ws_samples.append(headers)
        for c in ws_samples[1]:
            c.font = Font(bold=True)
        for row in sample_rows:
            ws_samples.append([row[h] for h in headers])

    if all_cluster_rows:
        ws_cl = wb.create_sheet("Clusters")
        headers = list(all_cluster_rows[0].keys())
        ws_cl.append(headers)
        for c in ws_cl[1]:
            c.font = Font(bold=True)
        for row in all_cluster_rows:
            ws_cl.append([row[h] for h in headers])

    if all_cluster_rows and hasattr(win, "_rgb_hex"):
        for i, row in enumerate(all_cluster_rows, start=2):
            color = None
            for st in getattr(win, "cluster_stats", []):
                if st["idx"] == row.get("cluster") and st["label"] == row.get("label"):
                    color = st.get("color")
                    break
            if color is not None:
                col_idx = headers.index("label") + 1
                ws_cl.cell(row=i, column=col_idx).fill = PatternFill(
                    "solid", fgColor=win._rgb_hex(color))

    wb.save(path)


def export_sample_maps(sample_dir: Path, d):
    """Write standard lifetime and photon PNG maps for one ``PhasorData`` sample.

    Exports thresholded photons, τ_φ, τ_m, and τ_n when arrays are present.

    Args:
        sample_dir: Directory created under ``samples/`` in the bundle.
        d: Processed ``PhasorData`` with lifetime maps.

    Returns:
        List of basenames of maps successfully written.
    """
    maps = [
        ("photons.png", d.mean_thr if d.mean_thr is not None else d.mean_raw, "photon"),
        ("tau_phi_ns.png", d.tau_phi, "viridis"),
        ("tau_mod_ns.png", d.tau_mod, "viridis"),
        ("tau_normal_ns.png", d.tau_normal, "viridis"),
    ]
    written = []
    for name, arr, style in maps:
        if arr is None:
            continue
        path = sample_dir / name
        ok = _write_photon_png(path, arr) if style == "photon" else _write_map_png(path, arr, cmap=style)
        if ok:
            written.append(name)
    return written


def _export_gmm_masks(sample_dir: Path, d, gmm_fit, written: list[str]):
    """Write per-cluster GMM ellipse masks as grayscale PNG files.

    Uses elliptic regions from the active GMM fit intersected with the sample
    valid phasor mask.

    Args:
        sample_dir: Per-sample export subdirectory.
        d: ``PhasorData`` with calibrated g/s maps.
        gmm_fit: GMM parameter tuple from the GUI fit.
        written: Mutable list of output paths appended in place.
    """
    from flim_phasors.analysis import masks_from_gmm_ellipses

    if d.real_cal is None:
        return
    valid = d.valid_mask()
    masks = masks_from_gmm_ellipses(d.real_cal, d.imag_cal, gmm_fit, valid)
    import matplotlib.pyplot as plt

    for k in range(masks.shape[0]):
        path = sample_dir / f"gmm_mask_{k + 1:02d}.png"
        plt.imsave(path, masks[k].astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
        written.append(str(path))


def export_analysis_bundle(win, out_dir: str | Path) -> dict:
    """Write a complete FLIM phasor analysis export folder.

    Creates phasor figure exports (PNG/PDF/SVG), active segmentation overlay,
    per-sample map subfolders, CSV tables, optional Excel workbook, session
    JSON, and a README describing contents. Lazy-loads samples that have paths
    but no decoded histogram yet.

    Args:
        win: ``MainWindow`` with phasor canvas, cluster stats, and datasets.
        out_dir: Destination directory (created if missing).

    Returns:
        Log dict with ``directory``, ``files`` (paths written), and ``n_samples``.

    Raises:
        ValueError: When no loaded sample is available to export.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    datasets = win._all_datasets() if hasattr(win, "_all_datasets") else [win.data]
    datasets = [
        d for d in datasets
        if d.sample_path or d.signal_full is not None
    ]
    if not datasets:
        raise ValueError("No loaded sample to export.")
    for d in datasets:
        if d.signal_full is None and d.sample_path:
            d.ensure_loaded(frame=getattr(d, "frame_index", -1))

    # Phasor plot (current view)
    phasor_path = out / "phasor_plot.png"
    win.phasor.fig.savefig(phasor_path, dpi=200, bbox_inches="tight")
    written.append(str(phasor_path))
    for ext, fmt in (("phasor_plot.pdf", "pdf"), ("phasor_plot.svg", "svg")):
        try:
            p = out / ext
            win.phasor.fig.savefig(p, format=fmt, bbox_inches="tight")
            written.append(str(p))
        except Exception:
            pass

    # Active segmentation overlay
    if getattr(win, "last_overlay", None) is not None:
        import matplotlib.pyplot as plt

        seg_path = out / "segmentation_active.png"
        plt.imsave(seg_path, np.clip(np.asarray(win.last_overlay), 0, 1))
        written.append(str(seg_path))

    # Per-sample subfolders
    sample_rows = []
    active = win.data
    for i, d in enumerate(datasets):
        sample_rows.append(sample_metadata_row(win, d, i))
        if d.real_cal is None:
            continue
        sub = out / "samples" / _sample_folder_name(d, i)
        sub.mkdir(parents=True, exist_ok=True)
        maps = export_sample_maps(sub, d)
        written.extend(str(sub / m) for m in maps)
        if getattr(win, "mode", "") == "gmm" and hasattr(win, "_gmm_fit") and d is active:
            _export_gmm_masks(sub, d, win._gmm_fit, written)

    # Tables
    write_samples_summary(out / "samples_summary.csv", sample_rows)
    written.append(str(out / "samples_summary.csv"))

    all_cluster_rows = []
    active = win.data
    if getattr(win, "cluster_stats", None):
        all_cluster_rows.extend(cluster_rows_for_dataset(win, active, win.cluster_stats))

    if all_cluster_rows:
        write_clusters_csv(out / "clusters.csv", all_cluster_rows)
        written.append(str(out / "clusters.csv"))

    # Excel workbook (optional)
    xlsx_path = out / "analysis_results.xlsx"
    try:
        write_excel_bundle(xlsx_path, win, all_cluster_rows, sample_rows)
        written.append(str(xlsx_path))
    except ImportError:
        (out / "README_export.txt").write_text(
            "Install openpyxl for Excel export: pip install openpyxl\n",
            encoding="utf-8",
        )

    # Session JSON
    session_path = out / "session.json"
    session_path.write_text(
        json.dumps(build_session_dict(win), indent=2),
        encoding="utf-8",
    )
    written.append(str(session_path))

    readme = out / "README_export.txt"
    readme.write_text(
        f"FLIM Phasors export ({__version__})\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "Contents:\n"
        "  phasor_plot.png          — phasor plot as shown in the app\n"
        "  segmentation_active.png — segmentation overlay (active sample, if painted)\n"
        "  samples/               — per-sample lifetime and photon maps\n"
        "  samples_summary.csv    — metadata for all loaded samples\n"
        "  clusters.csv           — cluster stats (active sample, after Paint)\n"
        "  analysis_results.xlsx  — summary workbook (if openpyxl installed)\n"
        "  session.json           — settings, paths, cursor positions\n",
        encoding="utf-8",
    )
    written.append(str(readme))

    return {"directory": str(out), "files": written, "n_samples": len(datasets)}
