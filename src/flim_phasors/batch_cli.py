"""Batch-process FLIM samples from the command line (no GUI).

Loads PicoQuant and Imspector histograms, applies reference calibration and
phasor filtering, optionally fits GMM clusters, and exports via the same
bundle writer as the GUI.

Config mode (recommended for GMM)::

    flim-phasor-batch examples/headless_gmm.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from flim_phasors.analysis import apply_gmm_to_dataset
from flim_phasors.calibration import compute_reference_phasor
from flim_phasors.data import PhasorData
from flim_phasors.export_bundle import (
    export_one_sample,
    export_sample_maps,
    write_clusters_csv,
    write_excel_bundle,
    write_samples_summary,
    _session_excel_path,
)
from flim_phasors.io import is_supported_flim_path

# Keys accepted in a headless ``*.txt`` config (key=value per line).
_CONFIG_DEFAULTS = {
    "input_folder": "",
    "output_folder": "",
    "reference": "",
    "ref_channel": 0,
    "ref_lifetime": 4.0,
    "harmonic": 1,
    "frequency": 80.0,
    "channel": 0,
    "filter": "median",
    "median_size": 3,
    "median_repeat": 1,
    "paw_sigma": 2.0,
    "paw_levels": 1,
    "min_photons": 0,
    "detect_harmonics": True,
    "gmm": True,
    "gmm_clusters": 3,
    "gmm_sigma": 2.0,
    "gmm_covariance": "full",
    "gmm_bic": False,
}


def _collect_paths(folder: Path) -> list[str]:
    """Gather supported FLIM file paths from a directory (``.ptu``/``.tif``)."""
    out = []
    for ext in (".ptu", ".tif", ".tiff"):
        out.extend(str(p) for p in sorted(folder.glob(f"*{ext}")))
    return [p for p in out if is_supported_flim_path(p)]


def _parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def load_config(path: str | Path) -> dict:
    """Load ``key=value`` config; ``#`` starts a comment."""
    cfg = dict(_CONFIG_DEFAULTS)
    text = Path(path).read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid config line (expected key=value): {line}")
        key, _, val = line.partition("=")
        key = key.strip().lower()
        val = val.strip().strip('"').strip("'")
        if key not in cfg:
            raise ValueError(f"Unknown config key: {key}")
        default = _CONFIG_DEFAULTS[key]
        if isinstance(default, bool):
            cfg[key] = _parse_bool(val)
        elif isinstance(default, int) and not isinstance(default, bool):
            cfg[key] = int(float(val))
        elif isinstance(default, float):
            cfg[key] = float(val)
        else:
            cfg[key] = val
    return cfg


def process_one(
    path: str,
    *,
    ref_path: str,
    ref_channel: int,
    ref_lifetime: float,
    harmonic: int,
    frequency: float,
    filter_mode: str,
    min_photons: int,
    channel: int = 0,
    median_size: int = 3,
    median_repeat: int = 1,
    paw_sigma: float = 2.0,
    paw_levels: int = 1,
    detect_harmonics: bool = True,
    ref_cal=None,
):
    """Load and process one FLIM sample (same Apply path as the GUI)."""
    d = PhasorData()
    d.load_sample(path, load_channel=int(channel))
    d.harmonic = harmonic
    d.frequency = frequency
    d.channel = int(channel)
    if ref_path:
        d.ref_path = ref_path
        d.ref_channel = ref_channel
    d.apply_processing(
        ref_calibration=ref_cal,
        ref_path=ref_path or None,
        ref_lifetime=ref_lifetime,
        filter_mode=filter_mode,
        median_size=int(median_size),
        median_repeat=int(median_repeat),
        paw_sigma=float(paw_sigma),
        paw_levels=int(paw_levels),
        intensity_min=float(min_photons),
        detect_harmonics=bool(detect_harmonics),
    )
    d.processing_settings = {
        "filter_mode": filter_mode,
        "median_size": int(median_size),
        "median_repeat": int(median_repeat),
        "paw_sigma": float(paw_sigma),
        "paw_levels": int(paw_levels),
        "intensity_min": float(min_photons),
        "detect_harmonics": bool(detect_harmonics),
        "ref_lifetime": float(ref_lifetime),
        "harmonic": int(harmonic),
        "frequency": float(frequency),
        "channel": int(channel),
    }
    return d


def _headless_win(datasets: list[PhasorData], cfg: dict, ref_cal):
    """Minimal stand-in so :func:`export_analysis_bundle` can run without Qt UI."""
    active = datasets[0]
    cal = ref_cal or SimpleNamespace(
        mean_g=0.0, mean_s=0.0, use_manual=False, manual_g=0.0, manual_s=0.0,
        harmonic_gs=None, is_active=False,
    )
    win = SimpleNamespace(
        data=active,
        datasets=datasets,
        active_idx=0,
        mode="gmm" if cfg.get("gmm") else "cursor",
        cluster_stats=list(getattr(active, "cluster_stats", None) or []),
        _gmm_fit=getattr(active, "gmm_fit", None),
        last_overlay=getattr(active, "last_overlay", None),
        phasor=SimpleNamespace(cursors=[], fig=None),
        ref_calibration=cal,
        shared_ref_path=cfg.get("reference") or "",
        shared_ref_channel=int(cfg.get("ref_channel", 0)),
        sp_freq=SimpleNamespace(value=lambda: float(cfg["frequency"])),
        sp_harm=SimpleNamespace(value=lambda: int(cfg["harmonic"])),
        sp_reflt=SimpleNamespace(value=lambda: float(cfg["ref_lifetime"])),
        sp_thr=SimpleNamespace(value=lambda: int(cfg["min_photons"])),
        cb_filter=SimpleNamespace(currentText=lambda: str(cfg["filter"])),
        chk_detect_harm=SimpleNamespace(isChecked=lambda: bool(cfg["detect_harmonics"])),
        chk_shared_ref=SimpleNamespace(isChecked=lambda: bool(cfg.get("reference"))),
    )
    win._all_datasets = lambda: datasets
    win._effective_ref_path = lambda d=None: cfg.get("reference") or ""
    win._rgb_hex = lambda rgb: "".join(
        f"{int(max(0.0, min(1.0, float(c))) * 255):02X}" for c in list(rgb)[:3]
    )
    return win


def run_from_config(cfg: dict) -> int:
    """Process FLIM files one-at-a-time (low RAM): load → GMM → export → free."""
    import gc
    from datetime import datetime, timezone

    from flim_phasors import __version__

    inp = Path(cfg["input_folder"])
    out = Path(cfg["output_folder"])
    if not inp.is_dir():
        print(f"input_folder not found: {inp}", file=sys.stderr)
        return 1
    if not cfg["output_folder"]:
        print("output_folder is required", file=sys.stderr)
        return 1

    paths = _collect_paths(inp)
    if not paths:
        print(f"No supported FLIM files in {inp}", file=sys.stderr)
        return 1

    filt = str(cfg["filter"])
    print("=== Headless FLIM Phasors (streaming, one sample in RAM) ===")
    print(f"  samples : {len(paths)} file(s) in {inp}")
    print(f"  output  : {out}")
    print(f"  sample channel={cfg['channel']}  harmonic={cfg['harmonic']}  "
          f"freq={cfg['frequency']} MHz")
    print(f"  filter  : {filt}", end="")
    if filt == "pawflim":
        print(f"  (paw_sigma={cfg['paw_sigma']}, paw_levels={cfg['paw_levels']})")
    elif filt in ("median", "gaussian", "signal median", "signal gaussian"):
        print(f"  (size={cfg['median_size']}, repeat={cfg['median_repeat']})")
    else:
        print()
    print(f"  threshold: min_photons={cfg['min_photons']}  "
          f"detect_harmonics={cfg['detect_harmonics']}")
    if cfg.get("gmm"):
        print(f"  GMM     : clusters={cfg['gmm_clusters']}  sigma={cfg['gmm_sigma']}  "
              f"cov={cfg['gmm_covariance']}  bic={cfg['gmm_bic']}")
    else:
        print("  GMM     : off (maps only)")
    print()

    ref_path = cfg["reference"] or ""
    ref_cal = None
    if ref_path:
        if not os.path.isfile(ref_path):
            print(f"Reference not found: {ref_path}", file=sys.stderr)
            return 1
        print(f"[ref] Decoding shared reference  {os.path.basename(ref_path)}  "
              f"(ch={cfg['ref_channel']}, lifetime={cfg['ref_lifetime']} ns)")
        ref_cal = compute_reference_phasor(
            ref_path, int(cfg["ref_channel"]), int(cfg["harmonic"]))
        print("[ref] Calibration ready (reused for all samples)")
    else:
        print("[ref] No reference — skipping calibration")

    out.mkdir(parents=True, exist_ok=True)
    samples_root = out / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict] = []
    all_cluster_rows: list[dict] = []
    n_files_written = 0

    for i, sp in enumerate(paths):
        tag = f"[{i + 1}/{len(paths)}]"
        print(f"{tag} Load sample     {os.path.basename(sp)}")
        print(f"{tag} Apply           calibrate + filter={filt} + threshold")
        d = process_one(
            sp,
            ref_path=ref_path,
            ref_channel=int(cfg["ref_channel"]),
            ref_lifetime=float(cfg["ref_lifetime"]),
            harmonic=int(cfg["harmonic"]),
            frequency=float(cfg["frequency"]),
            filter_mode=filt,
            min_photons=int(cfg["min_photons"]),
            channel=int(cfg["channel"]),
            median_size=int(cfg["median_size"]),
            median_repeat=int(cfg["median_repeat"]),
            paw_sigma=float(cfg["paw_sigma"]),
            paw_levels=int(cfg["paw_levels"]),
            detect_harmonics=bool(cfg["detect_harmonics"]),
            ref_cal=ref_cal,
        )
        n_valid = int(d.valid_mask().sum()) if d.real_cal is not None else 0
        print(f"{tag} Phasor ready    {n_valid} valid pixels")
        if cfg.get("gmm"):
            print(f"{tag} Fit GMM         …")
            _fit, stats = apply_gmm_to_dataset(
                d,
                clusters=int(cfg["gmm_clusters"]),
                sigma=float(cfg["gmm_sigma"]),
                covariance_type=str(cfg["gmm_covariance"]),
                use_bic=bool(cfg["gmm_bic"]),
            )
            print(f"{tag} GMM done        {len(stats)} cluster(s)")

        print(f"{tag} Export          → samples/…")
        win = _headless_win([d], cfg, ref_cal)
        one = export_one_sample(win, out, d, i)
        sample_rows.append(one["sample_row"])
        all_cluster_rows.extend(one["cluster_rows"])
        n_files_written += len(one["written"])
        print(f"{tag} Wrote           {one['folder']}")

        # Shared tables grow after each sample (Excel rewritten; small vs map RAM)
        write_samples_summary(out / "samples_summary.csv", sample_rows)
        if all_cluster_rows:
            write_clusters_csv(out / "clusters.csv", all_cluster_rows)
        try:
            write_excel_bundle(_session_excel_path(out), win, all_cluster_rows, sample_rows)
        except ImportError:
            pass

        # Drop large arrays before next sample
        del d, win, one
        gc.collect()
        print(f"{tag} Freed           sample memory")

    # Lightweight session.json (paths/settings only; maps already on disk)
    session = {
        "app_version": __version__,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "segmentation_mode": "gmm" if cfg.get("gmm") else "",
        "shared_reference": bool(ref_path),
        "shared_reference_path": ref_path,
        "headless_streaming": True,
        "calibration": {
            "frequency_MHz": float(cfg["frequency"]),
            "harmonic": int(cfg["harmonic"]),
            "reference_lifetime_ns": float(cfg["ref_lifetime"]),
            "filter": str(cfg["filter"]),
            "min_photons": int(cfg["min_photons"]),
            "harmonic_mask": bool(cfg["detect_harmonics"]),
            "reference_path": ref_path,
            "reference_channel": int(cfg["ref_channel"]),
        },
        "samples": sample_rows,
        "cursors": [],
        "gmm": bool(cfg.get("gmm")),
        "gmm_clusters": int(cfg["gmm_clusters"]),
        "gmm_sigma": float(cfg["gmm_sigma"]),
        "gmm_covariance": str(cfg["gmm_covariance"]),
        "gmm_bic": bool(cfg["gmm_bic"]),
    }
    (out / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    print()
    print(f"Done — {len(paths)} sample(s), ~{n_files_written} sample files → {out}")
    print("  (samples_summary.csv / clusters.csv / analysis.xlsx updated after each sample)")
    return 0


def main(argv=None) -> int:
    """CLI: config ``.txt`` (GMM + full export) or legacy folder/file args."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 1 and Path(argv[0]).is_file() and Path(argv[0]).suffix.lower() in (
        ".txt", ".cfg", ".ini",
    ):
        return run_from_config(load_config(argv[0]))

    p = argparse.ArgumentParser(
        description="Batch FLIM phasor processing (pass a .txt config for headless GMM)")
    p.add_argument("input", help="Sample file/folder, or a headless config .txt")
    p.add_argument("-o", "--output", help="Output folder (legacy mode)")
    p.add_argument("-r", "--reference", help="Reference .ptu/.tif for calibration")
    p.add_argument("--ref-channel", type=int, default=0)
    p.add_argument("--ref-lifetime", type=float, default=4.0)
    p.add_argument("--harmonic", type=int, default=1)
    p.add_argument("--frequency", type=float, default=80.0)
    p.add_argument(
        "--filter", default="median",
        choices=["median", "gaussian", "pawflim", "signal median", "signal gaussian", "none"],
    )
    p.add_argument("--min-photons", type=int, default=0)
    args = p.parse_args(argv)

    # Config path passed with other flags still works.
    if Path(args.input).is_file() and Path(args.input).suffix.lower() in (".txt", ".cfg", ".ini"):
        return run_from_config(load_config(args.input))

    if not args.output:
        print("Legacy mode requires -o/--output (or use a .txt config).", file=sys.stderr)
        return 1

    inp = Path(args.input)
    paths = _collect_paths(inp) if inp.is_dir() else [str(inp)]
    paths = [p for p in paths if is_supported_flim_path(p)]
    if not paths:
        print("No supported FLIM files found.", file=sys.stderr)
        return 1

    ref_cal = None
    if args.reference:
        if not os.path.isfile(args.reference):
            print(f"Reference not found: {args.reference}", file=sys.stderr)
            return 1
        ref_cal = compute_reference_phasor(args.reference, args.ref_channel, args.harmonic)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, sp in enumerate(paths):
        print(f"[{i + 1}/{len(paths)}] {os.path.basename(sp)}")
        d = process_one(
            sp,
            ref_path=args.reference or "",
            ref_channel=args.ref_channel,
            ref_lifetime=args.ref_lifetime,
            harmonic=args.harmonic,
            frequency=args.frequency,
            filter_mode=args.filter,
            min_photons=args.min_photons,
            ref_cal=ref_cal,
        )
        sub = out / f"{i + 1:02d}_{Path(sp).stem}"
        sub.mkdir(parents=True, exist_ok=True)
        export_sample_maps(sub, d)
        summary.append({
            "path": sp,
            "valid_pixels": int(d.valid_mask().sum()) if d.real_cal is not None else 0,
        })

    (out / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done — {len(paths)} sample(s) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
