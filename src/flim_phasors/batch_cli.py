"""Batch-process FLIM samples from the command line (no GUI).

Loads PicoQuant and Imspector histograms, applies reference calibration and
phasor filtering, and exports lifetime maps to an output folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from flim_phasors.calibration import compute_reference_phasor
from flim_phasors.data import PhasorData
from flim_phasors.export_bundle import export_sample_maps
from flim_phasors.io import is_supported_flim_path


def _collect_paths(folder: Path) -> list[str]:
    """Gather supported FLIM file paths from a directory.

    Batch mode handles histogram files only (``.ptu``/``.tif``); LIF phasor
    maps are GUI-only via :mod:`flim_phasors.lif_io`.

    Args:
        folder: Directory to scan for ``.ptu``, ``.tif``, and ``.tiff`` files.

    Returns:
        Sorted list of absolute paths that pass :func:`~flim_phasors.io.is_supported_flim_path`.
    """
    out = []
    for ext in (".ptu", ".tif", ".tiff"):
        out.extend(str(p) for p in sorted(folder.glob(f"*{ext}")))
    return [p for p in out if is_supported_flim_path(p)]


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
    ref_cal=None,
):
    """Load and process a single FLIM sample through the phasor pipeline.

    Mirrors the GUI's Apply step for a single sample: loads the raw
    histogram, sets acquisition parameters, and calls
    :meth:`~flim_phasors.data.PhasorData.apply_processing` to run
    calibration, spatial/signal filtering, and photon thresholding in one
    pass. Passing a precomputed ``ref_cal`` (as :func:`main` does for batch
    runs) avoids re-decoding the reference file once per sample.

    Args:
        path: Sample ``.ptu`` or ``.tif``/``.tiff`` path.
        ref_path: Reference file for G/S calibration (empty skips binding).
        ref_channel: Emission channel index on the reference stack.
        ref_lifetime: Known reference fluorophore lifetime in nanoseconds.
        harmonic: Phasor harmonic index (typically 1).
        frequency: Laser modulation frequency in MHz.
        filter_mode: Spatial or signal filter (``"median"``, ``"gaussian"``, etc.).
        min_photons: Minimum photon count per pixel for valid phasor pixels.
        ref_cal: Precomputed reference calibration, or ``None`` to derive from
            *ref_path* inside :meth:`~flim_phasors.data.PhasorData.apply_processing`.

    Returns:
        Processed :class:`~flim_phasors.data.PhasorData` with calibrated maps.
    """
    d = PhasorData()
    d.load_sample(path)
    d.harmonic = harmonic
    d.frequency = frequency
    d.channel = 0
    if ref_path:
        d.ref_path = ref_path
        d.ref_channel = ref_channel
    d.apply_processing(
        ref_calibration=ref_cal,
        ref_path=ref_path or None,
        ref_lifetime=ref_lifetime,  # ns; drives phasor_calibrate phase reference
        filter_mode=filter_mode,
        intensity_min=float(min_photons),
    )
    return d


def main(argv=None) -> int:
    """Run batch FLIM phasor processing from command-line arguments.

    Resolves the input (single file or directory of ``.ptu``/``.tif`` files),
    computes the reference calibration once and reuses it across all
    samples, then processes each sample with :func:`process_one` and writes
    its maps to a numbered subfolder under ``--output`` via
    :func:`~flim_phasors.export_bundle.export_sample_maps`. A
    ``batch_summary.json`` recording each sample's path and valid-pixel
    count is written at the end for quick review without opening the GUI.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        ``0`` on success, ``1`` if no input files or reference path is missing.
    """
    p = argparse.ArgumentParser(description="Batch FLIM phasor processing")
    p.add_argument("input", help="Sample file or folder of .ptu/.tif")
    p.add_argument("-o", "--output", required=True, help="Output folder")
    p.add_argument("-r", "--reference", help="Reference .ptu/.tif for calibration")
    p.add_argument("--ref-channel", type=int, default=0)
    p.add_argument("--ref-lifetime", type=float, default=4.0)
    p.add_argument("--harmonic", type=int, default=1)
    p.add_argument("--frequency", type=float, default=80.0)
    p.add_argument("--filter", default="median", choices=["median", "gaussian", "pawflim", "signal median", "signal gaussian"])
    p.add_argument("--min-photons", type=int, default=0)
    args = p.parse_args(argv)

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
        # One reference decode shared by all samples (same channel/harmonic as GUI).
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
        summary.append({"path": sp, "valid_pixels": int(d.valid_mask().sum()) if d.real_cal is not None else 0})

    (out / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done — {len(paths)} sample(s) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
