"""Save and load reference calibration to JSON (no full histogram)."""

from __future__ import annotations

import json
from pathlib import Path

from flim_phasors.calibration import ReferenceCalibration


def calibration_to_dict(cal: ReferenceCalibration, *, ui_extra: dict | None = None) -> dict:
    d = {
        "version": 1,
        "source_path": cal.source_path,
        "channel": cal.channel,
        "n_channels": cal.n_channels,
        "harmonic": cal.harmonic,
        "mean_g": cal.mean_g,
        "mean_s": cal.mean_s,
        "mean_intensity": cal.mean_intensity,
        "use_manual": cal.use_manual,
        "manual_g": cal.manual_g,
        "manual_s": cal.manual_s,
        "manual_mean": cal.manual_mean,
        "values_ready": cal.values_ready,
    }
    if ui_extra:
        d["ui"] = ui_extra
    return d


def calibration_from_dict(data: dict) -> ReferenceCalibration:
    cal = ReferenceCalibration(
        source_path=str(data.get("source_path", "")),
        channel=int(data.get("channel", 0)),
        n_channels=int(data.get("n_channels", 1)),
        harmonic=int(data.get("harmonic", 1)),
        mean_g=float(data.get("mean_g", 0.0)),
        mean_s=float(data.get("mean_s", 0.0)),
        mean_intensity=float(data.get("mean_intensity", 1.0)),
        use_manual=bool(data.get("use_manual", False)),
        manual_g=float(data.get("manual_g", 0.0)),
        manual_s=float(data.get("manual_s", 0.0)),
        manual_mean=float(data.get("manual_mean", 1.0)),
        values_ready=bool(
            data.get("values_ready", "mean_g" in data and "mean_s" in data)
        ),
    )
    return cal


def save_calibration(path: str | Path, cal: ReferenceCalibration, *, ui_extra: dict | None = None):
    Path(path).write_text(
        json.dumps(calibration_to_dict(cal, ui_extra=ui_extra), indent=2),
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> tuple[ReferenceCalibration, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ui = data.get("ui") or {}
    return calibration_from_dict(data), ui
