"""Save and load phasor cursor definitions as JSON."""

from __future__ import annotations

import json
from pathlib import Path


def cursors_to_list(cursors: list[dict]) -> list[dict]:
    out = []
    for c in cursors:
        out.append({
            "kind": c.get("kind", "circle"),
            "center_real": float(c["center_real"]),
            "center_imag": float(c["center_imag"]),
            "radius": float(c["radius"]),
            "radius_minor": c.get("radius_minor"),
            "angle": float(c.get("angle", 0.0)),
            "label": c.get("label", ""),
            "color": list(c.get("color", (0.5, 0.5, 0.5))),
        })
    return out


def save_cursors(path: str | Path, cursors: list[dict], *, sample_path: str = ""):
    payload = {"version": 1, "sample_path": sample_path, "cursors": cursors_to_list(cursors)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cursors(path: str | Path) -> tuple[list[dict], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cursors = []
    for c in data.get("cursors", []):
        color = tuple(c.get("color", (0.5, 0.5, 0.5))[:3])
        entry = {
            "kind": c.get("kind", "circle"),
            "center_real": float(c["center_real"]),
            "center_imag": float(c["center_imag"]),
            "radius": float(c["radius"]),
            "angle": float(c.get("angle", 0.0)),
            "label": c.get("label", ""),
            "color": color,
        }
        if c.get("radius_minor") is not None:
            entry["radius_minor"] = float(c["radius_minor"])
        cursors.append(entry)
    return cursors, str(data.get("sample_path", ""))
