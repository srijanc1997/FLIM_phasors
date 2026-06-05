# FLIM Phasors

Interactive **phasor analysis and segmentation** for fluorescence lifetime imaging (FLIM). Works with general TCSPC data from PicoQuant and Imspector-style TIFF stacks.

Built on [phasorpy](https://github.com/phasorpy/phasorpy) with a PySide6 + matplotlib desktop UI.

## Supported data

| Format | Extensions | Notes |
|--------|------------|--------|
| PicoQuant TCSPC | `.ptu` | Multi-channel; optional frame index |
| Imspector FLIM TIFF | `.tif`, `.tiff` | Multi-frame stacks can be summed or indexed |

If TIFF files lack laser metadata, set **frequency (MHz)** and **harmonic** under **Calibration** before **Apply**.

## Features

- **Sample & reference** — load multiple files at once; per-sample or **shared reference** calibration
- **Calibration** — reference phasor (maps only in RAM, not full histogram); manual g/s; save/load `calibration.json`; ref preview plot
- **Calibrate** then **Apply** — pick reference file, calibrate g/s, then preprocess samples
- **Filtering** — photon threshold; phasor and TCSPC spatial filters; pawFLIM optional
- **Segmentation** — circular or elliptic phasor cursors (undo, save/load JSON); GMM via `phasor_cluster_gmm`
- **Multi-sample** — **Multi-phasor** tab: sample table, groups, phasor overlay; **Setup** tab for filters; **Apply selected** / **Apply settings to all**
- **Image views** — photons (log scale, auto contrast), τ maps, scale bar (µm/px)
- **Phasor ↔ image** — click phasor to highlight nearest pixel on the image
- **Export all** — PNG/PDF/SVG phasor plot, per-sample maps, GMM masks, CSV, Excel, `session.json`
- **Save session** — one `.flimsession` bundle (processed maps + calibration + cursors; no PTU/TIF data)
- **Open session** — load `.flimsession` (standalone) or exported `session.json` (needs original files)
- **Batch CLI** — `flim-phasor-batch` for folder in → folder out
- **Cancel** long decode/processing jobs from the progress dialog

## Calibration (quick guide)

1. Load **Reference…** (path only) or **Load cal…** (reuse saved g/s — no `.ptu` decode).
2. Click **Calibrate** once to decode the reference and store **g / s** (scalar values only; the reference histogram is not kept in RAM).
3. Use **Manual ref phasor** to type g/s directly instead of a reference file.
4. Orange status text means harmonic/filter/channel changed — click **Calibrate** again, then **Apply**.
5. **Apply** uses the stored g/s values only; it does not reload the reference file.
6. **Save cal…** writes a small JSON; reference `.ptu` is not stored inside it.

## Requirements

- Python **3.10+**
- See `pyproject.toml` / `requirements.txt`

## Install

```bash
git clone https://github.com/srijanc1997/FLIM_phasors.git
cd FLIM_phasors
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[all]"
```

## Run

```bash
python -m flim_phasors
```

Or: `python flim_phasor_gui.py` / `flim-phasor-gui` after install.

## Batch processing (no GUI - in development)

```bash
flim-phasor-batch path/to/samples/ -o path/to/output -r reference.ptu --harmonic 1 --min-photons 10
```

## Quick workflow

1. **Sample…** — select one or more `.ptu` / `.tif` files (Ctrl/Shift+click for batch). Drag-and-drop onto the window also works.
2. **Reference…** (optional) — calibration file; **Shared ref** applies to all samples in multi-image mode.
3. **Reference…** — choose the calibration file (decoded on **Calibrate**, not on load).
4. **Calibrate** — compute reference g/s (check preview).
5. **Calibration** — frequency, harmonic, filters, **Frame** (if the stack has time).
6. **Apply** — preprocess on the **Setup** tab. With several images, use the **Multi-phasor** tab to pick samples, then **Apply selected** or **Apply settings to all**.
7. **Analyze** tab — segment with cursors or GMM (**Paint**), then **Export all…** or **File → Save session…** (`.flimsession` archive).

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| Ctrl+O | Sample… |
| Ctrl+R | Reference… |
| Ctrl+E | Export all… |
| Ctrl+Shift+O | Open session… |
| Ctrl+Shift+S | Save session… |
| F5 | Apply |
| Delete | Remove cursor |

## Export folder layout

```
export_folder/
  README_export.txt
  phasor_plot.png
  phasor_plot.pdf
  phasor_plot.svg
  segmentation_active.png    # if Paint was run on active sample
  samples_summary.csv
  clusters.csv                 # active sample clusters (after Paint)
  analysis_results.xlsx        # if openpyxl installed
  session.json                 # settings, paths, cursor positions, phasorpy version
  samples/
    01_myfile__groupA/
      photons.png
      tau_phi_ns.png
      gmm_mask_01.png          # if GMM was fit on active sample
      ...
```

## Session bundle (`.flimsession`)

**File → Save session…** writes one zip archive with processed phasor maps for every **Apply**-d sample (no raw PTU/TIF histograms). Use **File → Open session…** to restore segmentation, overlays, and tables on another machine without the original data files.

Typical size: ~1–3 MB per 256×256 image (compressed), vs tens–hundreds of MB per PTU in RAM.

```
my_experiment.flimsession   # zip
  manifest.json             # calibration, cursors, per-sample settings, UI state
  samples/000/maps.npz        # real/imag, photons, τ maps (float64, compressed)
  samples/001/maps.npz
  overlay.npz               # optional painted segmentation (active sample)
```

Re-opening a bundle does **not** let you change filters and re-**Apply** — that still needs the original files.

## Project layout

```
FLIM_phasors/
├── src/flim_phasors/
│   ├── app.py
│   ├── batch_cli.py
│   ├── busy.py
│   ├── calibration.py
│   ├── calibration_io.py
│   ├── cursors_io.py
│   ├── session_io.py
│   ├── session_bundle_io.py
│   ├── memory_est.py
│   ├── data.py
│   ├── io.py
│   ├── analysis.py
│   ├── export_bundle.py
│   ├── canvas/
│   └── gui/
│       ├── main_window.py
│       ├── enhancements.py
│       └── processing.py
├── pyproject.toml
└── flim_phasor_gui.py
```

## Development

```bash
pip install -e ".[all,dev]"
pytest
```

## License

See [LICENSE](LICENSE).
