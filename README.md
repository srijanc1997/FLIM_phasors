# FLIM Phasors

Interactive **phasor analysis and segmentation** for fluorescence lifetime imaging (FLIM), aimed at CAM and general TCSPC workflows.

Built on [phasorpy](https://github.com/biopaul/phasorpy) with a PySide6 + matplotlib desktop UI.

## Supported data

| Format | Extensions | Notes |
|--------|------------|--------|
| PicoQuant TCSPC | `.ptu` | Channel and frame selection via phasorpy |
| Imspector FLIM TIFF | `.tif`, `.tiff` | Multi-frame stacks are summed over `T` by default |

If TIFF files lack laser metadata, set **frequency (MHz)** and **harmonic** under **Calibration** before applying filters.

## Features

- **Sample & reference** — per-sample reference or one **shared reference** for all loaded datasets
- **Calibration** — reference phasor calibration; adjustable laser frequency and harmonic
- **Filtering** — photon threshold with optional **harmonic masking**; phasor median/Gaussian/pawFLIM; **TCSPC pre-filters** (signal median/Gaussian via phasorpy)
- **Segmentation** — **circular or elliptic** phasor cursors (`mask_from_*_cursor`); **GMM** via `phasor_cluster_gmm` with adjustable σ
- **Phasor tools** — click the plot for `phasor_to_lifetime_search` lifetime readout (activity log)
- **Image views** — masked photons, τφ, τmod, τ normal, and **τ search** phase/mod maps (phasorpy lifetime search)
- **Multi-sample mode** — load several files; assign **group names** (e.g. Tumor, Control); overlay table to pick which curves appear on a combined phasor plot
- **Export** — segmentation PNG, phasor PNG, cluster **CSV**; **Excel** with `openpyxl` (`pip install -e ".[excel]"`)

## Requirements

- Python **3.10+**
- Core dependencies are listed in `pyproject.toml` / `requirements.txt` (`numpy`, `matplotlib`, `PySide6`, `phasorpy`, `scikit-learn`)

## Install

```bash
git clone https://github.com/srijanc1997/FLIM_phasors.git
cd FLIM_phasors
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e .
```

Optional extras (pawFLIM filter, Excel export):

```bash
pip install -e ".[all]"
```

Alternatively, install dependencies from `requirements.txt` then the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run

Recommended:

```bash
python -m flim_phasors
```

Console script (after editable install):

```bash
flim-phasor-gui
```

Legacy launcher at the repo root (same application):

```bash
python flim_phasor_gui.py
```

## Quick workflow

1. **Choose sample** — load one or more `.ptu` / `.tif` files (Ctrl+click or Shift+click in the file dialog for batch load).
2. **Choose reference** (optional) — load a reference file; enable **Shared reference** to use one calibration for every sample in multi-image mode.
3. Set **Calibration** (frequency, harmonic, reference channel) and **Filters** (threshold, spatial filter).
4. Click **Apply** to compute phasors and lifetime maps.
5. Segment with **cursors** (add/move circles on the phasor plot) or **GMM**, then export results from the bottom of the control panel.

For comparing several acquisitions, turn on **multi-image mode**, load additional samples, and use the **overlay table** to choose which datasets appear on the multi-phasor plot.

## Project layout

```
FLIM_phasors/
├── src/flim_phasors/
│   ├── __init__.py
│   ├── __main__.py         # python -m flim_phasors
│   ├── app.py              # Qt/matplotlib bootstrap, main()
│   ├── constants.py        # file filters, plot/UI constants
│   ├── data.py             # PhasorData: load, calibrate, filter, lifetimes
│   ├── io.py               # load_flim_signal(), reference caches
│   ├── utils.py
│   ├── canvas/
│   │   ├── phasor.py       # interactive phasor plot + cursors
│   │   └── image.py        # lifetime / photon image panel
│   └── gui/
│       └── main_window.py  # main window and controls
├── pyproject.toml          # package metadata, optional deps, entry points
├── requirements.txt
└── flim_phasor_gui.py      # thin launcher → flim_phasors.app.main
```

## Development

Editable install is enough for day-to-day work:

```bash
pip install -e ".[all]"
python -m flim_phasors
```

The installable package lives under `src/flim_phasors/`. The root `flim_phasor_gui.py` only forwards to `flim_phasors.app.main` for backward compatibility.

## License

See [LICENSE](LICENSE).
