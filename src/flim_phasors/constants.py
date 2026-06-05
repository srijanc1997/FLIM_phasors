"""Shared UI, plot, and file-dialog constants for FLIM phasor analysis.

Centralizes default choices for compare-plot colormaps, legend layout, spatial
filter modes, lifetime image views, phasor cursor shapes, categorical cluster
colors, and supported FLIM file-type filters used across the GUI and I/O layers.
"""

COMPARE_CMAPS = ("Blues", "Oranges", "Greens", "Purples", "Reds", "Greys")
COMPARE_SCATTER_MAX = 8000
COMPARE_STYLE_MAP = {
    "Full density (overlay)": "cloud",
    "Subsample scatter": "scatter",
    "Mean ± σ": "summary",
}

LEGEND_FORMAT_ITEMS = (
    "Sample name",
    "Group · sample",
)

LEGEND_LOC_ITEMS = (
    "upper right",
    "upper left",
    "lower right",
    "lower left",
    "best",
)

LEGEND_SIZE_DEFAULT = 11
LEGEND_SIZE_MIN = 6
LEGEND_SIZE_MAX = 22

FILTER_MODES = (
    "none",
    "median",
    "gaussian",
    "pawflim",
    "signal median",
    "signal gaussian",
)

IMAGE_VIEW_ITEMS = (
    "Photons (masked)",
    "Photons (raw)",
    "τφ phase (ns)",
    "τmod (ns)",
    "τ normal (ns)",
)

CURSOR_SHAPES = ("Circle", "Ellipse")

CATEGORICAL_NAMES = (
    "red", "blue", "green", "pink", "purple", "lime",
    "cyan", "orange", "brown", "indigo", "teal", "slate",
)

# --- unused (focused cleanup): uncomment if needed; see io.is_supported_flim_path ---
# SUPPORTED_EXTENSIONS = (".ptu", ".tif", ".tiff")

FLIM_FILE_FILTER = (
    "FLIM files (*.ptu *.tif *.tiff *.lif *.xlef);;"
    "PicoQuant PTU (*.ptu);;"
    "Imspector TIFF (*.tif *.tiff);;"
    "Leica LIF (*.lif *.xlef);;"
    "All files (*.*)"
)
