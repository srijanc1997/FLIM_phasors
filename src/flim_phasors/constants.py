"""Shared UI and plot constants."""

COMPARE_CMAPS = ("Blues", "Oranges", "Greens", "Purples", "Reds", "Greys")
COMPARE_SCATTER_MAX = 8000
COMPARE_STYLE_MAP = {
    "Full density (overlay)": "cloud",
    "Subsample scatter": "scatter",
    "Mean ± σ": "summary",
}

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
    "τφ phase (ns)",
    "τmod (ns)",
    "τ normal (ns)",
    "τ search phase (ns)",
    "τ search mod (ns)",
)

CURSOR_SHAPES = ("Circle", "Ellipse")

CATEGORICAL_NAMES = (
    "red", "blue", "green", "pink", "purple", "lime",
    "cyan", "orange", "brown", "indigo", "teal", "slate",
)

# --- unused (focused cleanup): uncomment if needed; see io.is_supported_flim_path ---
# SUPPORTED_EXTENSIONS = (".ptu", ".tif", ".tiff")

FLIM_FILE_FILTER = (
    "FLIM files (*.ptu *.tif *.tiff);;"
    "PicoQuant PTU (*.ptu);;"
    "Imspector TIFF (*.tif *.tiff);;"
    "All files (*.*)"
)
