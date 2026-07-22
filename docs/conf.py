"""Sphinx configuration for the flim-phasors API reference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Import version from the package so docs stay in sync with pyproject.toml.
from flim_phasors import __version__ as _pkg_version  # noqa: E402

# -- Project information -----------------------------------------------------

project = "flim-phasors"
copyright = "2026, FLIM Phasors contributors"
author = "FLIM Phasors contributors"
release = _pkg_version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Prefer reading source in the documented order.
add_module_names = False
toc_object_entries_show_parents = "hide"

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "show-inheritance": True,
    "undoc-members": False,
    "inherited-members": False,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
# Keep imports from pulling a real display while docs build.
autodoc_mock_imports = []

autosummary_generate = True

# -- Napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_attr_annotations = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"flim-phasors {release} API"
html_short_title = "flim-phasors"
html_show_sourcelink = True
html_copy_source = False

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
}

# Set in CI for correct links on GitHub Pages (project site subpath).
html_baseurl = os.environ.get("SPHINX_HTML_BASE_URL", "")
