"""Phasor Lab UI themes (dark default + light variant).

Defines the two named themes used throughout the FLIM Phasors GUI — the dark
``phasor_lab`` theme (default) and the light ``phasor_lab_light`` variant —
along with helper functions that resolve a theme id (tolerant of legacy
boolean-era settings) to the concrete Qt stylesheets, activity-log style,
matplotlib toolbar style, and raw toolbar palette colors needed to fully
restyle the main window. ``EnhancementsMixin`` in ``enhancements.py`` is the
sole consumer of these helpers when applying or switching themes.
"""

from __future__ import annotations

THEME_PHASOR_LAB = "phasor_lab"
THEME_PHASOR_LAB_LIGHT = "phasor_lab_light"
DEFAULT_THEME = THEME_PHASOR_LAB

THEME_MENU_LABELS = {
    THEME_PHASOR_LAB: "Phasor Lab",
    THEME_PHASOR_LAB_LIGHT: "Phasor Lab Light",
}

PRIMARY_BUTTON_ATTRS = (
    # Widgets tagged with setProperty("primary", True) — styled via QPushButton[primary="true"] below.
    "btn_calibrate",
    "btn_apply",
    "btn_apply_all",
    "btn_export",
    "btn_paint",
)


def normalize_theme_id(raw: str) -> str:
    """Coerce any persisted, legacy, or user-supplied value to a known theme id.

    Used everywhere a theme id is read from ``QSettings`` or passed in from a
    menu action, so the rest of the theme module can assume it always has one
    of exactly two valid ids. Lowercases and strips ``raw``, normalizes hyphens
    to underscores, then maps it: ``"phasor_lab"``/``"dark"`` (old boolean-era
    naming) to :data:`THEME_PHASOR_LAB`, and ``"phasor_lab_light"``/``"light"``
    to :data:`THEME_PHASOR_LAB_LIGHT`. Any other value (unrecognized, empty, or
    corrupted settings) falls back to :data:`DEFAULT_THEME` rather than
    raising, so a bad settings file can never crash theme application.

    Args:
        raw: Theme id as read from settings or a UI action, or any other
            string-like value to normalize.

    Returns:
        One of :data:`THEME_PHASOR_LAB` or :data:`THEME_PHASOR_LAB_LIGHT`.
    """
    key = str(raw).lower().strip().replace("-", "_")
    if key in (THEME_PHASOR_LAB, "dark"):
        return THEME_PHASOR_LAB
    if key in (THEME_PHASOR_LAB_LIGHT, "light"):
        return THEME_PHASOR_LAB_LIGHT
    return DEFAULT_THEME


def is_dark_theme(theme: str) -> bool:
    """Return whether a given theme id corresponds to the dark palette.

    Used by ``EnhancementsMixin._init_enhancements``/``_apply_ui_theme`` to
    maintain the legacy ``self._dark_theme`` boolean flag (and the matching
    ``"dark_theme"`` QSettings key) alongside the newer named-theme system, for
    backward compatibility with any code or settings still expecting a simple
    boolean. Normalizes ``theme`` first so callers do not need to validate it
    themselves.

    Args:
        theme: Theme id, tolerant of legacy/typo'd values via
            :func:`normalize_theme_id`.

    Returns:
        ``True`` if the normalized theme is :data:`THEME_PHASOR_LAB` (dark);
        ``False`` otherwise.
    """
    return normalize_theme_id(theme) == THEME_PHASOR_LAB


def stylesheet_for(theme: str) -> str:
    """Return the full application-wide Qt stylesheet string for a theme.

    Called by ``EnhancementsMixin._apply_ui_theme`` via
    ``self.setStyleSheet(stylesheet_for(theme))`` to restyle every widget in
    the window at once. The returned string is a single Qt stylesheet covering
    generic widget backgrounds/text, group boxes, buttons (including the
    ``[primary="true"]`` accent-button variant), combo/spin boxes, sliders,
    tab bars, text edits, the table widget, header sections, scroll bars, and
    the matplotlib toolbar container — see ``_PHASOR_LAB_DARK_STYLESHEET`` and
    ``_PHASOR_LAB_LIGHT_STYLESHEET`` for the concrete rule sets. The slider
    rules exist specifically so the Brightness/Contrast sliders pick up the
    theme's accent color instead of the platform's default (often clashing)
    highlight color.

    Args:
        theme: Theme id, tolerant of legacy/typo'd values via
            :func:`normalize_theme_id`.

    Returns:
        The Qt stylesheet string for the normalized theme.
    """
    if normalize_theme_id(theme) == THEME_PHASOR_LAB_LIGHT:
        return _PHASOR_LAB_LIGHT_STYLESHEET
    return _PHASOR_LAB_DARK_STYLESHEET


def log_style_for(theme: str) -> str:
    """Return the stylesheet for the activity-log text widget in a theme.

    Called by ``EnhancementsMixin._apply_theme_widgets`` to restyle
    ``self.txt_log`` directly, separately from the global stylesheet, because
    the log widget uses a monospace font and a background distinct from other
    text edits (to visually set it apart as a console-like readout) that is
    easier to express as its own small stylesheet than to encode via a
    global ``QPlainTextEdit#activity_log`` selector alone.

    Args:
        theme: Theme id, tolerant of legacy/typo'd values via
            :func:`normalize_theme_id`.

    Returns:
        Qt stylesheet string setting the log widget's font, background, text
        color, and border for the normalized theme.
    """
    if normalize_theme_id(theme) == THEME_PHASOR_LAB_LIGHT:
        return _LOG_PHASOR_LAB_LIGHT
    return _LOG_PHASOR_LAB_DARK


def toolbar_style_for(theme: str) -> str:
    """Return the stylesheet for matplotlib's navigation toolbar widgets.

    Called by ``EnhancementsMixin._apply_theme_widgets`` for each of
    ``phasor_toolbar``/``image_toolbar``. Matplotlib's ``NavigationToolbar2QT``
    renders its own ``QToolButton``s and labels that need explicit background/
    hover/pressed styling to match the rest of the app, since it is not a
    plain Qt widget the global stylesheet fully reaches. The returned string
    also styles the toolbar's own background/border; note callers additionally
    set the widget's ``QPalette`` via :func:`toolbar_colors_for`, since some of
    matplotlib's toolbar painting uses palette colors rather than the
    stylesheet.

    Args:
        theme: Theme id, tolerant of legacy/typo'd values via
            :func:`normalize_theme_id`.

    Returns:
        Qt stylesheet string for the toolbar and its child buttons/labels.
    """
    if normalize_theme_id(theme) == THEME_PHASOR_LAB_LIGHT:
        return _MPL_TOOLBAR_PHASOR_LAB_LIGHT
    return _MPL_TOOLBAR_PHASOR_LAB_DARK


def toolbar_colors_for(theme: str) -> tuple[str, str]:
    """Return the raw background/foreground hex colors for plot toolbars.

    Called by ``EnhancementsMixin._apply_theme_widgets`` to build a
    ``QPalette`` (window, button, window-text, and button-text roles) applied
    directly to the matplotlib toolbar widgets, in addition to the stylesheet
    from :func:`toolbar_style_for`. This is necessary because parts of
    matplotlib's ``NavigationToolbar2QT`` painting read palette colors rather
    than stylesheet rules, so the two functions must be kept visually
    consistent with each other for the toolbar to look uniform.

    Args:
        theme: Theme id, tolerant of legacy/typo'd values via
            :func:`normalize_theme_id`.

    Returns:
        A ``(background_hex, foreground_hex)`` tuple for the normalized
        theme.
    """
    if normalize_theme_id(theme) == THEME_PHASOR_LAB_LIGHT:
        return "#e8eef5", "#1a2a33"
    return "#252a3d", "#e8eaf0"


_LOG_PHASOR_LAB_DARK = (
    "font-family: Consolas, monospace; font-size: 10px;"
    " background-color: #1e2235; color: #e8eaf0;"
    " border: 1px solid #3d4563;")

_LOG_PHASOR_LAB_LIGHT = (
    "font-family: Consolas, monospace; font-size: 10px;"
    " background-color: #eef6f8; color: #1a2a33;"
    " border: 1px solid #b8dde2;")

_MPL_TOOLBAR_PHASOR_LAB_DARK = (
    "background-color: #252a3d; border: 1px solid #3d4563;"
    " QToolButton { background-color: #252a3d; border: none; padding: 3px; }"
    " QToolButton:hover { background-color: #323852; }"
    " QToolButton:pressed { background-color: #3d4563; }"
    " QLabel { color: #e8eaf0; background: transparent; }")

_MPL_TOOLBAR_PHASOR_LAB_LIGHT = (
    "background-color: #e8eef5; border: 1px solid #b8c5d9;"
    " QToolButton { background-color: #e8eef5; border: none; padding: 3px; }"
    " QToolButton:hover { background-color: #d8e3ef; }"
    " QToolButton:pressed { background-color: #c8d5e3; }"
    " QLabel { color: #1a2a33; background: transparent; }")

_PHASOR_LAB_DARK_STYLESHEET = (
    "QWidget { background-color: #1a1d2e; color: #e8eaf0; }"
    "QLabel { background: transparent; }"
    "QGroupBox {"
    " border: 1px solid #3d4563; border-left: 3px solid #3db8c4;"
    " margin-top: 8px; padding-top: 8px; color: #e8eaf0;"
    " }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #3db8c4; }"
    "QPushButton {"
    " background-color: #323852; color: #e8eaf0; border: 1px solid #3d4563;"
    " border-radius: 4px; padding: 4px 10px; min-height: 18px;"
    " }"
    "QPushButton:hover { background-color: #3d4563; border-color: #4d5778; }"
    "QPushButton:pressed { background-color: #252a3d; }"
    "QPushButton:disabled { color: #6a7088; background-color: #252a3d; border-color: #323852; }"
    "QPushButton[primary=\"true\"] {"
    " background-color: #3db8c4; color: #0d1a1c; border: 1px solid #2e9aa5;"
    " font-weight: 600;"
    " }"
    "QPushButton[primary=\"true\"]:hover { background-color: #4dcad6; border-color: #3db8c4; }"
    "QPushButton[primary=\"true\"]:pressed { background-color: #2e9aa5; }"
    "QPushButton[primary=\"true\"]:disabled {"
    " color: #5a6a6e; background-color: #2a4a50; border-color: #2a4a50;"
    " }"
    "QRadioButton, QCheckBox { spacing: 6px; background: transparent; }"
    "QComboBox, QSpinBox, QDoubleSpinBox {"
    " background-color: #252a3d; color: #e8eaf0; border: 1px solid #3d4563;"
    " border-radius: 3px; padding: 2px 4px; min-height: 18px;"
    " }"
    "QComboBox::drop-down { subcontrol-origin: padding; border-left: 1px solid #3d4563; }"
    "QComboBox QAbstractItemView {"
    " background-color: #252a3d; color: #e8eaf0; selection-background-color: #3db8c4;"
    " selection-color: #0d1a1c;"
    " }"
    "QSlider::groove:horizontal, QSlider::sub-page:horizontal, QSlider::add-page:horizontal {"
    " height: 4px; background: #252a3d; border: 1px solid #3d4563; border-radius: 2px;"
    " }"
    "QSlider::handle:horizontal {"
    " background: #3db8c4; border: 1px solid #2e9aa5;"
    " width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;"
    " }"
    "QSlider::handle:horizontal:hover { background: #4dcad6; }"
    "QSlider::handle:horizontal:pressed { background: #2e9aa5; }"
    "QTabBar::tab {"
    " background: #252a3d; color: #a8b0c8; border: 1px solid #3d4563;"
    " padding: 5px 14px; margin-right: 2px; border-bottom: none;"
    " }"
    "QTabBar::tab:selected {"
    " background: #1a1d2e; color: #3db8c4; border-bottom: 3px solid #3db8c4;"
    " }"
    "QTabBar::tab:hover { color: #e8eaf0; }"
    "QPlainTextEdit, QTextEdit { background-color: #252a3d; color: #e8eaf0; }"
    "QPlainTextEdit#activity_log { background-color: #1e2235; color: #e8eaf0; }"
    "QTabWidget::pane { border: 1px solid #3d4563; background: #1a1d2e; }"
    "QTableWidget { background-color: #252a3d; gridline-color: #3d4563; color: #e8eaf0; }"
    "QHeaderView::section {"
    " background-color: #323852; color: #e8eaf0; border: 1px solid #3d4563; padding: 3px;"
    " }"
    "QScrollBar:vertical { background: #1a1d2e; width: 12px; }"
    "QScrollBar::handle:vertical { background: #3d4563; border-radius: 4px; min-height: 20px; }"
    "QWidget#mpl_toolbar { background-color: #252a3d; border: 1px solid #3d4563; }"
    "QWidget#mpl_toolbar QToolButton { background-color: #252a3d; border: none; }"
    "QWidget#mpl_toolbar QToolButton:hover { background-color: #323852; }")

_PHASOR_LAB_LIGHT_STYLESHEET = (
    "QWidget { background-color: #f4f6fa; color: #1a2a33; }"
    "QLabel { background: transparent; }"
    "QGroupBox {"
    " border: 1px solid #c8d5e3; border-left: 3px solid #2a9dad;"
    " margin-top: 8px; padding-top: 8px; color: #1a2a33;"
    " }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #1a6b74; }"
    "QPushButton {"
    " background-color: #ffffff; color: #1a2a33; border: 1px solid #b8c5d9;"
    " border-radius: 4px; padding: 4px 10px; min-height: 18px;"
    " }"
    "QPushButton:hover { background-color: #eef3f8; border-color: #9ab0c8; }"
    "QPushButton:pressed { background-color: #dde6f0; }"
    "QPushButton:disabled { color: #8a96a8; background-color: #f0f3f7; border-color: #d8e0ea; }"
    "QPushButton[primary=\"true\"] {"
    " background-color: #2a9dad; color: #ffffff; border: 1px solid #1f7f8d;"
    " font-weight: 600;"
    " }"
    "QPushButton[primary=\"true\"]:hover { background-color: #35b0c0; border-color: #2a9dad; }"
    "QPushButton[primary=\"true\"]:pressed { background-color: #1f7f8d; }"
    "QPushButton[primary=\"true\"]:disabled {"
    " color: #d8eef2; background-color: #8ab8c0; border-color: #8ab8c0;"
    " }"
    "QRadioButton, QCheckBox { spacing: 6px; background: transparent; }"
    "QComboBox, QSpinBox, QDoubleSpinBox {"
    " background-color: #ffffff; color: #1a2a33; border: 1px solid #b8c5d9;"
    " border-radius: 3px; padding: 2px 4px; min-height: 18px;"
    " }"
    "QComboBox::drop-down { subcontrol-origin: padding; border-left: 1px solid #b8c5d9; }"
    "QComboBox QAbstractItemView {"
    " background-color: #ffffff; color: #1a2a33; selection-background-color: #2a9dad;"
    " selection-color: #ffffff;"
    " }"
    "QSlider::groove:horizontal, QSlider::sub-page:horizontal, QSlider::add-page:horizontal {"
    " height: 4px; background: #ffffff; border: 1px solid #b8c5d9; border-radius: 2px;"
    " }"
    "QSlider::handle:horizontal {"
    " background: #2a9dad; border: 1px solid #1f7f8d;"
    " width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;"
    " }"
    "QSlider::handle:horizontal:hover { background: #35b0c0; }"
    "QSlider::handle:horizontal:pressed { background: #1f7f8d; }"
    "QTabBar::tab {"
    " background: #e8eef5; color: #5a6a7a; border: 1px solid #c8d5e3;"
    " padding: 5px 14px; margin-right: 2px; border-bottom: none;"
    " }"
    "QTabBar::tab:selected {"
    " background: #f4f6fa; color: #1a6b74; border-bottom: 3px solid #2a9dad;"
    " }"
    "QTabBar::tab:hover { color: #1a2a33; }"
    "QPlainTextEdit, QTextEdit { background-color: #ffffff; color: #1a2a33; }"
    "QPlainTextEdit#activity_log { background-color: #eef6f8; color: #1a2a33; }"
    "QTabWidget::pane { border: 1px solid #c8d5e3; background: #f4f6fa; }"
    "QTableWidget { background-color: #ffffff; gridline-color: #d8e0ea; color: #1a2a33; }"
    "QHeaderView::section {"
    " background-color: #e8eef5; color: #1a2a33; border: 1px solid #d8e0ea; padding: 3px;"
    " }"
    "QScrollBar:vertical { background: #f4f6fa; width: 12px; }"
    "QScrollBar::handle:vertical { background: #c8d5e3; border-radius: 4px; min-height: 20px; }"
    "QWidget#mpl_toolbar { background-color: #e8eef5; border: 1px solid #b8c5d9; }"
    "QWidget#mpl_toolbar QToolButton { background-color: #e8eef5; border: none; }"
    "QWidget#mpl_toolbar QToolButton:hover { background-color: #d8e3ef; }")
