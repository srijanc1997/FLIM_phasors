"""Dialog to pick one or more FLIM series inside Leica LIF files.

Presented when a LIF archive contains multiple phasor-bearing measurements so the
user can choose which series to load as separate samples.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from flim_phasors.lif_io import LifPhasorSeries


class LifSeriesDialog(QtWidgets.QDialog):
    """Modal picker for choosing which FLIM series to import from LIF file(s).

    A single Leica ``.lif`` archive can bundle many acquisitions; only some of
    them carry the phasor-relevant FLIM/TCSPC data needed by this app. This
    dialog presents every discovered phasor-bearing series — grouped under a
    non-selectable header per source file when multiple files are involved —
    as a checkable list (all checked by default), with "Select all"/"Select
    none" convenience buttons and standard OK/Cancel actions. The caller
    (typically the sample-loading code in ``enhancements.py``) shows the
    dialog modally, and on acceptance calls :meth:`selected_series` to get the
    list of series the user chose to load, each as a separate sample.
    """

    def __init__(self, series_by_file: dict[str, list[LifPhasorSeries]], parent=None):
        """Construct the dialog and populate the checkable series list.

        Sets the window title/size, then builds a vertical layout with an
        instructional label, the checkable ``QListWidget``
        (``self.list``, in ``NoSelection`` mode so mouse clicks toggle
        checkboxes rather than fighting with row-selection highlighting), the
        Select all/none button row, and a standard OK/Cancel button box wired
        to ``accept``/``reject``. Iterates ``series_by_file`` in
        case-insensitive base-name order; when more than one file is present,
        inserts a non-checkable header item (styled with the palette's
        foreground color) before that file's series so the user can tell which
        file each series came from. For each series, builds a label from its
        display name plus its pixel dimensions (if known), adds a checked,
        checkable list item storing the :class:`LifPhasorSeries` object itself
        as item data (``UserRole``), and appends it to ``self._series`` for
        bookkeeping.

        Args:
            series_by_file: Mapping from LIF file path to the
                :class:`~flim_phasors.lif_io.LifPhasorSeries` entries
                discovered in that file.
            parent: Optional parent widget for standard Qt dialog ownership/
                modality behavior.
        """
        super().__init__(parent)
        self.setWindowTitle("Select FLIM series")
        self.resize(520, 360)
        self._series: list[LifPhasorSeries] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "This Leica file contains multiple FLIM measurements with phasor images.\n"
                "Select one or more series to load as separate samples."
            )
        )

        self.list = QtWidgets.QListWidget()
        # NoSelection: clicks toggle checkboxes only; avoids row highlight stealing focus from checks.
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.list)

        for lif_path in sorted(series_by_file.keys(), key=lambda p: os.path.basename(p).lower()):
            entries = series_by_file[lif_path]
            if len(series_by_file) > 1:
                header = QtWidgets.QListWidgetItem(f"— {os.path.basename(lif_path)} —")
                # NoItemFlags: file header is label-only, not checkable.
                header.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                header.setForeground(self.palette().color(self.foregroundRole()))
                self.list.addItem(header)
            for s in entries:
                label = s.display_name
                if s.shape_yx:
                    label += f"  ({s.shape_yx[1]}×{s.shape_yx[0]})"
                item = QtWidgets.QListWidgetItem(label)
                item.setFlags(
                    item.flags()
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    | QtCore.Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, s)
                self.list.addItem(item)
                self._series.append(s)

        row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Select all")
        btn_none = QtWidgets.QPushButton("Select none")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        row.addWidget(btn_all)
        row.addWidget(btn_none)
        row.addStretch()
        layout.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self):
        """Check every series item in the list, bound to the "Select all" button.

        Iterates all rows in ``self.list`` and, for each item whose flags
        include ``ItemIsUserCheckable`` (i.e. an actual series row, not a
        non-checkable file-header row), sets its check state to ``Checked``.
        Lets the user quickly restore the default "import everything" state
        after using "Select none" or manually unchecking items.
        """
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.CheckState.Checked)

    def _select_none(self):
        """Uncheck every series item in the list, bound to the "Select none" button.

        Iterates all rows in ``self.list`` and, for each item whose flags
        include ``ItemIsUserCheckable`` (i.e. an actual series row, not a
        non-checkable file-header row), sets its check state to
        ``Unchecked``. Useful when the user wants to hand-pick just one or two
        series out of a long list rather than manually unchecking every item.
        """
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def selected_series(self) -> list[LifPhasorSeries]:
        """Return the series objects for all checked list items.

        Called when the user confirms the dialog to determine which LIF
        series to actually import; unchecked items (including
        non-checkable file-header rows) are skipped. Each list item stores
        its associated series object in the Qt ``UserRole`` data slot, set
        when the list was originally populated.

        Returns:
            List of :class:`~flim_phasors.lif_io.LifPhasorSeries` chosen by the user.
        """
        out: list[LifPhasorSeries] = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                s = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(s, LifPhasorSeries):
                    out.append(s)
        return out
