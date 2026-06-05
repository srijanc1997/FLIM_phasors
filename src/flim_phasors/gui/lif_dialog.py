"""Dialog to pick one or more FLIM series inside Leica LIF files.

Presented when a LIF archive contains multiple phasor-bearing measurements so the
user can choose which series to load as separate samples.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from flim_phasors.lif_io import LifPhasorSeries


class LifSeriesDialog(QtWidgets.QDialog):
    """Multi-select list of phasor-bearing FLIM series across one or more LIF files."""

    def __init__(self, series_by_file: dict[str, list[LifPhasorSeries]], parent=None):
        """Build the series picker dialog.

        Args:
            series_by_file: Mapping from LIF file path to discoverable
                :class:`~flim_phasors.lif_io.LifPhasorSeries` entries.
            parent: Optional parent widget.
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
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.list)

        for lif_path in sorted(series_by_file.keys(), key=lambda p: os.path.basename(p).lower()):
            entries = series_by_file[lif_path]
            if len(series_by_file) > 1:
                header = QtWidgets.QListWidgetItem(f"— {os.path.basename(lif_path)} —")
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
        """Check every selectable series item in the list."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.CheckState.Checked)

    def _select_none(self):
        """Uncheck every selectable series item in the list."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def selected_series(self) -> list[LifPhasorSeries]:
        """Return the series objects for all checked list items.

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
