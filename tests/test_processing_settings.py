"""Per-sample processing settings helpers."""

from flim_phasors.data import PhasorData
from flim_phasors.gui.processing import PROC_SETTING_KEYS, processing_params_for_dataset


class _WinStub:
    def __init__(self, *, multi: bool, stash: dict | None):
        self.data = PhasorData()
        self.datasets = [self.data, PhasorData()] if multi else [self.data]
        self.ref_calibration = type("C", (), {"is_active": False})()
        self.cb_filter = _Combo("median")
        self.sp_reflt = _Val(4.0)
        self.sp_msize = _Val(3)
        self.sp_mrep = _Val(1)
        self.sp_psigma = _Val(2.0)
        self.sp_plevels = _Val(1)
        self.sp_thr = _Val(100)
        self.chk_detect_harm = _Chk(True)
        self._stash = stash

    def _effective_ref_path(self, d):
        return ""

    def _active_calibration(self):
        return None


class _Chk:
    def __init__(self, v):
        self._v = v

    def isChecked(self):
        return self._v


class _Combo:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _Val:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v


def test_per_sample_uses_stash():
    d = PhasorData()
    d.processing_settings = {
        "filter_mode": "gaussian",
        "median_size": 5,
        "median_repeat": 2,
        "paw_sigma": 2.0,
        "paw_levels": 1,
        "intensity_min": 50.0,
        "detect_harmonics": False,
        "ref_lifetime": 3.5,
    }
    win = _WinStub(multi=True, stash=d.processing_settings)
    p = processing_params_for_dataset(win, d)
    assert p["filter_mode"] == "gaussian"
    assert p["median_size"] == 5
    assert p["intensity_min"] == 50.0
    assert p["detect_harmonics"] is False


def test_proc_setting_keys_complete():
    assert "filter_mode" in PROC_SETTING_KEYS
    assert "harmonic" in PROC_SETTING_KEYS
