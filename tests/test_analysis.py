"""Tests for phasor analysis helpers (lifetimes, GMM BIC selection)."""



import numpy as np

import pytest



from flim_phasors.analysis import lifetimes_at_phasor, select_gmm_clusters_bic





def test_lifetimes_at_phasor_apparent():

    """Return finite τφ, τmod, and τ normal for a valid (g, s) coordinate."""

    tp, tm, tn = lifetimes_at_phasor(0.5, 0.3, 80.0)

    assert np.isfinite(tp) and np.isfinite(tm) and np.isfinite(tn)





def test_select_gmm_clusters_bic_finds_three_clusters():

    """BIC search picks k=3 for well-separated synthetic phasor clusters."""

    pytest.importorskip("sklearn")

    rng = np.random.default_rng(0)

    centers = np.array([[0.2, 0.15], [0.5, 0.35], [0.75, 0.45]])

    chunks = [rng.normal(c, 0.02, size=(80, 2)) for c in centers]

    X = np.vstack(chunks)

    n, bic = select_gmm_clusters_bic(X, k_max=5, covariance_type="full")

    assert n == 3

    assert np.isfinite(bic)





def test_select_gmm_clusters_bic_caps_k_by_sample_count():

    """Never request more GMM components than phasor pixels in the fit set."""

    pytest.importorskip("sklearn")

    X = np.array([[0.3, 0.2], [0.31, 0.21], [0.7, 0.4], [0.71, 0.39]])

    n, _ = select_gmm_clusters_bic(X, k_max=12, covariance_type="full")

    assert 1 <= n <= X.shape[0]


