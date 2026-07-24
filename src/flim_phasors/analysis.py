"""Phasor-space analysis helpers for the FLIM Phasors GUI.

Wraps ``phasorpy`` routines for Gaussian-mixture-model (GMM) clustering in the
phasor plane (g, s), elliptic cluster masks, and lifetime extraction from a
single phasor coordinate. GMM fits treat each pixel as a point in normalized
phasor space; BIC selection chooses the component count; apparent lifetimes
(τ_φ, τ_m, τ_n) are derived from the excitation frequency and harmonic.
"""

from __future__ import annotations

import numpy as np
from phasorpy.cluster import phasor_cluster_gmm
from phasorpy.cursor import mask_from_circular_cursor, mask_from_elliptic_cursor
from phasorpy.lifetime import (
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_nearest_neighbor


def fit_phasor_gmm(
    real,
    imag,
    *,
    clusters: int,
    sigma: float = 2.0,
    covariance_type: str = "full",
    sort: str | None = "polar",
    **kwargs,
):
    """Fit a GMM to phasor pixels and return elliptic cluster parameters.

    Each finite (g, s) pixel is treated as a sample in phasor space. The fit
    yields one ellipse per cluster (center, major/minor radii, orientation)
    suitable for drawing cursors or building segmentation masks.

    Args:
        real: Real (g) component of the phasor map, any array broadcastable to
            pixel coordinates.
        imag: Imaginary (s) component of the phasor map.
        clusters: Number of Gaussian components (clusters) to fit.
        sigma: Width scaling for ellipse boundaries (phasorpy convention).
        covariance_type: Sklearn covariance model (``"full"``, ``"tied"``, etc.).
        sort: Optional cluster ordering (``"polar"`` sorts by polar angle).
        **kwargs: Forwarded to ``phasorpy.cluster.phasor_cluster_gmm``.

    Returns:
        Tuple ``(center_real, center_imag, radius_major, radius_minor, angle)``
        with one entry per cluster.
    """
    return phasor_cluster_gmm(
        real,
        imag,
        clusters=int(clusters),
        sigma=float(sigma),
        sort=sort,
        covariance_type=covariance_type,
        **kwargs,
    )


def select_gmm_clusters_bic(
    X: np.ndarray,
    *,
    k_max: int,
    covariance_type: str = "full",
    random_state: int = 0,
    max_points: int = 20_000,
) -> tuple[int, float]:
    """Choose GMM component count by minimum Bayesian Information Criterion (BIC).

    Evaluates sklearn ``GaussianMixture`` models with ``n_components`` from 1
    through ``k_max`` on phasor coordinates (typically stacked g and s values)
    and returns the count with the lowest BIC. When ``X`` has more than
    ``max_points`` rows, a random (but reproducible, seeded by
    ``random_state``) subsample is scanned instead of the full pixel set —
    a full-resolution image can have hundreds of thousands of valid pixels,
    and fitting up to ``k_max`` separate GMMs on all of them is slow without
    meaningfully changing which component count wins, since a few thousand
    points are already enough to estimate cluster structure.

    Args:
        X: 2-D array of shape ``(n_pixels, 2)`` with columns ``[g, s]``.
        k_max: Upper bound on the number of clusters to try.
        covariance_type: Sklearn GMM covariance type passed to each fit.
        random_state: Random seed for reproducible GMM initialization and
            subsampling.
        max_points: Maximum number of points to scan; larger inputs are
            randomly subsampled down to this many.

    Returns:
        ``(best_n, best_bic)`` where ``best_n`` is the selected component
        count and ``best_bic`` is the corresponding BIC value. Returns
        ``(1, inf)`` when fewer than two points are available.
    """
    from sklearn.mixture import GaussianMixture

    pts = np.asarray(X, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 1, float("inf")
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    k_hi = max(1, min(int(k_max), int(pts.shape[0])))
    best_n, best_bic = 1, float("inf")
    for n in range(1, k_hi + 1):
        gm = GaussianMixture(
            n, covariance_type=covariance_type, random_state=random_state,
        ).fit(pts)
        b = float(gm.bic(pts))
        if b < best_bic:
            best_bic, best_n = b, n
    return best_n, best_bic


def label_pixels_by_gmm(
    real,
    imag,
    center_real,
    center_imag,
    radius_major,
    *,
    distance_scale: float = 1.5,
):
    """Assign each pixel to the nearest GMM cluster center within a cutoff.

    Uses phasor-plane Euclidean distance from each pixel's (g, s) to cluster
    centers. Pixels farther than ``distance_scale`` times the largest major
    radius are left unassigned (label -1 in phasorpy convention).

    Args:
        real: Real (g) phasor map.
        imag: Imaginary (s) phasor map.
        center_real: Sequence of cluster center g coordinates from a GMM fit.
        center_imag: Sequence of cluster center s coordinates.
        radius_major: Major-axis radii of fitted ellipses (sets distance scale).
        distance_scale: Multiplier on the maximum major radius for ``distance_max``.

    Returns:
        Integer label array, same shape as ``real``, with cluster indices.
    """
    cr = np.asarray(center_real, dtype=float)
    ci = np.asarray(center_imag, dtype=float)
    rm = np.asarray(radius_major, dtype=float)
    g = np.asarray(real, dtype=float)
    s = np.asarray(imag, dtype=float)
    dmax = float(np.max(rm) * distance_scale) if rm.size else 0.1
    labels = phasor_nearest_neighbor(g, s, cr, ci, distance_max=dmax)
    return np.asarray(labels, dtype=int)


def cursor_masks_for_dataset(d, cursors) -> np.ndarray | None:
    """Boolean masks for phasor cursors on one dataset's calibrated g/s maps.

    Shared by the GUI and export so circle/ellipse geometry stays identical.

    Args:
        d: ``PhasorData`` with calibrated g/s (``real_cal`` / ``imag_cal``).
        cursors: Iterable of cursor dicts from the phasor canvas.

    Returns:
        Stacked boolean array ``(n_cursors, H, W)``, or ``None`` if unavailable.
    """
    if not cursors or getattr(d, "real_cal", None) is None:
        return None
    g, s = d.real_cal, d.imag_cal
    valid = d.valid_mask()
    masks = []
    for c in cursors:
        if c.get("kind") == "ellipse":
            radius_minor = c.get("radius_minor")
            if radius_minor is None:
                radius_minor = c["radius"] * 0.65
            mk = mask_from_elliptic_cursor(
                g, s, [c["center_real"]], [c["center_imag"]],
                radius=[c["radius"]],
                radius_minor=[radius_minor],
                angle=[c.get("angle", 0.0)],
            )
        else:
            mk = mask_from_circular_cursor(
                g, s, [c["center_real"]], [c["center_imag"]], radius=[c["radius"]])
        if mk.ndim == 3:
            mk = mk[0]
        masks.append(mk & valid)
    return np.stack(masks) if masks else None


def lifetimes_at_phasor(g, s, frequency_mhz):
    """Return (τ_φ, τ_m, τ_n) in ns for one (g, s) at ``frequency_mhz``."""
    g = float(g)
    s = float(s)
    freq = float(frequency_mhz)
    tp, tm = phasor_to_apparent_lifetime(g, s, freq)
    tn = phasor_to_normal_lifetime(g, s, freq)
    return float(tp), float(tm), float(tn)


def apply_gmm_to_dataset(
    d,
    *,
    clusters: int = 3,
    sigma: float = 2.0,
    covariance_type: str = "full",
    use_bic: bool = False,
    random_state: int = 0,
):
    """Fit GMM on calibrated maps and stash fit/stats/overlay on ``d``.

    Same labeling/stats path as the GUI's Fit GMM → Paint, without Qt.

    Returns:
        ``(fit, cluster_stats)``.
    """
    from flim_phasors.utils import categorical_name, categorical_rgb

    if getattr(d, "real_cal", None) is None:
        raise ValueError("No calibrated phasor maps on dataset")
    valid = d.valid_mask()
    if int(valid.sum()) < 10:
        raise ValueError("Not enough valid pixels for GMM")
    g, s = d.real_cal, d.imag_cal
    n = max(1, min(12, int(clusters)))
    cov = str(covariance_type or "full")
    sig = max(0.5, min(6.0, float(sigma)))
    if use_bic:
        X = np.column_stack([g[valid], s[valid]])
        n, _bic = select_gmm_clusters_bic(
            X, k_max=n, covariance_type=cov, random_state=random_state)
    fit = fit_phasor_gmm(
        g, s, clusters=n, sigma=sig, covariance_type=cov, random_state=random_state)
    cr, ci, rm, _ri, _ang = fit
    n_comp = len(cr)
    labelmap = label_pixels_by_gmm(g, s, cr, ci, rm)
    masks = np.stack([(labelmap == k) & valid for k in range(n_comp)])
    colors = [categorical_rgb(k) for k in range(n_comp)]
    total_valid = max(int(valid.sum()), 1)
    freq = float(d.work_frequency)
    stats = []
    for k in range(n_comp):
        cg, cs = float(cr[k]), float(ci[k])
        tp, tm, tn = lifetimes_at_phasor(cg, cs, freq)
        nk = int(masks[k].sum())
        stats.append(dict(
            idx=k + 1, color=colors[k], label=categorical_name(k),
            tp=tp, tm=tm, tn=tn, g=cg, s=cs, n=nk,
            area=100.0 * nk / total_valid,
        ))
    d.gmm_fit = fit
    d.cluster_stats = stats
    try:
        from phasorpy.cursor import pseudo_color

        intensity = getattr(d, "mean_thr", None)
        if intensity is None:
            intensity = getattr(d, "mean_raw", None)
        overlay = pseudo_color(
            *[masks[k] for k in range(n_comp)],
            intensity=intensity,
            colors=np.array(colors),
        )
        d.last_overlay = np.clip(np.asarray(overlay), 0, 1)
    except Exception:
        d.last_overlay = None
    return fit, stats
