"""Wrappers around phasorpy analysis helpers used by the GUI."""

from __future__ import annotations

import numpy as np
from phasorpy.cluster import phasor_cluster_gmm
from phasorpy.cursor import mask_from_elliptic_cursor
from phasorpy.lifetime import (
    phasor_center,
    phasor_to_apparent_lifetime,
    phasor_to_lifetime_search,
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
    """Fit GMM clusters; returns phasorpy ellipse parameter tuples."""
    return phasor_cluster_gmm(
        real,
        imag,
        clusters=int(clusters),
        sigma=float(sigma),
        sort=sort,
        covariance_type=covariance_type,
        **kwargs,
    )


def label_pixels_by_gmm(
    real,
    imag,
    center_real,
    center_imag,
    radius_major,
    *,
    distance_scale: float = 1.5,
):
    """Assign each pixel to nearest cluster center within scaled major radius."""
    cr = np.asarray(center_real, dtype=float)
    ci = np.asarray(center_imag, dtype=float)
    rm = np.asarray(radius_major, dtype=float)
    g = np.asarray(real, dtype=float)
    s = np.asarray(imag, dtype=float)
    dmax = float(np.max(rm) * distance_scale) if rm.size else 0.1
    labels = phasor_nearest_neighbor(g, s, cr, ci, distance_max=dmax)
    return np.asarray(labels, dtype=int)


def masks_from_gmm_ellipses(real, imag, gmm_fit, valid_mask):
    """Boolean mask per cluster from elliptic GMM regions."""
    cr, ci, rm, ri, ang = gmm_fit
    masks = []
    for k in range(len(cr)):
        mk = mask_from_elliptic_cursor(
            real,
            imag,
            [cr[k]],
            [ci[k]],
            radius=[rm[k]],
            radius_minor=[ri[k]],
            angle=[ang[k]],
        )
        if mk.ndim == 3:
            mk = mk[0]
        masks.append(mk & valid_mask)
    return np.stack(masks) if masks else np.zeros((0,) + np.shape(real), dtype=bool)


def lifetimes_at_phasor(g, s, frequency_mhz):
    """Apparent lifetimes at a single (g, s) phasor coordinate."""
    g = float(g)
    s = float(s)
    freq = float(frequency_mhz)
    tp, tm = phasor_to_lifetime_search(
        np.array([[g]], dtype=float),
        np.array([[s]], dtype=float),
        freq,
    )
    tn = phasor_to_normal_lifetime(g, s, freq)
    return float(np.asarray(tp).ravel()[0]), float(np.asarray(tm).ravel()[0]), float(tn)


def global_phasor_center(mean, real, imag, *, method="mean"):
    """Intensity-weighted phasor center (g, s) and mean photons."""
    m, gr, gi = phasor_center(mean, real, imag, method=method)
    return float(np.asarray(gr).ravel()[0]), float(np.asarray(gi).ravel()[0]), float(np.asarray(m).ravel()[0])
