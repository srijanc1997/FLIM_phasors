"""Per-cluster overlay brightness and the images written for one cloud.

The on-screen segmentation is an intensity overlay: a dark photon pixel stays
dark even when it belongs to a bright cluster color. Raising a cluster's
brightness multiplies those painted pixels and clips them at 1, which lifts
the cloud without recoloring it.
"""

from __future__ import annotations

import numpy as np


def normalize_intensity(intensity) -> np.ndarray:
    """Scale a photon image to ``[0, 1]`` the same way the overlay blend does.

    NaNs become 0. The maximum finite value maps to 1, matching
    ``phasorpy.cursor.pseudo_color`` with ``vmin=0``.
    """
    data = np.nan_to_num(np.asarray(intensity, dtype=np.float32), nan=0.0)
    if data.size == 0:
        return data
    peak = float(np.max(data))
    if peak > 0:
        data = data / peak
    np.clip(data, 0.0, 1.0, out=data)
    return data


def _overlay_blend(base_rgb: np.ndarray, color) -> np.ndarray:
    """Photoshop overlay of one RGB color onto pixels shaped ``(N, 3)``."""
    base = np.asarray(base_rgb, dtype=np.float32)
    tint = np.asarray(color, dtype=np.float32).reshape(1, 3)
    return np.where(
        base < 0.5,
        2.0 * base * tint,
        1.0 - 2.0 * (1.0 - base) * (1.0 - tint),
    )


def compose_cluster_overlay(intensity, masks, colors, gains=None) -> np.ndarray:
    """Paint cluster colors on the photon image, then apply each brightness.

    ``gains`` multiplies the painted pixels of that cluster. ``1`` matches the
    unscaled overlay. Pixels in no cluster stay grayscale. Overlapping masks
    are applied in order, each scaled by its own gain.

    Args:
        intensity: 2-D photon image. NaNs are treated as 0.
        masks: Boolean array ``(n, H, W)``.
        colors: Sequence of RGB triples in ``[0, 1]``, one per mask.
        gains: Per-cluster multipliers. Missing entries are ``1``.

    Returns:
        Float array ``(H, W, 3)`` clipped to ``[0, 1]``.
    """
    masks = np.asarray(masks, dtype=bool)
    base = normalize_intensity(intensity)
    if base.shape != masks.shape[1:]:
        raise ValueError(f"intensity shape {base.shape} != mask shape {masks.shape[1:]}")
    rgb = np.repeat(base[..., None], 3, axis=-1)
    n = int(masks.shape[0])
    for k in range(n):
        mask = masks[k]
        if not np.any(mask):
            continue
        gain = 1.0 if gains is None or k >= len(gains) else float(gains[k])
        painted = _overlay_blend(rgb[mask], colors[k])
        rgb[mask] = np.clip(painted * gain, 0.0, 1.0)
    return rgb


def cluster_color_image(intensity, mask, color, gain: float = 1.0) -> np.ndarray:
    """One cluster on a black background, with that cluster's brightness.

    Uses the same normalization as :func:`compose_cluster_overlay`, so a
    pixel here matches that cluster in the combined overlay when masks do
    not overlap.
    """
    base = normalize_intensity(intensity)
    mask = np.asarray(mask, dtype=bool)
    rgb = np.zeros((*base.shape, 3), dtype=np.float32)
    if base.shape != mask.shape or not np.any(mask):
        return rgb
    gray = np.repeat(base[mask, None], 3, axis=1)
    rgb[mask] = np.clip(_overlay_blend(gray, color) * float(gain), 0.0, 1.0)
    return rgb


def masks_shown_on_image(overlay, colors, fallback=None):
    """Exclusive masks of the pixels that show each cluster color.

    The segmentation image is the reference: a pixel belongs to the cluster
    whose color it was painted with. Where the overlay is gray, or so bright
    that it clipped to white, the painted mask is used so that pixel stays
    with the cluster that owns it on the image. Overlapping paints keep the
    last color, which is the one visible in the picture.

    Args:
        overlay: RGB image ``(H, W, 3)`` from the segmentation view, or None.
        colors: RGB triples, one per cluster, in cluster order.
        fallback: Boolean masks ``(n, H, W)`` used when the overlay has no
            hue. Returned unchanged when there is no overlay.

    Returns:
        Boolean array ``(n, H, W)``, or ``fallback`` when the overlay cannot
        be matched to the colors.
    """
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=bool)
    colors = list(colors or [])
    n = len(colors)
    if overlay is None or n == 0:
        return fallback
    rgb = np.clip(np.asarray(overlay, dtype=float)[..., :3], 0.0, 1.0)
    if rgb.ndim != 3 or (fallback is not None and rgb.shape[:2] != fallback.shape[1:]):
        return fallback
    if fallback is not None and fallback.shape[0] != n:
        return fallback
    cols = np.clip(np.asarray([list(c)[:3] for c in colors], dtype=float), 0.0, 1.0)
    gray = rgb.mean(axis=-1)
    delta = rgb - gray[..., None]
    chroma = np.max(np.abs(delta), axis=-1)
    color_delta = cols - cols.mean(axis=1, keepdims=True)
    scores = np.einsum("hwc,nc->hwn", delta, color_delta)
    shown = np.zeros((n,) + rgb.shape[:2], dtype=bool)
    colored = chroma > 0.02
    if np.any(colored):
        best = np.argmax(scores, axis=-1)
        peak = np.max(scores, axis=-1)
        for k in range(n):
            shown[k] = colored & (best == k) & (peak > 0)
    if fallback is not None:
        # Last painted cluster wins, matching compose_cluster_overlay.
        owner = np.full(rgb.shape[:2], -1, dtype=int)
        for k in range(n):
            owner[fallback[k]] = k
        quiet = ~colored
        for k in range(n):
            shown[k] |= quiet & (owner == k)
    return shown


def cluster_gains(stats, n: int) -> list[float]:
    """Brightness multipliers stored on cluster stat rows, defaulting to 1."""
    gains = []
    for k in range(n):
        if stats is not None and k < len(stats):
            gains.append(float(stats[k].get("brightness", 1.0) or 1.0))
        else:
            gains.append(1.0)
    return gains
