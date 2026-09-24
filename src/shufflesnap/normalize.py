"""Mapping input coordinates into the grid frame.

All costs are measured in the grid frame, where neighbouring cell centers are one
pitch apart.  The mapping is part of the problem definition: every method compared
in a benchmark must receive the same normalized coordinates.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid

__all__ = ["normalize_points", "MODES"]

MODES = ("bbox", "fit", "none")


def normalize_points(points, grid: Grid, mode: str = "bbox") -> np.ndarray:
    """Return ``points`` mapped into the coordinate frame of ``grid.centers``.

    ``"bbox"``  per-axis affine map of the point bounding box onto the bounding box of
                the usable cell centers (default; stretches each axis independently).
    ``"fit"``   uniform scale preserving the aspect ratio, fitted inside the cell
                bounding box and centered.
    ``"none"``  points are already in grid units; returned as float64 unchanged.

    An axis on which all points coincide is mapped to the middle of the cell range.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(P).all():
        raise ValueError("points contain NaN or infinite values")
    if mode == "none":
        return np.ascontiguousarray(P)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    C = grid.centers
    pmin, pmax = P.min(axis=0), P.max(axis=0)
    cmin, cmax = C.min(axis=0), C.max(axis=0)
    pext = pmax - pmin
    cext = cmax - cmin
    out = np.empty_like(P)
    if mode == "bbox":
        for a in range(2):
            if pext[a] > 0:
                out[:, a] = cmin[a] + (P[:, a] - pmin[a]) * (cext[a] / pext[a])
            else:
                out[:, a] = 0.5 * (cmin[a] + cmax[a])
    else:  # fit
        with np.errstate(divide="ignore"):
            ratios = np.where(pext > 0, cext / np.where(pext > 0, pext, 1.0), np.inf)
        s = float(np.min(ratios)) if np.isfinite(ratios).any() else 1.0
        for a in range(2):
            mid_p = 0.5 * (pmin[a] + pmax[a])
            mid_c = 0.5 * (cmin[a] + cmax[a])
            out[:, a] = mid_c + (P[:, a] - mid_p) * s
    return np.ascontiguousarray(out)
