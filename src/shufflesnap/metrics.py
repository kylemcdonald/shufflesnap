"""Validation and quality measures (pure NumPy/SciPy, independent of the kernels)."""

from __future__ import annotations

import numpy as np

from .grid import Grid

__all__ = [
    "validate_assignment",
    "assignment_cost",
    "displacement_stats",
    "knn_preservation",
    "grid_neighbor_preservation",
]


def validate_assignment(cell, n_points: int, grid: Grid) -> None:
    """Raise ``ValueError`` unless ``cell`` gives every point its own usable cell."""
    cell = np.asarray(cell)
    if cell.shape != (n_points,):
        raise ValueError(f"expected {n_points} cell ids, got shape {cell.shape}")
    if not np.issubdtype(cell.dtype, np.integer):
        raise ValueError("cell ids must be integers")
    if n_points and (cell.min() < 0 or cell.max() >= grid.n_cells):
        raise ValueError("cell id out of range")
    if len(np.unique(cell)) != n_points:
        raise ValueError("two points share a cell")
    lat = grid.lattice[cell]
    if not grid.mask[lat[:, 1], lat[:, 0]].all():
        raise ValueError("a point was placed on a masked cell")


def assignment_cost(points_normalized, grid: Grid, cell) -> float:
    """Total squared displacement between normalized points and their cell centers."""
    d = np.asarray(points_normalized, dtype=np.float64) - grid.centers[np.asarray(cell)]
    return float(np.einsum("ij,ij->", d, d))


def displacement_stats(points_normalized, grid: Grid, cell) -> dict:
    d = np.asarray(points_normalized, dtype=np.float64) - grid.centers[np.asarray(cell)]
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    return {
        "mean_sq": float(np.mean(r ** 2)),
        "rms": float(np.sqrt(np.mean(r ** 2))),
        "mean": float(np.mean(r)),
        "median": float(np.median(r)),
        "p95": float(np.quantile(r, 0.95)),
        "p99": float(np.quantile(r, 0.99)),
        "max": float(np.max(r)),
    }


def knn_preservation(points, grid: Grid, cell, k: int = 10, sample=None, seed: int = 0) -> float:
    """Mean fraction of each point's k nearest input neighbours that are also among its
    k nearest grid neighbours (grid distance between assigned cell centers; ties at the
    k-th distance are counted as neighbours).  ``sample`` limits the evaluation to a random
    subset of query points.
    """
    from scipy.spatial import cKDTree

    X = np.asarray(points, dtype=np.float64)
    Y = grid.centers[np.asarray(cell)]
    n = len(X)
    q = np.arange(n)
    if sample is not None and sample < n:
        q = np.random.default_rng(seed).choice(n, sample, replace=False)
    _, nx = cKDTree(X).query(X[q], k=k + 1)
    ty = cKDTree(Y)
    dk, _ = ty.query(Y[q], k=k + 1)
    rk = dk[:, -1]
    hits = 0
    for row, (i, r) in enumerate(zip(q, rk)):
        ids = ty.query_ball_point(Y[i], r + 1e-9)
        s = set(ids)
        s.discard(i)
        nb = [j for j in nx[row] if j != i][:k]
        hits += sum(1 for j in nb if j in s)
    return hits / (len(q) * k)


def grid_neighbor_preservation(points, grid: Grid, cell, sample=None, seed: int = 0) -> dict:
    """For each point, compare its 8 lattice neighbours (occupied cells at Chebyshev
    distance 1) with its nearest input neighbours.

    Returns the mean input-space rank of lattice neighbours relative to the number of
    points (lower is better, normalized to [0, 1]) and the fraction of lattice
    neighbours that are among the point's 8 nearest input neighbours.
    """
    from scipy.spatial import cKDTree

    X = np.asarray(points, dtype=np.float64)
    cell = np.asarray(cell)
    n = len(X)
    occ = np.full(grid.mask.shape, -1, dtype=np.int64)
    lat = grid.lattice[cell]
    occ[lat[:, 1], lat[:, 0]] = np.arange(n)
    q = np.arange(n)
    if sample is not None and sample < n:
        q = np.random.default_rng(seed).choice(n, sample, replace=False)
    _, nx = cKDTree(X).query(X[q], k=9)
    H, W = grid.mask.shape
    hit = tot = 0
    for row, i in enumerate(q):
        x, y = lat[i]
        s = set(nx[row][1:].tolist())
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                xx, yy = x + dx, y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    j = occ[yy, xx]
                    if j >= 0:
                        tot += 1
                        hit += j in s
    return {"lattice_nbr_in_input_8nn": hit / max(tot, 1)}
