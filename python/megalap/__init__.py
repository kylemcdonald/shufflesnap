from __future__ import annotations

import math

import numpy as np

from ._core import _linear_sum_assignment, _window_cleanup

DEFAULT_CLEANUP_SECONDS = 10.0

__all__ = [
    "DEFAULT_CLEANUP_SECONDS",
    "default_stride_schedule",
    "linear_sum_assignment",
    "window_cleanup",
    "snap_to_grid",
]


def _build_target_grid(width: int, height: int, margin: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width > 1:
        xs = np.linspace(margin, 1.0 - margin, width, dtype=np.float64)
    else:
        xs = np.array([0.5], dtype=np.float64)
    if height > 1:
        ys = np.linspace(margin, 1.0 - margin, height, dtype=np.float64)
    else:
        ys = np.array([0.5], dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])


def _choose_grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be positive")

    exact_candidates: list[tuple[int, int]] = []
    root = int(math.isqrt(n))
    for height in range(1, root + 1):
        if n % height != 0:
            continue
        width = n // height
        if width < height:
            width, height = height, width
        if width / height <= 2.0:
            exact_candidates.append((width, height))

    if exact_candidates:
        return min(exact_candidates, key=lambda wh: (wh[0] - wh[1], wh[0]))

    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    max_height = int(math.ceil(math.sqrt(n)))
    for height in range(1, max_height + 1):
        width = math.ceil(n / height)
        if width < height:
            width, height = height, width
        if width / height > 2.0:
            continue
        area = width * height
        key = (area, width - height, width)
        if best_key is None or key < best_key:
            best_key = key
            best = (width, height)

    if best is None:
        side = int(math.ceil(math.sqrt(n)))
        return side, side
    return best


def default_stride_schedule(rows: int, cols: int, window_size: int = 6) -> list[tuple[int, int]]:
    """Coarse-to-fine stride schedule: starts at the smallest power-of-two
    stride whose windows span the whole grid, halving down to (1, 1). One
    round is run at each entry; the final entry repeats until the budget
    expires or the assignment converges."""
    stride_r = 1
    while stride_r * window_size < rows:
        stride_r *= 2
    stride_c = 1
    while stride_c * window_size < cols:
        stride_c *= 2
    schedule: list[tuple[int, int]] = []
    while stride_r > 1 or stride_c > 1:
        schedule.append((stride_r, stride_c))
        stride_r = max(1, stride_r // 2)
        stride_c = max(1, stride_c // 2)
    schedule.append((1, 1))
    return schedule


def _resolve_stride_schedule(strides, rows: int, cols: int, window_size: int) -> list[int]:
    if strides is None:
        schedule = default_stride_schedule(rows, cols, window_size)
    elif isinstance(strides, int):
        schedule = [(strides, strides)]
    else:
        schedule = []
        for entry in strides:
            if isinstance(entry, int):
                schedule.append((entry, entry))
            else:
                stride_r, stride_c = entry
                schedule.append((int(stride_r), int(stride_c)))
    if not schedule:
        raise ValueError("strides must contain at least one entry")
    flat: list[int] = []
    for stride_r, stride_c in schedule:
        if stride_r < 1 or stride_c < 1:
            raise ValueError("strides must be positive")
        flat.extend((int(stride_r), int(stride_c)))
    return flat


def _normalize_num_threads(num_threads: int | None) -> int:
    if num_threads is None:
        return 0
    num_threads = int(num_threads)
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    return num_threads


def linear_sum_assignment(cost_matrix):
    cost = np.asarray(cost_matrix, dtype=np.float64, order="C")
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost_matrix must be a square 2D array")
    row_ind, col_ind, total_cost = _linear_sum_assignment(cost)
    return (
        np.asarray(row_ind, dtype=np.int64),
        np.asarray(col_ind, dtype=np.int64),
        float(total_cost),
    )


def _spread_seed_assignment(points: np.ndarray, total_cells: int) -> np.ndarray:
    """Cheap legal seed: points sorted by (y, x) are placed on evenly spread
    cells in row-major order. Exactly the identity raster when n == cells."""
    n = int(points.shape[0])
    order = np.lexsort((points[:, 0], points[:, 1]))
    cells = np.floor(np.arange(n, dtype=np.float64) * (total_cells / n)).astype(np.int64)
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = cells
    return assignment


def window_cleanup(
    points,
    initial_assignment,
    rows: int,
    cols: int,
    budget_seconds: float | None = None,
    window_size: int = 6,
    margin: float = 0.03,
    num_threads: int | None = None,
    fixed_suffix_count: int = 0,
    strides=None,
    trace_rounds: bool = False,
):
    """Improve a legal assignment with multiscale window cleanup.

    Runs one round per stride-schedule entry (coarse to fine by default), then
    repeats the finest stride until ``budget_seconds`` expires or a full
    stride-(1, 1) round changes nothing (``converged``). ``budget_seconds=None``
    runs until convergence. Points may number fewer than ``rows * cols``; the
    unassigned cells act as holes that drift to wherever they least hurt the
    objective.
    """
    pts = np.asarray(points, dtype=np.float64, order="C")
    assignment = np.asarray(initial_assignment, dtype=np.int64, order="C")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if assignment.ndim != 1 or assignment.shape[0] != pts.shape[0]:
        raise ValueError("initial_assignment must have shape (n,)")
    if int(rows) * int(cols) < pts.shape[0]:
        raise ValueError("rows * cols must be at least the number of points")
    budget = math.inf if budget_seconds is None else float(budget_seconds)
    result = _window_cleanup(
        pts,
        assignment,
        int(rows),
        int(cols),
        budget,
        int(window_size),
        float(margin),
        int(fixed_suffix_count),
        _normalize_num_threads(num_threads),
        _resolve_stride_schedule(strides, int(rows), int(cols), int(window_size)),
        bool(trace_rounds),
    )
    result["assignment"] = np.asarray(result["assignment"], dtype=np.int64)
    if trace_rounds:
        result["round_elapsed_s"] = np.asarray(result["round_elapsed_s"], dtype=np.float64)
        result["round_costs"] = np.asarray(result["round_costs"], dtype=np.float64)
        result["round_strides"] = np.asarray(result["round_strides"], dtype=np.int64)
    return result


def snap_to_grid(
    points,
    width: int | None = None,
    height: int | None = None,
    cleanup_seconds: float | None = None,
    window_size: int = 6,
    margin: float = 0.03,
    num_threads: int | None = None,
):
    """Assign a 2D point cloud to distinct cells of a regular grid.

    Any ``n <= width * height`` is supported directly: leftover cells stay
    empty and drift toward the sparsest parts of the cloud during cleanup.
    ``cleanup_seconds`` caps the cleanup budget (default
    ``DEFAULT_CLEANUP_SECONDS``); cleanup stops early once converged.
    """
    pts = np.asarray(points, dtype=np.float64, order="C")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    n = int(pts.shape[0])
    if width is None and height is None:
        width, height = _choose_grid_shape(n)
    elif width is None or height is None:
        raise ValueError("width and height must be provided together")
    else:
        width = int(width)
        height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    total_cells = width * height
    if total_cells < n:
        raise ValueError("width * height must be at least the number of source points")

    assignment = _spread_seed_assignment(pts, total_cells)
    budget = DEFAULT_CLEANUP_SECONDS if cleanup_seconds is None else float(cleanup_seconds)

    if budget > 0.0:
        cleanup = window_cleanup(
            pts,
            assignment,
            rows=height,
            cols=width,
            budget_seconds=budget,
            window_size=window_size,
            margin=margin,
            num_threads=num_threads,
        )
        assignment = cleanup["assignment"]

    target_points = _build_target_grid(width, height, float(margin))
    grid_points = target_points[assignment]
    return grid_points, assignment.copy(), (width, height)
