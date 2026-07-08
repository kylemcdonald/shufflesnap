from __future__ import annotations

import math

import numpy as np

from ._core import _linear_sum_assignment, _window_cleanup

__all__ = [
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
    elif isinstance(strides, (int, np.integer)):
        schedule = [(int(strides), int(strides))]
    else:
        schedule = []
        for entry in strides:
            if isinstance(entry, (int, np.integer)):
                schedule.append((int(entry), int(entry)))
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


def _spread_seed_assignment(points: np.ndarray, usable_cells: np.ndarray) -> np.ndarray:
    """Cheap legal seed: points sorted by (y, x) are placed on evenly spread
    usable cells in row-major order. Exactly the identity raster when the
    usable cells are all cells and n == cells."""
    n = int(points.shape[0])
    total = int(usable_cells.shape[0])
    order = np.lexsort((points[:, 0], points[:, 1]))
    picks = np.floor(np.arange(n, dtype=np.float64) * (total / n)).astype(np.int64)
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = usable_cells[picks]
    return assignment


def _resolve_cell_mask(cell_mask, rows: int, cols: int) -> list[int]:
    """Normalize a usable-cell mask to the kernel's blocked-cell id list."""
    if cell_mask is None:
        return []
    mask = np.asarray(cell_mask, dtype=bool)
    if mask.ndim == 2:
        if mask.shape != (rows, cols):
            raise ValueError("cell_mask must have shape (rows, cols)")
        mask = mask.reshape(-1)
    elif mask.ndim != 1 or mask.shape[0] != rows * cols:
        raise ValueError("cell_mask must have shape (rows, cols) or (rows * cols,)")
    return np.flatnonzero(~mask).tolist()


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
    cell_mask=None,
):
    """Improve a legal assignment with multiscale window cleanup.

    Runs one round per stride-schedule entry (coarse to fine by default), then
    repeats the finest stride until ``budget_seconds`` expires or a full
    stride-(1, 1) round changes nothing (``converged``). ``budget_seconds=None``
    runs until convergence. Points may number fewer than ``rows * cols``; the
    unassigned cells act as holes that drift to wherever they least hurt the
    objective. ``cell_mask`` (bool, ``(rows, cols)`` or flat) marks which cells
    may be used; masked-out cells are never assigned.
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
    schedule = _resolve_stride_schedule(strides, int(rows), int(cols), int(window_size))
    if math.isinf(budget) and tuple(schedule[-2:]) != (1, 1):
        raise ValueError(
            "budget_seconds=None runs until a stride-(1, 1) round converges, "
            "so the stride schedule must end with 1 (or pass a finite budget)"
        )
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
        schedule,
        bool(trace_rounds),
        _resolve_cell_mask(cell_mask, int(rows), int(cols)),
    )
    result["assignment"] = np.asarray(result["assignment"], dtype=np.int64)
    if trace_rounds:
        result["round_elapsed_s"] = np.asarray(result["round_elapsed_s"], dtype=np.float64)
        result["round_costs"] = np.asarray(result["round_costs"], dtype=np.float64)
        result["round_strides"] = np.asarray(result["round_strides"], dtype=np.int64)
        result["round_strides_col"] = np.asarray(result["round_strides_col"], dtype=np.int64)
    return result


def snap_to_grid(
    points,
    width: int | None = None,
    height: int | None = None,
    cleanup_seconds: float | None = None,
    window_size: int = 6,
    margin: float = 0.03,
    num_threads: int | None = None,
    mask=None,
):
    """Assign a 2D point cloud to distinct cells of a regular grid.

    Any ``n <= width * height`` is supported directly: leftover cells stay
    empty and drift toward the sparsest parts of the cloud during cleanup.
    Pass ``mask`` (bool array, shape ``(height, width)``) to restrict which
    cells may be used, e.g. to shape the atlas or to place the empty cells by
    hand; the grid shape is taken from the mask when ``width``/``height`` are
    omitted. By default cleanup runs until it converges (no window can improve
    the assignment); pass ``cleanup_seconds`` to cap the time instead, or
    ``0.0`` to return the raw seed.
    """
    pts = np.asarray(points, dtype=np.float64, order="C")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if pts.shape[0] == 0:
        raise ValueError("points must be non-empty")

    n = int(pts.shape[0])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D boolean array of shape (height, width)")
        if width is None and height is None:
            height, width = mask.shape
        else:
            if (int(height), int(width)) != mask.shape:
                raise ValueError("mask shape must match (height, width)")
        width = int(width)
        height = int(height)
    elif width is None and height is None:
        width, height = _choose_grid_shape(n)
    elif width is None or height is None:
        raise ValueError("width and height must be provided together")
    else:
        width = int(width)
        height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    if mask is not None:
        usable_cells = np.flatnonzero(mask.reshape(-1))
    else:
        usable_cells = np.arange(width * height, dtype=np.int64)
    if usable_cells.shape[0] < n:
        raise ValueError("the grid (after masking) must have at least as many cells as points")

    assignment = _spread_seed_assignment(pts, usable_cells)
    budget = None if cleanup_seconds is None else float(cleanup_seconds)

    if budget is None or budget > 0.0:
        cleanup = window_cleanup(
            pts,
            assignment,
            rows=height,
            cols=width,
            budget_seconds=budget,
            window_size=window_size,
            margin=margin,
            num_threads=num_threads,
            cell_mask=mask,
        )
        assignment = cleanup["assignment"]

    target_points = _build_target_grid(width, height, float(margin))
    grid_points = target_points[assignment]
    return grid_points, assignment.copy(), (width, height)
