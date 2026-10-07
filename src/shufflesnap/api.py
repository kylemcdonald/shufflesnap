"""Public ShuffleSnap interface."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import _core
from .grid import Grid
from .normalize import normalize_points

__all__ = [
    "assign",
    "Result",
    "build_schedule",
    "default_offsets",
    "initial_assignment",
    "run_schedule",
    "TRACE_COLUMNS",
    "PRESETS",
]

# Chosen on development instances (seed 0) with the geometric window solver.  "fast" and
# "balanced" take about as long as BSP-OT with 16 and 64 plans (one core, 1e5 points), so
# the two can be compared at equal time; "quality" is one window step further.  Stride
# ratio sqrt 2 was best or close to best at every window size.
PRESETS = {
    "fast": dict(window=5, schedule="geometric", ratio=2 ** 0.5),
    # default: 8x8 windows (64-cell exact solves)
    "balanced": dict(window=8, schedule="geometric", ratio=2 ** 0.5),
    "quality": dict(window=12, schedule="geometric", ratio=2 ** 0.5),
}

TRACE_COLUMNS = (
    "elapsed_s",  # seconds since assign() was called (includes normalization and start)
    "cost",  # total squared displacement after the phase
    "stage",  # stage index in the schedule (-1 for the starting assignment)
    "round",  # round index within the stage
    "phase",  # offset index within the round
    "sx",
    "sy",
    "windows",  # windows in the phase
    "solved",  # windows actually solved (others skipped as unchanged)
    "improved",  # windows whose occupants were reassigned
    "moved",  # points that changed cell
)


def _start_stride(length: int, window: int) -> int:
    """Smallest power of two s with window * s >= length."""
    s = 1
    while window * s < length:
        s *= 2
    return s


def default_offsets(window: int = 6):
    """The four tilings of one round: shifts (0,0), (0,h), (h,0), (h,h), h = window // 2."""
    h = window // 2
    if h == 0:
        return np.array([[0, 0]], dtype=np.int32)
    return np.array([[0, 0], [0, h], [h, 0], [h, h]], dtype=np.int32)


def _stride_sequence(length: int, window: int, ratio: float, start_stride=None):
    s0 = _start_stride(length, window)
    if start_stride is not None:
        s0 = min(s0, int(start_stride))
    out, x = [], float(s0)
    while x > 1.0 + 1e-9:
        out.append(max(1, int(round(x))))
        x /= ratio
    return out


def build_schedule(width: int, height: int, window: int = 6, schedule="halving", start_stride=None,
                   rounds_per_stride: int = 1, final_rounds=None, ratio: float = 2.0) -> np.ndarray:
    """Return the stage table ``(sx, sy, max_rounds, stop_when_no_moves)``.

    ``"halving"``: per axis of length L start at the smallest power of two
    s with ``window * s >= L``; run ``rounds_per_stride`` rounds; halve each stride
    (never below 1) until both are 1; then run stride-1 rounds until a full round makes
    no move (or ``final_rounds`` rounds, if given).  ``"geometric"`` divides the strides
    by ``ratio`` instead of 2 (rounded to integers, repeated values dropped); ratio 2
    reproduces ``"halving"``.  ``start_stride`` caps the starting stride on both axes.
    ``"flat"`` skips the coarse levels (stride 1 only).  A list of
    ``(sx, sy, rounds, stop)`` rows is used verbatim.
    """
    fin = -1 if final_rounds is None else int(final_rounds)
    if not isinstance(schedule, str):
        return np.ascontiguousarray(np.asarray(schedule, dtype=np.int32).reshape(-1, 4))
    if schedule == "flat":
        return np.array([[1, 1, fin, 1]], dtype=np.int32)
    if schedule == "halving":
        ratio = 2.0
    elif schedule != "geometric":
        raise ValueError("schedule must be 'halving', 'geometric', 'flat' or an explicit stage list")
    if not ratio > 1.0:
        raise ValueError("ratio must be > 1")
    sx = _stride_sequence(width, window, ratio, start_stride)
    sy = _stride_sequence(height, window, ratio, start_stride)
    n = max(len(sx), len(sy))
    sx += [1] * (n - len(sx))
    sy += [1] * (n - len(sy))
    stages = []
    for a, b in zip(sx, sy):
        if (a, b) != (1, 1) and (not stages or stages[-1][:2] != (a, b)):
            stages.append((a, b, int(rounds_per_stride), 0))
    stages.append((1, 1, fin, 1))
    return np.array(stages, dtype=np.int32)


def _rowsort(P: np.ndarray, grid: Grid) -> np.ndarray:
    """Start that sorts points into rows by y, then along each row by x."""
    N, M = len(P), grid.n_cells
    lat = grid.lattice
    rows, counts = np.unique(lat[:, 1], return_counts=True)
    if N == M:
        quota = counts.copy()
    else:
        exact = counts * (N / M)
        quota = np.floor(exact).astype(np.int64)
        rem = N - quota.sum()
        order = np.argsort(-(exact - quota), kind="stable")
        quota[order[:rem]] += 1
    order_y = np.lexsort((P[:, 0], P[:, 1]))
    pos = np.empty(N, dtype=np.int32)
    start = 0
    for r, cnt, q in zip(rows, counts, quota):
        if q == 0:
            continue
        pts = order_y[start:start + q]
        start += q
        pts = pts[np.lexsort((P[pts, 1], P[pts, 0]))]
        cells = np.nonzero(lat[:, 1] == r)[0]  # row-major => sorted by x
        if q < cnt:
            pick = np.round(np.linspace(0, cnt - 1, q)).astype(np.int64)
            cells = cells[pick]
        pos[pts] = cells.astype(np.int32)
    return pos


def initial_assignment(P: np.ndarray, grid: Grid, init="random", seed=0) -> np.ndarray:
    """Starting cell of every point (int32, injective)."""
    N, M = len(P), grid.n_cells
    if isinstance(init, np.ndarray) or isinstance(init, (list, tuple)):
        pos = np.array(init, dtype=np.int64)
        if pos.shape != (N,):
            raise ValueError("explicit init must have one cell per point")
        _check_cells(pos, M)
        return np.ascontiguousarray(pos.astype(np.int32))
    if init == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(M)[:N].astype(np.int32)
    if init == "bisect":
        pos = np.empty(N, dtype=np.int32)
        _core.init_bisect(P, grid.centers, pos)
        return pos
    if init == "rowsort":
        return _rowsort(P, grid)
    raise ValueError("init must be 'random', 'bisect', 'rowsort' or an array")


SOLVERS = {"hungarian": 0, "jv": 1, "hungarian_greedy": 2, "auction": 3, "geometric": 4}


def _check_cells(pos, M):
    if len(pos) and (pos.min() < 0 or pos.max() >= M):
        raise ValueError("cell index out of range")
    if len(np.unique(pos)) != len(pos):
        raise ValueError("two points share a cell")


def _check_budget(time_budget):
    """``None`` (no limit) or a non-negative number of seconds; ``0`` returns the start."""
    if time_budget is None:
        return None
    tb = float(time_budget)
    if math.isnan(tb) or tb < 0:
        raise ValueError("time_budget must be None or a non-negative number of seconds")
    return tb


def run_schedule(P, grid: Grid, pos: np.ndarray, stages, window=6, offsets=None, time_budget=None,
                 threads=0, tol_rel=1e-12, skip_clean=True, solver="geometric"):
    """Low-level: improve ``pos`` (int32, C-contiguous, one distinct cell per point) in place
    under a stage table.  Returns (trace, finished)."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    N, M = len(P), grid.n_cells
    if not (isinstance(pos, np.ndarray) and pos.dtype == np.int32 and pos.flags.c_contiguous and pos.shape == (N,)):
        raise TypeError("pos must be a C-contiguous int32 array with one entry per point (it is updated in place)")
    _check_cells(pos, M)
    occ = np.full(M, -1, dtype=np.int32)
    occ[pos] = np.arange(N, dtype=np.int32)
    offsets = default_offsets(window) if offsets is None else np.ascontiguousarray(offsets, dtype=np.int32)
    stages = np.ascontiguousarray(stages, dtype=np.int32)
    tb = _check_budget(time_budget)
    if stages.ndim != 2 or stages.shape[1] != 4 or len(stages) == 0:
        raise ValueError("stages must be a non-empty (S, 4) table of (sx, sy, max_rounds, stop_when_no_moves)")
    if (stages[:, :2] < 1).any():
        raise ValueError("strides must be at least 1")
    if tb is None and ((stages[:, 2] < 0) & (stages[:, 3] == 0)).any():
        raise ValueError("a stage with unlimited rounds never stops without stop_when_no_moves=1 or a time budget")
    tb = -1.0 if tb is None else tb
    flat, finished = _core.run_schedule(P, grid.index, grid.centers, occ, pos, stages, offsets, int(window), tb,
                                        int(threads), float(tol_rel), bool(skip_clean), SOLVERS[solver])
    trace = np.asarray(flat, dtype=np.float64).reshape(-1, len(TRACE_COLUMNS))
    return trace, bool(finished)


@dataclass
class Result:
    """Outcome of :func:`assign`.

    ``cell[i]`` is the id of the cell given to point ``i``; ``xy`` gives its lattice
    column/row.  ``cost`` is the total squared displacement in grid units between the
    normalized points and their cell centers.

    With ``assign(..., exact=True)``, ``lower_bound`` is a proven lower bound on the
    optimal cost, ``gap = cost - lower_bound``, and ``certified`` says whether the gap is
    within the solver's tolerance (i.e. ``cell`` is optimal).  Otherwise all three are None.
    """

    cell: np.ndarray
    grid: Grid
    points: np.ndarray
    cost: float
    finished: bool
    trace: np.ndarray
    timings: dict
    config: dict
    snapshots: list = field(default_factory=list)
    lower_bound: float | None = None
    gap: float | None = None
    certified: bool | None = None

    @property
    def xy(self) -> np.ndarray:
        return self.grid.lattice[self.cell]

    @property
    def mean_sq_displacement(self) -> float:
        return self.cost / max(1, len(self.cell))

    def trace_dict(self) -> dict:
        return {c: self.trace[:, i] for i, c in enumerate(TRACE_COLUMNS)}


def assign(points, grid: Grid | None = None, *, preset: str = "balanced", normalize: str = "bbox", init="random",
           seed: int = 0, window: int | None = None, schedule=None, ratio: float | None = None, start_stride=None,
           rounds_per_stride: int = 1,
           final_rounds=None, offsets=None, time_budget=None, threads: int = 0, skip_clean: bool = True,
           tol_rel: float = 1e-12, snapshots: bool = False, solver: str = "geometric", exact: bool = False,
           exact_time_limit: float | None = None) -> Result:
    """Assign every point to its own grid cell, minimizing total squared displacement.

    Parameters
    ----------
    points : (N, 2) array
        Input coordinates (e.g. a UMAP embedding).
    grid : Grid, optional
        Target cells.  Defaults to ``Grid.for_count(N)`` (near-square, exactly N cells).
        With more cells than points the solver also chooses which cells stay empty.
    preset : {"balanced", "fast", "quality"}
        Bundle of ``window``/``schedule``/``ratio`` (see ``PRESETS``); explicit arguments
        override it.  All three use stride ratio sqrt 2, with 5x5 (``"fast"``), 8x8
        (``"balanced"``, the default) and 12x12 (``"quality"``) windows.
    normalize : {"bbox", "fit", "none"}
        How points are mapped into the grid frame (see :func:`normalize_points`).
    init : {"random", "bisect", "rowsort"} or array
        Starting assignment.  ``"random"`` (the default) is a seeded random permutation;
        structured starts are available but give worse results.
    window : int
        Window side in subgrid cells, 1 to 32 (8 means at most 64 cells per window).
    schedule, ratio, start_stride, rounds_per_stride, final_rounds
        Stride schedule; see :func:`build_schedule`.
    time_budget : float, optional
        Wall-clock budget in seconds for the whole call, including normalization and the
        start.  The best assignment so far is returned when it expires; ``0`` returns the
        start itself.  Negative or NaN budgets are rejected.  With ``exact=True`` it
        bounds only the ShuffleSnap phase; use ``exact_time_limit`` for the exact phase.
    threads : int
        Worker threads for the window solves (0 = OpenMP default).  Results do not
        depend on the thread count.
    snapshots : bool
        Keep a copy of the assignment after every stage (for figures).
    exact : bool
        After ShuffleSnap, solve the assignment exactly with
        :func:`shufflesnap.exact.solve_certified` (requires ``pip install
        shufflesnap[exact]``), seeded by the ShuffleSnap result, and fill in
        ``Result.lower_bound``/``gap``/``certified``.  This can take far longer than
        ShuffleSnap itself (seconds at 10k points, up to tens of minutes at 100k).
    exact_time_limit : float, optional
        Seconds after which the exact phase stops adding candidate edges and returns its
        best solution with ``certified=False``.  It is checked between rounds, so one
        long min-cost-flow solve can overrun it.
    """
    t_start = time.perf_counter()
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    time_budget = _check_budget(time_budget)
    exact_time_limit = _check_budget(exact_time_limit)
    if exact:
        try:
            import ortools  # noqa: F401
            import scipy  # noqa: F401
        except ImportError as e:
            raise ImportError("assign(exact=True) needs OR-Tools and SciPy: pip install shufflesnap[exact]") from e
    cfg = PRESETS[preset]
    window = cfg["window"] if window is None else int(window)
    if schedule is None:
        schedule = cfg["schedule"]
    ratio = cfg["ratio"] if ratio is None else float(ratio)
    P = np.asarray(points, dtype=np.float64)
    N = P.shape[0]
    if grid is None:
        grid = Grid.for_count(N)
    if N > grid.n_cells:
        raise ValueError(f"{N} points do not fit into {grid.n_cells} cells")
    Pn = normalize_points(P, grid, normalize)
    t_norm = time.perf_counter()
    pos = initial_assignment(Pn, grid, init, seed)
    t_init = time.perf_counter()
    stages = build_schedule(grid.width, grid.height, window, schedule, start_stride, rounds_per_stride, final_rounds,
                            ratio)
    remaining = None if time_budget is None else max(0.0, float(time_budget) - (t_init - t_start))
    snaps = []
    if snapshots:
        snaps.append(("start", pos.copy()))
        traces, finished = [], True
        for si, st in enumerate(stages):
            t_stage = time.perf_counter()
            budget = None if remaining is None else max(0.0, remaining - (t_stage - t_init))
            if budget is not None and budget <= 0:
                finished = False
                break
            tr, fin = run_schedule(Pn, grid, pos, st[None, :], window, offsets, budget, threads, tol_rel, skip_clean,
                                   solver)
            tr = tr if si == 0 else tr[1:]
            tr[:, 0] += t_stage - t_init
            tr[:, 2] = np.where(tr[:, 2] >= 0, si, tr[:, 2])
            traces.append(tr)
            snaps.append((f"stage{si}:s={st[0]}x{st[1]}", pos.copy()))
            if not fin:
                finished = False
                break
        trace = np.concatenate(traces) if traces else np.zeros((0, len(TRACE_COLUMNS)))
    else:
        trace, finished = run_schedule(Pn, grid, pos, stages, window, offsets, remaining, threads, tol_rel, skip_clean,
                                       solver)
    t_ss = time.perf_counter()
    if len(trace):
        trace[:, 0] += t_init - t_start
    cost = float(_core.assignment_cost(Pn, grid.centers, pos, threads))
    lower_bound = gap = certified = exact_info = None
    if exact:
        from . import exact as _exact

        ex = _exact.solve_certified(Pn, grid, pos, threads=threads, time_limit=exact_time_limit)
        exact_info = dict(shufflesnap_cost=cost, rounds=ex["rounds"], tolerance=ex["tolerance"])
        if ex["cost"] <= cost:  # guard against integer cost scaling ever making it worse
            pos, cost = ex["cell"], ex["cost"]
        lower_bound = ex["lower_bound"]
        gap = cost - lower_bound
        certified = bool(gap <= ex["tolerance"])
    t_end = time.perf_counter()
    timings = {
        "normalize_s": t_norm - t_start,
        "init_s": t_init - t_norm,
        "solve_s": t_ss - t_init,
        "exact_s": t_end - t_ss,
        "total_s": t_end - t_start,
    }
    config = dict(preset=preset, normalize=normalize, init=init if isinstance(init, str) else "explicit", seed=seed,
                  window=window, schedule=schedule if isinstance(schedule, str) else "custom", ratio=ratio,
                  start_stride=start_stride, rounds_per_stride=rounds_per_stride, final_rounds=final_rounds,
                  time_budget=time_budget, threads=threads, skip_clean=skip_clean, tol_rel=tol_rel, solver=solver,
                  stages=stages.tolist(), exact=exact, exact_time_limit=exact_time_limit)
    if exact_info is not None:
        config["exact_info"] = exact_info
    return Result(cell=pos, grid=grid, points=Pn, cost=cost, finished=finished, trace=trace, timings=timings,
                  config=config, snapshots=snaps, lower_bound=lower_bound, gap=gap, certified=certified)
