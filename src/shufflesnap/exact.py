"""Certified exact point-to-grid assignment (optional; requires ``ortools``).

The solver never forms the dense N x M cost matrix.  It alternates

1. an exact min-cost-flow solve (OR-Tools cost scaling) restricted to a sparse set of
   candidate point-cell edges, with costs scaled to integers;
2. integer shortest-path cell potentials for that solution (SPFA on the cell graph);
3. a global pricing pass over *all* usable cells, computing the Lagrangian lower bound
   ``LB(v) = sum_i min_j (c_ij - v_j) + sum_j v_j - (M - N) max_j v_j`` and every edge
   whose reduced cost undercuts the point's current cell;

adding violating edges until none remain.  The returned lower bound is valid for any
potentials, so ``cost - lower_bound`` is a certified optimality gap in real arithmetic
(the only approximation is the integer scaling, whose effect is bounded by ``N * 2/S``
and reported).
"""

from __future__ import annotations

import time

import numpy as np

from . import _core
from .grid import Grid

__all__ = ["candidate_edges", "solve_certified", "lower_bound", "solve_dense"]


def _sqdist(P, C, rows, cols):
    d = P[rows] - C[cols]
    return np.einsum("ij,ij->i", d, d)


def candidate_edges(P: np.ndarray, grid: Grid, cell0: np.ndarray | None, k: int = 12, radius: int = 2):
    """Sparse candidate edges: each point's k nearest cells, plus the (2r+1)^2 lattice
    neighbourhood of its cell in ``cell0`` (which guarantees a feasible matching)."""
    from scipy.spatial import cKDTree

    N, M = len(P), grid.n_cells
    rows_l, cols_l = [], []
    kk = min(k, M)
    if kk > 0:
        _, idx = cKDTree(grid.centers).query(P, k=kk)
        idx = idx.reshape(N, kk)
        rows_l.append(np.repeat(np.arange(N), kk))
        cols_l.append(idx.ravel())
    if cell0 is not None:
        lat = grid.lattice[cell0]
        H, W = grid.mask.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = lat[:, 0] + dx
                y = lat[:, 1] + dy
                ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                ids = np.full(N, -1, dtype=np.int64)
                ids[ok] = grid.index[y[ok], x[ok]]
                ok &= ids >= 0
                rows_l.append(np.nonzero(ok)[0])
                cols_l.append(ids[ok])
    rows = np.concatenate(rows_l).astype(np.int64)
    cols = np.concatenate(cols_l).astype(np.int64)
    key = np.unique(rows * M + cols)
    return (key // M).astype(np.int64), (key % M).astype(np.int64)


def lower_bound(P: np.ndarray, grid: Grid, v: np.ndarray, cell: np.ndarray, eps: float = 0.0, max_viol: int = 0,
                block: int = 16, threads: int = 0):
    """Global Lagrangian bound for potentials ``v`` (see module docstring)."""
    return _core.dual_bound(np.ascontiguousarray(P, dtype=np.float64), grid.index, grid.centers,
                            np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(cell, dtype=np.int32),
                            float(eps), int(max_viol), int(block), int(threads))


def _mcf(N, M, rows, cols, cint):
    from ortools.graph.python import min_cost_flow

    mcf = min_cost_flow.SimpleMinCostFlow()
    sink = N + M
    tails = np.concatenate([rows, N + np.arange(M, dtype=np.int64)])
    heads = np.concatenate([N + cols, np.full(M, sink, dtype=np.int64)])
    costs = np.concatenate([cint, np.zeros(M, dtype=np.int64)])
    caps = np.ones(len(tails), dtype=np.int64)
    mcf.add_arcs_with_capacity_and_unit_cost(tails, heads, caps, costs)
    sup = np.zeros(N + M + 1, dtype=np.int64)
    sup[:N] = 1
    sup[sink] = -N
    mcf.set_nodes_supplies(np.arange(N + M + 1), sup)
    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError(f"min cost flow failed with status {status}")
    f = mcf.flows(np.arange(len(rows)))
    sel = np.nonzero(f)[0]
    cell = np.full(N, -1, dtype=np.int64)
    cell[rows[sel]] = cols[sel]
    if (cell < 0).any():
        raise RuntimeError("min cost flow returned an incomplete matching")
    return cell


def _neighborhood_edges(grid: Grid, rows: np.ndarray, cells: np.ndarray, radius: int):
    """Edges from each point in `rows` to the lattice neighbourhood of the matching cell."""
    lat = grid.lattice[cells]
    H, W = grid.mask.shape
    rr, cc = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = lat[:, 0] + dx
            y = lat[:, 1] + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            ids = np.full(len(rows), -1, dtype=np.int64)
            ids[ok] = grid.index[y[ok], x[ok]]
            ok &= ids >= 0
            rr.append(rows[ok])
            cc.append(ids[ok])
    return np.concatenate(rr).astype(np.int64), np.concatenate(cc).astype(np.int64)


def solve_certified(P: np.ndarray, grid: Grid, cell0: np.ndarray, k: int = 12, radius: int = 2, max_viol: int = 8,
                    max_rounds: int = 50, rel_gap: float = 1e-9, threads: int = 0, block: int = 16,
                    verbose: bool = False, time_limit: float | None = None, expand: int = 1) -> dict:
    """Exact assignment of normalized points ``P`` to ``grid`` with an optimality certificate.

    ``cell0`` is any feasible starting assignment (it seeds the candidate edges).  After
    each pricing pass, violating edges are added together with the ``expand``-radius
    lattice neighbourhood of every violating point's best-priced cell.
    Returns a dict with ``cell``, ``cost``, ``lower_bound``, ``gap`` (= cost - bound),
    ``certified`` (gap within tolerance), per-round statistics and timings.
    """
    t0 = time.perf_counter()
    P = np.ascontiguousarray(P, dtype=np.float64)
    N, M = len(P), grid.n_cells
    rows, cols = candidate_edges(P, grid, np.asarray(cell0), k=k, radius=radius)
    stats = []
    result = None
    for rnd in range(max_rounds):
        tr = time.perf_counter()
        creal = _sqdist(P, grid.centers, rows, cols)
        cmax = max(float(creal.max()), 1.0)
        # keep |cost| * (#nodes) well inside int64 for the cost-scaling solver
        S = float(2.0 ** 56 / ((N + M + 2) * cmax))
        S = min(S, 1e12)
        cint = np.rint(creal * S).astype(np.int64)
        cell = _mcf(N, M, rows, cols, cint)
        t_mcf = time.perf_counter() - tr
        # integer potentials on the cell graph: arc cell(i) -> j, weight c(i,j) - c(i,cell(i))
        own = np.empty(N, dtype=np.int64)
        m_own = cols == cell[rows]
        own[rows[m_own]] = cint[m_own]
        other = ~m_own
        tail = cell[rows[other]].astype(np.int32)
        head = cols[other].astype(np.int32)
        w = cint[other] - own[rows[other]]
        ts = time.perf_counter()
        ok, d = _core.spfa(M, tail, head, w)
        t_spfa = time.perf_counter() - ts
        if not ok:
            raise RuntimeError("negative cycle in the potential graph (solution not optimal on candidates)")
        v = d.astype(np.float64) / S
        tb = time.perf_counter()
        eps = 4.0 / S
        lb, u, arg, vi, vj = lower_bound(P, grid, v, cell, eps=eps, max_viol=max_viol, block=block, threads=threads)
        t_lb = time.perf_counter() - tb
        cost = float(_core.assignment_cost(P, grid.centers, cell.astype(np.int32), threads))
        gap = cost - lb
        tol = max(rel_gap * cost, N * 4.0 / S)
        st = dict(round=rnd, edges=int(len(rows)), cost=cost, lower_bound=float(lb), gap=float(gap),
                  violations=int(len(vi)), scale=S, t_mcf=t_mcf, t_spfa=t_spfa, t_bound=t_lb,
                  t_round=time.perf_counter() - tr)
        stats.append(st)
        if verbose:
            print(st, flush=True)
        result = dict(cell=cell.astype(np.int32), cost=cost, lower_bound=float(lb), gap=float(gap),
                      certified=bool(gap <= tol and len(vi) == 0) or bool(gap <= tol), tolerance=tol,
                      rounds=stats)
        if gap <= tol:
            break
        if len(vi) == 0:
            # bound not tight although no edge violates by more than eps: tighten eps
            break
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            break
        add_r, add_c = vi.astype(np.int64), vj.astype(np.int64)
        if expand > 0:
            # also offer each violating point the neighbourhood of its best-priced cell
            vp = np.unique(add_r)
            er, ec = _neighborhood_edges(grid, vp, arg[vp], expand)
            add_r = np.concatenate([add_r, er])
            add_c = np.concatenate([add_c, ec])
        key = np.unique(np.concatenate([rows * M + cols, add_r * M + add_c]))
        rows, cols = key // M, key % M
    result["time_s"] = time.perf_counter() - t0
    return result


def solve_dense(P: np.ndarray, grid: Grid, method: str = "scipy") -> np.ndarray:
    """Exact assignment through a dense N x M cost matrix (memory ~ 8 N M bytes)."""
    C = grid.centers
    D = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    if method == "scipy":
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(D)
        cell = np.empty(len(P), dtype=np.int32)
        cell[r] = c
        return cell
    if method == "lap":
        import lap

        _, x, _ = lap.lapjv(D, extend_cost=D.shape[0] != D.shape[1])
        return x.astype(np.int32)
    raise ValueError("method must be 'scipy' or 'lap'")
