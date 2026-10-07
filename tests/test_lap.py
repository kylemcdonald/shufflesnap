"""The window solver must return an optimal injective assignment (checked independently)."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from shufflesnap import _core


def brute_force(A):
    k, m = A.shape
    best = np.inf
    for cols in itertools.permutations(range(m), k):
        best = min(best, A[np.arange(k), list(cols)].sum())
    return best


@pytest.mark.parametrize("solver", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_small(seed, solver):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        k = int(rng.integers(0, 6))
        m = int(rng.integers(max(k, 1), 7))
        A = rng.random((k, m)) * rng.choice([1e-3, 1.0, 1e4])
        if rng.random() < 0.4:
            A = np.round(A * 3) / 3  # many ties
        ans, tot = _core.solve_lap(np.ascontiguousarray(A), solver)
        assert len(ans) == k and len(set(ans)) == k
        assert all(0 <= c < m for c in ans)
        assert np.isclose(tot, A[np.arange(k), ans].sum(), rtol=1e-12, atol=1e-12)
        assert np.isclose(tot, brute_force(A), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("solver", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(36, 36), (20, 36), (1, 36), (64, 64), (5, 64), (36, 40), (144, 144), (100, 144)])
def test_matches_scipy(shape, solver):
    rng = np.random.default_rng(sum(shape))
    for trial in range(100):
        A = rng.random(shape) * 100
        if trial % 3 == 0:
            A = np.floor(A / 10)  # heavy ties
        if trial % 5 == 0:  # squared distances like the real use case
            P = rng.random((shape[0], 2)) * 6
            Q = np.stack(np.meshgrid(np.arange(12), np.arange(12)), -1).reshape(-1, 2)[: shape[1]]
            A = ((P[:, None] - Q[None]) ** 2).sum(-1)
        ans, tot = _core.solve_lap(np.ascontiguousarray(A), solver)
        r, c = linear_sum_assignment(A)
        assert len(set(ans)) == shape[0]
        assert np.isclose(tot, A[r, c].sum(), rtol=1e-12, atol=1e-9)


def test_rejects_more_rows_than_columns():
    with pytest.raises(ValueError):
        _core.solve_lap(np.zeros((3, 2)))


def test_rejects_unknown_solver():
    with pytest.raises(ValueError):
        _core.solve_lap(np.zeros((2, 2)), 5)


@pytest.mark.parametrize("solver", [2, 4])
def test_window_geometry_matches_scipy(solver):
    """Windows as the kernel sees them, including the clustered and rectangular cases the
    geometric start is built for, and degenerate geometry (coincident points, one row)."""
    rng = np.random.default_rng(7)
    for trial in range(400):
        w = int(rng.integers(1, 9))
        h = int(rng.integers(1, 9))
        s = float(rng.choice([1, 3, 16]))
        C = np.stack(np.meshgrid(np.arange(w) * s, np.arange(h) * s), -1).reshape(-1, 2).astype(float)
        C += rng.integers(0, 1000)
        m = len(C)
        k = m if trial % 3 else int(rng.integers(1, m + 1))
        kind = trial % 5
        if kind == 0:  # near their cells
            P = C[rng.permutation(m)[:k]] + rng.normal(0, s, (k, 2))
        elif kind == 1:  # a tight cluster far away
            P = C.mean(0) + rng.normal(0, 1, 2) * 500 + rng.normal(0, 0.3, (k, 2))
        elif kind == 2:  # coincident points
            P = np.repeat(C[:1] + 5.0, k, axis=0)
        elif kind == 3:  # elongated diagonal cluster
            t = rng.normal(0, 1, k)
            P = C.mean(0) + 40 + np.stack([t * 10, t * 10 + rng.normal(0, 0.5, k)], 1)
        else:  # lattice-aligned ties
            P = np.round(C.mean(0) + rng.normal(0, 2 * s, (k, 2)))
        P = np.ascontiguousarray(P)
        ans, tot = _core.solve_window(P, np.ascontiguousarray(C), solver)
        A = ((P[:, None] - C[None]) ** 2).sum(-1)
        r, c = linear_sum_assignment(A)
        assert len(set(ans)) == k and all(0 <= a < m for a in ans)
        assert np.isclose(tot, A[np.arange(k), ans].sum(), rtol=1e-12, atol=1e-9)
        assert np.isclose(tot, A[r, c].sum(), rtol=1e-10, atol=1e-7)
