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


@pytest.mark.parametrize("solver", [0, 1, 2])
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


@pytest.mark.parametrize("solver", [0, 1, 2])
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
