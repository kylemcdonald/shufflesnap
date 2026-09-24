import numpy as np
import pytest

import shufflesnap as ss

pytest.importorskip("ortools")
from shufflesnap import exact  # noqa: E402


@pytest.mark.parametrize("dist", ["uniform", "normal", "clusters"])
@pytest.mark.parametrize("partial", ["center", "free"])
def test_certified_matches_dense(dist, partial):
    rng = np.random.default_rng(11)
    n = 700
    if dist == "uniform":
        P = rng.random((n, 2))
    elif dist == "normal":
        P = rng.normal(size=(n, 2))
    else:
        P = np.concatenate([rng.normal(size=(n // 2, 2)) * 0.1, rng.normal(size=(n - n // 2, 2)) * 0.1 + 2])
    g = ss.Grid.for_count(n, partial=partial, aspect=1.3)
    res = ss.assign(P, g, seed=0)
    ex = exact.solve_certified(res.points, g, res.cell)
    ss.validate_assignment(ex["cell"], n, g)
    dense = exact.solve_dense(res.points, g)
    opt = ss.assignment_cost(res.points, g, dense)
    assert ex["certified"]
    assert np.isclose(ex["cost"], opt, rtol=1e-9)
    assert ex["lower_bound"] <= opt * (1 + 1e-12) + 1e-9
    assert res.cost >= opt * (1 - 1e-12)


def test_lower_bound_valid_for_any_potentials():
    rng = np.random.default_rng(5)
    n = 400
    P = rng.random((n, 2))
    g = ss.Grid.for_count(n, partial="free")
    res = ss.assign(P, g, seed=0)
    opt = ss.assignment_cost(res.points, g, exact.solve_dense(res.points, g))
    for _ in range(5):
        v = rng.normal(size=g.n_cells) * 3
        lb = exact.lower_bound(res.points, g, v, res.cell)[0]
        assert lb <= opt + 1e-9


@pytest.mark.parametrize("grid", ["free", "wide", "disk", "mask"])
def test_certified_on_masks_and_free_cells(grid):
    # clustered data on masked / surplus-cell grids needs several pricing rounds; this
    # used to trip a false negative-cycle detection in the potential computation
    rng = np.random.default_rng(3)
    n = 1500
    k = 10
    means = rng.uniform(-10, 10, size=(k, 2))
    P = means[rng.integers(0, k, n)] + rng.normal(size=(n, 2)) * 0.7
    if grid == "free":
        g = ss.Grid.for_count(int(n * 1.1), partial="free")
    elif grid == "wide":
        g = ss.Grid.for_count(n, aspect=4.0)
    elif grid == "disk":
        g = ss.Grid.disk(n)
    else:
        m = np.ones((50, 50), dtype=bool)
        m[10:40, 20:30] = False
        g = ss.Grid.from_mask(m)
        n = min(n, g.n_cells)
        P = P[:n]
    res = ss.assign(P, g, seed=0)
    ex = exact.solve_certified(res.points, g, res.cell)
    opt = ss.assignment_cost(res.points, g, exact.solve_dense(res.points, g))
    assert ex["certified"]
    assert np.isclose(ex["cost"], opt, rtol=1e-9)
