import numpy as np
import pytest

import shufflesnap as ss


def check(res, n):
    ss.validate_assignment(res.cell, n, res.grid)
    tr = res.trace
    costs = tr[:, 1]
    assert np.all(np.diff(costs) <= 1e-9 * np.maximum(costs[:-1], 1.0)), "cost must not increase"
    assert np.isclose(res.cost, ss.assignment_cost(res.points, res.grid, res.cell), rtol=1e-10)
    assert np.isclose(tr[-1, 1], res.cost, rtol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 36, 37, 101, 997, 5003])
def test_prime_and_small_counts(n):
    rng = np.random.default_rng(n)
    P = rng.normal(size=(n, 2))
    for partial in ("center", "start", "free"):
        g = ss.Grid.for_count(n, partial=partial)
        res = ss.assign(P, g, seed=1)
        check(res, n)
        if partial != "free":
            assert g.n_cells == n


@pytest.mark.parametrize("init", ["random", "bisect", "rowsort"])
def test_inits_valid(init):
    rng = np.random.default_rng(3)
    P = rng.random((2000, 2))
    for g in (ss.Grid.for_count(2000), ss.Grid.rectangle(50, 45), ss.Grid.disk(2000)):
        res = ss.assign(P, g, init=init)
        check(res, 2000)


def test_masks():
    rng = np.random.default_rng(1)
    mask = rng.random((60, 60)) < 0.7
    mask[20:40, 20:40] = False  # a hole
    g = ss.Grid.from_mask(mask)
    n = g.n_cells - 37
    P = rng.normal(size=(n, 2))
    res = ss.assign(P, g)
    check(res, n)


def test_thread_count_does_not_change_result():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(20000, 2)) ** 3
    g = ss.Grid.for_count(20000, partial="free", aspect=1.5)
    a = ss.assign(P, g, threads=1, seed=5)
    b = ss.assign(P, g, threads=8, seed=5)
    assert np.array_equal(a.cell, b.cell)


def test_skip_clean_is_exact():
    rng = np.random.default_rng(4)
    P = rng.random((10000, 2))
    a = ss.assign(P, seed=2, skip_clean=True)
    b = ss.assign(P, seed=2, skip_clean=False)
    assert np.array_equal(a.cell, b.cell)
    assert a.trace[:, 8].sum() < b.trace[:, 8].sum()  # fewer windows solved


def test_recovers_permuted_grid_from_sorted_start():
    # Points exactly on the lattice: the optimum is the identity with zero cost.
    # From a random start the windowed search can stop at closed loops of points shifted
    # by one cell (a known local optimum, see the paper); from a bisection start it
    # recovers the identity.
    g = ss.Grid.rectangle(37, 23)
    P = g.centers.astype(float).copy()
    res = ss.assign(P, g, normalize="none", init="bisect")
    assert res.cost == 0.0
    assert np.array_equal(res.cell, np.arange(g.n_cells))
    res = ss.assign(P, g, normalize="none", seed=0)
    assert res.cost >= 0.0


def test_duplicates_and_degenerate_axes():
    P = np.zeros((500, 2))
    res = ss.assign(P)
    check(res, 500)
    P = np.c_[np.linspace(0, 1, 500), np.zeros(500)]
    res = ss.assign(P)
    check(res, 500)
    P = np.repeat(np.random.default_rng(0).random((10, 2)), 50, axis=0)
    res = ss.assign(P)
    check(res, 500)


def test_time_budget():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(200000, 2))
    res = ss.assign(P, time_budget=0.05, threads=1)
    ss.validate_assignment(res.cell, len(P), res.grid)
    assert not res.finished
    assert res.timings["total_s"] < 0.5


def test_final_cost_not_above_start_and_windows_locally_optimal():
    # after convergence, re-running a full stride-1 round must not move anything
    rng = np.random.default_rng(7)
    P = rng.random((3000, 2))
    res = ss.assign(P, seed=3)
    pos = res.cell.copy()
    stages = np.array([[1, 1, 1, 1]], dtype=np.int32)
    tr, fin = ss.run_schedule(res.points, res.grid, pos, stages)
    assert np.array_equal(pos, res.cell)
    assert tr[1:, 10].sum() == 0


def test_anisotropic_pitch():
    rng = np.random.default_rng(0)
    P = rng.random((1200, 2))
    g = ss.Grid.for_count(1200, pitch=(4.0, 3.0))
    res = ss.assign(P, g)
    check(res, 1200)


def test_solvers_agree_on_cost_and_validity():
    rng = np.random.default_rng(9)
    P = rng.random((5000, 2))
    for g in (ss.Grid.for_count(5000), ss.Grid.for_count(5000, partial="free")):
        a = ss.assign(P, g, solver="hungarian", seed=1)
        b = ss.assign(P, g, solver="jv", seed=1)
        check(a, 5000)
        check(b, 5000)
        # both solvers are exact, so each window reaches the same optimal cost; ties may
        # be broken differently, so assignments can differ slightly in the end
        assert abs(a.cost - b.cost) <= 0.02 * a.cost


def test_presets_and_explicit_overrides():
    rng = np.random.default_rng(12)
    P = rng.normal(size=(6000, 2))
    base = ss.assign(P, preset="baseline", seed=3)
    explicit = ss.assign(P, preset="balanced", window=6, schedule="halving", seed=3)
    assert np.array_equal(base.cell, explicit.cell)
    assert base.config["stages"][0][:2] == [16, 16]
    for name in ss.PRESETS:
        r = ss.assign(P, preset=name, seed=3)
        check(r, 6000)
    with pytest.raises(ValueError):
        ss.assign(P, preset="nope")


def _rotated_ring(W, H, x0, y0, rw, rh):
    g = ss.Grid.rectangle(W, H)
    P = g.centers.astype(float).copy()  # points on cell centers: the optimum is the identity (cost 0)
    ring = ([(x, y0) for x in range(x0, x0 + rw)] + [(x0 + rw - 1, y) for y in range(y0 + 1, y0 + rh)]
            + [(x, y0 + rh - 1) for x in range(x0 + rw - 2, x0 - 1, -1)] + [(x0, y) for y in range(y0 + rh - 2, y0, -1)])
    ids = [g.index[y, x] for x, y in ring]
    pos = np.arange(g.n_cells, dtype=np.int32)
    for k in range(len(ids)):
        pos[ids[k]] = ids[(k + 1) % len(ids)]
    return g, P, pos, len(ids)


def test_rotated_ring_larger_than_windows_is_a_fixed_point():
    """Documents the central limitation: a closed loop of one-cell shifts that no window
    contains is left untouched at every stride, although the optimum has cost 0."""
    for w in (6, 8, 12):
        g, P, pos, L = _rotated_ring(64, 48, 10, 10, w + 1, w + 1)
        stages = ss.build_schedule(g.width, g.height, window=w, schedule="geometric", ratio=2 ** 0.5)
        tr, _ = ss.run_schedule(P, g, pos, stages, window=w)
        assert tr[:, 10].sum() == 0
        assert ss.assignment_cost(P, g, pos) == L
        # a ring that fits in one of the four shifted tilings is repaired
        g, P, pos, L = _rotated_ring(64, 48, 10, 10, w // 2 + 1, w // 2 + 1)
        ss.run_schedule(P, g, pos, stages, window=w)
        assert ss.assignment_cost(P, g, pos) == 0



def test_invalid_explicit_start_is_rejected():
    rng = np.random.default_rng(0)
    P = rng.random((100, 2))
    g = ss.Grid.for_count(100)
    with pytest.raises(ValueError):
        ss.assign(P, g, init=np.zeros(100, dtype=int))  # duplicates
    with pytest.raises(ValueError):
        ss.assign(P, g, init=np.arange(100) + 1)  # out of range
    pos = np.arange(100, dtype=np.int64)
    with pytest.raises(TypeError):
        ss.run_schedule(P, g, pos, ss.build_schedule(g.width, g.height))
