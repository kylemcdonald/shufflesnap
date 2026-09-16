from __future__ import annotations

import numpy as np
import pytest

import shufflesnap


def _grid_targets(width: int, height: int, margin: float = 0.03) -> np.ndarray:
    xs = np.linspace(margin, 1.0 - margin, width) if width > 1 else np.array([0.5])
    ys = np.linspace(margin, 1.0 - margin, height) if height > 1 else np.array([0.5])
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


def _assignment_cost(points: np.ndarray, targets: np.ndarray, assignment: np.ndarray) -> float:
    d = points - targets[assignment]
    return float((d * d).sum())


def test_linear_sum_assignment_finds_known_optimum() -> None:
    cost = np.array(
        [
            [4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    row_ind, col_ind, total_cost = shufflesnap.linear_sum_assignment(cost)

    assert row_ind.tolist() == [0, 1, 2]
    assert col_ind.tolist() == [1, 0, 2]
    assert total_cost == pytest.approx(5.0)


def test_linear_sum_assignment_matches_brute_force() -> None:
    from itertools import permutations

    rng = np.random.default_rng(0)
    cost = rng.random((6, 6))
    _, col_ind, total_cost = shufflesnap.linear_sum_assignment(cost)
    best = min(sum(cost[i, p[i]] for i in range(6)) for p in permutations(range(6)))
    assert total_cost == pytest.approx(best)
    assert sorted(col_ind.tolist()) == list(range(6))


def test_snap_to_grid_returns_unique_assignment_for_nonfactorable_count() -> None:
    points = np.array(
        [
            [0.10, 0.20],
            [0.85, 0.15],
            [0.20, 0.80],
            [0.75, 0.70],
            [0.50, 0.45],
        ],
        dtype=np.float64,
    )

    grid_points, assignment, grid_size = shufflesnap.snap_to_grid(points, cleanup_seconds=0.0)

    assert grid_points.shape == points.shape
    assert assignment.shape == (points.shape[0],)
    assert len(set(assignment.tolist())) == points.shape[0]
    assert grid_size[0] * grid_size[1] >= points.shape[0]


def test_snap_to_grid_near_optimal_on_small_cloud() -> None:
    rng = np.random.default_rng(1)
    n = 20 * 20
    points = 0.03 + 0.94 * rng.random((n, 2))

    grid_points, assignment, (width, height) = shufflesnap.snap_to_grid(points, width=20, height=20)

    assert sorted(assignment.tolist()) == list(range(n))

    from scipy.optimize import linear_sum_assignment as scipy_lsa

    targets = _grid_targets(width, height)
    cost = ((points[:, None, :] - targets[None, :, :]) ** 2).sum(-1)
    ri, ci = scipy_lsa(cost)
    optimal = float(cost[ri, ci].sum())
    achieved = _assignment_cost(points, targets, assignment)
    # uniform clouds at this size have a near-zero optimum, which inflates
    # the relative gap; a few percent is still window-optimal
    assert achieved <= optimal * 1.03


def test_snap_to_grid_with_spare_cells_keeps_occupied_subset_fixed() -> None:
    rng = np.random.default_rng(2)
    n = 700
    points = 0.03 + 0.94 * rng.random((n, 2))

    grid_points, assignment, (width, height) = shufflesnap.snap_to_grid(points, width=30, height=30)

    assert len(set(assignment.tolist())) == n
    assert assignment.min() >= 0
    assert assignment.max() < width * height

    targets = _grid_targets(width, height)
    cleaned = _assignment_cost(points, targets, assignment)
    _, seed_assignment, _ = shufflesnap.snap_to_grid(points, width=30, height=30, cleanup_seconds=0.0)
    seeded = _assignment_cost(points, targets, seed_assignment)
    assert set(assignment.tolist()) == set(seed_assignment.tolist())
    assert cleaned < seeded


def test_window_cleanup_converges_from_random_permutation() -> None:
    rng = np.random.default_rng(3)
    rows = cols = 24
    n = rows * cols
    points = 0.03 + 0.94 * rng.random((n, 2))
    initial = rng.permutation(n).astype(np.int64)

    result = shufflesnap.window_cleanup(points, initial, rows=rows, cols=cols)

    assert result["converged"]
    assert sorted(result["assignment"].tolist()) == list(range(n))

    from scipy.optimize import linear_sum_assignment as scipy_lsa

    targets = _grid_targets(cols, rows)
    cost = ((points[:, None, :] - targets[None, :, :]) ** 2).sum(-1)
    ri, ci = scipy_lsa(cost)
    optimal = float(cost[ri, ci].sum())
    assert result["final_cost"] <= optimal * 1.01


def test_window_cleanup_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    rows = cols = 16
    n = rows * cols
    points = 0.03 + 0.94 * rng.random((n, 2))
    initial = rng.permutation(n).astype(np.int64)

    first = shufflesnap.window_cleanup(points, initial, rows=rows, cols=cols, num_threads=4)
    second = shufflesnap.window_cleanup(points, initial, rows=rows, cols=cols, num_threads=1)

    assert first["assignment"].tolist() == second["assignment"].tolist()


def test_window_cleanup_trace_rounds_monotone() -> None:
    rng = np.random.default_rng(5)
    rows = cols = 24
    n = rows * cols
    points = 0.03 + 0.94 * rng.random((n, 2))
    initial = rng.permutation(n).astype(np.int64)

    result = shufflesnap.window_cleanup(
        points, initial, rows=rows, cols=cols, trace_rounds=True
    )

    costs = result["round_costs"]
    assert len(costs) == result["rounds_completed"]
    assert np.all(np.diff(costs) <= 1e-12)
    assert result["round_strides"][0] > result["round_strides"][-1]


def test_window_cleanup_respects_fixed_suffix() -> None:
    rng = np.random.default_rng(6)
    rows = cols = 8
    n = rows * cols
    points = 0.03 + 0.94 * rng.random((n, 2))
    initial = np.arange(n, dtype=np.int64)
    fixed = 8

    result = shufflesnap.window_cleanup(
        points, initial, rows=rows, cols=cols, fixed_suffix_count=fixed
    )

    assignment = result["assignment"]
    for point_id in range(n):
        if initial[point_id] >= n - fixed:
            assert assignment[point_id] == initial[point_id]


def test_window_cleanup_rejects_duplicate_assignment() -> None:
    points = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    initial = np.array([0, 0], dtype=np.int64)

    with pytest.raises(RuntimeError):
        shufflesnap.window_cleanup(points, initial, rows=1, cols=2, budget_seconds=0.0)


def test_window_cleanup_zero_budget_runs_one_round() -> None:
    points = np.array(
        [
            [0.03, 0.03],
            [0.97, 0.03],
            [0.03, 0.97],
            [0.97, 0.97],
        ],
        dtype=np.float64,
    )
    initial_assignment = np.arange(4, dtype=np.int64)

    result = shufflesnap.window_cleanup(
        points,
        initial_assignment,
        rows=2,
        cols=2,
        budget_seconds=0.0,
        num_threads=1,
    )

    assert result["assignment"].dtype == np.int64
    assert sorted(result["assignment"].tolist()) == [0, 1, 2, 3]
    assert result["rounds_completed"] >= 1
    assert result["elapsed_s"] >= 0.0


def test_window_cleanup_converges_with_duplicate_points() -> None:
    # Tied window optima must not flip forever: duplicate points make ties
    # ubiquitous, and budget_seconds=None must still terminate.
    rng = np.random.default_rng(7)
    base = rng.integers(0, 2, size=(24, 2)).astype(np.float64) * 0.9 + 0.05
    initial = rng.permutation(36)[:24].astype(np.int64)

    result = shufflesnap.window_cleanup(base, initial, rows=6, cols=6)

    assert result["converged"]
    assert len(set(result["assignment"].tolist())) == 24
    assert set(result["assignment"].tolist()) == set(initial.tolist())


def test_window_cleanup_infinite_budget_requires_stride_one_schedule() -> None:
    points = np.random.default_rng(8).random((16, 2))
    initial = np.arange(16, dtype=np.int64)

    with pytest.raises(ValueError):
        shufflesnap.window_cleanup(points, initial, rows=4, cols=4, strides=2)

    result = shufflesnap.window_cleanup(points, initial, rows=4, cols=4,
                                    strides=2, budget_seconds=0.05)
    assert sorted(result["assignment"].tolist()) == list(range(16))


def test_window_cleanup_accepts_numpy_integer_strides() -> None:
    points = np.random.default_rng(9).random((16, 2))
    initial = np.arange(16, dtype=np.int64)

    result = shufflesnap.window_cleanup(
        points, initial, rows=4, cols=4, strides=np.array([2, 1]),
    )
    assert result["converged"]


def test_window_cleanup_trace_costs_never_increase_with_ties() -> None:
    rng = np.random.default_rng(10)
    base = np.repeat(rng.random((18, 2)), 2, axis=0)  # every point duplicated
    initial = rng.permutation(36).astype(np.int64)

    result = shufflesnap.window_cleanup(base, initial, rows=6, cols=6, trace_rounds=True)

    assert result["converged"]
    assert np.all(np.diff(result["round_costs"]) <= 0.0)


def test_snap_to_grid_seed_is_random_permutation() -> None:
    rng = np.random.default_rng(14)
    n = 12 * 12
    points = 0.03 + 0.94 * rng.random((n, 2))

    _, seed_full, _ = shufflesnap.snap_to_grid(points, width=12, height=12,
                                           cleanup_seconds=0.0)
    assert sorted(seed_full.tolist()) == list(range(n))
    # must not resemble the raster-sorted seed
    order = np.lexsort((points[:, 0], points[:, 1]))
    raster = np.empty(n, dtype=np.int64)
    raster[order] = np.arange(n)
    assert (seed_full == raster).mean() < 0.1
    # deterministic across calls
    _, again, _ = shufflesnap.snap_to_grid(points, width=12, height=12,
                                       cleanup_seconds=0.0)
    assert seed_full.tolist() == again.tolist()


def test_snap_to_grid_spare_cell_seed_uses_fixed_random_subset() -> None:
    rng = np.random.default_rng(15)
    n = 100
    points = 0.03 + 0.94 * rng.random((n, 2))

    _, seed, _ = shufflesnap.snap_to_grid(points, width=12, height=12,
                                      cleanup_seconds=0.0)
    assert len(set(seed.tolist())) == n
    # The mapping is unstructured, rather than a raster ordering over the
    # preselected occupied cells.
    order = np.lexsort((points[:, 0], points[:, 1]))
    assert (np.diff(seed[order]) > 0).mean() < 0.7
    _, cleaned, _ = shufflesnap.snap_to_grid(points, width=12, height=12)
    assert set(cleaned.tolist()) == set(seed.tolist())


def test_all_offset_polish_matches_manual_single_sweep() -> None:
    rng = np.random.default_rng(16)
    side = 20
    points = 0.03 + 0.94 * rng.random((side * side, 2))

    _, base, _ = shufflesnap.snap_to_grid(points, width=side, height=side)
    _, polished, _ = shufflesnap.snap_to_grid(
        points, width=side, height=side, polish_all_offsets=True
    )
    manual = shufflesnap.window_cleanup(
        points,
        base,
        rows=side,
        cols=side,
        budget_seconds=0.0,
        strides=[1],
        all_offsets=True,
    )["assignment"]

    targets = _grid_targets(side, side)
    assert polished.tolist() == manual.tolist()
    assert _assignment_cost(points, targets, polished) <= _assignment_cost(points, targets, base)


def test_all_offset_polish_preserves_fixed_occupied_subset() -> None:
    rng = np.random.default_rng(17)
    points = 0.03 + 0.94 * rng.random((100, 2))

    _, seed, _ = shufflesnap.snap_to_grid(
        points, width=12, height=12, cleanup_seconds=0.0
    )
    _, polished, _ = shufflesnap.snap_to_grid(
        points, width=12, height=12, polish_all_offsets=True
    )

    assert set(polished.tolist()) == set(seed.tolist())


def test_all_offset_polish_rejects_raw_seed_mode() -> None:
    points = np.random.default_rng(18).random((16, 2))
    with pytest.raises(ValueError):
        shufflesnap.snap_to_grid(
            points,
            width=4,
            height=4,
            cleanup_seconds=0.0,
            polish_all_offsets=True,
        )


def test_snap_to_grid_rejects_empty_points() -> None:
    with pytest.raises(ValueError):
        shufflesnap.snap_to_grid(np.empty((0, 2)), width=4, height=4)


def test_snap_to_grid_with_circle_mask() -> None:
    rng = np.random.default_rng(11)
    side = 20
    yy, xx = np.mgrid[0:side, 0:side]
    mask = (xx - (side - 1) / 2) ** 2 + (yy - (side - 1) / 2) ** 2 <= (side / 2) ** 2
    n = int(mask.sum()) - 10
    points = 0.03 + 0.94 * rng.random((n, 2))

    grid_points, assignment, (width, height) = shufflesnap.snap_to_grid(points, mask=mask)
    _, seed, _ = shufflesnap.snap_to_grid(points, mask=mask, cleanup_seconds=0.0)

    assert (width, height) == (side, side)
    assert len(set(assignment.tolist())) == n
    assert mask.reshape(-1)[assignment].all()  # every point inside the mask
    assert set(assignment.tolist()) == set(seed.tolist())


def test_window_cleanup_rejects_assignment_outside_mask() -> None:
    points = np.random.default_rng(12).random((4, 2))
    initial = np.arange(4, dtype=np.int64)
    mask = np.ones(16, dtype=bool)
    mask[0] = False  # cell 0 is masked out but point 0 sits there

    with pytest.raises(RuntimeError):
        shufflesnap.window_cleanup(points, initial, rows=4, cols=4,
                               budget_seconds=0.0, cell_mask=mask)


def test_snap_to_grid_defaults_to_convergence() -> None:
    rng = np.random.default_rng(13)
    points = 0.03 + 0.94 * rng.random((15 * 15, 2))

    _, first, _ = shufflesnap.snap_to_grid(points, width=15, height=15)
    _, second, _ = shufflesnap.snap_to_grid(points, width=15, height=15)

    assert first.tolist() == second.tolist()  # converged runs are deterministic


def test_default_stride_schedule_shapes() -> None:
    schedule = shufflesnap.default_stride_schedule(96, 96)
    assert schedule[0] == (16, 16)
    assert schedule[-1] == (1, 1)

    anisotropic = shufflesnap.default_stride_schedule(50, 1000)
    assert anisotropic[0][0] < anisotropic[0][1]
    assert anisotropic[-1] == (1, 1)


@pytest.mark.parametrize("extent", [7, 8, 13, 14, 200])
@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("strides", [1, None])
def test_clipped_shifted_window_connects_trailing_cells(extent, vertical, strides):
    """The last unshifted band needs its clipped half-offset bridge."""
    rows, cols = (extent, 1) if vertical else (1, extent)
    points = _grid_targets(cols, rows)
    initial = np.arange(extent, dtype=np.int64)
    boundary = 6 * (extent // 6)
    initial[boundary - 1], initial[boundary] = initial[boundary], initial[boundary - 1]
    kwargs = {} if strides is None else {"strides": strides}
    result = shufflesnap.window_cleanup(points, initial, rows=rows, cols=cols, **kwargs)
    np.testing.assert_array_equal(result["assignment"], np.arange(extent))
