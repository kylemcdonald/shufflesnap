# megalap

`megalap` assigns large 2D point clouds to regular grids: every point gets its own grid cell, and the total squared movement is nearly minimal. It is a Python package with a native C++ core and `nanobind` bindings. The main use case is turning embeddings (UMAP, t-SNE, Isomap) into image atlases.

Instead of building an `n x n` cost matrix, `megalap` refines a trivial legal assignment by solving exact linear assignment problems inside small grid windows at multiple scales — coarse-to-fine strided windows repair global structure in a handful of rounds, then stride-1 windows polish. The assignment is legal after every round, the cost never increases, memory stays linear in `n`, and even a random initial permutation converges to a fraction of a percent above the exact optimum.

The public API has three functions:

1. `snap_to_grid(points, width=None, height=None, cleanup_seconds=None, ...)`
2. `window_cleanup(points, initial_assignment, rows, cols, budget_seconds=None, ...)`
3. `linear_sum_assignment(cost_matrix)`

## Install

```bash
python -m pip install megalap
```

To run the matplotlib example from the source tree:

```bash
python -m pip install -e '.[examples]'
python examples/basic_usage.py
```

## API

### `snap_to_grid(points, width=None, height=None, cleanup_seconds=None, ...)`

High-level wrapper for snapping a 2D point cloud onto a destination grid.

Behavior:

- chooses a destination grid automatically when `width` and `height` are omitted: an exact factorization of `n` with aspect ratio in `[1:1, 2:1]` when one exists, otherwise a slightly larger near-square grid
- supports any `n <= width * height` directly: leftover cells stay empty and drift toward the sparsest parts of the cloud during cleanup (no padding, no ghost points)
- `cleanup_seconds` caps the cleanup budget (default `10.0`); cleanup stops early once converged; `0.0` returns the raw seed

Returns:

- `grid_points`: `(n, 2)` float64 NumPy array of assigned destination points in original point order
- `assignment`: `(n,)` int64 NumPy array of destination-grid cell ids in original point order
- `(width, height)`: destination-grid size tuple

### `window_cleanup(points, initial_assignment, rows, cols, budget_seconds=None, ...)`

Improve any legal assignment with the native multiscale window cleanup kernel.

Key options:

- `budget_seconds=None` runs until converged (a full stride-1 round changes nothing)
- `strides=None` uses the automatic coarse-to-fine schedule; pass a list to override
- `window_size=6`
- `num_threads=None` to use `std::thread::hardware_concurrency()`
- `fixed_suffix_count` to keep a suffix of target cells fixed
- `trace_rounds=True` to record per-round cost, elapsed time, and stride

Returns a dict with `assignment`, `rounds_completed`, `elapsed_s`, `final_cost`, and `converged`.

### `linear_sum_assignment(cost_matrix)`

Solve a dense square linear assignment problem exactly with the native C++ Jonker-Volgenant implementation.

Returns:

- `row_ind`: `int64` NumPy array of shape `(n,)`
- `col_ind`: `int64` NumPy array of shape `(n,)`
- `total_cost`: Python `float`

## More

- Source repository: https://github.com/kylemcdonald/megalap
- Issue tracker: https://github.com/kylemcdonald/megalap/issues
- Example scripts: https://github.com/kylemcdonald/megalap/tree/main/examples
