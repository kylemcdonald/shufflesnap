# shufflesnap

`shufflesnap` assigns large 2D point clouds to regular grids: every point gets its own grid cell, and the solver seeks a small total squared movement using multiscale local descent. It is a Python package with a native C++ core and `nanobind` bindings. The main use case is turning embeddings (UMAP, t-SNE, Isomap) into image atlases.

Instead of building an `n x n` cost matrix, `shufflesnap` refines a trivial legal assignment by solving exact linear assignment problems inside small grid windows at multiple scales — coarse-to-fine strided windows repair global structure in a handful of rounds, then stride-1 windows polish. The assignment is legal after every round, the cost never increases, memory is linear in the number of points plus grid cells, and random initialization works well on the tested distributions. Final accuracy and the number of polishing rounds have no worst-case guarantee.

The public API has three functions:

1. `snap_to_grid(points, width=None, height=None, cleanup_seconds=None, ...)`
2. `window_cleanup(points, initial_assignment, rows, cols, budget_seconds=None, ...)`
3. `linear_sum_assignment(cost_matrix)`

## Install

```bash
python -m pip install shufflesnap
```

To run the matplotlib example from the source tree:

```bash
python -m pip install -e '.[examples]'
python examples/basic_usage.py
```

## API

### `snap_to_grid(points, width=None, height=None, cleanup_seconds=None, ...)`

High-level wrapper for snapping a 2D point cloud onto a destination grid. Inputs must use the same coordinate system as the target grid (range `[0, 1]` per axis); the API does not normalize inputs.

Behavior:

- chooses a destination grid automatically when `width` and `height` are omitted: an exact factorization of `n` with aspect ratio in `[1:1, 2:1]` when one exists, otherwise a slightly larger near-square grid
- supports any `n <= width * height` directly: the occupied cell subset is selected before cleanup and remains fixed
- by default cleanup runs until none of the four scheduled half-offset tilings can improve the assignment; `cleanup_seconds` caps the time instead, and `0.0` returns the raw seed (a deterministic random permutation, with a fixed random cell subset when the grid has more cells than points)
- `polish_all_offsets=True` adds one stride-1 sweep over every complete window placement after normal cleanup, reducing the remaining local error at additional cost
- pass `mask` (a `(height, width)` bool array) to restrict which cells may be used — shaped atlases (circles, cut corners, a half-empty last row) work out of the box

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
- `all_offsets=True` uses all `window_size ** 2` tiling offsets instead of the default four and is substantially slower
- `num_threads=None` to use `std::thread::hardware_concurrency()`
- `fixed_suffix_count` to keep a suffix of target cells fixed; `cell_mask` to mark which cells may be used at all
- `trace_rounds=True` to record per-round cost, elapsed time, and stride

Returns a dict with `assignment`, `rounds_completed`, `elapsed_s`, `final_cost`, and `converged`.

### `linear_sum_assignment(cost_matrix)`

Solve a dense square linear assignment problem exactly with the native C++ Jonker-Volgenant implementation.

Returns:

- `row_ind`: `int64` NumPy array of shape `(n,)`
- `col_ind`: `int64` NumPy array of shape `(n,)`
- `total_cost`: Python `float`

## More

- Source repository: https://github.com/kylemcdonald/shufflesnap
- Issue tracker: https://github.com/kylemcdonald/shufflesnap/issues
- Example scripts: https://github.com/kylemcdonald/shufflesnap/tree/main/examples

### Coordinate convention (next release)

Grid centers span `[0, 1]` on each axis; a one-cell axis is centered at `0.5`.
The `margin` argument has been removed. Normalize input coordinates explicitly
when needed, and add visual padding when rendering. The API does not normalize
inputs. PyPI 0.3.0 retains the previous inset; this source change is unreleased.
