# Changelog

## 0.4.0 (2026-10-06)

Complete rewrite with a new API; not compatible with 0.3.0. `snap_to_grid`,
`window_cleanup`, `default_stride_schedule` and `linear_sum_assignment` are gone; use
`shufflesnap.assign(xy, ...)`, which returns a `Result`. Releases up to 0.3.0 remain on PyPI
and their source is in the Git history (tag `v0.3.0`).

- Strided windowed exact reassignment (C++/OpenMP kernel, nanobind bindings).
- Presets `fast` (5x5 windows), `balanced` (8x8, default) and `quality` (12x12), all with
  stride ratio sqrt 2, chosen on development instances with the geometric window solver so that
  `fast` and `balanced` take about as long as BSP-OT with 16 and 64 plans. The
  former `baseline` preset (6x6 windows, halving strides) was removed; it remains available as
  `window=6, schedule="halving"`. Windows up to 32x32 can be set explicitly.
- Grids: rectangles, exact-fit partial rows, surplus cells, disks, arbitrary masks, image masks,
  anisotropic pitch.
- Optional certified exact solver (`shufflesnap.exact`, requires OR-Tools and SciPy).
- `assign(..., exact=True)` runs the certified exact solver seeded by the ShuffleSnap result
  and reports `lower_bound`, `gap` and `certified` on `Result` (`exact_time_limit` caps it).
- Window solvers (`solver=`): `geometric` (default; exact shortest augmenting paths started
  from closed-form per-axis optimal-transport duals, 2-6x faster than `hungarian_greedy`,
  which made full runs 6-8x faster at 100k points with the same results up to ties),
  `auction` (eps-scaled auction warm start, exact finish), `hungarian`, `hungarian_greedy`,
  `jv`.
- Serial NumPy/SciPy reference implementation (`shufflesnap.reference`), tested to reproduce
  the compiled kernel's assignments exactly.
- Fix: `time_budget=0` (or a budget used up by normalization and the start) ran the whole
  schedule; it now returns the start. Negative and NaN budgets are rejected.
- Test documenting that masks can disconnect exchanges between usable cells.
- `run_schedule` rejects empty stage tables, strides below 1, and stages with unlimited rounds
  and no stop rule unless a time budget bounds them (such a stage never returned).
