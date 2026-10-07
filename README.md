# ShuffleSnap

ShuffleSnap assigns 2D points (for example a UMAP or t-SNE layout) one-to-one to the
cells of a grid, minimizing the total squared distance each point moves. It is meant
for 50k–1M points, where a dense exact linear assignment does not fit in memory.

```python
import numpy as np
import shufflesnap as ss

xy = np.random.default_rng(0).normal(size=(100_000, 2))  # your embedding
res = ss.assign(xy)                  # near-square grid with exactly N cells
res.xy                               # (N, 2) integer column/row of every point
res.cost / len(xy)                   # mean squared displacement, in cell units
```

## How it works

The algorithm is a coarse-to-fine local search, loosely inspired by Shellsort:

1. **Random start.** Points are placed in random cells (a seeded permutation).
2. **Strided windows.** At stride *s* the lattice splits into *s × s* interleaved
   subgrids, one per row/column residue mod *s*. Each subgrid is tiled with *w × w*
   windows (default *w* = 8). A window holds at most *w²*
   cells spaced *s* apart, so it spans (*w*−1)*s*+1 cells per axis.
3. **Exact window solves.** Inside every window, the current occupants are reassigned
   among the window's cells by an exact dense linear assignment on squared distances to
   the original coordinates. A window's new assignment is accepted only if it lowers the
   cost by more than a relative 10⁻¹², so the total cost never increases. The default
   window solver (`solver="geometric"`) starts from dual potentials computed in closed form
   from per-axis optimal transport between the window's points and cells, then finishes
   exactly with shortest augmenting paths; on clustered data this is about 5× faster than a
   Hungarian solver with a greedy start, with the same results.
4. **Shifted tilings.** A round consists of four tilings, shifted by (0,0), (0,*w*/2),
   (*w*/2,0), and (*w*/2,*w*/2) subgrid cells. The windows within one tiling are disjoint
   and are solved in parallel.
5. **Schedule.** Per axis of length *L*, the start stride is the smallest power of two
   with *w·s ≥ L*. After one round, strides are divided by √2 and rounded (any ratio can
   be set; 2 halves them), until both reach 1. Stride-1 rounds then
   repeat until a full round makes no move, or until the time budget runs out.

**Boundary rules.** Windows that extend past a subgrid's edge are clipped, never dropped,
on every side. So every lattice position belongs to exactly one window of every tiling,
and the lower-right boundary cells exchange with the interior just like the upper-left
ones. Windows with fewer than two usable cells are skipped, since nothing can move in
them.

**Rectangular grids.** Each axis has its own start stride. Both strides shrink on every
level; the shorter axis reaches 1 first and stays there.

**Masks and surplus cells.** A `Grid` is a boolean mask on a W × H lattice. Masked cells
never enter a window. Empty (unmasked, unoccupied) cells do take part, so points can move
into them. With more cells than points, the solver therefore also decides which cells
stay empty.

**Skipping unchanged windows.** A window whose cells have not changed occupant since it
was last solved in the same stage is already optimal and is skipped. This does not change
the result, and the result is also independent of the number of threads (both are tested).

## Presets

| preset | window | stride ratio | notes |
|---|---|---|---|
| `fast` | 5 | √2 | about the time of BSP-OT with 16 plans |
| `balanced` (default) | 8 | √2 | about the time of BSP-OT with 64 plans |
| `quality` | 12 | √2 | 144-cell exact window solves |

The presets were chosen on development instances (not the benchmark instances) so that
`fast` and `balanced` take about as long as BSP-OT with 16 and 64 plans (one core,
100,000 points), which makes the two methods comparable at equal time. Larger windows are
better in every test but slower; ratio √2 was best or close to best at every window size
from 6 to 24; windows up to 32 can be set with `window=`.

Explicit arguments (`window=`, `ratio=`, `schedule=`, `rounds_per_stride=`,
`start_stride=`, `init=`) override the preset.

## Measured performance

Held-out instances, 100,000 points, square grid, i9-12900KS. The table gives excess over a
**certified** optimum (mean of three instances), with time on one core (24 threads in
parentheses). The full benchmark, including failures, is in the paper repository.

| method | uniform | Gaussian mixture | spiral | checkerboard | time |
|---|---|---|---|---|---|
| `fast` | 2.6% | 0.013% | 0.022% | 0.13% | 0.77 s (0.09 s) |
| BSP-OT (16 plans) | 45% | 5.6% | 10% | 50% | 0.71 s (0.13 s) |
| `balanced` (default) | 0.84% | 0.002% | 0.004% | 0.057% | 2.3 s (0.19 s) |
| BSP-OT (64 plans) | 32% | 3.7% | 8.1% | 36% | 2.9 s (0.41 s) |
| `quality` | 0.44% | <0.001% | 0.001% | 0.020% | 6.7 s (0.53 s) |
| RasterFairy (upstream) | 45% | 17% | 27% | 425% | 3.0 s |
| certified exact (`shufflesnap.exact`) | 0 | 0 | 0 | 0 | median 117 s |

At equal time, `fast` and `balanced` are closer to the optimum than BSP-OT with 16 and 64
plans on 8 and 9 of the 10 benchmark distributions (BSP-OT wins on a near-line and on two
clusters with 90/10 mass). On the full MNIST and Fashion-MNIST UMAP embeddings (70,000 points
each), `balanced` is within 0.002% of the optimum. Where ShuffleSnap is weakest:

- near-uniform data (0.84% at 100k points, growing with N to 1.3% at 1M);
- grids with surplus cells (3.1% for uniform points with less than a row of surplus);
- masks with concave bays, where coarse schedules route mass across the bays (uniform
  points on a blocky S: `balanced` 1.4%, BSP-OT with 64 plans 9.3%).

## Grids

```python
ss.Grid.rectangle(400, 250)                  # full rectangle
ss.Grid.for_count(99991)                     # near-square, last row partially filled (exactly N cells)
ss.Grid.for_count(99991, partial="free")     # full rectangle; solver picks the empty cells
ss.Grid.for_count(50_000, aspect=16 / 9)
ss.Grid.disk(50_000)                         # disk with exactly N cells
ss.Grid.from_mask(mask_bool_array)           # arbitrary shape (True = usable)
ss.Grid.from_image(img, 50_000)              # dark pixels of an image, exactly N cells
ss.Grid.for_count(n, pitch=(4, 3))           # anisotropic cells, e.g. 4:3 thumbnails
```

## Normalization

Costs are measured in the grid frame (one cell pitch = one unit). By default
(`normalize="bbox"`), each axis of the point bounding box is mapped onto the bounding
box of the usable cell centers. `"fit"` preserves the aspect ratio. `"none"` means the
points are already in grid units.

## Certified exact solutions (optional)

`pip install shufflesnap[exact]` adds `shufflesnap.exact.solve_certified`. It never
builds the dense N × M matrix; instead it alternates three steps until a Lagrangian
lower bound computed over **all** cells meets the primal cost:

1. an exact min-cost-flow solve (OR-Tools) on sparse candidate edges;
2. integer shortest-path cell potentials;
3. a global pricing pass that adds any violating edges.

```python
res = ss.assign(xy, exact=True)      # ShuffleSnap, then the certified exact solve
res.cost, res.lower_bound, res.gap, res.certified
```

The ShuffleSnap result only chooses the candidate edges (OR-Tools has no warm start), but
it is a good choice: the first sparse solve is typically already within 0.001% of the
optimum, which makes this much faster than starting OR-Tools from a random or bisection
assignment. It can still take far longer than ShuffleSnap itself (seconds at 10k points,
up to tens of minutes at 100k); `exact_time_limit=` returns the best solution so far with
`certified=False`. The same solve is available directly:

```python
from shufflesnap import exact
r = exact.solve_certified(res.points, res.grid, res.cell)
r["cost"], r["lower_bound"], r["gap"], r["certified"]
```

## Known limitations

The method is a local search. Its remaining excess over the optimum is dominated by a
few large closed loops of points, each shifted by about one cell. No window smaller than
such a loop can remove it. Structured starts (bisection, row sorting) create seams that
make results worse, which is why the start is random. Masks with narrow features or
concave bays, and heavily surplus grids, are the hardest cases. See the accompanying
paper for measurements, including failure cases.

A mask guarantees a valid layout, not that every pair of usable cells can trade
occupants. Two cells exchange only if some scheduled window contains both. On a one-row
mask whose only usable columns are 0 and 9, stride 2 separates them and an 8-wide
stride-1 window never holds both, so a swapped pair stays swapped under `balanced`
(`quality`'s 12-wide windows repair this example, not the general problem).

## Reference implementation

`shufflesnap.reference.snap(points, mask, w=8, ratio=2**0.5, seed=0)` is a serial
implementation of the same algorithm in about 70 lines of NumPy and SciPy, meant for
reading. On inputs without exact ties it returns the same assignment as `assign` with
`normalize="none"` and the matching preset (tested for all presets on rectangles,
partial rows, surplus cells, disks and random masks). It is slow: use it to understand
or check the method, not to run it.

## Development

```bash
pip install -e ".[test]"        # or ./dev_build.sh for an in-place build
pytest -q
```

The compiled kernel is C++17 (nanobind), parallelized with OpenMP when available. On the
benchmark machine, compiling takes 2.6 s with Ninja (8.1 s on one core); an isolated
source build (`pip install .`) takes about 35 s, most of it setting up the build
environment. There is no JIT, so nothing is compiled on first call.

## License

MIT
