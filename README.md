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
   windows (default *w* = 8; the original method uses 6). A window holds at most *w²*
   cells spaced *s* apart, so it spans (*w*−1)*s*+1 cells per axis.
3. **Exact window solves.** Inside every window, the current occupants are reassigned
   among the window's cells by an exact dense linear assignment on squared distances to
   the original coordinates. A window's new assignment is accepted only if it lowers the
   cost by more than a relative 10⁻¹², so the total cost never increases.
4. **Shifted tilings.** A round consists of four tilings, shifted by (0,0), (0,*w*/2),
   (*w*/2,0), and (*w*/2,*w*/2) subgrid cells. The windows within one tiling are disjoint
   and are solved in parallel.
5. **Schedule.** Per axis of length *L*, the start stride is the smallest power of two
   with *w·s ≥ L*. After one round, strides are divided by a ratio (2 in the original
   method, √2 in the presets below) and rounded, until both reach 1. Stride-1 rounds then
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
| `baseline` | 6 | 2 (halving) | the original specification |
| `fast` | 6 | √2 | same cost per level as `baseline`, more levels |
| `balanced` (default) | 8 | √2 | |
| `quality` | 12 | √2 | 144-cell exact window solves |

Explicit arguments (`window=`, `ratio=`, `schedule=`, `rounds_per_stride=`,
`start_stride=`, `init=`) override the preset.

## Measured performance

Held-out instances, 100,000 points, square grid, i9-12900KS. The table gives excess over a
**certified** optimum (mean of three instances), with time on one core (24 threads in
parentheses). The full benchmark, including failures, is in the paper repository.

| method | uniform | Gaussian mixture | spiral | jittered lattice | time |
|---|---|---|---|---|---|
| `baseline` | 2.5% | 0.011% | 0.023% | 33% | 5.1 s (0.35 s) |
| `fast` | 1.5% | 0.008% | 0.012% | 23% | 6.9 s (0.47 s) |
| `balanced` | 0.84% | 0.002% | 0.004% | 13% | 16 s (1.1 s) |
| `quality` | 0.44% | <0.001% | 0.001% | 7.4% | 52 s (3.9 s) |
| BSP-OT (64 plans) | 32% | 3.7% | 8.1% | 46% | 2.9 s (0.42 s) |
| RasterFairy (upstream) | 45% | 17% | 27% | 13% | 3.0 s |
| certified exact (`shufflesnap.exact`) | 0 | 0 | 0 | 0 | median 128 s |

On the full MNIST and Fashion-MNIST UMAP embeddings (70,000 points each), `balanced` is
within 0.002% of the optimum. Where ShuffleSnap is weakest:

- near-uniform data, whose excess grows with N (5.3% for `baseline` at 1M points);
- the jittered lattice;
- masks with concave bays, where `baseline` was 18% above optimal on an S shape;
- grids with surplus cells.

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

## Development

```bash
pip install -e ".[test]"        # or ./dev_build.sh for an in-place build
pytest -q
```

The compiled kernel is C++17 (nanobind), parallelized with OpenMP when available. On the
benchmark machine, compiling takes 2.6 s with Ninja (7.8 s on one core); an isolated
source build (`pip install .`) takes about 35 s, most of it setting up the build
environment. There is no JIT, so nothing is compiled on first call.

## License

MIT
