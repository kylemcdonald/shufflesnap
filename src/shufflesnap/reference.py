"""Serial reference implementation of ShuffleSnap, in NumPy and SciPy only.

Written to be read: no threads, no skipped windows, no time budget, unit cell pitch.
On inputs without exact ties it returns the same assignment as
``shufflesnap.assign(points, grid, normalize="none", ...)`` (tested); with ties the
two can pick different optimal window solutions.  ``snap(P, mask)`` is the default
(balanced) preset; ``w=5`` and ``w=12`` give the fast and quality presets.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def strides(length, w, ratio):
    """From the smallest power of two s with w*s >= length, divide by ratio, round."""
    s = 1
    while w * s < length:
        s *= 2
    out, x = [], float(s)
    while x > 1 + 1e-9:
        out.append(max(1, int(round(x))))
        x /= ratio
    return out


def coarse_stages(width, height, w, ratio):
    """Stride pairs (sx, sy) before the stride-1 stage; the shorter axis stays at 1."""
    sx, sy = strides(width, w, ratio), strides(height, w, ratio)
    n = max(len(sx), len(sy))
    out = []
    for pair in zip(sx + [1] * (n - len(sx)), sy + [1] * (n - len(sy))):
        if pair != (1, 1) and (not out or out[-1] != pair):
            out.append(pair)
    return out


def windows(mask, sx, sy, w, ox, oy):
    """Lattice rows and columns of the cells of every window of one phase."""
    H, W = mask.shape
    for b in range(min(sy, H)):
        rows = np.arange(b, H, sy)  # one residue class of rows
        for a in range(min(sx, W)):
            cols = np.arange(a, W, sx)  # one residue class of columns
            for y0 in range(-((w - oy) % w), len(rows), w):
                for x0 in range(-((w - ox) % w), len(cols), w):
                    ys, xs = rows[max(y0, 0):y0 + w], cols[max(x0, 0):x0 + w]  # clipped
                    yy, xx = np.meshgrid(ys, xs, indexing="ij")
                    usable = mask[yy, xx]  # masked cells never enter a window
                    if usable.sum() >= 2:
                        yield yy[usable], xx[usable]


def phase(P, C, index, occ, pos, wins, tol):
    """Reassign each window's occupants exactly; return the number of changed windows."""
    changed = 0
    for yy, xx in wins:
        cells = index[yy, xx]
        held = np.nonzero(occ[cells] >= 0)[0]  # window columns that hold a point
        if len(held) == 0:
            continue
        pts = occ[cells[held]]
        cost = ((P[pts, None, :] - C[None, cells, :]) ** 2).sum(-1)
        col = linear_sum_assignment(cost)[1]
        cur = best = 0.0
        for r in range(len(pts)):  # sequential sums, as in the compiled kernel
            cur += cost[r, held[r]]
            best += cost[r, col[r]]
        if best < cur - tol * cur:  # strict improvement: the cost never increases
            occ[cells] = -1
            occ[cells[col]] = pts
            pos[pts] = cells[col]
            changed += 1
    return changed


def snap(P, mask, w=8, ratio=2 ** 0.5, seed=0, tol=1e-12):
    """Cell id (row-major over usable cells) of each point; P is (N, 2) in grid units."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    index = np.full(mask.shape, -1)
    index[ys, xs] = np.arange(len(xs))
    C = np.stack([xs, ys], axis=1).astype(float)  # cell centers (x, y)
    pos = np.random.default_rng(seed).permutation(len(C))[:len(P)]  # random start
    occ = np.full(len(C), -1)
    occ[pos] = np.arange(len(P))
    h = w // 2

    def one_round(sx, sy):  # four tilings, shifted by half a window
        return sum(phase(P, C, index, occ, pos, windows(mask, sx, sy, w, ox, oy), tol)
                   for ox, oy in [(0, 0), (0, h), (h, 0), (h, h)])

    for sx, sy in coarse_stages(mask.shape[1], mask.shape[0], w, ratio):
        one_round(sx, sy)
    while one_round(1, 1):  # stride-1 rounds until one changes nothing
        pass
    return pos
