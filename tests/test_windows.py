"""Every lattice position lies in exactly one window of every phase (clipped windows kept)."""

import numpy as np
import pytest

import shufflesnap as ss
from shufflesnap import _core


def phase_cover(mask, sx, sy, w, ox, oy):  # coverage by the kernel's windows
    H, W = mask.shape
    grid = ss.Grid.from_mask(mask).index
    count = np.zeros((H, W), dtype=int)
    for a, b, i0, i1, j0, j1 in _core.list_windows(grid, sx, sy, w, ox, oy):
        assert 0 <= i0 < i1 <= i0 + w and 0 <= j0 < j1 <= j0 + w
        xs = a + sx * np.arange(i0, i1)
        ys = b + sy * np.arange(j0, j1)
        assert xs.max() < W and ys.max() < H
        count[np.ix_(ys, xs)] += 1
    return count


def reference_windows(mask, sx, sy, w, ox, oy):
    """Independent re-implementation of the tiling: every residue subgrid (a, b) is cut
    into w x w blocks whose grid is shifted by (ox, oy); blocks are clipped to the subgrid."""
    H, W = mask.shape
    out = []
    for b in range(min(sy, H)):
        Hb = len(range(b, H, sy))
        for a in range(min(sx, W)):
            Wa = len(range(a, W, sx))
            ystarts = sorted({((j - oy) // w) * w + oy for j in range(Hb)})
            xstarts = sorted({((i - ox) // w) * w + ox for i in range(Wa)})
            for v0 in ystarts:
                for u0 in xstarts:
                    i0, i1 = max(0, u0), min(Wa, u0 + w)
                    j0, j1 = max(0, v0), min(Hb, v0 + w)
                    xs = a + sx * np.arange(i0, i1)
                    ys = b + sy * np.arange(j0, j1)
                    out.append(((a, b, i0, i1, j0, j1), int(mask[np.ix_(ys, xs)].sum())))
    return out


@pytest.mark.parametrize("W,H", [(1, 1), (7, 5), (13, 13), (40, 9), (61, 97), (100, 100)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (4, 1), (8, 16), (32, 32)])
def test_tiling_matches_reference(W, H, stride):
    sx, sy = stride
    rng = np.random.default_rng(W * H + sx)
    for mask in (np.ones((H, W), dtype=bool), rng.random((H, W)) < 0.7):
        if not mask.any():
            continue
        grid = ss.Grid.from_mask(mask).index
        for ox, oy in ss.default_offsets(6):
            ref = reference_windows(mask, sx, sy, 6, ox, oy)
            # every lattice position is in exactly one reference window
            cover = np.zeros((H, W), dtype=int)
            for (a, b, i0, i1, j0, j1), _ in ref:
                cover[np.ix_(b + sy * np.arange(j0, j1), a + sx * np.arange(i0, i1))] += 1
            assert (cover == 1).all()
            expected = sorted(win for win, n in ref if n >= 2)
            got = sorted(tuple(x) for x in _core.list_windows(grid, sx, sy, 6, int(ox), int(oy)))
            assert got == expected


def test_clipped_edges_cover_lower_right_boundary():
    # 6k+2 subgrid length: the last window of the unshifted tiling is 2 wide, and the
    # shifted tiling has clipped windows at both ends; all must be present.
    mask = np.ones((14, 14), dtype=bool)
    for ox, oy in ss.default_offsets(6):
        c = phase_cover(mask, 1, 1, 6, ox, oy)
        assert (c == 1).all()
        wins = _core.list_windows(ss.Grid.from_mask(mask).index, 1, 1, 6, ox, oy)
        assert any(i1 == 14 for _, _, _, i1, _, _ in wins)
        assert any(j1 == 14 for _, _, _, _, _, j1 in wins)


def test_masked_cells_never_in_windows():
    rng = np.random.default_rng(0)
    mask = rng.random((30, 30)) < 0.6
    g = ss.Grid.from_mask(mask)
    for s in (1, 2, 4):
        for ox, oy in ss.default_offsets(6):
            c = phase_cover(mask, s, s, 6, ox, oy)
            assert c.max() <= 1


def test_start_stride_rule():
    st = ss.build_schedule(317, 316)
    assert tuple(st[0][:2]) == (64, 64)  # 6*64 = 384 >= 317 > 6*32
    st = ss.build_schedule(1000, 50)
    assert tuple(st[0][:2]) == (256, 16)
    assert tuple(st[-1]) == (1, 1, -1, 1)
    # both strides halve and floor at 1
    for a, b in zip(st[:-1], st[1:]):
        assert b[0] == max(1, a[0] // 2) and b[1] == max(1, a[1] // 2)
    assert tuple(ss.build_schedule(6, 6)[0]) == (1, 1, -1, 1)


def test_geometric_ratio_two_equals_halving():
    for W, H in [(317, 316), (1000, 50), (7, 300), (6, 6)]:
        a = ss.build_schedule(W, H, schedule="halving")
        b = ss.build_schedule(W, H, schedule="geometric", ratio=2.0)
        assert np.array_equal(a, b)


def test_geometric_sqrt2_is_decreasing_and_ends_at_one():
    st = ss.build_schedule(317, 316, schedule="geometric", ratio=2 ** 0.5)
    sx, sy = st[:, 0], st[:, 1]
    assert sx[0] == 64 and tuple(st[-1]) == (1, 1, -1, 1)
    assert np.all(np.diff(sx) <= 0) and np.all(np.diff(sy) <= 0)
    assert len(st) > len(ss.build_schedule(317, 316))
