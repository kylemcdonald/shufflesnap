"""The serial NumPy/SciPy reference (shufflesnap.reference) must reproduce the compiled
kernel exactly: same schedule, tilings, clipping, acceptance rule and stopping rule."""
import numpy as np
import pytest

import shufflesnap as ss
from shufflesnap import reference

# the three presets, plus 6x6 windows with halving strides as an explicit setting
PRESETS = [("fast", 5, 2 ** 0.5), ("balanced", 8, 2 ** 0.5), ("quality", 12, 2 ** 0.5), (None, 6, 2.0)]


def _grids():
    rng = np.random.default_rng(5)
    mask = rng.random((28, 31)) > 0.3
    return {
        "exact fit": (ss.Grid.for_count(900), 900),
        "partial last row": (ss.Grid.for_count(997), 997),
        "surplus cells": (ss.Grid.for_count(997, partial="free"), 997),
        "rectangle": (ss.Grid.rectangle(47, 9), 400),
        "disk": (ss.Grid.disk(700), 700),
        "random mask": (ss.Grid.from_mask(mask), int(mask.sum()) - 20),
    }


GRIDS = _grids()


@pytest.mark.parametrize("preset,w,ratio", PRESETS)
@pytest.mark.parametrize("name", list(GRIDS))
def test_reference_matches_kernel(name, preset, w, ratio):
    g, n = GRIDS[name]
    rng = np.random.default_rng(len(name) + w)
    P = np.concatenate([rng.normal(size=(n // 2, 2)), 0.4 * rng.normal(size=(n - n // 2, 2)) + [3.0, 1.0]])
    Pn = ss.normalize_points(P, g, "bbox")
    if preset is None:
        native = ss.assign(Pn, g, normalize="none", window=w, schedule="halving", seed=11, threads=2)
    else:
        native = ss.assign(Pn, g, normalize="none", preset=preset, seed=11, threads=2)
    serial = reference.snap(Pn, g.mask, w=w, ratio=ratio, seed=11)
    np.testing.assert_array_equal(native.cell, serial)
    assert native.finished


def test_reference_schedule_matches_package():
    for W, H in [(317, 316), (1000, 50), (7, 7), (1, 40)]:
        for w, ratio, kind in [(6, 2.0, "halving"), (8, 2 ** 0.5, "geometric"), (12, 2 ** 0.5, "geometric")]:
            stages = ss.build_schedule(W, H, window=w, schedule=kind, ratio=ratio)
            assert [tuple(s[:2]) for s in stages[:-1]] == reference.coarse_stages(W, H, w, ratio)
