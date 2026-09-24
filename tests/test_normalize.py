import numpy as np

import shufflesnap as ss


def test_bbox_maps_extremes_to_cell_box():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(1000, 2)) * [3, 0.1] + [5, -2]
    g = ss.Grid.rectangle(40, 20)
    Q = ss.normalize_points(P, g, "bbox")
    assert np.allclose(Q.min(0), [0, 0]) and np.allclose(Q.max(0), [39, 19])


def test_fit_preserves_aspect():
    rng = np.random.default_rng(0)
    P = rng.random((1000, 2)) * [2, 1]
    g = ss.Grid.rectangle(30, 30)
    Q = ss.normalize_points(P, g, "fit")
    ext = Q.max(0) - Q.min(0)
    assert np.isclose(ext[0] / ext[1], (P.max(0) - P.min(0))[0] / (P.max(0) - P.min(0))[1])
    assert ext.max() <= 29 + 1e-9
