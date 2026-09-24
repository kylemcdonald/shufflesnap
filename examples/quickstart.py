"""Minimal ShuffleSnap example: snap a 2D embedding to a grid and save an image.

    python examples/quickstart.py
"""
import numpy as np

import shufflesnap as ss

rng = np.random.default_rng(0)
# a toy "embedding": 8 anisotropic clusters, 50k points
centers = rng.uniform(-10, 10, size=(8, 2))
labels = rng.integers(0, 8, 50_000)
xy = centers[labels] + rng.normal(size=(50_000, 2)) * rng.uniform(0.3, 1.5, size=(8, 2))[labels]

res = ss.assign(xy)  # default preset "balanced"; grid with exactly N cells
print(res.grid, f"mean squared displacement {res.mean_sq_displacement:.2f} cells^2",
      f"time {res.timings['total_s']:.2f} s")

# same points on a disk, and on a rectangle with 5% surplus cells
disk = ss.assign(xy, ss.Grid.disk(len(xy)))
wide = ss.assign(xy, ss.Grid.for_count(int(len(xy) * 1.05), aspect=16 / 9, partial="free"))
print("disk", disk.mean_sq_displacement, "wide with surplus", wide.mean_sq_displacement)

try:
    import matplotlib.pyplot as plt

    img = np.ones((res.grid.height, res.grid.width, 3))
    colors = plt.get_cmap("tab10")(labels % 10)[:, :3]
    img[res.xy[:, 1], res.xy[:, 0]] = colors
    plt.imsave("quickstart_grid.png", img)
    print("wrote quickstart_grid.png")
except ImportError:
    pass
