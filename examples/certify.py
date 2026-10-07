"""Certify how far a layout is from the optimum (needs `pip install shufflesnap[exact]`)."""
import numpy as np

import shufflesnap as ss
from shufflesnap import exact

xy = np.random.default_rng(1).normal(size=(20_000, 2))
res = ss.assign(xy, preset="fast")
opt = exact.solve_certified(res.points, res.grid, res.cell)
print(f"ShuffleSnap cost {res.cost:.1f}; certified optimum {opt['cost']:.1f} "
      f"(lower bound {opt['lower_bound']:.1f}, gap {opt['gap']:.2e}); "
      f"excess {100 * (res.cost / opt['cost'] - 1):.4f}%")
