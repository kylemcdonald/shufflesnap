"""Target cell sets: rectangles, partial rectangles, and arbitrary masks.

A :class:`Grid` is a boolean mask on a ``width x height`` lattice.  ``True`` marks a
usable cell.  Lattice position ``(x, y)`` means column ``x`` and row ``y``; cell ids
enumerate the usable positions in row-major order.  Cell centers default to the
integer lattice coordinates, so one grid unit equals one cell pitch.  A different
``pitch`` gives anisotropic cells (for example 4:3 thumbnails) without changing the
lattice topology used by the solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Grid"]


@dataclass(frozen=True, eq=False)
class Grid:
    mask: np.ndarray
    pitch: tuple = (1.0, 1.0)
    index: np.ndarray = field(init=False, repr=False)
    lattice: np.ndarray = field(init=False, repr=False)
    centers: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        mask = np.ascontiguousarray(np.asarray(self.mask, dtype=bool))
        if mask.ndim != 2 or mask.shape[0] < 1 or mask.shape[1] < 1:
            raise ValueError("mask must be a non-empty 2D array (height, width)")
        if not mask.any():
            raise ValueError("mask has no usable cells")
        ys, xs = np.nonzero(mask)  # row-major order
        index = np.full(mask.shape, -1, dtype=np.int32)
        index[ys, xs] = np.arange(len(xs), dtype=np.int32)
        lattice = np.stack([xs, ys], axis=1).astype(np.int32)
        px, py = float(self.pitch[0]), float(self.pitch[1])
        if not (px > 0 and py > 0):
            raise ValueError("pitch must be positive")
        centers = np.ascontiguousarray(lattice * np.array([px, py]), dtype=np.float64)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "pitch", (px, py))
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "centers", centers)

    # ------------------------------------------------------------------ builders
    @classmethod
    def rectangle(cls, width: int, height: int, pitch=(1.0, 1.0)) -> "Grid":
        """Full ``width x height`` rectangle."""
        return cls(np.ones((int(height), int(width)), dtype=bool), pitch)

    @classmethod
    def for_count(cls, n: int, aspect: float = 1.0, partial: str = "center", pitch=(1.0, 1.0)) -> "Grid":
        """Smallest ``width x height`` lattice with ``width/height ~ aspect`` holding ``n`` cells.

        ``partial`` controls the cells beyond ``n`` in the last row:

        * ``"center"``: the last row keeps ``r`` cells centered horizontally, so the grid
          has exactly ``n`` cells (a bijection).
        * ``"start"``: the last row keeps its first ``r`` cells (left-aligned).
        * ``"free"``: the full rectangle is kept; the solver chooses which cells stay empty.
        """
        n = int(n)
        if n < 1:
            raise ValueError("n must be positive")
        if partial not in ("center", "start", "free"):
            raise ValueError("partial must be 'center', 'start' or 'free'")
        width = max(1, int(math.ceil(math.sqrt(n * float(aspect)))))
        height = int(math.ceil(n / width))
        mask = np.ones((height, width), dtype=bool)
        extra = width * height - n
        if extra and partial != "free":
            r = width - extra
            mask[-1, :] = False
            start = (width - r) // 2 if partial == "center" else 0
            mask[-1, start:start + r] = True
        return cls(mask, pitch)

    @classmethod
    def from_mask(cls, mask, pitch=(1.0, 1.0)) -> "Grid":
        """Arbitrary mask; ``True`` (or nonzero) marks usable cells."""
        return cls(np.asarray(mask).astype(bool), pitch)

    @classmethod
    def disk(cls, n: int, pitch=(1.0, 1.0)) -> "Grid":
        """Disk-shaped mask with exactly ``n`` cells (outermost cells trimmed by angle)."""
        n = int(n)
        r = math.sqrt(n / math.pi) + 2
        size = int(2 * math.ceil(r) + 1)
        c = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        d2 = (xx - c) ** 2 + (yy - c) ** 2
        ang = np.arctan2(yy - c, xx - c)
        order = np.lexsort((ang.ravel(), d2.ravel()))
        mask = np.zeros(size * size, dtype=bool)
        mask[order[:n]] = True
        mask = mask.reshape(size, size)
        rows = np.nonzero(mask.any(1))[0]
        cols = np.nonzero(mask.any(0))[0]
        return cls(mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], pitch)

    @classmethod
    def from_image(cls, image, n: int, threshold: float = 0.5, invert: bool = False, pitch=(1.0, 1.0)) -> "Grid":
        """Mask shaped like a grayscale image (dark = inside) with exactly ``n`` cells.

        The image is resampled so that about ``n`` lattice cells fall inside the shape;
        the cells nearest the shape boundary are then added or removed (by distance to
        the thresholded shape, ties broken by position) to reach exactly ``n``.
        """
        img = np.asarray(image, dtype=np.float64)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        if img.max() > 1.0:
            img = img / 255.0
        inside = img < threshold
        if invert:
            inside = ~inside
        frac = inside.mean()
        if frac <= 0:
            raise ValueError("image has no inside pixels")
        h0, w0 = inside.shape
        scale = math.sqrt(n / (frac * h0 * w0))
        best = None
        for _ in range(40):
            H = max(1, int(round(h0 * scale)))
            W = max(1, int(round(w0 * scale)))
            yi = np.minimum(((np.arange(H) + 0.5) / H * h0).astype(int), h0 - 1)
            xi = np.minimum(((np.arange(W) + 0.5) / W * w0).astype(int), w0 - 1)
            sub = img[np.ix_(yi, xi)]
            m = (sub < threshold) if not invert else (sub >= threshold)
            cnt = int(m.sum())
            best = (sub, m, H, W)
            if cnt >= n:
                break
            scale *= 1.02
        sub, m, H, W = best
        # signed "depth" score: gray value relative to threshold, used to trim/extend.
        score = (threshold - sub) if not invert else (sub - threshold)
        order = np.lexsort((np.arange(H * W), -score.ravel()))
        mask = np.zeros(H * W, dtype=bool)
        mask[order[:n]] = True
        mask = mask.reshape(H, W)
        rows = np.nonzero(mask.any(1))[0]
        cols = np.nonzero(mask.any(0))[0]
        return cls(mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], pitch)

    # --------------------------------------------------------------- properties
    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.lattice.shape[0])

    def __repr__(self) -> str:
        return f"Grid(width={self.width}, height={self.height}, n_cells={self.n_cells}, pitch={self.pitch})"
