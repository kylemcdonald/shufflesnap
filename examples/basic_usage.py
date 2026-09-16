from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

import shufflesnap


def make_meandering_points(n: int, seed: int = 0, margin: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=1.0, size=(n, 2))
    points = np.cumsum(steps, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, np.finfo(np.float64).eps)
    points = (points - mins) / span
    points = margin + (1.0 - 2.0 * margin) * points
    return np.asarray(points, dtype=np.float64)


def lab_to_srgb(points: np.ndarray) -> np.ndarray:
    l = np.full(points.shape[0], 72.0, dtype=np.float64)
    a = (points[:, 0] * 2.0 - 1.0) * 80.0
    b = (points[:, 1] * 2.0 - 1.0) * 80.0

    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def invf(t: np.ndarray) -> np.ndarray:
        t3 = t * t * t
        return np.where(t3 > epsilon, t3, (116.0 * t - 16.0) / kappa)

    x = 0.95047 * invf(fx)
    y = invf(fy)
    z = 1.08883 * invf(fz)

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb_lin = np.clip(np.column_stack([r_lin, g_lin, b_lin]), 0.0, 1.0)

    threshold = 0.0031308
    rgb = np.where(
        rgb_lin <= threshold,
        12.92 * rgb_lin,
        1.055 * np.power(rgb_lin, 1.0 / 2.4) - 0.055,
    )
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    points = make_meandering_points(32 * 32, seed=0)
    grid_points, assignment, grid_size = shufflesnap.snap_to_grid(
        points,
        width=32,
        height=32,
        cleanup_seconds=0.5,
    )

    interp = points + 0.8 * (grid_points - points)
    colors = lab_to_srgb(points)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150, facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(interp[:, 0], interp[:, 1], s=1, c=colors, marker="s", linewidths=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"shufflesnap snap_to_grid · grid={grid_size[0]}x{grid_size[1]}", color="white")
    fig.tight_layout()
    output = pathlib.Path(__file__).with_name("basic_usage_output.png")
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print("grid_size:", grid_size)
    print("assignment shape:", assignment.shape)
    print("first five assigned indices:", assignment[:5])
    print("wrote image:", output)


if __name__ == "__main__":
    main()
