from __future__ import annotations

import os
import statistics

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


def summarize(label: str, rows: int, cols: int, num_threads: int | None) -> None:
    n = rows * cols
    points = make_meandering_points(n)
    assignment = np.arange(n, dtype=np.int64)
    elapsed = []
    passes = []
    final_cost = []
    for _ in range(3):
        result = shufflesnap.window_cleanup(
            points,
            assignment,
            rows=rows,
            cols=cols,
            budget_seconds=1.0,
            window_size=6,
            num_threads=num_threads,
        )
        elapsed.append(float(result["elapsed_s"]))
        passes.append(int(result["rounds_completed"]))
        final_cost.append(float(result["final_cost"]))

    rates = [p / e for p, e in zip(passes, elapsed)]
    print(label)
    print("  median_elapsed_s:", statistics.median(elapsed))
    print("  median_passes:", statistics.median(passes))
    print("  median_passes_per_s:", statistics.median(rates))
    print("  final_costs:", final_cost)


def main() -> None:
    rows, cols = 256, 256
    print("cpu_count:", os.cpu_count())
    print("grid:", (rows, cols))
    summarize("serial (num_threads=1)", rows, cols, 1)
    summarize("auto (num_threads=None)", rows, cols, None)


if __name__ == "__main__":
    main()
