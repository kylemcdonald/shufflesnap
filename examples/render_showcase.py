from __future__ import annotations

import argparse
import pathlib
import struct
import zlib

import numpy as np

import shufflesnap


def make_meandering_points(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=1.0, size=(n, 2))
    points = np.cumsum(steps, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, np.finfo(np.float64).eps)
    return np.asarray((points - mins) / span, dtype=np.float64)


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
    return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def render_points(
    source_points: np.ndarray,
    dest_points: np.ndarray,
    width: int,
    height: int,
    interp: float,
    colors: np.ndarray,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    interp_points = source_points + interp * (dest_points - source_points)

    px = np.rint(interp_points[:, 0] * (width - 1)).astype(np.int32)
    py = np.rint((1.0 - interp_points[:, 1]) * (height - 1)).astype(np.int32)
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    order = np.random.default_rng(0).permutation(interp_points.shape[0])
    canvas[py[order], px[order]] = colors[order]
    return canvas


def render_triptych(
    source_points: np.ndarray,
    dest_points: np.ndarray,
    panel_width: int,
    panel_height: int,
    mid_interp: float,
) -> np.ndarray:
    colors = lab_to_srgb(source_points)
    panels = [
        render_points(source_points, dest_points, panel_width, panel_height, 0.0, colors),
        render_points(source_points, dest_points, panel_width, panel_height, mid_interp, colors),
        render_points(source_points, dest_points, panel_width, panel_height, 1.0, colors),
    ]
    return np.concatenate(panels, axis=1)


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", crc)
    )


def write_png(path: pathlib.Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 array of shape (height, width, 3)")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a shufflesnap showcase PNG without matplotlib.")
    parser.add_argument("--grid-width", type=int, default=512)
    parser.add_argument("--grid-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cleanup-seconds", type=float, default=30.0)
    parser.add_argument("--mid-interp", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "assets" / "showcase_triptych_512.png",
    )
    args = parser.parse_args()

    n = args.grid_width * args.grid_height
    points = make_meandering_points(n, seed=args.seed)
    dest_points, _, _ = shufflesnap.snap_to_grid(
        points,
        width=args.grid_width,
        height=args.grid_height,
        cleanup_seconds=args.cleanup_seconds,
        num_threads=None if args.num_threads == 0 else args.num_threads,
    )

    rgb = render_triptych(
        points,
        dest_points,
        args.image_width,
        args.image_height,
        args.mid_interp,
    )
    write_png(args.output, rgb)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
