# Changelog

## 0.1.0 (unreleased)

- Strided windowed exact reassignment (C++/OpenMP kernel, nanobind bindings).
- Presets: `baseline` (6x6 windows, halving strides, as originally specified), `fast`,
  `balanced` (default), `quality`.
- Grids: rectangles, exact-fit partial rows, surplus cells, disks, arbitrary masks, image masks,
  anisotropic pitch.
- Optional certified exact solver (`shufflesnap.exact`, requires OR-Tools and SciPy).
