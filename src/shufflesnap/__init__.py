"""ShuffleSnap: point-to-grid assignment by strided windowed exact reassignment."""

from . import _core
from .api import (
    TRACE_COLUMNS,
    Result,
    assign,
    build_schedule,
    default_offsets,
    initial_assignment,
    run_schedule,
)
from .grid import Grid
from .metrics import assignment_cost, displacement_stats, validate_assignment
from .normalize import normalize_points

__version__ = "0.1.0"

__all__ = [
    "assign",
    "Result",
    "Grid",
    "normalize_points",
    "build_schedule",
    "default_offsets",
    "initial_assignment",
    "run_schedule",
    "validate_assignment",
    "assignment_cost",
    "displacement_stats",
    "TRACE_COLUMNS",
    "__version__",
]
