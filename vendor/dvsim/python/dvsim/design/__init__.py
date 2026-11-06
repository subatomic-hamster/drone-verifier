"""dvsim.design: parts library, port snapping, fastener selection, assembly -> MJCF builder. Stdlib + numpy only; does not
need the C++ kernel (assemblies are only simulated by loading the generated MJCF into dvsim.Sim)."""

from .assembly import Assembly, dumps, load, loads, save
from .build import build, compute, format_budget
from .drc import Finding, check
from .edit import Design, EditResult
from .library import DesignError, Library, Part
from .snap import (
    MateSpec,
    Tree,
    find_compatible_ports,
    hole_count,
    hole_positions,
    mate_transform,
    port_frame,
    select_fasteners,
    solve_tree,
    thread_of,
)

__all__ = [
    "Assembly",
    "Design",
    "DesignError",
    "EditResult",
    "Finding",
    "Library",
    "MateSpec",
    "Part",
    "Tree",
    "build",
    "check",
    "compute",
    "dumps",
    "find_compatible_ports",
    "format_budget",
    "hole_count",
    "hole_positions",
    "load",
    "loads",
    "mate_transform",
    "port_frame",
    "save",
    "select_fasteners",
    "solve_tree",
    "thread_of",
]
