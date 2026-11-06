"""STL reader (binary or ASCII) for vendor meshes used as visual elements."""

import re
import struct
from pathlib import Path

import numpy as np

from .mesh import Mesh


def read_stl(path) -> Mesh:
    b = Path(path).read_bytes()
    if len(b) >= 84 and 84 + 50 * struct.unpack("<I", b[80:84])[0] == len(b):
        n = struct.unpack("<I", b[80:84])[0]
        rec = np.frombuffer(
            b,
            np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
            count=n,
            offset=84,
        )
        v = rec["v"].reshape(-1, 3).astype(float)
    else:
        v = np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", b), dtype=float)
    return Mesh(v, np.arange(len(v)).reshape(-1, 3))
