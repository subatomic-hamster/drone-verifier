import os
from pathlib import Path

# DVSIM_ROOT defaults to the vendored dvsim tree; set it only to point at another dvsim tree (hf.sh does, for the demo box).
os.environ.setdefault(
    "DVSIM_ROOT", str(Path(__file__).resolve().parent.parent / "vendor" / "dvsim")
)
