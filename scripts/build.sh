#!/usr/bin/env bash
# One command: venv + vendored dvsim (C++ kernel and viewer, built from source) + Python deps.  scripts/build.sh
# Needs on the system (see README.md): uv, git, g++ 13+ (C++20). Python 3.12 is fetched by uv if missing. No X11 or Vulkan dev packages are needed to build.
# Network: cmake fetches the third-party sources pinned in vendor/dvsim/cmake/deps.cmake (GitHub, gitlab.com, freedesktop.org) and the slangc release.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=$ROOT/vendor/dvsim/.venv
cd "$ROOT"
for c in uv git g++; do command -v "$c" >/dev/null || { echo "build.sh: missing '$c' (see README.md prerequisites)" >&2; exit 1; }; done
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# build tools, same pins as vendor/dvsim/tools/remote-requirements.txt (cmake >= 3.25 is required)
uv pip install --python "$VENV/bin/python" cmake==4.4.4 ninja==1.13.2
export PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}
uv pip install --python "$VENV/bin/python" -r requirements.txt
"$VENV/bin/python" -c 'import dvsim; from dvsim import _dvsim; print("dvsim", dvsim.__file__); print("viewer built:", _dvsim.has_viewer)'
echo "OK: activate with 'source vendor/dvsim/.venv/bin/activate' (dv/ finds vendor/dvsim by itself; DVSIM_ROOT overrides)"
