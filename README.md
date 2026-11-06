# drone-verifier

A benchmark that watches AI agents design a hopper drone and proves or disproves their work in simulation. An agent picks real
parts from a catalog (budget, mass, envelope), writes a flight controller, and a verifier flies it through gates (static, lift,
nominal, wind, rain, robustness). The score is gate points on held-out seeds. Design notes: `docs/DESIGN.md`.

The simulator is dvsim (proprietary robot design and MuJoCo-based sim kernel), **vendored** at `vendor/dvsim` (commit and date in
`vendor/dvsim/VENDORED`). A fresh clone builds it from source; no dvsim checkout is needed anywhere.

## Prerequisites

Linux x86_64 (the build and the viewer are tested on Ubuntu 24.04, GCC 13).

System packages (apt names):

- `git`, `curl`, `ca-certificates`, `build-essential` (g++ 13 or newer, C++20), `uv` (https://docs.astral.sh/uv/; fetches Python 3.12 if missing)
- GUI/video only: a Vulkan 1.3 driver (`mesa-vulkan-drivers`, or the vendor's), `libvulkan1`, `vulkan-tools` (for `vulkaninfo`);
  `gstreamer1.0-tools`, `gstreamer1.0-plugins-base`, `gstreamer1.0-plugins-good`, `gstreamer1.0-plugins-bad` and a VA-API driver
  (`mesa-va-drivers`) because `dv/gui.py` encodes with `vah264enc`; `libx11-6` (GLFW opens libX11 at run time)
- Agents: the `claude` CLI (Claude Code), logged in, on PATH. Only `harness/` uses it.
- Benchmark and demo drivers (`harness/bench.py`, `harness/demo.py`, `scripts/hf.sh`) additionally need `ssh` and `rsync` and a Linux
  GPU host named `hfbox` in `~/.ssh/config` (see Demo). Running a session on the local machine needs none of this.

Fetched from the network by the build (pinned in `vendor/dvsim/cmake/deps.cmake`, none is GPL):

- Python (PyPI): cmake 4.4.4, ninja 1.13.2, scikit-build-core 1.1.1, numpy 2.5.3, mcp 2.3.0 (+ starlette, uvicorn, anyio, ...), gymnasium 1.4.0, pytest
- C/C++ sources: MuJoCo 3.7.0 (Apache-2.0) and its own deps (tinyxml2, qhull, lodepng, ccd, tinyobjloader, MarchingCubeCpp), FlatBuffers,
  GoogleTest, nanobind, Vulkan-Headers, volk, VulkanMemoryAllocator, mavlink c_library_v2, toml++, Eigen (MPL-2.0), OSQP, GLFW 3.4,
  Dear ImGui 1.92.9b (incl. its Roboto font), ImGuiColorTextEdit, stb_image, X11 compile-time headers (xorgproto, libX11, libXrandr,
  libXrender, libXi, libXinerama, libXcursor, libXext, libXfixes; MIT/X11) from GitHub, gitlab.com and gitlab.freedesktop.org
- Build tool: the slangc 2026.19 release archive from github.com/shader-slang/slang (Apache-2.0 WITH LLVM-exception; SHA256 pinned,
  build-time only, never shipped)
- Vendored in the repo: `vendor/dvsim/meshes/koch` and `so101` carry their own Apache-2.0 LICENSE and NOTICE files.

Build time is a few minutes to ~20 minutes on a first build and about 2 GB of disk for `vendor/dvsim/build`.

## Build

    scripts/build.sh

It creates `vendor/dvsim/.venv`, installs cmake and ninja into it, then builds dvsim (kernel, nanobind module, Vulkan viewer) and installs
the Python dependencies from `requirements.txt`. Then `source vendor/dvsim/.venv/bin/activate`. `dv/` finds `vendor/dvsim` by itself;
set `DVSIM_ROOT` only to use another dvsim tree. Checks: `python vendor/dvsim/tools/check_parts.py` (part library, expects 0 problems).

## Run one agent (locally)

    export DV_RUNS=$PWD/runs
    python -m dv.server --session A --port 8801 --model haiku &       # add --gui for the dvsim window + video (needs the GUI packages)
    python -m dv.render_prompt > /tmp/prompt.txt
    python harness/run_agent.py --model haiku --port 8801 --run-dir results/haiku-0 --prompt-file /tmp/prompt.txt

The agent works only through the MCP tools at `http://127.0.0.1:8801/mcp`. The run dir holds `events.jsonl`, `controller.py`,
`design.json`, `results.json`. Positive and negative controls (reference scores 100, do-nothing and open-loop score no flight points):
`python tests/check_controls.py`.

## Run the benchmark

    python harness/bench.py --models haiku,sonnet,opus,fable --trials 1 --parallel 4

Models: `haiku sonnet opus fable` run through Claude Code; `nemotron` (NVIDIA NIM, free, `nvidia/nemotron-3.5-super-vl-120b-a12b`),
`deepseek-flash` and `deepseek-pro` (paid, direct DeepSeek provider) run through the `omp` CLI, isolated the same way (only our MCP tools, no built-ins).

`bench.py` starts one headless session per run through `scripts/hf.sh` (ports 8804-8811), runs the agents from this machine,
and pulls results into `results/`. `python harness/leaderboard.py` builds the leaderboard. Where the sessions run is one knob, `DV_HOST`:
an ssh host name (default `hfbox`; `DV_DIR` sets its work dir, default `/workspace/wt/dv`), or `local`, which runs the same commands as
local subprocesses with no ssh, rsync or bridge: runs and logs go under `.dv/` in the repo, `DVSIM_ROOT` defaults to `vendor/dvsim`
(and `DV_RUNS`, `DV_PORT_BASE` override the run dir and first port). Needs `tmux`. Example, everything on this machine:

    DV_HOST=local python harness/bench.py --models haiku --trials 1 --parallel 1

## Run the demo

    python harness/demo.py single MODEL        # one live agent, GUI session on hfbox, dashboard in the browser
    python harness/demo.py versus LOW HIGH     # two models side by side
    python harness/demo.py replay              # saved results, no remote host at all

Live demos use `scripts/hf.sh` (rsync of this repo to hfbox, dvsim build there) and `harness/bridge.py` (ssh-exec bridge, because hfbox
refuses port forwarding). On a remote host `scripts/hf.sh` points `DVSIM_ROOT` at an existing dvsim build there, not at `vendor/dvsim`;
`harness/demo.py` does not support `DV_HOST=local` yet.
