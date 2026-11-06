#!/usr/bin/env bash
# Run dvsim builds/tests on a remote build host. The laptop only edits, runs git and syncs.
#
#   tools/remote.sh bootstrap         one-time: user-local toolchain venv (cmake, ninja, uv) on the host
#   tools/remote.sh sync              rsync the working tree to the host
#   tools/remote.sh run [--gpu] [--nolock] [--slots N] CMD
#                                     sync, then run CMD in the remote source dir (blocks, streams output);
#                                     waits for N CPU slots (default 1 of DVSIM_SLOTS=2, 8 threads each; N=2 is the whole
#                                     box) unless --nolock; killed after DVSIM_TIMEOUT s (3600). The job gets
#                                     DVSIM_THREADS / OMP_NUM_THREADS / CMAKE_BUILD_PARALLEL_LEVEL = its thread count.
#   tools/remote.sh bg [--gpu] [--slots N] NAME CMD
#                                     sync, then run CMD in tmux session dvsim-NAME; output to logs/NAME.log
#   tools/remote.sh log NAME [N]      last N lines of that log (no follow) + whether it is still running
#   tools/remote.sh pull PATH...      copy remote paths (relative to the remote dir) back into ./remote-out/
#
# Example:
#   tools/remote.sh run 'cmake --preset dev && cmake --build --preset dev && ctest --preset dev --output-on-failure'
#
# Host-agnostic: today hfbox, later an AWS EC2 Ubuntu 24.04 instance. Point at another host with
#   DVSIM_REMOTE=<ssh alias>  DVSIM_REMOTE_DIR=<abs dir>  DVSIM_GPU_LOCK=  (empty = dedicated host, no shared GPU lock)
# Parallel git worktrees: one DVSIM_REMOTE_DIR each. The CPU lane ($DVSIM_LANES/cpu.0..DVSIM_SLOTS-1, default
# /workspace/wt/dvsim-cache/lanes) is shared by all worktrees: DVSIM_SLOTS jobs run at once, DVSIM_JOBS/DVSIM_SLOTS threads
# each; a --slots N job waits until it holds N slots. --gpu also waits for the one GPU lane ($DVSIM_LANES/gpu). Never
# flock a lane file inside a job (deadlock).
# Baseline the code must build on: Ubuntu 24.04, GCC 13, Python 3.12.
set -euo pipefail

REMOTE=${DVSIM_REMOTE:-hfbox}
DIR=${DVSIM_REMOTE_DIR:-/workspace/wt/dvsim}
JOBS=${DVSIM_JOBS:-16}  # hfbox is dedicated to dvsim (user, 2026-10-10): use all 16 cores
SLOTS=${DVSIM_SLOTS:-2}  # CPU lane: SLOTS lock files, JOBS/SLOTS threads each (DVSIM_CPU_LOCK is gone)
LANES=${DVSIM_LANES:-/workspace/wt/dvsim-cache/lanes}  # shared by every worktree on the host
# GPU lane: one --gpu job at a time. VRAM overflow spills into host GTT memory, which no container memory cap
# counts; two big GPU jobs OOM-killed hfbox on 2026-10-10. DVSIM_GPU_LOCK= (empty) turns it off.
GPU_LOCK=${DVSIM_GPU_LOCK-$LANES/gpu}
# Shared sccache DISK cache (CMakeLists uses sccache when on PATH; a no-op on hosts without $CACHE/bin/sccache). Each checkout gets
# its OWN sccache server in preprocessor mode (SCCACHE_DIRECT=false; port from the source dir, see CMakeLists.txt): the old single
# direct-mode server served STALE objects after a shared header (src/core/solver.h) changed. A server reads BASEDIRS at start, so
# list every worktree dir: object keys then ignore the /workspace/wt/dvsim-X prefix and builds of the same commit in different
# worktrees still hit each other.
CACHE=${DVSIM_CACHE_DIR:-/workspace/wt/dvsim-cache}
BASEDIRS=${DVSIM_SCCACHE_BASEDIRS:-\$(ls -d /workspace/wt/dvsim /workspace/wt/dvsim-* | grep -v -- -cache | paste -sd:)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# ssh exit 255 = connection failure (hfbox funnel outages last ~15 min); retry with backoff.
# retry "<connection-failure exit codes>" cmd...
retry() {
  local codes=" $1 " delay=15 rc; shift
  for _ in 1 2 3 4 5 6; do
    rc=0; "$@" || rc=$?
    [[ $codes != *" $rc "* ]] && return $rc
    echo "remote.sh: $REMOTE unreachable (rc=$rc), retrying in ${delay}s" >&2
    sleep "$delay"; delay=$((delay * 2))
  done
  return $rc
}
rssh() { retry 255 ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "$@"; }
# rsync: 255 = ssh failed, 10/12 = socket/stream dropped mid-transfer
rrsync() { retry "255 10 12" rsync -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' "$@"; }

sync_tree() {
  rssh "mkdir -p $DIR/src $DIR/logs"
  rrsync -az --delete \
    --exclude /build/ --exclude /remote-out/ --exclude .git/ --exclude .venv/ \
    --exclude __pycache__/ --exclude .ruff_cache/ \
    "$ROOT/" "$REMOTE:$DIR/src/"
}

# Remote shell prefix: toolchain on PATH, then the CPU lane. The host has K slot lock files ($LANES/cpu.0..K-1, shared by
# every worktree), JOBS/K threads each. A job holds `slots` of them for its whole run (flock -n on each, all-or-nothing and
# retried, so two jobs can never deadlock each other) and gets slots*JOBS/K threads in DVSIM_THREADS and the usual
# *_NUM_THREADS / parallel-level variables. --nolock = no slot (tiny commands), threads of one slot.
# The lane fds are inherited by everything the job starts: never `flock` a lane file inside a job, and start daemons
# (this checkout's sccache server is started here, before the slots are taken) outside it.
prefix() {
  local gpu=$1 lock=${2:-1} slots=${3:-1} n
  (( slots >= 1 && slots <= SLOTS )) || { echo "remote.sh: --slots must be 1..$SLOTS (DVSIM_SLOTS)" >&2; exit 2; }
  n=$((JOBS * (lock == 1 ? slots : 1) / SLOTS))
  local p="cd $DIR/src && export PATH=$DIR/env/bin:$CACHE/bin:\$PATH SCCACHE_DIR=$CACHE/sccache SCCACHE_CACHE_SIZE=4G SCCACHE_BASEDIRS=$BASEDIRS DVSIM_THREADS=$n CMAKE_BUILD_PARALLEL_LEVEL=$n CTEST_PARALLEL_LEVEL=$n OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n LP_NUM_THREADS=$n &&"
  # this checkout's sccache server (port + preprocessor mode exactly as CMakeLists' launcher sets them), started before the slots
  # are taken so it does not inherit the lane fds
  p+=" { SCCACHE_DIRECT=false SCCACHE_SERVER_PORT=\$((4300 + 0x\$(printf %s $DIR/src | md5sum | cut -c1-2))) sccache --start-server >/dev/null 2>&1 || true; } &&"
  if [[ $lock == 1 ]]; then
    p+=" mkdir -p $LANES && w=0 && while :; do g=0; for i in \$(seq 0 $((SLOTS - 1))); do
      if (( g < $slots )); then eval \"exec \$((30 + i))>$LANES/cpu.\$i\"; if flock -n \$((30 + i)); then g=\$((g + 1)); else eval \"exec \$((30 + i))>&-\"; fi; fi; done;
      (( g == $slots )) && break; for i in \$(seq 0 $((SLOTS - 1))); do eval \"exec \$((30 + i))>&-\"; done;
      (( w++ == 0 )) && echo \"remote.sh: waiting for $slots of $SLOTS cpu slots\" >&2; sleep 2; done &&"
  fi
  [[ $gpu == 1 && -n $GPU_LOCK ]] && p+=" mkdir -p $LANES && flock $GPU_LOCK"
  printf '%s' "$p"
}

cmd=${1:-}; shift || true
case $cmd in
  bootstrap)
    sync_tree
    rssh "set -e; command -v g++ rsync flock tmux >/dev/null || { echo 'need g++ rsync flock tmux on host' >&2; exit 1; }
          [ -x $DIR/env/bin/cmake ] || { python3 -m venv $DIR/env && $DIR/env/bin/pip install -q -r $DIR/src/tools/remote-requirements.txt; }
          $DIR/env/bin/cmake --version | head -1; $DIR/env/bin/ninja --version; $DIR/env/bin/uv --version; g++ --version | head -1"
    ;;
  sync) sync_tree ;;
  run)
    gpu=0 lock=1 slots=1
    while [[ ${1:-} == --* ]]; do
      case $1 in --gpu) gpu=1 ;; --nolock) lock=0 ;; --slots) slots=$2; shift ;; *) echo "unknown option $1" >&2; exit 2 ;; esac; shift
    done
    sync_tree
    # A foreground job never holds its slots past DVSIM_TIMEOUT (default 1 h): a hung or
    # forgotten server must not block every other dvsim job. Long work belongs in `bg`.
    rssh "$(prefix $gpu $lock $slots) timeout ${DVSIM_TIMEOUT:-3600} bash -c $(printf '%q' "$*")"
    ;;
  bg)
    gpu=0 slots=1
    while [[ ${1:-} == --* ]]; do
      case $1 in --gpu) gpu=1 ;; --slots) slots=$2; shift ;; *) echo "unknown option $1" >&2; exit 2 ;; esac; shift
    done
    name=$1; shift
    sync_tree
    rssh "tmux new -d -s dvsim-$name $(printf '%q' "$(prefix $gpu 1 $slots) bash -c $(printf '%q' "$*") >$DIR/logs/$name.log 2>&1")"
    echo "started tmux session dvsim-$name on $REMOTE; tools/remote.sh log $name"
    ;;
  log) rssh "tail -n ${2:-50} $DIR/logs/$1.log; tmux has-session -t dvsim-$1 2>/dev/null && echo '[running]' || echo '[finished]'" ;;
  pull)
    mkdir -p "$ROOT/remote-out"
    for p in "$@"; do rrsync -az "$REMOTE:$DIR/$p" "$ROOT/remote-out/"; done
    ;;
  *) sed -n '2,25p' "$0"; exit 2 ;;
esac
