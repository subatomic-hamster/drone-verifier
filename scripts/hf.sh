#!/usr/bin/env bash
# Run on hfbox. hf.sh sync | hf.sh run CMD | hf.sh bg NAME CMD | hf.sh log NAME [N] | hf.sh stop NAME | hf.sh tunnel PORT...
# Code in /workspace/wt/dv/src, dvsim build in /workspace/wt/dv-dvsim/src, runs in /workspace/wt/dv/runs. Long-lived sessions are tmux dv-NAME.
set -euo pipefail
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/run/user/1000/hfbox-agent.sock}
HOST=${DV_HOST:-hfbox}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=/workspace/wt/dv
ENV="export OMP_NUM_THREADS=1 PYTHONPATH=$DIR/src:/workspace/wt/dv-dvsim/src/examples/hopper DV_RUNS=$DIR/runs DVSIM_ROOT=/workspace/wt/dv-dvsim/src PATH=/workspace/wt/dv-dvsim/src/.venv/bin:\$PATH; cd $DIR/src"
s() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "$@" 2> >(grep -Ev '^(depth=|verify return|Connecting to)' >&2); }

sync() { rsync -a --delete --exclude .git --exclude runs --exclude results --exclude __pycache__ -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' "$ROOT/" "$HOST:$DIR/src/" 2> >(grep -Ev '^(depth=|verify return|Connecting to)' >&2); }

case ${1:?usage} in
sync) s "mkdir -p $DIR/src $DIR/runs $DIR/logs $DIR/slots"; sync ;;
run) shift; sync; s "$ENV; $*" ;;
bg)
    name=$2; shift 2; sync
    s "tmux new-session -d -s dv-$name $(printf '%q' "bash -c $(printf '%q' "$ENV; $* 2>&1 | tee $DIR/logs/$name.log")")"
    echo "started dv-$name" ;;
log) s "tail -n ${3:-40} $DIR/logs/$2.log; tmux has-session -t dv-$2 2>/dev/null && echo RUNNING || echo STOPPED" ;;
stop) s "tmux kill-session -t dv-$2" ;;
tunnel)
    shift; args=(); for p in "$@"; do args+=(-L "$p:127.0.0.1:$p"); done
    exec ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -N "${args[@]}" "$HOST" ;;
*) echo "usage: hf.sh sync|run|bg|log|stop|tunnel" >&2; exit 2 ;;
esac
