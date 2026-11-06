#!/usr/bin/env bash
# Run on hfbox. hf.sh sync | hf.sh run CMD | hf.sh bg NAME CMD | hf.sh log NAME [N] | hf.sh stop NAME | hf.sh tunnel PORT...
# Code in /workspace/wt/dv/src, dvsim build in /workspace/wt/dv-dvsim/src, runs in /workspace/wt/dv/runs. Long-lived sessions are tmux dv-NAME.
set -euo pipefail
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/run/user/1000/hfbox-agent.sock}
HOST=${DV_HOST:-hfbox}   # an ssh host, or "local": same commands as local subprocesses, no ssh, no rsync (harness/hosts.py reads the same knob)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "$HOST" = local ]; then
    DIR=${DV_DIR:-$ROOT/.dv}; SRC=$ROOT; DVSIM=${DVSIM_ROOT:-$ROOT/vendor/dvsim}; RUNS=${DV_RUNS:-$DIR/runs}
    s() { bash -c "$1"; }
    sync() { :; }
else
    DIR=${DV_DIR:-/workspace/wt/dv}; SRC=$DIR/src; DVSIM=/workspace/wt/dv-dvsim/src; RUNS=$DIR/runs
    s() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "$@" 2> >(grep -Ev '^(depth=|verify return|Connecting to)' >&2); }
fi
ENV="export OMP_NUM_THREADS=1 PYTHONPATH=$SRC:$DVSIM/examples/hopper DV_RUNS=$RUNS DVSIM_ROOT=$DVSIM PATH=$DVSIM/.venv/bin:\$PATH; cd $SRC"

[ "$HOST" = local ] || sync() { rsync -a --delete --exclude .git --exclude runs --exclude results --exclude __pycache__ --exclude vendor/dvsim/.venv --exclude vendor/dvsim/build -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' "$ROOT/" "$HOST:$DIR/src/" 2> >(grep -Ev '^(depth=|verify return|Connecting to)' >&2); }

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
