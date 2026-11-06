"""Parameters & telemetry: a scenario declares the knobs a user may turn and the channels a program streams, a UI (or the CLI) lists, edits and plots them.

Scenario TOML (extra top-level tables are ignored by `dvsim program`, so this is additive):

    [[parameters]]                    # one knob
    name = "wind_ms"                  # unique; the key in DVSIM_PARAMS
    type = "float"                    # float | int | bool | choice
    default = 2.0
    min = 0.0  max = 8.0              # float / int only (inclusive)
    # choices = ["a", "b"]            # choice only
    unit = "m/s"  group = "environment"  label = "Mean wind"  doc = "text shown as the tooltip"

    [[telemetry]]                     # one channel of the JSON-lines stream (`telemetry.jsonl` in the run dir)
    name = "est.alt_m"                # dotted path into each record
    unit = "m"  label = "Altitude (estimate)"  group = "state"  kind = "number"   # number | string | bool
    min = 0  max = 30                 # optional plot range
    doc = "..."

Contract (JSON, what a UI exchanges):
  describe(scenario)  -> {"parameters": [ {name, type, default, min?, max?, choices?, unit, group, label, doc} ... ],
                          "telemetry":  [ {name, unit, label, group, kind, min?, max?, doc} ... ]}
  A run is started with the environment variable DVSIM_PARAMS = a JSON object {name: value} (unset names take their defaults; unknown names and
  out-of-range values are errors) -- the program calls `values(scenario)` to read them back.
  The telemetry stream is one JSON object per line: {"t": <sim s>, ...nested fields...}; `get(record, "est.alt_m")` resolves a channel.
"""

import json
import math
import os
import tomllib
from pathlib import Path

PARAM_TYPES = ("float", "int", "bool", "choice")
CHANNEL_KINDS = ("number", "string", "bool")


def default_workers(gb_per_worker: float = 1.0) -> int:
    """Default worker count for a Monte Carlo driver: DVSIM_THREADS (the job's CPU slots, tools/remote.sh) or every core,
    bounded by MemAvailable / gb_per_worker."""
    cpus = int(os.environ.get("DVSIM_THREADS") or os.cpu_count() or 1)
    try:
        with open("/proc/meminfo") as f:
            kb = next(
                int(line.split()[1]) for line in f if line.startswith("MemAvailable:")
            )
        return max(1, min(cpus, int(kb / 2**20 / gb_per_worker)))
    except (OSError, StopIteration):
        return cpus


class ParamError(ValueError):
    pass


def _load(scenario):
    try:
        return tomllib.loads(Path(scenario).read_text())
    except (OSError, ValueError) as e:
        raise ParamError(f"cannot read scenario '{scenario}': {e}") from None


def declared(scenario):
    """-> (parameters, telemetry) lists of dicts as declared (validated)."""
    sc = _load(scenario)
    params, seen = [], set()
    for i, p in enumerate(sc.get("parameters", [])):
        n = p.get("name")
        if not n or n in seen:
            raise ParamError(f"[[parameters]] {i}: name missing or duplicated ({n!r})")
        seen.add(n)
        t = p.get("type")
        if t not in PARAM_TYPES:
            raise ParamError(f"parameter '{n}': type must be one of {PARAM_TYPES}")
        if "default" not in p:
            raise ParamError(f"parameter '{n}': default is required")
        if t == "choice" and not p.get("choices"):
            raise ParamError(f"parameter '{n}': choice needs choices = [...]")
        q = {"unit": "", "group": "", "label": n, "doc": "", **p}
        _check_value(q, p["default"])
        params.append(q)
    chans, seen = [], set()
    for i, c in enumerate(sc.get("telemetry", [])):
        n = c.get("name")
        if not n or n in seen:
            raise ParamError(f"[[telemetry]] {i}: name missing or duplicated ({n!r})")
        seen.add(n)
        if c.get("kind", "number") not in CHANNEL_KINDS:
            raise ParamError(f"channel '{n}': kind must be one of {CHANNEL_KINDS}")
        chans.append(
            {"unit": "", "group": "", "label": n, "doc": "", "kind": "number", **c}
        )
    return params, chans


def describe(scenario):
    """The JSON a UI reads."""
    p, c = declared(scenario)
    return {"parameters": p, "telemetry": c}


def _check_value(spec, v):
    n, t = spec["name"], spec["type"]
    if t == "bool":
        if not isinstance(v, bool):
            raise ParamError(f"parameter '{n}': expected a bool, got {v!r}")
    elif t == "choice":
        if v not in spec["choices"]:
            raise ParamError(f"parameter '{n}': {v!r} not in {spec['choices']}")
    else:
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
        ):
            raise ParamError(f"parameter '{n}': expected a number, got {v!r}")
        if t == "int" and v != int(v):
            raise ParamError(f"parameter '{n}': expected an integer, got {v!r}")
        if "min" in spec and v < spec["min"] or "max" in spec and v > spec["max"]:
            raise ParamError(
                f"parameter '{n}': {v} outside [{spec.get('min', '-inf')}, {spec.get('max', 'inf')}]"
            )


def resolve(scenario, overrides=None):
    """Values of every declared parameter: defaults overlaid with `overrides` (validated; unknown names are errors)."""
    params, _ = declared(scenario)
    spec = {p["name"]: p for p in params}
    overrides = overrides or {}
    if bad := sorted(set(overrides) - set(spec)):
        raise ParamError(f"unknown parameters {bad} (declared: {sorted(spec)})")
    out = {}
    for n, p in spec.items():
        v = overrides.get(n, p["default"])
        _check_value(p, v)
        out[n] = (
            int(v) if p["type"] == "int" else float(v) if p["type"] == "float" else v
        )
    return out


def env_overrides(environ=None):
    """The overrides a UI passed in DVSIM_PARAMS (JSON object), {} when unset."""
    raw = (environ or os.environ).get("DVSIM_PARAMS", "")
    if not raw:
        return {}
    try:
        o = json.loads(raw)
    except ValueError as e:
        raise ParamError(f"DVSIM_PARAMS is not JSON: {e}") from None
    if not isinstance(o, dict):
        raise ParamError("DVSIM_PARAMS must be a JSON object")
    return o


def values(scenario, environ=None):
    """What a program calls: the declared parameters with DVSIM_PARAMS applied."""
    return resolve(scenario, env_overrides(environ))


def get(record, name):
    """Value of the channel `name` ('a.b.c') in a telemetry record; None when absent."""
    for k in name.split("."):
        if not isinstance(record, dict) or k not in record:
            return None
        record = record[k]
    return record


class TelemetryWriter:
    """<run dir>/telemetry.jsonl, one compact JSON object per line, flushed per record; `callbacks` get every record (a live panel)."""

    def __init__(self, path, callbacks=()):
        self.f = open(path, "w")
        self.callbacks = list(callbacks)
        self.n = 0

    def emit(self, rec):
        self.f.write(json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n")
        self.f.flush()
        self.n += 1
        for cb in self.callbacks:
            cb(rec)

    def close(self):
        self.f.close()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def channel_series(records, name):
    """[(t, value)] of a channel over a list of records, skipping records without it."""
    return [(r["t"], v) for r in records if (v := get(r, name)) is not None]


def main(argv=None):
    """Command line of the contract, what a panel shells out to:
    python -m dvsim.params describe <scenario.toml>             -> the describe() JSON
    python -m dvsim.params table <telemetry.jsonl> <ch>[,<ch>]  -> a text table of channels, one row per record (the plain panel)"""
    import sys

    a = list(sys.argv[1:] if argv is None else argv)
    out = sys.stdout.write
    if len(a) == 2 and a[0] == "describe":
        out(json.dumps(describe(a[1]), indent=1) + "\n")
    elif len(a) == 3 and a[0] == "table":
        recs, names = read_jsonl(a[1]), a[2].split(",")
        out(" ".join(["t".rjust(8), *(n[-14:].rjust(14) for n in names)]) + "\n")
        for r in recs:
            cells = [f"{r['t']:8.3f}"]
            for n in names:
                c = get(r, n)
                cells.append(
                    f"{c:14.3f}" if isinstance(c, (int, float)) else str(c).rjust(14)
                )
            out(" ".join(cells) + "\n")
    else:
        out(main.__doc__ + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
