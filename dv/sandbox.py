"""Agent controller sandbox (docs/DESIGN.md, "Agent code on a shared box"). Layers, none of them a true sandbox:
AST allowlist before anything runs, a child that drops to uid nobody, empty env and cwd, rlimits (no files, no processes, CPU, memory),
restricted builtins, a per-step wall deadline in the parent. The parent never unpickles child output: JSON lines, validated.

Run as a script (`--child`) it is the child. Parent side: check_source(), Controller."""

import ast
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import time

NOBODY = 65534
ALLOWED_IMPORTS = {"math", "numpy", "numpy.linalg", "numpy.fft", "numpy.random"}
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "input", "getattr", "setattr", "delattr", "hasattr", "globals", "locals", "vars", "dir", "breakpoint", "help", "memoryview",
    "type", "object", "classmethod", "staticmethod", "property", "exit", "quit", "id", "bytearray", "bytes", "license", "credits", "copyright",
}  # fmt: skip
BANNED_ATTRS = {
    # file and process surfaces of numpy
    "os", "sys", "subprocess", "builtins", "importlib", "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "fromregex", "memmap", "DataSource", "dump", "dumps",
    "ctypes", "ctypeslib", "f2py", "distutils", "testing", "test", "show_config", "show_runtime", "get_include", "lookfor", "source", "info", "who", "from_dlpack",
    "setflags", "getfield", "setfield", "seterrcall", "frombuffer", "tobytes", "fromstring", "byte_bounds", "add_newdoc", "deprecate", "safe_eval",
    # string formatting walks attributes from a string, frames and generators reach globals
    "format", "format_map", "mro", "tb_frame", "tb_next", "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "with_traceback",
}  # fmt: skip
SAFE_NUMPY_MODULES = {"linalg", "fft", "random"}
MAX_SOURCE = 200_000


def _numpy_module_attrs():
    """Every module reachable as an attribute of the numpy namespaces we allow (np.os, np.lib, np._core, ...): banned attribute names, except the three safe sub-modules."""
    import types

    import numpy as np

    out = set()
    for ns in (np, np.linalg, np.fft, np.random):
        out |= {k for k, v in vars(ns).items() if isinstance(v, types.ModuleType)}
    return out - SAFE_NUMPY_MODULES


def check_source(code, banned_attrs=None):
    """-> [{line, message}], empty when the source may run. Runs in the parent (for the agent's error) and again in the child."""
    if len(code) > MAX_SOURCE:
        return [
            dict(
                line=1, message=f"file too large ({len(code)} bytes, max {MAX_SOURCE})"
            )
        ]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [dict(line=e.lineno or 1, message=f"syntax error: {e.msg}")]
    banned = BANNED_ATTRS | (
        _numpy_module_attrs() if banned_attrs is None else banned_attrs
    )
    errs = []

    def bad(n, msg):
        errs.append(dict(line=getattr(n, "lineno", 1), message=msg))

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name not in ALLOWED_IMPORTS:
                    bad(n, f"import of '{a.name}' is not allowed: only math and numpy")
        elif isinstance(n, ast.ImportFrom):
            if n.level or n.module not in ALLOWED_IMPORTS:
                bad(n, f"import from '{n.module}' is not allowed: only math and numpy")
            for a in n.names:
                if a.name == "*":
                    bad(n, "'import *' is not allowed")
                elif (
                    a.name.startswith("_") or a.name in banned or a.name in BANNED_NAMES
                ):
                    bad(n, f"importing '{a.name}' is not allowed")
        elif isinstance(n, ast.Name):
            if n.id in BANNED_NAMES:
                bad(n, f"'{n.id}' is not allowed")
            elif n.id.startswith("__"):
                bad(n, f"dunder name '{n.id}' is not allowed")
        elif isinstance(n, ast.Attribute):
            if n.attr.startswith("__"):
                bad(n, f"dunder attribute '{n.attr}' is not allowed")
            elif n.attr in banned:
                bad(
                    n,
                    f"attribute name '{n.attr}' is not allowed (rename it if it is your own)",
                )
        elif isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)):
            if ("__" if isinstance(n.value, str) else b"__") in n.value:
                bad(n, "strings containing a double underscore are not allowed")
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            if n.name.startswith("__") and n.name != "__init__":
                bad(n, f"defining '{n.name}' is not allowed")
            if isinstance(n, ast.ClassDef) and n.keywords:
                bad(n, "class keywords (metaclass=...) are not allowed")
        elif isinstance(
            n,
            (
                ast.AsyncFunctionDef,
                ast.AsyncFor,
                ast.AsyncWith,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            bad(n, "async code and generators are not allowed")
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            for name in n.names:
                if name.startswith("__"):
                    bad(n, f"dunder name '{name}' is not allowed")
        elif isinstance(n, ast.alias) and n.asname and n.asname.startswith("__"):
            bad(n, f"dunder alias '{n.asname}' is not allowed")
    if not errs and not any(
        isinstance(n, ast.ClassDef) and n.name == "Controller" for n in tree.body
    ):
        errs.append(
            dict(
                line=1,
                message="the file must define `class Controller` at the top level",
            )
        )
    return errs


def scrub(msg):
    return str(msg)[:400]


# ---------------------------------------------------------------------------------------------------------------- child
def _lockdown():
    import resource

    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(NOBODY)
        os.setuid(NOBODY)
        if os.getuid() != NOBODY or os.geteuid() != NOBODY:
            raise SystemExit("could not drop to nobody")
    elif not os.environ.get("DV_SANDBOX_NOROOT"):
        raise SystemExit(
            "sandbox refuses to run agent code: not root, so it cannot drop to nobody"
        )
    gb = 1 << 30
    for lim, v in (
        (resource.RLIMIT_NOFILE, 3),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_NPROC, 0),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_AS, 2 * gb),
        (resource.RLIMIT_CPU, 300),
    ):
        resource.setrlimit(lim, (v, v))


class NS:
    """cfg and s of the controller: attribute and item access, lists of numbers are numpy arrays."""

    def __init__(self, d, np):
        for k, v in d.items():
            setattr(self, k, _conv(v, np))

    def __getitem__(self, k):
        return getattr(self, k)


def _conv(v, np):
    if isinstance(v, dict):
        return NS(v, np)
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return np.array(v, float)
    if isinstance(v, list):
        return [_conv(x, np) for x in v]
    return v


def _err_line(e):
    tb = e.__traceback__
    line = None
    while tb:
        if tb.tb_frame.f_code.co_filename == "controller.py":
            line = tb.tb_lineno
        tb = tb.tb_next
    return line


def child_main():
    import builtins

    import numpy as np
    import numpy.fft  # everything lazy is loaded before the uid drop
    import numpy.linalg

    # whatever numpy loads lazily is loaded now: after the lockdown no file can be opened
    import numpy.ma
    import numpy.polynomial
    import numpy.random

    a = np.eye(3)
    (
        np.median(a),
        np.percentile(a, 50),
        np.linalg.pinv(a),
        np.linalg.solve(a, a),
        np.linalg.norm(a),
        np.fft.fft(a),
        np.random.default_rng(0).normal(),
        a.min(),
        a.max(),
        a.mean(),
        a.std(),
        np.cross(a, a),
        np.interp(0.5, [0, 1], [0, 1]),
        str(a),
    )
    banned = _numpy_module_attrs()
    os.chdir(sys.argv[2])
    _lockdown()
    stdin, stdout = sys.stdin, sys.stdout

    def send(o):
        stdout.write(json.dumps(o) + "\n")
        stdout.flush()

    try:
        job = json.loads(stdin.readline())
        errs = check_source(job["code"], banned)
        if errs:
            send(dict(ok=False, error=errs[0]["message"], line=errs[0]["line"]))
            return

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            # numpy's C code imports its own internals through the calling frame's builtins, so an already loaded numpy.* module must pass; the agent's own import statements were checked by the AST
            internal = name in sys.modules and name.split(".")[0] == "numpy"
            if level or not (name in ALLOWED_IMPORTS or internal):
                raise ImportError(f"import of {name} is not allowed")
            return builtins.__import__(name, globals, locals, fromlist, level)

        keep = [
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "frozenset",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "pow",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "slice",
            "sorted",
            "str",
            "sum",
            "super",
            "tuple",
            "zip",
            "None",
            "True",
            "False",
            "ArithmeticError",
            "AttributeError",
            "Exception",
            "FloatingPointError",
            "IndexError",
            "KeyError",
            "LookupError",
            "NameError",
            "OverflowError",
            "RuntimeError",
            "StopIteration",
            "TypeError",
            "ValueError",
            "ZeroDivisionError",
            "ImportError",
        ]
        safe = {k: getattr(builtins, k) for k in keep}
        safe.update(
            __build_class__=builtins.__build_class__,
            __import__=guarded_import,
            print=lambda *a, **k: None,
        )
        ns = {"__builtins__": safe, "__name__": "controller"}
        exec(compile(job["code"], "controller.py", "exec"), ns)
        ctl = ns["Controller"](_conv(job["cfg"], np))
    except BaseException as e:  # noqa: BLE001  # agent code: anything can be raised
        send(dict(ok=False, error=f"{type(e).__name__}: {scrub(e)}", line=_err_line(e)))
        return
    send(dict(ok=True))
    step = ctl.step
    while True:
        raw = stdin.readline()
        if not raw:
            return
        try:
            m = json.loads(raw)
            s = _conv(m["s"], np)
            cpu0 = time.process_time()
            out = step(m["t"], s)
            duty, vanes = float(out["duty"]), [float(x) for x in out["vanes"]]
            cpu = time.process_time() - cpu0
            if len(vanes) != 4:
                raise ValueError("vanes must have 4 entries")
            send(dict(duty=duty, vanes=vanes, cpu=cpu))
        except BaseException as e:  # noqa: BLE001
            send(dict(error=f"{type(e).__name__}: {scrub(e)}", line=_err_line(e)))
            return


# ---------------------------------------------------------------------------------------------------------------- parent
class Controller:
    """One sandboxed controller for one flight. step() -> (duty, vanes): holds the last command when step() used more than deadline_s of CPU; after `dead` is set the command is zero
    (motor off). The sim waits for the reply, so lateness is CPU time, not wall time: other jobs on a shared box cannot fail a flight or change it. No reply in hang_s of wall time = hung.
    Fields: late (steps held), steps, error (first error text, with the line), dead."""

    def __init__(self, code, cfg, deadline_s=0.025, init_s=5.0, hang_s=1.0):
        self.deadline_s, self.hang_s = deadline_s, hang_s
        self.late = self.steps = 0
        self.error = None
        self.dead = False
        self.last = (0.0, [0.0, 0.0, 0.0, 0.0])
        self.buf = b""
        self.cwd = tempfile.mkdtemp(prefix="dv-empty-")
        os.chmod(self.cwd, 0o555)
        env = dict(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        if os.environ.get("DV_SANDBOX_NOROOT"):
            env["DV_SANDBOX_NOROOT"] = "1"
        self.p = subprocess.Popen(
            [sys.executable, "-I", os.path.abspath(__file__), "--child", self.cwd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=self.cwd,
            bufsize=0,
        )
        self._send(dict(code=code, cfg=cfg))
        r = self._recv(init_s)
        if r is None or not r.get("ok"):
            why = (
                "__init__ or import took too long"
                if r is None
                else r.get("error", "failed")
            )
            self._die(
                f"controller did not start: {why}"
                + (f" (line {r['line']})" if r and r.get("line") else "")
            )

    def _send(self, o):
        try:
            self.p.stdin.write((json.dumps(o) + "\n").encode())
        except (BrokenPipeError, OSError):
            self._die("controller process died")

    def _recv(self, timeout):
        """One JSON line or None at the deadline; a dead child sets the error."""
        end = time.perf_counter() + timeout
        while b"\n" not in self.buf:
            left = end - time.perf_counter()
            if left < 0:
                return None
            r, _, _ = select.select([self.p.stdout], [], [], left)
            if not r:
                return None
            chunk = os.read(self.p.stdout.fileno(), 65536)
            if not chunk:
                self._die(
                    "controller process died (out of memory or CPU limit, or killed)"
                )
                return dict(error=self.error)
            self.buf += chunk
            if len(self.buf) > 1 << 20:
                self._die("controller wrote too much output")
                return dict(error=self.error)
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            r = json.loads(line)
            return r if isinstance(r, dict) else dict(error="bad reply")
        except ValueError:
            return dict(error="bad reply")

    def _die(self, why):
        if self.error is None:
            self.error = why
        self.dead = True
        self.last = (0.0, [0.0, 0.0, 0.0, 0.0])
        self.close()

    def step(self, t, s):
        self.steps += 1
        if self.dead:
            return self.last
        self._send(dict(t=t, s=s))
        if self.dead:
            return self.last
        r = self._recv(self.hang_s)
        if r is None:
            self.late += 1
            self._die(
                f"controller hung at t={t:.2f} s: step() did not return within {self.hang_s} s"
            )
            return self.last
        if "error" in r:
            self._die(
                f"step() raised at t={t:.2f} s: {r['error']}"
                + (f" (line {r['line']})" if r.get("line") else "")
            )
            return self.last
        try:
            duty, vanes = float(r["duty"]), [float(v) for v in r["vanes"]]
            if not (
                math.isfinite(duty)
                and all(math.isfinite(v) for v in vanes)
                and len(vanes) == 4
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            self._die(f"step() returned NaN, inf or a bad command at t={t:.2f} s")
            return self.last
        if r.get("cpu", 0.0) > self.deadline_s:
            self.late += 1
            return self.last
        self.last = (duty, vanes)
        return self.last

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
        try:
            self.p.wait(timeout=2)
            self.p.stdin.close()
            self.p.stdout.close()
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            os.rmdir(self.cwd)
        except OSError:
            pass


if __name__ == "__main__" and "--child" in sys.argv:
    child_main()
