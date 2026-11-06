"""The PROGRAM page's backend ops: a small IDE service (files, search, go-to-definition / rename / hover, syntax check) and a line
shell (pty terminals). `Prog(backend).handle(tool, args)` is called from `Backend._dispatch` for every `prog.*` / `term.*` request.

These are deliberately NOT MCP tools (mcp_server.py exposes none of them): like `dvsim program` they let the window's user run and
change code, so they exist only on the in-process viewer path. All lines / columns here are 0-based; paths are relative to the
program root and confined to it (no absolute paths, no `..`, no symlink escapes).

    prog.files {}                              -> {root, files:[rel]}
    prog.read {path}                           -> {text}
    prog.write {path, text}                    -> {}
    prog.search {query, regex?, case?, word?}  -> {matches:[{path,line,col,text}], truncated}
    prog.check {path, text}                    -> {problems:[{line,col,end_col,message}]}     (python / toml / json)
    prog.define {path,line,col,text?}          -> {path,line,col} | {external} | {}          (python, jedi)
    prog.rename {path,line,col,new}            -> {edits:[{path,line,col,end_line,end_col,text}]}; the files are written here
    prog.hover {path,line,col,text?}           -> {entity?, markdown?, pins?:[{component,port,pin,connects:[{component,port,pin}]}]}
    prog.ui {cmd, ...}                         -> {}   (scripts the window: open / type / find / replace / search / goto / define / rename ...; screenshot tools)
    term.open {cwd?}                        -> {id};  term.write {id,data};  term.read {id} -> {data, alive};  term.kill {id}
"""

import atexit
import codecs
import fcntl
import itertools
import json
import keyword
import os
import re
import shutil
import signal
import struct
import subprocess
import termios
import threading
from pathlib import Path, PurePosixPath

from .library import DesignError

UI_CMDS = {
    "open", "type", "cmd", "term", "hover", "dock", "find", "find_next", "find_prev", "replace", "replace_all", "search", "goto",
    "define", "rename", "quick_open",
}  # fmt: skip
SKIP_DIRS = {"__pycache__", "node_modules", "build", "dist", "venv", "env"}
BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm", ".mkv", ".avi", ".npz", ".npy", ".so", ".o", ".a",
    ".pyc", ".pyd", ".dll", ".exe", ".bin", ".stl", ".obj", ".glb", ".gltf", ".ttf", ".otf", ".zip", ".gz", ".tar", ".xz",
    ".pdf", ".ico", ".mp3", ".wav", ".h5", ".onnx", ".pt", ".db", ".sqlite", ".mcap",
}  # fmt: skip
MAX_FILES, MAX_BYTES, MAX_MATCHES = 4000, 4_000_000, 500
TERM_COLS, TERM_ROWS, TERM_BUF = 160, 40, 4_000_000


def _need_jedi():
    try:
        import jedi

        return jedi
    except ImportError:
        raise DesignError(
            "python editing tools need jedi: pip install 'dvsim[ide]'"
        ) from None


def _inst_part(d, iid):
    for i in d.doc["instances"]:
        if i["id"] == iid:
            return i
    return None


def _end_pins(d, w, end):
    """The pin refs ('J8.8') one end of wire `w` stands for: its explicit `<end>_pins`, else every pin of the port."""
    if w.get(end + "_pins"):
        return list(w[end + "_pins"])
    iid, _, port = str(w.get(end, "")).partition(".")
    inst = _inst_part(d, iid)
    if inst is None:
        return []
    from . import pins

    try:
        return list(pins.port_pins(d._part(inst["part"]), port))
    except Exception:
        return []


def pin_connections(d, component, port=None, limit=16):
    """What a component's port(s) are wired to, pin by pin: [{component, port, pin, connects:[{component, port, pin}]}].
    `port` None = every port that has a wire. A wire between ports without pin lists gives one row with pin ''."""
    rows, wires = [], d.doc.get("wires", [])
    for w in wires:
        for me, other in (("from", "to"), ("to", "from")):
            c, _, p = str(w.get(me, "")).partition(".")
            oc, _, op = str(w.get(other, "")).partition(".")
            if c != component or (port is not None and p != port):
                continue
            mine, theirs = _end_pins(d, w, me), _end_pins(d, w, other)
            if mine and len(mine) == len(theirs):
                for a, b in zip(mine, theirs):
                    rows.append(
                        {
                            "component": c,
                            "port": p,
                            "pin": a,
                            "connects": [{"component": oc, "port": op, "pin": b}],
                        }
                    )
            else:
                rows.append(
                    {
                        "component": c,
                        "port": p,
                        "pin": "",
                        "connects": [{"component": oc, "port": op, "pin": ""}],
                    }
                )
    merged = {}
    for r in rows:  # one row per (port, pin): a pin with several wires lists every peer
        k = (r["port"], r["pin"])
        if k in merged:
            merged[k]["connects"] += r["connects"]
        else:
            merged[k] = r
    out = list(merged.values())
    if port is not None and not out:
        out = [{"component": component, "port": port, "pin": "", "connects": []}]
    return out[:limit]


# ---- the pty terminals -----------------------------------------------------------------------------------------------------
class Term:
    """One shell on a pty. A reader thread drains the master into a buffer so a chatty child never blocks on a full pipe."""

    def __init__(self, cwd, shell=None):
        self.cwd = str(cwd)
        shell = shell or os.environ.get("SHELL") or "/bin/bash"
        if not os.path.exists(shell):
            shell = shutil.which("bash") or "/bin/sh"
        name = os.path.basename(shell)
        # a line shell: the window owns the input line, so no line editor on the other side (readline / zle would echo twice)
        args = {"bash": ["--noediting", "-i"], "zsh": ["-i", "+Z"], "fish": ["-i"]}.get(
            name, ["-i"]
        )
        self.master, slave = os.openpty()
        fcntl.ioctl(
            slave, termios.TIOCSWINSZ, struct.pack("HHHH", TERM_ROWS, TERM_COLS, 0, 0)
        )
        env = {
            **os.environ,
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "DVSIM_TERMINAL": "1",
        }
        env.pop("PROMPT_COMMAND", None)

        def child():  # new session with the pty as its controlling terminal: ^C then reaches the foreground job
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(
            [shell, *args],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=self.cwd,
            env=env,
            preexec_fn=child,
            close_fds=True,
        )
        os.close(slave)
        self.buf, self.lock, self.alive = bytearray(), threading.Lock(), True
        self.dec = codecs.getincrementaldecoder("utf-8")("replace")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            try:
                data = os.read(self.master, 65536)
            except OSError:
                data = b""
            if not data:
                break
            with self.lock:
                self.buf += data
                if len(self.buf) > TERM_BUF:
                    del self.buf[: len(self.buf) - TERM_BUF]
        self.alive = False

    def write(self, data: str):
        if self.alive:
            os.write(self.master, data.encode())

    def read(self) -> str:
        with self.lock:
            raw, self.buf = bytes(self.buf), bytearray()
        return self.dec.decode(raw)

    def kill(self):
        self.alive = False
        try:
            os.killpg(self.proc.pid, signal.SIGHUP)
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.close(self.master)
        except OSError:
            pass


class Prog:
    def __init__(self, backend, root=None):
        self.b = backend
        self.explicit_root = Path(root).resolve() if root else None
        self.terms, self._ids = {}, itertools.count(1)
        self._jedi_project = self._jedi_root = None
        atexit.register(self.close)

    def close(self):
        """Kill every shell (the viewer closed)."""
        for t in list(self.terms.values()):
            t.kill()
        self.terms.clear()

    # ---- paths -------------------------------------------------------------------------------------------------------------
    def root(self) -> Path:
        """The program root: the `program_root` argument, else `examples/<name>` beside an `assemblies/<name>.toml` design (the
        convention of this repository), else the design file's directory, else the working directory."""
        if self.explicit_root:
            return self.explicit_root
        d = self.b.get_design()
        p = Path(d.path).resolve() if d is not None and d.path else None
        if p is not None:
            ex = p.parent.parent / "examples" / p.stem
            if p.parent.name == "assemblies" and ex.is_dir():
                return ex
            return p.parent
        return Path.cwd().resolve()

    def confine(self, rel, must_exist=False) -> Path:
        root = self.root()
        r = PurePosixPath(str(rel))
        if not str(rel) or r.is_absolute() or ".." in r.parts or "\x00" in str(rel):
            raise DesignError(f"path {rel!r} is outside the program root")
        p = (root / r).resolve()
        if p != root and root not in p.parents:
            raise DesignError(f"path {rel!r} is outside the program root")
        if must_exist and not p.exists():
            raise DesignError(f"{rel}: no such file")
        return p

    def rel(self, p) -> str:
        return Path(p).resolve().relative_to(self.root()).as_posix()

    def _walk(self):
        root, n = self.root(), 0
        for dp, dns, fns in os.walk(root, followlinks=False):
            dns[:] = sorted(
                x
                for x in dns
                if not x.startswith(".")
                and x not in SKIP_DIRS
                and not x.endswith(".egg-info")
            )
            for f in sorted(fns):
                if f.startswith(".") or Path(f).suffix.lower() in BINARY_EXT:
                    continue
                p = Path(dp) / f
                if p.is_symlink() and root not in p.resolve().parents:
                    continue
                yield p
                n += 1
                if n >= MAX_FILES:
                    return

    def _read_text(self, p: Path) -> str:
        if p.stat().st_size > MAX_BYTES:
            raise DesignError(f"{p.name}: larger than {MAX_BYTES // 1_000_000} MB")
        b = p.read_bytes()
        if b"\x00" in b[:8192]:
            raise DesignError(f"{p.name}: binary file")
        return b.decode("utf-8", "replace")

    # ---- dispatch ----------------------------------------------------------------------------------------------------------
    def handle(self, tool, a) -> dict:
        fn = getattr(self, "op_" + tool.replace(".", "_"), None)
        if fn is None:
            raise DesignError(f"unknown tool {tool!r}")
        return {"ok": True, "result": fn(a)}

    # ---- scripting the window --------------------------------------------------------------------------------------------
    def op_prog_ui(self, a):
        """Queue a window event for the PROGRAM page (what a person would click / type): cmd = open {path,line,col,len} |
        type {text} | cmd {name} | term {text} | hover {line,col} | dock {name} | find {text,replace?,case?,word?,regex?} |
        find_next | find_prev | replace | replace_all | search {text,case?,word?,regex?,send?} (find in files) | goto {line} |
        define (F12) | rename {new} (F2) | quick_open {text,pick?}. Screenshot tools and tests drive the page with it."""
        cmd = str(a.get("cmd", ""))
        if cmd not in UI_CMDS:
            raise DesignError(f"prog.ui: unknown cmd {cmd!r}")
        if cmd == "open":
            self.confine(a["path"], True)
        self.b.events.append(
            {**{k: v for k, v in a.items() if k != "cmd"}, "cmd": "prog_" + cmd}
        )
        return {}

    # ---- files -------------------------------------------------------------------------------------------------------------
    def op_prog_files(self, a):
        root = self.root()
        if not root.is_dir():
            raise DesignError(f"program root {root} is not a directory")
        return {"root": str(root), "files": [self.rel(p) for p in self._walk()]}

    def op_prog_read(self, a):
        return {"text": self._read_text(self.confine(a["path"], True))}

    def op_prog_write(self, a):
        p = self.confine(a["path"])
        if p.is_dir():
            raise DesignError(f"{a['path']} is a directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".dvsim-tmp")
        tmp.write_bytes(str(a["text"]).encode())
        if p.exists():
            shutil.copymode(p, tmp)
        os.replace(tmp, p)
        return {}

    def op_prog_search(self, a):
        q = str(a.get("query", ""))
        if not q:
            return {"matches": [], "truncated": False}
        flags = 0 if a.get("case") else re.IGNORECASE
        pat = q if a.get("regex") else re.escape(q)
        if a.get("word"):
            pat = rf"\b(?:{pat})\b"
        try:
            rx = re.compile(pat, flags)
        except re.error as e:
            raise DesignError(f"bad regex: {e}") from None
        out = []
        for p in self._walk():
            try:
                text = self._read_text(p)
            except (DesignError, OSError):
                continue
            for ln, line in enumerate(text.split("\n")):
                m = rx.search(line)
                if m:
                    out.append(
                        {
                            "path": self.rel(p),
                            "line": ln,
                            "col": m.start(),
                            "text": line.strip()[:200],
                        }
                    )
                    if len(out) >= MAX_MATCHES:
                        return {"matches": out, "truncated": True}
        return {"matches": out, "truncated": False}

    # ---- diagnostics -------------------------------------------------------------------------------------------------------
    def op_prog_check(self, a):
        path, text = str(a["path"]), str(a.get("text", ""))
        ext, probs = Path(path).suffix.lower(), []
        try:
            if ext in (".py", ".pyi"):
                compile(text, path, "exec", dont_inherit=True)
            elif ext == ".toml":
                import tomllib

                tomllib.loads(text)
            elif ext == ".json":
                json.loads(text)
        except SyntaxError as e:
            col = max((e.offset or 1) - 1, 0)
            end = (
                max((e.end_offset or e.offset or 1) - 1, col + 1)
                if (e.end_lineno or e.lineno) == e.lineno
                else col + 1
            )
            probs.append(
                {
                    "line": max((e.lineno or 1) - 1, 0),
                    "col": col,
                    "end_col": end,
                    "message": e.msg,
                }
            )
        except json.JSONDecodeError as e:
            probs.append(
                {
                    "line": e.lineno - 1,
                    "col": e.colno - 1,
                    "end_col": e.colno,
                    "message": e.msg,
                }
            )
        except ValueError as e:  # tomllib.TOMLDecodeError
            m = re.search(r"line (\d+), column (\d+)", str(e))
            ln, col = (int(m[1]) - 1, int(m[2]) - 1) if m else (0, 0)
            probs.append(
                {
                    "line": ln,
                    "col": col,
                    "end_col": col + 1,
                    "message": re.sub(r"\s*\(at line.*", "", str(e)),
                }
            )
        return {"problems": probs}

    # ---- python intelligence (jedi) ----------------------------------------------------------------------------------------
    def _script(self, a):
        jedi = _need_jedi()
        p = self.confine(a["path"], True)
        if p.suffix not in (".py", ".pyi"):
            raise DesignError("go to definition / rename work on python files")
        if self._jedi_project is None or self._jedi_root != self.root():
            self._jedi_root = self.root()
            self._jedi_project = jedi.Project(str(self._jedi_root))
        code = a["text"] if a.get("text") is not None else self._read_text(p)
        return jedi.Script(code=code, path=str(p), project=self._jedi_project)

    def op_prog_define(self, a):
        names = self._script(a).goto(
            a["line"] + 1, a["col"], follow_imports=True, follow_builtin_imports=False
        )
        for n in names:
            if n.module_path is None or n.line is None:
                continue
            mp = Path(n.module_path).resolve()
            if self.root() in mp.parents:
                return {"path": self.rel(mp), "line": n.line - 1, "col": n.column}
            return {"external": f"{n.full_name or n.name} is defined in {mp}"}
        return {}

    def op_prog_rename(self, a):
        new = str(a["new"])
        if not new.isidentifier() or keyword.iskeyword(new):
            raise DesignError(f"{new!r} is not a valid python name")
        sc = self._script(a)
        refs = sc.get_references(
            a["line"] + 1, a["col"], include_builtins=False, scope="project"
        )
        if not refs:
            raise DesignError("rename: no symbol under the cursor")
        edits, seen, root, olds = [], set(), self.root(), {}
        for r in refs:
            if r.module_path is None:
                continue
            mp = Path(r.module_path).resolve()
            if root not in mp.parents:
                raise DesignError(
                    f"rename: {r.name!r} is defined outside the program root ({mp}); not renaming library code"
                )
            k = (mp, r.line, r.column)
            if k in seen:
                continue
            seen.add(k)
            olds[(self.rel(mp), r.line - 1, r.column)] = r.name
            edits.append(
                {
                    "path": self.rel(mp),
                    "line": r.line - 1,
                    "col": r.column,
                    "end_line": r.line - 1,
                    "end_col": r.column + len(r.name),
                    "text": new,
                }
            )
        edits.sort(key=lambda e: (e["path"], e["line"], e["col"]))
        by_path = {}
        for e in edits:
            by_path.setdefault(e["path"], []).append(e)
        new_text = {}
        for (
            rel,
            es,
        ) in by_path.items():  # compute everything first: a failure writes nothing
            lines = self._read_text(self.confine(rel, True)).split("\n")
            for e in reversed(es):
                s = lines[e["line"]]
                if s[e["col"] : e["end_col"]] != olds[(rel, e["line"], e["col"])]:
                    raise DesignError(
                        f"rename: {rel}:{e['line'] + 1} changed on disk; save and retry"
                    )
                lines[e["line"]] = s[: e["col"]] + new + s[e["end_col"] :]
            new_text[rel] = "\n".join(lines)
        for rel, t in new_text.items():
            self.op_prog_write({"path": rel, "text": t})
        return {"edits": edits, "files": sorted(new_text)}

    # ---- hover: design entities in code ------------------------------------------------------------------------------------
    @staticmethod
    def _token_at(line: str, col: int):
        """The path-like token around `col` ('/devices/asv_imu', 'bat.power_out', 'pdb') and the plain word under it."""
        col = min(max(col, 0), len(line))
        ch = r"[\w./\-:]"
        s = col
        while s > 0 and re.match(ch, line[s - 1]):
            s -= 1
        e = col
        while e < len(line) and re.match(ch, line[e]):
            e += 1
        tok = line[s:e].strip(".:/-")
        ws = s + (
            len(line[s:e]) - len(line[s:e].lstrip(".:/-"))
        )  # word under the cursor within the token
        parts, pos = [], ws
        for m in re.finditer(r"[\w\-]+", tok):
            parts.append((m.group(), ws + m.start(), ws + m.end()))
        word = next((w for w, a, b in parts if a <= col <= b), "")
        return tok, word, parts

    def op_prog_hover(self, a):
        p = self.confine(a["path"], True)
        text = a["text"] if a.get("text") is not None else self._read_text(p)
        lines = text.split("\n")
        if not 0 <= a["line"] < len(lines):
            return {}
        tok, word, _ = self._token_at(lines[a["line"]], a["col"])
        out = {}
        d = self.b.get_design()
        if d is not None and tok:
            out = self._entity(d, tok, word)
        if not out and p.suffix == ".py" and word and not word.isdigit():
            try:  # no design entity: the python symbol's signature / first docstring line
                for h in self._script({**a, "text": text}).help(
                    a["line"] + 1, a["col"]
                )[:1]:
                    doc = (h.docstring() or "").strip()
                    if doc:
                        out = {"markdown": doc[:600]}
            except Exception:  # ttsr-ignore: py/blind-except-pass  (a hover must never fail: jedi raises many kinds on odd code)
                pass
        return out

    @staticmethod
    def _entity(d, tok, word):
        doc = d.doc
        insts = {i["id"]: i for i in doc["instances"]}
        # the runtime path of a device is /devices/<assembly>_<name> (skydio_imu, asv_imu); the bare name works too
        devs = {v["name"]: v for v in doc.get("devices", [])}
        devs.update({f"{d.asm.name}_{n}": v for n, v in list(devs.items())})
        segs = [s for s in re.split(r"[/:]", tok) if s]
        # 1. a device by its runtime name ('/devices/asv_imu', or the bare name in a string / variable)
        dev = devs.get(word) or next(
            (devs[s] for s in reversed(segs) if s in devs), None
        )
        if dev is not None and word not in insts:
            iid, port = dev.get("instance", ""), dev.get("port") or dev.get("a") or ""
            part = insts.get(iid, {}).get("part", "")
            rows = (
                pin_connections(d, iid)
                if not dev.get("port") and not dev.get("a")
                else []
            )
            for pn in {x for x in (dev.get("port"), dev.get("a"), dev.get("b")) if x}:
                rows += pin_connections(d, iid, pn)
            md = (
                f"/devices/{d.asm.name}_{dev['name']}  type {dev['type']}\non component {iid}"
                + (f", port {port}" if port else "")
            )
            return {
                "entity": {
                    "kind": "device",
                    "id": dev["name"],
                    "instance": iid,
                    "part": part,
                },
                "markdown": md,
                "pins": rows[:16],
            }
        # 2. 'inst.port' / 'inst.CONN.PIN' references
        if "." in tok:
            iid, _, rest = tok.partition(".")
            if iid in insts:
                port = rest.split(".")[0]
                inst = insts[iid]
                try:
                    ports = [q["name"] for q in d._part(inst["part"]).ports]
                except Exception:
                    ports = []
                if port in ports:
                    rows = pin_connections(d, iid, port)
                    return {
                        "entity": {
                            "kind": "port",
                            "id": f"{iid}.{port}",
                            "instance": iid,
                            "part": inst["part"],
                        },
                        "markdown": f"port {port} of component {iid}",
                        "pins": rows,
                    }
        # 3. a component by its instance id
        if word in insts:
            inst = insts[word]
            rows = pin_connections(d, word)
            return {
                "entity": {"kind": "instance", "id": word, "part": inst["part"]},
                "markdown": f"component {word}"
                + (f", group {inst['group']}" if inst.get("group") else ""),
                "pins": rows,
            }
        return {}

    # ---- terminals ---------------------------------------------------------------------------------------------------------
    def _term(self, a) -> Term:
        t = self.terms.get(a.get("id"))
        if t is None:
            raise DesignError(f"no terminal {a.get('id')!r}")
        return t

    def op_term_open(self, a):
        cwd = self.confine(a["cwd"], True) if a.get("cwd") else self.root()
        if not cwd.is_dir():
            cwd = self.root()
        tid = next(self._ids)
        self.terms[tid] = Term(cwd)
        return {"id": tid, "cwd": str(cwd)}

    def op_term_write(self, a):
        self._term(a).write(str(a.get("data", "")))
        return {}

    def op_term_read(self, a):
        t = self._term(a)
        data = t.read()
        return {"data": data, "alive": t.alive or bool(data)}

    def op_term_kill(self, a):
        t = self.terms.pop(a.get("id"), None)
        if t is not None:
            t.kill()
        return {}
