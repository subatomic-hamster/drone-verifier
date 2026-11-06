"""Escapes that must fail, and a legal numpy controller that must work. Run on hfbox as root (`pytest tests/test_sandbox.py`): the uid drop is part of what is tested."""

import os
import signal
import stat
import tempfile
import threading

import pytest
from dv.sandbox import Controller, check_source

CFG = dict(mass=1.2, inertia=[0.01, 0.01, 0.01])
S = dict(pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], height=None, armed=True)
SHELL = "class Controller:\n    def __init__(self, cfg):\n{init}\n    def step(self, t, s):\n{step}\n        return dict(duty=0.5, vanes=[0.0, 0.0, 0.0, 0.0])\n"


def src(init="        pass", step="        pass", head=""):
    return head + SHELL.format(init=init, step=step)


def run(code, n=3, **kw):
    c = Controller(code, CFG, **kw)
    out = None
    for i in range(n):
        out = c.step(i * 0.004, S)
    c.close()
    return c, out


@pytest.mark.parametrize(
    "code",
    [
        "import os\nclass Controller: pass",
        "import subprocess\nclass Controller: pass",
        "from os import system\nclass Controller: pass",
        "import numpy.ctypeslib\nclass Controller: pass",
        src("        open('/etc/shadow')"),
        src("        eval('1')"),
        src("        self.x = ().__class__.__bases__"),
        src("        getattr(self, 'x')"),
        src("        x = '__class__'"),
        src("        import numpy as np\n        np.load('/etc/shadow')"),
        src("        import numpy as np\n        np.fromfile('/etc/shadow')"),
        src("        import numpy as np\n        np.os.system('id')"),
        src("        import numpy as np\n        np.lib.npyio"),
        src("        x = '{0.__class__}'.format(1)"),
        src("        type('x', (), {})"),
        "class Controller(metaclass=abs): pass",
        "x = 1",
    ],
)
def test_rejected_by_ast(code):
    assert check_source(code), code


def test_child_cannot_read_root_only_file_even_past_the_ast():
    """The uid and rlimits hold if the AST check is bypassed: the child itself, run with an empty ban list, tries a real open."""
    fd, path = tempfile.mkstemp(prefix="dv-secret-")
    os.write(fd, b"secret")
    os.close(fd)
    os.chmod(path, stat.S_IRUSR)
    # numpy's file reader is the one route left when a name slips the AST: drive the child's own lockdown by a probe script
    import subprocess
    import sys

    probe = (
        "import os,sys,resource\n"
        "sys.path.insert(0,sys.argv[1])\n"
        "from dv import sandbox\n"
        "sandbox._lockdown()\n"
        "print('uid',os.getuid())\n"
        "for name,fn in dict(open=lambda:open(sys.argv[2]).read(), write=lambda:open('/tmp/dv-x','w').write('x'), fork=os.fork, sh=lambda:os.system('id')).items():\n"
        "    try:\n"
        "        r=fn(); print(name,'ESCAPED',r)\n"
        "        if r==0: os._exit(0)\n"
        "    except OSError as e: print(name,'denied',e.errno)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe, os.getcwd(), path],
        capture_output=True,
        text=True,
        cwd="/",
    )
    os.unlink(path)
    print(r.stdout, r.stderr)
    assert "uid 65534" in r.stdout
    # os.system reports a spawn that RLIMIT_NPROC refused as exit status 127 (32512), not as an OSError: the shell never ran
    assert (
        "open ESCAPED" not in r.stdout
        and "write ESCAPED" not in r.stdout
        and "fork ESCAPED" not in r.stdout
    )
    assert "sh ESCAPED 0" not in r.stdout
    assert (
        "open denied" in r.stdout
        and "write denied" in r.stdout
        and "fork denied" in r.stdout
    )


def test_nobody_can_import_numpy_from_the_venv():
    import subprocess
    import sys

    fd, path = tempfile.mkstemp(prefix="dv-secret-")
    os.close(fd)
    os.chmod(path, stat.S_IRUSR)
    probe = (
        "import os,sys\nos.setgroups([]);os.setgid(65534);os.setuid(65534)\nimport numpy,numpy.linalg,numpy.random\nprint(os.getuid(),numpy.__version__)\n"
        "try:\n    open(sys.argv[1]); print('READ ROOT FILE')\nexcept PermissionError: print('root-only file denied by the uid alone')\n"
    )
    r = subprocess.run(
        [sys.executable, "-I", "-c", probe, path],
        capture_output=True,
        text=True,
        cwd="/",
        env=dict(OMP_NUM_THREADS="1"),
    )
    os.unlink(path)
    print(r.stdout, r.stderr)
    assert r.stdout.startswith("65534 "), r.stderr
    assert "root-only file denied by the uid alone" in r.stdout


@pytest.mark.parametrize(
    "body",
    [
        "        while True: pass",
        "        x = [0.0] * (10 ** 11)\n        return x",
    ],
)
def test_hang_or_memory_in_step_ends_the_controller(body):
    c, out = run(src(step=body), n=80, deadline_s=0.02, hang_s=0.3)
    assert c.dead and c.error
    assert out == (0.0, [0.0, 0.0, 0.0, 0.0])


def test_lateness_is_cpu_time_not_wall_time():
    """A stalled child (a busy shared box) is not late; a step that burns CPU is, and its command is dropped."""
    code = (
        "class Controller:\n"
        "    def __init__(self, cfg):\n"
        "        self.n = 0\n"
        "    def step(self, t, s):\n"
        "        self.n += 1\n"
        "        x = 0\n"
        "        for i in range(5000000 if self.n == 2 else 0):\n"
        "            x += i\n"
        "        return dict(duty=self.n / 10.0, vanes=[0.0, 0.0, 0.0, 0.0])\n"
    )
    c = Controller(code, CFG, deadline_s=0.02)
    os.kill(c.p.pid, signal.SIGSTOP)
    threading.Timer(0.2, os.kill, (c.p.pid, signal.SIGCONT)).start()
    assert c.step(0.0, S)[0] == 0.1 and c.late == 0  # stalled 0.2 s, no CPU used
    assert c.step(0.004, S)[0] == 0.1 and c.late == 1  # burned CPU: held
    assert c.step(0.008, S)[0] == 0.3 and c.late == 1
    c.close()


def test_hang_in_init_does_not_start():
    c = Controller(src("        while True: pass"), CFG, init_s=0.5)
    assert c.dead and "did not start" in c.error
    c.close()


def test_exception_reports_the_line():
    c, _ = run(src(step="        raise ValueError('boom')"))
    assert c.dead and "boom" in c.error and "line" in c.error


def test_nan_command_kills_the_flight():
    code = "class Controller:\n    def __init__(self, cfg): pass\n    def step(self, t, s): return dict(duty=float('nan'), vanes=[0.0, 0.0, 0.0, 0.0])\n"
    c, _ = run(code)
    assert c.dead and "NaN" in c.error


def test_legal_numpy_controller_works():
    code = (
        "import math\nimport numpy as np\n"
        "class Controller:\n"
        "    def __init__(self, cfg):\n"
        "        self.m = cfg.mass; self.I = np.diag(cfg.inertia); self._h = np.linalg.inv(self.I)\n"
        "    def step(self, t, s):\n"
        "        a = np.linalg.pinv(np.eye(3)) + np.cross(np.eye(3), np.eye(3)); b = float(a.min() + a.max() + a.std() + np.median(a) + np.interp(0.5, [0, 1], [0, 1]) + np.random.default_rng(1).normal())\n"
        "        c = str(a) + repr(np.percentile(a, 50))\n"
        "        d = float(np.clip(self.m * 0.3 + math.sin(t) * 0.0, 0, 1))\n"
        "        return {'duty': d, 'vanes': list(np.zeros(4))}\n"
    )
    c, out = run(code)
    assert c.error is None, c.error
    assert out[0] == pytest.approx(0.36) and out[1] == [0.0] * 4


def test_child_really_runs_as_nobody():
    code = (
        "class Controller:\n"
        "    def __init__(self, cfg):\n"
        "        pass\n"
        "    def step(self, t, s):\n"
        "        return dict(duty=0.0, vanes=[0.0, 0.0, 0.0, 0.0])\n"
    )
    c = Controller(code, CFG)
    uid = os.stat(f"/proc/{c.p.pid}").st_uid
    c.close()
    assert uid == 65534
