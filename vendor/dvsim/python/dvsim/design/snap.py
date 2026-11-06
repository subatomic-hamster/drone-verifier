"""Snapping: port frames, mate transforms, compatible-port search, kinematic tree solve, fastener selection.

Design rules (constants of the method, not physical data):
  ENGAGEMENT_FACTOR = 1.5   tapped hole: thread engagement = 1.5 x nominal diameter (rule of thumb, a design rule)
  PROTRUSION_PITCHES = 2    through-bolt: the screw must stick out of the nut by 2 thread pitches
"""

import math
import re
from dataclasses import dataclass

import numpy as np

from .library import DesignError, Library, Part, genders_ok, types_ok

ENGAGEMENT_FACTOR = 1.5
PROTRUSION_PITCHES = 2
EPS_LEN = 1e-9  # slack on length >= L_min comparisons (float noise only)
RX_PI = np.diag([1.0, -1.0, -1.0, 1.0])  # rotation by pi about x: z -> -z, y -> -y


# ---- frames -----------------------------------------------------------------------------------------------------
def _vec3(v, what: str) -> np.ndarray:
    try:
        a = np.array(v, dtype=float)
    except (TypeError, ValueError):
        raise DesignError(f"{what}: not a 3-vector: {v!r}") from None
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise DesignError(f"{what}: not a 3-vector: {v!r}")
    return a


def inv_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def Rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def port_frame(port: dict, where: str = "") -> np.ndarray:
    """4x4 P: columns x' = ref, y' = axis x ref, z' = axis, origin = pos (all in the part frame)."""
    w = f"{where}port '{port.get('name', '?')}'"
    if "axis" not in port or "ref" not in port:
        raise DesignError(f"{w}: mech port needs 'axis' and 'ref'")
    z = _vec3(port["axis"], f"{w} axis")
    x = _vec3(port["ref"], f"{w} ref")
    if abs(np.linalg.norm(z) - 1) > 1e-6:
        raise DesignError(f"{w}: axis is not a unit vector")
    if abs(np.linalg.norm(x) - 1) > 1e-6:
        raise DesignError(f"{w}: ref is not a unit vector")
    if abs(z @ x) > 1e-6:
        raise DesignError(f"{w}: ref is not perpendicular to axis (dot = {z @ x:.3g})")
    z = z / np.linalg.norm(z)
    x = (
        x - (x @ z) * z
    )  # remove the <= 1e-6 residue so the frame is exactly orthonormal
    x = x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    T[:3, 3] = _vec3(port.get("pos", [0, 0, 0]), f"{w} pos")
    return T


def symmetry(port: dict) -> int:
    n = port.get("symmetry", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise DesignError(
            f"port '{port.get('name', '?')}': symmetry must be an integer >= 1, got {n!r}"
        )
    return n


# ---- mating -----------------------------------------------------------------------------------------------------
def incompatible(pa: dict, pb: dict) -> str | None:
    """Reason the two mech ports cannot mate, or None."""
    if not types_ok(pa.get("type", ""), pb.get("type", "")):
        return f"type mismatch: {pa.get('type')!r} vs {pb.get('type')!r}"
    if not genders_ok(pa.get("gender", ""), pb.get("gender", "")):
        return f"gender mismatch: {pa.get('gender')!r} with {pb.get('gender')!r}"
    return None


def n_mate(pa: dict, pb: dict) -> int:
    return math.gcd(symmetry(pa), symmetry(pb))


def mate_transform(Ta: np.ndarray, pa: dict, pb: dict, k: int = 0) -> np.ndarray:
    """T_world_b = T_world_a P_a Rz(2 pi k / n) Rx(pi) inv(P_b); n = gcd(symmetries), 0 <= k < n."""
    why = incompatible(pa, pb)
    if why:
        raise DesignError(
            f"cannot mate '{pa.get('name')}' with '{pb.get('name')}': {why}"
        )
    n = n_mate(pa, pb)
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n:
        raise DesignError(
            f"rotation {k!r} out of range: ports '{pa.get('name')}'/'{pb.get('name')}' allow 0..{n - 1}"
        )
    return Ta @ port_frame(pa) @ Rz(2 * math.pi * k / n) @ RX_PI @ inv_T(port_frame(pb))


def find_compatible_ports(
    part_a: Part, part_b: Part, port_a: str | None = None, port_b: str | None = None
):
    """All (name_a, name_b) mech port pairs that can mate (optionally restricted to named ports), in declaration order."""
    out = []
    for pa in part_a.ports:
        if pa.get("kind") != "mech" or (port_a and pa.get("name") != port_a):
            continue
        for pb in part_b.ports:
            if pb.get("kind") != "mech" or (port_b and pb.get("name") != port_b):
                continue
            if incompatible(pa, pb) is None:
                out.append((pa["name"], pb["name"]))
    return out


# ---- kinematic tree ---------------------------------------------------------------------------------------------
@dataclass
class MateSpec:
    name: str
    a: str  # instance ids; a = parent, b = child
    b: str
    pa: dict  # port dicts
    pb: dict
    rotation: int = 0
    closure: bool = False
    has_joint: bool = False

    def flipped(self) -> "MateSpec":
        # M(b, a, -k) = M(a, b, k)^-1
        return MateSpec(
            self.name,
            self.b,
            self.a,
            self.pb,
            self.pa,
            (-self.rotation) % n_mate(self.pa, self.pb),
            self.closure,
            self.has_joint,
        )

    def matrix(self) -> np.ndarray:
        """T_a_b: pose of b in a's frame."""
        return mate_transform(np.eye(4), self.pa, self.pb, self.rotation)


def rpy_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis rpy (radians): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(x) for x in rpy)
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def matrix_rpy(R: np.ndarray) -> list:
    """Inverse of rpy_matrix (pitch in [-pi/2, pi/2]; at gimbal lock roll = 0)."""
    sp = -R[2, 0]
    p = math.asin(max(-1.0, min(1.0, sp)))
    if abs(sp) > 1 - 1e-9:
        return [0.0, p, math.atan2(-R[0, 1], R[1, 1])]
    return [math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])]


def pose_matrix(pos, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rpy_matrix(rpy), pos
    return T


@dataclass
class PoseSpec(MateSpec):
    """A free-placed instance (`pos` / `rpy` on the instance) as a tree edge from the root: not a mate, no ports."""

    T: np.ndarray = None

    def matrix(self) -> np.ndarray:
        return self.T


@dataclass
class Tree:
    T: dict  # instance id -> 4x4 world pose (root = identity)
    parent: dict  # child id -> (parent id, MateSpec as used: .a = parent, .b = child)
    order: list  # ids, parents before children
    closures: list  # (MateSpec, position error m, rotation error rad)


def rot_angle(R: np.ndarray) -> float:
    """Rotation angle of R (stable for small angles)."""
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return math.atan2(np.linalg.norm(v), (np.trace(R) - 1) / 2)


def _chain(parent: dict, x: str) -> list:
    out = [x]
    while out[-1] in parent:
        out.append(parent[out[-1]][0])
    return out


def solve_tree(
    root: str,
    ids: list,
    mates: list,
    tol_pos: float = 1e-6,
    tol_rot: float = 1e-6,
    placed: dict | None = None,
) -> Tree:
    """Place every instance from `root`. A mate between two already placed instances is a cycle (error naming the loop)
    unless it is flagged `closure`, in which case it is only checked against the tree. `placed` = {id: 4x4 pose in the root
    frame} of free-placed instances: each is a tree child of the root, mates may continue from it."""
    known = set(ids)
    for m in mates:
        for end in (m.a, m.b):
            if end not in known:
                raise DesignError(
                    f"mate '{m.name}': unknown or offboard instance '{end}'"
                )
    T = {root: np.eye(4)}
    parent: dict = {}
    order = [root]
    for pid, PT in (placed or {}).items():
        if pid not in known or pid == root:
            continue
        T[pid] = PT
        parent[pid] = (root, PoseSpec(f"{pid}:pose", root, pid, {}, {}, T=PT))
        order.append(pid)
    pending = [m for m in mates if not m.closure]
    while pending:
        rest, progress = [], False
        for m in pending:
            if m.a in T and m.b in T:
                ca, cb = _chain(parent, m.a), _chain(parent, m.b)
                common = next(x for x in ca if x in cb)
                loop = ca[: ca.index(common) + 1] + cb[: cb.index(common)][::-1]
                raise DesignError(
                    f"cycle: {' -> '.join(loop + [loop[0]])} (mate '{m.name}'); "
                    "set closure = true on a mate that only closes a loop"
                )
            if m.a in T:
                use = m
            elif m.b in T:
                if m.has_joint:
                    raise DesignError(
                        f"mate '{m.name}': has a joint but a ('{m.a}') is the child; swap a and b"
                    )
                use = m.flipped()
            else:
                rest.append(m)
                continue
            T[use.b] = T[use.a] @ use.matrix()
            parent[use.b] = (use.a, use)
            order.append(use.b)
            progress = True
        if not progress:
            break
        pending = rest
    missing = [i for i in ids if i not in T]
    if missing:
        raise DesignError(
            f"instances not connected to root '{root}' by mates: {missing}"
        )
    closures = []
    for m in mates:
        if not m.closure:
            continue
        err = inv_T(T[m.b]) @ (T[m.a] @ m.matrix())
        ep, er = float(np.linalg.norm(err[:3, 3])), rot_angle(err[:3, :3])
        if ep > tol_pos or er > tol_rot:
            raise DesignError(
                f"closure mate '{m.name}' is inconsistent with the tree: off by {ep:.3g} m and {er:.3g} rad "
                f"(limits {tol_pos:g} m, {tol_rot:g} rad)"
            )
        closures.append((m, ep, er))
    return Tree(T, parent, order, closures)


# ---- fasteners --------------------------------------------------------------------------------------------------
_THREAD = re.compile(
    r"(?:^|\.)(m[0-9]+(?:\.[0-9]+)?)$"
)  # last dotted segment; joins decimals: thread.m2.5
_NUM = r"[0-9]+(?:\.[0-9]+)?"


def thread_of(port: dict) -> str | None:
    """'m3' from the type's trailing m<d> segment (bolt-circle.25.4.m3 -> m3), else the port's `thread` key."""
    m = _THREAD.search(str(port.get("type", "")))
    if m:
        return m.group(1)
    t = port.get("thread")
    return (
        t.lower()
        if isinstance(t, str) and re.fullmatch(r"m[0-9.]+", t.lower())
        else None
    )


def hole_positions(port: dict) -> list:
    """Hole centres [x, y] (m) in the port frame, before any `holes` override."""
    hp = port.get("hole_pos")
    if hp:
        return [[float(x), float(y)] for x, y in hp]
    t = str(port.get("type", ""))
    m = re.search(rf"bolt-pattern\.({_NUM})x({_NUM})", t)
    if m:
        a, b = float(m.group(1)) / 2000, float(m.group(2)) / 2000
        return [[a, b], [-a, b], [-a, -b], [a, -b]]
    m = re.search(rf"bolt-circle\.({_NUM})\.([0-9]+)", t)
    if m:
        r, n = float(m.group(1)) / 2000, int(m.group(2))
        return [
            [r * math.cos(2 * math.pi * j / n), r * math.sin(2 * math.pi * j / n)]
            for j in range(n)
        ]
    return [[0.0, 0.0]]


def hole_count(port: dict) -> int:
    pos = hole_positions(port)
    n = port.get("holes", len(pos))
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(pos):
        raise DesignError(
            f"port '{port.get('name')}': holes={n!r} but the pattern has {len(pos)} positions"
        )
    return n


def nut_height(nut: Part) -> float | None:
    for k in ("height", "height_max"):
        if nut.value(k) is not None:
            return nut.value(k)
    for p in nut.ports:
        if isinstance(p.get("length"), (int, float)):
            return float(p["length"])
    return None


def _fastener_parts(lib: Library, thread: str):
    """(screws sorted by (length, id), nuts sorted by (height, id)) of this thread."""
    typ = f"thread.{thread}"
    screws = [
        p
        for p in lib.by_category("fastener")
        if p.has_port(typ, "male") and p.value("length") is not None
    ]
    nuts = [
        p
        for p in lib.by_category("fastener")
        if p.has_port(typ, "female") and nut_height(p) is not None
    ]
    return (
        sorted(screws, key=lambda p: (p.value("length"), p.id)),
        sorted(nuts, key=lambda p: (nut_height(p), p.id)),
    )


def select_fasteners(
    lib: Library, mate: str, pa: dict, pb: dict, spec, nut_id: str | None = None
):
    """Choose the fastener for one mate. spec = "auto" or {"part": id, "count": n, "nut": id}.
    Returns (record, screw Part, nut Part|None, hole positions used, warnings [(code, message)])."""
    ta, tb = thread_of(pa), thread_of(pb)
    if ta and tb and ta != tb:
        raise DesignError(f"mate '{mate}': thread mismatch {ta} vs {tb}")
    thread = ta or tb
    if not thread:
        raise DesignError(
            f"mate '{mate}': cannot determine the thread (no m<d> type suffix or `thread` key)"
        )
    d = float(thread[1:]) / 1000.0
    screws, nuts = _fastener_parts(lib, thread)
    tapped = [p for p in (pa, pb) if p.get("tapped")]
    if len(tapped) == 2:
        raise DesignError(f"mate '{mate}': both ports are tapped")
    stack = float(sum(p.get("thickness", 0.0) for p in (pa, pb) if not p.get("tapped")))
    nut = None
    if tapped:
        eng = ENGAGEMENT_FACTOR * d
        depth = tapped[0].get("thread_depth")
        if depth is not None:
            eng = min(eng, float(depth))
    else:
        if nut_id:
            nut = lib.get(nut_id)
            if not nut.has_port(f"thread.{thread}", "female"):
                raise DesignError(
                    f"mate '{mate}': nut '{nut_id}' has no female thread.{thread} port"
                )
        elif nuts:
            nut = nuts[0]
        else:
            raise DesignError(
                f"mate '{mate}': through-bolt needs a nut but the library has none for {thread}"
            )
        h = nut_height(nut)
        if h is None:
            raise DesignError(f"mate '{mate}': nut '{nut.id}' has no height")
        explicit = (
            lib.get(spec["part"]) if isinstance(spec, dict) and "part" in spec else None
        )
        pitch = None
        for src in ([explicit] if explicit else []) + screws + [nut]:
            if src is not None and src.value("thread_pitch") is not None:
                pitch = src.value("thread_pitch")
                break
        if pitch is None:
            raise DesignError(
                f"mate '{mate}': no thread_pitch known for {thread}; cannot size a through-bolt"
            )
        eng = h + PROTRUSION_PITCHES * pitch
    l_min = stack + eng
    warnings = []
    positions = hole_positions(pa)
    holes = hole_count(pa)
    if isinstance(spec, dict):
        screw = lib.get(spec["part"])
        if (
            not screw.has_port(f"thread.{thread}", "male")
            or screw.value("length") is None
        ):
            raise DesignError(
                f"mate '{mate}': part '{screw.id}' is not a {thread} screw with a length"
            )
        if screw.value("length") < l_min - EPS_LEN:
            warnings.append(
                (
                    "FASTENER_SHORT",
                    f"mate '{mate}': {screw.id} length {screw.value('length')} m < L_min {l_min} m",
                )
            )
        count = spec.get("count", holes)
    else:
        ok = [s for s in screws if s.value("length") >= l_min - EPS_LEN]
        if not ok:
            longest = screws[-1].value("length") if screws else None
            raise DesignError(
                f"mate '{mate}': no {thread} screw fits: L_min = {l_min:.6g} m, "
                f"longest available = {longest if longest is not None else 'none in library'}"
            )
        screw, count = ok[0], holes
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= len(positions)
    ):
        raise DesignError(
            f"mate '{mate}': fastener count {count!r} not in 1..{len(positions)}"
        )
    if screw.value("mass") is None or (nut is not None and nut.value("mass") is None):
        raise DesignError(
            f"mate '{mate}': no mass for part '{screw.id if screw.value('mass') is None else nut.id}'"
        )
    mass = count * (screw.value("mass") + (nut.value("mass") if nut else 0.0))
    rec = {
        "mate": mate,
        "thread": thread,
        "holes": holes,
        "stack": stack,
        "engagement": eng,
        "L_min": l_min,
        "part": screw.id,
        "length": screw.value("length"),
        "nut": nut.id if nut else None,
        "count": count,
        "mass": mass,
    }
    return rec, screw, nut, positions[:count], warnings
