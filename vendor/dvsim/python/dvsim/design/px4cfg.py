"""PX4 bridge config from a design (`dvsim program` scenario `[design] path = ...`), with the motor-order / prop-direction check.

    findings = check_px4(asm, xml)              # [Finding]: errors that would crash a first flight, each with its fix
    cfg_path = bridge_config(asm, xml, out)     # writes the dvsim-px4-bridge TOML; raises DesignError listing the errors

What is derived (nothing is typed by hand):
  * motor index <-> ESC device <-> PX4 actuator output: the wire fc.pwmN -> escK.signal means PX4 output N drives ESC K. The
    airframe (px4/airframes/4902_dvsim_x500v2, = PX4's 4001_quad_x geometry) sets PWM_MAIN_FUNCN = 100 + N (Motor N), so PX4 `controls[N-1]`
    is what the bridge writes to that ESC's `throttle` (`[motors] esc = [...]` in PX4 motor order).
  * rotor geometry: the rotor body position in the root frame (the bridge's FLU base: +x forward, +y left) relative to the
    centroid of the four rotors, and the spin sense from the joint axis (the ESC's positive drive spins the prop about +axis; a
    CW prop has the reversed axis and rotor dir = -1, assemblies/x500v2.toml header) -> corner + CW / CCW.
  * sensors / battery: the first imu / baro / mag / gnss / battery device; home = the gnss lat0 / lon0 / alt0, gps_rate_hz = its rate.

PX4 reference layout (Quadrotor x, "motors in green rotate clockwise, blue counterclockwise"; numbering as in PX4's airframe
reference https://docs.px4.io/main/en/airframes/airframe_reference.html#quadrotor-x and ROMFS/px4fmu_common/init.d/airframes/4001_quad_x,
CA_ROTORn_PX / PY in FRD, CA_ROTORn_KM > 0 = CCW, < 0 = CW per control_allocator/module.yaml):
    motor 1 front-right CCW   motor 2 rear-left CCW   motor 3 front-left CW   motor 4 rear-right CW
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .assembly import Assembly, load
from .drc import Finding
from .library import DesignError, Library

# PX4 motor index (0-based = controls[i]) -> (forward sign, right sign, CCW) of 4001_quad_x / 4902 CA_ROTORi_PX, PY, KM
QUAD_X = [(1, 1, True), (-1, -1, True), (1, -1, False), (-1, 1, False)]
AIRFRAME = "4001_quad_x / 4902_dvsim_x500v2"
SENSORS = (
    ("imu", "imus"),
    ("baro", "baros"),
    ("mag", "mags"),
    ("gnss", "gnss_receivers"),
    ("battery", "batteries"),
)


def _corner(fwd, right):
    return ("front" if fwd > 0 else "rear") + "-" + ("right" if right > 0 else "left")


def _quat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _model(xml):
    """-> (device JSON, root body name, {joint: (world axis, world pos of its body)}, {body: world pos}) of the built MJCF."""
    root = ET.parse(xml).getroot()
    dev = json.loads(
        next(t for t in root.iter("text") if t.get("name") == "dvsim.devices").get(
            "data"
        )
    )
    joints, bodies, first = {}, {}, []

    def walk(el, p0, R0):
        for b in el.findall("body"):
            p = p0 + R0 @ np.array(b.get("pos", "0 0 0").split(), float)
            R = R0 @ _quat(np.array(b.get("quat", "1 0 0 0").split(), float))
            bodies[b.get("name")] = p
            first.append(b.get("name"))
            for j in b.findall("joint"):
                joints[j.get("name")] = R @ np.array(
                    j.get("axis", "0 0 1").split(), float
                )
            walk(b, p, R)

    walk(root.find("worldbody"), np.zeros(3), np.eye(3))
    return dev, first[0], joints, bodies


def _channels(doc):
    """{ESC instance: [PX4 output N]} from the wires fc.pwmN <-> escK.signal."""
    ch = {}
    for w in doc.get("wires", []):
        for a, b in ((w["from"], w["to"]), (w["to"], w["from"])):
            m = re.fullmatch(r"[^.]+\.pwm(\d+)", a)
            if m and b.endswith(".signal"):
                ch.setdefault(b.split(".")[0], []).append(int(m[1]))
    return ch


def _motors(doc, dev, joints, bodies):
    """One record per ESC device: instance, PX4 outputs wired to it, rotor name / position / spin / thrust direction."""
    inst = {
        d["name"]: d["instance"]
        for d in doc.get("devices", [])
        if d.get("type") == "esc" and "instance" in d
    }
    ch = _channels(doc)
    rotors = {r["joint"]: r for r in dev.get("rotors", [])}
    out = []
    for e in dev.get("escs", []):
        r = rotors.get(e["joint"])
        m = dict(
            esc=e["name"],
            instance=inst.get(e["name"]),
            channels=ch.get(inst.get(e["name"]), []),
            rotor=None,
        )
        if r:
            ax = joints[e["joint"]]
            m.update(
                rotor=r["name"],
                pos=bodies[r["body"]],
                ccw=bool(ax[2] > 0),
                thrust_z=float(r.get("dir", 1) * ax[2]),
            )
        out.append(m)
    return out


def check_px4(asm, xml) -> list:
    """Motor-order / prop-direction / wiring findings (all errors) of the design against the PX4 quad-X airframe. `asm`: Assembly
    (or its path), `xml`: the built MJCF. Empty list = the bridge can be generated."""
    doc = (asm if isinstance(asm, Assembly) else load(asm)).doc
    dev, _, joints, bodies = _model(xml)
    ms = _motors(doc, dev, joints, bodies)
    F = []

    def err(code, msg, where, fix):
        F.append(Finding(code, "error", msg, list(where), fix))

    if len(ms) != len(QUAD_X):
        err("PX4_MOTOR_COUNT", f"PX4 airframe {AIRFRAME} is a quad (4 motors), the design has {len(ms)} ESC devices",
            [m["esc"] for m in ms], "bind exactly four esc devices (one per motor), or use a PX4 airframe with that rotor count")  # fmt: skip
        return F
    for m in ms:
        w = [f"fc.pwm{n}" for n in m["channels"]]
        if not m["channels"]:
            err("PX4_PWM_UNWIRED", f"ESC {m['esc']} has no signal wire from an FC PWM output: PX4 would never drive it", [m["esc"]],
                f"wire fc.pwmN -> {m['instance']}.signal (N = the PX4 motor number, 1..4)")  # fmt: skip
        elif len(m["channels"]) > 1:
            err("PX4_PWM_DUPLICATE", f"ESC {m['esc']} is wired to {len(w)} PWM outputs ({', '.join(w)})", [m["esc"], *w],
                "keep one signal wire per ESC")  # fmt: skip
        elif not 1 <= m["channels"][0] <= len(QUAD_X):
            err("PX4_PWM_RANGE", f"ESC {m['esc']} is on {w[0]}, but the airframe's motors are outputs 1..4 (PWM_MAIN_FUNC1..4 = Motor1..4)",
                [m["esc"], w[0]], "move the signal lead to fc.pwm1..fc.pwm4")  # fmt: skip
        if m["rotor"] is None:
            err("PX4_NO_ROTOR", f"ESC {m['esc']} drives no rotor device (no rotor on its joint)", [m["esc"]], "bind a rotor device on the prop's joint")  # fmt: skip
        elif m["thrust_z"] <= 0:
            err("PX4_PROP_THRUST", f"rotor {m['rotor']} (ESC {m['esc']}) pushes DOWN: thrust direction (dir x joint axis) z = {m['thrust_z']:+.0f}", [m["rotor"]],
                "a CW prop needs BOTH the reversed joint axis (joint reverse = true) and rotor dir = -1; a CCW prop neither")  # fmt: skip
    if F:
        return F
    chans = [m["channels"][0] for m in ms]
    if sorted(chans) != [1, 2, 3, 4]:
        err("PX4_PWM_DUPLICATE", f"PX4 outputs {sorted(chans)} drive the four ESCs, expected 1, 2, 3, 4 once each", [m["esc"] for m in ms], "wire fc.pwm1..fc.pwm4 to the four ESCs, one each")  # fmt: skip
        return F
    c = np.mean([m["pos"] for m in ms], axis=0)
    for m in ms:  # corner of the rotor relative to the rotor centroid (root frame: +x forward, +y left)
        d = m["pos"] - c
        if min(abs(d[0]), abs(d[1])) < 0.2 * max(abs(d[0]), abs(d[1])):
            err("PX4_GEOMETRY", f"rotor {m['rotor']} is not at a corner of an X (offset from the rotor centroid x {d[0]:+.3f}, y {d[1]:+.3f} m)", [m["rotor"]], f"{AIRFRAME} needs a square quad-X, rotors on the diagonals")  # fmt: skip
        m["corner"] = (1 if d[0] > 0 else -1, -1 if d[1] > 0 else 1)  # (forward, right)
    if F:
        return F
    at = {m["corner"]: m for m in ms}
    if len(at) != 4:
        err("PX4_GEOMETRY", "two rotors share a corner of the quad-X", [m["rotor"] for m in ms], f"{AIRFRAME} needs one rotor per corner")  # fmt: skip
        return F
    fixes = {
        n: at[QUAD_X[n - 1][:2]] for n in range(1, 5)
    }  # PX4 motor n -> the ESC whose rotor sits where it belongs
    for m in sorted(ms, key=lambda m: m["channels"][0]):
        n = m["channels"][0]
        want = QUAD_X[n - 1]
        if m["corner"] != want[:2]:
            err("PX4_MOTOR_ORDER", f"PX4 motor {n} (fc.pwm{n} -> {m['esc']}) drives the rotor at {_corner(*m['corner'])}, but in {AIRFRAME} motor {n} is {_corner(*want[:2])} "
                f"({'CCW' if want[2] else 'CW'}): PX4 would apply its roll / pitch / yaw commands to the wrong rotors and flip at takeoff", [f"fc.pwm{n}", m["esc"]],
                f"wire fc.pwm{n} -> {fixes[n]['instance']}.signal (the {_corner(*want[:2])} ESC); full order: "
                + ", ".join(f"fc.pwm{k} -> {fixes[k]['instance']}" for k in range(1, 5)))  # fmt: skip
        elif m["ccw"] != want[2]:
            err("PX4_PROP_DIRECTION", f"rotor {m['rotor']} (PX4 motor {n}, {_corner(*want[:2])}) spins {'CCW' if m['ccw'] else 'CW'} but {AIRFRAME} motor {n} is {'CCW' if want[2] else 'CW'} "
                "(CA_ROTORn_KM sign): the yaw torque would have the wrong sign and the quad spins up at takeoff", [m["rotor"], m["esc"]],
                "CW prop: joint reverse = true on the prop mate and rotor dir = -1; CCW prop: neither (assemblies/x500v2.toml header)")  # fmt: skip
    return F


def bridge_config(asm, xml, out, parts=None) -> Path:
    """Write the dvsim-px4-bridge TOML for the design to `out` (-> Path). Raises DesignError with every finding of check_px4 (and the fix
    of each) before anything is written. `parts`: the parts dir, for the battery's cell count (else 4)."""
    a = asm if isinstance(asm, Assembly) else load(asm)
    if bad := check_px4(a, xml):
        raise DesignError(
            "PX4 bridge config: the design does not match the PX4 airframe:\n"
            + "\n".join(f"  [{f.code}] {f.message}\n    fix: {f.fix}" for f in bad)
        )
    dev, root, joints, bodies = _model(xml)
    ms = sorted(_motors(a.doc, dev, joints, bodies), key=lambda m: m["channels"][0])
    names = {}
    for key, k in SENSORS:
        if not dev.get(k):
            raise DesignError(
                f"PX4 bridge config: the design has no {key} device (the bridge needs imu, baro, mag, gnss and battery)"
            )
        names[key] = dev[k][0]["name"]
    g = dev["gnss_receivers"][0]
    cells = 4
    if parts is not None:
        bat = next(
            (d for d in a.doc.get("devices", []) if d.get("type") == "battery"), None
        )
        part = next(
            (
                i["part"]
                for i in a.doc["instances"]
                if bat and i["id"] == bat.get("instance")
            ),
            None,
        )
        if part:
            cells = int(
                Library(parts).get(part).params.get("cells_series", {}).get("value", 4)
            )
    L = [
        f"# dvsim-px4-bridge config generated from design '{a.name}' by dvsim.design.px4cfg (motor order checked against {AIRFRAME})"
    ]
    L += [f'model = "{Path(xml).resolve()}"', f'base = "/world/{root}"']
    L += [f'{k} = "{v}"' for k, v in names.items()]
    L += [f"gps_rate_hz = {g['rate_hz']:g}", "satellites = 10", "", "[motors]"]
    L += [
        "# entry i = PX4 controls[i] = fc.pwm{i+1}: "
        + ", ".join(
            f"{m['esc']} (fc.pwm{m['channels'][0]}, rotor {m['rotor']})" for m in ms
        )
    ]
    L += [
        "esc = [" + ", ".join(f'"{m["esc"]}"' for m in ms) + "]",
        "",
        "[home]  # the gnss device's lat0 / lon0 / alt0",
    ]
    L += [f"lat = {g['lat0']!r}", f"lon = {g['lon0']!r}", f"alt = {g['alt0']!r}", ""]
    L += [
        "[px4]",
        "sysid = 1",
        "compid = 180",
        "battery_rate_hz = 5",
        f"cells = {cells}",
        "",
    ]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out
