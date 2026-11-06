"""Scene of the hopper: assemblies/hopper.toml standing on its four feet on the floor. `dvsim program` calls build(outdir) -> MJCF path; hop.py and mc.py call build_seeded() for a
per-run noise seed and Dryden intensity (the device seeds and `w20` are build-time constants)."""

import json
import re
from pathlib import Path

from dvsim import design
from dvsim.design.assembly import load

REPO = Path(__file__).resolve().parents[2]
ASSEMBLY = REPO / "assemblies" / "hopper.toml"
_LIB = None


M8N = dict(
    horiz_std=2.1233, vert_std=3.185, vel_std=0.05, corr_time_s=600.0, corr_frac=0.9
)  # the lander's NEO-M8N figures (assemblies/tvc-lander.toml), for the sensitivity run


def scatter_xml(xml, sc):
    """Apply the build-time part of one flight's physical scatter to the MJCF text. `sc` keys (all optional): ct_scale (thrust AND torque coefficients of the fan), mass_scale (every geom),
    batt_dx / batt_dy (m, battery position: moves the centre of mass 0.25 x as far), imu_noise / baro_noise / lidar_noise (x noise densities), gnss (dict of GNSS device fields)."""
    m = re.search(r"(<text name=\"dvsim.devices\" data=')(.*?)('\s*/>)", xml, re.DOTALL)
    d = json.loads(m.group(2))
    for r in d["rotors"]:
        r["ct"] = [c * sc.get("ct_scale", 1.0) for c in r["ct"]]
        r["cq"] = [c * sc.get("ct_scale", 1.0) for c in r["cq"]]
    for i in d["imus"]:
        for ax in ("gyro", "accel"):
            i[ax]["noise_density"] *= sc.get("imu_noise", 1.0)
    for b in d["baros"]:
        b["noise_std"] *= sc.get("baro_noise", 1.0)
    for r in d["rangefinders"]:
        r["sigma0"] *= sc.get("lidar_noise", 1.0)
        r["sigma1"] *= sc.get("lidar_noise", 1.0)
    if "soc0" in sc:
        d["batteries"][0]["soc0"] = sc["soc0"]
    for g in d["gnss_receivers"]:
        g.update(sc.get("gnss", {}))
    xml = (
        xml[: m.start(2)]
        + json.dumps(d, sort_keys=True, separators=(",", ":"))
        + xml[m.end(2) :]
    )
    k = sc.get("mass_scale", 1.0)
    if k != 1.0:
        xml = re.sub(
            r'(<geom [^>]*?\bmass=")([-0-9.eE+]+)(")',
            lambda g: f"{g.group(1)}{float(g.group(2)) * k:.9g}{g.group(3)}",
            xml,
        )
    dx, dy = sc.get("batt_dx", 0.0), sc.get("batt_dy", 0.0)
    if dx or dy:

        def shift(g):
            x, y, z = (float(v) for v in g.group(2).split())
            return f'{g.group(1)}{x + dx:.9g} {y + dy:.9g} {z:.9g}"'

        xml = re.sub(r'(<body name="batt" pos=")([^"]*)"', shift, xml, count=1)
    return xml


def build_seeded(out, seed=1, w20=None, scatter=None):
    """MJCF of the hopper with every noisy device seeded `seed`, the Dryden intensity `w20` (m/s; None = the assembly's value) and the build-time physical scatter `scatter` (scatter_xml).
    -> path of scene.xml"""
    global _LIB
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    asm = load(ASSEMBLY)
    asm.doc["assembly"]["seed"] = int(seed)
    if w20 is not None:
        env = next(d for d in asm.doc["devices"] if d["type"] == "environment")
        env["set"]["w20"]["value"] = float(w20)
    if _LIB is None:
        _LIB = design.Library(REPO / "parts")
    xml, _ = design.compute(asm, REPO / "parts", out, lib=_LIB, visual=False)
    if scatter:
        xml = scatter_xml(xml, scatter)
    (out / "scene.xml").write_text(xml)
    return out / "scene.xml"


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xml, _ = design.compute(ASSEMBLY, REPO / "parts", out)
    (out / "scene.xml").write_text(xml)
    return out / "scene.xml"
