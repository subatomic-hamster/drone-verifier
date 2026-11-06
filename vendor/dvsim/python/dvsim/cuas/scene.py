"""Multi-vehicle scene composition for the counter-UAS demo: several built assemblies (the turret and any number of drones) in ONE MJCF /
ONE World, each under its own namespace.

    scene = compose(outdir, [Vehicle("drone1", "x500v2", pos=(30, 5, 0.0), yaw=180), Vehicle("turret", "turret", pos=(0, 0, 1.0))])
    scene["model"]      # path of the MJCF
    scene["manifest"]   # the same dict as <outdir>/scene.json: prim paths, device names, body-id -> (vehicle, body) table

Namespacing. Every vehicle sits in a container body named like the vehicle ("drone1", placed at `pos`, yawed `yaw` degrees about z); every
element name inside is prefixed "drone1/" (MuJoCo names are global), so a body prim is /world/drone1/<leaf-path>, e.g. /world/drone1/bottom.
Device names cannot contain '/', so they get "drone1_": /devices/drone1_esc1. Device references to joints / bodies / sites are rewritten to
the prefixed MJCF names, electrical node numbers are offset per vehicle (each vehicle keeps its own network; the kernel solves the block-diagonal
system), and the per-vehicle `environment` device is replaced by ONE scene environment device `air` (wind, Dryden turbulence).
Extras: a fixed floor (the first vehicle's), a pedestal under the turret, a pool of 16 mocap tracer bodies `pellet0..15` (visual only, geom group 5)
that the turret's launcher device moves onto the flying pellets.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import design
from ..terrain import GEOM_NAME, add_terrain

REPO = Path(__file__).resolve().parents[3]
N_PELLETS = 16  # kMaxPellets of the kernel
TRACER_RADIUS = 0.015  # m: VISUAL sphere of a flying pellet (the physical pellet is 4.5 mm; enlarged so the viewer shows it, still < 1 px beyond 15 m)
NAME_TAGS = ("body", "geom", "joint", "freejoint", "site", "mesh")
DEV_REFS = (
    "joint",
    "backlash_joint",
    "body",
    "ref_body",
    "stator_body",
    "duct",
    "site",
)


@dataclass
class Vehicle:
    name: str  # namespace and container body name ([A-Za-z0-9_]+)
    assembly: str  # "x500v2" (assemblies/<name>.toml) or a path
    pos: tuple = (
        0.0,
        0.0,
        0.0,
    )  # container position (the vehicle's own frame origin is built on the ground / at its base)
    yaw: float = 0.0  # deg about +z
    pedestal: float = 0.0  # m: a static box from the ground up to `pos` z (the turret's mast), 0 = none


@dataclass
class Environment:
    wind: tuple = (0.0, 0.0, 0.0)  # steady wind, world (m/s)
    w20: float = 0.0  # Dryden turbulence level, wind speed at 6.1 m (m/s); 0 = none
    origin_alt: float = 0.0
    seed: int = 1
    # Rain (kernel `rain` device, docs/models.md "Rain"): present iff rain or rain_rate > 0, so every existing scene keeps its state layout
    # and replay hash. rain_rate is the initial rate in mm/h (the harness changes it at run time: sim.set("/devices/rain", "rate", R)).
    # rain_vehicle: the vehicle whose root body catches rain (drag by the drops' momentum + a water film that adds weight);
    # rain_catch_area / rain_wet_area (m^2, ESTIMATES by the caller): area facing the falling rain / exposed surface that holds a film.
    rain: bool = False
    rain_rate: float = 0.0
    rain_vehicle: str = ""
    rain_catch_area: float = 0.0
    rain_wet_area: float = 0.0


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def _asm_path(a):
    p = Path(a)
    return p if p.suffix else REPO / "assemblies" / f"{a}.toml"


def _quat_z(deg):
    h = np.radians(deg) / 2
    return [np.cos(h), 0.0, 0.0, np.sin(h)]


def _bodies_dfs(root):
    """Body names in MuJoCo id order (document order, depth first), world = id 0."""
    out = ["world"]

    def walk(el):
        for b in el.findall("body"):
            out.append(b.get("name"))
            walk(b)

    walk(root.find("worldbody"))
    return out


def _paths(root):
    """{full body name: prim path}: a body prim shows only the leaf after the last '/', under its parent's prim path."""
    out = {}

    def walk(el, prefix):
        for b in el.findall("body"):
            p = f"{prefix}/{b.get('name').rsplit('/', 1)[-1]}"
            out[b.get("name")] = p
            walk(b, p)

    walk(root.find("worldbody"), "/world")
    return out


def _prefix_vehicle(root, p):
    """Prefix every name in the vehicle's MJCF tree with 'p/' (in place) and return its worldbody children."""
    for el in root.iter():
        if el.tag in NAME_TAGS and el.get("name"):
            el.set("name", f"{p}/{el.get('name')}")
        if el.tag == "geom" and el.get("mesh"):
            el.set("mesh", f"{p}/{el.get('mesh')}")


def compose(
    outdir,
    vehicles,
    env=None,
    parts=REPO / "parts",
    seed=1,
    floor_size=(0, 0, 1),
    visual=True,
    terrain=None,
    wind_field=None,
) -> dict:
    """`terrain`: a dvsim.terrain.Terrain (replaces the floor with its hfield, pads, boulders); `wind_field`: a dvsim.terrain.WindField (terrain-aware wind device
    `wind`, needs the environment; its hfield is the terrain's). The manifest gets `pads` {name: {x, y, z, radius}} and `terrain` {kind, size, seed}."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    env = env or Environment()
    root = None
    devs: dict = {}
    node_base = 0
    man = {"vehicles": {}, "seed": seed}
    taken = set()
    roots = {}  # vehicle name -> its first top-level body (the one a rain film sits on)
    for v in vehicles:
        if v.name in taken or "/" in v.name or "_" in v.name[:1]:
            raise ValueError(
                f"vehicle name {v.name!r}: unique, no '/', not starting with '_'"
            )
        taken.add(v.name)
        text, report = design.compute(_asm_path(v.assembly), parts, out, visual=visual)
        tree = ET.fromstring(text)
        d = json.loads(tree.find("custom/text").get("data"))
        _prefix_vehicle(tree, v.name)
        wb = tree.find("worldbody")
        for g in [g for g in wb.findall("geom") if g.get("name") == f"{v.name}/floor"]:
            wb.remove(g)
        # MuJoCo allows a free joint only on a top-level body, so no container: each top-level body is placed in the world directly
        # (offset + yaw composed into its pose) and named "<vehicle>_<leaf>" so that prim paths /world/<name> stay unique.
        renames = {}
        for b in wb.findall("body"):
            old = b.get("name")
            renames[old] = b.attrib["name"] = f"{v.name}_{old.split('/', 1)[1]}"
            if b.get("euler") or b.get("axisangle"):
                raise ValueError(f"{old}: top-level body pose must use pos/quat")
            p0 = np.array([float(x) for x in b.get("pos", "0 0 0").split()])
            q0 = np.array([float(x) for x in b.get("quat", "1 0 0 0").split()])
            c, s = np.cos(np.radians(v.yaw)), np.sin(np.radians(v.yaw))
            b.set(
                "pos",
                _fmt(
                    np.array([c * p0[0] - s * p0[1], s * p0[0] + c * p0[1], p0[2]])
                    + np.array(v.pos)
                ),
            )
            qz = _quat_z(v.yaw)
            b.set("quat", _fmt([qz[0] * q0[0] - qz[3] * q0[3], qz[0] * q0[1] - qz[3] * q0[2],
                                qz[0] * q0[2] + qz[3] * q0[1], qz[0] * q0[3] + qz[3] * q0[0]]))  # fmt: skip
        tops = list(wb.findall("body"))
        roots[v.name] = tops[0].get("name") if tops else ""
        if root is None:
            root = ET.Element("mujoco", model="cuas_scene")
            ET.SubElement(root, "compiler", angle="radian")
            ET.SubElement(root, "option", timestep=tree.find("option").get("timestep"))
            ET.SubElement(root, "asset")
            rw = ET.SubElement(root, "worldbody")
            ET.SubElement(rw, "geom", name="floor", type="plane", size=_fmt(floor_size), contype="1", conaffinity="1",
                          group="0", rgba="0.36 0.42 0.30 1")  # fmt: skip
            ET.SubElement(root, "custom")
        elif tree.find("option").get("timestep") != root.find("option").get("timestep"):
            raise ValueError("vehicles must share one timestep")
        asset = tree.find("asset")
        if (
            asset is not None
        ):  # visual materials ("m_<name>") are shared across vehicles: keep one, a different one is a clash
            have = {(e.tag, e.get("name")): e.attrib for e in root.find("asset")}
            for e in asset:
                k = (e.tag, e.get("name"))
                if k in have and have[k] != e.attrib:
                    raise ValueError(f"MJCF asset clash between vehicles: {k}")
                if k not in have:
                    root.find("asset").append(e)
        root.find("worldbody").extend(tops)
        if v.pedestal > 0:
            ET.SubElement(root.find("worldbody"), "geom", name=f"{v.name}/pedestal", type="box", group="0", contype="1", conaffinity="1",
                          size=_fmt([0.12, 0.12, v.pedestal / 2]), pos=_fmt([v.pos[0], v.pos[1], v.pedestal / 2]), rgba="0.55 0.55 0.58 1")  # fmt: skip
        # devices: rename, re-reference, offset the nodes
        top = 0
        for key, lst in d.items():
            if not isinstance(lst, list):
                continue
            if key == "environments":
                continue
            for dev in lst:
                dev["name"] = f"{v.name}_{dev['name']}"
                for r in DEV_REFS:
                    if dev.get(r):
                        dev[r] = renames.get(f"{v.name}/{dev[r]}", f"{v.name}/{dev[r]}")
                if isinstance(
                    dev.get("rotor"), str
                ):  # a jet vane names the rotor DEVICE it sits in the jet of
                    dev["rotor"] = f"{v.name}_{dev['rotor']}"
                for f in ("node", "a", "b"):
                    if dev.get(f, 0) > 0:
                        top = max(top, dev[f])
                        dev[f] += node_base
                devs.setdefault(key, []).append(dev)
        node_base += top
        man["vehicles"][v.name] = dict(assembly=str(v.assembly), pos=list(v.pos), yaw=v.yaw,
                                       mass=float(report["mass"]["total"]),
                                       instances=[dict(body=i.get("body"), mass=i["mass"]) for i in report["instances"] if i.get("body") and i["mass"] > 0])  # fmt: skip
    if node_base > 128:
        raise ValueError(
            f"{node_base} electrical nodes in the scene; the kernel supports 128"
        )
    # scene environment + tracer pool + launcher visuals
    devs["environments"] = [
        dict(
            name="air",
            origin_alt=env.origin_alt,
            wind=list(env.wind),
            w20=env.w20,
            seed=env.seed,
            ref_height=10.0,
        )
    ]
    if env.rain or env.rain_rate > 0:
        if env.rain_vehicle and env.rain_vehicle not in roots:
            raise ValueError(
                f"rain_vehicle {env.rain_vehicle!r} is not a vehicle of the scene"
            )
        devs["rains"] = [
            dict(
                name="rain",
                rate=env.rain_rate,
                body=roots.get(env.rain_vehicle, ""),
                catch_area=env.rain_catch_area,
                wet_area=env.rain_wet_area,
            )
        ]
    rw = root.find("worldbody")
    names = []
    for k in range(N_PELLETS):
        b = ET.SubElement(rw, "body", name=f"pellet{k}", mocap="true", pos="0 0 -1000")
        ET.SubElement(
            b,
            "geom",
            type="sphere",
            size=f"{TRACER_RADIUS}",
            group="5",
            contype="0",
            conaffinity="0",
            rgba="1 0.55 0.1 1",
        )
        names.append(f"pellet{k}")
    for g in devs.get("launchers", []):
        g["visual_bodies"] = (
            names  # one launcher per scene is assumed for the tracer pool
        )
    if len(devs.get("launchers", [])) > 1:
        raise ValueError("one launcher per scene (they would share the tracer pool)")
    devs["seed"] = seed
    if wind_field is not None:
        devs["wind_fields"] = [
            wind_field.device(GEOM_NAME if terrain is not None else None)
        ]
    root.find("custom").append(
        ET.Element(
            "text",
            name="dvsim.devices",
            data=json.dumps(devs, sort_keys=True, separators=(",", ":")),
        )
    )
    if terrain is not None:
        add_terrain(root, terrain)
        man["terrain"] = dict(
            kind=terrain.kind, size=list(terrain.size), seed=terrain.seed, n=terrain.nx
        )
        man["pads"] = {
            p.name: dict(x=p.x, y=p.y, z=p.z, radius=p.radius)
            for p in terrain.pads.values()
        }
    ET.indent(root)
    path = out / "scene.xml"
    path.write_text(ET.tostring(root, encoding="unicode"))
    man["model"] = str(path)
    man["dt"] = float(root.find("option").get("timestep"))
    man["body_names"] = _bodies_dfs(root)
    man["body_paths"] = _paths(root)
    man["devices"] = {
        k: [x["name"] for x in v] for k, v in devs.items() if isinstance(v, list)
    }
    man["device_cfg"] = {
        x["name"]: x for k, v in devs.items() if isinstance(v, list) for x in v
    }
    (out / "scene.json").write_text(json.dumps(man, indent=1))
    return dict(model=str(path), manifest=man)
