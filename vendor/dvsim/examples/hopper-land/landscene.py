"""Scenes of examples/hopper-land: assemblies/hopper-land.toml (a copy of the hopper) standing on a take-off pad of a seeded terrain at 2500-4000 m, with the terrain-aware wind field, the
rain device when it rains, or on a flat floor beside a scripted moving deck.

    build_model(out, p, seed, scatter, visual)   -> (scene.xml, Info)        p = resolved scenario parameters (landing.PARAMS)
    build(out) / build_for(scenario_toml)(out)   the `[model] script` entry point of a scenario file (its defaults)

Sites (everything synthetic and seeded: dvsim.terrain; the shapes are plausible, not surveyed terrain). Pad radii 2 m (the hopper stands on a 0.29 m square of feet); cells of 0.6-0.8 m, so
every pad is a flat disc of ~3.2 m radius (terrain.Pad.flat_radius). Relief 30-45 m: the hopper's battery is a hover of about two minutes (examples/hopper/README.md), a cliff of hundreds of
metres is not a hop.

    mesa    plateau and lowland across a straight 30 m cliff (cliff width 10 m), pads 24 m either side of the brink          origin_alt 3200 m
    ridge   a ridge crest 45 m high, summit pad and a second pad 46 m along the crest, wind across the ridge                  origin_alt 3800 m
    ledge   the same ridge with a steep lee face, summit pad and a carved ledge pad on the lee face 13 m lower               origin_alt 3400 m
    moor    rolling moorland, hilltop pad and a pad 28 m away on the slope                                                  origin_alt 2800 m
    deck    flat floor at sea level (no terrain) and a scripted moving platform: heave, roll, pitch, drift
"""

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "examples" / "hopper"))
sys.path.insert(0, str(HERE))

import scene as hopper_scene  # examples/hopper/scene.py: scatter_xml, M8N (the original, unchanged)
from dvsim import design
from dvsim.design.assembly import load
from dvsim.terrain import PadSpec, WindField, add_terrain, generate

ASSEMBLY = REPO / "assemblies" / "hopper-land.toml"
FRAME_Z = 0.2509  # m, the frame body's height over the feet in the built MJCF (assembly: `<body name="frame" pos="0 0 0.2509">`)
_LIB = None


@dataclass(frozen=True)
class Site:
    kind: str  # dvsim.terrain kind, or "deck" (flat floor + moving platform)
    size: tuple = (300.0, 220.0)
    n: int = 385
    relief: float = 40.0
    roughness: float = 0.5
    hurst: float = 0.8
    seed: int = 1
    pads: tuple = ()  # (name, at, radius): `at` an anchor name or an (x, y) in the terrain frame (m)
    takeoff: str = ""
    land: str = ""
    shape: dict = field(default_factory=dict)
    rocks: int = 0
    origin_alt: float = 3200.0
    wind_dir_deg: float = 0.0  # direction the wind blows TOWARDS (0 = +x east)
    ridge_len: float = 25.0
    lee_turb: float = 2.0
    lee_down: float = 0.2
    z0: float = 0.01  # roughness length m: bare rock / sparse alpine ground (Wieringa 1992: 0.001-0.01 rock, snow; 0.03 short grass)


SITES = {
    "mesa": Site(
        "mesa",
        (240.0, 180.0),
        385,
        30.0,
        0.3,
        seed=5,
        shape=dict(cliff_width=10.0, wobble=2.0),
        pads=(("plateau", (4.8, 0.0), 2.0), ("lowland", (52.8, 0.0), 2.0)),
        takeoff="plateau",
        land="lowland",
        rocks=30,
        origin_alt=3200.0,
    ),
    "ridge": Site(
        "ridge",
        (300.0, 220.0),
        385,
        45.0,
        0.5,
        seed=3,
        pads=(("summit", "summit", 2.0), ("crest", (-4.2, 72.0), 2.0)),
        takeoff="summit",
        land="crest",
        rocks=30,
        origin_alt=3800.0,
        ridge_len=25.0,
    ),
    "ledge": Site(
        "ledge",
        (300.0, 220.0),
        385,
        45.0,
        0.5,
        seed=4,
        pads=(("summit", "summit", 2.0), ("ledge", "ledge", 2.0)),
        takeoff="summit",
        land="ledge",
        rocks=30,
        origin_alt=3400.0,
        ridge_len=25.0,
        lee_turb=2.5,
        lee_down=0.25,
    ),
    "moor": Site(
        "moorland",
        (300.0, 225.0),
        385,
        30.0,
        0.4,
        seed=2,
        pads=(("hilltop", "hilltop", 2.0), ("slope", (82.0, -4.5), 2.0)),
        takeoff="hilltop",
        land="slope",
        rocks=40,
        origin_alt=2800.0,
        z0=0.03,
        ridge_len=20.0,
    ),
    "deck": Site("deck", origin_alt=0.0),
    "osv": Site(
        "osv", origin_alt=0.0
    ),  # the 12.6 m support vessel of dvsim.cuas.ship underway in a seeded JONSWAP sea, the hopper on its helipad
}
# the ISA rho at those altitudes, kg/m3: 3200 m 0.891, 3800 m 0.837, 3400 m 0.873, 2800 m 0.927 (landveh.isa_rho)


@dataclass
class Info:
    """What the harness needs to know about the built scene (the mission planner's survey data, never the flight computer's sensors)."""

    site: str
    terrain: object  # dvsim.terrain.Terrain or None
    takeoff: tuple  # (x, y, z) world surface point of the take-off pad
    land: tuple  # same for the landing pad
    origin_alt: float
    pad_radius: float
    flat_radius: float
    deck: dict = field(
        default_factory=dict
    )  # deck platform geometry when site == "deck"


def make_terrain(site: Site, reverse=False, seed=0):
    """The site's terrain; `seed` > 0 replaces the site's seed (new fBm detail and crest/cliff phases, same pad positions)."""
    pads = [PadSpec(n, at=a, radius=r, blend=15.0) for n, a, r in site.pads]
    if reverse:  # the datum (z = 0) is the take-off pad: the second pad goes first
        pads = pads[::-1]
    return generate(
        site.kind,
        site.size,
        site.n,
        site.relief,
        site.roughness,
        site.hurst,
        seed or site.seed,
        pads,
        site.rocks,
        **site.shape,
    )


def _devices_edit(root, fn):
    dev = root.find("custom/text[@name='dvsim.devices']")
    d = json.loads(dev.get("data"))
    fn(d)
    dev.set("data", json.dumps(d, sort_keys=True, separators=(",", ":")))


def _compute(out, seed, w20, visual):
    global _LIB
    asm = load(ASSEMBLY)
    asm.doc["assembly"]["seed"] = int(seed)
    if w20 is not None:
        env = next(d for d in asm.doc["devices"] if d["type"] == "environment")
        env["set"]["w20"]["value"] = float(w20)
    if _LIB is None:
        _LIB = design.Library(REPO / "parts")
    xml, _ = design.compute(asm, REPO / "parts", out, lib=_LIB, visual=visual)
    return xml


def add_deck(root, y_m, x_m=0.0, size=(1.6, 1.6)):
    """The scripted deck: a body on three slides (x, y, z) and two hinges (roll about x, pitch about y) driven by position actuators (stiff, gravity-compensated; landing.Deck commands them each control
    tick). A box deck surface (contact, ray-visible group 0) with a painted H, 0.5 m above the floor at rest. Joints/actuators: deck_x deck_y deck_z deck_r deck_p."""
    wb = root.find("worldbody")
    b = ET.SubElement(wb, "body", name="deck", pos=f"{x_m} {y_m} 0.5", gravcomp="1")
    for nm, ax, typ in (
        ("x", "1 0 0", "slide"),
        ("y", "0 1 0", "slide"),
        ("z", "0 0 1", "slide"),
        ("r", "1 0 0", "hinge"),
        ("p", "0 1 0", "hinge"),
    ):
        ET.SubElement(b, "joint", name=f"deck_{nm}", type=typ, axis=ax, damping="0")
    ET.SubElement(b, "geom", name="deck_top", type="box", size=f"{size[0]} {size[1]} 0.05", pos="0 0 -0.05", mass="60", contype="1", conaffinity="1", group="0", rgba="0.35 0.37 0.40 1", friction="1.0 0.005 0.0001")  # fmt: skip
    ET.SubElement(b, "geom", name="deck_ring", type="cylinder", size=f"{0.9} 0.002", pos="0 0 0.003", mass="0.001", contype="0", conaffinity="0", group="4", rgba="0.95 0.45 0.08 1")  # fmt: skip
    ET.SubElement(b, "geom", name="deck_H", type="box", size="0.35 0.06 0.002", pos="0 0 0.006", mass="0.001", contype="0", conaffinity="0", group="4", rgba="0.96 0.96 0.96 1")  # fmt: skip
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for nm, kp, kv in (
        ("x", 4e5, 6e4),
        ("y", 4e5, 6e4),
        ("z", 4e5, 6e4),
        ("r", 4e5, 6e4),
        ("p", 4e5, 6e4),
    ):
        ET.SubElement(
            act,
            "position",
            name=f"deck_{nm}",
            joint=f"deck_{nm}",
            kp=str(kp),
            kv=str(kv),
        )


def _build_osv(out, p, seed, oa, visual):
    """The hopper standing on the helipad of the support vessel (dvsim.cuas.ship.ship_scene; the lander is the hopper-land assembly, which composes as `hopper_*`). The vehicle scatter
    of the other sites (mass / thrust / CG) is not applied here: ship_scene composes the assembly as it is; sensor biases and servo scatter still are (landhop.fly)."""
    from dvsim.cuas.ship import ship_scene

    sc = ship_scene(
        out,
        lander="hopper-land",
        lander_name="hopper",
        lander_start="deck",
        hs=float(p["hs_m"]),
        tp=None if p["tp_s"] <= 0 else float(p["tp_s"]),
        wave_dir=float(p["wave_dir_deg"]),
        origin_alt=oa,
        seed=int(seed),
        visual=visual,
    )
    pad = tuple(float(v) for v in sc["pad"])
    info = Info("osv", None, pad, pad, oa, 2.0, 0.0)
    info.deck = dict(x=pad[0], y=pad[1], z=pad[2], half=1.5, ship=True)
    return Path(sc["model"]), info


def build_model(out, p, seed=1, scatter=None, visual=False):
    """MJCF of one flight. `p` keys: site, reverse (land on the take-off pad of the site's reverse hop), origin_alt (-1 = the site's), w20_ms, rain_mmh (device present iff > 0), deck_y_m (site deck),
    wind_field (False = uniform wind, no terrain wind). -> (path of scene.xml, Info)"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    site = SITES[p["site"]]
    oa = site.origin_alt if p.get("origin_alt", -1.0) < 0 else float(p["origin_alt"])
    if site.kind == "osv":
        return _build_osv(out, p, seed, oa, visual)
    xml = _compute(out, seed, p["w20_ms"] if p["w20_ms"] >= 0 else p["wind_ms"], visual)
    if scatter:
        xml = hopper_scene.scatter_xml(xml, scatter)
    root = ET.fromstring(xml)
    frame = next(
        b for b in root.find("worldbody").findall("body") if b.get("name") == "frame"
    )
    rain = float(p.get("rain_mmh", 0.0)) > 0.0

    def devs(d):
        for e in d["environments"]:
            e["origin_alt"] = oa
        for g in d["gnss_receivers"]:
            g["alt0"] = (
                oa  # the receiver reports MSL height: the flight computer reads its site altitude from it
            )
        if rain:
            # ESTIMATES (flagged in the README): the hopper's catch area is the duct inlet (80 mm: 0.005 m2) plus the deck seen from above (0.2 x 0.2 m); wet area the deck, duct and battery.
            d["rains"] = [
                dict(
                    name="rain", rate=0.0, body="frame", catch_area=0.045, wet_area=0.12
                )
            ]

    _devices_edit(root, devs)
    info = Info(p["site"], None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), oa, 2.0, 0.0)
    if site.kind == "deck":
        add_deck(root, y_m=float(p.get("deck_y_m", 6.0)))
        info.deck = dict(x=0.0, y=float(p.get("deck_y_m", 6.0)), z=0.5, half=1.6)
        info.takeoff, info.land = (
            (0.0, 0.0, 0.0),
            (0.0, float(p.get("deck_y_m", 6.0)), 0.5),
        )
    else:
        t = make_terrain(
            site, bool(p.get("reverse", False)), int(p.get("terrain_seed", 0))
        )
        a, b = (
            (site.land, site.takeoff)
            if p.get("reverse", False)
            else (site.takeoff, site.land)
        )
        pa, pb = t.pad(a), t.pad(b)
        px, py, pz = (
            frame.get("pos").split()[:3] if frame.get("pos") else (0, 0, FRAME_Z)
        )
        frame.set("pos", f"{pa.x} {pa.y} {pa.z + float(pz)}")
        wf = (
            WindField(
                z0=site.z0,
                ridge_len=site.ridge_len,
                lee_turb=site.lee_turb,
                lee_down=site.lee_down,
            )
            if p.get("wind_field", True)
            else WindField(z0=0.0, ridge_gain=0.0, lee_turb=1.0, lee_down=0.0)
        )
        add_terrain(root, t, wind=wf, origin_alt=oa)
        info.terrain, info.takeoff, info.land = (
            t,
            (pa.x, pa.y, pa.z),
            (pb.x, pb.y, pb.z),
        )
        info.flat_radius = pa.flat_radius
    ET.indent(root)
    path = out / "scene.xml"
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path, info


# ---- the `[model] script` entry points of the scenario files ---------------------------------------------------------------------------------------
def build_for(scenario_toml):
    """-> build(out): the scenario's model at its default parameters (what `dvsim program` loads and the viewer shows)."""
    import landing

    def build(out):
        p = landing.resolve(scenario_toml)
        h = landing.draw_hidden(p["seed"], p)
        return build_model(out, p, p["seed"], h["scatter"], visual=True)[0]

    return build
