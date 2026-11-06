"""Parts library: parses parts/**/*.toml. A file that does not parse is recorded in `Library.errors` and only raises
(DesignError) when an assembly asks for that part."""

import tomllib
from pathlib import Path

from . import pins as _pins

GENDERS = ("male", "female", "neutral")


class DesignError(Exception):
    """Any user-facing assembly / library problem (the CLI turns it into exit code 2)."""


def genders_ok(a: str, b: str) -> bool:
    """male+female, or either side neutral."""
    return a == "neutral" or b == "neutral" or {a, b} == {"male", "female"}


_PAIRED_FAMILIES = ({"shaft", "bore"}, {"tube", "tube-clamp"})


def types_ok(a: str, b: str) -> bool:
    """Same port type, or paired families with the same size: shaft.<d> / bore.<d>, tube.<od> / tube-clamp.<od>."""
    if a == b:
        return True
    (fa, _, sa), (fb, _, sb) = a.partition("."), b.partition(".")
    return bool(sa) and sa == sb and {fa, fb} in _PAIRED_FAMILIES


class Part:
    def __init__(self, doc: dict, path: str):
        self.raw = doc
        self.path = path
        self.id = doc["id"]
        self.category = doc.get("category", "")
        self.params = doc.get("params", {})
        self.ports = doc.get("ports", [])
        self.geometry = doc.get("geometry", {})

    @property
    def mirror(self) -> dict:
        """The part's `[mirror]` table: `symmetry = "xy"|"yz"|"xz"` (symmetric about that local plane: a mirrored copy is the same
        part) or `opposite = "<part-id>"` (+ `plane`, default "xz": the opposite-hand part is this one reflected in that local
        plane, its ports the reflection of this one's). {} = cannot be mirrored."""
        m = self.raw.get("mirror")
        return m if isinstance(m, dict) else {}

    def mirror_plane(self):
        """Local reflection plane S of a mirrored copy (None when the part declares neither symmetry nor opposite)."""
        m = self.mirror
        return m.get("symmetry") or (
            m.get("plane", "xz") if m.get("opposite") else None
        )

    @property
    def connectors(self) -> dict:
        """Pin-level connectors by id (pins.py); {} for a part without a pinout."""
        return _pins.connectors(self)

    def port(self, name: str) -> dict:
        for p in self.ports:
            if p.get("name") == name:
                return p
        for p in _pins.synthetic_ports(self):  # a connector's unclaimed pins (pins.py)
            if p["name"] == name:
                return p
        names = [p.get("name") for p in self.ports]
        raise DesignError(f"part {self.id} has no port '{name}' (ports: {names})")

    def value(self, name: str):
        """Numeric value of params[name] or None."""
        e = self.params.get(name)
        v = e.get("value") if isinstance(e, dict) else None
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def has_port(self, ptype: str, gender: str | None = None) -> bool:
        return any(
            p.get("type") == ptype and (gender is None or p.get("gender") == gender)
            for p in self.ports
        )


class Library:
    def __init__(self, parts_dir):
        self.dir = Path(parts_dir)
        self.root = (
            self.dir.resolve().parent
        )  # repo root: mesh paths are relative to it
        self.parts: dict[str, Part] = {}
        self.errors: list[str] = []  # "relative/path.toml: message"
        self.broken: dict[str, str] = {}  # file stem -> message
        for f in sorted(self.dir.rglob("*.toml")):
            rel = f.relative_to(self.dir).as_posix()
            if rel.startswith("_") or "/_" in rel:  # _prices/ and other non-part data
                continue
            try:
                doc = tomllib.loads(f.read_text(encoding="utf-8"))
                pid = doc.get("id")
                if not isinstance(pid, str) or not pid:
                    raise ValueError("missing 'id'")
                if not isinstance(doc.get("ports", []), list) or not all(
                    isinstance(p, dict) for p in doc.get("ports", [])
                ):
                    raise ValueError("'ports' must be an array of tables")
                if not isinstance(doc.get("params", {}), dict):
                    raise ValueError("'params' must be a table")
                self.errors += [
                    f"{rel}: {m}" for m in _pins.validate(doc)
                ]  # the part still loads
                if pid in self.parts:
                    raise ValueError(
                        f"duplicate id '{pid}' (also {self.parts[pid].path})"
                    )
            except (tomllib.TOMLDecodeError, ValueError, OSError) as e:
                self.errors.append(f"{rel}: {e}")
                self.broken[f.stem] = f"{rel}: {e}"
                continue
            self.parts[pid] = Part(doc, rel)
        self._merge_prices()

    def _merge_prices(self):
        """parts/_prices/*.toml `[prices]` -> params: value -> price (USD), price_per_m (USD/m), price_per_kg (USD/kg),
        not_for_sale. A param already in the part's own TOML wins."""
        units = {
            "value": ("price", "USD"),
            "price_per_m": ("price_per_m", "USD/m"),
            "price_per_kg": ("price_per_kg", "USD/kg"),
        }
        for f in sorted((self.dir / "_prices").glob("*.toml")):
            rel = f.relative_to(self.dir).as_posix()
            try:
                prices = tomllib.loads(f.read_text(encoding="utf-8")).get("prices", {})
            except (tomllib.TOMLDecodeError, OSError) as e:
                self.errors.append(f"{rel}: {e}")
                continue
            for pid, e in prices.items():
                key = next(
                    (
                        k
                        for k in (*units, "not_for_sale")
                        if isinstance(e, dict) and k in e
                    ),
                    None,
                )
                v = e.get(key) if key else None
                if pid not in self.parts:
                    self.errors.append(f"{rel}: unknown part id '{pid}'")
                elif key is None or (
                    v is not True
                    if key == "not_for_sale"
                    else isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or v <= 0
                ):
                    self.errors.append(
                        f"{rel}: '{pid}': need a positive value / price_per_m / price_per_kg, or not_for_sale = true"
                    )
                else:
                    name, unit = units.get(key, ("not_for_sale", "1"))
                    rec = {
                        "value": v,
                        "unit": unit,
                        "fidelity": e.get("fidelity", "estimated"),
                    }
                    rec |= {k: e[k] for k in ("rationale", "src") if k in e}
                    self.parts[pid].params.setdefault(name, rec)

    def get(self, pid: str) -> Part:
        if pid in self.parts:
            return self.parts[pid]
        if pid in self.broken:
            raise DesignError(f"part '{pid}' failed to load: {self.broken[pid]}")
        raise DesignError(f"unknown part '{pid}'")

    def by_category(self, prefix: str) -> list[Part]:
        """Parts whose dotted category equals `prefix` or starts with `prefix.`; sorted by id."""
        return [
            p
            for _, p in sorted(self.parts.items())
            if p.category == prefix or p.category.startswith(prefix + ".")
        ]

    def by_port_type(self, ptype: str) -> list[Part]:
        return [p for _, p in sorted(self.parts.items()) if p.has_port(ptype)]

    def compatible(self, ptype: str, gender: str) -> list[tuple[Part, dict]]:
        """(part, port) pairs that can mate a port of this type and gender; sorted by part id."""
        return [
            (p, port)
            for _, p in sorted(self.parts.items())
            for port in p.ports
            if types_ok(port.get("type", ""), ptype)
            and genders_ok(port.get("gender", ""), gender)
        ]
