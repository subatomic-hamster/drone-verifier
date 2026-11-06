"""Named finishes for the visual layer: (rgb, specular, shininess, metallic, roughness) = MuJoCo <material> attributes
(all 0..1). Cosmetic only; a `colors = { slot = [r, g, b] | "name" }` entry in a part file overrides a slot."""

MATERIALS = {
    "alu-anodised-black": ((0.07, 0.07, 0.08), 0.6, 0.7, 0.85, 0.30),
    "alu-anodised-grey": ((0.34, 0.35, 0.38), 0.6, 0.7, 0.85, 0.32),
    "alu-anodised-red": ((0.62, 0.05, 0.05), 0.6, 0.7, 0.85, 0.30),
    "alu-anodised-blue": ((0.07, 0.20, 0.62), 0.6, 0.7, 0.85, 0.30),
    "gunmetal": ((0.20, 0.21, 0.23), 0.6, 0.7, 0.85, 0.35),
    "alu-extrusion": ((0.72, 0.73, 0.75), 0.5, 0.6, 0.9, 0.40),
    "alu-champagne": ((0.66, 0.60, 0.52), 0.6, 0.7, 0.85, 0.35),
    "dxl-black": ((0.085, 0.085, 0.09), 0.45, 0.55, 0.0, 0.38),
    "servo-black": ((0.075, 0.075, 0.08), 0.40, 0.5, 0.0, 0.42),
    "alu-brushed": ((0.70, 0.71, 0.73), 0.5, 0.6, 0.9, 0.38),
    "steel": ((0.62, 0.63, 0.65), 0.7, 0.8, 0.95, 0.25),
    "steel-dark": ((0.30, 0.31, 0.33), 0.6, 0.7, 0.9, 0.35),
    "brass": ((0.78, 0.62, 0.24), 0.7, 0.8, 0.95, 0.30),
    "copper": ((0.80, 0.38, 0.17), 0.6, 0.7, 0.95, 0.30),
    "gold": ((0.83, 0.68, 0.22), 0.8, 0.9, 1.0, 0.22),
    "solder": ((0.72, 0.73, 0.75), 0.7, 0.8, 0.9, 0.3),
    "plastic-black": ((0.045, 0.045, 0.05), 0.30, 0.45, 0.0, 0.50),
    "plastic-dark": ((0.12, 0.12, 0.13), 0.30, 0.45, 0.0, 0.50),
    "plastic-white": ((0.86, 0.86, 0.84), 0.30, 0.45, 0.0, 0.50),
    "plastic-cream": ((0.80, 0.74, 0.58), 0.30, 0.45, 0.0, 0.50),
    "plastic-blue": ((0.04, 0.45, 0.80), 0.35, 0.5, 0.0, 0.45),
    "plastic-yellow": ((0.90, 0.72, 0.08), 0.35, 0.5, 0.0, 0.45),
    "plastic-red": ((0.70, 0.07, 0.06), 0.35, 0.5, 0.0, 0.45),
    "pla-matte": ((0.85, 0.45, 0.10), 0.15, 0.25, 0.0, 0.75),
    "pla-white": ((0.88, 0.88, 0.86), 0.15, 0.25, 0.0, 0.75),
    "carbon-fibre": ((0.075, 0.077, 0.085), 0.55, 0.6, 0.15, 0.35),
    "fiberglass": ((0.56, 0.60, 0.33), 0.5, 0.6, 0.0, 0.35),
    "kraft-paper": ((0.62, 0.47, 0.30), 0.12, 0.2, 0.0, 0.85),
    "kraft-dark": ((0.40, 0.28, 0.17), 0.12, 0.2, 0.0, 0.85),
    "ceramic-grey": ((0.46, 0.45, 0.43), 0.15, 0.25, 0.0, 0.8),
    "hull-orange": ((0.90, 0.38, 0.05), 0.45, 0.55, 0.0, 0.40),
    "rubber": ((0.025, 0.025, 0.027), 0.08, 0.2, 0.0, 0.90),
    "silicone-red": ((0.65, 0.04, 0.04), 0.2, 0.3, 0.0, 0.8),
    "soldermask-green": (
        (0.04, 0.42, 0.19),
        0.35,
        0.5,
        0.0,
        0.45,
    ),  # brightened to match Raspberry Pi 5 product photo
    "soldermask-black": ((0.025, 0.025, 0.028), 0.35, 0.5, 0.0, 0.45),
    "soldermask-blue": ((0.03, 0.10, 0.45), 0.35, 0.5, 0.0, 0.45),
    "soldermask-purple": ((0.25, 0.05, 0.36), 0.35, 0.5, 0.0, 0.45),
    "soldermask-red": ((0.55, 0.04, 0.04), 0.35, 0.5, 0.0, 0.45),
    "ic-black": ((0.02, 0.02, 0.022), 0.3, 0.4, 0.0, 0.55),
    "label-white": ((0.90, 0.90, 0.88), 0.2, 0.3, 0.0, 0.7),
    "shrink-blue": ((0.06, 0.12, 0.45), 0.55, 0.6, 0.0, 0.30),
    "shrink-black": ((0.04, 0.04, 0.045), 0.55, 0.6, 0.0, 0.30),
    "shrink-red": ((0.55, 0.05, 0.05), 0.55, 0.6, 0.0, 0.30),
    "glass-dark": ((0.02, 0.03, 0.06), 1.0, 1.0, 0.0, 0.05),
    "wire-red": ((0.75, 0.04, 0.04), 0.3, 0.4, 0.0, 0.6),
    "wire-black": ((0.03, 0.03, 0.03), 0.3, 0.4, 0.0, 0.6),
}


def resolve_material(spec) -> dict:
    """Material name, or [r, g, b] (a plain painted finish) -> {name, rgba, specular, shininess, metallic, roughness}."""
    if isinstance(spec, str):
        if spec not in MATERIALS:
            raise ValueError(f"unknown material '{spec}'")
        rgb, sp, sh, me, ro = MATERIALS[spec]
        name = spec
    else:
        if not (
            isinstance(spec, (list, tuple))
            and len(spec) == 3
            and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in spec)
        ):
            raise ValueError("colour must be [r, g, b] in 0..1 or a material name")
        rgb, (sp, sh, me, ro) = tuple(float(x) for x in spec), (0.3, 0.45, 0.0, 0.5)
        name = "rgb_%02x%02x%02x" % tuple(round(x * 255) for x in rgb)
    return {
        "name": name,
        "rgba": (*rgb, 1.0),
        "specular": sp,
        "shininess": sh,
        "metallic": me,
        "roughness": ro,
    }
