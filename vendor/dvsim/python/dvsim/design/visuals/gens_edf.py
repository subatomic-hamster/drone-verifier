"""Generators for the EDF hopper (examples/hopper): the 12-blade 80 mm ducted-fan rotor, its housing, the flat jet vane and the
printed deck frame. Frames follow each part's own: fan hub centre at the blade plane (+z = shaft towards the inlet); duct on the
same axis (blade plane z = 0, exit plane at -0.075, shoulder at -0.045); vane at its horn seat (plate normal x, chord z, span -y);
frame with the deck's top face at z = 0."""

import math

import numpy as np

from . import gen
from .mesh import circle, extrude, merge, rbox, revolve, rot, rot_to, rrect

TAU = 2 * math.pi


@gen("edf_fan", blade="plastic-dark", hub="alu-anodised-grey")
def edf_fan(c, **_):
    """Hub with a conical spinner and `blade_count` short broad blades pitched 35 deg, out to `diameter` / 2."""
    R = c.p("diameter", default=0.08) / 2
    n = int(c.p("blade_count", default=12))
    rh = 0.0175
    hub = revolve(
        [
            (0, -0.006),
            (rh, -0.006),
            (rh, 0.004),
            (0.012, 0.012),
            (0.005, 0.024),
            (0, 0.027),
        ],
        48,
    )
    outline = np.array(
        [
            (rh - 0.001, -0.007),
            (R - 0.0004, -0.0055),
            (R - 0.0004, 0.0055),
            (rh - 0.001, 0.007),
        ]
    )
    blade = extrude(outline, -0.0005, 0.0005).at(R=rot([1, 0, 0], math.radians(35)))
    blades = merge(
        blade.at((0, 0, 0.0005), R=rot([0, 0, 1], TAU * k / n)) for k in range(n)
    )
    return {"hub": hub, "blade": blades}


@gen("edf_duct", shell="plastic-dark", stator="alu-anodised-grey")
def edf_duct(c, **_):
    """Housing: 1.2 mm wall from the inlet lip (z = +0.028) to the shoulder (-0.045), a contraction to the exit tube whose
    inner radius sets the nozzle area (Ae = pi 0.0329^2 = 3.4e-3 m2), exit at -0.075; seven stator vanes behind the fan."""
    ro, ri, rt, ti = 0.0415, 0.0403, 0.0359, 0.0329
    wall = revolve(
        [
            (ti, -0.075), (rt, -0.075), (rt, -0.045), (ro, -0.045), (ro, 0.024), (ro + 0.002, 0.024),
            (ro + 0.002, 0.028), (ri, 0.028), (ri, -0.030), (ti, -0.040),
        ],
        64,
    )  # fmt: skip
    vane = rbox((0.0228, 0.0016, 0.020), 0.0004).at((0.0289, 0.0, -0.022))
    stator = merge(vane.at(R=rot([0, 0, 1], TAU * k / 7)) for k in range(7))
    return {"shell": wall, "stator": stator}


@gen("jet_vane", plate="alu-anodised-grey", boss="steel-dark")
def jet_vane(c, **_):
    """Flat aluminium plate in the jet: normal x, chord z (hinge at 25 % of the chord from the top edge), span from the horn
    seat inwards (-y)."""
    ch = c.p("chord", default=0.045)
    sp = c.p("span", default=0.034)
    th = c.p("thickness", default=0.0015)
    plate = rbox((th, sp, ch), 0.0003).at((0, -0.003 - sp / 2, ch / 4 - ch / 2))
    boss = revolve([(0, -0.004), (0.0065, -0.004), (0.0065, 0.0), (0, 0.0)], 32).at(
        R=rot_to([0, 1, 0])
    )
    return {"plate": plate, "boss": boss}


@gen("hopper_frame", frame="pla-matte")
def hopper_frame(c, **_):
    """Printed deck: 0.20 m square plate (deck top face at z = 0, 4 mm thick) with the duct hole, four diagonal arms to the
    leg hinges and four servo brackets hanging 0.0674 m from the axis on the x / y axes."""
    s, t = c.p("side", default=0.20), c.p("thickness", default=0.004)
    rh = c.p("duct_hole_radius", default=0.0375)
    deck = extrude(rrect(s, s, 0.012), -t, 0.0, holes=[circle(rh, 64)[::-1]])
    arm = rbox((0.057, 0.02, t), 0.001).at((0.119, 0, -t / 2))
    arms = merge(
        arm.at(R=rot([0, 0, 1], math.pi / 4 + k * math.pi / 2)) for k in range(4)
    )
    br = rbox((0.003, 0.032, 0.1085), 0.0005).at((0.0674, 0, -t - 0.05425))
    brackets = merge(br.at(R=rot([0, 0, 1], k * math.pi / 2)) for k in range(4))
    return {"frame": merge([deck, arms, brackets])}
