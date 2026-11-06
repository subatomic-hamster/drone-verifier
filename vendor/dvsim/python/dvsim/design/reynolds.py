"""Reynolds / rotor-speed factor of the rotor device (N4b), shared by the thrust-stand bench and the assembly builder."""

import math

import numpy as np

# UIUC static C_T / C_P vs rpm tables (APC props, all literature data in the parts DB) pooled into the Reynolds factor.
UIUC_STATIC = (
    "uiuc-apcsf-8x3.8",
    "uiuc-apcsf-9x4.7",
    "uiuc-apce-9x4.5",
    "uiuc-apce-11x7",
    "uiuc-apce-14x12",
    "uiuc-apce-17x12",
    "uiuc-apc-sf-10x4.7",
)
RE_X_REF = 4.5  # m^2/s: x = n D^2 at which f = 1 (all seven UIUC tables cover it); = 10 in prop at ~4100 rpm


def reynolds_shape(lib) -> dict:
    """Reynolds / rotor-speed factors f_T(x), f_Q(x), x = n D^2 (rev/s m^2), from the UIUC static tables (J = 0).

    Per prop: f_i(x) = C(x) / C(RE_X_REF) from its own C_T(rpm) (C_Q = C_P / 2 pi: same ratio as C_P). Pooled: at each grid
    point the MEDIAN of ln f_i over the props whose measured x range covers it (>= 2 props; no extrapolation), then
    re-normalised to f(RE_X_REF) = 1. Beyond the covered range the device clamps. Reynolds number at 0.75 R is
    Re = rho (2 pi n) (0.75 R) c / mu ~ n D^2 (c / D): x stands for Re of geometrically similar blades (equal c/D, same
    air); the shape is a property of the blade-section drag (Cd falls ~ Re^-0.2 .. -0.5 over this range) so it transfers to
    other props only to the extent they are APC-like. It is not fitted to any stand row. `lib` = a design Library."""
    curves = []
    for pid in UIUC_STATIC:
        part = lib.get(pid).raw
        d = part["params"]["diameter"]["value"]
        r = np.array(part["tables"]["static_ct_cp"]["rows"], float)
        curves.append((r[:, 0] / 60 * d * d, r[:, 1], r[:, 2]))  # x, C_T, C_P
    grid = sorted({*np.geomspace(2.2, 10.0, 12).tolist(), RE_X_REF})
    xs, lt, lq, cnt = [], [], [], []
    for x0 in grid:
        cov = [c for c in curves if c[0][0] <= x0 <= c[0][-1]]
        if len(cov) < 2:
            continue
        xs.append(x0)
        cnt.append(len(cov))
        for out, col in ((lt, 1), (lq, 2)):
            out.append(
                float(
                    np.median(
                        [
                            math.log(
                                np.interp(x0, c[0], c[col])
                                / np.interp(RE_X_REF, c[0], c[col])
                            )
                            for c in cov
                        ]
                    )
                )
            )
    k = xs.index(RE_X_REF)
    return {
        "x": xs,
        "ft": [math.exp(v - lt[k]) for v in lt],
        "fq": [math.exp(v - lq[k]) for v in lq],
        "n_props": cnt,
        "x_ref": RE_X_REF,
        "props": list(UIUC_STATIC),
    }


def ref_scale(shape: dict, diameter: float, rpm, coef, fkey: str) -> float:
    """k minimising sum (C_i - k f(x_i))^2 over a prop's own static table (x_i = rpm_i / 60 D^2): the coefficient AT
    x_ref that makes `C_ref f(x)` reproduce that table (the table's C themselves sit at their own, higher x)."""
    x = np.asarray(rpm, float) / 60 * diameter**2
    f = np.interp(x, shape["x"], shape[fkey])
    c = np.asarray(coef, float)
    return float(c @ f / (f @ f))
