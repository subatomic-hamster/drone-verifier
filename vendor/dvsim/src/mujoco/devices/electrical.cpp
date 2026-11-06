// Electrical network: dense nodal analysis (ground = node 0), quasi-static per tick.
//
// Fidelity notes (averaged / quasi-static model):
//  - No inductance, no PWM ripple, no bus capacitance: the network is solved algebraically each tick from the state at the
//    start of the tick (rotor speed, throttle, battery SoC / RC voltage, temperature). Real ESC input capacitors and
//    wiring inductance smooth / delay the bus on a sub-ms scale; below our tick rate they are not resolved.
//  - ESC = buck stage of duty d = throttle (linear), linearised for the bus as I_bus = (d + sw) I_m + i_q with
//    I_m = (d V - Ke w)/(R(T) + r_on): g = (d + sw) d/R, i0 = (d + sw) Ke w/R (see eval_esc). r_on (FET conduction),
//    sw (switching + dead-time, fraction of the motor-side power) and i_q (electronics) default to 0 = ideal stage.
//  - A zero-throttle ESC is open (coast), also when `complementary`: real BLHeli ESCs idle with the FETs off; braking
//    applies only while d > 0.
//  - Non-regenerating phase: I_m < 0 -> open (diode freewheel), solved by re-solving until the modes settle.
//  - ESC current limit: reduces |d| (raises it when braking) so |I_m| = i_limit; on the bus that is a constant-power-like
//    sink I_bus = d I_m, handled as a constant current source updated by fixed-point iteration.
//  - Servo H-bridge (S3) goes through the same code (Drv): signed duty d in [-1, 1], synchronous so the phase current may
//    reverse and regenerate into the bus (I_bus = d I_m < 0), I_bus = d I_m linearised exactly as for the ESC. d = 0 with the
//    torque enabled shorts the phase (electrical brake); torque disabled = open phase (I_m = 0, no bus load). Duty is not
//    re-clamped to [-1, 1] when the current limit raises it while braking (as for the ESC).
//  - PSU: ideal source behind a series resistance; unlike a bench supply it can also sink current (regen from a
//    complementary ESC would flow back into it). Current limit = constant-current mode.
//  - gmin = 1e-12 S from every node to ground (SPICE-style) so a floating node solves to 0 V instead of singular.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mujoco/devices/devices.h"

namespace devs {
namespace {
constexpr double kGmin = 1e-12;
constexpr int kMaxIter = 32;  // mode / current-limit re-solves per tick (chatter at a boundary is harmless: continuous)

using Mat = double[kMaxNodes][kMaxNodes + 1];  // augmented [G | rhs]; row = node - 1

// Gaussian elimination with partial pivoting (strict '>' keeps the lowest row on ties: deterministic).
void gauss_solve(int n, Mat& a, double* x) {
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int r = k + 1; r < n; ++r)
      if (std::abs(a[r][k]) > std::abs(a[p][k])) p = r;
    if (a[p][k] == 0.0) throw std::runtime_error("electrical network is singular");
    if (p != k)
      for (int c = k; c <= n; ++c) std::swap(a[p][c], a[k][c]);
    for (int r = k + 1; r < n; ++r) {
      if (a[r][k] == 0.0) continue;  // block-diagonal networks (several vehicles): skipping a zero multiplier changes no value
      const double f = a[r][k] / a[k][k];
      for (int c = k; c <= n; ++c) a[r][c] -= f * a[k][c];
    }
  }
  for (int k = n - 1; k >= 0; --k) {
    double s = a[k][n];
    for (int c = k + 1; c < n; ++c) s -= a[k][c] * x[c];
    x[k] = s / a[k][k];
  }
}

double ocv_at(const BatteryCfg& c, double soc) {  // piecewise linear, clamped at both ends
  const std::size_t n = c.ocv_soc.size();
  if (soc <= c.ocv_soc[0]) return c.ocv_v[0];
  if (soc >= c.ocv_soc[n - 1]) return c.ocv_v[n - 1];
  std::size_t k = 1;
  while (c.ocv_soc[k] < soc) ++k;
  const double f = (soc - c.ocv_soc[k - 1]) / (c.ocv_soc[k] - c.ocv_soc[k - 1]);
  return c.ocv_v[k - 1] + f * (c.ocv_v[k] - c.ocv_v[k - 1]);
}

// One drive stage (ESC or servo H-bridge) + motor as the network sees it. cmd: duty command (throttle >= 0 for an ESC,
// signed for a servo); omega: motor shaft speed; open: phase open (ESC: zero throttle, servo: torque disabled);
// comp: current may reverse (servo: always, synchronous H-bridge).
// r_on / sw / iq: ESC losses (EscCfg), all 0 for a servo and for an ideal ESC (then every expression below reduces to the
// original one bit for bit: x + 0.0 == x).
struct Drv { int node; const MotorCfg* m; double cmd, omega, temp, i_limit; bool comp, open; double r_on = 0, sw = 0, iq = 0; };
Drv drv_of(const EscCfg& c, const EscState& s) {
  // forward-only: open unless throttle > 0 (a negative value never gets here: set() rejects it); bidirectional: open only at exactly 0
  return {c.node, &c.motor, s.throttle, s.omega, s.temp, c.i_limit, c.complementary,
          (c.bidirectional ? s.throttle == 0 : !(s.throttle > 0)) || s.fault_open > 0.5, c.r_on, c.sw_loss, c.i_q};
}
double r_tot(const Drv& g) { return g.m->r_at(g.temp) + g.r_on; }  // winding + FET conduction resistance
Drv drv_of(const ServoCfg& c, const ServoState& s) {
  return {c.node, &c.motor, s.cmd, s.omega, s.temp, c.i_limit, true, servo_open(s)};
}

// mode 0 normal, 1 phase open, 2 current-limited; ic: constant bus current in mode 2
struct DrvEval { int mode; double ic; };

// Switching loss of the drive as a signed duty offset: it follows the direction of the command (a reversing bidirectional ESC; for cmd >= 0
// this is +sw, the original expression).
double sw_signed(const Drv& g) { return g.cmd < 0 ? -g.sw : g.sw; }

// Motor current that makes |I_m| = i_limit and the duty that gives it, at bus voltage V.
void limited(const Drv& g, double V, double& im, double& duty) {
  const double r = r_tot(g), ke = g.m->ke();
  im = std::copysign(g.i_limit, (g.cmd * V - ke * g.omega) / r);
  duty = (im * r + ke * g.omega) / V;
}

// Which regime the drive is in at bus voltage V (from the free, unlimited current at d = cmd).
DrvEval eval_drv(const Drv& g, double V) {
  if (g.open) return {1, 0};
  const double i_free = (g.cmd * V - g.m->ke() * g.omega) / r_tot(g);
  if (!g.comp && (g.cmd < 0 ? i_free > 0 : i_free < 0)) return {1, 0};  // diode freewheel: the current cannot oppose the drive
  if (g.i_limit > 0 && std::abs(i_free) > g.i_limit && V > 1e-9) {
    double im, duty;
    limited(g, V, im, duty);
    return {2, g.sw != 0 ? (duty + sw_signed(g)) * im : duty * im};
  }
  return {0, 0};
}

// Duty and phase current of a drive in `mode` at its node voltage V.
void drv_out(const Drv& g, int mode, double V, double& duty, double& i_m) {
  if (mode == 0) {
    duty = g.cmd;
    i_m = (duty * V - g.m->ke() * g.omega) / r_tot(g);
  } else if (mode == 2) {
    limited(g, V, i_m, duty);
  } else {
    duty = 0;
    i_m = 0;
  }
}
}  // namespace

void solve_electrical(Devices& dv, double dt) {
  const int n = dv.n_nodes;
  const std::size_t ne = dv.esc.cfg.size(), nd = ne + dv.servo.cfg.size(), np = dv.psu.cfg.size();
  auto drv = [&](std::size_t k) {  // drive k: ESCs first, then servos
    return k < ne ? drv_of(dv.esc.cfg[k], dv.esc.st[k]) : drv_of(dv.servo.cfg[k - ne], dv.servo.st[k - ne]);
  };
  dv.drv_mode.assign(nd, 0);
  dv.drv_ic.assign(nd, 0.0);
  dv.psu_lim.assign(np, 0);
  for (std::size_t k = 0; k < nd; ++k) dv.drv_mode[k] = drv(k).open ? 1 : 0;  // canonical start

  Mat a;
  double v[kMaxNodes];
  auto volt = [&](int node) { return node ? v[node - 1] : 0.0; };
  auto cond = [&](int i, int j, double g) {  // conductance between nodes i and j
    if (i) a[i - 1][i - 1] += g;
    if (j) a[j - 1][j - 1] += g;
    if (i && j) { a[i - 1][j - 1] -= g; a[j - 1][i - 1] -= g; }
  };
  auto inject = [&](int i, double amps) { if (i) a[i - 1][n] += amps; };  // current flowing INTO node i

  for (int it = 0; it < kMaxIter; ++it) {
    for (int r = 0; r < n; ++r) {
      std::fill(a[r], a[r] + n + 1, 0.0);
      a[r][r] = kGmin;
    }
    // Battery / PSU as Norton equivalents (E/R into the node, 1/R to ground).
    for (std::size_t i = 0; i < dv.battery.cfg.size(); ++i) {
      const BatteryCfg& c = dv.battery.cfg[i];
      const double e = ocv_at(c, dv.battery.st[i].soc) - dv.battery.st[i].v_rc + dv.battery.st[i].ocv_offset;
      const double r0 = c.r0 + dv.battery.st[i].r_fault;  // F3 cell fault: extra series resistance / lost EMF (0 = healthy)
      cond(c.node, 0, 1.0 / r0);
      inject(c.node, e / r0);
    }
    for (std::size_t i = 0; i < np; ++i) {
      const PsuCfg& c = dv.psu.cfg[i];
      if (!(dv.psu.st[i].enable > 0.5)) continue;  // disabled = open circuit
      if (dv.psu_lim[i]) {
        inject(c.node, c.i_limit);  // current-limited: constant current source
      } else {
        cond(c.node, 0, 1.0 / c.r_series);
        inject(c.node, c.voltage / c.r_series);
      }
    }
    for (const WireCfg& c : dv.wire.cfg) cond(c.a, c.b, 1.0 / c.r);
    // A load draws I = g V - i0 out of its node: conductance g plus a source of i0 into the node.
    for (const LoadCfg& c : dv.load.cfg) {
      cond(c.node, 0, c.g);
      inject(c.node, c.i0);
    }
    for (std::size_t k = 0; k < nd; ++k) {
      const Drv g = drv(k);
      if (dv.drv_mode[k] == 0) {  // I_bus = (d + sw) I_m = ((d + sw) d / R) V - (d + sw) Ke w / R, R = winding + r_on
        const double r = r_tot(g), d = g.cmd, db = g.sw != 0 ? d + sw_signed(g) : d;  // db: bus-side duty (keeps d's bits when sw = 0)
        cond(g.node, 0, db * d / r);
        inject(g.node, db * g.m->ke() * g.omega / r);
      } else if (dv.drv_mode[k] == 2) {
        inject(g.node, -dv.drv_ic[k]);
      }
      inject(g.node, -g.iq);  // quiescent draw of the ESC electronics, in every mode (also at zero throttle)
    }
    // QDD actuators (qdd.cpp): the bus current of the previous tick as a constant sink (their FOC limits need this tick's V first).
    for (std::size_t i = 0; i < dv.qdd.cfg.size(); ++i) inject(dv.qdd.cfg[i].node, -dv.qdd.st[i].bus_i);
    gauss_solve(n, a, v);

    bool changed = false;
    for (std::size_t i = 0; i < np; ++i) {
      const PsuCfg& c = dv.psu.cfg[i];
      if (!(dv.psu.st[i].enable > 0.5) || !(c.i_limit > 0)) continue;
      const double vn = volt(c.node);
      const int lim = dv.psu_lim[i] ? !(vn >= c.voltage - c.r_series * c.i_limit)  // stays limited until V_node recovers
                                    : (c.voltage - vn) / c.r_series > c.i_limit;
      changed |= lim != dv.psu_lim[i];
      dv.psu_lim[i] = lim;
    }
    for (std::size_t k = 0; k < nd; ++k) {
      const Drv g = drv(k);
      const DrvEval ev = eval_drv(g, volt(g.node));
      changed |= ev.mode != dv.drv_mode[k] ||
                 (ev.mode == 2 && std::abs(ev.ic - dv.drv_ic[k]) > 1e-13 * std::abs(ev.ic));
      dv.drv_mode[k] = ev.mode;
      dv.drv_ic[k] = ev.ic;
    }
    if (!changed) break;
  }

  // Outputs from the final node voltages; battery RC / SoC advance with this tick's solved current.
  for (std::size_t i = 0; i < dv.battery.cfg.size(); ++i) {
    const BatteryCfg& c = dv.battery.cfg[i];
    BatteryState& s = dv.battery.st[i];
    s.ocv = ocv_at(c, s.soc);
    s.v = volt(c.node);
    s.i = (s.ocv - s.v_rc + s.ocv_offset - s.v) / (c.r0 + s.r_fault);
    if (c.r1 > 0) {  // exact exponential step of dv1/dt = (I R1 - v1)/tau for I held over the tick
      const double decay = std::exp(-dt / (c.r1 * c.c1));
      s.v_rc = s.v_rc * decay + s.i * c.r1 * (1.0 - decay);
    }
    s.soc -= s.i * dt / (3600.0 * c.capacity_ah);  // not clamped: < 0 flags over-discharge (OCV lookup clamps)
  }
  for (std::size_t i = 0; i < np; ++i) {
    const PsuCfg& c = dv.psu.cfg[i];
    PsuState& s = dv.psu.st[i];
    s.v = volt(c.node);
    s.limited = dv.psu_lim[i];
    s.i = !(s.enable > 0.5) ? 0.0 : dv.psu_lim[i] ? c.i_limit : (c.voltage - s.v) / c.r_series;
  }
  for (std::size_t i = 0; i < dv.wire.cfg.size(); ++i) {
    dv.wire.st[i].v = volt(dv.wire.cfg[i].a) - volt(dv.wire.cfg[i].b);
    dv.wire.st[i].i = dv.wire.st[i].v / dv.wire.cfg[i].r;
  }
  for (std::size_t i = 0; i < dv.load.cfg.size(); ++i) {
    dv.load.st[i].v = volt(dv.load.cfg[i].node);
    dv.load.st[i].i = dv.load.cfg[i].g * dv.load.st[i].v - dv.load.cfg[i].i0;
  }
  auto drive_out = [&](auto& s, const Drv& g, int mode) {  // s: EscState or ServoState
    s.v = volt(g.node);
    drv_out(g, mode, s.v, s.duty, s.i_m);
    s.i_bus = s.duty * s.i_m;  // guarded so a servo / an ideal ESC keeps its bits (0 * -x = -0, -0 + 0 = +0)
    if (g.sw != 0) s.i_bus += sw_signed(g) * s.i_m;
    if (g.iq != 0) s.i_bus += g.iq;
  };
  for (std::size_t e = 0; e < ne; ++e) {
    dv.esc.st[e].mode = dv.drv_mode[e];
    drive_out(dv.esc.st[e], drv(e), dv.drv_mode[e]);
  }
  for (std::size_t i = 0; i < dv.servo.cfg.size(); ++i) drive_out(dv.servo.st[i], drv(ne + i), dv.drv_mode[ne + i]);
  for (std::size_t i = 0; i < dv.qdd.cfg.size(); ++i) dv.qdd.st[i].vbus = volt(dv.qdd.cfg[i].node);
}

}  // namespace devs
