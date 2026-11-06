// Solid rocket motors and rocket airframe aerodynamics (phase 5). Fidelity notes live in docs/models.md.
//
// Motor. Thrust is the published curve (RASP .eng points, linear between points). With the integral I(t) of that piecewise-linear
// curve known in closed form per segment (cum[] at the knots + a quadratic inside a segment) the tick force is the MEAN over the
// tick, (I(t1) - I(t0)) / dt: the impulse delivered is the curve's integral exactly, for any dt, and the propellant mass burned is
// proportional to it (progress = I(t) / I_total). Thrust is applied at the nozzle site along its +z; the torque about the body CoM is
// (nozzle - CoM) x F, so a motor that sits on a gimbal pushes the vehicle through the gimbal's hinge joints, a rigidly mounted one
// through its body.
//
// Changing a mass in MuJoCo: the dynamics read body_mass / body_inertia / body_ipos / body_iquat (composite inertia, mj_crb) and
// body_subtreemass (subtree CoM) from the mjModel, once per mj_step1. prepare_motors rewrites exactly those for the motor body at
// the START of every tick, as a pure function of the motor state (mass from the burn progress, inertia of the casing + remaining
// propellant from the config), before mj_step1 builds the composite inertias from them. Nothing else caches the mass: the constants
// of mj_setConst (dof_M0 of "simple" bodies, the invweight0 of the constraint reference acceleration, meaninertia) keep their
// compile-time values; dof_M0 is refused at bind() (a simple motor body), the others only scale the constraint softness and the
// solver tolerance (a few % for a rocket that loses 5 % of its mass). Because the arrays are recomputed from the saved state every
// tick, a restored, cloned or replayed run (whose model holds whatever it held) is bitwise identical. Subtree masses are rebuilt
// with the same loop as mj_setConst (no accumulated deltas), for the same reason.
//
// Airframe. See schema Airframe; velocities of the CP point include the angular velocity, so pitch damping from the fins' normal
// force comes out of the model.
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {
namespace {
inline double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Integral of the piecewise-linear thrust curve from 0 to tb (tb clamped to [0, burn time]).
double icum(const SolidMotorCfg& c, double tb) {
  if (tb <= c.t.front()) return 0.0;
  if (tb >= c.t.back()) return c.i_total;
  std::size_t k = 1;
  while (c.t[k] < tb) ++k;  // tb in (t[k-1], t[k]]
  const double t0 = c.t[k - 1], w = c.t[k] - t0, tau = tb - t0, f0 = c.f[k - 1], f1 = c.f[k];
  return c.cum[k - 1] + f0 * tau + 0.5 * (f1 - f0) / w * tau * tau;
}
}  // namespace

void prepare_motors(mjModel* m, Devices& dv) {
  if (dv.motor.cfg.empty()) return;
  for (std::size_t i = 0; i < dv.motor.cfg.size(); ++i) {
    const SolidMotorCfg& c = dv.motor.cfg[i];
    const SolidMotorState& s = dv.motor.st[i];
    const int b = c.bid;
    const double mp = c.prop_mass * (1.0 - s.progress), mc = c.total_mass - c.prop_mass;  // propellant left, casing (+ delay element)
    const double r = 0.5 * c.diameter, l2 = c.length * c.length / 12.0;
    // Casing: thin cylindrical shell; propellant: solid cylinder (the same radius), both uniform over the length.
    const double i_ax = mc * r * r + 0.5 * mp * r * r;
    const double i_lat = mc * (0.5 * r * r + l2) + mp * (0.25 * r * r + l2);
    m->body_mass[b] = mc + mp;
    m->body_inertia[3 * b + 0] = m->body_inertia[3 * b + 1] = i_lat;
    m->body_inertia[3 * b + 2] = i_ax;
    // Principal axes = the nozzle site's (z = motor axis); CoM half a length ahead of the nozzle exit, on the axis.
    const double* sq = m->site_quat + 4 * c.sid;
    const double ez[3] = {0, 0, 1};
    double ax[3];
    mju_rotVecQuat(ax, ez, sq);
    for (int k = 0; k < 3; ++k) m->body_ipos[3 * b + k] = m->site_pos[3 * c.sid + k] + 0.5 * c.length * ax[k];
    for (int k = 0; k < 4; ++k) m->body_iquat[4 * b + k] = sq[k];
  }
  // Same loop (same order of additions) as mj_setConst, so an untouched model keeps its compile-time values bit for bit.
  for (int i = 0; i < m->nbody; ++i) m->body_subtreemass[i] = m->body_mass[i];
  for (int i = m->nbody - 1; i > 0; --i) m->body_subtreemass[m->body_parentid[i]] += m->body_subtreemass[i];
}

void step_motors(const mjModel* m, mjData* d, Devices& dv) {
  const double dt = m->opt.timestep;
  for (std::size_t i = 0; i < dv.motor.cfg.size(); ++i) {
    const SolidMotorCfg& c = dv.motor.cfg[i];
    SolidMotorState& s = dv.motor.st[i];
    for (int k = 0; k < 3; ++k) {
      s.pos[k] = d->site_xpos[3 * c.sid + k];
      s.dir[k] = d->site_xmat[9 * c.sid + 3 * k + 2];
    }
    if (s.ignite > 0.5) {
      if (s.phase == 0) {
        s.phase = 1;
        s.t_ign = 0;
      }
      s.ignite = 0;
    }
    double thrust = 0;
    if (s.phase >= 1) {
      const double tburn = c.t.back(), tb0 = s.t_ign - c.delay, tb1 = tb0 + dt;
      const double i0 = icum(c, std::clamp(tb0, 0.0, tburn)), i1 = icum(c, std::clamp(tb1, 0.0, tburn));
      thrust = s.thrust_scale * (i1 - i0) / dt;
      s.impulse += s.thrust_scale * (i1 - i0);
      s.progress = i1 / c.i_total;
      s.t_ign += dt;
      s.t_burn = std::clamp(tb1, 0.0, tburn);
      s.phase = tb1 <= 0 ? 1.0 : tb1 >= tburn ? 3.0 : 2.0;
    }
    s.thrust = thrust;
    s.burned = c.prop_mass * s.progress;
    s.mass = c.total_mass - s.burned;
    if (thrust != 0) {
      double f[3], rr[3], tq[3];
      for (int k = 0; k < 3; ++k) {
        f[k] = thrust * s.dir[k];
        rr[k] = s.pos[k] - d->xipos[3 * c.bid + k];
      }
      mju_cross(tq, rr, f);
      for (int k = 0; k < 3; ++k) {
        d->xfrc_applied[6 * c.bid + k] += f[k];
        d->xfrc_applied[6 * c.bid + 3 + k] += tq[k];
      }
    }
  }
}

void step_airframes(const mjModel* m, mjData* d, Devices& dv) {
  if (dv.airframe.cfg.empty()) return;
  double wind[3];
  wind_at(dv, wind);
  for (std::size_t i = 0; i < dv.airframe.cfg.size(); ++i) {
    const AirframeCfg& c = dv.airframe.cfg[i];
    AirframeState& s = dv.airframe.st[i];
    double z[3];
    for (int k = 0; k < 3; ++k) z[k] = d->site_xmat[9 * c.sid + 3 * k + 2];
    double vel[6];
    mj_objectVelocity(m, d, mjOBJ_SITE, c.sid, vel, 0);  // rot:lin of the site point, world axes
    double arm[3], cp[3], vcp[3], w[3] = {vel[0], vel[1], vel[2]};
    for (int k = 0; k < 3; ++k) {
      arm[k] = c.x_cp * z[k];
      cp[k] = d->site_xpos[3 * c.sid + k] + arm[k];
    }
    mju_cross(vcp, w, arm);
    double vr[3];
    for (int k = 0; k < 3; ++k) vr[k] = vel[3 + k] + vcp[k] - wind[k];
    const double va = dot3(vr, z);
    double vp[3];
    for (int k = 0; k < 3; ++k) vp[k] = vr[k] - va * z[k];
    const double vpn = std::sqrt(dot3(vp, vp)), half_rho = 0.5 * rho_at(dv, cp[2]);
    const double cd = va >= 0 ? c.cd0 : c.cd_rev;
    const double ka = -half_rho * c.s_ref * cd * std::abs(va);
    const double kc = -half_rho * (c.s_ref * c.cn_alpha * std::abs(va) + c.s_plan * c.cd_cross * vpn);
    double f[3], rr[3], tq[3];
    for (int k = 0; k < 3; ++k) {
      f[k] = ka * va * z[k] + kc * vp[k];
      rr[k] = cp[k] - d->xipos[3 * c.bid + k];
      s.force[k] = f[k];
    }
    mju_cross(tq, rr, f);
    for (int k = 0; k < 3; ++k) {
      d->xfrc_applied[6 * c.bid + k] += f[k];
      d->xfrc_applied[6 * c.bid + 3 + k] += tq[k];
    }
    s.alpha = std::atan2(vpn, va);  // angle between the axis and the air velocity (0 when there is no flow)
    s.speed = std::sqrt(dot3(vr, vr));
  }
}

}  // namespace devs
