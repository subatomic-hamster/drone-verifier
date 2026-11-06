// Environment (ISA, steady wind, Dryden turbulence), rotor aero, body drag. Phase 5 of the device tick: everything here
// only adds to d->xfrc_applied (force at the body CoM xipos, torque about it); nothing writes qfrc_applied, so the motor
// torque (joint-internal, electrical) and the aero torque (external, on the rotor body) stay distinct.
//
// Fidelity notes:
//  - Rotor: steady-state C_T(J), C_Q(J) tables (quasi-static: no inflow dynamics, no blade flapping, no Mach dependence,
//    no tip-vortex or wake interaction between rotors). Reynolds dependence only when the optional re_x / re_ft / re_fq
//    tables are given: C *= f(x), x = |n| D^2 (docs/models.md). Thrust and aero torque are evaluated from the
//    state at the start of the tick (rotor speed, CoM velocity). Below |n| = 1e-9 rev/s J is set to 0 (static values).
//  - Reverse spin: thrust reverses (T ~ sign(dir w) n^2), J is measured in the thrust direction, and the same C_T, C_Q
//    tables are used mirrored; a real propeller's reverse performance differs: only a constant thrust factor (`rev_thrust`,
//    default 1) models it, the J dependence of the reverse curve and the torque are the mirrored forward ones.
//  - Aero torque magnitude is C_Q rho n^2 D^5 and always opposes w; for C_Q < 0 (windmilling) it assists the spin.
//  - H-force: F = -k_h |w| v_perp at the CoM (no moment about the hub, no blade-flapping coupling).
//  - Ground effect: Cheeseman-Bennett T_IGE / T_OGE = 1 / (1 - (R / 4z)^2), z from one ray along the downwash (-s a, opposite
//    the thrust vector; -a at w = 0) from the CoM (distance along the axis, not height: a tilted rotor sees 1 / cos). Clamped to z >= R / 2 (ratio <= 4/3), 1 on
//    no hit. Applied to thrust only (torque unchanged). The ray ignores the rotor body and geoms outside `ray_groups`:
//    the vehicle's own geoms must sit in a group outside the mask or the ray hits them. mj_ray also skips fully transparent geoms.
//  - Air density: ISA troposphere at the body CoM altitude (origin_alt + z), pressure / temperature offsets = 0
//    (standard day). Without an environment device: ISA with origin altitude 0, zero wind.
//  - Wind: ONE steady + turbulence vector for the whole scene (frozen field: no spatial correlation, no wake or shear),
//    evaluated for the turbulence statistics at a single reference height. World z = 0 is ground level (MIL-F-8785C h
//    is AGL, so origin_alt is not added for turbulence). The wind acts on rotors and body drag only, not on lift surfaces.
//  - Dryden (MIL-F-8785C, h < 1000 ft): h_ft = clamp(h / 0.3048, 10, 1000) (clamped outside the validity range of the
//    low-altitude form); sigma_w = 0.1 W20, sigma_u = sigma_v = sigma_w / k^0.4, k = 0.177 + 0.000823 h_ft,
//    L_w = h_ft ft, L_u = L_v = h_ft / k^1.2 ft. The filter time scale is L / V with V = max(|steady wind|, 2 m/s): the
//    turbulence is advected at the steady wind speed, a hovering vehicle gets a floor (documented, not from the spec).
//    u is along the horizontal steady-wind direction (world x when |wind_xy| < 1e-9), v to its left, w along world +z
//    (the spec has w positive down; the statistics are symmetric).
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {
namespace {
constexpr double kFt = 0.3048;      // m per ft (exact)
constexpr double kVFloor = 2.0;     // m/s, floor of the Dryden airspeed
constexpr double kJZero = 1e-9;     // rev/s: below this |n| the advance ratio is taken as 0
// ISA troposphere (ICAO Doc 7488): T0 = 288.15 K, p0 = 101325 Pa, L = 6.5 K/km, g0 = 9.80665, R = 287.05287 J/(kg K).
constexpr double kT0 = 288.15, kP0 = 101325.0, kLapse = 0.0065, kG0 = 9.80665, kR = 287.05287;
const double kIsaExp = kG0 / (kR * kLapse);  // 5.25588...

double isa_rho(double h) {
  h = std::clamp(h, -5000.0, 11000.0);  // troposphere only (< 11 km)
  const double t = kT0 - kLapse * h;
  return kP0 * std::pow(t / kT0, kIsaExp) / (kR * t);
}
double isa_p(double h) {  // same clamp and temperature as isa_rho; rho = p / (R T)
  h = std::clamp(h, -5000.0, 11000.0);
  return kP0 * std::pow((kT0 - kLapse * h) / kT0, kIsaExp);
}

double sgn(double x) { return x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0; }
double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Piecewise linear, clamped at both ends (x strictly increasing, validated by parse).
double interp(const std::vector<double>& x, const std::vector<double>& y, double v) {
  const std::size_t n = x.size();
  if (v <= x[0]) return y[0];
  if (v >= x[n - 1]) return y[n - 1];
  std::size_t k = 1;
  while (x[k] < v) ++k;
  return y[k - 1] + (v - x[k - 1]) / (x[k] - x[k - 1]) * (y[k] - y[k - 1]);
}

// Dryden shaping filters, exact ZOH discretisation. Driving noise n = sqrt(pi / dt) g (g ~ N(0,1) per tick) is a held
// band-limited white noise of one-sided unit PSD, for which the continuous filters
//   H_u = sigma sqrt(2T/pi) / (1 + T s),   H_v,w = sigma sqrt(T/pi) (1 + sqrt3 T s) / (1 + T s)^2,   T = L / V
// have output variance sigma^2. Stages a = n / (1 + Ts), b = a / (1 + Ts); H_v,w n = K ((1 - sqrt3) b + sqrt3 a).
// With e = exp(-dt/T), eps = dt/T and n held over the step the stages advance exactly:
//   a+ = e a + (1 - e) n,   b+ = e b + eps e a + (1 - e - eps e) n.
// Stationary variance error vs sigma^2: O((dt/T)^2) for u (1 - eps^2/12), O(dt/T) for v, w (dt/T <= ~1e-2 here).
void dryden_step(const EnvCfg& c, EnvState& s, double z_ref, double dt) {
  const double v_air = std::max(std::sqrt(dot3(s.wind, s.wind)), kVFloor);
  const double h_ft = std::clamp(z_ref / kFt, 10.0, 1000.0);
  const double k = 0.177 + 0.000823 * h_ft;
  const double sig_w = 0.1 * c.w20, sig_u = sig_w / std::pow(k, 0.4);
  const double len_w = h_ft * kFt, len_u = h_ft / std::pow(k, 1.2) * kFt;
  const double amp = std::sqrt(kPi / dt);
  const double noise[3] = {amp * s.rng.gauss(), amp * s.rng.gauss(), amp * s.rng.gauss()};
  double y[3];
  const double sigma[3] = {sig_u, sig_u, sig_w}, len[3] = {len_u, len_u, len_w};
  for (int i = 0; i < 3; ++i) {
    const double t = len[i] / v_air, e = std::exp(-dt / t), eps = dt / t;
    const double a_old = s.xa[i];
    s.xa[i] = e * a_old + (1 - e) * noise[i];
    if (i == 0) {
      y[i] = sigma[i] * std::sqrt(2 * t / kPi) * s.xa[i];
    } else {
      s.xb[i] = e * s.xb[i] + eps * e * a_old + (1 - e - eps * e) * noise[i];
      y[i] = sigma[i] * std::sqrt(t / kPi) * ((1 - std::sqrt(3.0)) * s.xb[i] + std::sqrt(3.0) * s.xa[i]);
    }
  }
  const double hm = std::hypot(s.wind[0], s.wind[1]);
  const double cs = hm > 1e-9 ? s.wind[0] / hm : 1.0, sn = hm > 1e-9 ? s.wind[1] / hm : 0.0;
  s.turb[0] = y[0] * cs - y[1] * sn;
  s.turb[1] = y[0] * sn + y[1] * cs;
  s.turb[2] = y[2];
}
}  // namespace

double rho_at(const Devices& dv, double z) {
  return isa_rho((dv.env.cfg.empty() ? 0.0 : dv.env.cfg[0].origin_alt) + z);
}

double pressure_at(const Devices& dv, double z) {
  return isa_p((dv.env.cfg.empty() ? 0.0 : dv.env.cfg[0].origin_alt) + z);
}

void wind_at(const Devices& dv, double out[3]) {
  for (int i = 0; i < 3; ++i) out[i] = dv.env.st.empty() ? 0.0 : dv.env.st[0].wind[i] + dv.env.st[0].turb[i];
}

void step_forces(const mjModel* m, mjData* d, Devices& dv) {
  const double dt = m->opt.timestep;
  // Environment first: rotors and drag see this tick's wind. W20 = 0: turbulence exactly 0 and no RNG draws.
  for (std::size_t e = 0; e < dv.env.cfg.size(); ++e) {
    const EnvCfg& c = dv.env.cfg[e];
    EnvState& s = dv.env.st[e];
    s.density = rho_at(dv, 0.0);
    if (c.w20 > 0) {
      double z_ref = c.bid >= 0 ? d->xpos[3 * c.bid + 2] : c.ref_height;
      if (c.bid >= 0 && !dv.wfield.cfg.empty()) z_ref -= ground_height(dv, d->xpos[3 * c.bid], d->xpos[3 * c.bid + 1]);  // AGL of the local ground
      dryden_step(c, s, z_ref, dt);
    }
  }
  step_wind_field(m, d, dv);  // gusts, WindField outputs (no-op without a wind_field device)
  const bool field = !dv.wfield.cfg.empty();  // wind per point (windfield.cpp) instead of the one uniform vector
  double wind[3];
  wind_at(dv, wind);
  step_rain(m, d, dv, wind);  // rain.cpp: LWC / fall speed of this tick, catch force + film weight on the film body
  const bool wet = rain_rate(dv) > 0;
  int n_wet = 0;
  double s_loss_sum = 0.0;

  // Rotors. All velocities in the world frame, rotor CoM (xipos); xfrc_applied acts at xipos.
  for (std::size_t r = 0; r < dv.rotor.cfg.size(); ++r) {
    const RotorCfg& c = dv.rotor.cfg[r];
    RotorState& s = dv.rotor.st[r];
    const double* ax = d->xaxis + 3 * c.jnt;  // unit thrust axis a
    const double w = d->qvel[c.dof], n = w / (2.0 * kPi), an = std::abs(n);
    double vel[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, c.bid, vel, 0);  // rot:lin, world axes, about the body CoM
    // A marine propeller (c.water) sees the sea: its density, its current instead of the wind, and the surface above the hub.
    double cur[3] = {0, 0, 0};
    if (c.water) sea_current(dv, cur);
    double loc[3] = {0, 0, 0};
    if (field && !c.water) wind_at_point(dv, d->xipos + 3 * c.bid, loc);
    const double* air = c.water ? cur : field ? loc : wind;
    const double vrel[3] = {vel[3] - air[0], vel[4] - air[1], vel[5] - air[2]};
    const double v_ax = dot3(vrel, ax);
    const double vperp[3] = {vrel[0] - v_ax * ax[0], vrel[1] - v_ax * ax[1], vrel[2] - v_ax * ax[2]};
    // s_dir = +1 when spinning in the design direction (pushes along +a), -1 reversed.
    const double s_dir = sgn(c.dir * w);
    const double adv = an > kJZero ? v_ax * s_dir / (an * c.diameter) : 0.0;
    const double* hub = d->xipos + 3 * c.bid;
    const double rho = c.water ? dv.sea.cfg[0].rho : rho_at(dv, hub[2]);
    const double imm = c.water ? std::clamp(0.5 + (sea_eta(dv, hub[0], hub[1]) - hub[2]) / c.diameter, 0.0, 1.0) : 1.0;

    double ge = 1.0;
    if (c.ground_effect && !c.water) {
      mjtByte groups[mjNGROUP] = {0};
      for (int g = 0; g < 6; ++g) groups[g] = (c.ray_groups >> g) & 1;
      // Downwash is opposite the thrust vector (thrust = s_dir * a): -s_dir * a. s_dir == 0 (no spin, zero thrust):
      // fall back to -a.
      const double sd = s_dir == 0.0 ? 1.0 : s_dir;
      const double down[3] = {-sd * ax[0], -sd * ax[1], -sd * ax[2]};
      const double dist = mj_ray(m, d, d->xipos + 3 * c.bid, down, groups, 1, c.bid, nullptr, nullptr);
      if (dist >= 0) {
        const double rad = 0.5 * c.diameter, q = rad / (4.0 * std::max(dist, 0.5 * rad));
        ge = 1.0 / (1.0 - q * q);
      }
    }
    const double d4 = c.diameter * c.diameter * c.diameter * c.diameter;
    double ft = 1.0, fq = 1.0;  // Reynolds factors (off: exactly 1, results bitwise unchanged)
    if (!c.re_x.empty()) {
      const double x = an * c.diameter * c.diameter;
      ft = interp(c.re_x, c.re_ft, x);
      fq = interp(c.re_x, c.re_fq, x);
    }
    const double healthy = 1.0 - s.damage;  // F3 prop damage: fraction of thrust and torque lost (0 = x * 1.0, bitwise unchanged)
    double thrust = s_dir * interp(c.j, c.ct, adv) * ft * rho * n * n * d4 * ge * healthy * imm * (s_dir < 0 ? c.rev_thrust : 1.0);
    double q_aero = interp(c.j, c.cq, adv) * fq * rho * n * n * d4 * c.diameter * healthy * imm;
    double* xf = d->xfrc_applied + 6 * c.bid;
    if (wet && !c.water) {  // rain (rain.cpp, docs/models.md "Rain"): wetted blades lose lift / gain drag, drops hit the disk
      const RainState& rs = dv.rain.st[0];
      const double x = rs.lwc / 23.0;
      const double kt = std::max(0.5, 1.0 - c.rain_kt * x);
      thrust *= kt;
      q_aero *= 1.0 + c.rain_kq * x;
      s_loss_sum += 1.0 - kt;
      ++n_wet;
      // Momentum of the drops stopped by the disk: u = drop velocity - hub velocity along the axis; half of the disk intercepts (estimate).
      const double u_ax = (wind[0] - vel[3]) * ax[0] + (wind[1] - vel[4]) * ax[1] + (wind[2] - rs.v_fall - vel[5]) * ax[2];
      const double f_drop = 0.5 * rs.lwc * 1e-3 * (0.25 * kPi * c.diameter * c.diameter) * std::abs(u_ax) * u_ax;
      for (int i = 0; i < 3; ++i) xf[i] += f_drop * ax[i];
    }
    for (int i = 0; i < 3; ++i) {
      xf[i] += thrust * ax[i] - c.k_h * std::abs(w) * vperp[i];
      xf[3 + i] += -sgn(w) * q_aero * ax[i];
    }
    if (c.stator_bid > 0) {  // EDF stator vanes hand the recovered share of the fan's swirl torque back to the housing
      double* xs = d->xfrc_applied + 6 * c.stator_bid;
      for (int i = 0; i < 3; ++i) xs[3 + i] += c.swirl_recovery * sgn(w) * q_aero * ax[i];
    }
    s.omega = w; s.thrust = thrust; s.torque = q_aero; s.advance_ratio = adv; s.ge_ratio = c.water ? imm : ge; s.rho = rho;  // damage is an input: keep
  }

  if (n_wet > 0) dv.rain.st[0].thrust_loss = s_loss_sum / n_wet;

  // Body drag.
  for (std::size_t b = 0; b < dv.drag.cfg.size(); ++b) {
    const DragCfg& c = dv.drag.cfg[b];
    double vel[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, c.bid, vel, 0);
    double loc[3] = {0, 0, 0};
    if (field) wind_at_point(dv, d->xipos + 3 * c.bid, loc);
    const double* air = field ? loc : wind;
    const double vrel[3] = {vel[3] - air[0], vel[4] - air[1], vel[5] - air[2]};
    const double k = -0.5 * rho_at(dv, d->xipos[3 * c.bid + 2]) * c.cd * c.area * std::sqrt(dot3(vrel, vrel));
    for (int i = 0; i < 3; ++i) {
      dv.drag.st[b].force[i] = k * vrel[i];
      d->xfrc_applied[6 * c.bid + i] += k * vrel[i];
    }
  }
  advance_wind_clock(dv, dt);
}

}  // namespace devs
