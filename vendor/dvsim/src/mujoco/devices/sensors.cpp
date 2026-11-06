// Sensors (S4): IMU, barometer, magnetometer, GNSS. Post-step hook of the device tick (after mj_step2).
//
// Timing. mj_step2 integrates qpos / qvel, but the derived data of the tick (xpos, site_xpos / site_xmat, cvel) still
// refer to the state s_k the tick started from, and so does the acceleration qacc it solved. Every sample is computed
// there, from derived data only, so position, velocity and acceleration of one sample belong to the same instant s_k.
// The sample is stored in the state; get() and the next tick's sensors-read phase see the held value. Fidelity: a
// sample describes the state at the START of the tick that produced it and becomes visible at its end: a latency of
// one tick (dt) on top of the configured GNSS latency, and the first sample appears after the first tick (before it
// the outputs read 0; GNSS reads its origin). Between samples the value is held (zero-order hold, no interpolation).
// Acceleration needs mj_rnePostConstraint (MuJoCo computes it lazily, only for models with acceleration sensors, and
// then before integrating): when it has not run we call it ourselves, with qvel temporarily set back to s_k (it
// multiplies cdof_dot by qvel, and qvel is s_{k+1} by now). Euler (default): qacc is the continuous-time acceleration
// at s_k. Implicit integrators overwrite qacc of free joints with the one-step finite difference of qvel (MuJoCo
// midpoint integration), which the IMU then sees instead.
//
// Fidelity notes (all sensors): ideal mounting: the sensor frame IS the site frame, no mount compliance, vibration or
// lever-arm dynamics beyond the rigid-body kinematics of the site point; no cross-axis sensitivity or misalignment;
// no temperature dependence; no sampling jitter, no bus / interface delay; white noise is Gaussian, drawn per sample
// from the device's own RNG (draw order: IMU gyro x y z, then accel x y z, each axis: bias walk, then noise; a zero
// parameter draws nothing), so equal seeds give bitwise equal streams.
//  - IMU: bias = constant + random walk (no scale-factor or temperature drift, no g-sensitivity of the gyro). Order
//    per axis: scale / bias / noise, then quantisation (round to nearest LSB, ties away from zero), then saturation.
//    A site on a static body (welded to the world) reads +R^T (-gravity): MuJoCo's own acceleration of a static body
//    is 0 (it omits gravity), but a bench IMU at rest reads 1 g.
//  - Barometer: ISA pressure of the site altitude (standard day, no weather, no wind / dynamic pressure, no
//    temperature drift); noise white, quantisation to `resolution`.
//  - Magnetometer: one uniform world-frame field (no declination map, no soft / hard iron beyond the constant bias,
//    no vehicle field); the field rotates with the site: B_site = R^T B_world.
//  - GNSS: world x = East, y = North, z = Up, origin (lat0, lon0, alt0). Flat earth: lat = lat0 + N / M, lon = lon0 +
//    E / (R_n cos lat0), alt = alt0 + U with the WGS84 meridian radius M and prime-vertical radius R_n at lat0:
//    accurate to ~ (distance / R)^2, i.e. cm over ~10 km; the altitude adds no height scaling to M, R_n and no geoid.
//    Position errors: white Gaussian by default; with corr_time_s > 0 each of north / east / up is a first-order
//    Gauss-Markov process (variance share corr_frac, time constant corr_time_s) plus white noise (the rest), total
//    sigma unchanged (real receivers are strongly time-correlated: ionosphere, orbit, multipath). Velocity errors
//    are always white. Velocity is the site point's world velocity as NED. Latency: output = the sample taken
//    round(latency_s rate_hz) samples ago, from a fixed ring of kGnssRing samples (delay <= 63, checked at load);
//    until that many samples exist the output stays at the origin / zero.
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {
namespace {
constexpr double kDeg = 180.0 / kPi;
// WGS84: semi-major axis a, flattening f -> e^2 = f (2 - f).
constexpr double kWgsA = 6378137.0, kWgsF = 1.0 / 298.257223563, kWgsE2 = kWgsF * (2.0 - kWgsF);

double quantise(double v, double lsb) { return lsb > 0 ? std::round(v / lsb) * lsb : v; }

// One triad: truth -> out, advancing the bias walk. See schema InertialErr for the equations.
void measure(const InertialErrCfg& e, double rate, const double truth[3], double bias[3], Rng& rng, double out[3]) {
  const double sigma = e.noise_density * std::sqrt(rate / 2.0), walk = e.rw_density * std::sqrt(1.0 / rate);
  const double lsb = e.bits > 0 ? 2.0 * e.full_scale / std::ldexp(1.0, e.bits) : 0.0;
  for (int i = 0; i < 3; ++i) {
    if (walk > 0) bias[i] += walk * rng.gauss();
    double v = (1.0 + e.scale[i]) * truth[i] + bias[i];
    if (sigma > 0) v += sigma * rng.gauss();
    v = quantise(v, lsb);
    out[i] = e.full_scale > 0 ? std::clamp(v, -e.full_scale, e.full_scale) : v;
  }
}

// Specific force at the site, site frame: mj_rnePostConstraint with the qvel of s_k (see the header comment).
void site_accel(const mjModel* m, mjData* d, Devices& dv, int sid, double out[3]) {
  const int body = m->site_bodyid[sid];
  if (m->body_weldid[body] == 0) {  // static: MuJoCo returns 0 (no gravity); at rest the specific force is -gravity
    double g[3] = {0, 0, 0};
    if (!(m->opt.disableflags & mjDSBL_GRAVITY)) mju_scl3(g, m->opt.gravity, -1.0);
    mju_mulMatTVec3(out, d->site_xmat + 9 * sid, g);
    return;
  }
  if (!d->flg_rnepost) {
    dv.qvel_k1.assign(d->qvel, d->qvel + m->nv);
    mju_copy(d->qvel, dv.qvel_k.data(), m->nv);
    mj_rnePostConstraint(m, d);
    mju_copy(d->qvel, dv.qvel_k1.data(), m->nv);
  }
  double a[6];
  mj_objectAcceleration(m, d, mjOBJ_SITE, sid, a, 1);
  mju_copy3(out, a + 3);
}
}  // namespace

void sample_sensors(const mjModel* m, mjData* d, Devices& dv) {
  for (std::size_t i = 0; i < dv.imu.cfg.size(); ++i) {
    const ImuCfg& c = dv.imu.cfg[i];
    ImuState& s = dv.imu.st[i];
    if (s.ticks++ % static_cast<std::uint64_t>(c.period) != 0) continue;
    double v[6], truth[3];
    mj_objectVelocity(m, d, mjOBJ_SITE, c.sid, v, 1);  // rot:lin in the site frame
    measure(c.gyro, c.rate_hz, v, s.gyro_bias, s.rng, s.gyro);
    site_accel(m, d, dv, c.sid, truth);
    measure(c.accel, c.rate_hz, truth, s.accel_bias, s.rng, s.accel);
  }
  for (std::size_t i = 0; i < dv.baro.cfg.size(); ++i) {
    const BaroCfg& c = dv.baro.cfg[i];
    BaroState& s = dv.baro.st[i];
    if (s.ticks++ % static_cast<std::uint64_t>(c.period) != 0) continue;
    s.bias_extra += s.drift_rate * m->opt.timestep * c.period;  // F3 baro drift: a slow bias ramp (Pa/s); 0 = none
    double p = pressure_at(dv, d->site_xpos[3 * c.sid + 2]) + c.bias + s.bias_extra;
    if (c.noise_std > 0) p += c.noise_std * s.rng.gauss();
    s.pressure = quantise(p, c.resolution);
  }
  for (std::size_t i = 0; i < dv.mag.cfg.size(); ++i) {
    const MagCfg& c = dv.mag.cfg[i];
    MagState& s = dv.mag.st[i];
    if (s.ticks++ % static_cast<std::uint64_t>(c.period) != 0) continue;
    mju_mulMatTVec3(s.field, d->site_xmat + 9 * c.sid, c.field);
    for (int k = 0; k < 3; ++k) {
      s.field[k] += c.bias[k];
      if (c.noise_std > 0) s.field[k] += c.noise_std * s.rng.gauss();
    }
  }
  for (std::size_t i = 0; i < dv.gnss.cfg.size(); ++i) {
    const GnssCfg& c = dv.gnss.cfg[i];
    GnssState& s = dv.gnss.st[i];
    if (s.ticks++ % static_cast<std::uint64_t>(c.period) != 0) continue;
    const double sin_lat = std::sin(c.lat0 / kDeg), w = 1.0 - kWgsE2 * sin_lat * sin_lat;
    const double r_mer = kWgsA * (1.0 - kWgsE2) / (w * std::sqrt(w)), r_pv = kWgsA / std::sqrt(w);
    const double* p = d->site_xpos + 3 * c.sid;
    double v[6];
    mj_objectVelocity(m, d, mjOBJ_SITE, c.sid, v, 0);  // world axes, of the site point
    double n[6] = {0, 0, 0, 0, 0, 0};                  // north, east, up, vn, ve, vd errors
    const bool gm = c.corr_time_s > 0 && c.corr_frac > 0;  // else white, bitwise the pre-Gauss-Markov stream
    const double phi = gm ? std::exp(-1.0 / (c.rate_hz * c.corr_time_s)) : 0.0;
    for (int k = 0; k < 6; ++k) {
      const double sd = k < 2 ? c.horiz_std : k == 2 ? c.vert_std : c.vel_std;
      if (sd <= 0) continue;
      if (gm && k < 3) {  // total variance sd^2 = Gauss-Markov (corr_frac) + white (the rest); GM starts stationary
        const double sc = sd * std::sqrt(c.corr_frac), sw = sd * std::sqrt(1.0 - c.corr_frac);
        s.gm[k] = s.samples == 0 ? sc * s.rng.gauss() : phi * s.gm[k] + std::sqrt(1.0 - phi * phi) * sc * s.rng.gauss();
        n[k] = s.gm[k] + (sw > 0 ? sw * s.rng.gauss() : 0.0);
      } else {
        n[k] = sd * s.rng.gauss();
      }
    }
    double* slot = s.ring[s.samples % kGnssRing];
    n[0] += s.jump[1]; n[1] += s.jump[0]; n[2] += s.jump[2];  // F3 gnss_jump: ENU step -> north, east, up error
    slot[0] = c.lat0 + (p[1] + n[0]) / r_mer * kDeg;
    slot[1] = c.lon0 + (p[0] + n[1]) / (r_pv * std::cos(c.lat0 / kDeg)) * kDeg;
    slot[2] = c.alt0 + p[2] + n[2];
    slot[3] = v[4] + n[3];
    slot[4] = v[3] + n[4];
    slot[5] = -v[5] + n[5];
    if (s.samples >= static_cast<std::uint64_t>(c.delay) && !(s.fix_lost > 0.5)) {  // fix lost: the output holds
      const double* o = s.ring[(s.samples - static_cast<std::uint64_t>(c.delay)) % kGnssRing];
      s.lat = o[0];
      s.lon = o[1];
      s.alt = o[2];
      std::copy(o + 3, o + 6, s.vel_ned);
    }
    ++s.samples;
  }
}

}  // namespace devs
