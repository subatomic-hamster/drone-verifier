#pragma once
// Pure conversions between dvsim (world ENU, z up; body FLU: x forward, y left, z up) and PX4 / MAVLink (world NED; body
// FRD: x forward, y right, z down), plus the unit scaling of the HIL messages. No I/O, header-only (tests/cpp/test_px4.cpp).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace px4f {

using V3 = std::array<double, 3>;
using Q4 = std::array<double, 4>;  // w, x, y, z (Hamilton, same order as MuJoCo and dvsim `quat`)

// ---- frames ----
// Body vector FLU -> FRD: (x, y, z) -> (x, -y, -z). Applies to accel, gyro, mag and body rates.
inline V3 flu_to_frd(const V3& v) { return {v[0], -v[1], -v[2]}; }
// World vector ENU (E, N, U) -> NED (N, E, -D = -U): (E, N, U) -> (N, E, -U). Applies to position, velocity.
inline V3 enu_to_ned(const V3& v) { return {v[1], v[0], -v[2]}; }

inline Q4 qmul(const Q4& a, const Q4& b) {
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
inline V3 qrot(const Q4& q, const V3& v) {  // q v q^-1 for a unit q
  const V3 u{q[1], q[2], q[3]};
  const V3 t{2 * (u[1] * v[2] - u[2] * v[1]), 2 * (u[2] * v[0] - u[0] * v[2]), 2 * (u[0] * v[1] - u[1] * v[0])};
  return {v[0] + q[0] * t[0] + (u[1] * t[2] - u[2] * t[1]), v[1] + q[0] * t[1] + (u[2] * t[0] - u[0] * t[2]),
          v[2] + q[0] * t[2] + (u[0] * t[1] - u[1] * t[0])};
}
// Attitude: q_ned<-frd = q_ned<-enu * q_enu<-flu * q_flu<-frd. q_ned<-enu is the proper rotation that swaps E and N and
// flips U (180 deg about (1,1,0)/sqrt2); q_flu<-frd is 180 deg about body x. Result has w >= 0.
// Identity dvsim attitude (nose East) -> yaw +90 deg; nose North (dvsim yaw +90 deg) -> identity.
inline Q4 q_ned_frd(const Q4& q_enu_flu) {
  constexpr double s = 0.70710678118654752440;
  Q4 q = qmul(qmul(Q4{0, s, s, 0}, q_enu_flu), Q4{0, 1, 0, 0});
  if (q[0] < 0)
    for (double& c : q) c = -c;
  return q;
}

// ---- units ----
inline double tesla_to_gauss(double t) { return t * 1e4; }
inline double pa_to_hpa(double p) { return p / 100.0; }

// ISA troposphere, the constants of the device model (docs/models.md "ISA").
constexpr double kG0 = 9.80665, kRair = 287.05287, kLapse = 0.0065, kT0 = 288.15, kP0 = 101325.0;
constexpr double kExp = kRair * kLapse / kG0;  // T / T0 = (p / p0)^kExp
inline double isa_pressure(double h) { return kP0 * std::pow(1.0 - kLapse * h / kT0, 1.0 / kExp); }
inline double pressure_alt(double p_pa) { return kT0 / kLapse * (1.0 - std::pow(p_pa / kP0, kExp)); }  // inverse of isa_pressure
inline double isa_temp_c(double p_pa) { return kT0 * std::pow(p_pa / kP0, kExp) - 273.15; }  // ISA temperature at that pressure

inline std::int32_t deg_e7(double deg) { return static_cast<std::int32_t>(std::llround(deg * 1e7)); }
inline std::int32_t mm(double m) { return static_cast<std::int32_t>(std::llround(m * 1e3)); }
inline std::int16_t cm_s_i16(double ms) {  // m/s -> cm/s, saturating
  return static_cast<std::int16_t>(std::clamp(std::llround(ms * 100.0), -32768LL, 32767LL));
}
inline std::uint16_t cm_u16(double m) {  // m (or m/s) -> cm, saturating, never negative
  return static_cast<std::uint16_t>(std::clamp(std::llround(m * 100.0), 0LL, 65535LL));
}
// Course over ground, centidegrees in [0, 36000); 65535 (unknown) when the ground speed is ~ 0.
inline std::uint16_t cog_cdeg(double vn, double ve) {
  if (std::hypot(vn, ve) < 0.01) return 65535;
  double d = std::atan2(ve, vn) * 180.0 / 3.14159265358979323846;
  if (d < 0) d += 360.0;
  return static_cast<std::uint16_t>(std::llround(d * 100.0) % 36000);
}

// HIL_SENSOR fields_updated: PX4 tests whole groups (all three bits).
constexpr std::uint32_t kAccel = 7, kGyro = 56, kMag = 448, kBaro = 6656;  // baro = abs_pressure + pressure_alt + temperature
inline std::uint32_t fields_updated(bool imu, bool baro, bool mag) {
  return (imu ? kAccel | kGyro : 0u) | (baro ? kBaro : 0u) | (mag ? kMag : 0u);
}

// True iff a sensor with `period` ticks per sample took a sample in ticks [tick0, tick0 + n): a sample is taken when
// (tick counter) % period == 0 and the counter equals the world tick the tick starts from.
inline bool sampled(std::uint64_t tick0, std::uint64_t n, std::uint64_t period) {
  return (tick0 + period - 1) / period * period < tick0 + n;
}

}  // namespace px4f
