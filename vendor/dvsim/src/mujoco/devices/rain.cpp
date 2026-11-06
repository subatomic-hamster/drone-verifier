// Rain (schema Rain): one precipitation field for the whole scene. Phase 5 of the device tick, right after the environment: the
// rotors (aero.cpp) and the rangefinders (rangefinder.cpp) read this state. Model and sources: docs/models.md "Rain".
//
//   LWC   = 0.072 R^0.88 g/m^3                  liquid water content of rain at rate R mm/h (the briefing's fit to the Marshall-Palmer
//                                               N(D) = 8000 exp(-4.1 R^-0.21 D) m^-3 mm^-1; the pure MP integral is 0.089 R^0.84)
//   v_fall = R * 2.7778e-4 kg/(m^2 s per mm/h) / (LWC * 1e-3)   mass-weighted fall speed: the water flux through a horizontal plane equals R
//                                               exactly (3.9 m/s at 1 mm/h, 5.7 at 25, 6.7 at 100; Atlas-Ulbrich 3.78 D_m^0.67 at the
//                                               mass-weighted D_m = 0.976 R^0.21 mm gives 3.8 / 5.9 / 7.4: the same within ~10 %)
//   Drops fall at (wind_xy, wind_z - v_fall): they drift with the steady wind + turbulence of the environment device.
//   Film body: u = drop velocity - body CoM velocity; catch force F = LWC A |u| u (inelastic impact: the drops lose all relative
//   momentum; splash and rebound, up to a factor 2, are ignored) at the CoM; film mass dm/dt = collect_eff LWC A |u| - m / film_tau,
//   capped at rho_w film_depth wet_area; its weight m g acts at the CoM (the water adds no inertia in this model: <~2 % error on the
//   mass at the cap of a small drone, and none on the CoM).
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {
namespace {
constexpr double kFluxPerMmH = 1.0 / 3600.0;  // 1 mm/h of rain = 1 kg/(m^2 h) = 2.7778e-4 kg/(m^2 s)
constexpr double kG = 9.80665, kRhoW = 1000.0;
}  // namespace

double rain_rate(const Devices& dv) { return dv.rain.st.empty() ? 0.0 : dv.rain.st[0].rate; }

void step_rain(const mjModel* m, mjData* d, Devices& dv, const double wind[3]) {
  if (dv.rain.cfg.empty()) return;
  const RainCfg& c = dv.rain.cfg[0];
  RainState& s = dv.rain.st[0];
  const double r = s.rate;
  s.lwc = r > 0 ? 0.072 * std::pow(r, 0.88) : 0.0;
  s.v_fall = r > 0 ? r * kFluxPerMmH / (s.lwc * 1e-3) : 0.0;
  s.thrust_loss = 0.0;  // accumulated by the rotor loop of this tick
  for (double& f : s.force) f = 0.0;
  if (c.bid < 0 || (r <= 0 && s.film_mass <= 0)) return;
  double catch_rate = 0.0;
  if (r > 0) {
    double vel[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, c.bid, vel, 0);  // rot:lin, world axes, about the CoM
    const double u[3] = {wind[0] - vel[3], wind[1] - vel[4], wind[2] - s.v_fall - vel[5]};
    const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    const double k = s.lwc * 1e-3 * c.catch_area * un;  // kg/s of water meeting the body
    for (int i = 0; i < 3; ++i) s.force[i] = k * u[i];
    catch_rate = c.collect_eff * k;
  }
  const double dt = m->opt.timestep, cap = kRhoW * c.film_depth * c.wet_area;
  s.film_mass = std::clamp(s.film_mass + dt * (catch_rate - s.film_mass / c.film_tau), 0.0, cap);
  s.force[2] -= s.film_mass * kG;
  for (int i = 0; i < 3; ++i) d->xfrc_applied[6 * c.bid + i] += s.force[i];
}

}  // namespace devs
