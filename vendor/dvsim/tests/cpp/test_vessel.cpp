// Sea + Vessel (buoyancy, Fossen damping and added mass, waves, current) and the marine propeller option of the rotor
// (src/mujoco/devices/vessel.cpp, aero.cpp). Expectations are closed forms (hydrostatics, linear and quadratic damping ODEs,
// the single-DOF wave response), written independently of the implementation. Boat models are generated per test into a temp file.
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "mujoco/mujoco_solver.h"

namespace {
const std::string kDir = std::string(DVSIM_TEST_DATA_DIR) + "/devices/";
constexpr double kPi = 3.14159265358979323846;
constexpr double kRho = 998.2, kG = 9.81;

const MujocoSolver& ms(const Solver& s) { return static_cast<const MujocoSolver&>(s); }
std::vector<double> getv(const Solver& s, const std::string& prim, const std::string& attr) {
  std::vector<double> o;
  const Result r = s.get(prim, attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << prim << "." << attr << ": " << r.message;
  return o;
}
double get(const Solver& s, const std::string& prim, const std::string& attr) {
  const auto o = getv(s, prim, attr);
  return o.empty() ? NAN : o[0];
}
void put_ok(Solver& s, const std::string& prim, const std::string& attr, std::vector<double> v) {
  const Result r = s.set(prim, attr, "", v);
  EXPECT_EQ(r.code, Result::ok) << r.message;
}

// A box hull (1.0 x 0.4 x 0.2 m) floating in the sea, generated MJCF. Defaults: 20 kg, at the derived draft, hydrodynamics as the
// heave / roll tests need them. thrust_n > 0 adds a flat-curve solid motor (a constant force source) pushing along the body +x from
// y = thrust_y.
struct Boat {
  double mass = 20.0, roll_deg = 0.0, thrust_n = 0.0, thrust_y = 0.0, current_x = 0.0, wave_a = 0.0, wave_t = 4.0;
  double aft_kg = 0.0;  // a welded child 0.4 m aft and 0.05 m up: shifts the CoM and gives the inertia a product (Ixz)
  std::string added = "[5,5,4,0.1,0.5,1]", lin = "[4,4,200,1.0,2,4]", quad = "[6,6,0,0,0,0]", extra_sea, fins;  // fins: the JSON array body
  bool no_sea = false, heading_frame = false;  // heading_frame: the vessel's translational damping and added mass in the yaw-only frame
};
std::string boat_xml(const Boat& b) {
  const double total = b.mass + b.aft_kg + (b.thrust_n > 0 ? 0.1 : 0.0);
  const double z0 = 0.1 - total / (kRho * 0.4);
  const double q0 = std::cos(0.5 * b.roll_deg * kPi / 180), q1 = std::sin(0.5 * b.roll_deg * kPi / 180);
  std::ostringstream x;
  x << "<mujoco model=\"boat\"><option timestep=\"0.001\"/><worldbody>"
    << "<geom name=\"seabed\" type=\"plane\" pos=\"0 0 -20\" size=\"100 100 1\" contype=\"0\" conaffinity=\"0\"/>"
    << "<body name=\"hull\" pos=\"0 0 " << z0 << "\" quat=\"" << q0 << " " << q1 << " 0 0\"><freejoint/>"
    << "<geom type=\"box\" size=\"0.5 0.2 0.1\" mass=\"" << b.mass << "\" contype=\"0\" conaffinity=\"0\"/>";
  if (b.aft_kg > 0)
    x << "<body name=\"aft\" pos=\"-0.4 0 0.05\"><geom type=\"box\" size=\"0.05 0.05 0.05\" mass=\"" << b.aft_kg
      << "\" contype=\"0\" conaffinity=\"0\"/></body>";
  if (b.thrust_n > 0)
    x << "<body name=\"motor\" pos=\"0 " << b.thrust_y << " 0\"><geom type=\"box\" size=\"0.02 0.02 0.02\" mass=\"0.1\" contype=\"0\" "
      << "conaffinity=\"0\"/><site name=\"nozzle\" euler=\"0 90 0\"/></body>";
  x << "</body></worldbody><custom><text name=\"dvsim.devices\" data='{";
  if (!b.no_sea) {
    x << "\"seas\": [{\"name\": \"sea\", \"rho\": " << kRho << ", \"current\": [" << b.current_x << ", 0, 0]";
    if (b.wave_a > 0) x << ", \"waves\": [{\"amplitude\": " << b.wave_a << ", \"period\": " << b.wave_t << "}]";
    x << b.extra_sea << "}], ";
  }
  x << "\"vessels\": [{\"name\": \"v\", \"body\": \"hull\", \"hulls\": [{\"shape\": \"box\", \"pos\": [0,0,0], \"size\": [1.0, 0.4, 0.2], "
    << "\"n\": [20, 10, 10]}], \"added_mass\": " << b.added << ", \"damp_lin\": " << b.lin << ", \"damp_quad\": " << b.quad
    << (b.fins.empty() ? "" : ", \"fins\": [" + b.fins + "]") << (b.heading_frame ? ", \"heading_frame\": true" : "") << "}]";
  if (b.thrust_n > 0)
    x << ", \"solid_motors\": [{\"name\": \"fan\", \"body\": \"motor\", \"site\": \"nozzle\", \"time\": [0.001, 200, 200.001], "
      << "\"thrust\": [" << b.thrust_n << ", " << b.thrust_n << ", 0], \"propellant_mass\": 1e-6, \"total_mass\": 0.1, "
      << "\"diameter\": 0.02, \"length\": 0.04}]";
  x << "}'/></custom></mujoco>";
  return x.str();
}
std::unique_ptr<Solver> make(const Boat& b) {
  static int n = 0;
  const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_boat_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".xml")).string();
  std::ofstream(path) << boat_xml(b);
  auto s = load_mujoco(path);
  std::remove(path.c_str());
  return s;
}
double roll_of(const mjData& d) { return 2.0 * std::atan2(d.qpos[4], d.qpos[3]); }  // pure x rotation
}  // namespace

// Hydrostatics: a 20 kg box released at the waterline sinks to the draft m / (rho A) = 50.09 mm, displaces its own mass, and stays.
TEST(vessel, floats_at_the_derived_draft_and_settles) {
  Boat b;
  auto s = make(b);
  put_ok(*s, "/devices/sea", "wave_scale", {0});
  const double draft = b.mass / (kRho * 0.4), want_z = 0.1 - draft;
  // restart from the waterline (hull centre at 0): the factory pose is already at equilibrium, so push it down by running from there
  ASSERT_EQ(s->step(10000), 10000);
  const mjData& d = ms(*s).data();
  EXPECT_NEAR(d.qpos[2], want_z, 1e-6);
  EXPECT_NEAR(get(*s, "/devices/v", "submerged"), b.mass / kRho, 1e-6 * b.mass / kRho);
  EXPECT_NEAR(getv(*s, "/devices/v", "buoyancy")[2], b.mass * kG, 1e-4 * b.mass * kG);
  EXPECT_NEAR(get(*s, "/devices/v", "wetted"), 1.0, 1e-6);
  EXPECT_NEAR(get(*s, "/devices/v", "eta"), 0.0, 1e-12);
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(d.qvel[i], 0.0, 1e-6) << "dof " << i;
  // a dropped hull (1 m above) settles to the same draft: energy leaves through the heave damping
  Boat hi = b;
  auto t = make(hi);
  put_ok(*t, "/devices/sea", "wave_scale", {0});
  ASSERT_EQ(t->step(20), 20);
  EXPECT_NEAR(ms(*t).data().qpos[2], want_z, 1e-3);
  EXPECT_FALSE(s->diverged());
}

// Free decay in roll: T = 2 pi sqrt((Ixx + K_pdot) / (Delta g GM)), GM = KB + BM - KG from the box geometry.
TEST(vessel, roll_free_decay_period_matches_metacentric_theory) {
  Boat b;
  b.roll_deg = 5.0;
  auto s = make(b);
  put_ok(*s, "/devices/sea", "wave_scale", {0});
  const double draft = b.mass / (kRho * 0.4);
  const double kb = draft / 2, bm = 0.4 * 0.4 / (12 * draft), kg = 0.1, gm = kb + bm - kg;
  const double ixx = b.mass / 12 * (0.4 * 0.4 + 0.2 * 0.2), added = 0.1;  // added roll inertia: wetted fraction 1
  const double period = 2 * kPi * std::sqrt((ixx + added) / (b.mass * kG * gm));
  std::vector<double> up;  // times of upward zero crossings of the roll angle
  double prev = roll_of(ms(*s).data());
  for (int i = 0; i < 8000 && up.size() < 7; ++i) {
    ASSERT_EQ(s->step(1), 1);
    const double r = roll_of(ms(*s).data());
    if (prev < 0 && r >= 0) up.push_back(ms(*s).data().time - 0.001 * r / (r - prev));
    prev = r;
  }
  ASSERT_GE(up.size(), 7u);
  const double measured = (up[6] - up[1]) / 5.0;
  EXPECT_NEAR(measured, period, 0.03 * period) << "GM " << gm << " m, predicted " << period << " s";
  // the oscillation decays (damped): the amplitude after 6 cycles is well below the start
  EXPECT_LT(std::abs(roll_of(ms(*s).data())), 0.05);
}

// Heave in a regular wave: the single-DOF response of the hull to the surface, RAO = k / |k - w^2 M + i w c|, k = rho g A_wp.
TEST(vessel, heave_response_to_a_regular_wave) {
  Boat b;
  b.wave_a = 0.1;
  b.wave_t = 4.0;
  auto s = make(b);
  const double w = 2 * kPi / b.wave_t, k = kRho * kG * 0.4, mm = b.mass + 4.0, c = 200.0;
  const double rao = k / std::hypot(k - w * w * mm, w * c);
  ASSERT_EQ(s->step(24000), 24000);  // transient: e^(-zeta wn t) with zeta 0.33, wn 12.8: gone after a few seconds
  double lo = 1e9, hi = -1e9;
  for (int i = 0; i < 8000; ++i) {  // two wave periods
    ASSERT_EQ(s->step(1), 1);
    const double z = ms(*s).data().qpos[2];
    lo = std::min(lo, z);
    hi = std::max(hi, z);
  }
  EXPECT_NEAR(0.5 * (hi - lo), rao * 0.1, 0.03 * rao * 0.1) << "RAO " << rao;
  EXPECT_NEAR(get(*s, "/devices/sea", "time"), 32.0, 1e-9);
  // the surface function the device exposes is the one the physics used: eta at the hull origin
  const double t = get(*s, "/devices/sea", "time") - 0.001;  // the tick that produced the vessel output started one dt earlier
  const auto wt = getv(*s, "/devices/sea", "waves");  // a, k, w, theta, phi
  const double x = ms(*s).data().qpos[0], y = ms(*s).data().qpos[1];
  EXPECT_NEAR(get(*s, "/devices/v", "eta"), wt[0] * std::cos(wt[1] * (x * std::cos(wt[3]) + y * std::sin(wt[3])) - wt[2] * t + wt[4]), 2e-3);
}

// A passive hull in a regular sea must not drift: nothing in the model pushes it (no orbital velocities, no current, no thrust; buoyancy is
// vertical). With the damping and added mass in the BODY frame the heave force of a pitched hull is tilted by the pitch angle; heave
// velocity and pitch are correlated in a following sea, so the horizontal part does not average out and the hull propels itself
// (F_x ~ 1/2 d_heave a^2 omega k). In the heading frame (Vessel.heading_frame) the heave force is vertical and the mean drift is zero.
TEST(vessel, passive_hull_in_waves_drifts_only_with_body_frame_heave_damping) {
  double drift[2];
  for (const bool hf : {false, true}) {
    Boat b;
    b.wave_a = 0.12;
    b.wave_t = 2.0;  // 6 m wavelength on a 1 m hull: a pitch amplitude near the wave slope a k = 0.12 rad
    b.heading_frame = hf;
    auto s = make(b);
    ASSERT_EQ(s->step(30000), 30000);  // transients (surge damping time constant ~ 6 s) are gone
    const double x0 = ms(*s).data().qpos[0], t0 = ms(*s).data().time;
    ASSERT_EQ(s->step(60000), 60000);  // 30 wave periods
    drift[hf] = (ms(*s).data().qpos[0] - x0) / (ms(*s).data().time - t0);
    EXPECT_NEAR(ms(*s).data().qpos[1], 0.0, 1e-3) << "no sideways drift";
    EXPECT_FALSE(s->diverged());
  }
  EXPECT_GT(std::abs(drift[0]), 0.02) << "the artefact of the body frame: " << drift[0] << " m/s";
  EXPECT_LT(std::abs(drift[1]), 0.002) << "heading frame: " << drift[1] << " m/s";
}

// Added mass against an inertia with a product: heave + pitch added mass on a hull whose welded aft mass puts Ixz into the body inertia
// (the solve is (I + W A) y = z; (I + A W) gave a heave-pitch oscillation that grew without bound). Released at the waterline it must
// settle, not oscillate: the last 2 s of heave stay within 0.1 mm.
TEST(vessel, added_mass_with_product_of_inertia_stays_stable) {
  Boat b;
  b.aft_kg = 6.0;
  b.added = "[5,5,60,0.1,4,1]";
  b.lin = "[4,4,200,1.0,30,4]";
  auto s = make(b);
  put_ok(*s, "/devices/sea", "wave_scale", {0});
  ASSERT_EQ(s->step(10000), 10000);
  double lo = 1e9, hi = -1e9;
  for (int i = 0; i < 2000; ++i) {
    ASSERT_EQ(s->step(1), 1);
    lo = std::min(lo, ms(*s).data().qpos[2]);
    hi = std::max(hi, ms(*s).data().qpos[2]);
  }
  EXPECT_LT(hi - lo, 1e-4);
  EXPECT_FALSE(s->diverged());
}

// Surge under a constant thrust T against M du/dt = T - c1 u - c2 |u| u, M = m + added mass (the added mass is exact, also when it is
// 3x the hull: the lagged form would diverge), and the terminal speed is the root of c2 u^2 + c1 u = T (here exactly 1 m/s).
TEST(vessel, surge_speed_matches_the_damping_model_and_added_mass_is_exact) {
  for (const double au : {5.0, 60.0}) {
    Boat b;
    b.thrust_n = 10.0;
    b.added = "[" + std::to_string(au) + ",5,4,0.1,0.5,1]";
    auto s = make(b);
    put_ok(*s, "/devices/sea", "wave_scale", {0});
    put_ok(*s, "/devices/fan", "ignite", {1});
    const double mm = b.mass + 0.1 + au, c1 = 4, c2 = 6, T = 10;
    double u = 0, t = 0;
    const double dt = 1e-3;
    auto f = [&](double v) { return (T - c1 * v - c2 * std::abs(v) * v) / mm; };
    for (const double stop : {0.05, 2.0, 6.0, 15.0, 40.0}) {
      const int n = static_cast<int>(std::lround((stop - t) / dt));
      for (int i = 0; i < n; ++i) {  // RK4 of the scalar ODE at the sim's step
        const double k1 = f(u), k2 = f(u + 0.5 * dt * k1), k3 = f(u + 0.5 * dt * k2), k4 = f(u + dt * k3);
        u += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
      }
      t = stop;
      ASSERT_EQ(s->step(static_cast<int>(std::lround(stop * 1000)) - static_cast<int>(ms(*s).data().time * 1000 + 0.5)), static_cast<int>(std::lround(stop * 1000)) - static_cast<int>(ms(*s).data().time * 1000 + 0.5));
      const double got = ms(*s).data().qvel[0];  // world x = body x (no yaw)
      EXPECT_NEAR(got, u, 0.01 * std::max(u, 0.01)) << "added mass " << au << " at t = " << stop;
    }
    EXPECT_NEAR(ms(*s).data().qvel[0], 1.0, 0.01);  // terminal speed
    EXPECT_NEAR(ms(*s).data().qvel[1], 0.0, 1e-6);   // straight line
    EXPECT_FALSE(s->diverged());
  }
}

// Yaw: an off-centre thrust gives a constant torque tau = T y_off about the CoM; with decoupled yaw damping the rate follows
// (Izz + N_rdot) dr/dt = tau - N_r r (surge-sway terms leave yaw untouched when added_mass_x = added_mass_y).
TEST(vessel, yaw_rate_from_an_offset_thrust) {
  Boat b;
  b.thrust_n = 10.0;
  b.thrust_y = -0.2;  // torque_z = -y F_x > 0
  b.added = "[5,5,4,0.1,0.5,1]";
  b.lin = "[4,4,200,1.0,2,4]";
  b.quad = "[0,0,0,0,0,0]";
  auto s = make(b);
  put_ok(*s, "/devices/sea", "wave_scale", {0});
  put_ok(*s, "/devices/fan", "ignite", {1});
  const double tau = 10.0 * 0.2, nr = 4.0, mz = (b.mass / 12 * (1.0 + 0.16)) + 1.0 + 0.1 * 0.2 * 0.2;  // box Izz + added + motor
  ASSERT_EQ(s->step(10000), 10000);
  const double r_ss = tau / nr;
  EXPECT_NEAR(ms(*s).data().qvel[5], r_ss, 0.02 * r_ss) << "steady yaw rate";
  Boat b2 = b;
  auto t = make(b2);
  put_ok(*t, "/devices/sea", "wave_scale", {0});
  put_ok(*t, "/devices/fan", "ignite", {1});
  ASSERT_EQ(t->step(730), 730);  // one time constant (Izz + a)/N_r = 0.73 s: r = r_ss (1 - 1/e)
  EXPECT_NEAR(ms(*t).data().qvel[5], r_ss * (1 - std::exp(-730e-3 * nr / mz)), 0.04 * r_ss);
}

// A current pushes the still hull to the current's speed through the relative-velocity drag.
TEST(vessel, current_drags_the_hull_to_the_water_speed) {
  Boat b;
  b.current_x = 1.0;
  auto s = make(b);
  put_ok(*s, "/devices/sea", "wave_scale", {0});
  ASSERT_EQ(s->step(60000), 60000);
  EXPECT_NEAR(ms(*s).data().qvel[0], 1.0, 0.01);
  put_ok(*s, "/devices/sea", "current", {0, 0.5, 0});
  ASSERT_EQ(s->step(60000), 60000);
  EXPECT_NEAR(ms(*s).data().qvel[1], 0.5, 0.01);
  EXPECT_NEAR(ms(*s).data().qvel[0], 0.0, 0.01);
}

// The wave table: dispersion w^2 = g k (deep) or g k tanh(k h) (depth), a JONSWAP sea synthesises exactly Hm0 = Hs, and equal seeds give
// equal seas (different seeds different phases).
TEST(vessel, sea_dispersion_and_jonswap) {
  Boat b;
  b.wave_a = 0.1;
  b.wave_t = 6.0;
  auto s = make(b);
  auto w = getv(*s, "/devices/sea", "waves");
  ASSERT_EQ(w.size(), 5u);
  const double om = 2 * kPi / 6.0;
  EXPECT_NEAR(w[0], 0.1, 1e-15);
  EXPECT_NEAR(w[2], om, 1e-15);
  EXPECT_NEAR(w[1], om * om / 9.80665, 1e-12);
  Boat d = b;
  d.extra_sea = ", \"depth\": 3.0";  // shallow: k from w^2 = g k tanh(k h)
  auto sd = make(d);
  const auto wd = getv(*sd, "/devices/sea", "waves");
  EXPECT_NEAR(9.80665 * wd[1] * std::tanh(wd[1] * 3.0), om * om, 1e-9);
  EXPECT_GT(wd[1], w[1]);
  Boat j;
  j.extra_sea = ", \"jonswap\": {\"hs\": 0.6, \"tp\": 5.0, \"n\": 30, \"seed\": 11}";
  auto sj = make(j), sj2 = make(j);
  j.extra_sea = ", \"jonswap\": {\"hs\": 0.6, \"tp\": 5.0, \"n\": 30, \"seed\": 12}";
  auto sj3 = make(j);
  const auto a = getv(*sj, "/devices/sea", "waves"), a2 = getv(*sj2, "/devices/sea", "waves"), a3 = getv(*sj3, "/devices/sea", "waves");
  ASSERT_EQ(a.size(), 150u);
  double m0 = 0;
  for (std::size_t i = 0; i < 30; ++i) m0 += 0.5 * a[5 * i] * a[5 * i];
  EXPECT_NEAR(4 * std::sqrt(m0), 0.6, 1e-12);
  EXPECT_EQ(a, a2);
  EXPECT_NE(a, a3);
  // spectral peak: the largest components sit near 2 pi / tp
  double wp = 0, amax = 0;
  for (std::size_t i = 0; i < 30; ++i)
    if (a[5 * i] > amax) amax = a[5 * i], wp = a[5 * i + 2];
  EXPECT_NEAR(wp, 2 * kPi / 5.0, 0.25 * 2 * kPi / 5.0);
}

// Marine propeller: thrust = K_T rho_w n^2 D^4 from the rotor's own outputs, scaled by the immersion of the hub: 1 when submerged by
// half a diameter, 1/2 at the surface, 0 above. The current enters the advance ratio.
TEST(vessel, water_rotor_thrust_immersion_and_current) {
  auto make_prop = [&](double hub_z, double cur_x) {
    std::ostringstream x;
    x << "<mujoco model=\"prop\"><option timestep=\"0.001\" gravity=\"0 0 0\"/><worldbody>"
      << "<body name=\"stator\" pos=\"0 0 " << hub_z << "\"><geom type=\"box\" size=\"0.02 0.02 0.02\" mass=\"0.1\" contype=\"0\" "
      << "conaffinity=\"0\"/><body name=\"rotor\"><joint name=\"shaft\" type=\"hinge\" axis=\"1 0 0\"/><geom type=\"sphere\" size=\"0.001\" "
      << "mass=\"0.001\" contype=\"0\" conaffinity=\"0\"/></body></body></worldbody><custom><text name=\"dvsim.devices\" data='{"
      << "\"seas\": [{\"name\": \"sea\", \"rho\": 998.2, \"current\": [" << cur_x << ", 0, 0]}], "
      << "\"psus\": [{\"name\": \"bench\", \"node\": 1, \"voltage\": 12, \"r_series\": 0.001}], "
      << "\"escs\": [{\"name\": \"m1\", \"node\": 1, \"joint\": \"shaft\", \"motor\": {\"kv\": 900, \"r20\": 0.1, \"i0\": 0.5, "
      << "\"rotor_inertia\": 3e-5, \"c_th\": 2000, \"r_th\": 3}}], "
      << "\"rotors\": [{\"name\": \"prop\", \"body\": \"rotor\", \"joint\": \"shaft\", \"diameter\": 0.05, \"water\": true, "
      << "\"j\": [0, 0.5, 1.0], \"ct\": [0.40, 0.25, 0.05], \"cq\": [0.04, 0.03, 0.01]}]}'/></custom></mujoco>";
    const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_prop_" + std::to_string(::getpid()) + ".xml")).string();
    std::ofstream(path) << x.str();
    auto s = load_mujoco(path);
    std::remove(path.c_str());
    put_ok(*s, "/devices/m1", "throttle", {0.3});
    return s;
  };
  const double d4 = std::pow(0.05, 4);
  for (const double depth : {-0.5, -0.025, 0.0, 0.5}) {  // hub z: 0.5 m deep ... 0.5 m above the surface
    auto s = make_prop(depth, 0.0);
    ASSERT_EQ(s->step(2500), 2500);
    const double n = get(*s, "/devices/prop", "omega") / (2 * kPi), thrust = get(*s, "/devices/prop", "thrust");
    const double imm = std::clamp(0.5 + (0.0 - depth) / 0.05, 0.0, 1.0);
    EXPECT_NEAR(get(*s, "/devices/prop", "ge_ratio"), imm, 1e-12) << "hub z " << depth;
    EXPECT_NEAR(get(*s, "/devices/prop", "rho"), 998.2, 1e-9);
    EXPECT_GT(std::abs(n), 10.0);
    // advance ratio ~ 0 (stator fixed, no current): static coefficient 0.40 (the table clamps J < 0 at j[0], J is tiny here)
    EXPECT_NEAR(thrust, imm * 0.40 * 998.2 * n * n * d4, 1e-3 * imm * 0.40 * 998.2 * n * n * d4 + 1e-12) << "hub z " << depth;
  }
  auto c = make_prop(-0.5, 2.0);
  ASSERT_EQ(c->step(2500), 2500);
  const double n = get(*c, "/devices/prop", "omega") / (2 * kPi);
  // axis +x, spinning in the design direction: v_ax = (0 - 2) -> J = -2 / (|n| D)
  EXPECT_NEAR(get(*c, "/devices/prop", "advance_ratio"), -2.0 / (std::abs(n) * 0.05), 1e-9);
}

// Bidirectional ESC + reverse thrust: a negative throttle spins the propeller the other way, and the thrust is the forward coefficients' value scaled by
// `rev_thrust` (a T200 gives 2.92 / 3.71 = 0.787 of its forward thrust in reverse at the same power). An ESC that is not bidirectional refuses a negative
// throttle at the interface; both directions leave the forward behaviour (hashes of the existing models) untouched.
TEST(vessel, bidirectional_esc_reverse_thrust_and_rev_ratio) {
  auto make_bench = [&](bool bidi, double rev) {
    std::ostringstream x;
    x << "<mujoco model=\"prop\"><option timestep=\"0.001\" gravity=\"0 0 0\"/><worldbody>"
      << "<body name=\"stator\" pos=\"0 0 -0.5\"><geom type=\"box\" size=\"0.02 0.02 0.02\" mass=\"0.1\" contype=\"0\" conaffinity=\"0\"/>"
      << "<body name=\"rotor\"><joint name=\"shaft\" type=\"hinge\" axis=\"1 0 0\"/><geom type=\"sphere\" size=\"0.001\" mass=\"0.001\" "
      << "contype=\"0\" conaffinity=\"0\"/></body></body></worldbody><custom><text name=\"dvsim.devices\" data='{"
      << "\"seas\": [{\"name\": \"sea\", \"rho\": 998.2}], "
      << "\"psus\": [{\"name\": \"bench\", \"node\": 1, \"voltage\": 12, \"r_series\": 0.001}], "
      << "\"escs\": [{\"name\": \"m1\", \"node\": 1, \"joint\": \"shaft\", \"bidirectional\": " << (bidi ? "true" : "false")
      << ", \"motor\": {\"kv\": 900, \"r20\": 0.1, \"i0\": 0.5, \"rotor_inertia\": 3e-5, \"c_th\": 2000, \"r_th\": 3}}], "
      << "\"rotors\": [{\"name\": \"prop\", \"body\": \"rotor\", \"joint\": \"shaft\", \"diameter\": 0.05, \"water\": true, \"rev_thrust\": " << rev
      << ", \"j\": [0, 0.5, 1.0], \"ct\": [0.40, 0.25, 0.05], \"cq\": [0.04, 0.03, 0.01]}]}'/></custom></mujoco>";
    const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_bidi_" + std::to_string(::getpid()) + ".xml")).string();
    std::ofstream(path) << x.str();
    auto s = load_mujoco(path);
    std::remove(path.c_str());
    return s;
  };
  const double d4 = std::pow(0.05, 4), rev = 0.787;
  auto f = make_bench(true, rev), r = make_bench(true, rev);
  put_ok(*f, "/devices/m1", "throttle", {0.3});
  put_ok(*r, "/devices/m1", "throttle", {-0.3});
  ASSERT_EQ(f->step(2500), 2500);
  ASSERT_EQ(r->step(2500), 2500);
  const double nf = get(*f, "/devices/prop", "omega") / (2 * kPi), nr = get(*r, "/devices/prop", "omega") / (2 * kPi);
  EXPECT_GT(nf, 10.0);
  EXPECT_LT(nr, -10.0);  // the other way round
  EXPECT_NEAR(std::abs(nr), std::abs(nf), 0.02 * std::abs(nf));  // a symmetric bridge: the same speed for the same duty
  const double tf = get(*f, "/devices/prop", "thrust"), tr = get(*r, "/devices/prop", "thrust");
  EXPECT_NEAR(tf, 0.40 * kRho * nf * nf * d4, 2e-3 * tf);
  EXPECT_LT(tr, 0.0);
  EXPECT_NEAR(tr, -rev * 0.40 * kRho * nr * nr * d4, 2e-3 * std::abs(tr));
  // a unidirectional ESC refuses the negative throttle, and the same one with rev_thrust = 1 gives the symmetric thrust
  auto u = make_bench(false, 1.0);
  const Result bad = u->set("/devices/m1", "throttle", "", std::vector<double>{-0.3});
  EXPECT_NE(bad.code, Result::ok);
  auto s1 = make_bench(true, 1.0);
  put_ok(*s1, "/devices/m1", "throttle", {-0.3});
  ASSERT_EQ(s1->step(2500), 2500);
  EXPECT_NEAR(get(*s1, "/devices/prop", "thrust") / (tr / -rev) * -1.0, 1.0, 5e-3);  // rev_thrust 1: |T| = K_T rho n^2 D^4 in both directions
}

// Skeg (Vessel.fins): a fin aft of the centre sees the sideslip and pushes the stern back to the flow. Water 1 m/s along the hull and 0.1 m/s across
// (a current past a hull at rest, no other hydrodynamics): F_y = -sum imm (rho / 2) A (cl_alpha |u| w + cd |w| w) at each fin, a yaw torque -x_f F_y
// that turns the bow INTO the flow; without fins nothing acts.
TEST(vessel, skeg_force_and_yaw_moment_oppose_sideslip) {
  const double A = 0.03, cla = 1.02, cd = 1.1, xf = -0.45;
  for (const bool with_fins : {true, false}) {
    Boat b;
    b.added = "[0,0,0,0,0,0]";
    b.lin = "[0,0,0,0,0,0]";
    b.quad = "[0,0,0,0,0,0]";
    // the body origin floats 0.05 m above the water (draft 0.05 m): fins 0.15 m below it are 0.1 m deep, fully immersed (imm = 1, span 0.1 m)
    if (with_fins)
      b.fins = "{\"pos\": [-0.45, 0.1, -0.15], \"area\": 0.03, \"span\": 0.1, \"cl_alpha\": 1.02, \"cd_cross\": 1.1}, "
               "{\"pos\": [-0.45, -0.1, -0.15], \"area\": 0.03, \"span\": 0.1, \"cl_alpha\": 1.02, \"cd_cross\": 1.1}";
    auto s = make(b);
    put_ok(*s, "/devices/sea", "wave_scale", {0});
    put_ok(*s, "/devices/sea", "current", {-1.0, -0.1, 0.0});  // water moves past the hull: body-relative velocity (+1.0, +0.1)
    ASSERT_EQ(s->step(1), 1);
    const double fy = getv(*s, "/devices/v", "hydro")[1];
    const double want = with_fins ? -2.0 * (kRho / 2) * A * (cla * 1.0 * 0.1 + cd * 0.1 * 0.1) : 0.0;
    EXPECT_NEAR(fy, want, 1e-3 + 1e-3 * std::abs(want)) << (with_fins ? "with fins" : "without");
    ASSERT_EQ(s->step(9), 9);  // 10 ms in all: the sway has not yet changed the relative flow
    const double r = ms(*s).data().qvel[5];
    if (with_fins) {
      // torque about the CoM (x_com = 0 for the symmetric hull): tau = x_f F_y; r = tau t / Izz, Izz = m (L^2 + W^2) / 12 of the 1.0 x 0.4 box
      const double tau = xf * want, izz = b.mass * (1.0 + 0.16) / 12.0;
      EXPECT_NEAR(r, tau * 0.01 / izz, 0.05 * std::abs(tau * 0.01 / izz));
      EXPECT_GT(r, 0.0) << "positive sideslip water velocity at the stern turns the bow to port, into the relative flow";
    } else {
      EXPECT_NEAR(r, 0.0, 1e-9);
    }
    EXPECT_FALSE(s->diverged());
  }
}

// Vessel in waves, thrusting: save / restore / clone and a model reloaded from bytes are bitwise (sea clock, wave table in the state).
TEST(vessel, save_restore_clone_bitwise) {
  Boat b;
  b.wave_a = 0.08;
  b.thrust_n = 10;
  b.extra_sea = ", \"jonswap\": {\"hs\": 0.3, \"tp\": 4.0, \"n\": 8, \"seed\": 5}";
  auto a = make(b), c0 = make(b);
  for (auto* s : {a.get(), c0.get()}) {
    put_ok(*s, "/devices/fan", "ignite", {1});
    ASSERT_EQ(s->step(1500), 1500);
  }
  const auto snap = c0->save();
  auto cl = c0->clone();
  ASSERT_EQ(c0->step(500), 500);
  ASSERT_NE(c0->state_hash(), a->state_hash());
  c0->restore(snap);
  EXPECT_EQ(c0->state_hash(), a->state_hash());
  auto r = load_mujoco_bytes(a->model_bytes());
  r->restore(a->save());
  EXPECT_EQ(r->state_hash(), a->state_hash());
  for (auto* s : {a.get(), c0.get(), cl.get(), r.get()}) ASSERT_EQ(s->step(500), 500);
  EXPECT_EQ(a->state_hash(), c0->state_hash());
  EXPECT_EQ(a->state_hash(), cl->state_hash());
  EXPECT_EQ(a->state_hash(), r->state_hash());
}

// A vessel or a water rotor without a sea is refused at load.
TEST(vessel, needs_a_sea) {
  Boat b;
  b.no_sea = true;
  EXPECT_THROW(make(b), std::runtime_error);
}
