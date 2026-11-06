// Solid rocket motor + rocket airframe (src/mujoco/devices/rocket.cpp). Expectations are written independently of the implementation
// (closed forms, the NAR / thrustcurve.org numbers, a separate 1-D integration). Fixtures: rocket_free.xml (Estes F15 on a 0.9 kg
// airframe, zero gravity), rocket_tilt.xml (nozzle tilted 10 deg), rocket_alpha.xml (Estes Alpha III class on a C6, standing on the ground).
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "mujoco/mujoco_solver.h"

namespace {
const std::string kDir = std::string(DVSIM_TEST_DATA_DIR) + "/devices/";
constexpr double kPi = 3.14159265358979323846;

std::unique_ptr<Solver> load(const std::string& f) { return load_mujoco(kDir + f); }
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
void put_ok(Solver& s, const std::string& prim, const std::string& attr, double v) {
  const Result r = s.set(prim, attr, "", std::span<const double>(&v, 1));
  EXPECT_EQ(r.code, Result::ok) << r.message;
}
int body_id(const Solver& s, const char* name) { return mj_name2id(&ms(s).model(), mjOBJ_BODY, name); }

// Estes F15 (NAR certification curve, thrustcurve.org simfile 5f4294d20002e900000007e5), and the published totals.
const double kF15t[] = {0.148, 0.228, 0.294, 0.353, 0.382, 0.419, 0.477, 0.520, 0.593, 0.688, 0.855, 1.037, 1.205,
                        1.423, 1.452, 1.503, 1.736, 1.955, 2.210, 2.494, 2.763, 3.120, 3.382, 3.404, 3.418, 3.450};
const double kF15f[] = {7.638, 12.253, 16.391, 20.210, 22.756, 25.260, 23.074, 20.845, 19.093, 17.500, 16.225, 15.427, 14.948,
                        14.627, 15.741, 14.785, 14.623, 14.303, 14.141, 13.819, 13.338, 13.334, 13.013, 9.352, 4.895, 0.0};
constexpr double kF15Cert = 49.61, kF15Prop = 0.060, kF15Total = 0.103;
double trapezoid(const double* t, const double* f, int n) {  // (0, 0) implied before the first point
  double i = 0.5 * t[0] * f[0];
  for (int k = 1; k < n; ++k) i += 0.5 * (f[k] + f[k - 1]) * (t[k] - t[k - 1]);
  return i;
}

double isa(double h) {
  const double t = 288.15 - 0.0065 * h;
  return 101325.0 * std::pow(t / 288.15, 9.80665 / (287.05287 * 0.0065)) / (287.05287 * t);
}
}  // namespace

// The delivered impulse is the curve's integral and matches the certified total; the propellant burns to the certified mass; the
// velocity gain is the rocket equation dv = (I / m_p) ln(M0 / Mf) (no gravity, no drag), which ties force, mass update and integration.
TEST(rocket, impulse_mass_and_rocket_equation) {
  auto s = load("rocket_free.xml");
  const double table = trapezoid(kF15t, kF15f, 26);
  EXPECT_NEAR(table, kF15Cert, 0.001 * kF15Cert);  // the curve file against the NAR number
  EXPECT_EQ(get(*s, "/devices/f15", "mass"), kF15Total);
  put_ok(*s, "/devices/f15", "ignite", 1);
  ASSERT_EQ(s->step(4200), 4200);  // 0.3 s delay + 3.45 s burn + coast
  EXPECT_EQ(get(*s, "/devices/f15", "phase"), 3.0);
  EXPECT_NEAR(get(*s, "/devices/f15", "impulse"), table, 1e-9 * table);
  EXPECT_NEAR(get(*s, "/devices/f15", "progress"), 1.0, 1e-12);
  EXPECT_NEAR(get(*s, "/devices/f15", "burned_mass"), kF15Prop, 1e-12);
  EXPECT_NEAR(get(*s, "/devices/f15", "mass"), kF15Total - kF15Prop, 1e-12);
  const mjModel& m = ms(*s).model();
  EXPECT_NEAR(m.body_mass[body_id(*s, "motor")], kF15Total - kF15Prop, 1e-12);        // the model itself changed
  EXPECT_NEAR(m.body_subtreemass[body_id(*s, "rocket")], 0.9 + kF15Total - kF15Prop, 1e-12);
  EXPECT_NEAR(m.body_subtreemass[0], 0.9 + kF15Total - kF15Prop, 1e-12);
  EXPECT_EQ(get(*s, "/devices/f15", "thrust"), 0.0);
  const double m0 = 0.9 + kF15Total, mf = m0 - kF15Prop;
  const double dv = table / kF15Prop * std::log(m0 / mf);
  EXPECT_NEAR(ms(*s).data().qvel[2], dv, 2e-3 * dv) << "rocket equation";
  EXPECT_NEAR(ms(*s).data().qvel[0], 0.0, 1e-9);  // on-axis thrust: no side velocity, no spin
  EXPECT_NEAR(ms(*s).data().qvel[4], 0.0, 1e-9);
}

// Nothing happens before the ignition delay; the ignite command clears itself and a second one while burning is ignored; inside the
// first curve segment (0, 0) -> (0.148, 7.638) thrust is linear in time, so the impulse after tb is 25.8 tb^2 exactly.
TEST(rocket, ignition_delay_single_shot_and_start_ramp) {
  auto s = load("rocket_free.xml");
  ASSERT_EQ(s->step(200), 200);
  EXPECT_EQ(get(*s, "/devices/f15", "phase"), 0.0);  // armed, no command yet
  put_ok(*s, "/devices/f15", "ignite", 1);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "/devices/f15", "ignite"), 0.0);
  EXPECT_EQ(get(*s, "/devices/f15", "phase"), 1.0);
  ASSERT_EQ(s->step(289), 289);  // 290 ticks = 0.29 s after the command
  EXPECT_EQ(get(*s, "/devices/f15", "thrust"), 0.0);
  EXPECT_EQ(get(*s, "/devices/f15", "impulse"), 0.0);
  EXPECT_EQ(get(*s, "/devices/f15", "phase"), 1.0);
  ASSERT_EQ(s->step(40), 40);  // 330 ticks: 0.03 s of burn
  EXPECT_EQ(get(*s, "/devices/f15", "phase"), 2.0);
  const double tb = 0.03, slope = 7.638 / 0.148;
  EXPECT_NEAR(get(*s, "/devices/f15", "impulse"), 0.5 * slope * tb * tb, 1e-6);
  EXPECT_NEAR(get(*s, "/devices/f15", "burn_time"), tb, 1e-9);
  put_ok(*s, "/devices/f15", "ignite", 1);  // already lit: nothing restarts
  ASSERT_EQ(s->step(1), 1);
  EXPECT_NEAR(get(*s, "/devices/f15", "burn_time"), tb + 0.001, 1e-9);
  // thrust_scale scales the force, not the burn: half scale = half the impulse, the propellant still burns out on schedule
  auto h = load("rocket_free.xml");
  put_ok(*h, "/devices/f15", "thrust_scale", 0.5);
  put_ok(*h, "/devices/f15", "ignite", 1);
  ASSERT_EQ(h->step(4200), 4200);
  EXPECT_NEAR(get(*h, "/devices/f15", "impulse"), 0.5 * kF15Cert, 0.001 * kF15Cert);
  EXPECT_NEAR(get(*h, "/devices/f15", "burned_mass"), kF15Prop, 1e-12);
}

// A motor axis tilted 10 deg: the impulse vector points along the axis, and the angular momentum about the system CoM is
// (nozzle - CoM) x impulse; the device also owns the motor body's inertia (cylinders, principal axes along the site z).
TEST(rocket, tilted_nozzle_force_torque_and_inertia) {
  auto s = load("rocket_tilt.xml");
  put_ok(*s, "/devices/f15", "ignite", 1);
  ASSERT_EQ(s->step(5), 5);
  const mjModel& m = ms(*s).model();
  mjData* d = mj_copyData(nullptr, &m, &ms(*s).data());
  mj_forward(&m, d);
  mj_subtreeVel(&m, d);
  const double mass = m.body_subtreemass[0], th = 10.0 * kPi / 180.0;
  const double imp = 0.5 * (7.638 / 0.148) * 0.005 * 0.005;  // N s over the first 5 ticks (linear start segment)
  const double px = d->subtree_linvel[0] * mass, pz = d->subtree_linvel[2] * mass, py = d->subtree_linvel[1] * mass;
  EXPECT_NEAR(px, imp * std::sin(th), 5e-9);  // (the mass changed by 1e-6 relative while the impulse was delivered)
  EXPECT_NEAR(pz, imp * std::cos(th), 5e-9);
  EXPECT_NEAR(py, 0.0, 1e-12);
  // L about the system CoM = r x J, r = nozzle - CoM (the rocket has rotated by ~1e-4 rad: second order)
  const double cz = d->subtree_com[2], nozzle_z = getv(*s, "/devices/f15", "pos")[2];
  const double l_want = (nozzle_z - cz) * imp * std::sin(th);
  EXPECT_NEAR(d->subtree_angmom[1], l_want, 2e-3 * std::abs(l_want));
  EXPECT_NEAR(d->subtree_angmom[0], 0.0, 1e-9);
  mj_deleteData(d);
  // The motor body: CoM half a length ahead of the nozzle along the (tilted) axis, inertia of casing + remaining propellant
  // (rewritten at the start of a tick from the progress of the one before, here 5 ticks of a 3.45 s burn: propellant essentially full).
  const int b = body_id(*s, "motor");
  const double len = 0.114, r = 0.0145, mc = kF15Total - kF15Prop;
  const double i_ax = mc * r * r + 0.5 * kF15Prop * r * r;
  const double i_lat = mc * (0.5 * r * r + len * len / 12) + kF15Prop * (0.25 * r * r + len * len / 12);
  EXPECT_NEAR(m.body_inertia[3 * b + 2], i_ax, 1e-3 * i_ax);
  EXPECT_NEAR(m.body_inertia[3 * b + 0], i_lat, 1e-3 * i_lat);
  EXPECT_EQ(m.body_inertia[3 * b + 0], m.body_inertia[3 * b + 1]);
  EXPECT_NEAR(m.body_ipos[3 * b + 0], 0.5 * len * std::sin(th), 1e-12);
  EXPECT_NEAR(m.body_ipos[3 * b + 2], -0.057 + 0.5 * len * std::cos(th), 1e-12);
}

// An unlit rocket on the ground stays put: no thrust, the contact holds it, the idle device changes nothing.
TEST(rocket, passive_rocket_stays_on_the_pad) {
  auto s = load("rocket_alpha.xml");
  ASSERT_EQ(s->step(200), 200);
  const std::vector<double> q0(ms(*s).data().qpos, ms(*s).data().qpos + 7);
  ASSERT_EQ(s->step(3000), 3000);
  const mjData& d = ms(*s).data();
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(d.qpos[i], q0[static_cast<std::size_t>(i)], 2e-4) << "axis " << i;
  EXPECT_NEAR(d.qpos[3], 1.0, 1e-4);  // quaternion w: no tilt
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(d.qvel[i], 0.0, 2e-3) << "dof " << i;
  EXPECT_EQ(get(*s, "/devices/c6", "phase"), 0.0);
  EXPECT_EQ(get(*s, "/devices/c6", "thrust"), 0.0);
  EXPECT_NEAR(get(*s, "/devices/c6", "mass"), 0.0231, 1e-12);
}

// Unguided launch of an Alpha III class rocket on a C6: apogee against (a) an independent 1-D integration with the same physics and
// (b) the band around Estes' published "up to 1150 ft (351 m)" for the kit (a manufacturer's claim, +-20 %).
TEST(rocket, alpha_iii_apogee_matches_reference_and_estes_figure) {
  auto s = load("rocket_alpha.xml");
  ASSERT_EQ(s->step(100), 100);
  put_ok(*s, "/devices/c6", "ignite", 1);
  double zmax = 0, t_apogee = 0;
  double max_tilt = 0;
  for (int i = 0; i < 20000; ++i) {
    ASSERT_EQ(s->step(5), 5);
    const mjData& d = ms(*s).data();
    if (d.qpos[2] > zmax) zmax = d.qpos[2], t_apogee = d.time;
    max_tilt = std::max(max_tilt, std::acos(std::clamp(1.0 - 2.0 * (d.qpos[4] * d.qpos[4] + d.qpos[5] * d.qpos[5]), -1.0, 1.0)));
    if (d.time > 3.0 && d.qvel[2] < -5.0) break;
  }
  // reference: z'' = (T - 1/2 rho cd A |v| v) / m - g, mass linear in the delivered impulse
  const double tc[] = {0.031, 0.092, 0.139, 0.192, 0.209, 0.231, 0.248, 0.292, 0.370, 0.475, 0.671, 0.702,
                       0.723, 0.850, 1.063, 1.211, 1.242, 1.303, 1.468, 1.656, 1.821, 1.834, 1.847, 1.860};
  const double fc[] = {0.946, 4.826, 9.936, 14.090, 11.446, 7.381, 6.151, 5.489, 4.921, 4.448, 4.258, 4.542,
                       4.164, 4.448, 4.353, 4.353, 4.069, 4.258, 4.353, 4.448, 4.448, 2.933, 1.325, 0.0};
  const double itot = trapezoid(tc, fc, 24), area = 0.000483, cd = 0.65;
  EXPECT_NEAR(itot, 8.82, 0.01 * 8.82);  // NAR certified total impulse of the C6
  auto thrust = [&](double t) {
    double prev_t = 0, prev_f = 0;
    for (int k = 0; k < 24; ++k) {
      if (t <= tc[k]) return prev_f + (fc[k] - prev_f) * (t - prev_t) / (tc[k] - prev_t);
      prev_t = tc[k], prev_f = fc[k];
    }
    return 0.0;
  };
  double z = 0.0005, v = 0, t = 0, cum = 0, zref = 0;
  const double dt = 1e-4, delay = 0.2 + 0.001;
  while (!(v < 0 && t > 3)) {
    const double tb = std::max(t - delay, 0.0), f = thrust(tb);
    cum += f * dt;
    const double mass = 0.034 + 0.0231 - 0.0108 * std::min(cum / itot, 1.0);
    const double a = (f - 0.5 * isa(z) * cd * area * std::abs(v) * v) / mass - 9.81;
    v += a * dt;
    z += v * dt;
    t += dt;
    zref = std::max(zref, z);
    if (z < 0.0005) z = 0.0005, v = std::max(v, 0.0);  // on the ground until the thrust beats the weight
  }
  EXPECT_NEAR(zmax, zref, 0.02 * zref) << "device apogee " << zmax << " m, reference " << zref << " m";
  EXPECT_GT(zmax, 0.8 * 351.0);
  EXPECT_LT(zmax, 1.2 * 351.0);
  EXPECT_LT(max_tilt, 0.05) << "a stable rocket flies straight up";
  EXPECT_GT(t_apogee, 5.0);
}

// A rocket in a crosswind weathercocks into it (CP behind the CG): the nose tilts towards the wind source, not away.
TEST(rocket, weathercocks_into_the_wind) {
  auto s = load("rocket_alpha.xml");
  ASSERT_EQ(s->step(100), 100);
  put_ok(*s, "/devices/c6", "ignite", 1);
  const double w[3] = {0, 5.0, 0};  // air moves towards +y
  ASSERT_EQ(s->set("/devices/air", "wind", "", std::span<const double>(w, 3)).code, Result::ok);
  ASSERT_EQ(s->step(1200), 1200);
  const mjData& d = ms(*s).data();
  // body z axis (nose) in the world: third column of the rotation of the root body
  double R[9];
  mju_quat2Mat(R, d.qpos + 3);
  EXPECT_LT(R[5], -0.005) << "the nose should lean towards -y (into the wind that blows towards +y): z axis y component " << R[5];
  EXPECT_GT(R[8], 0.95);
}

// Save / restore / clone and a model reloaded from the bytes of a model that was saved MID-BURN (its masses already reduced) all stay
// bitwise on the same trajectory: the masses are recomputed from the saved state every tick.
TEST(rocket, save_restore_clone_and_remodel_bitwise) {
  auto a = load("rocket_alpha.xml"), b = load("rocket_alpha.xml");
  for (auto* s : {a.get(), b.get()}) {
    put_ok(*s, "/devices/c6", "ignite", 1);
    ASSERT_EQ(s->step(700), 700);  // mid-burn
  }
  const std::vector<std::byte> snap = b->save();
  auto c = b->clone();
  ASSERT_EQ(b->step(500), 500);
  ASSERT_NE(b->state_hash(), a->state_hash());
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), a->state_hash());
  const auto bytes = a->model_bytes();  // masses of a, mid-burn
  auto r = load_mujoco_bytes(bytes);
  r->restore(a->save());
  EXPECT_EQ(r->state_hash(), a->state_hash());
  ASSERT_EQ(a->step(500), 500);
  ASSERT_EQ(b->step(500), 500);
  ASSERT_EQ(c->step(500), 500);
  ASSERT_EQ(r->step(500), 500);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->state_hash(), c->state_hash());
  EXPECT_EQ(a->state_hash(), r->state_hash());
  EXPECT_GT(get(*a, "/devices/c6", "progress"), 0.5);  // 1.2 s of a 2.06 s run: still burning
  EXPECT_LT(get(*a, "/devices/c6", "progress"), 0.7);
}

// Input validation and bind checks.
TEST(rocket, input_validation) {
  auto s = load("rocket_free.xml");
  EXPECT_EQ(s->set("/devices/f15", "ignite", "", std::vector<double>{0.5}).code, Result::invalid_argument);
  EXPECT_EQ(s->set("/devices/f15", "thrust", "", std::vector<double>{1.0}).code, Result::invalid_argument);  // read-only
  EXPECT_EQ(s->set("/devices/f15", "thrust_scale", "", std::vector<double>{3.0}).code, Result::invalid_argument);
  EXPECT_EQ(s->unit("/devices/f15", "impulse"), "N*s");
}
