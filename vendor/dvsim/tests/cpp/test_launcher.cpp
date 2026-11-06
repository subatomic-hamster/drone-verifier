// Counter-UAS pellet launcher (src/mujoco/devices/launcher.cpp) and the camera pose output. Expectations are written
// independently of the implementation (closed forms, a separate high-accuracy integration).
// Fixtures: launcher_nodrag.xml (cd 0, ground 10 m below the muzzle, barrel along +x, muzzle at (0.05, 0, 10)) and launcher.xml
// (cd 0.45, 0.55 g, 4.5 mm, 200 m/s, environment, a 2 mm plate 40 m downrange, one mocap visual, one camera on the muzzle site).
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

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
Result put(Solver& s, const std::string& prim, const std::string& attr, std::vector<double> v) { return s.set(prim, attr, "", v); }
void put_ok(Solver& s, const std::string& prim, const std::string& attr, std::vector<double> v) {
  const Result r = put(s, prim, attr, v);
  EXPECT_EQ(r.code, Result::ok) << r.message;
}
void fire(Solver& s) { put_ok(s, "/devices/gun", "fire", {1.0}); }
int body_id(const Solver& s, const char* name) { return mj_name2id(&ms(s).model(), mjOBJ_BODY, name); }

// ISA troposphere from the definition (as test_devices.cpp).
double isa(double h) {
  const double t = 288.15 - 0.0065 * h;
  return 101325.0 * std::pow(t / 288.15, 9.80665 / (287.05287 * 0.0065)) / (287.05287 * t);
}
// Independent reference: classical RK4 at 10 us over the whole path, rho(z) re-evaluated at every stage (the device holds it per tick).
void ref_flight(double p[3], double v[3], const double w[3], double cd, double t_end) {
  const double area = kPi * 0.0045 * 0.0045 / 4, m = 0.00055;
  auto acc = [&](const double* pp, const double* vv, double* a) {
    const double r[3] = {vv[0] - w[0], vv[1] - w[1], vv[2] - w[2]};
    const double k = isa(pp[2]) * cd * area / (2 * m) * std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    a[0] = -k * r[0];
    a[1] = -k * r[1];
    a[2] = -9.81 - k * r[2];
  };
  const int n = static_cast<int>(std::lround(t_end / 1e-5));
  const double h = t_end / n;
  for (int i = 0; i < n; ++i) {
    double y[6] = {p[0], p[1], p[2], v[0], v[1], v[2]}, k[4][6];
    for (int st = 0; st < 4; ++st) {
      double q[6];
      const double c = st == 0 ? 0 : st == 3 ? 1 : 0.5;
      for (int j = 0; j < 6; ++j) q[j] = y[j] + (st ? c * h * k[st - 1][j] : 0.0);
      for (int j = 0; j < 3; ++j) k[st][j] = q[3 + j];
      acc(q, q + 3, k[st] + 3);
    }
    for (int j = 0; j < 6; ++j) y[j] += h / 6 * (k[0][j] + 2 * k[1][j] + 2 * k[2][j] + k[3][j]);
    for (int j = 0; j < 3; ++j) p[j] = y[j], v[j] = y[3 + j];
  }
}
}  // namespace

// Cd 0: the flight is the closed-form parabola (RK4 is exact on a quadratic): x = x0 + v0 t, z = z0 - g t^2 / 2; then the ground
// ray ends it: ground counter 1, hit body 0 (world), the hit point on z = 0 at the analytic range and time.
TEST(launcher, no_drag_parabola_and_ground_hit) {
  auto s = load("launcher_nodrag.xml");
  fire(*s);
  ASSERT_EQ(s->step(300), 300);
  const auto p = getv(*s, "/devices/gun", "pos"), v = getv(*s, "/devices/gun", "vel");
  const double t = 0.3;
  EXPECT_NEAR(p[0], 0.05 + 200 * t, 1e-9);
  EXPECT_NEAR(p[1], 0.0, 1e-12);
  EXPECT_NEAR(p[2], 10 - 0.5 * 9.81 * t * t, 1e-9);
  EXPECT_NEAR(v[0], 200, 1e-12);
  EXPECT_NEAR(v[2], -9.81 * t, 1e-9);
  EXPECT_EQ(get(*s, "/devices/gun", "rounds"), 1);
  ASSERT_EQ(s->step(1200), 1200);  // the flight lasts sqrt(2 * 10 / 9.81) = 1.428 s
  EXPECT_EQ(get(*s, "/devices/gun", "ground"), 1);
  EXPECT_EQ(get(*s, "/devices/gun", "hits"), 0);
  EXPECT_EQ(get(*s, "/devices/gun", "hit_body"), 0);
  const auto hp = getv(*s, "/devices/gun", "hit_pos");
  EXPECT_NEAR(hp[2], 0.0, 1e-9);
  EXPECT_NEAR(hp[0], 0.05 + 200 * std::sqrt(2 * 10 / 9.81), 2e-3);  // chord vs. curve: sagitta g dt^2 / 8 ~ 1e-6 m
  EXPECT_EQ(getv(*s, "/devices/gun", "active")[0], 0.0);
  EXPECT_NEAR(get(*s, "/devices/gun", "hit_t"), std::sqrt(2 * 10 / 9.81), 2e-3);
}

// Cd > 0: the pellet follows an independent high-accuracy integration of the same law, loses speed, and with a 5 m/s crosswind
// the deflection has the right sign (+y wind -> +y) and the size of the lag rule w (t - x / v0).
TEST(launcher, drag_matches_reference_and_wind_deflects) {
  for (const double wy : {0.0, 5.0}) {
    auto s = load("launcher.xml");
    put_ok(*s, "/devices/air", "wind", {0, wy, 0});
    fire(*s);
    ASSERT_EQ(s->step(150), 150);
    double rp[3] = {0.05, 0, 10}, rv[3] = {200, 0, 0};
    const double w[3] = {0, wy, 0};
    ref_flight(rp, rv, w, 0.45, 0.150);
    const auto p = getv(*s, "/devices/gun", "pos"), v = getv(*s, "/devices/gun", "vel");
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(p[i], rp[i], 2e-4) << "wind " << wy << " axis " << i;
      EXPECT_NEAR(v[i], rv[i], 2e-3) << "wind " << wy << " axis " << i;
    }
    EXPECT_LT(v[0], 195);
    EXPECT_GT(v[0], 150);
    if (wy > 0) {
      EXPECT_GT(p[1], 0);
      const double lag = wy * (0.150 - (p[0] - 0.05) / 200);
      EXPECT_NEAR(p[1], lag, 0.25 * lag) << "crosswind deflection vs the lag rule";
    } else {
      EXPECT_EQ(p[1], 0.0);
    }
  }
}

// A 2 mm plate is crossed by a 200 m/s pellet in ~10 us while a tick moves it 0.2 m: only the swept segment test catches it. The
// hit is recorded on the plate's body at its front face, the pellet is removed, the mocap visual is parked again.
TEST(launcher, swept_hit_on_a_thin_plate) {
  auto s = load("launcher.xml");
  fire(*s);
  ASSERT_EQ(s->step(10), 10);
  const mjData& d = ms(*s).data();
  const int mc = ms(*s).model().body_mocapid[body_id(*s, "pellet0")];
  const auto p = getv(*s, "/devices/gun", "pos");
  EXPECT_EQ(d.mocap_pos[3 * mc + 0], p[0]);  // the visual follows the pellet
  EXPECT_EQ(d.mocap_pos[3 * mc + 2], p[2]);
  for (int i = 0; i < 400 && getv(*s, "/devices/gun", "active")[0] > 0.5; ++i) ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(getv(*s, "/devices/gun", "active")[0], 0.0);
  EXPECT_EQ(get(*s, "/devices/gun", "hits"), 1);
  EXPECT_EQ(get(*s, "/devices/gun", "ground"), 0);
  EXPECT_EQ(get(*s, "/devices/gun", "hit_n"), 1);
  EXPECT_EQ(get(*s, "/devices/gun", "hit_body"), body_id(*s, "target"));
  EXPECT_NEAR(getv(*s, "/devices/gun", "hit_pos")[0], 40 - 0.001, 1e-6);
  // v = v0 exp(-k x), k = rho cd A / (2 m) = 0.00797 /m: 200 exp(-0.319) = 145 m/s at 40 m
  EXPECT_NEAR(get(*s, "/devices/gun", "hit_speed"), 200 * std::exp(-0.00797 * 40), 3.0);
  EXPECT_EQ(get(*s, "/devices/gun", "hit_round"), 1);
  EXPECT_EQ(d.mocap_pos[3 * mc + 2], -1000.0);
}

// Pool of 16: a 17th shot while 16 fly is refused and counted. fire is a 0/1 input that clears itself.
TEST(launcher, pool_full_and_input_validation) {
  auto s = load("launcher_nodrag.xml");
  for (int i = 0; i < 17; ++i) {
    fire(*s);
    ASSERT_EQ(s->step(1), 1);
    EXPECT_EQ(get(*s, "/devices/gun", "fire"), 0.0);
  }
  EXPECT_EQ(get(*s, "/devices/gun", "rounds"), 16);
  EXPECT_EQ(get(*s, "/devices/gun", "dropped"), 1);
  EXPECT_EQ(put(*s, "/devices/gun", "fire", {0.5}).code, Result::invalid_argument);
  EXPECT_EQ(put(*s, "/devices/gun", "hits", {0}).code, Result::invalid_argument);  // read-only
  EXPECT_EQ(s->unit("/devices/gun", "pos"), "m");
}

// Mid-flight save / restore and clone reproduce the state hash bitwise (pellets, ring, counters are in the saved state).
TEST(launcher, save_restore_clone_bitwise) {
  auto a = load("launcher.xml"), b = load("launcher.xml");
  for (auto* s : {a.get(), b.get()}) {
    put_ok(*s, "/devices/air", "wind", {1, -2, 0});
    fire(*s);
    ASSERT_EQ(s->step(60), 60);
  }
  const std::vector<std::byte> snap = b->save();
  auto c = b->clone();
  ASSERT_EQ(b->step(300), 300);  // through the plate
  ASSERT_NE(b->state_hash(), a->state_hash());
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), a->state_hash());
  ASSERT_EQ(a->step(300), 300);
  ASSERT_EQ(b->step(300), 300);
  ASSERT_EQ(c->step(300), 300);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->state_hash(), c->state_hash());
  EXPECT_EQ(get(*a, "/devices/gun", "hits"), 1);
}

// Camera pose outputs follow the site: looks along the site z axis (here world +x), image up = the site x axis (here world -z).
TEST(launcher, camera_pose_follows_the_site) {
  auto s = load("launcher.xml");
  ASSERT_EQ(s->step(1), 1);
  const auto pos = getv(*s, "/devices/cam", "pos"), fwd = getv(*s, "/devices/cam", "fwd"), up = getv(*s, "/devices/cam", "up");
  const double ep[3] = {0.05, 0, 10}, ef[3] = {1, 0, 0}, eu[3] = {0, 0, -1};
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(pos[i], ep[i], 1e-12);
    EXPECT_NEAR(fwd[i], ef[i], 1e-12);
    EXPECT_NEAR(up[i], eu[i], 1e-12);
  }
}
