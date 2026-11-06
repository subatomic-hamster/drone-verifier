// Jet vane (src/mujoco/devices/vane.cpp) and the EDF stator torque of the rotor (aero.cpp). Fixture: jetvane.xml. Expectations are
// closed forms written independently of the implementation (rotating the plate by hand, the momentum-theory exit speed).
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
void put_ok(Solver& s, const std::string& prim, const std::string& attr, double v) {
  const Result r = s.set(prim, attr, "", std::span<const double>(&v, 1));
  EXPECT_EQ(r.code, Result::ok) << r.message;
}
const double* xfrc(const Solver& s, const char* body) {
  return ms(s).data().xfrc_applied + 6 * mj_name2id(&ms(s).model(), mjOBJ_BODY, body);
}
double rel(double got, double want) { return std::abs(got - want) / std::abs(want); }

constexpr double kArea = 0.0012, kExit = 0.0045, kCnA = 2.0, kCnMax = 1.2, kCd0 = 0.05, kJf = 0.9;
// The vane force in the world frame for a plate deflected by `delta` about y (R_y(delta) turns chord (0,0,-1) into (-sin, 0, -cos),
// normal (1,0,0) into (cos, 0, -sin)), exhaust direction e = (0,0,sgn), jet pressure q.
struct F { double x, z; };
F expected(double q, double delta, double sgn) {
  double cx = -std::sin(delta), cz = -std::cos(delta), nx = std::cos(delta), nz = -std::sin(delta);
  if (cz * sgn < 0) cx = -cx, cz = -cz, nx = -nx, nz = -nz;  // chord turned downstream
  const double alpha = std::atan2(sgn * nz, sgn * cz), cn = kCnMax * std::tanh(kCnA * std::sin(alpha) / kCnMax);
  return {q * kArea * (cn * nx + kCd0 * cx), q * kArea * (cn * nz + kCd0 * cz)};
}
}  // namespace

// Thrust through the rotor table, then the exit speed and dynamic pressure of the momentum theory; the plate at zero deflection only
// sees chord-wise drag, a deflected one a normal force; the force is applied at the site (torque = r x F about the plate body CoM).
TEST(jetvane, force_follows_the_plate_formula) {
  const double w = 2000, n = w / (2 * kPi);
  for (const double delta : {0.0, 0.2, -0.35, 1.2}) {
    auto s = load("jetvane.xml");
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", w);
    put_ok(*s, "/world/carrier/plate/flap", "qpos", delta);
    ASSERT_EQ(s->step(1), 1);
    const double rho = get(*s, "/devices/fan", "rho"), thrust = get(*s, "/devices/fan", "thrust");
    EXPECT_LT(rel(thrust, 1.3 * rho * n * n * std::pow(0.08, 4)), 1e-9);
    const double ve = kJf * std::sqrt(thrust / (rho * kExit)), q = 0.5 * rho * ve * ve;
    const F want = expected(q, delta, -1.0);
    const auto f = getv(*s, "/devices/vane", "force");
    EXPECT_LT(rel(get(*s, "/devices/vane", "jet_speed"), ve), 1e-9) << delta;
    EXPECT_NEAR(f[0], want.x, 1e-9 * q * kArea + 1e-12) << delta;
    EXPECT_NEAR(f[1], 0.0, 1e-12);
    EXPECT_NEAR(f[2], want.z, 1e-9 * q * kArea + 1e-12) << delta;
    const double* xf = xfrc(*s, "plate");
    EXPECT_NEAR(xf[0], f[0], 1e-12);
    EXPECT_NEAR(xf[2], f[2], 1e-12);
    // Torque about the plate's CoM (its origin): r = site - CoM = R_y(delta) (0, 0, -0.02) = -0.02 (sin, 0, cos), tau_y = r_z F_x - r_x F_z.
    const double rx = -0.02 * std::sin(delta), rz = -0.02 * std::cos(delta);
    EXPECT_NEAR(xf[4], rz * f[0] - rx * f[2], 1e-9 * q * kArea * 0.02 + 1e-12) << "torque y, delta " << delta;
    EXPECT_NEAR(xf[3], 0.0, 1e-12);
    EXPECT_NEAR(xf[5], 0.0, 1e-12);
    if (delta == 0.0) {
      EXPECT_NEAR(f[0], 0.0, 1e-12);  // no normal force at zero angle of attack
      EXPECT_LT(rel(f[2], -q * kArea * kCd0), 1e-9);
    }
  }
}

// Reversed spin: the exhaust goes up, the plate is symmetric, so the lateral force of the same deflection flips sign with the flow
// (the chord is turned downstream) and the speed is the same.
TEST(jetvane, reverse_flow_and_no_flow) {
  const double w = 2000;
  const double delta = 0.2;
  double fx_fwd = 0, ve = 0;
  {
    auto s = load("jetvane.xml");
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", w);
    put_ok(*s, "/world/carrier/plate/flap", "qpos", delta);
    ASSERT_EQ(s->step(1), 1);
    fx_fwd = getv(*s, "/devices/vane", "force")[0];
    ve = get(*s, "/devices/vane", "jet_speed");
  }
  {
    auto s = load("jetvane.xml");
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", -w);
    put_ok(*s, "/world/carrier/plate/flap", "qpos", delta);
    ASSERT_EQ(s->step(1), 1);
    EXPECT_LT(rel(get(*s, "/devices/vane", "jet_speed"), ve), 1e-9);
    const double rho = get(*s, "/devices/fan", "rho"), q = 0.5 * rho * ve * ve;
    const F want = expected(q, delta, +1.0);
    const auto f = getv(*s, "/devices/vane", "force");
    EXPECT_NEAR(f[0], want.x, 1e-9 * q * kArea + 1e-12);
    EXPECT_NEAR(f[2], want.z, 1e-9 * q * kArea + 1e-12);
    EXPECT_LT(f[0] * fx_fwd, 0.0);  // opposite lateral direction
  }
  {  // a stopped fan blows nothing
    auto s = load("jetvane.xml");
    put_ok(*s, "/world/carrier/plate/flap", "qpos", delta);
    ASSERT_EQ(s->step(1), 1);
    for (double v : getv(*s, "/devices/vane", "force")) EXPECT_EQ(v, 0.0);
    EXPECT_EQ(get(*s, "/devices/vane", "jet_speed"), 0.0);
  }
}

// Stator: the fan torque of the air is -sgn(w) Q on the rotor body; the stator body gets +0.8 of it back, so the vehicle sees 20 %.
TEST(jetvane, stator_returns_the_swirl_torque) {
  const double w = 2000, n = w / (2 * kPi);
  auto s = load("jetvane.xml");
  put_ok(*s, "/world/carrier/rotor/shaft", "qvel", w);
  ASSERT_EQ(s->step(1), 1);
  const double rho = get(*s, "/devices/fan", "rho"), q = 0.2 * rho * n * n * std::pow(0.08, 5);
  EXPECT_LT(rel(get(*s, "/devices/fan", "torque"), q), 1e-9);
  EXPECT_LT(rel(xfrc(*s, "rotor")[5], -q), 1e-9);
  EXPECT_LT(rel(xfrc(*s, "carrier")[5], 0.8 * q), 1e-9);
  EXPECT_NEAR(xfrc(*s, "rotor")[5] + xfrc(*s, "carrier")[5], -0.2 * q, 1e-9 * q);
}

// A vane naming something that is not a rotor, or a swirl share without a stator, fails to load with a message naming the device.
TEST(jetvane, bad_config_names_the_device) {
  struct Case { const char* file; const char* needle; };
  for (const Case& c : {Case{"bad_vane.xml", "vane_bad"}, Case{"bad_swirl.xml", "fan_bad"}}) {
    try {
      load(c.file);
      ADD_FAILURE() << c.file << " loaded";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(c.needle), std::string::npos) << c.file << ": " << e.what();
    }
  }
}
