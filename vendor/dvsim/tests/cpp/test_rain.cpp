// Rain device (src/mujoco/devices/rain.cpp, aero.cpp, rangefinder.cpp; docs/models.md "Rain"). Expectations are the closed-form
// model: LWC = 0.072 R^0.88, v_fall = R / LWC (the water flux equals R), the rotor thrust factor 1 - kt LWC / 23, the film cap, and
// the sensor statistics (monotone in R, exactly the dry result at R = 0). Models are generated per test into a temp file.
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "mujoco/mujoco_solver.h"

namespace {
std::unique_ptr<Solver> make(const std::string& xml) {
  static int n = 0;
  const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_rain_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".xml")).string();
  std::ofstream(path) << xml;
  auto s = load_mujoco(path);
  std::remove(path.c_str());
  return s;
}
double get(const Solver& s, const char* dev, const char* attr, size_t i = 0) {
  std::vector<double> o;
  const Result r = s.get(std::string("/devices/") + dev, attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << attr << ": " << r.message;
  return o.size() > i ? o[i] : NAN;
}
void put(Solver& s, const char* dev, const char* attr, double v) {
  const double a[1] = {v};
  ASSERT_EQ(s.set(std::string("/devices/") + dev, attr, "", a).code, Result::ok);
}

// A rotor held in place (hinge only, no gravity), spinning at 300 rad/s, in calm air, with a rain device (film body = the rotor body).
std::string rotor_xml(bool rain) {
  return std::string(R"(<mujoco><option timestep="0.001" gravity="0 0 0"/><worldbody>
  <body name="rotor" pos="0 0 2"><joint name="shaft" type="hinge" axis="0 0 1"/>
  <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/></body></worldbody>
  <custom><text name="dvsim.devices" data='{
  "rotors": [{"name": "prop", "body": "rotor", "joint": "shaft", "diameter": 0.254, "j": [0, 0.4, 0.8], "ct": [0.12, 0.08, 0.02],
              "cq": [0.006, 0.005, 0.002]}],
  "environments": [{"name": "air"}])") +
         (rain ? R"(, "rains": [{"name": "rain", "body": "rotor", "catch_area": 0.05, "wet_area": 0.3}])" : "") + "}'/></custom></mujoco>";
}
const std::string kRotor = rotor_xml(true);

double thrust_at(double rate) {
  auto s = make(kRotor);
  put(*s, "rain", "rate", rate);
  const double w[1] = {300.0};
  EXPECT_EQ(s->set("/world/rotor/shaft", "qvel", "", w).code, Result::ok);
  EXPECT_EQ(s->step(1), 1);
  return get(*s, "prop", "thrust");
}
}  // namespace

TEST(rain, lwc_and_fall_speed_follow_the_closed_forms) {
  auto s = make(kRotor);
  for (const double r : {1.0, 25.0, 100.0}) {
    put(*s, "rain", "rate", r);
    ASSERT_EQ(s->step(1), 1);
    const double lwc = 0.072 * std::pow(r, 0.88);
    EXPECT_NEAR(get(*s, "rain", "lwc"), lwc, 1e-12);
    EXPECT_NEAR(get(*s, "rain", "v_fall") * lwc * 1e-3, r / 3600.0, 1e-12);  // water flux through a horizontal plane = R
  }
  put(*s, "rain", "rate", 0.0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "rain", "lwc"), 0.0);
  EXPECT_EQ(get(*s, "rain", "v_fall"), 0.0);
}

// Thrust loss grows with the rate, is negligible in drizzle and a few percent in a downpour (kt = 0.10 per 23 g/m^3 of LWC).
TEST(rain, rotor_thrust_loss_is_monotone_and_small_at_light_rain) {
  const double t0 = thrust_at(0.0);
  ASSERT_GT(t0, 1.0);
  double prev = 0.0;
  for (const double r : {1.0, 5.0, 25.0, 50.0, 100.0, 200.0}) {
    const double loss = 1.0 - thrust_at(r) / t0;
    EXPECT_GT(loss, prev) << "rate " << r;
    prev = loss;
    if (r == 1.0) EXPECT_LT(loss, 0.002);
    if (r == 100.0) {
      EXPECT_GT(loss, 0.01);
      EXPECT_LT(loss, 0.03);
    }
  }
}

// R = 0 is the dry model exactly: same bits with and without a rain device (the film also stays empty).
TEST(rain, dry_rain_device_changes_nothing) {
  auto a = make(rotor_xml(false)), b = make(kRotor);
  const double w[1] = {300.0};
  a->set("/world/rotor/shaft", "qvel", "", w);
  b->set("/world/rotor/shaft", "qvel", "", w);
  ASSERT_EQ(a->step(200), 200);
  ASSERT_EQ(b->step(200), 200);
  EXPECT_EQ(get(*a, "prop", "thrust"), get(*b, "prop", "thrust"));
  EXPECT_EQ(get(*b, "rain", "film_mass"), 0.0);
}

// The film fills towards min(cap, collect_eff LWC A |u| tau) and drains when the rain stops; its weight shows in `force`.
TEST(rain, film_mass_is_bounded_and_drains) {
  auto s = make(kRotor);
  put(*s, "rain", "rate", 100.0);
  ASSERT_EQ(s->step(60000), 60000);  // 60 s = 3 tau
  const double cap = 1000.0 * 3e-4 * 0.3;  // 0.09 kg
  const double m = get(*s, "rain", "film_mass");
  const double eq = 0.3 * get(*s, "rain", "lwc") * 1e-3 * 0.05 * get(*s, "rain", "v_fall") * 20.0;  // catch rate x tau
  EXPECT_NEAR(m, eq * (1.0 - std::exp(-3.0)), 0.01 * eq);
  EXPECT_LE(m, cap + 1e-12);
  EXPECT_NEAR(get(*s, "rain", "force", 2), -m * 9.80665 - get(*s, "rain", "lwc") * 1e-3 * 0.05 * std::pow(get(*s, "rain", "v_fall"), 2), 1e-9);
  put(*s, "rain", "rate", 0.0);
  ASSERT_EQ(s->step(60000), 60000);
  EXPECT_LT(get(*s, "rain", "film_mass"), 0.06 * m);
}

namespace {
// A lidar at (0, 0, 1) looking along +x at a wall 15 m ahead (max range 20 m); counts valid returns over many frames.
struct Stats { double valid_frac, near_frac, sigma; };
Stats lidar(double rate, double extra_det) {
  std::ostringstream x;
  x << R"(<mujoco><option timestep="0.001" gravity="0 0 0"/><worldbody>
    <geom name="wall" type="box" pos="15.5 0 1" size="0.5 20 20"/>
    <body name="sensor" pos="0 0 1"><site name="eye" xyaxes="0 0 1  0 -1 0"/></body></worldbody>
    <custom><text name="dvsim.devices" data='{"seed": 3,
    "rangefinders": [{"name": "lid", "site": "eye", "rate_hz": 1000, "projection": "spherical", "fov_h": 0.3, "fov_v": 0.3,
    "n_h": 16, "n_v": 16, "min_range": 0.1, "max_range": 20, "sigma0": 0.02, "seed": 5, "rain_det": )"
    << extra_det << R"(}],
    "rains": [{"name": "rain", "rate": )" << rate << R"(}]}'/></custom></mujoco>)";
  auto s = make(x.str());
  double valid = 0, near = 0, n = 0, s1 = 0, s2 = 0, nr = 0;
  for (int f = 0; f < 200; ++f) {
    s->step(1);
    std::vector<double> img;
    s->get("/devices/lid", "range", "", img);
    for (const double v : img) {
      ++n;
      if (v > 0) ++valid;
      if (v > 0 && v < 5.0) ++near;
      if (v > 14) { s1 += v; s2 += v * v; ++nr; }
    }
  }
  const double mean = s1 / nr;
  return {valid / n, near / n, std::sqrt(s2 / nr - mean * mean)};
}
}  // namespace

TEST(rain, rangefinder_dropout_noise_and_false_returns_rise_with_rate) {
  const Stats d0 = lidar(0.0, 0.01), d25 = lidar(25.0, 0.01), d100 = lidar(100.0, 0.01);
  EXPECT_EQ(d0.valid_frac, 1.0);  // dry: every ray returns, nothing near
  EXPECT_EQ(d0.near_frac, 0.0);
  EXPECT_LT(d25.valid_frac, d0.valid_frac);
  EXPECT_LT(d100.valid_frac, d25.valid_frac);
  EXPECT_GT(d100.valid_frac, 0.5);  // still a working sensor at 100 mm/h
  EXPECT_GT(d25.near_frac, 0.0);
  EXPECT_GT(d100.near_frac, d25.near_frac);
  EXPECT_GT(d100.sigma, d0.sigma);  // range noise rises with the loss
  const Stats off = lidar(100.0, 0.0);  // rain_det = 0: no false returns
  EXPECT_EQ(off.near_frac, 0.0);
}
