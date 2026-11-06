// Range sensor (depth camera / ToF / lidar, src/mujoco/devices/rangefinder.cpp). Expectations are plane geometry (a wall 5 m ahead,
// a floor 1.5 m below), the configured noise sigma, and the sampling schedule. Models are generated per test into a temp file.
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
#include <unistd.h>
#include <vector>

#include "mujoco/mujoco_solver.h"

namespace {
std::vector<double> getv(const Solver& s, const std::string& attr) {
  std::vector<double> o;
  const Result r = s.get("/devices/cam", attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << attr << ": " << r.message;
  return o;
}
double get(const Solver& s, const std::string& attr) {
  const auto o = getv(s, attr);
  return o.empty() ? NAN : o[0];
}

// Sensor at (0, 0, 1.0 or 1.5). dir "wall": looks along world +x at a wall whose face is at x = 5; "floor": looks down at the plane z = 0.
struct Cfg {
  std::string look = "wall", proj = "pinhole", extra;
  double rate = 100, sigma0 = 0, min_r = 0.1, max_r = 20, dropout = 0, res = 0;
  int nh = 8, nv = 6, wall_group = 0, seed = 1;
  unsigned groups = 63;
};
std::unique_ptr<Solver> make(const Cfg& c) {
  static int n = 0;
  const bool wall = c.look == "wall";
  std::ostringstream x;
  x << "<mujoco model=\"rf\"><option timestep=\"0.001\" gravity=\"0 0 0\"/><worldbody>"
    << "<geom name=\"floor\" type=\"plane\" pos=\"0 0 " << (wall ? -50 : 0) << "\" size=\"50 50 1\" group=\"" << (wall ? 5 : 0) << "\"/>"
    << "<geom name=\"wall\" type=\"box\" pos=\"5.5 0 1\" size=\"0.5 5 5\" group=\"" << c.wall_group << "\"/>"
    << "<body name=\"sensor\" pos=\"0 0 " << (wall ? 1.0 : 1.5) << "\"><site name=\"eye\" xyaxes=\""
    << (wall ? "0 0 1  0 -1 0" : "1 0 0  0 -1 0") << "\"/></body></worldbody><custom><text name=\"dvsim.devices\" data='{"
    << "\"rangefinders\": [{\"name\": \"cam\", \"site\": \"eye\", \"rate_hz\": " << c.rate << ", \"projection\": \"" << c.proj
    << "\", \"fov_h\": 1.0, \"fov_v\": 0.7, \"n_h\": " << c.nh << ", \"n_v\": " << c.nv << ", \"min_range\": " << c.min_r
    << ", \"max_range\": " << c.max_r << ", \"sigma0\": " << c.sigma0 << ", \"resolution\": " << c.res << ", \"dropout\": " << c.dropout
    << ", \"ray_groups\": " << c.groups << ", \"seed\": " << c.seed << c.extra << "}]}'/></custom></mujoco>";
  const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_rf_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".xml")).string();
  std::ofstream(path) << x.str();
  auto s = load_mujoco(path);
  std::remove(path.c_str());
  return s;
}
}  // namespace

// Pinhole depth is the distance along the optical axis, so every pixel of a flat wall reads exactly the wall distance; the same for
// the floor under a downward camera. Spherical range is the ray length: never shorter than the perpendicular distance, longer off-axis.
TEST(rangefinder, pinhole_depth_of_planes_and_spherical_range) {
  for (const auto& [look, dist] : {std::pair<const char*, double>{"wall", 5.0}, {"floor", 1.5}}) {
    Cfg c;
    c.look = look;
    auto s = make(c);
    ASSERT_EQ(s->step(1), 1);
    const auto img = getv(*s, "range");
    ASSERT_EQ(img.size(), 48u);
    for (double v : img) EXPECT_NEAR(v, dist, 1e-9) << look;
    EXPECT_EQ(get(*s, "valid"), 48.0);
    Cfg sp = c;
    sp.proj = "spherical";
    auto t = make(sp);
    ASSERT_EQ(t->step(1), 1);
    const auto r = getv(*t, "range");
    ASSERT_EQ(r.size(), 48u);
    const double lo = *std::min_element(r.begin(), r.end()), hi = *std::max_element(r.begin(), r.end());
    EXPECT_GE(lo, dist - 1e-9) << look;
    EXPECT_GT(hi, lo + 0.01 * dist) << look;
    EXPECT_LT(hi, dist / std::cos(0.5) / std::cos(0.35) + 1e-9) << look;  // the half-FOV corner bounds the grid
  }
}

// Gaussian depth noise: sample std matches sigma0 (and mean the true depth) over many pixels and frames; quantisation rounds to the
// resolution; dropout removes about the configured fraction.
TEST(rangefinder, noise_quantisation_and_dropout) {
  Cfg c;
  c.nh = 32;
  c.nv = 24;
  c.sigma0 = 0.02;
  auto s = make(c);
  double sum = 0, sq = 0;
  std::size_t n = 0;
  for (int f = 0; f < 6; ++f) {
    ASSERT_EQ(s->step(10), 10);
    for (double v : getv(*s, "range")) sum += v, sq += v * v, ++n;
  }
  const double mean = sum / static_cast<double>(n), sd = std::sqrt(sq / static_cast<double>(n) - mean * mean);
  EXPECT_NEAR(mean, 5.0, 0.003);
  EXPECT_NEAR(sd, 0.02, 0.002);
  Cfg q;
  q.proj = "spherical";  // pinhole depth = quantised range x cos, so quantise the range itself
  q.sigma0 = 0.02;
  q.res = 0.05;
  auto t = make(q);
  ASSERT_EQ(t->step(1), 1);
  for (double v : getv(*t, "range")) EXPECT_NEAR(v / 0.05, std::round(v / 0.05), 1e-9);
  Cfg d;
  d.nh = 40;
  d.nv = 30;
  d.dropout = 0.3;
  auto u = make(d);
  ASSERT_EQ(u->step(1), 1);
  EXPECT_NEAR(get(*u, "valid") / 1200.0, 0.7, 0.05);
}

// Range gating (outside [min, max] reads 0 = invalid) and the geom-group mask.
TEST(rangefinder, gating_and_ray_groups) {
  Cfg c;
  c.min_r = 6.0;
  auto near = make(c);
  ASSERT_EQ(near->step(1), 1);
  EXPECT_EQ(get(*near, "valid"), 0.0);
  for (double v : getv(*near, "range")) EXPECT_EQ(v, 0.0);
  Cfg f;
  f.max_r = 4.0;
  auto far = make(f);
  ASSERT_EQ(far->step(1), 1);
  EXPECT_EQ(get(*far, "valid"), 0.0);
  Cfg g;
  g.wall_group = 3;
  g.groups = 7;  // groups 0..2 only: the wall (group 3) is invisible
  auto blind = make(g);
  ASSERT_EQ(blind->step(1), 1);
  EXPECT_EQ(get(*blind, "valid"), 0.0);
  g.groups = 15;
  auto seen = make(g);
  ASSERT_EQ(seen->step(1), 1);
  EXPECT_EQ(get(*seen, "valid"), 48.0);
}

// Schedule: the first tick samples, then every 1/rate; the image is held in between.
TEST(rangefinder, frame_rate_and_hold) {
  Cfg c;
  c.sigma0 = 0.01;
  auto s = make(c);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "frames"), 1.0);
  const auto first = getv(*s, "range");
  ASSERT_EQ(s->step(9), 9);
  EXPECT_EQ(get(*s, "frames"), 1.0);
  EXPECT_EQ(getv(*s, "range"), first);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "frames"), 2.0);
  EXPECT_NE(getv(*s, "range"), first);
  ASSERT_EQ(s->step(90), 90);
  EXPECT_EQ(get(*s, "frames"), 11.0);  // 101 ticks: samples at ticks 0, 10, ..., 100
}

// Same seed, same images; another seed, other noise; save / restore / clone continue bitwise (RNG and tick counter in the state).
TEST(rangefinder, deterministic_and_restorable) {
  Cfg c;
  c.sigma0 = 0.02;
  c.dropout = 0.1;
  auto a = make(c), b = make(c);
  Cfg o = c;
  o.seed = 2;
  auto d = make(o);
  for (auto* s : {a.get(), b.get(), d.get()}) ASSERT_EQ(s->step(25), 25);
  EXPECT_EQ(getv(*a, "range"), getv(*b, "range"));
  EXPECT_NE(getv(*a, "range"), getv(*d, "range"));
  EXPECT_EQ(a->state_hash(), b->state_hash());
  const auto snap = b->save();
  auto cl = b->clone();
  ASSERT_EQ(b->step(20), 20);
  ASSERT_NE(b->state_hash(), a->state_hash());
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), a->state_hash());
  for (auto* s : {a.get(), b.get(), cl.get()}) ASSERT_EQ(s->step(33), 33);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->state_hash(), cl->state_hash());
  EXPECT_EQ(getv(*a, "range"), getv(*cl, "range"));
}
