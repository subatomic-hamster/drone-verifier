// Rangefinder "rosette" projection (Livox Mid-360-like, src/mujoco/devices/rangefinder.cpp; docs/models.md "Rosette scan pattern").
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "mujoco/devices/devices.h"
#include "mujoco/mujoco_solver.h"

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

std::string lidar_json(int rays, double rate = 10, unsigned groups = 63, int seed = 1, double max_range = 100) {
  std::ostringstream o;
  o.precision(17);
  o << R"("rangefinders": [{"name": "cam", "site": "s", "rate_hz": )" << rate << R"(, "projection": "rosette", "fov_h": )" << 2 * kPi
    << R"(, "fov_v_min": )" << -7 * kDeg << R"(, "fov_v_max": )" << 52 * kDeg << R"(, "rays_per_frame": )" << rays
    << R"(, "min_range": 0.1, "max_range": )" << max_range << R"(, "ray_groups": )" << groups << R"(, "seed": 1, "pattern_seed": )" << seed << "}]";
  return o.str();
}
// Lidar at z = 1 with site +x pointing up (the spherical frame's elevation axis), a floor and optionally a ceiling at z = 4, and a
// torso sphere (group 4, the robot's own geom) enclosing the sensor.
std::string model(const std::string& dev, bool ceiling, bool floor_, bool torso, const std::string& extra_assets = "", const std::string& hf = "") {
  std::ostringstream o;
  o << "<mujoco><option timestep=\"0.001\" gravity=\"0 0 0\"/><asset>" << extra_assets << "</asset><worldbody>"
    << (floor_ ? "<geom name=\"floor\" type=\"plane\" size=\"500 500 0.1\" contype=\"0\" conaffinity=\"0\"/>" : "")
    << (ceiling ? "<geom name=\"ceil\" type=\"plane\" pos=\"0 0 4\" euler=\"180 0 0\" size=\"500 500 0.1\" contype=\"0\" conaffinity=\"0\"/>" : "") << hf
    << (torso ? "<body name=\"torso\" pos=\"0 0 1\"><geom type=\"sphere\" size=\"0.3\" group=\"4\" contype=\"0\" conaffinity=\"0\"/></body>" : "")
    << "<body name=\"head\" pos=\"0 0 1\"><joint type=\"slide\" axis=\"1 0 0\"/><geom type=\"sphere\" size=\"0.02\" mass=\"1\" group=\"4\" contype=\"0\" conaffinity=\"0\"/>"
       "<site name=\"s\" euler=\"0 -90 0\"/></body></worldbody>"
    << "<custom><text name=\"dvsim.devices\" data='{" << dev << "}'/></custom></mujoco>";
  return o.str();
}
std::unique_ptr<Solver> make(const std::string& xml) {
  static int n = 0;
  const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_ros_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".xml")).string();
  std::ofstream(path) << xml;
  auto s = load_mujoco(path);
  std::remove(path.c_str());
  return s;
}
std::vector<double> image(const Solver& s) {
  std::vector<double> o;
  EXPECT_EQ(s.get("/devices/cam", "range", "", o).code, Result::ok);
  return o;
}
double frames(const Solver& s) {
  std::vector<double> o;
  EXPECT_EQ(s.get("/devices/cam", "frames", "", o).code, Result::ok);
  return o[0];
}
devs::RangefinderCfg cfg_of(int rays, int seed = 1) {
  devs::Devices dv = devs::parse("{" + lidar_json(rays, 10, 63, seed) + "}");
  return dv.range.cfg.at(0);
}
}  // namespace

// All generated directions lie in the field of view, and the image is 1 x rays_per_frame.
TEST(rosette, rays_stay_inside_the_fov) {
  const auto c = cfg_of(2000);
  EXPECT_EQ(c.n_h, 2000);
  EXPECT_EQ(c.n_v, 1);
  for (std::uint64_t m = 0; m < 200000; ++m) {
    double d[3];
    devs::rosette_dir(c, m, d);
    EXPECT_NEAR(d[0] * d[0] + d[1] * d[1] + d[2] * d[2], 1.0, 1e-12);
    const double el = std::asin(d[0]), az = std::atan2(d[1], d[2]);
    ASSERT_GE(el, -7 * kDeg - 1e-12) << m;
    ASSERT_LE(el, 52 * kDeg + 1e-12) << m;
    ASSERT_GE(az, -kPi - 1e-12);
    ASSERT_LE(az, kPi + 1e-12);
  }
}

// Coverage after 1 s (10 frames of 20000 rays = the Mid-360's 200 000 points/s) >= 95 % of the 1 x 1 degree bins, and the
// frames do not repeat: the first frame alone covers little, and no direction is generated twice.
TEST(rosette, covers_the_fov_after_one_second_without_repeating) {
  const auto c = cfg_of(20000);
  const int naz = 360, nel = 59;
  auto cover = [&](std::uint64_t n_rays) {
    std::set<int> bins;
    for (std::uint64_t m = 0; m < n_rays; ++m) {
      double d[3];
      devs::rosette_dir(c, m, d);
      const int a = std::clamp(static_cast<int>(std::floor((std::atan2(d[1], d[2]) + kPi) / kDeg)), 0, naz - 1);
      const int e = std::clamp(static_cast<int>(std::floor((std::asin(d[0]) / kDeg) + 7.0)), 0, nel - 1);
      bins.insert(a * nel + e);
    }
    return static_cast<double>(bins.size()) / (naz * nel);
  };
  const double one = cover(20000), ten = cover(200000);
  std::printf("[rosette] 1x1 deg bin coverage: 1 frame (0.1 s) %.1f %%, 10 frames (1 s) %.1f %%\n", 100 * one, 100 * ten);
  EXPECT_GE(ten, 0.95);
  EXPECT_LT(one, ten);
  double a[3], b[3];
  devs::rosette_dir(c, 5, a);
  devs::rosette_dir(c, 5 + 20000, b);  // the same slot of the next frame
  EXPECT_GT(std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]) + std::abs(a[2] - b[2]), 1e-6);
  // pattern_seed moves the pattern
  const auto c2 = cfg_of(20000, 2);
  devs::rosette_dir(c2, 5, b);
  devs::rosette_dir(c, 5, a);
  EXPECT_GT(std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]) + std::abs(a[2] - b[2]), 1e-6);
}

// Through the solver, against analytic planes: over a floor only the rays below the horizon (-7..0 deg, 7/59 of them) return, each at
// range >= 1 / sin(7 deg); under a ceiling 3 m above the sensor every ray above the horizon returns at range >= 3 / sin(52 deg).
TEST(rosette, ranges_against_a_floor_and_a_ceiling_match_the_fov) {
  {
    auto s = make(model(lidar_json(20000), false, true, false));
    ASSERT_EQ(s->step(1), 1);
    const auto img = image(*s);
    ASSERT_EQ(img.size(), 20000u);
    int valid = 0;
    for (const double r : img)
      if (r > 0) {
        ++valid;
        ASSERT_GE(r, 1.0 / std::sin(7 * kDeg) - 1e-9);
      }
    EXPECT_NEAR(static_cast<double>(valid) / 20000, 7.0 / 59.0, 0.012) << "fraction of rays below the horizon (those inside 1 / sin(e) < 100)";
  }
  {
    auto s = make(model(lidar_json(20000), true, false, false));
    ASSERT_EQ(s->step(1), 1);
    for (const double r : image(*s))
      if (r > 0) ASSERT_GE(r, 3.0 / std::sin(52 * kDeg) - 1e-9);
  }
}

// A later frame continues the sequence (different image), and an identical model replays the same image: determinism + restore.
TEST(rosette, frames_advance_and_replay_bitwise) {
  const std::string xml = model(lidar_json(2000), true, true, false);
  auto a = make(xml), b = make(xml);
  ASSERT_EQ(a->step(1), 1);
  const auto f0 = image(*a);
  ASSERT_EQ(a->step(100), 100);  // next sample: frame 1
  EXPECT_EQ(frames(*a), 2.0);
  EXPECT_NE(f0, image(*a));
  ASSERT_EQ(b->step(150), 150);
  const auto snap = b->save();
  ASSERT_EQ(b->step(100), 100);
  b->restore(snap);  // back at tick 150
  ASSERT_EQ(a->step(400 - 101), 400 - 101);  // a: 101 ticks so far
  ASSERT_EQ(b->step(400 - 150), 400 - 150);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(image(*a), image(*b));
}

// Review Focus 4: rays start inside the robot's own torso geom (group 4). Unmasked they all return the torso (0.3 m); with the
// group mask 63 & ~16 they never see it.
TEST(rosette, self_hits_are_excluded_by_the_group_mask) {
  {
    auto s = make(model(lidar_json(3000, 10, 63), false, true, true));
    ASSERT_EQ(s->step(1), 1);
    const auto img = image(*s);
    int torso = 0;
    for (const double r : img) torso += std::abs(r - 0.3) < 1e-6;
    EXPECT_GT(torso, 2900) << "unmasked: the torso is hit from inside";
  }
  {
    auto s = make(model(lidar_json(3000, 10, 63 & ~16u), false, true, true));
    ASSERT_EQ(s->step(1), 1);
    for (const double r : image(*s)) ASSERT_TRUE(r == 0.0 || r > 8.0) << r;
  }
}

// Cost: rays per second of the rosette on a 2001 x 2001 heightfield (100 m at 5 cm, rolling relief), the whole step including physics.
TEST(rosette, cost_on_a_2001_squared_hfield) {
  const int n = 2001;
  const std::string bin = (std::filesystem::temp_directory_path() / ("dvsim_ros_hf_" + std::to_string(::getpid()) + ".bin")).string();
  {
    std::vector<float> h(static_cast<std::size_t>(n) * n);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c)
        h[static_cast<std::size_t>(r) * n + c] = static_cast<float>(0.5 + 0.25 * std::sin(r * 0.05) * std::cos(c * 0.07) + 0.1 * std::sin(r * 0.9 + c * 0.6));
    std::ofstream f(bin, std::ios::binary);
    const int32_t dims[2] = {n, n};
    f.write(reinterpret_cast<const char*>(dims), sizeof dims);
    f.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * sizeof(float)));
  }
  const std::string assets = "<hfield name=\"hf\" file=\"" + bin + "\" size=\"50 50 1 0.1\"/>";
  const std::string hf = "<geom name=\"terr\" type=\"hfield\" hfield=\"hf\" pos=\"0 0 0\" contype=\"0\" conaffinity=\"0\"/>";
  for (const double max_range : {100.0, 40.0, 20.0}) {
    const int rays = 5000;
    auto s = make(model(lidar_json(rays, 10, 63, 1, max_range), false, false, false, assets, hf));
    ASSERT_EQ(s->step(1), 1);  // frame 0 (includes any lazy setup)
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_EQ(s->step(300), 300);  // 3 frames at 10 Hz
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[rosette cost] %d rays/frame, max_range %.0f m, 2001^2 hfield: %.0f rays/s (%.2f s wall per sim-second at 10 Hz)\n", rays,
                max_range, rays * 3 / dt, dt / 0.3);
  }
  std::remove(bin.c_str());
}
