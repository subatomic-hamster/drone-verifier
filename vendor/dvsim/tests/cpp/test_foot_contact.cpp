// FootContact device (src/mujoco/devices/contact.cpp; docs/models.md "FootContact"). A ball on a flat hfield whose left half is
// covered by a flush box ("rock"): resting on the seam it touches two geoms at once. Expectations are independent sums of
// MuJoCo's own contact forces, read from the solver's mjData after each tick.
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "mujoco/mujoco_solver.h"

namespace {
struct Cfg {
  double mass = 1.0, z = 0.0505, f_max = 500, noise = 0, on = 2.0, off = 1.0, debounce = 0.005, rate = 1000;
  int seed = 1;
};

std::string xml(const Cfg& c) {
  std::ostringstream o;
  o << R"(<mujoco><option timestep="0.001"/><asset>
  <hfield name="h" nrow="3" ncol="3" size="1 1 0.1 0.1" elevation="0 0 0 0 0 0 0 0 0"/></asset><worldbody>
  <geom name="ground" type="hfield" hfield="h" contype="1" conaffinity="1"/>
  <geom name="rock" type="box" pos="-0.4 0 -0.05" size="0.5 0.5 0.05" contype="1" conaffinity="1"/>
  <body name="ball" pos="0.09 0 )" << c.z << R"("><freejoint name="fj"/>
  <geom name="foot" type="sphere" size="0.05" mass=")" << c.mass << R"(" contype="1" conaffinity="1"/></body></worldbody>
  <custom><text name="dvsim.devices" data='{
  "foot_contacts": [{"name": "f1", "parts": ["test-foot"], "geom": "foot", "mode": 0, "f_max": )" << c.f_max << R"(,
    "noise_std": )" << c.noise << R"(, "hysteresis_on": )" << c.on << R"(, "hysteresis_off": )" << c.off << R"(,
    "debounce_s": )" << c.debounce << R"(, "rate_hz": )" << c.rate << R"(, "seed": )" << c.seed << R"(}]
  }'/></custom></mujoco>)";
  return o.str();
}

std::unique_ptr<Solver> make(const Cfg& c = {}) {
  static int n = 0;
  const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_foot_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".xml")).string();
  std::ofstream(path) << xml(c);
  auto s = load_mujoco(path);
  std::remove(path.c_str());
  return s;
}
const MujocoSolver& ms(const Solver& s) { return static_cast<const MujocoSolver&>(s); }
double get(const Solver& s, const char* attr) {
  std::vector<double> o;
  const Result r = s.get("/devices/f1", attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << attr << ": " << r.message;
  return o.empty() ? NAN : o[0];
}
// Sum of the normal forces of every contact of the "foot" geom, straight from MuJoCo (and the partner geoms seen).
double truth(const Solver& s, std::set<std::string>* partners = nullptr) {
  const mjModel& m = ms(s).model();
  const mjData& d = ms(s).data();
  const int g = mj_name2id(&m, mjOBJ_GEOM, "foot");
  double sum = 0;
  for (int k = 0; k < d.ncon; ++k) {
    const mjContact& c = d.contact[k];
    if (c.geom1 != g && c.geom2 != g) continue;
    mjtNum f[6];
    mj_contactForce(&m, &d, k, f);
    sum += f[0];
    if (partners) partners->insert(mj_id2name(&m, mjOBJ_GEOM, c.geom1 == g ? c.geom2 : c.geom1));
  }
  return sum;
}
}  // namespace

// Review Focus 3: a foot on the rock/hfield seam touches both; the device sums them, and at rest the total is the weight.
TEST(foot_contact, sums_the_forces_of_all_contacts_of_the_geom) {
  auto s = make();
  ASSERT_EQ(s->step(1500), 1500);
  std::set<std::string> partners;
  const double sum = truth(*s, &partners);
  EXPECT_EQ(partners, (std::set<std::string>{"ground", "rock"}));
  EXPECT_NEAR(get(*s, "force"), sum, 1e-9 * sum);
  EXPECT_NEAR(sum, 1.0 * 9.81, 0.03 * 9.81) << "at rest the contact forces carry the weight";
  EXPECT_EQ(get(*s, "contact"), 1.0);
  // Each single contact is smaller than the total (the weight really is shared by more than one contact).
  const mjModel& m = ms(*s).model();
  const mjData& d = ms(*s).data();
  const int g = mj_name2id(&m, mjOBJ_GEOM, "foot");
  int n = 0;
  for (int k = 0; k < d.ncon; ++k) {
    if (d.contact[k].geom1 != g && d.contact[k].geom2 != g) continue;
    mjtNum f[6];
    mj_contactForce(&m, &d, k, f);
    EXPECT_LT(f[0], 0.99 * sum);
    ++n;
  }
  EXPECT_GE(n, 2);
}

// The reading saturates at f_max (a 5 kg ball weighs 49 N against f_max = 20 N).
TEST(foot_contact, saturates_at_f_max) {
  Cfg c;
  c.mass = 5;
  c.f_max = 20;
  auto s = make(c);
  ASSERT_EQ(s->step(1500), 1500);
  EXPECT_GT(truth(*s), 40.0);
  EXPECT_EQ(get(*s, "force"), 20.0);
  EXPECT_EQ(get(*s, "contact"), 1.0);
}

// Debounce timing: dropped from 1 cm, the comparator (force > on) first differs at tick t_on; the output follows once the
// difference has lasted debounce_s, i.e. at tick t_on + 5 (samples every tick). Thrown upward, the same for off.
TEST(foot_contact, hysteresis_and_debounce_timing) {
  Cfg c;
  c.z = 0.06;
  c.debounce = 0.005;
  auto s = make(c);
  int t_on = -1, t_on_out = -1;
  for (int t = 0; t < 400; ++t) {
    ASSERT_EQ(s->step(1), 1);
    if (t_on < 0 && truth(*s) > c.on) t_on = t;
    if (t_on_out < 0 && get(*s, "contact") > 0.5) t_on_out = t;
  }
  ASSERT_GE(t_on, 0);
  EXPECT_EQ(t_on_out, t_on + 5);
  // Settled; now throw the ball up: contact vanishes at once (force 0 <= off), the output drops debounce_s later.
  const double v[6] = {0, 0, 3.0, 0, 0, 0};
  ASSERT_EQ(s->set("/world/ball/fj", "qvel", "", v).code, Result::ok);
  int t_off = -1, t_off_out = -1;
  for (int t = 0; t < 100; ++t) {
    ASSERT_EQ(s->step(1), 1);
    if (t_off < 0 && truth(*s) <= c.off) t_off = t;
    if (t_off_out < 0 && get(*s, "contact") < 0.5) t_off_out = t;
  }
  ASSERT_GE(t_off, 0);
  EXPECT_EQ(t_off_out, t_off + 5);
  EXPECT_EQ(get(*s, "force"), 0.0);
}

// Samples are held between rate_hz periods.
TEST(foot_contact, held_between_samples) {
  Cfg c;
  c.rate = 100;  // every 10 ticks
  c.z = 0.06;
  auto s = make(c);
  double last = get(*s, "force");
  int changes = 0;
  for (int t = 0; t < 300; ++t) {
    ASSERT_EQ(s->step(1), 1);
    const double f = get(*s, "force");
    if (f != last) {
      ++changes;
      EXPECT_EQ((t) % 10, 0) << "force changed between samples at tick " << t;
    }
    last = f;
  }
  EXPECT_GE(changes, 2);
}

// Determinism: noise from the device's own seeded stream; equal seeds equal hashes, a restored run continues bitwise.
TEST(foot_contact, noise_is_seeded_and_replays) {
  Cfg c;
  c.noise = 0.5;
  auto a = make(c), b = make(c);
  ASSERT_EQ(a->step(600), 600);
  ASSERT_EQ(b->step(250), 250);
  const auto snap = b->save();
  const std::uint64_t h = b->state_hash();
  ASSERT_EQ(b->step(100), 100);
  ASSERT_NE(b->state_hash(), h);
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), h);
  ASSERT_EQ(b->step(350), 350);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->save(), b->save());
  Cfg d = c;
  d.seed = 2;
  auto o = make(d);
  ASSERT_EQ(o->step(600), 600);
  EXPECT_NE(o->state_hash(), a->state_hash());
  EXPECT_GT(std::abs(get(*a, "force") - truth(*a)), 0.0);  // the noise is really there
}

TEST(foot_contact, bad_config_fails_load) {
  Cfg c;
  c.on = 0.5;  // hysteresis_on < hysteresis_off
  EXPECT_THROW(make(c), std::exception);
  Cfg d;
  d.rate = 300;  // not a whole number of ticks
  EXPECT_THROW(make(d), std::exception);
}
