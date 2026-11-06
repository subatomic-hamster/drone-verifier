#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/world.h"
#include "mujoco/mujoco_solver.h"

namespace {
// Pure function of (seed, tick, actuator index): counter-based splitmix64 -> [-1, 1].
double ctrl_value(std::uint64_t seed, std::uint64_t tick, int i) {
  std::uint64_t z = seed * 0x9E3779B97F4A7C15ull + tick * 0xD1B54A32D192ED03ull + static_cast<std::uint64_t>(i) + 1;
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

void run_to(World& w, std::uint64_t seed, std::uint64_t to) {
  std::vector<std::string> acts;  // actuator prims, in registry (= actuator index) order
  for (const Prim& p : w.solver().prims())
    if (p.kind == "actuator") acts.push_back(p.path);
  while (w.tick() < to) {
    for (std::size_t i = 0; i < acts.size(); ++i) {
      const double u = ctrl_value(seed, w.tick(), static_cast<int>(i));
      ASSERT_EQ(w.solver().set(acts[i], "ctrl", "", std::span<const double>(&u, 1)).code, Result::ok);
    }
    w.step(1);
  }
}

World make() { return World(load_mujoco, DVSIM_HUMANOID_XML); }

std::vector<double> qpos(const World& w) {
  const mjData& d = static_cast<const MujocoSolver&>(w.solver()).data();
  const mjModel& m = static_cast<const MujocoSolver&>(w.solver()).model();
  return {d.qpos, d.qpos + m.nq};
}

constexpr std::uint64_t kSeed = 42, kEnd = 10000, kMid = 5000;
}  // namespace

TEST(determinism, two_worlds_equal) {
  World a = make(), b = make();
  for (std::uint64_t t = 1000; t <= kEnd; t += 1000) {
    run_to(a, kSeed, t);
    run_to(b, kSeed, t);
    ASSERT_EQ(a.state_hash(), b.state_hash()) << "tick " << t;
  }
  EXPECT_EQ(a.save(), b.save());
  EXPECT_FALSE(a.diverged());
}

TEST(determinism, humanoid_moves_and_seed_matters) {
  World a = make(), other = make();
  const std::vector<double> q0 = qpos(a);
  const std::uint64_t h0 = a.state_hash();
  run_to(a, kSeed, kEnd);
  run_to(other, kSeed + 1, kEnd);
  EXPECT_NE(qpos(a), q0);
  EXPECT_NE(a.state_hash(), h0);
  EXPECT_NE(a.state_hash(), other.state_hash());
  // Contact happened: the root must have dropped/moved by more than numerical noise.
  double maxdiff = 0;
  for (std::size_t i = 0; i < q0.size(); ++i) maxdiff = std::max(maxdiff, std::abs(qpos(a)[i] - q0[i]));
  EXPECT_GT(maxdiff, 0.1);
}

TEST(determinism, restore_continues_bitwise) {
  World a = make();
  run_to(a, kSeed, kEnd);

  World b = make();
  run_to(b, kSeed, kMid);
  const std::vector<std::byte> snap = b.save();

  World c = make();  // freshly loaded
  c.restore(snap);
  EXPECT_EQ(c.tick(), kMid);
  run_to(c, kSeed, kEnd);
  EXPECT_EQ(c.save(), a.save());
  EXPECT_EQ(c.state_hash(), a.state_hash());

  run_to(b, kSeed, 7000);  // restore into the same world after it moved on
  b.restore(snap);
  EXPECT_EQ(b.tick(), kMid);
  run_to(b, kSeed, kEnd);
  EXPECT_EQ(b.save(), a.save());
}

TEST(determinism, clone_continues_bitwise) {
  World a = make();
  run_to(a, kSeed, kEnd);

  World b = make();
  run_to(b, kSeed, kMid);
  World c = b.clone();
  EXPECT_EQ(c.tick(), kMid);
  EXPECT_EQ(c.save(), b.save());
  run_to(c, kSeed, kEnd);
  EXPECT_EQ(c.save(), a.save());
  run_to(b, kSeed, kEnd);  // original unaffected by the clone
  EXPECT_EQ(b.save(), a.save());
}

TEST(determinism, load_errors_throw_with_message) {
  try {
    World w(load_mujoco, "/nonexistent/model.xml");
    FAIL() << "expected throw";
  } catch (const std::runtime_error& e) {
    std::cout << "[ load error ] " << e.what() << "\n";
    EXPECT_NE(std::string(e.what()).find("/nonexistent/model.xml"), std::string::npos);
  }
  World w = make();
  EXPECT_THROW(w.restore(std::vector<std::byte>(3)), std::runtime_error);
  EXPECT_THROW(w.restore(std::vector<std::byte>(64)), std::runtime_error);
}

// Proves mju_error -> C++ throw unwinds through MuJoCo's C frames (needs -fexceptions on libmujoco).
TEST(determinism, mju_error_throws_through_c_frames) {
  install_mujoco_handlers();
  char err[1000] = "";
  mjModel* m = mj_loadXML(DVSIM_HUMANOID_XML, nullptr, err, sizeof err);
  ASSERT_NE(m, nullptr) << err;
  m->narena = 64;  // arena too small for mj_forward's stack allocations
  mjData* d = mj_makeData(m);
  ASSERT_NE(d, nullptr);
  EXPECT_THROW(mj_forward(m, d), std::runtime_error);
  mj_deleteData(d);
  mj_deleteModel(m);
}
