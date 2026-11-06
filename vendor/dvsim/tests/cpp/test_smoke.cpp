#include <mujoco/mujoco.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

#include "mujoco/mujoco_solver.h"

// The design builder names a cable's tendon "wire:<i>:<c>|<end a>|<end b>": the viewer hides a cable by those instance ids.
TEST(smoke, tubes_carry_their_end_instances) {
  const std::string xml = R"(<mujoco><worldbody>
    <body name="a"><geom size=".05"/><site name="s0" pos="0 0 .1"/></body>
    <body name="b" pos="1 0 0"><geom size=".05"/><site name="s1" pos="0 0 .1"/></body>
    </worldbody><tendon>
    <spatial name="wire:3:0|esc 1|motor-2" width=".002" rgba="1 0 0 1"><site site="s0"/><site site="s1"/></spatial>
    <spatial name="plain" width=".002" rgba="0 1 0 1"><site site="s0"/><site site="s1"/></spatial>
    </tendon></mujoco>)";
  const auto path = std::filesystem::temp_directory_path() / "dvsim_tube_ends.xml";
  std::ofstream(path) << xml;
  const auto s = load_mujoco(path.string());
  const auto e = s->tube_ends();
  ASSERT_EQ(s->tubes().size(), 2u);
  ASSERT_EQ(e.size(), 2u);  // index-parallel to tubes()
  EXPECT_EQ(e[0].first, "esc 1");
  EXPECT_EQ(e[0].second, "motor-2");
  EXPECT_TRUE(e[1].first.empty() && e[1].second.empty());  // not a design cable: no ends known, never hidden
}

TEST(smoke, pendulum_steps_finite) {
  char err[1000] = "";
  mjModel* m = mj_loadXML(DVSIM_TEST_DATA_DIR "/pendulum.xml", nullptr, err, sizeof err);
  ASSERT_NE(m, nullptr) << err;
  mjData* d = mj_makeData(m);
  ASSERT_NE(d, nullptr);
  d->qpos[0] = 1.0;  // start off-equilibrium so it actually moves
  for (int i = 0; i < 1000; ++i) mj_step(m, d);
  for (int i = 0; i < m->nq; ++i) EXPECT_TRUE(std::isfinite(d->qpos[i])) << "qpos[" << i << "]";
  EXPECT_NE(d->qpos[0], 1.0);
  mj_deleteData(d);
  mj_deleteModel(m);
}
