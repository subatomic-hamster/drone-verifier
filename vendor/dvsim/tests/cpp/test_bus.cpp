// Bus tests, driven through the agent-facing JSON edge (execute_json).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "core/world.h"
#include "mujoco/mujoco_solver.h"

namespace {
const std::string kData = DVSIM_TEST_DATA_DIR;
const std::string kPole = "/world/pole", kHinge = "/world/pole/hinge";

Session session(std::vector<std::string> caps, int max_steps = 100000) {
  Session s;
  s.caps = std::move(caps);
  s.max_steps_per_call = max_steps;
  s.root = kData;
  return s;
}
const Session kRC = session({"read", "control"});
const Session kAll = session({"read", "control", "destructive"});

World pendulum() { return World(load_mujoco, kData + "/pendulum.xml"); }

std::string q(const std::string& cmd_type, const std::string& cmd) {
  return R"({"id":7,"source":"cli","cmd_type":")" + cmd_type + R"(","cmd":)" + cmd + "}";
}
std::string query(const std::string& path, const std::string& attr, const std::string& frame = "") {
  return q("Query", R"({"path":")" + path + R"(","attr":")" + attr + R"(","frame":")" + frame + R"("})");
}
std::string step(int n) { return q("Step", R"({"n":)" + std::to_string(n) + "}"); }
std::string load(const std::string& path) { return q("LoadModel", R"({"path":")" + path + R"("})"); }

// Raw text of a top-level-ish JSON field value ("key": value), tolerant of whitespace. Arrays include brackets.
std::string raw(const std::string& j, const std::string& key) {
  std::size_t p = j.find("\"" + key + "\"");
  if (p == std::string::npos) return "";
  p = j.find(':', p) + 1;
  while (p < j.size() && j[p] == ' ') ++p;
  std::size_t e = p;
  if (j[p] == '"') {
    e = j.find('"', p + 1);
    return j.substr(p + 1, e - p - 1);  // strings come back unquoted
  } else if (j[p] == '[') {
    e = j.find(']', p) + 1;
  } else {
    while (e < j.size() && j[e] != ',' && j[e] != '}' && j[e] != ' ') ++e;
  }
  return j.substr(p, e - p);
}
bool ok(const std::string& r) { return raw(r, "ok") == "true"; }
std::string error(const std::string& r) { return raw(r, "error"); }
std::vector<double> values(const std::string& r) {
  std::vector<double> v;
  const std::string a = raw(r, "values");
  for (std::size_t p = a.find('['); p != std::string::npos && p + 1 < a.size();) {
    ++p;
    char* end = nullptr;
    const double x = std::strtod(a.c_str() + p, &end);
    if (end == a.c_str() + p) break;
    v.push_back(x);
    p = a.find(',', p);
  }
  return v;
}
}  // namespace

TEST(bus, query_pos_three_values_in_meters) {
  World w = pendulum();
  const std::string req = query(kPole, "pos"), rep = execute_json(w, req, kRC);
  std::cout << "[ example ] request: " << req << "\n[ example ] reply:   " << rep << "\n";
  ASSERT_TRUE(ok(rep)) << rep;
  EXPECT_EQ(raw(rep, "id"), "7");
  EXPECT_EQ(raw(rep, "unit"), "m");
  const std::vector<double> v = values(rep);
  ASSERT_EQ(v.size(), 3u) << rep;
  EXPECT_DOUBLE_EQ(v[2], 1.0);  // pole body sits at z=1
}

TEST(bus, query_empty_path_lists_prims) {
  World w = pendulum();
  const std::string rep = execute_json(w, query("", ""), kRC);
  ASSERT_TRUE(ok(rep)) << rep;
  for (const char* p : {"/world/pole", "/world/pole/hinge", "/actuators/torque", "/sensors/angle"})
    EXPECT_NE(rep.find(p), std::string::npos) << p << " in " << rep;
}

TEST(bus, unknown_field_is_rejected_naming_it) {
  World w = pendulum();
  const std::string rep = execute_json(
      w, q("Query", R"({"path":"/world/pole","attr":"pos","bogus_field":1})"), kRC);
  std::cout << "[ unknown field ] " << rep << "\n";
  EXPECT_FALSE(ok(rep));
  EXPECT_NE(rep.find("bogus_field"), std::string::npos) << rep;
  const std::string top = execute_json(w, R"({"id":1,"nonsense":3})", kRC);  // envelope-level too
  EXPECT_FALSE(ok(top));
  EXPECT_NE(top.find("nonsense"), std::string::npos) << top;
}

TEST(bus, setattr_then_step_changes_qpos) {
  World w = pendulum();
  ASSERT_TRUE(ok(execute_json(w, q("SetAttr", R"({"path":")" + kHinge + R"(","attr":"qpos","values":[1.0]})"), kRC)));
  const double before = values(execute_json(w, query(kHinge, "qpos"), kRC)).at(0);
  EXPECT_DOUBLE_EQ(before, 1.0);
  const std::string st = execute_json(w, step(200), kRC);
  ASSERT_TRUE(ok(st)) << st;
  EXPECT_EQ(raw(st, "tick"), "200");
  const std::string rep = execute_json(w, query(kHinge, "qpos"), kRC);
  EXPECT_EQ(raw(rep, "unit"), "rad");
  const double after = values(rep).at(0);
  EXPECT_GT(std::abs(after - before), 0.1) << before << " -> " << after;  // pendulum swings down

  // actuator ctrl round-trips and drives the joint
  World w2 = pendulum();
  ASSERT_TRUE(ok(execute_json(w2, q("SetAttr", R"({"path":"/actuators/torque","attr":"ctrl","values":[5.0]})"), kRC)));
  EXPECT_DOUBLE_EQ(values(execute_json(w2, query("/actuators/torque", "ctrl"), kRC)).at(0), 5.0);
  execute_json(w2, step(100), kRC);
  EXPECT_GT(std::abs(values(execute_json(w2, query(kHinge, "qpos"), kRC)).at(0)), 1e-3);
}

TEST(bus, snapshot_restore_roundtrip) {
  World w = pendulum();
  execute_json(w, q("SetAttr", R"({"path":")" + kHinge + R"(","attr":"qpos","values":[1.0]})"), kRC);
  const std::string snap = execute_json(w, q("Snapshot", "{}"), kRC);
  ASSERT_TRUE(ok(snap)) << snap;
  const std::uint64_t h = w.state_hash();
  execute_json(w, step(50), kRC);
  EXPECT_NE(w.state_hash(), h);
  const std::string rep = execute_json(w, q("Restore", R"({"id":)" + raw(snap, "snapshot_id") + "}"), kRC);
  ASSERT_TRUE(ok(rep)) << rep;
  EXPECT_EQ(w.state_hash(), h);
  EXPECT_EQ(w.tick(), 0u);
}

TEST(bus, destructive_needs_destructive_cap) {
  World w = pendulum();
  const std::string req = load("two_link.urdf");
  const std::string denied = execute_json(w, req, kRC);
  std::cout << "[ denied ] " << denied << "\n";
  EXPECT_FALSE(ok(denied));
  EXPECT_EQ(error(denied), "permission_denied");
  EXPECT_NE(denied.find("LoadModel"), std::string::npos);
  EXPECT_NE(denied.find("destructive"), std::string::npos);
  EXPECT_EQ(w.solver().prims().size(), pendulum().solver().prims().size());  // untouched

  const std::string granted = execute_json(w, req, kAll);
  EXPECT_TRUE(ok(granted)) << granted;
}

TEST(bus, load_urdf_has_two_joint_prims) {
  World w = pendulum();
  w.step(10);
  ASSERT_TRUE(ok(execute_json(w, load("two_link.urdf"), kAll)));
  EXPECT_EQ(w.tick(), 0u);  // tick reset by LoadModel
  int joints = 0;
  for (const Prim& p : w.solver().prims()) {
    std::cout << "[ prim ] " << p.kind << " " << p.path << "\n";
    joints += p.kind == "joint";
  }
  EXPECT_EQ(joints, 2);
}

TEST(bus, error_paths) {
  World w = pendulum();
  const Session small = session({"read", "control", "destructive"}, 1000);

  const std::string lim = execute_json(w, step(1001), small);
  EXPECT_FALSE(ok(lim));
  EXPECT_EQ(error(lim), "limit_exceeded") << lim;
  EXPECT_TRUE(ok(execute_json(w, step(1000), small)));

  for (const char* bad : {"../../etc/passwd", "/etc/passwd", "../tests/data/pendulum.xml"}) {
    const std::string rep = execute_json(w, load(bad), kAll);
    EXPECT_EQ(error(rep), "permission_denied") << bad << " -> " << rep;
  }

  EXPECT_EQ(error(execute_json(w, query("/world/nope", "pos"), kRC)), "not_found");
  EXPECT_EQ(error(execute_json(w, query(kPole, "nope"), kRC)), "not_found");
  EXPECT_EQ(error(execute_json(w, q("SetAttr", R"({"path":")" + kHinge + R"(","attr":"qpos","values":[1,2]})"), kRC)),
            "invalid_argument");
  EXPECT_EQ(error(execute_json(w, q("SetAttr", R"({"path":")" + kPole + R"(","attr":"pos","values":[1,2,3]})"), kRC)),
            "invalid_argument");  // read-only
  EXPECT_EQ(error(execute_json(w, load("does_not_exist.xml"), kAll)), "solver_error");
  EXPECT_EQ(error(execute_json(w, query(kPole, "pos", "/world/nope"), kRC)), "not_found");
}

TEST(bus, frame_relative_body_pos) {
  World w = pendulum();
  for (double x : values(execute_json(w, query(kPole, "pos", kPole), kRC))) EXPECT_NEAR(x, 0.0, 1e-12);

  // two_link: link2 sits 0.5 m along link1's z axis whatever the shoulder angle; in world it does not.
  ASSERT_TRUE(ok(execute_json(w, load("two_link.urdf"), kAll)));
  std::string l1, l2, shoulder;
  for (const Prim& p : w.solver().prims()) {
    if (p.path.ends_with("/link1")) l1 = p.path;
    if (p.path.ends_with("/link2")) l2 = p.path;
    if (p.path.ends_with("/shoulder")) shoulder = p.path;
  }
  ASSERT_FALSE(l1.empty() || l2.empty() || shoulder.empty());
  ASSERT_TRUE(ok(execute_json(w, q("SetAttr", R"({"path":")" + shoulder + R"(","attr":"qpos","values":[0.7]})"), kRC)));
  const std::vector<double> rel = values(execute_json(w, query(l2, "pos", l1), kRC));
  ASSERT_EQ(rel.size(), 3u);
  EXPECT_NEAR(rel[0], 0.0, 1e-9);
  EXPECT_NEAR(rel[1], 0.0, 1e-9);
  EXPECT_NEAR(rel[2], 0.5, 1e-9);
  const std::vector<double> p1 = values(execute_json(w, query(l1, "pos"), kRC));
  const std::vector<double> p2 = values(execute_json(w, query(l2, "pos"), kRC));
  EXPECT_NEAR(p2[0] - p1[0], 0.5 * std::sin(0.7), 1e-9);  // world frame is rotated
  EXPECT_NEAR(p2[2] - p1[2], 0.5 * std::cos(0.7), 1e-9);
}
