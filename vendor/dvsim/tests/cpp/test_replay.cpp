// Replay-log tests: record a seeded random session, replay it in-process and through the `dvsim-replay` exe,
// re-record byte-identically, and check that damaged / altered logs fail loudly (or change the hash).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "core/world.h"
#include "mujoco/mujoco_solver.h"

namespace fs = std::filesystem;

namespace {
const std::string kData = DVSIM_TEST_DATA_DIR;
const std::string kHinge = "/world/pole/hinge";
using Bytes = std::vector<std::byte>;

Session session(std::vector<std::string> caps) {
  Session s;
  s.caps = std::move(caps);
  s.root = kData;
  return s;
}
const Session kRC = session({"read", "control"});
const Session kAll = session({"read", "control", "destructive"});

World pendulum() { return World(load_mujoco, kData + "/pendulum.xml"); }
World replay_of(const Bytes& log, bool record = false, const std::string& root = kData) {
  return replay(log, load_mujoco_bytes, load_mujoco, root, record);
}

std::string num(double v) {
  char b[40];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}
std::string cmd(const std::string& type, const std::string& body) {
  return R"({"id":1,"source":"test","cmd_type":")" + type + R"(","cmd":)" + body + "}";
}
std::string setattr(const std::string& path, const std::string& attr, const std::string& values) {
  return cmd("SetAttr", R"({"path":")" + path + R"(","attr":")" + attr + R"(","values":[)" + values + "]}");
}
std::string step(int n) { return cmd("Step", R"({"n":)" + std::to_string(n) + "}"); }
std::string query(const std::string& path, const std::string& attr) {
  return cmd("Query", R"({"path":")" + path + R"(","attr":")" + attr + R"("})");
}

// Walks the log: number of records after the header (also validates the layout described in replay.h).
std::size_t count_records(const Bytes& b) {
  auto u = [&](std::size_t p, std::size_t n) { std::uint64_t v = 0; std::memcpy(&v, b.data() + p, n); return v; };
  std::size_t p = 20;
  p += 4 + u(p, 4);
  p += 8 + u(p, 8);
  p += 8 + u(p, 8);
  std::size_t n = 0;
  const std::size_t end = b.size() - kReplayFooterSize;
  for (; p < end; ++n) p += 13 + u(p + 9, 4);
  EXPECT_EQ(p, end);
  EXPECT_EQ(u(end, 8), n);  // footer record count
  return n;
}

// After editing a log's bytes on purpose: recompute the footer checksum so the edit reaches the check under test.
void reseal(Bytes& b) {
  const std::uint64_t s = fnv1a64(std::span<const std::byte>(b.data(), b.size() - 8));
  std::memcpy(b.data() + b.size() - 8, &s, 8);
}

struct Live {
  Bytes log;
  std::uint64_t hash = 0, tick = 0;
  Bytes save;
  std::size_t logged = 0;       // commands the generator expects to be logged
  Bytes before_last;            // complete log as it stood before the final command
};

// 300 seeded random commands, reads interleaved. Rejected-at-the-gate commands are mixed in too.
Live record_session(std::uint64_t seed) {
  std::mt19937_64 g(seed);
  auto u = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); };
  World w = pendulum();
  w.start_log();
  Live l;
  std::uint64_t snaps = 0;
  for (int i = 0; i < 300; ++i) {
    std::string c;
    switch (g() % 13) {
      case 0: case 1: c = setattr(kHinge, "qpos", num(u(-1.5, 1.5))); break;
      case 2: c = setattr(kHinge, "qvel", num(u(-2, 2))); break;
      case 3: case 4: c = setattr("/actuators/torque", "ctrl", num(u(-5, 5))); break;
      case 5: case 6: c = step(1 + static_cast<int>(g() % 50)); break;
      case 7: c = cmd("Snapshot", "{}"); ++snaps; break;
      case 8: c = snaps ? cmd("Restore", R"({"id":)" + std::to_string(1 + g() % snaps) + "}") : cmd("Snapshot", "{}");
              if (!snaps) ++snaps; break;
      case 9: c = query("/world/pole", g() % 2 ? "pos" : "quat"); break;  // mj_kinematics; must not perturb
      case 10: c = query(kHinge, "qpos"); break;
      case 11: c = setattr(kHinge, "qpos", "1,2"); break;                // logged, fails: invalid_argument
      default: c = g() % 2 ? step(200000)                                // limit_exceeded: refused, not logged
                           : cmd("LoadModel", R"({"path":"two_link.urdf"})");  // permission_denied: refused
    }
    const std::size_t before = w.log()->bytes().size();
    execute_json(w, c, kRC);
    l.logged += w.log()->bytes().size() > before;
  }
  l.before_last = w.log()->bytes();
  execute_json(w, setattr(kHinge, "qpos", "0.321"), kRC);  // last record: dropping it must change the hash
  l.log = w.log()->bytes();
  l.hash = w.state_hash();
  l.tick = w.tick();
  l.save = w.save();
  return l;
}

fs::path tmp(const std::string& name) {
  const fs::path d = fs::path(testing::TempDir()) / "dvsim_replay";
  fs::create_directories(d);
  return d / (std::string(testing::UnitTest::GetInstance()->current_test_info()->name()) + "_" + name);
}
void write_file(const fs::path& p, const Bytes& b) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}
Bytes read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::vector<char> c((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  Bytes b(c.size());
  std::memcpy(b.data(), c.data(), c.size());
  return b;
}
// Runs a shell command, stdout+stderr to a file; returns exit status (0 = success) and the output.
int run(const std::string& command, std::string& output) {
  const fs::path out = tmp("cmd.out");
  const int rc = std::system((command + " > \"" + out.string() + "\" 2>&1").c_str());
  const Bytes b = read_file(out);
  output.assign(reinterpret_cast<const char*>(b.data()), b.size());
  return rc;
}
std::string hash_line(std::uint64_t tick, std::uint64_t hash) {
  char b[80];
  std::snprintf(b, sizeof b, "tick=%llu state_hash=%016llx", static_cast<unsigned long long>(tick),
                static_cast<unsigned long long>(hash));
  return b;
}
// what() of the std::runtime_error replay throws, or "" if it replayed cleanly.
std::string replay_error(const Bytes& log, const std::string& root = kData) {
  try {
    replay_of(log, false, root);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}
}  // namespace

TEST(replay, random_session_replays_to_equal_state) {
  const Live l = record_session(12345);
  EXPECT_EQ(count_records(l.log), l.logged + 1);  // reads and gate-refused commands are not in the log
  EXPECT_GT(l.logged, 100u);
  std::cout << "[ live ] records=" << l.logged + 1 << " tick=" << l.tick << " hash=" << std::hex << l.hash << std::dec << "\n";
  World r = replay_of(l.log);
  EXPECT_EQ(r.state_hash(), l.hash);
  EXPECT_EQ(r.tick(), l.tick);
  EXPECT_EQ(r.save(), l.save);  // every state byte, not just the hash
}

TEST(replay, rerecord_is_byte_identical) {
  const Live l = record_session(777);
  World r = replay_of(l.log, /*record=*/true);
  ASSERT_NE(r.log(), nullptr);
  EXPECT_TRUE(r.log()->bytes() == l.log) << "re-recorded log differs from the original";
  EXPECT_EQ(r.state_hash(), l.hash);
}

TEST(replay, dvsim_exe_matches_live_hash) {
  const Live l = record_session(2024);
  const fs::path log = tmp("live.dvsimog"), again = tmp("again.dvsimog");
  write_file(log, l.log);
  std::string out;
  ASSERT_EQ(run(std::string("\"") + DVSIM_CLI_EXE + "\" \"" + log.string() + "\" --record \"" + again.string() + "\"", out), 0) << out;
  std::cout << "[ dvsim-replay ] " << out;
  EXPECT_EQ(out, hash_line(l.tick, l.hash) + "\n");
  EXPECT_TRUE(read_file(again) == l.log) << "exe --record output differs";
}

TEST(replay, bad_logs_fail_cleanly) {
  const Live l = record_session(99);
  const Bytes& good = l.log;
  auto show = [&](const char* what, const Bytes& b, const char* needle) {
    const std::string e = replay_error(b);
    std::cout << "[ " << what << " ] " << e << "\n";
    EXPECT_NE(e.find(needle), std::string::npos) << what << ": '" << e << "'";
  };
  for (std::size_t cut : {std::size_t{0}, std::size_t{10}, std::size_t{30}, good.size() - 3})
    if (cut) show("truncated", Bytes(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(cut)), "truncated");
  show("empty", Bytes{}, "bad magic");

  Bytes b = good;
  b[0] ^= std::byte{0xFF};
  show("bad magic", b, "bad magic");
  b = good; b[8] = std::byte{9};
  show("format version", b, "format version");
  b = good; b[12] ^= std::byte{1};
  show("schema hash", b, "schema hash");
  b = good; b[24] ^= std::byte{1};  // first char of the solver version string
  show("damaged, not resealed", b, "checksum mismatch");
  reseal(b);
  show("solver version", b, "solver version mismatch");

  // Footer: truncation exactly at a record boundary, damaged model bytes, a footer that lies about the count.
  b = l.before_last;
  b.resize(b.size() - kReplayFooterSize);
  show("cut at record boundary", b, "checksum mismatch");
  std::uint32_t vlen;
  std::memcpy(&vlen, good.data() + 20, 4);
  b = good; b[24 + vlen + 8 + 5] ^= std::byte{1};  // inside the model bytes, which must never reach mj_loadModelBuffer
  show("model byte flipped", b, "checksum mismatch");
  b = good; b[b.size() - kReplayFooterSize] ^= std::byte{1};  // first byte of the footer's record count
  reseal(b);
  show("record count", b, "footer claims");

  // One record whose envelope bytes are garbage: must be rejected by the Verifier before execution.
  World w = pendulum();
  w.start_log();
  const std::size_t before = w.log()->bytes().size() - kReplayFooterSize;  // offset of the record
  execute_json(w, setattr(kHinge, "qpos", "0.5"), kRC);
  b = w.log()->bytes();
  for (std::size_t i = before + 13; i < b.size() - kReplayFooterSize; ++i) b[i] = std::byte{0xFF};
  reseal(b);
  show("corrupt envelope", b, "corrupt envelope");

  // The exe: non-zero exit + message on a bad file, and on a missing one.
  const fs::path bad = tmp("bad.dvsimog");
  write_file(bad, Bytes(good.begin(), good.begin() + 30));
  std::string out;
  EXPECT_NE(run(std::string("\"") + DVSIM_CLI_EXE + "\" \"" + bad.string() + "\"", out), 0);
  std::cout << "[ dvsim-replay bad file ] " << out;
  EXPECT_NE(out.find("dvsim-replay: error: truncated"), std::string::npos) << out;
  EXPECT_NE(run(std::string("\"") + DVSIM_CLI_EXE + "\" /nonexistent.dvsimog", out), 0);
  EXPECT_NE(out.find("cannot open"), std::string::npos) << out;
}

TEST(replay, test_can_fail_dropped_record_changes_hash) {
  const Live l = record_session(5);
  World r = replay_of(l.before_last);  // log with the final record dropped: valid, but a different history
  EXPECT_NE(r.state_hash(), l.hash);
  EXPECT_NE(r.save(), l.save);
  EXPECT_EQ(replay_of(l.log).state_hash(), l.hash);  // control: the full log still matches
}

TEST(replay, load_model_replays_under_root) {
  World w = pendulum();
  w.start_log();
  w.step(10);
  const std::string loaded = execute_json(w, cmd("LoadModel", R"({"path":"two_link.urdf"})"), kAll);
  ASSERT_TRUE(loaded.find("\"error\": \"none\"") != std::string::npos || loaded.find("\"error\":\"none\"") != std::string::npos ||
              loaded.find("\"ok\": true") != std::string::npos)
      << loaded;
  std::string shoulder;
  for (const Prim& p : w.solver().prims())
    if (p.path.ends_with("/shoulder")) shoulder = p.path;
  ASSERT_FALSE(shoulder.empty());
  execute_json(w, setattr(shoulder, "qpos", "0.7"), kRC);
  execute_json(w, step(25), kRC);
  const Bytes log = w.log()->bytes();

  EXPECT_EQ(replay_of(log).state_hash(), w.state_hash());  // LoadModel resolves under root
  const std::string e = replay_error(log, "");               // no root: LoadModel is refused -> loud mismatch
  std::cout << "[ load_model without root ] " << e << "\n";
  EXPECT_NE(e.find("diverged from the live run"), std::string::npos) << e;
}
