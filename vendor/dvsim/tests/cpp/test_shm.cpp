// shm lockstep (M11): a CHILD PROCESS (shm_test_client, header-only client) drives 10k lockstep rounds against a server
// in this process; hashes must equal an in-process run of the same ctrl sequence and the server's replay log. Also
// the failure paths: layout/version rejection, request validation, client death, server death, segment cleanup.
#include <gtest/gtest.h>
#include <sys/wait.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "core/shm.h"
#include "mujoco/mujoco_solver.h"
#include "shm_ctrl.h"

namespace {
namespace shm = dvsim::shm;
using Clock = std::chrono::steady_clock;

const std::string kData = DVSIM_TEST_DATA_DIR;
const std::string kHinge = "/world/pole/hinge";
const std::vector<std::string> kActs = {"/actuators/torque", "/actuators/servo"};
constexpr std::uint64_t kSeed = 20240607, kRounds = 10000;

const ShmServer::Obs kObs = {{kHinge, "qpos"}, {kHinge, "qvel"}};
Session session() {
  Session s;
  s.caps = {"read", "control"};
  s.root = kData;
  return s;
}
World pendulum() { return World(load_mujoco, kData + "/pendulum.xml"); }
std::string seg(const char* tag) { return "/dvsim-test-" + std::to_string(getpid()) + "-" + tag; }
bool exists(const std::string& name) { return access(("/dev/shm" + name).c_str(), F_OK) == 0; }
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// Child process via popen (stdout piped back). Returns the output and the wait status.
struct Child {
  FILE* f;
  explicit Child(const std::string& args) : f(popen((std::string(DVSIM_SHM_CLIENT_EXE) + " " + args).c_str(), "r")) {}
  std::string out_and_wait(int& status) {
    std::string out;
    char buf[256];
    while (std::fgets(buf, sizeof buf, f)) out += buf;
    status = pclose(f);
    f = nullptr;
    return out;
  }
};

// Serves on its own thread (the World is touched only there until join()).
struct Served {
  std::thread t;
  std::string err;
  Served(ShmServer& s, double idle) : t([this, &s, idle] {
    try {
      s.serve(idle);
    } catch (const std::exception& e) {
      err = e.what();
    }
  }) {}
  ~Served() { if (t.joinable()) t.join(); }
};

// Applies the shared ctrl sequence in-process on a fresh World through the solver (not the bus or shm).
World reference(std::uint64_t rounds) {
  World w = pendulum();
  for (std::uint64_t r = 0; r < rounds; ++r) {
    for (int i = 0; i < 2; ++i) {
      const double v = shm_ctrl_at(kSeed, r, i);
      EXPECT_EQ(w.solver().set(kActs[i], "ctrl", "", std::span<const double>(&v, 1)).code, Result::ok);
    }
    w.step(1);
  }
  return w;
}

std::string attach_error(const std::string& n, std::uint32_t nu = shm::kAny, std::uint32_t nobs = shm::kAny) {
  try {
    shm::Client c(n, nu, nobs, 0.3);
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}
void tamper(const std::string& name, const std::function<void(shm::Header&)>& fn) {
  const int fd = shm_open(name.c_str(), O_RDWR, 0);
  ASSERT_GE(fd, 0);
  void* p = mmap(nullptr, sizeof(shm::Header), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  ASSERT_NE(p, MAP_FAILED);
  fn(*static_cast<shm::Header*>(p));
  munmap(p, sizeof(shm::Header));
}
}  // namespace

TEST(shm_lockstep, child_process_10k_steps_matches_in_process_and_replay) {
  const std::string name = seg("main");
  World w = pendulum();
  w.start_log();
  std::string out;
  int status = -1;
  const auto t0 = Clock::now();
  {
    ShmServer server(w, session(), name, kObs);
    EXPECT_TRUE(exists(name));
    Child child(name + " " + std::to_string(kSeed) + " " + std::to_string(kRounds));
    ASSERT_NE(child.f, nullptr);
    EXPECT_EQ(server.serve(10), kRounds);
    out = child.out_and_wait(status);
  }
  const double wall = seconds_since(t0);
  EXPECT_FALSE(exists(name)) << "segment not unlinked";
  ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "child status " << status << " out: " << out;

  std::uint64_t tick = 0, child_hash = 0;
  double obs0 = 0, maxlat_us = 0;
  ASSERT_EQ(std::sscanf(out.c_str(), "tick=%" SCNu64 " hash=%" SCNx64 " obs0=%lf maxlat_us=%lf", &tick, &child_hash, &obs0,
                        &maxlat_us), 4) << out;
  World ref = reference(kRounds);
  std::vector<double> q;
  ASSERT_EQ(ref.solver().get(kHinge, "qpos", "", q).code, Result::ok);

  World replayed = replay(w.log()->bytes(), load_mujoco_bytes, load_mujoco, kData);
  std::printf("[shm] child hash=%016" PRIx64 " server hash=%016" PRIx64 " in-process hash=%016" PRIx64
              " replay hash=%016" PRIx64 "\n[shm] ticks child=%" PRIu64 " server=%" PRIu64 " ref=%" PRIu64
              "; %" PRIu64 " rounds in %.3f s wall, worst round latency %.0f us\n",
              child_hash, w.state_hash(), ref.state_hash(), replayed.state_hash(), tick, w.tick(), ref.tick(), kRounds, wall,
              maxlat_us);
  EXPECT_EQ(tick, kRounds);
  EXPECT_EQ(child_hash, ref.state_hash());
  EXPECT_EQ(w.state_hash(), ref.state_hash());
  EXPECT_EQ(replayed.state_hash(), ref.state_hash());
  EXPECT_EQ(replayed.tick(), ref.tick());
  EXPECT_EQ(obs0, q.at(0)) << "obs[0] (hinge qpos) as seen by the client";
  // Timeout is 10 s: a wake-up lost across processes would show up as a worst-case latency of seconds.
  EXPECT_LT(maxlat_us, 1e6);
}

TEST(shm_lockstep, attach_rejects_bad_layout_version_magic_size) {
  const std::string name = seg("bad");
  World w = pendulum();
  ShmServer server(w, session(), name, kObs);
  EXPECT_EQ(attach_error(name), "");  // control: a valid attach works (nu=2, nobs=2)
  EXPECT_EQ(attach_error(name, 2, 2), "");
  EXPECT_NE(attach_error(name, 3, shm::kAny).find("layout mismatch"), std::string::npos);
  EXPECT_NE(attach_error(name, shm::kAny, 5).find("layout mismatch"), std::string::npos);
  {
    shm::Client first(name);
    EXPECT_NE(attach_error(name).find("already has a client"), std::string::npos);
  }
  tamper(name, [](shm::Header& h) { h.version = 2; });
  EXPECT_NE(attach_error(name).find("layout version 2"), std::string::npos);
  tamper(name, [](shm::Header& h) { h.version = shm::kVersion; h.magic[0] = 'X'; });
  EXPECT_NE(attach_error(name).find("bad magic"), std::string::npos);
  tamper(name, [](shm::Header& h) { h.magic[0] = 'N'; h.nu = 7; });  // sizes no longer add up to the segment
  EXPECT_NE(attach_error(name).find("inconsistent header sizes"), std::string::npos);
  tamper(name, [](shm::Header& h) { h.nu = 2; h.ready.store(0); });
  EXPECT_NE(attach_error(name).find("not ready"), std::string::npos);

  const std::string tiny = seg("tiny");  // not a dvsim block at all
  const int fd = shm_open(tiny.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 16), 0);
  close(fd);
  EXPECT_NE(attach_error(tiny).find("not a dvsim block"), std::string::npos);
  shm_unlink(tiny.c_str());
  EXPECT_NE(attach_error(seg("missing")).find("shm_open"), std::string::npos);
  EXPECT_THROW(ShmServer(w, session(), name, kObs), std::runtime_error);  // exclusive create
}

TEST(shm_lockstep, server_validates_requests) {
  const std::string name = seg("req");
  World w = pendulum();
  ShmServer server(w, session(), name, kObs, 50);
  Served served(server, 10);
  shm::Client c(name, 2, 2);
  c.ctrl()[0] = std::nan("");
  EXPECT_EQ(c.step(1).status, 1u) << "NaN ctrl: dvsim.Err.invalid_argument";
  EXPECT_EQ(c.step(1).tick, 0u) << "a refused ctrl must not step";
  c.ctrl()[0] = 1.0;
  EXPECT_EQ(c.step(51).status, shm::kBadRequest);
  const shm::Client::Reply ok = c.step(50);
  EXPECT_EQ(ok.status, 0u);
  EXPECT_EQ(ok.tick, 50u);
  EXPECT_EQ(c.step(0).tick, 50u);  // observe only
  EXPECT_NE(c.obs()[1], 0.0) << "qvel after 50 ticks of torque";
  c.close_session();
}

TEST(shm_lockstep, client_death_times_out_and_cleans_up) {
  const std::string name = seg("die");
  World w = pendulum();
  w.start_log();
  int status = -1;
  double wall = 0;
  std::string err;
  {
    ShmServer server(w, session(), name, kObs);
    Child child(name + " 7 1000 100");  // SIGKILLs itself after 100 rounds
    ASSERT_NE(child.f, nullptr);
    const auto t0 = Clock::now();
    try {
      server.serve(0.5);
    } catch (const std::runtime_error& e) {
      err = e.what();
    }
    wall = seconds_since(t0);
    child.out_and_wait(status);
  }
  std::printf("[shm] client killed at round 100: server error \"%s\" after %.3f s wall (idle timeout 0.5 s; includes the 100 "
              "rounds)\n", err.c_str(), wall);
  EXPECT_NE(err.find("no request for"), std::string::npos);
  EXPECT_GE(wall, 0.5);
  EXPECT_LT(wall, 3.0);
  // popen runs the child under `sh -c`, which reports a SIGKILLed child as exit status 128 + 9.
  EXPECT_TRUE((WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL) || (WIFEXITED(status) && WEXITSTATUS(status) == 137))
      << status;
  EXPECT_EQ(w.tick(), 100u);
  EXPECT_FALSE(exists(name)) << "segment not unlinked after the server unwound";
  // The log of the interrupted run is still a valid, replayable log.
  EXPECT_EQ(replay(w.log()->bytes(), load_mujoco_bytes, load_mujoco, kData).state_hash(), w.state_hash());
}

TEST(shm_lockstep, server_gone_or_idle_client_times_out_and_unblocks) {
  const std::string name = seg("srv");
  World w = pendulum();
  auto server = std::make_unique<ShmServer>(w, session(), name, kObs);
  {
    shm::Client c(name);
    const auto t0 = Clock::now();  // server alive but never serving
    EXPECT_THROW(c.step(1, 0.2), std::runtime_error);
    EXPECT_GE(seconds_since(t0), 0.2);
    EXPECT_LT(seconds_since(t0), 2.0);
  }
  shm::Client c(name);  // server destroyed while the client waits (5 s timeout): it must be woken at once
  std::string err;
  double wall = 0;
  std::thread t([&] {
    const auto t0 = Clock::now();
    try {
      c.step(1, 5);
    } catch (const std::exception& e) {
      err = e.what();
    }
    wall = seconds_since(t0);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  server.reset();
  t.join();
  std::printf("[shm] server closed under a waiting client: \"%s\" after %.3f s wall (client timeout 5 s)\n", err.c_str(), wall);
  EXPECT_NE(err.find("server closed"), std::string::npos);
  EXPECT_LT(wall, 2.0);
  EXPECT_FALSE(exists(name));
}
