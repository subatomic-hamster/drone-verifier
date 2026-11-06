// dvsim-px4-bridge (src/px4): pure frame / unit conversions, and a loopback test where this process plays PX4 (MAVLink over TCP
// + UDP, c_library_v2) against the built bridge on tests/data/devices/all.xml. See docs/research/px4-sitl.md.
#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <common/mavlink.h>

#include "px4/frames.h"

namespace {
using namespace px4f;
namespace fs = std::filesystem;

const std::string kData = DVSIM_TEST_DATA_DIR;
const std::string kCfg = kData + "/px4/bridge_all.toml";

// ---------------------------------------------------------------- frames
double norm(const V3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
Q4 unit(Q4 q) {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (double& c : q) c /= n;
  return q;
}
void expect_v(const V3& a, const V3& b, double tol) {
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(a[static_cast<std::size_t>(i)], b[static_cast<std::size_t>(i)], tol) << "component " << i;
}

TEST(Px4Frames, body_and_world_vectors) {
  expect_v(flu_to_frd({0, 0, 9.81}), {0, 0, -9.81}, 0);  // rest accel: specific force up in FLU = -z in FRD
  expect_v(flu_to_frd({1, 2, 3}), {1, -2, -3}, 0);       // gyro / mag
  expect_v(enu_to_ned({1, 2, 3}), {2, 1, -3}, 0);        // E=1 N=2 U=3 -> N=2 E=1 D=-3
}

TEST(Px4Frames, attitude_convention) {
  const double s = std::sqrt(0.5);
  Q4 q = q_ned_frd({1, 0, 0, 0});  // identity dvsim attitude: nose East = yaw +90 deg in NED
  EXPECT_NEAR(q[0], s, 1e-12);
  EXPECT_NEAR(q[1], 0, 1e-12);
  EXPECT_NEAR(q[2], 0, 1e-12);
  EXPECT_NEAR(q[3], s, 1e-12);
  q = q_ned_frd({s, 0, 0, s});  // dvsim yaw +90 deg about z (nose North) -> identity
  EXPECT_NEAR(q[0], 1, 1e-12);
  EXPECT_NEAR(q[1], 0, 1e-12);
  EXPECT_NEAR(q[2], 0, 1e-12);
  EXPECT_NEAR(q[3], 0, 1e-12);
}

TEST(Px4Frames, rotation_commutes_with_frame_change) {
  // R_ned_frd * v_frd == ENU->NED(R_enu_flu * v_flu), for several attitudes (roll / pitch / yaw mixes) and vectors
  for (const Q4 raw : {Q4{0.8, 0.3, -0.2, 0.45}, Q4{0.1, 0.9, 0.2, -0.3}, Q4{-0.5, 0.2, 0.7, 0.4}, Q4{0.3, -0.1, 0.05, -0.9}}) {
    const Q4 q = unit(raw);
    for (const V3 v : {V3{1, 0, 0}, V3{0, 1, 0}, V3{0, 0, 1}, V3{0.3, -1.2, 2.5}}) {
      expect_v(qrot(q_ned_frd(q), flu_to_frd(v)), enu_to_ned(qrot(q, v)), 1e-12);
    }
  }
}

TEST(Px4Units, scaling) {
  EXPECT_NEAR(tesla_to_gauss(4.8e-5), 0.48, 1e-12);
  EXPECT_NEAR(pa_to_hpa(101325), 1013.25, 1e-12);
  EXPECT_NEAR(pressure_alt(isa_pressure(488.0)), 488.0, 0.01);
  EXPECT_NEAR(pressure_alt(isa_pressure(-100.0)), -100.0, 0.01);
  EXPECT_NEAR(isa_temp_c(kP0), 15.0, 1e-9);
  EXPECT_NEAR(isa_temp_c(isa_pressure(1000.0)), 15.0 - 6.5, 1e-9);
  EXPECT_EQ(deg_e7(47.3769), 473769000);
  EXPECT_EQ(deg_e7(1.00000004), 10000000);
  EXPECT_EQ(deg_e7(1.00000006), 10000001);
  EXPECT_EQ(deg_e7(-1.00000006), -10000001);
  EXPECT_EQ(mm(488.0004), 488000);
  EXPECT_EQ(cm_s_i16(1.234), 123);
  EXPECT_EQ(cm_s_i16(-1.236), -124);
  EXPECT_EQ(cm_s_i16(400), 32767);
  EXPECT_EQ(cm_u16(0.5), 50);
  EXPECT_EQ(cm_u16(-1), 0);
}

TEST(Px4Units, course_over_ground) {
  EXPECT_EQ(cog_cdeg(1, 0), 0);       // north
  EXPECT_EQ(cog_cdeg(0, 1), 9000);    // east
  EXPECT_EQ(cog_cdeg(-1, 0), 18000);  // south
  EXPECT_EQ(cog_cdeg(0, -1), 27000);  // west (negative atan2 wraps)
  EXPECT_EQ(cog_cdeg(1, -1e-9), 0);   // 35999.99999 rounds to 36000 -> wraps to 0, never 36000
  EXPECT_EQ(cog_cdeg(0.001, 0.001), 65535);
  EXPECT_EQ(cog_cdeg(0, 0), 65535);
}

TEST(Px4Units, fields_updated_and_sampling) {
  EXPECT_EQ(fields_updated(true, true, true), 7167u);
  EXPECT_EQ(fields_updated(true, false, false), 63u);
  EXPECT_EQ(fields_updated(true, true, false), 6719u);
  EXPECT_EQ(fields_updated(true, false, true), 511u);
  EXPECT_TRUE(sampled(0, 4, 20));
  EXPECT_TRUE(sampled(16, 4, 20) == false);  // ticks 16..19: next multiple of 20 is 20
  EXPECT_TRUE(sampled(17, 4, 20));           // ticks 17..20
  EXPECT_FALSE(sampled(4, 4, 20));
  EXPECT_TRUE(sampled(1234, 4, 1));
}

// ---------------------------------------------------------------- loopback harness
int free_port(int type) {
  const int fd = socket(AF_INET, type, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t len = sizeof a;
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a), 0);
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
  close(fd);
  return ntohs(a.sin_port);
}

// Child process with stdin / stdout pipes. Killed on destruction so a failed assertion leaves no stray bridge.
struct Proc {
  pid_t pid = -1;
  int in = -1, out = -1;
  std::string buf;
  bool reaped = false;
  explicit Proc(std::vector<std::string> args, bool merge_err = false) {
    int pi[2], po[2];
    if (pipe(pi) != 0 || pipe(po) != 0) std::abort();
    pid = fork();
    if (pid == 0) {
      dup2(pi[0], 0);
      dup2(po[1], 1);
      if (merge_err) dup2(po[1], 2);
      for (int fd : {pi[0], pi[1], po[0], po[1]}) close(fd);
      std::vector<char*> av;
      for (std::string& a : args) av.push_back(a.data());
      av.push_back(nullptr);
      execv(av[0], av.data());
      _exit(127);
    }
    close(pi[0]);
    close(po[1]);
    in = pi[1];
    out = po[0];
  }
  ~Proc() {
    if (pid > 0 && !reaped) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
    if (in >= 0) close(in);
    if (out >= 0) close(out);
  }
  void write(const std::string& s) const { EXPECT_EQ(::write(in, s.data(), s.size()), static_cast<ssize_t>(s.size())); }
  // Next stdout line, or false on timeout / EOF.
  bool line(std::string& s, double timeout_s) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout_s * 1000));
    for (;;) {
      const auto nl = buf.find('\n');
      if (nl != std::string::npos) {
        s = buf.substr(0, nl);
        buf.erase(0, nl + 1);
        return true;
      }
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - std::chrono::steady_clock::now()).count();
      pollfd p{out, POLLIN, 0};
      if (left <= 0 || poll(&p, 1, static_cast<int>(left)) <= 0) return false;
      char b[1024];
      const ssize_t r = read(out, b, sizeof b);
      if (r <= 0) return false;
      buf.append(b, static_cast<std::size_t>(r));
    }
  }
  // Waits for a line starting with `prefix` (other lines are kept in `seen`).
  bool expect(const std::string& prefix, std::string& s, double timeout_s, std::vector<std::string>* seen = nullptr) {
    while (line(s, timeout_s)) {
      if (s.rfind(prefix, 0) == 0) return true;
      if (seen) seen->push_back(s);
    }
    return false;
  }
  int wait(double timeout_s = 20) {
    std::string rest;
    for (int i = 0; i < static_cast<int>(timeout_s * 100); ++i) {
      int st = 0;
      if (waitpid(pid, &st, WNOHANG) == pid) {
        reaped = true;
        return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
      }
      usleep(10000);
    }
    return -999;
  }
};

// MAVLink endpoint over a connected TCP socket or a bound UDP socket.
struct Link {
  int fd;
  std::uint8_t chan;
  std::uint8_t buf[4096];
  ssize_t len = 0, pos = 0;
  mavlink_status_t st{};
  Link(int f, std::uint8_t c) : fd(f), chan(c) {}
  bool next(mavlink_message_t& m, int timeout_ms) {
    for (;;) {
      while (pos < len)
        if (mavlink_parse_char(chan, buf[pos++], &m, &st) == MAVLINK_FRAMING_OK) return true;
      pollfd p{fd, POLLIN, 0};
      if (poll(&p, 1, timeout_ms) <= 0) return false;
      len = recv(fd, buf, sizeof buf, 0);
      pos = 0;
      if (len <= 0) { len = 0; return false; }
    }
  }
  void send(const mavlink_message_t& m) const {
    std::uint8_t b[MAVLINK_MAX_PACKET_LEN];
    const std::size_t n = mavlink_msg_to_send_buffer(b, &m);
    EXPECT_EQ(::send(fd, b, n, 0), static_cast<ssize_t>(n));
  }
  void ctrl(float c0, float c1, float c2, float c3, std::uint8_t mode) const {
    float c[16] = {c0, c1, c2, c3};
    mavlink_message_t m;
    mavlink_msg_hil_actuator_controls_pack(1, 1, &m, 0, c, mode, 0);
    send(m);
  }
};

int connect_to(int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  if (connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}
int bind_udp_at(int port) {
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a), 0);
  return fd;
}

// The bridge trace CSV, re-read from disk (the bridge flushes every row).
struct Trace {
  std::map<std::string, std::size_t> col;
  std::vector<std::vector<double>> rows;
  explicit Trace(const std::string& path) {
    std::ifstream f(path);
    std::string l;
    if (!std::getline(f, l)) return;
    std::stringstream h(l);
    std::string c;
    for (std::size_t i = 0; std::getline(h, c, ','); ++i) col[c] = i;
    while (std::getline(f, l)) {
      std::stringstream r(l);
      std::vector<double> row;
      std::string v;
      while (std::getline(r, v, ',')) row.push_back(std::stod(v));
      if (row.size() == col.size()) rows.push_back(row);  // a half-written last row is dropped
    }
  }
  double at(std::size_t row, const std::string& name) const { return rows.at(row).at(col.at(name)); }
  // Row with this tick, or -1.
  int row_of(std::uint64_t tick) const {
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (static_cast<std::uint64_t>(rows[i][col.at("tick")]) == tick) return static_cast<int>(i);
    return -1;
  }
};
int wait_row(const std::string& path, std::uint64_t tick) {
  for (int i = 0; i < 200; ++i) {
    const int r = Trace(path).row_of(tick);
    if (r >= 0) return r;
    usleep(10000);
  }
  return -1;
}

double num(const std::string& json, const std::string& key) {
  const auto p = json.find("\"" + key + "\":");
  return p == std::string::npos ? NAN : std::stod(json.substr(p + key.size() + 3));
}

TEST(Px4Bridge, connect_timeout_exits_1) {
  Proc p({DVSIM_PX4_BRIDGE_EXE, "--config", kCfg, "--port", std::to_string(free_port(SOCK_STREAM)), "--udp-bind",
          std::to_string(free_port(SOCK_DGRAM)), "--connect-timeout", "1"},
         true);
  std::string out, l;
  while (p.line(l, 10)) out += l + "\n";
  EXPECT_EQ(p.wait(), 1);
  EXPECT_NE(out.find("timed out"), std::string::npos) << out;
}

TEST(Px4Bridge, rejects_inconsistent_home) {
  const std::string bad = (fs::temp_directory_path() / ("px4_bad_home_" + std::to_string(getpid()) + ".toml")).string();
  {
    std::ifstream in(kCfg);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string t = ss.str();
    t.replace(t.find("alt = 408.0"), 11, "alt = 400.0");
    t.replace(t.find("../devices/all.xml"), 18, kData + "/devices/all.xml");
    std::ofstream(bad) << t;
  }
  Proc p({DVSIM_PX4_BRIDGE_EXE, "--config", bad, "--connect-timeout", "1"}, true);
  std::string out, l;
  while (p.line(l, 10)) out += l + "\n";
  EXPECT_EQ(p.wait(), 1);
  EXPECT_NE(out.find("[home] differs"), std::string::npos) << out;
  fs::remove(bad);
}

// GPS-denied config (no `gnss` key): the stream has HIL_SENSOR and ground truth but no HIL_GPS; and the ctl socket's `imu <since>` returns the
// IMU samples that went to PX4, one per HIL step, with the world tick they were taken at (a companion computer's estimator reads these).
TEST(Px4Bridge, gps_denied_and_imu_history_over_ctl) {
  const fs::path dir = fs::temp_directory_path() / ("px4_nogps_" + std::to_string(getpid()));
  fs::create_directories(dir);
  const std::string cfg = (dir / "bridge.toml").string(), sock = (dir / "ctl.sock").string();
  {
    std::ifstream in(kCfg);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string t = ss.str();
    t.replace(t.find("gnss = \"gnss\""), 13, "");
    t.replace(t.find("../devices/all.xml"), 18, kData + "/devices/all.xml");
    std::ofstream(cfg) << t;
  }
  const int port = free_port(SOCK_STREAM), px4_udp = free_port(SOCK_DGRAM), our_udp = free_port(SOCK_DGRAM);
  Proc bridge({DVSIM_PX4_BRIDGE_EXE, "--config", cfg, "--port", std::to_string(port), "--px4-udp", std::to_string(px4_udp), "--udp-bind",
               std::to_string(our_udp), "--ctl", sock, "--connect-timeout", "20"});  // real time: 250 steps/s, so the 16 s history never wraps here
  std::string l;
  ASSERT_TRUE(bridge.expect("listening", l, 30));
  const int fd = connect_to(port);
  ASSERT_GE(fd, 0);
  Link px4(fd, MAVLINK_COMM_0);
  mavlink_message_t m;
  mavlink_msg_heartbeat_pack(1, 1, &m, MAV_TYPE_QUADROTOR, MAV_AUTOPILOT_PX4, 0, 0, MAV_STATE_ACTIVE);
  px4.send(m);
  ASSERT_TRUE(bridge.expect("connected", l, 10));
  std::vector<mavlink_hil_sensor_t> sens;
  int gps = 0;
  while (sens.size() < 100) {  // no HIL_ACTUATOR_CONTROLS is ever sent: the bridge free-runs
    ASSERT_TRUE(px4.next(m, 10000));
    if (m.msgid == MAVLINK_MSG_ID_HIL_SENSOR) mavlink_msg_hil_sensor_decode(&m, &sens.emplace_back());
    else if (m.msgid == MAVLINK_MSG_ID_HIL_GPS) ++gps;
  }
  EXPECT_EQ(gps, 0);

  const int cfd = socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un ua{};
  ua.sun_family = AF_UNIX;
  std::strncpy(ua.sun_path, sock.c_str(), sizeof ua.sun_path - 1);
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  ASSERT_EQ(connect(cfd, reinterpret_cast<sockaddr*>(&ua), sizeof ua), 0);
  ASSERT_EQ(::write(cfd, "imu 0\n", 6), 6);
  std::string reply;
  for (char c; reply.empty() || reply.back() != '\n';) {
    pollfd p{cfd, POLLIN, 0};
    ASSERT_GT(poll(&p, 1, 10000), 0);
    ASSERT_EQ(::read(cfd, &c, 1), 1);
    reply += c;
  }
  close(cfd);
  std::istringstream is(reply);
  std::string ok;
  std::uint64_t tick = 0, cnt = 0;
  is >> ok >> tick >> cnt;
  ASSERT_EQ(ok, "ok");
  ASSERT_GE(cnt, sens.size());
  for (std::uint64_t i = 0; i < cnt; ++i) {
    std::uint64_t counter;
    double g[3], a[3];
    ASSERT_TRUE(is >> counter >> g[0] >> g[1] >> g[2] >> a[0] >> a[1] >> a[2]) << "record " << i;
    EXPECT_EQ(counter, 4 * i + 2) << "all.xml IMU is 500 Hz: step i covers ticks [4i, 4i + 4), its last sample was taken at counter 4i + 2";
    if (i < sens.size()) {  // the same sample HIL_SENSOR carried, FLU instead of FRD
      EXPECT_NEAR(a[0], sens[i].xacc, 1e-4);
      EXPECT_NEAR(a[1], -sens[i].yacc, 1e-4);
      EXPECT_NEAR(a[2], -sens[i].zacc, 1e-4);
      EXPECT_NEAR(g[0], sens[i].xgyro, 1e-5);
      EXPECT_NEAR(g[1], -sens[i].ygyro, 1e-5);
      EXPECT_NEAR(g[2], -sens[i].zgyro, 1e-5);
    }
  }
  fs::remove_all(dir);
}

TEST(Px4Bridge, loopback_lockstep_all_xml) {
  const fs::path dir = fs::temp_directory_path() / ("px4_loop_" + std::to_string(getpid()));
  fs::create_directories(dir);
  const std::string trace = (dir / "trace.csv").string(), log = (dir / "run.log").string();
  const int port = free_port(SOCK_STREAM), px4_udp = free_port(SOCK_DGRAM), our_udp = free_port(SOCK_DGRAM);

  const int udp_fd = bind_udp_at(px4_udp);  // PX4's MAVLink UDP port: the bridge sends BATTERY_STATUS here
  Link udp(udp_fd, MAVLINK_COMM_1);
  Proc bridge({DVSIM_PX4_BRIDGE_EXE, "--config", kCfg, "--port", std::to_string(port), "--px4-udp", std::to_string(px4_udp), "--udp-bind",
               std::to_string(our_udp), "--speed", "0", "--trace", trace, "--log", log, "--connect-timeout", "20"});
  std::string l;
  ASSERT_TRUE(bridge.expect("listening", l, 30)) << "no listening line";
  const int fd = connect_to(port);
  ASSERT_GE(fd, 0);
  Link px4(fd, MAVLINK_COMM_0);
  mavlink_message_t m;
  mavlink_msg_heartbeat_pack(1, 1, &m, MAV_TYPE_QUADROTOR, MAV_AUTOPILOT_PX4, 0, 0, MAV_STATE_ACTIVE);
  px4.send(m);
  ASSERT_TRUE(bridge.expect("connected", l, 10)) << "bridge did not print connected";

  // ---- free-running until the first HIL_ACTUATOR_CONTROLS: 300 steps ----
  std::vector<mavlink_hil_sensor_t> sens;
  std::vector<mavlink_hil_gps_t> gps;
  std::vector<mavlink_hil_state_quaternion_t> states;
  while (sens.size() < 300) {
    ASSERT_TRUE(px4.next(m, 10000)) << "stream stopped after " << sens.size() << " sensor messages";
    if (m.msgid == MAVLINK_MSG_ID_HIL_SENSOR) mavlink_msg_hil_sensor_decode(&m, &sens.emplace_back());
    else if (m.msgid == MAVLINK_MSG_ID_HIL_GPS) mavlink_msg_hil_gps_decode(&m, &gps.emplace_back());
    else if (m.msgid == MAVLINK_MSG_ID_HIL_STATE_QUATERNION) mavlink_msg_hil_state_quaternion_decode(&m, &states.emplace_back());
  }
  for (std::size_t i = 0; i < sens.size(); ++i) {
    EXPECT_EQ(sens[i].time_usec, 4000u * (i + 1)) << "step " << i;  // exactly one 4 ms HIL step per message
    EXPECT_EQ(sens[i].id, 0);
    // all.xml: IMU 500 Hz (every step), baro 50 Hz (every 5th step), mag 125 Hz (every 2nd), message i covers ticks [4i, 4i + 4)
    const std::uint32_t want = 7 + 56 + (i % 2 == 0 ? 448 : 0) + (i % 5 == 0 ? 6656 : 0);
    EXPECT_EQ(sens[i].fields_updated, want) << "step " << i;
  }
  EXPECT_EQ(sens[0].fields_updated, 7167u);
  {  // first message: quad at rest at 20 m, falling freely
    const auto& s = sens[0];
    EXPECT_LT(norm({s.xacc, s.yacc, s.zacc}), 0.5);  // free fall: specific force ~ 0 (+ bias)
    EXPECT_LT(norm({s.xgyro, s.ygyro, s.zgyro}), 0.05);
    EXPECT_NEAR(norm({s.xmag, s.ymag, s.zmag}), 0.48, 0.01);  // |(2.1, 0.4, -4.3)| e-5 T in gauss
    EXPECT_NEAR(s.abs_pressure, isa_pressure(300 + 20.04) / 100, 0.2);
    EXPECT_NEAR(s.pressure_alt, 320.04, 2.0);
    EXPECT_NEAR(s.temperature, 15.0 - 0.0065 * 320.04, 0.3);
  }
  // GNSS 10 Hz = every 25th step: steps 0, 25, ..., 275
  ASSERT_EQ(gps.size(), 12u);
  for (std::size_t j = 0; j < gps.size(); ++j) {
    EXPECT_EQ(gps[j].time_usec, sens[25 * j].time_usec);
    EXPECT_EQ(gps[j].fix_type, 3);
    EXPECT_EQ(gps[j].satellites_visible, 10);
    EXPECT_EQ(gps[j].eph, 50);   // horiz_std 0.5 m -> cm
    EXPECT_EQ(gps[j].epv, 100);  // vert_std 1.0 m
    EXPECT_NEAR(gps[j].lat, 473769000, 20000);
    EXPECT_NEAR(gps[j].lon, 85417000, 20000);
    EXPECT_GT(gps[j].alt, 400000);
    EXPECT_LT(gps[j].alt, 440000);
  }
  // ground truth every step; identity dvsim attitude = yaw +90 deg in NED; 20 m up -> alt = home + 20 m
  ASSERT_GE(states.size(), 299u);
  EXPECT_EQ(states[0].time_usec, 4000u);
  EXPECT_NEAR(states[0].attitude_quaternion[0], std::sqrt(0.5), 1e-3);
  EXPECT_NEAR(states[0].attitude_quaternion[3], std::sqrt(0.5), 1e-3);
  EXPECT_NEAR(states[0].alt, 428000, 100);
  EXPECT_NEAR(states[0].lat, 473769000, 20);
  EXPECT_GT(states[298].vz, 1000);  // after 1.2 s of free fall: ~ 11 m/s DOWN = positive in NED (cm/s)

  // ---- battery over the UDP link: step 50k (k = 0..5) -> trace row with tick 200k + 4 ----
  std::vector<mavlink_battery_status_t> bats;
  int heartbeats = 0;
  while (udp.next(m, 500)) {
    EXPECT_EQ(m.sysid, 1);
    EXPECT_EQ(m.compid, 180);
    if (m.msgid == MAVLINK_MSG_ID_BATTERY_STATUS) mavlink_msg_battery_status_decode(&m, &bats.emplace_back());
    else if (m.msgid == MAVLINK_MSG_ID_HEARTBEAT) ++heartbeats;
  }
  ASSERT_GE(bats.size(), 6u);
  EXPECT_GE(heartbeats, 2);  // 1 Hz of sim time
  for (std::size_t k = 0; k < 6; ++k) {
    const int row = wait_row(trace, k == 0 ? 4 : 200 * k + 4);
    ASSERT_GE(row, 0);
    const Trace tr(trace);
    const double v = tr.at(static_cast<std::size_t>(row), "batt_v"), soc = tr.at(static_cast<std::size_t>(row), "batt_soc");
    const auto& b = bats[k];
    for (int c = 0; c < 3; ++c) EXPECT_NEAR(b.voltages[c], v / 3 * 1000, 0.6) << "msg " << k << " cell " << c;  // 3S pack
    for (int c = 3; c < 10; ++c) EXPECT_EQ(b.voltages[c], UINT16_MAX);
    EXPECT_GT(b.voltages[0], 3000);
    EXPECT_LT(b.voltages[0], 4300);
    EXPECT_NEAR(b.current_battery, tr.at(static_cast<std::size_t>(row), "batt_i") * 100, 0.6);
    EXPECT_EQ(b.battery_remaining, std::lround(soc * 100));
    EXPECT_EQ(b.battery_function, MAV_BATTERY_FUNCTION_ALL);
    EXPECT_EQ(b.type, MAV_BATTERY_TYPE_LIPO);
    EXPECT_EQ(b.id, 0);
    EXPECT_GE(b.current_consumed, 0);
  }

  // ---- first controls: armed throttles. The bridge answers with one more step, then blocks. ----
  auto drain_until_quiet = [&] {  // returns the last HIL_SENSOR time_usec; asserts nothing arrives afterwards
    std::uint64_t last = 0;
    while (px4.next(m, 500))
      if (m.msgid == MAVLINK_MSG_ID_HIL_SENSOR) {
        mavlink_hil_sensor_t s;
        mavlink_msg_hil_sensor_decode(&m, &s);
        last = s.time_usec;
      }
    EXPECT_FALSE(px4.next(m, 300)) << "bridge kept stepping without HIL_ACTUATOR_CONTROLS";
    return last;
  };
  auto next_sensor = [&](std::uint64_t& t_us) {  // the next HIL_SENSOR (other messages skipped)
    while (px4.next(m, 5000))
      if (m.msgid == MAVLINK_MSG_ID_HIL_SENSOR) {
        mavlink_hil_sensor_t s;
        mavlink_msg_hil_sensor_decode(&m, &s);
        t_us = s.time_usec;
        return true;
      }
    return false;
  };
  px4.ctrl(0.5f, 0.25f, 0.75f, 1.0f, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | MAV_MODE_FLAG_SAFETY_ARMED);
  std::uint64_t t_us = drain_until_quiet();
  ASSERT_GT(t_us, 1200000u);
  {
    const int row = wait_row(trace, t_us / 1000);  // the step after the controls were applied
    ASSERT_GE(row, 0);
    const Trace tr(trace);
    const auto r = static_cast<std::size_t>(row);
    EXPECT_EQ(tr.at(r, "throttle0"), 0.5);
    EXPECT_EQ(tr.at(r, "throttle1"), 0.25);
    EXPECT_EQ(tr.at(r, "throttle2"), 0.75);
    EXPECT_EQ(tr.at(r, "throttle3"), 1.0);
    EXPECT_EQ(tr.at(r, "armed"), 1);
    for (int i = 0; i < 4; ++i) EXPECT_NE(tr.at(r, "omega" + std::to_string(i)), 0.0);
    EXPECT_EQ(tr.at(0, "throttle0"), 0.0);  // and nothing before it
    EXPECT_EQ(tr.at(0, "armed"), 0);
    EXPECT_NEAR(tr.at(0, "wall_s"), std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(), 120);
    for (std::size_t i = 1; i < tr.rows.size(); ++i) EXPECT_EQ(tr.at(i, "tick") - tr.at(i - 1, "tick"), 4);
  }
  // one controls message releases exactly one step, then it blocks again
  px4.ctrl(0.5f, 0.25f, 0.75f, 1.0f, 129);
  std::uint64_t t2 = 0;
  ASSERT_TRUE(next_sensor(t2));
  EXPECT_EQ(t2, t_us + 4000);
  EXPECT_EQ(drain_until_quiet(), 0u);  // nothing more (the 300 ms silence check is inside)
  // disarmed: motor commands are ignored, throttles go to 0
  px4.ctrl(0.9f, 0.9f, 0.9f, 0.9f, 1);
  ASSERT_TRUE(next_sensor(t_us));
  EXPECT_EQ(t_us, t2 + 4000);
  {
    const int row = wait_row(trace, t_us / 1000);
    ASSERT_GE(row, 0);
    const Trace tr(trace);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(tr.at(static_cast<std::size_t>(row), "throttle" + std::to_string(i)), 0.0);
    EXPECT_EQ(tr.at(static_cast<std::size_t>(row), "armed"), 0);
  }

  // ---- junk on the MAVLink link is skipped and counted, never fatal ----
  {
    mavlink_message_t bad;
    mavlink_msg_heartbeat_pack(1, 1, &bad, MAV_TYPE_QUADROTOR, MAV_AUTOPILOT_PX4, 0, 0, MAV_STATE_ACTIVE);
    std::uint8_t b[MAVLINK_MAX_PACKET_LEN];
    const std::size_t n = mavlink_msg_to_send_buffer(b, &bad);
    b[n - 1] ^= 0xFF;  // corrupt the checksum
    EXPECT_EQ(::send(fd, b, n, 0), static_cast<ssize_t>(n));
    mavlink_msg_system_time_pack(1, 1, &bad, 1, 2);  // valid but not something the bridge handles
    px4.send(bad);
    const std::uint8_t junk[5] = {0x01, 0x02, 0x03, 0x04, 0x00};  // no STX byte (0xFD / 0xFE): a half frame would swallow the next real one
    EXPECT_EQ(::send(fd, junk, sizeof junk, 0), static_cast<ssize_t>(sizeof junk));
  }

  // ---- monitors on the true state, every tick ----
  bridge.write("monitor ok /devices/pack voltage 0 5 20 0.02\n");
  bridge.write("monitor bad /devices/pack voltage 0 100 200 0.008\n");
  bridge.write("monitor late /devices/pack voltage 0 0 100 10\n");  // outlives the run: reported at shutdown
  bridge.write("monitor nonsense /devices/nope voltage 0 0 1 1\n");  // refused (stderr), the run goes on
  usleep(200000);  // the bridge is blocked in its controls wait and reads stdin there
  const std::uint64_t tick_armed = t_us / 1000;
  std::vector<std::string> results;
  std::uint64_t end_us = t_us;
  for (int i = 0; i < 6; ++i) {
    px4.ctrl(0.4f, 0.4f, 0.4f, 0.4f, 129);
    ASSERT_TRUE(next_sensor(end_us));
  }
  EXPECT_EQ(end_us, t_us + 6 * 4000);
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(bridge.expect("monitor_result ", l, 10)) << "missing monitor_result " << i;
    results.push_back(l);
  }
  std::map<std::string, std::string> by_name;
  for (const std::string& r : results) by_name[r.substr(r.find("\"name\":\"") + 8, r.find('"', r.find("\"name\":\"") + 8) - r.find("\"name\":\"") - 8)] = r;
  ASSERT_EQ(by_name.count("ok"), 1u);
  ASSERT_EQ(by_name.count("bad"), 1u);
  EXPECT_NE(by_name["ok"].find("\"pass\":true"), std::string::npos) << by_name["ok"];
  EXPECT_EQ(num(by_name["ok"], "samples"), 20);  // 0.02 s at dt 1 ms: one sample per tick
  EXPECT_GT(num(by_name["ok"], "min"), 5);
  EXPECT_LT(num(by_name["ok"], "max"), 20);
  EXPECT_NE(by_name["bad"].find("\"pass\":false"), std::string::npos) << by_name["bad"];
  EXPECT_NE(by_name["bad"].find("\"hit\":true"), std::string::npos);
  EXPECT_EQ(num(by_name["bad"], "hit_tick"), static_cast<double>(tick_armed + 1));  // first tick after arming
  EXPECT_EQ(num(by_name["bad"], "samples"), 8);
  EXPECT_GT(num(by_name["bad"], "hit_value"), 5);

  // ---- clean shutdown ----
  bridge.write("quit\n");
  std::string final_line, stats_line, late_line;
  while (bridge.line(l, 10)) {
    if (l.rfind("monitor_result ", 0) == 0) late_line = l;
    else if (l.rfind("stats ", 0) == 0) stats_line = l;
    else if (l.rfind("final ", 0) == 0) final_line = l;
  }
  EXPECT_EQ(bridge.wait(), 0);
  EXPECT_NE(late_line.find("\"name\":\"late\""), std::string::npos) << late_line;  // unfinished monitor reported at shutdown
  EXPECT_EQ(num(late_line, "samples"), 24);                                          // 6 steps x 4 ticks since arming
  EXPECT_NE(late_line.find("\"pass\":true"), std::string::npos);
  std::uint64_t fin_tick = 0;
  char hash[32] = {};
  ASSERT_EQ(std::sscanf(final_line.c_str(), "final tick=%" SCNu64 " state_hash=%31s", &fin_tick, hash), 2) << final_line;
  EXPECT_EQ(fin_tick, end_us / 1000);
  EXPECT_NE(stats_line.find("set_errors=0"), std::string::npos) << stats_line;
  EXPECT_EQ(stats_line.find("skipped=0 "), std::string::npos) << stats_line;   // the SYSTEM_TIME frame
  EXPECT_EQ(stats_line.find("bad_frames=0 "), std::string::npos) << stats_line;  // the corrupted heartbeat

  // ---- the written log replays to the same state: the throttle SetAttrs were recorded ----
  ASSERT_TRUE(fs::exists(log));
  Proc replay({DVSIM_CLI_EXE, log});
  std::string rl;
  ASSERT_TRUE(replay.line(rl, 60));
  EXPECT_EQ(replay.wait(60), 0);
  char rhash[32] = {};
  std::uint64_t rtick = 0;
  ASSERT_EQ(std::sscanf(rl.c_str(), "tick=%" SCNu64 " state_hash=%31s", &rtick, rhash), 2) << rl;
  EXPECT_EQ(rtick, fin_tick);
  EXPECT_STREQ(rhash, hash);
  close(fd);
  close(udp_fd);
  fs::remove_all(dir);
}
}  // namespace
