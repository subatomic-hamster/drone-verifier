// dvsim-px4-bridge --config <toml> [--model <xml>] [--port 4560] [--px4-udp 18570] [--udp-bind 14550] [--speed 1.0]
//                 [--trace <csv>] [--log <replay log>] [--connect-timeout <s>] [--sysid <MAV_SYS_ID of PX4, default [px4] sysid>]
// PX4 SITL <-> dvsim World, lockstep, over the Simulator MAVLink API (docs/research/px4-sitl.md). We are the TCP SERVER on
// 127.0.0.1:<port>; PX4 connects. Per step (4 ms of sim time): step the World, send HIL_SENSOR (id 0, drives PX4's lockstep
// clock) + HIL_GPS when the GNSS sampled + HIL_STATE_QUATERNION, then block for HIL_ACTUATOR_CONTROLS and apply it to the
// ESC `throttle` attrs through World::execute (so the replay log records it). The battery goes as BATTERY_STATUS over a
// normal MAVLink UDP link (HIL_SENSOR has no battery field and simulator_mavlink does not dispatch BATTERY_STATUS).
//
// stdin: `monitor <name> <prim> <attr> <index> <lo> <hi> <duration_s>` | `quit`.   (one `always` Monitor per line)
// --ctl <unix socket>: a text-line control socket onto the same World (set / get / mget / tick / imu, see Bridge::on_ctl), so a second
// robot's user-side driver (the Feetech pty of `dvsim program`) or a companion computer's sensors (examples/skydio/px4_fly.py: depth frames, the
// IMU history) act in this lockstep world; its writes are logged and replayed. No `gnss` key in the config = GPS-denied (no HIL_GPS).
// stdout: `listening <port>`, `connected`, `monitor_result {json}`, `stats ...`, `final tick=<n> state_hash=<hex>`.
// Exit: 0 clean shutdown (PX4 disconnect, quit, SIGINT/SIGTERM), 1 error / connect timeout / solver divergence, 2 usage.
//
// Startup lockstep: until the FIRST HIL_ACTUATOR_CONTROLS has arrived we do not wait (PX4 is still booting: it answers
// sensor messages only once its modules run, and waiting there would deadlock; the Gazebo plugin behaves the same). From
// then on every step waits for a fresh HIL_ACTUATOR_CONTROLS (one sent after that step's HIL_SENSOR).
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <common/mavlink.h>
#include <toml++/toml.hpp>

#include "core/monitor.h"
#include "core/world.h"
#include "flatbuffers/flatbuffers.h"
#include "mujoco/mujoco_solver.h"
#include "dvsim_generated.h"
#include "px4/frames.h"

namespace {
using namespace px4f;
using Clock = std::chrono::steady_clock;

volatile std::sig_atomic_t g_signal = 0;
void on_signal(int) { g_signal = 1; }

constexpr double kHilDt = 0.004;  // s of sim time per lockstep step (PX4 SITL IMU_INTEG_RATE 250)
constexpr double kEarthR = 6378137.0;

struct Cfg {
  std::string model, base, imu, baro, mag, gnss, battery;
  std::vector<std::string> esc;  // PX4 motor index -> ESC device
  double home[3] = {0, 0, 0};    // lat deg, lon deg, alt m MSL
  double gps_rate_hz = 0;        // optional cross-check against the model (0 = unset)
  int sats = 10, sysid = 1, compid = 180, cells = 4;
  double battery_rate_hz = 5;
};

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error(m); }

Cfg load_cfg(const std::string& path, const std::string& model_override) {
  const toml::table t = toml::parse_file(path);
  auto req = [&](const toml::table& tb, const char* k) {
    auto v = tb[k].value<std::string>();
    if (!v) fail(std::string("config: missing string key '") + k + "'");
    return *v;
  };
  Cfg c;
  c.model = model_override;
  if (c.model.empty()) {
    c.model = req(t, "model");
    if (std::filesystem::path(c.model).is_relative()) c.model = (std::filesystem::path(path).parent_path() / c.model).string();
  }
  c.base = req(t, "base");
  c.imu = req(t, "imu");
  c.baro = req(t, "baro");
  c.mag = req(t, "mag");
  c.gnss = t["gnss"].value_or("");  // optional: GPS-denied flight (vision odometry from a companion computer) sends no HIL_GPS
  c.battery = req(t, "battery");
  c.gps_rate_hz = t["gps_rate_hz"].value_or(0.0);
  c.sats = t["satellites"].value_or(10);
  const toml::array* esc = t["motors"]["esc"].as_array();
  if (!esc || esc->empty()) fail("config: [motors] esc must be a non-empty array of ESC device names");
  for (const toml::node& n : *esc) {
    auto s = n.value<std::string>();
    if (!s) fail("config: [motors] esc entries must be strings");
    c.esc.push_back(*s);
  }
  if (esc->size() > 16) fail("config: at most 16 motors");
  auto h = [&](const char* k) {
    auto v = t["home"][k].value<double>();
    if (!v) fail(std::string("config: missing [home] ") + k);
    return *v;
  };
  c.home[0] = h("lat"), c.home[1] = h("lon"), c.home[2] = h("alt");
  c.sysid = t["px4"]["sysid"].value_or(1);
  c.compid = t["px4"]["compid"].value_or(180);
  c.battery_rate_hz = t["px4"]["battery_rate_hz"].value_or(5.0);
  c.cells = t["px4"]["cells"].value_or(4);
  if (c.cells < 1 || c.cells > 10) fail("config: px4.cells must be 1..10");
  if (c.battery_rate_hz <= 0) fail("config: px4.battery_rate_hz must be > 0");
  return c;
}

template <class G> const typename G::Cfg& find(const G& g, const std::string& name, const char* what) {
  for (const auto& c : g.cfg)
    if (c.meta.name == name) return c;
  fail(std::string("model has no ") + what + " device '" + name + "'");
}

int bind_listen(int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  if (bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || listen(fd, 1) != 0) {
    const std::string e = std::strerror(errno);
    close(fd);
    fail("cannot listen on 127.0.0.1:" + std::to_string(port) + ": " + e);
  }
  return fd;
}

int bind_udp(int port) {
  const int fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<std::uint16_t>(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  if (bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
    const std::string e = std::strerror(errno);
    close(fd);
    fail("cannot bind udp 127.0.0.1:" + std::to_string(port) + ": " + e);
  }
  return fd;
}

int bind_unix(const std::string& path) {  // --ctl: a unix stream socket a second robot's user-side driver talks to
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  if (path.size() >= sizeof a.sun_path) fail("--ctl path too long: " + path);
  std::strncpy(a.sun_path, path.c_str(), sizeof a.sun_path - 1);
  const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
  ::unlink(path.c_str());
  // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
  if (bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || listen(fd, 1) != 0) {
    const std::string e = std::strerror(errno);
    close(fd);
    fail("cannot listen on unix socket " + path + ": " + e);
  }
  return fd;
}

std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}

struct Bridge {
  Cfg cfg;
  std::unique_ptr<World> w;
  Session session;
  const devs::Devices* dv = nullptr;
  double dt = 0;
  std::uint64_t n = 0;                          // ticks per HIL step
  std::uint64_t p_imu = 0, p_baro = 0, p_mag = 0, p_gnss = 0, p_batt = 0, p_hb = 0;  // ticks per sample / message
  std::string imu, baro, mag, gnss, batt, base_joint;  // prim paths
  std::vector<std::string> esc;                 // prim paths, PX4 motor index order
  double capacity_ah = 0, soc0 = 0;
  // io
  int tcp = -1, udp = -1;
  sockaddr_in px4_udp{};
  bool stdin_eof = false, quit = false, peer_gone = false;
  std::string inbuf;
  // protocol state
  bool ever_ctrl = false, fresh_ctrl = false, armed = false;
  std::vector<double> throttle;  // last applied, per motor
  std::uint64_t bad = 0, skipped = 0, steps = 0, set_errors = 0;
  struct Mon { std::unique_ptr<Monitor> m; std::string prim, attr; int index; std::uint64_t to; };
  std::vector<Mon> mons;
  FILE* trace = nullptr;
  // The IMU samples sent to PX4 (one per HIL step, device FLU frame), kept for 16 s of sim time so a companion computer's estimator on the
  // ctl socket sees every sample, not whichever one is latest when it polls (`imu <since>`, below).
  struct ImuRec { std::uint64_t counter; V3 gyro, accel; };
  static constexpr std::size_t kImuHist = 4000;
  std::deque<ImuRec> imu_hist;

  bool stopping() const { return g_signal || quit || peer_gone; }

  std::vector<double> q(const std::string& prim, const std::string& attr) const {
    std::vector<double> v;
    const Result r = w->solver().get(prim, attr, "", v);
    if (r.code != Result::ok) fail("query " + prim + " " + attr + ": " + r.message);
    return v;
  }
  V3 q3(const std::string& prim, const std::string& attr) const {
    const auto v = q(prim, attr);
    return {v.at(0), v.at(1), v.at(2)};
  }
  double q1(const std::string& prim, const std::string& attr) const { return q(prim, attr).at(0); }

  // ---- MAVLink out ----
  bool send_tcp(const mavlink_message_t& m) {
    std::uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    const std::size_t len = mavlink_msg_to_send_buffer(buf, &m);
    std::size_t off = 0;
    while (off < len) {
      const ssize_t r = ::send(tcp, buf + off, len - off, MSG_NOSIGNAL);
      if (r > 0) off += static_cast<std::size_t>(r);
      else if (r < 0 && errno == EINTR) { if (g_signal) return false; }
      else if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        pollfd p{tcp, POLLOUT, 0};
        ::poll(&p, 1, 100);
        if (stopping()) return false;
      } else {
        peer_gone = true;
        return false;
      }
    }
    return true;
  }
  void send_udp(const mavlink_message_t& m) {
    std::uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    const std::size_t len = mavlink_msg_to_send_buffer(buf, &m);
    // ttsr-ignore: cpp/no-reinterpret-const-cast (the BSD sockets API takes sockaddr*)
    ::sendto(udp, buf, len, 0, reinterpret_cast<const sockaddr*>(&px4_udp), sizeof px4_udp);  // lossy by nature
  }

  // ---- controls in ----
  void set_throttle(std::size_t i, double v) {
    flatbuffers::FlatBufferBuilder fb;
    const auto sa = dvsim::CreateSetAttr(fb, fb.CreateString(esc[i]), fb.CreateString("throttle"), 0, fb.CreateVector(&v, 1));
    fb.Finish(dvsim::CreateEnvelope(fb, 0, fb.CreateString("px4"), dvsim::Command::SetAttr, sa.Union()));
    const auto reply = w->execute({fb.GetBufferPointer(), fb.GetSize()}, session);
    if (!flatbuffers::GetRoot<dvsim::Reply>(reply.data())->ok()) {
      ++set_errors;
      std::fprintf(stderr, "dvsim-px4-bridge: SetAttr %s throttle=%g failed\n", esc[i].c_str(), v);
    } else {
      throttle[i] = v;
    }
  }
  void on_controls(const mavlink_hil_actuator_controls_t& c) {
    armed = (c.mode & MAV_MODE_FLAG_SAFETY_ARMED) != 0;
    for (std::size_t i = 0; i < esc.size(); ++i) {
      const double raw = c.controls[i];
      const double v = armed && std::isfinite(raw) ? std::clamp(raw, 0.0, 1.0) : 0.0;
      if (v != throttle[i]) set_throttle(i, v);
    }
    ever_ctrl = fresh_ctrl = true;
  }

  // ---- io pump: tcp + udp + stdin; timeout_ms < 0 never used ----
  void on_msg(const mavlink_message_t& m) {
    if (m.msgid == MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS) {
      mavlink_hil_actuator_controls_t c;
      mavlink_msg_hil_actuator_controls_decode(&m, &c);
      on_controls(c);
    } else if (m.msgid != MAVLINK_MSG_ID_HEARTBEAT) {
      ++skipped;
    } else {
      saw_heartbeat = true;
    }
  }
  bool saw_heartbeat = false;
  std::uint8_t last_parse_error = 0;

  void pump_tcp() {
    std::uint8_t buf[4096];
    mavlink_message_t m;
    mavlink_status_t st;
    for (;;) {
      const ssize_t r = ::recv(tcp, buf, sizeof buf, 0);
      if (r == 0) { peer_gone = true; return; }
      if (r < 0) {
        if (errno == EINTR) return;
        if (errno != EAGAIN && errno != EWOULDBLOCK) peer_gone = true;
        return;
      }
      for (ssize_t i = 0; i < r; ++i) {
        if (mavlink_parse_char(MAVLINK_COMM_0, buf[i], &m, &st) == MAVLINK_FRAMING_OK) on_msg(m);
        // mavlink_parse_char returns 0 for a bad CRC / signature; the channel's cumulative (uint8) parse_error counter is the signal
        const std::uint8_t pe = mavlink_get_channel_status(MAVLINK_COMM_0)->parse_error;
        bad += static_cast<std::uint8_t>(pe - last_parse_error);
        last_parse_error = pe;
      }
    }
  }
  void pump_stdin() {
    char buf[1024];
    const ssize_t r = ::read(0, buf, sizeof buf);
    if (r == 0) { stdin_eof = true; return; }
    if (r < 0) return;
    inbuf.append(buf, static_cast<std::size_t>(r));
    for (std::size_t nl; (nl = inbuf.find('\n')) != std::string::npos;) {
      on_line(inbuf.substr(0, nl));
      inbuf.erase(0, nl + 1);
    }
  }
  // ---- ctl socket (--ctl): one client at a time, one text line per request, one per reply. Runs between steps on the World's
  // thread, and every state change goes through World::execute (so the replay log records it, like the ESC throttles):
  //   set <source> <prim> <attr> <v>...  -> ok | err <message>
  //   get <prim> <attr> [frame]          -> ok <tick> <v>...        (frame: body prim path, as Query)
  //   mget <prim> <attr>...              -> ok <tick> <v>...        (one scalar per attr: a whole device in one round trip)
  //   tick                               -> ok <tick>
  //   imu <since>                        -> ok <tick> <n> (<counter> <gx gy gz> <ax ay az>)...   the n IMU samples with counter > since, oldest first
  //                                         (counter = the world tick the sample was taken at; device FLU frame, rad/s and m/s^2; 16 s history)
  int ctl_listen = -1, ctl_fd = -1;
  std::string ctl_in;
  void close_ctl() {
    if (ctl_fd >= 0) close(ctl_fd);
    ctl_fd = -1;
    ctl_in.clear();
  }
  void ctl_reply(const std::string& s) {
    const std::string out = s + "\n";
    std::size_t off = 0;
    while (off < out.size() && ctl_fd >= 0) {
      const ssize_t r = ::send(ctl_fd, out.data() + off, out.size() - off, MSG_NOSIGNAL);
      if (r > 0) off += static_cast<std::size_t>(r);
      else if (r < 0 && (errno == EAGAIN || errno == EINTR)) { pollfd p{ctl_fd, POLLOUT, 0}; ::poll(&p, 1, 100); }
      else close_ctl();
    }
  }
  void on_ctl(const std::string& line) {
    std::istringstream is(line);
    std::string cmd, a, b, c;
    is >> cmd;
    const std::string tk = std::to_string(w->tick());
    if (cmd == "tick") return ctl_reply("ok " + tk);
    if (cmd == "imu") {
      std::uint64_t since = 0;
      if (!(is >> since)) return ctl_reply("err usage: imu <since counter>");
      std::string out;
      std::size_t cnt = 0;
      for (const ImuRec& r : imu_hist) {
        if (r.counter <= since) continue;
        out += " " + std::to_string(r.counter);
        for (double d : r.gyro) out += " " + jnum(d);
        for (double d : r.accel) out += " " + jnum(d);
        ++cnt;
      }
      return ctl_reply("ok " + tk + " " + std::to_string(cnt) + out);
    }
    if (cmd == "set") {
      std::vector<double> v;
      double x;
      if (!(is >> a >> b >> c)) return ctl_reply("err usage: set <source> <prim> <attr> <v>...");
      while (is >> x) v.push_back(x);
      flatbuffers::FlatBufferBuilder fb;
      const auto sa = dvsim::CreateSetAttr(fb, fb.CreateString(b), fb.CreateString(c), fb.CreateString(""), fb.CreateVector(v));
      fb.Finish(dvsim::CreateEnvelope(fb, 0, fb.CreateString(a), dvsim::Command::SetAttr, sa.Union()));
      const auto reply = w->execute({fb.GetBufferPointer(), fb.GetSize()}, session);
      const auto* r = flatbuffers::GetRoot<dvsim::Reply>(reply.data());
      if (r->ok()) return ctl_reply("ok");
      std::string m = r->message() ? r->message()->str() : "failed";
      std::replace(m.begin(), m.end(), '\n', ' ');
      return ctl_reply("err " + m);
    }
    if (cmd == "get" || cmd == "mget") {
      if (!(is >> a)) return ctl_reply("err usage: get <prim> <attr> [frame] | mget <prim> <attr>...");
      std::vector<std::string> attrs;
      while (is >> b) attrs.push_back(b);
      if (attrs.empty()) return ctl_reply("err missing attr");
      std::string out = "ok " + tk;
      for (std::size_t i = 0; i < (cmd == "get" ? 1 : attrs.size()); ++i) {
        std::vector<double> v;
        const Result r = w->solver().get(a, attrs[i], cmd == "get" && attrs.size() > 1 ? attrs[1] : "", v);
        if (r.code != Result::ok) return ctl_reply("err " + a + " " + attrs[i] + ": " + r.message);
        if (cmd == "mget" && v.size() != 1) return ctl_reply("err mget needs scalar attrs: " + attrs[i]);
        for (double d : v) out += " " + jnum(d);
      }
      return ctl_reply(out);
    }
    ctl_reply("err unknown ctl command '" + cmd + "'");
  }
  void pump_ctl() {
    if (ctl_fd < 0) return;
    char buf[2048];
    for (;;) {
      const ssize_t r = ::recv(ctl_fd, buf, sizeof buf, 0);
      if (r == 0) return close_ctl();
      if (r < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) close_ctl();
        return;
      }
      ctl_in.append(buf, static_cast<std::size_t>(r));
      for (std::size_t nl; ctl_fd >= 0 && (nl = ctl_in.find('\n')) != std::string::npos;) {
        const std::string line = ctl_in.substr(0, nl);
        ctl_in.erase(0, nl + 1);
        on_ctl(line);
      }
    }
  }
  void io(int timeout_ms) {
    pollfd p[5] = {{tcp, POLLIN, 0}, {udp, POLLIN, 0}, {0, POLLIN, 0}, {ctl_listen, POLLIN, 0}, {ctl_fd, POLLIN, 0}};  // fd -1: ignored
    if (stdin_eof) p[2].fd = -1;
    if (::poll(p, 5, timeout_ms) <= 0) return;
    if (p[0].revents) pump_tcp();
    if (p[1].revents) {
      char d[2048];  // PX4 telemetry on the UDP link: drained, ignored
      while (::recvfrom(udp, d, sizeof d, 0, nullptr, nullptr) >= 0) {}
    }
    if (!stdin_eof && p[2].revents) pump_stdin();
    if (p[3].revents) {  // a new client replaces the old one
      const int fd = ::accept4(ctl_listen, nullptr, nullptr, SOCK_NONBLOCK);
      if (fd >= 0) { close_ctl(); ctl_fd = fd; }
    }
    if (p[4].revents && ctl_fd == p[4].fd) pump_ctl();
  }

  // ---- monitors ----
  void on_line(const std::string& line) {
    std::istringstream is(line);
    std::string cmd;
    is >> cmd;
    if (cmd == "quit") { quit = true; return; }
    if (cmd.empty()) return;
    if (cmd != "monitor") { std::fprintf(stderr, "dvsim-px4-bridge: unknown stdin command '%s'\n", cmd.c_str()); return; }
    std::string name, prim, attr;
    int index = 0;
    double lo = 0, hi = 0, dur = 0;
    if (!(is >> name >> prim >> attr >> index >> lo >> hi >> dur) || dur <= 0 || index < 0 ||
        name.find_first_of("\"\\") != std::string::npos) {
      std::fprintf(stderr, "dvsim-px4-bridge: bad monitor line: %s\n", line.c_str());
      return;
    }
    std::vector<double> v;
    const Result r = w->solver().get(prim, attr, "", v);
    if (r.code != Result::ok || static_cast<std::size_t>(index) >= v.size()) {
      std::fprintf(stderr, "dvsim-px4-bridge: monitor '%s': cannot read %s %s[%d]\n", name.c_str(), prim.c_str(), attr.c_str(), index);
      return;
    }
    const std::uint64_t from = w->tick() + 1, len = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::llround(dur / dt)));
    mons.push_back({std::make_unique<Monitor>(name, true, lo, hi, from, from + len - 1), prim, attr, index, from + len - 1});
  }
  void print_monitor(const Monitor& m) {
    const MonitorResult r = m.result();
    std::printf("monitor_result {\"name\":\"%s\",\"pass\":%s,\"hit\":%s,\"hit_tick\":%" PRIu64 ",\"hit_value\":%s,\"samples\":%" PRIu64
                ",\"min\":%s,\"max\":%s,\"rms\":%s,\"final\":%s}\n",
                r.name.c_str(), r.pass ? "true" : "false", r.hit ? "true" : "false", r.hit_tick, jnum(r.hit_value).c_str(), r.samples,
                jnum(r.min_value).c_str(), jnum(r.max_value).c_str(), jnum(r.rms).c_str(), jnum(r.final_value).c_str());
    std::fflush(stdout);
  }
  // Every tick while any monitor is armed (single-tick steps), else one w->step(n).
  void advance() {
    if (mons.empty()) { w->step(static_cast<int>(n)); return; }
    for (std::uint64_t i = 0; i < n; ++i) {
      w->step(1);
      for (Mon& m : mons) {
        std::vector<double> v;
        const Result r = w->solver().get(m.prim, m.attr, "", v);
        m.m->sample(w->tick(), r.code == Result::ok && static_cast<std::size_t>(m.index) < v.size() ? v[static_cast<std::size_t>(m.index)] : NAN);
      }
      for (std::size_t k = mons.size(); k-- > 0;)
        if (w->tick() >= mons[k].to) { print_monitor(*mons[k].m); mons.erase(mons.begin() + static_cast<std::ptrdiff_t>(k)); }
    }
  }

  // ---- outgoing sensor messages for the step that just finished (ticks [tick0, tick0 + n)) ----
  void send_step(std::uint64_t tick0, V3& acc_frd) {
    const std::uint64_t tick = w->tick();
    const auto t_us = static_cast<std::uint64_t>(std::llround(static_cast<double>(tick) * dt * 1e6));
    const std::uint8_t sys = static_cast<std::uint8_t>(cfg.sysid);
    mavlink_message_t m;

    const V3 acc_flu = q3(imu, "accel"), gyr_flu = q3(imu, "gyro");
    acc_frd = flu_to_frd(acc_flu);
    const V3 gyr = flu_to_frd(gyr_flu);
    imu_hist.push_back({(tick - 1) / p_imu * p_imu, gyr_flu, acc_flu});  // the sample taken at counter c, visible after tick c + 1
    if (imu_hist.size() > kImuHist) imu_hist.pop_front();
    V3 mg = flu_to_frd(q3(mag, "field"));
    for (double& c : mg) c = tesla_to_gauss(c);
    const double p = q1(baro, "pressure");
    const bool fb = sampled(tick0, n, p_baro), fm = sampled(tick0, n, p_mag);
    mavlink_msg_hil_sensor_pack(sys, 1, &m, t_us, static_cast<float>(acc_frd[0]), static_cast<float>(acc_frd[1]), static_cast<float>(acc_frd[2]),
                                static_cast<float>(gyr[0]), static_cast<float>(gyr[1]), static_cast<float>(gyr[2]), static_cast<float>(mg[0]),
                                static_cast<float>(mg[1]), static_cast<float>(mg[2]), static_cast<float>(pa_to_hpa(p)), 0.0f,
                                static_cast<float>(pressure_alt(p)), static_cast<float>(isa_temp_c(p)),
                                fields_updated(sampled(tick0, n, p_imu), fb, fm), 0);
    if (!send_tcp(m)) return;

    if (gnss_cfg && sampled(tick0, n, p_gnss)) {
      const auto& g = *gnss_cfg;
      const V3 vn = q3(gnss, "vel_ned");
      // Fault gnss_outage (device attr fix_lost = 1): a receiver without a fix reports fix_type 0, no satellites, unknown accuracy.
      const bool lost = q1(gnss, "fix_lost") > 0.5;
      mavlink_msg_hil_gps_pack(sys, 1, &m, t_us, lost ? 0 : 3, deg_e7(q1(gnss, "lat")), deg_e7(q1(gnss, "lon")), mm(q1(gnss, "alt")),
                               lost ? UINT16_MAX : cm_u16(g.horiz_std), lost ? UINT16_MAX : cm_u16(g.vert_std), cm_u16(std::hypot(vn[0], vn[1])),
                               cm_s_i16(vn[0]), cm_s_i16(vn[1]), cm_s_i16(vn[2]), cog_cdeg(vn[0], vn[1]),
                               static_cast<std::uint8_t>(lost ? 0 : cfg.sats), 0, 0);
      if (!send_tcp(m)) return;
    }

    // Ground truth from the base body.
    const V3 pos = q3(cfg.base, "pos");
    const auto qq = q(cfg.base, "quat");
    const auto qv = q(base_joint, "qvel");  // free joint: linear world, angular body (FLU)
    const Q4 qn = q_ned_frd({qq.at(0), qq.at(1), qq.at(2), qq.at(3)});
    const float qf[4] = {static_cast<float>(qn[0]), static_cast<float>(qn[1]), static_cast<float>(qn[2]), static_cast<float>(qn[3])};
    const V3 w_frd = flu_to_frd({qv.at(3), qv.at(4), qv.at(5)});
    const V3 v_ned = enu_to_ned({qv.at(0), qv.at(1), qv.at(2)});
    const V3 p_ned = enu_to_ned(pos);
    const double lat = cfg.home[0] + p_ned[0] / kEarthR * 180.0 / 3.14159265358979323846;
    const double lon = cfg.home[1] + p_ned[1] / (kEarthR * std::cos(cfg.home[0] * 3.14159265358979323846 / 180.0)) * 180.0 / 3.14159265358979323846;
    auto mg_i16 = [](double a) { return static_cast<std::int16_t>(std::clamp(std::llround(a / kG0 * 1000.0), -32768LL, 32767LL)); };
    mavlink_msg_hil_state_quaternion_pack(sys, 1, &m, t_us, qf, static_cast<float>(w_frd[0]), static_cast<float>(w_frd[1]), static_cast<float>(w_frd[2]),
                                          deg_e7(lat), deg_e7(lon), mm(cfg.home[2] - p_ned[2]), cm_s_i16(v_ned[0]), cm_s_i16(v_ned[1]),
                                          cm_s_i16(v_ned[2]), 0, 0, mg_i16(acc_frd[0]), mg_i16(acc_frd[1]), mg_i16(acc_frd[2]));
    send_tcp(m);
  }

  void send_udp_link(std::uint64_t tick0) {
    const std::uint8_t sys = static_cast<std::uint8_t>(cfg.sysid), comp = static_cast<std::uint8_t>(cfg.compid);
    mavlink_message_t m;
    if (sampled(tick0, n, p_hb)) {
      mavlink_msg_heartbeat_pack(sys, comp, &m, MAV_TYPE_ONBOARD_CONTROLLER, MAV_AUTOPILOT_INVALID, 0, 0, MAV_STATE_ACTIVE);
      send_udp(m);
    }
    if (sampled(tick0, n, p_batt)) {
      std::uint16_t volts[10], ext[4] = {0, 0, 0, 0};
      std::fill(volts, volts + 10, UINT16_MAX);
      const double v = q1(batt, "voltage"), i = q1(batt, "current"), soc = q1(batt, "soc");
      for (int c = 0; c < cfg.cells; ++c) volts[c] = static_cast<std::uint16_t>(std::clamp(std::llround(v / cfg.cells * 1000.0), 0LL, 65534LL));
      const auto consumed = static_cast<std::int32_t>(std::llround((soc0 - soc) * capacity_ah * 1000.0));  // mAh since start
      mavlink_msg_battery_status_pack(sys, comp, &m, 0, MAV_BATTERY_FUNCTION_ALL, MAV_BATTERY_TYPE_LIPO, INT16_MAX, volts,
                                      static_cast<std::int16_t>(std::clamp(std::llround(i * 100.0), -32768LL, 32767LL)), consumed, -1,
                                      static_cast<std::int8_t>(std::clamp(std::llround(soc * 100.0), 0LL, 100LL)), 0, MAV_BATTERY_CHARGE_STATE_UNDEFINED,
                                      ext, MAV_BATTERY_MODE_UNKNOWN, 0);
      send_udp(m);
    }
  }

  void write_trace() {
    if (!trace) return;
    const double wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const V3 pos = q3(cfg.base, "pos");
    const auto qq = q(cfg.base, "quat"), qv = q(base_joint, "qvel");
    std::fprintf(trace, "%.6f,%.6f,%" PRIu64 ",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g", wall,
                 static_cast<double>(w->tick()) * dt, w->tick(), pos[0], pos[1], pos[2], qv.at(0), qv.at(1), qv.at(2), qq.at(0), qq.at(1), qq.at(2), qq.at(3));
    for (const char* a : {"throttle", "bus_current", "current", "omega"})
      for (const std::string& e : esc) std::fprintf(trace, ",%.9g", q1(e, a));
    std::fprintf(trace, ",%.9g,%.9g,%.9g,%d\n", q1(batt, "voltage"), q1(batt, "current"), q1(batt, "soc"), armed ? 1 : 0);
    std::fflush(trace);
  }

  const devs::GnssCfg* gnss_cfg = nullptr;
};

int usage() {
  std::fprintf(stderr,
               "usage: dvsim-px4-bridge --config <toml> [--model <xml>] [--port 4560] [--px4-udp 18570] [--udp-bind 14550]\n"
               "                       [--speed 1.0] [--trace <csv>] [--log <out>] [--connect-timeout <s>] [--sysid <n>] [--ctl <unix socket>]\n");
  return 2;
}

// Accept PX4 and read its HEARTBEAT, within `timeout` seconds. False on timeout / stop.
bool wait_for_px4(Bridge& b, int listen_fd, double timeout) {
  const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
  while (b.tcp < 0) {
    if (g_signal || Clock::now() >= deadline) return false;
    pollfd p{listen_fd, POLLIN, 0};
    if (::poll(&p, 1, 100) > 0) {
      b.tcp = ::accept(listen_fd, nullptr, nullptr);
      if (b.tcp >= 0) {
        int one = 1;
        setsockopt(b.tcp, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        fcntl(b.tcp, F_SETFL, fcntl(b.tcp, F_GETFL) | O_NONBLOCK);
      }
    }
  }
  while (!b.saw_heartbeat) {
    if (g_signal || b.peer_gone || Clock::now() >= deadline) return false;
    b.io(100);
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  std::string cfg_path, model, trace_path, log_path, ctl_path;
  int port = 4560, px4_udp = 18570, udp_bind = 14550, sysid = -1;
  double speed = 1.0, connect_timeout = 60;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (i + 1 >= argc) return usage();
      const std::string v = argv[++i];
      if (a == "--config") cfg_path = v;
      else if (a == "--model") model = v;
      else if (a == "--port") port = std::stoi(v);
      else if (a == "--px4-udp") px4_udp = std::stoi(v);
      else if (a == "--udp-bind") udp_bind = std::stoi(v);
      else if (a == "--speed") speed = std::stod(v);
      else if (a == "--trace") trace_path = v;
      else if (a == "--log") log_path = v;
      else if (a == "--connect-timeout") connect_timeout = std::stod(v);
      else if (a == "--sysid") sysid = std::stoi(v);
      else if (a == "--ctl") ctl_path = v;
      else return usage();
    }
  } catch (const std::exception&) {
    return usage();
  }
  if (cfg_path.empty() || speed < 0) return usage();

  struct sigaction sa{};
  sa.sa_handler = on_signal;  // no SA_RESTART: poll() returns EINTR and the loops see g_signal
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);

  Bridge b;
  int listen_fd = -1;
  int rc = 0;
  try {
    b.cfg = load_cfg(cfg_path, model);
    if (sysid >= 0) b.cfg.sysid = sysid;  // PX4 instance N runs with MAV_SYS_ID N + 1 and drops BATTERY_STATUS from another system
    const Cfg& c = b.cfg;
    b.w = std::make_unique<World>(load_mujoco, c.model);
    if (!log_path.empty()) b.w->start_log();
    b.session.caps = {"read", "control"};
    const auto* ms = dynamic_cast<const MujocoSolver*>(&b.w->solver());
    b.dv = ms ? ms->devices() : nullptr;
    if (!b.dv) fail("model '" + c.model + "' declares no dvsim.devices");
    b.dt = b.w->solver().dt();
    b.n = static_cast<std::uint64_t>(std::llround(kHilDt / b.dt));
    if (b.n < 1 || std::fabs(static_cast<double>(b.n) * b.dt - kHilDt) > 1e-9) fail("model dt must divide the 4 ms HIL step");
    const devs::Devices& dv = *b.dv;
    const auto& imu = find(dv.imu, c.imu, "imu");
    const auto& baro = find(dv.baro, c.baro, "baro");
    const auto& mag = find(dv.mag, c.mag, "mag");
    if (!c.gnss.empty()) b.gnss_cfg = &find(dv.gnss, c.gnss, "gnss");
    const auto& bat = find(dv.battery, c.battery, "battery");
    b.p_imu = static_cast<std::uint64_t>(imu.period), b.p_baro = static_cast<std::uint64_t>(baro.period);
    b.p_mag = static_cast<std::uint64_t>(mag.period);
    if (b.gnss_cfg) b.p_gnss = static_cast<std::uint64_t>(b.gnss_cfg->period);
    if (b.p_imu < 1 || b.p_imu > b.n) fail("the IMU must sample at least once per 4 ms HIL step (rate_hz >= 250)");
    if (b.gnss_cfg && c.gps_rate_hz > 0 && std::fabs(c.gps_rate_hz - b.gnss_cfg->rate_hz) > 1e-9) fail("config gps_rate_hz differs from the gnss device rate_hz in the model");
    const double bt = 1.0 / (c.battery_rate_hz * b.dt), ht = 1.0 / b.dt;
    if (std::fabs(bt - std::round(bt)) > 1e-6) fail("1 / (battery_rate_hz * dt) must be an integer");
    b.p_batt = static_cast<std::uint64_t>(std::llround(bt)), b.p_hb = static_cast<std::uint64_t>(std::llround(ht));
    if (b.gnss_cfg && (std::fabs(b.gnss_cfg->lat0 - c.home[0]) > 1e-7 || std::fabs(b.gnss_cfg->lon0 - c.home[1]) > 1e-7 ||
                       std::fabs(b.gnss_cfg->alt0 - c.home[2]) > 1e-3))
      fail("config [home] differs from the gnss device origin (lat0/lon0/alt0) in the model");
    if (!dv.env.cfg.empty() && std::fabs(dv.env.cfg[0].origin_alt - c.home[2]) > 1e-3)
      std::fprintf(stderr, "dvsim-px4-bridge: warning: environment origin_alt %g != [home] alt %g\n", dv.env.cfg[0].origin_alt, c.home[2]);
    for (const std::string& e : c.esc) {
      find(dv.esc, e, "esc");
      b.esc.push_back("/devices/" + e);
    }
    b.throttle.assign(b.esc.size(), 0.0);
    b.imu = "/devices/" + c.imu, b.baro = "/devices/" + c.baro, b.mag = "/devices/" + c.mag, b.gnss = "/devices/" + c.gnss;
    b.batt = "/devices/" + c.battery;
    b.capacity_ah = bat.capacity_ah;
    b.soc0 = b.q1(b.batt, "soc");
    for (const Prim& p : b.w->solver().prims())  // the base body's free joint carries its velocity
      if (p.kind == "joint" && p.path.rfind(c.base + "/", 0) == 0) {
        std::vector<double> v;
        if (b.w->solver().get(p.path, "qpos", "", v).code == Result::ok && v.size() == 7) { b.base_joint = p.path; break; }
      }
    if (b.base_joint.empty()) fail("base body '" + c.base + "' has no free joint");
    b.q3(c.base, "pos");  // base prim exists

    if (!trace_path.empty()) {
      b.trace = std::fopen(trace_path.c_str(), "w");
      if (!b.trace) fail("cannot open trace '" + trace_path + "'");
      std::fprintf(b.trace, "wall_s,t,tick,x,y,z,vx,vy,vz,qw,qx,qy,qz");
      for (const char* a : {"throttle", "bus_current", "current", "omega"})
        for (std::size_t i = 0; i < b.esc.size(); ++i) std::fprintf(b.trace, ",%s%zu", a, i);
      std::fprintf(b.trace, ",batt_v,batt_i,batt_soc,armed\n");
    }

    b.udp = bind_udp(udp_bind);
    if (!ctl_path.empty()) b.ctl_listen = bind_unix(ctl_path);
    b.px4_udp.sin_family = AF_INET;
    b.px4_udp.sin_port = htons(static_cast<std::uint16_t>(px4_udp));
    b.px4_udp.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    listen_fd = bind_listen(port);
    std::printf("listening %d\n", port);
    std::fflush(stdout);
    if (!wait_for_px4(b, listen_fd, connect_timeout)) {
      if (g_signal) b.quit = true;
      else if (b.peer_gone) b.quit = true;
      else fail("timed out after " + std::to_string(connect_timeout) + " s waiting for PX4 on tcp 127.0.0.1:" + std::to_string(port));
    } else {
      std::printf("connected\n");
      std::fflush(stdout);
    }
    const auto wall0 = Clock::now();
    V3 acc{};
    while (!b.stopping()) {
      b.io(0);
      if (b.stopping()) break;
      const std::uint64_t tick0 = b.w->tick();
      b.advance();
      if (b.w->diverged()) fail("solver diverged at tick " + std::to_string(b.w->tick()));
      b.fresh_ctrl = false;  // only a HIL_ACTUATOR_CONTROLS sent after this step's sensors counts
      b.send_step(tick0, acc);
      b.send_udp_link(tick0);
      b.write_trace();
      ++b.steps;
      while (b.ever_ctrl && !b.fresh_ctrl && !b.stopping()) b.io(50);
      while (speed > 0 && !b.stopping()) {
        const double ahead = static_cast<double>(b.w->tick()) * b.dt / speed - std::chrono::duration<double>(Clock::now() - wall0).count();
        if (ahead <= 0) break;
        b.io(static_cast<int>(std::min(50.0, std::ceil(ahead * 1000.0))));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dvsim-px4-bridge: error: %s\n", e.what());
    rc = 1;
  }
  for (auto& m : b.mons) b.print_monitor(*m.m);  // unfinished
  b.mons.clear();
  if (b.tcp >= 0) close(b.tcp);
  if (b.udp >= 0) close(b.udp);
  if (listen_fd >= 0) close(listen_fd);
  b.close_ctl();
  if (b.ctl_listen >= 0) { close(b.ctl_listen); ::unlink(ctl_path.c_str()); }
  if (b.trace) std::fclose(b.trace);
  if (b.w) {
    try {
      if (!log_path.empty() && b.w->log()) b.w->log()->write(log_path);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dvsim-px4-bridge: cannot write log: %s\n", e.what());
      rc = 1;
    }
    if (rc == 0) {
      std::printf("stats steps=%" PRIu64 " skipped=%" PRIu64 " bad_frames=%" PRIu64 " set_errors=%" PRIu64 "\n", b.steps, b.skipped, b.bad, b.set_errors);
      std::printf("final tick=%" PRIu64 " state_hash=%016" PRIx64 "\n", b.w->tick(), b.w->state_hash());
    }
  }
  std::fflush(stdout);
  return rc;
}
