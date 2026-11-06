// Qdd device (src/mujoco/devices/qdd.cpp; docs/models.md "Qdd"). Motor numbers are those of the CubeMars AK70-9 V3.0 KV60 part
// (kv 60, R 0.475 line-to-line, N 9, rotor 8.66e-5 kg m^2); efficiency, current limit and thermal values are TEST values.
// Expectations are closed forms written independently of the implementation. Models are generated per test into a temp file.
#include <gtest/gtest.h>

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
constexpr double kPi = 3.14159265358979323846;
constexpr double kKv = 60, kR20 = 0.475, kAlpha = 0.00393, kTamb = 25, kN = 9, kEta = 0.9;
double ke() { return 60.0 / (2.0 * kPi * kKv); }
double r_at(double t) { return kR20 * (1 + kAlpha * (t - 20)); }

struct Cfg {
  double volts = 24, i_limit = 20, c_th = 150, r_th = 1.5, latency = 0, timeout = 1e3, kd_timeout = 2, cmd_hz = 1000, i0 = 0.05;
  double derate_start = 60, t_max = 100;
  int mode = 0;
  std::string drop;  // a field to leave out (-> 0 -> the load must fail)
};

std::string xml(const Cfg& c) {
  std::ostringstream o;
  o << R"(<mujoco><option timestep="0.001" gravity="0 0 0"/><worldbody>
  <body name="link" pos="0 0 0.1"><joint name="out" type="hinge" axis="0 0 1"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.05" contype="0" conaffinity="0"/></body></worldbody>
  <custom><text name="dvsim.devices" data='{
  "psus": [{"name": "bench", "node": 1, "voltage": )" << c.volts << R"(, "r_series": 0.01}],
  "qdds": [{"name": "q1", "node": 1, "joint": "out", "parts": ["test-qdd"],
    "motor": {"kv": 60, "r20": 0.475, "i0": )" << c.i0 << R"(, "rotor_inertia": 8.66307e-5, "w_s": 5, "c_th": )" << c.c_th
    << R"(, "r_th": )" << c.r_th << R"(, "t_amb": 25},
    "gear_ratio": 9, "efficiency": 0.9, "encoder_cpr": 16384, "cmd_hz": )" << c.cmd_hz << R"(, "latency_s": )" << c.latency
    << R"(, "i_limit": )" << c.i_limit << R"(, "kp_max": 500, "kd_max": 10, "tau_ff_max": 100,
    "timeout_s": )" << c.timeout << R"(, "timeout_mode": )" << c.mode << R"(, "kd_timeout": )" << c.kd_timeout
    << R"(, "t_max": )" << c.t_max << R"(, "derate_start": )" << c.derate_start << R"(}]
  }'/></custom></mujoco>)";
  std::string s = o.str();
  if (!c.drop.empty()) {  // blank out one field by renaming it to the default-zero twin: replace its value with 0
    const std::size_t k = s.find("\"" + c.drop + "\":");
    if (k != std::string::npos) {
      const std::size_t e = s.find_first_of(",}", k);
      s.replace(k, e - k, "\"" + c.drop + "\": 0");
    }
  }
  return s;
}

std::unique_ptr<Solver> make(const Cfg& c = {}) {
  static int n = 0;
  const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_qdd_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".xml")).string();
  std::ofstream(path) << xml(c);
  try {
    auto s = load_mujoco(path);
    std::remove(path.c_str());
    return s;
  } catch (...) {
    std::remove(path.c_str());
    throw;
  }
}
double get(const Solver& s, const char* attr, const char* dev = "q1") {
  std::vector<double> o;
  const Result r = s.get(std::string("/devices/") + dev, attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << attr << ": " << r.message;
  return o.empty() ? NAN : o[0];
}
void put(Solver& s, const char* attr, double v, const char* dev = "q1") {
  const double a[1] = {v};
  ASSERT_EQ(s.set(std::string("/devices/") + dev, attr, "", a).code, Result::ok) << attr;
}
void set_qvel(Solver& s, double w) {
  const double a[1] = {w};
  ASSERT_EQ(s.set("/world/link/out", "qvel", "", a).code, Result::ok);
}
double qpos(const Solver& s) {
  std::vector<double> o;
  EXPECT_EQ(s.get("/world/link/out", "qpos", "", o).code, Result::ok);
  return o.empty() ? NAN : o[0];
}
// Motor-torque command at the limit: the driver saturates against its current / voltage limits.
void saturate(Solver& s) {
  put(s, "tau_ff", 100);
  put(s, "enable", 1);
}
// Closed-form averaged-FOC torque at output speed w (rad/s), bus voltage v, winding temperature t, for a positive command.
double envelope(double w, double v, double t, double i_limit) {
  const double wm = kN * w, r = r_at(t), kt = ke();
  const double lo = std::max(-i_limit, (-v - kt * wm) / r), hi = std::min(i_limit, (v - kt * wm) / r);
  const double i = lo <= hi ? std::clamp(100.0 / (kt * kN * kEta), lo, hi)  // command 100 N m (tau_ff_max)
                            : std::clamp(wm > 0 ? (v - kt * wm) / r : (-v - kt * wm) / r, -i_limit, i_limit);
  const double g = std::tanh(wm / 5.0) * (i > 0 ? 1.0 : i < 0 ? -1.0 : 0.0);
  const double k = g >= 0 ? kEta : kEta + (1 - kEta) * -g;
  return kt * kN * k * i;
}
}  // namespace

// (a) Stall: the torque is min(i_limit, V / R) Kt N eta, for a current-limited (24 V) and a voltage-limited (6 V) supply.
TEST(qdd, stall_torque_is_the_current_or_voltage_limit) {
  for (const double volts : {24.0, 6.0}) {
    Cfg c;
    c.volts = volts;
    auto s = make(c);
    saturate(*s);
    set_qvel(*s, 0);
    ASSERT_EQ(s->step(1), 1);
    const double v = get(*s, "voltage"), want_i = std::min(c.i_limit, v / r_at(kTamb));
    EXPECT_NEAR(get(*s, "i_q"), want_i, 1e-9 * want_i) << volts << " V";
    const double want = want_i * ke() * kN * kEta;
    EXPECT_NEAR(get(*s, "tau_est"), want, 0.01 * want) << volts << " V";
    EXPECT_GT(get(*s, "bus_i"), 0.0);  // power drawn from the supply
  }
}

// (b) Torque-speed sweep against the analytic envelope; a command above it must saturate to it, never exceed it.
TEST(qdd, torque_speed_envelope_and_no_load_speed) {
  for (const double volts : {24.0, 48.0}) {
    Cfg c;
    c.volts = volts;
    auto s = make(c);
    saturate(*s);
    const double w0 = volts / (ke() * kN);  // no-load output speed
    double prev_w = -1, prev_tau = 0, w_zero = NAN;
    for (int k = 0; k <= 24; ++k) {
      const double w = w0 * k / 20.0;  // 0 .. 1.2 w0
      set_qvel(*s, w);
      const double t_before = get(*s, "temp");
      ASSERT_EQ(s->step(1), 1);
      const double v = get(*s, "voltage"), tau = get(*s, "tau_est");
      const double want = envelope(w, v, t_before, c.i_limit);
      EXPECT_NEAR(tau, want, 1e-6 * std::max(1.0, std::abs(want))) << volts << " V, w " << w;
      EXPECT_LE(tau, get(*s, "tau_cmd") + 1e-9);  // never more than commanded
      if (k > 0 && prev_tau > 0 && tau <= 0) w_zero = prev_w + (w - prev_w) * prev_tau / (prev_tau - tau);
      prev_w = w;
      prev_tau = tau;
    }
    ASSERT_FALSE(std::isnan(w_zero)) << "no zero crossing";
    EXPECT_NEAR(w_zero, w0, 0.02 * w0) << volts << " V";
    if (volts == 48.0) EXPECT_NEAR(w_zero, 33.510, 0.05 * 33.510) << "published no-load speed of the AK70-9 (320 rpm)";
  }
}

// (c) kp / kd hold: a 0.3 rad position step settles within the friction band tau_f / kp = N Kt i0 / kp with no overshoot growth.
TEST(qdd, position_step_settles) {
  auto s = make();
  put(*s, "kp", 20);
  put(*s, "kd", 1);
  put(*s, "p_des", 0.3);
  put(*s, "enable", 1);
  double peak = 0;
  for (int t = 0; t < 1500; ++t) {
    ASSERT_EQ(s->step(1), 1);
    peak = std::max(peak, qpos(*s));
  }
  const double band = kN * ke() * 0.05 / 20.0;  // Coulomb friction band
  EXPECT_NEAR(qpos(*s), 0.3, band + 2e-3);
  EXPECT_LT(peak, 0.3 + band + 0.05);
  EXPECT_LT(std::abs(get(*s, "dq")), 1e-3);
  EXPECT_FALSE(s->diverged());
}

// (d) CAN latency: the command is effective exactly latency_s later (5 ticks), also with a 2 ms command period.
TEST(qdd, latency_delays_the_response_exactly) {
  for (const double hz : {1000.0, 500.0}) {
    Cfg c;
    c.latency = 0.006;
    c.cmd_hz = hz;
    auto s = make(c);
    set_qvel(*s, 0);
    put(*s, "tau_ff", 1);
    put(*s, "enable", 1);
    int first = -1;
    for (int t = 0; t < 12 && first < 0; ++t) {
      ASSERT_EQ(s->step(1), 1);
      if (get(*s, "tau_est") != 0.0) first = t;
    }
    EXPECT_EQ(first, 6) << "cmd_hz " << hz;  // sampled at tick 0, effective 6 ticks later
  }
}

// (e) CAN timeout: with frames arriving, no timeout; without, after timeout_s the driver damps (mode 0) or opens (mode 1).
TEST(qdd, command_loss_engages_the_safe_mode) {
  Cfg t50;
  t50.timeout = 0.05;
  {
    auto s = make(t50);
    put(*s, "enable", 1);  // one frame at t = 0, then silence
    set_qvel(*s, 5.0);
    ASSERT_EQ(s->step(40), 40);
    EXPECT_EQ(get(*s, "timeout"), 0.0);
    EXPECT_EQ(get(*s, "tau_est"), 0.0);  // zero gains: nothing commanded
    ASSERT_EQ(s->step(15), 15);          // 55 ms
    EXPECT_EQ(get(*s, "timeout"), 1.0);
    const double dq = get(*s, "dq");
    ASSERT_GT(dq, 0.1);
    EXPECT_NEAR(get(*s, "tau_cmd"), -2.0 * dq, 1e-12);
    EXPECT_LT(get(*s, "tau_est"), 0.0);
    // a fresh frame ends it
    put(*s, "enable", 1);
    ASSERT_EQ(s->step(1), 1);
    EXPECT_EQ(get(*s, "timeout"), 0.0);
  }
  {
    auto s = make(t50);
    put(*s, "enable", 1);
    for (int t = 0; t < 300; ++t) {  // a frame every 10 ms keeps it alive
      if (t % 10 == 0) put(*s, "kp", 0);
      ASSERT_EQ(s->step(1), 1);
      ASSERT_EQ(get(*s, "timeout"), 0.0) << t;
    }
  }
  {
    Cfg c = t50;
    c.mode = 1;
    auto s = make(c);
    put(*s, "tau_ff", 3);
    put(*s, "enable", 1);
    ASSERT_EQ(s->step(40), 40);
    EXPECT_GT(get(*s, "tau_est"), 0.0);
    ASSERT_EQ(s->step(30), 30);
    EXPECT_EQ(get(*s, "timeout"), 1.0);
    EXPECT_EQ(get(*s, "tau_est"), 0.0);
    EXPECT_EQ(get(*s, "i_q"), 0.0);
    EXPECT_EQ(get(*s, "bus_i"), 0.0);
  }
}

// (f) Thermal: a sustained stall at the current limit heats the winding as C dT/dt = i^2 R - (T - Tamb)/R_th, and past
// derate_start the current limit (and with it the torque) falls.
TEST(qdd, sustained_stall_heats_and_derates) {
  Cfg c;
  c.c_th = 2;
  c.r_th = 5;
  auto s = make(c);
  saturate(*s);
  double t_prev = kTamb, tau0 = NAN;
  for (int t = 0; t < 4000; ++t) {
    set_qvel(*s, 0);
    ASSERT_EQ(s->step(1), 1);
    if (t == 0) {
      tau0 = get(*s, "tau_est");
      // first tick: explicit Euler of the thermal ODE from T_amb
      const double i = get(*s, "i_q"), want = kTamb + 0.001 / c.c_th * i * i * r_at(kTamb);
      EXPECT_NEAR(get(*s, "temp"), want, 1e-9);
    }
    const double T = get(*s, "temp");
    EXPECT_GE(T, t_prev - 1e-9) << t;  // monotone while the dissipation exceeds the loss to ambient
    t_prev = T;
  }
  EXPECT_GT(t_prev, c.derate_start);
  EXPECT_LT(t_prev, c.t_max);
  EXPECT_LT(get(*s, "derate"), 1.0);
  EXPECT_LT(get(*s, "tau_est"), 0.6 * tau0);
  EXPECT_EQ(get(*s, "fault"), 0.0);
}

// (f2) A heat step past t_max (tiny thermal mass: one tick of stall heats by ~190 K) faults the driver: torque off, latched
// while enabled, cleared by enable = 0. (With a realistic thermal mass the linear foldback to 0 at t_max settles below it.)
TEST(qdd, over_temperature_faults_and_latches) {
  Cfg c;
  c.c_th = 0.001;
  c.r_th = 5;
  auto s = make(c);
  saturate(*s);
  bool faulted = false;
  for (int t = 0; t < 300 && !faulted; ++t) {
    set_qvel(*s, 0);
    ASSERT_EQ(s->step(1), 1);
    faulted = get(*s, "fault") > 0.5;
  }
  ASSERT_TRUE(faulted);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "tau_est"), 0.0);
  for (int t = 0; t < 400; ++t) {  // it cools (tau = 50 ms) but stays latched while enabled
    set_qvel(*s, 0);
    ASSERT_EQ(s->step(1), 1);
  }
  EXPECT_LT(get(*s, "temp"), c.t_max);
  EXPECT_EQ(get(*s, "fault"), 1.0);
  put(*s, "enable", 0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "fault"), 0.0);
  put(*s, "enable", 1);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_GT(get(*s, "tau_est"), 0.0);
}

// Gains, feed-forward are clamped to kp_max / kd_max / tau_ff_max; the encoder quantises q.
TEST(qdd, command_limits_and_encoder) {
  auto s = make();
  put(*s, "kp", 5000);      // above kp_max = 500
  put(*s, "kd", 100);       // above kd_max = 10
  put(*s, "p_des", 1.0);
  put(*s, "tau_ff", 1000);  // above tau_ff_max = 100
  put(*s, "enable", 1);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_NEAR(get(*s, "tau_cmd"), 500 * (1.0 - get(*s, "q")) + 10 * (0.0 - get(*s, "dq")) + 100, 1e-9);
  put(*s, "enable", 0);
  set_qvel(*s, 1.0);
  ASSERT_EQ(s->step(100), 100);
  const double counts = get(*s, "q") * 16384 / (2 * kPi);
  EXPECT_GT(counts, 100.0);
  EXPECT_NEAR(counts, std::round(counts), 1e-6);  // on the count grid
}

// (g) Determinism: save / restore / clone mid-run (between command samples, with a latency in flight) equals the uninterrupted run.
TEST(qdd, save_restore_bitwise) {
  Cfg c;
  c.latency = 0.003;
  c.cmd_hz = 500;
  auto run = [](Solver& s, int t0, int t1) {
    for (int t = t0; t < t1; ++t) {
      if (t % 7 == 0) {
        put(s, "kp", 10 + (t % 5));
        put(s, "p_des", std::sin(t * 0.01));
        put(s, "kd", 0.5);
        put(s, "enable", 1);
      }
      ASSERT_EQ(s.step(1), 1);
    }
  };
  auto a = make(c), b = make(c);
  run(*a, 0, 1500);
  run(*b, 0, 701);
  const std::vector<std::byte> snap = b->save();
  const std::uint64_t h_mid = b->state_hash();
  run(*b, 701, 1000);
  ASSERT_NE(b->state_hash(), h_mid);
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), h_mid);
  auto cl = b->clone();
  run(*b, 701, 1500);
  run(*cl, 701, 1500);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->state_hash(), cl->state_hash());
  EXPECT_EQ(a->save(), b->save());
  EXPECT_FALSE(a->diverged());
  EXPECT_GT(std::abs(qpos(*a)), 1e-3);  // it really moved
}

// No physics defaults: a missing parameter fails the load and names the device.
TEST(qdd, missing_parameter_fails_load) {
  for (const char* field : {"gear_ratio", "efficiency", "encoder_cpr", "cmd_hz", "i_limit", "kp_max", "kd_max", "tau_ff_max", "timeout_s", "t_max"}) {
    Cfg c;
    c.drop = field;
    try {
      make(c);
      ADD_FAILURE() << field << ": load succeeded";
    } catch (const std::exception& e) {
      EXPECT_NE(std::string(e.what()).find("q1"), std::string::npos) << field << ": " << e.what();
    }
  }
  Cfg bad;
  bad.latency = 0.0004;  // not a whole number of 1 ms ticks
  EXPECT_THROW(make(bad), std::exception);
}
