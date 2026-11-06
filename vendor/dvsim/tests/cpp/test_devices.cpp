// Device framework + electrical network + battery / PSU / wire / load + ESC / BLDC motor (milestone N2, stage S1).
// Every analytic expectation is written independently of the implementation (closed forms, not a copy of the solver).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/world.h"
#include "mujoco/mujoco_solver.h"

namespace {
const std::string kDir = std::string(DVSIM_TEST_DATA_DIR) + "/devices/";
constexpr double kPi = 3.14159265358979323846;

std::unique_ptr<Solver> load(const std::string& f) { return load_mujoco(kDir + f); }
const MujocoSolver& ms(const Solver& s) { return static_cast<const MujocoSolver&>(s); }

double get(const Solver& s, const std::string& prim, const std::string& attr) {
  std::vector<double> o;
  const Result r = s.get(prim, attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << prim << "." << attr << ": " << r.message;
  return o.empty() ? NAN : o[0];
}
Result put(Solver& s, const std::string& prim, const std::string& attr, double v) { return s.set(prim, attr, "", std::span<const double>(&v, 1)); }
void put_ok(Solver& s, const std::string& prim, const std::string& attr, double v) {
  const Result r = put(s, prim, attr, v);
  EXPECT_EQ(r.code, Result::ok) << r.message;
}

double rel(double got, double want) { return std::abs(got - want) / std::abs(want); }

// Motor of the fixtures: 920 KV, R20 0.117, I0 0.5, ambient 25 degC, alpha 0.00393.
constexpr double kKv = 920, kR20 = 0.117, kI0 = 0.5, kAlpha = 0.00393, kTamb = 25;
double ke() { return 60.0 / (2.0 * kPi * kKv); }
double r_at(double t) { return kR20 * (1 + kAlpha * (t - 20)); }

// Throttle schedule for the determinism tests: a function of the tick only.
void run(Solver& s, int t0, int t1, const std::string& esc = "/devices/m1") {
  for (int t = t0; t < t1; ++t) {
    put_ok(s, esc, "throttle", t < 50 ? 1.0 : 0.5 + 0.5 * std::sin(t * 0.004));
    ASSERT_EQ(s.step(1), 1);
  }
}
}  // namespace

// (a) no-load speed: steady w = Kv (V_node - I0 R(T)), with the actual V_node and T read from the device.
TEST(devices, motor_no_load_speed) {
  auto s = load("motor.xml");
  put_ok(*s, "/devices/m1", "throttle", 1.0);
  ASSERT_EQ(s->step(3000), 3000);  // tau_m = J R / (Kt Ke) ~ 33 ms: 3 s is ~90 time constants
  const double v = get(*s, "/devices/m1", "voltage"), t = get(*s, "/devices/m1", "temperature");
  const double want = (kKv * 2 * kPi / 60) * (v - kI0 * r_at(t));
  const double w = get(*s, "/devices/m1", "omega");
  EXPECT_GT(w, 1000);  // really spinning (about 1.5e3 rad/s)
  EXPECT_LT(rel(w, want), 0.005) << "omega " << w << " analytic " << want;
  EXPECT_GT(t, kTamb);  // the winding heated up (I0^2 R > 0)
  EXPECT_FALSE(s->diverged());
}

// (a) stall: at w = 0 the phase current is exactly d V_node / R(T_ambient).
TEST(devices, motor_stall_current) {
  auto s = load("motor.xml");
  const double d = 0.6;
  put_ok(*s, "/devices/m1", "throttle", d);
  ASSERT_EQ(s->step(1), 1);
  const double v = get(*s, "/devices/m1", "voltage");
  EXPECT_EQ(get(*s, "/devices/m1", "omega"), 0.0);
  EXPECT_LT(rel(get(*s, "/devices/m1", "current"), d * v / r_at(kTamb)), 1e-12);
  EXPECT_LT(rel(get(*s, "/devices/m1", "bus_current"), d * d * v / r_at(kTamb)), 1e-12);  // I_bus = d I_m
  EXPECT_GT(get(*s, "/devices/m1", "torque"), 0);
}

// (b) Thevenin battery into a constant-current sink. With the exact exponential RC update and a constant current the
// recursion v1 <- v1 a + I R1 (1 - a) equals the analytic curve, so only rounding remains (error ~ n eps, plus the 1e-12 S
// gmin leak = 1e-12 relative of the current): 1e-9 relative is generous. The terminal voltage reported after n steps was
// solved at the start of step n, i.e. with v_rc of n-1 steps.
TEST(devices, battery_step_response) {
  auto s = load("battery_step.xml");
  const double ocv = 12, i = 10, r0 = 0.05, r1 = 0.03, tau = 0.03 * 100, dt = 0.001, cap = 2;
  int n = 0;
  for (int target : {1, 10, 1000, 3000, 6000}) {
    ASSERT_EQ(s->step(target - n), target - n);
    n = target;
    const double v_want = ocv - i * r0 - i * r1 * (1 - std::exp(-(n - 1) * dt / tau));
    EXPECT_LT(rel(get(*s, "/devices/pack", "voltage"), v_want), 1e-9) << "n=" << n;
    EXPECT_LT(rel(get(*s, "/devices/pack", "v_rc"), i * r1 * (1 - std::exp(-n * dt / tau))), 1e-9) << "n=" << n;
    EXPECT_LT(rel(get(*s, "/devices/pack", "current"), i), 1e-9);
    EXPECT_LT(rel(1.0 - get(*s, "/devices/pack", "soc"), i * n * dt / (3600 * cap)), 1e-9) << "n=" << n;
    EXPECT_EQ(get(*s, "/devices/pack", "ocv"), ocv);
    EXPECT_LT(rel(get(*s, "/devices/sink", "current"), i), 1e-9);
  }
}

// (d) star: PSU -> trunk (AWG 12, 0.5 m, 2 conductors) -> PDB -> 4 branch wires -> 4 static loads.
// Branch k: I_k = (g_k V2 - i0_k) / (1 + g_k Rw_k); trunk node: V2 = (Vs + Rt sum i0_k/(1+g_k Rw_k)) / (1 + Rt sum g_k/(1+g_k Rw_k)).
TEST(devices, network_star) {
  auto s = load("star.xml");
  ASSERT_EQ(s->step(1), 1);
  auto awg_r = [](int awg, double len, int n) {
    const double d = 0.127e-3 * std::pow(92.0, (36.0 - awg) / 39.0);
    return 1.724e-8 * len * n / (kPi * d * d / 4);
  };
  const double vs = 24, rs = 0.05, r_trunk = awg_r(12, 0.5, 2);
  const double rw[4] = {0.02, 0.03, awg_r(16, 0.4, 2), 0.05}, g[4] = {0.5, 0.25, 1.0, 0.1}, i0[4] = {0, 0, 0, -3};
  const double rt = rs + r_trunk;
  double sa = 0, sb = 0;
  for (int k = 0; k < 4; ++k) {
    sa += i0[k] / (1 + g[k] * rw[k]);
    sb += g[k] / (1 + g[k] * rw[k]);
  }
  const double v2 = (vs + rt * sa) / (1 + rt * sb);
  double itot = 0;
  for (int k = 0; k < 4; ++k) {
    const double ik = (g[k] * v2 - i0[k]) / (1 + g[k] * rw[k]);
    itot += ik;
    const std::string l = "/devices/l" + std::to_string(k + 1), b = "/devices/br" + std::to_string(k + 1);
    EXPECT_LT(rel(get(*s, l, "voltage"), v2 - rw[k] * ik), 1e-9) << "load " << k + 1;
    EXPECT_LT(rel(get(*s, l, "current"), ik), 1e-9) << "load " << k + 1;
    EXPECT_LT(rel(get(*s, b, "current"), ik), 1e-9) << "branch " << k + 1;
  }
  EXPECT_LT(rel(get(*s, "/devices/trunk", "voltage"), r_trunk * itot), 1e-9);  // drop across the trunk
  EXPECT_LT(rel(get(*s, "/devices/src", "current"), itot), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/src", "voltage"), vs - rs * itot), 1e-9);
}

// (d) chain: six loads daisy-chained. Ladder from the far end: a load node k sees I_in = alpha_k V_k - beta_k with
// alpha_k = g_k + alpha_{k+1}/(1 + alpha_{k+1} R_{k+1}), beta_k = i0_k + beta_{k+1}/(1 + alpha_{k+1} R_{k+1});
// then forward from the source: V_{k+1} = V_k - R_{k+1} (alpha_{k+1} V_k - beta_{k+1}) / (1 + alpha_{k+1} R_{k+1}).
TEST(devices, network_chain) {
  auto s = load("chain.xml");
  ASSERT_EQ(s->step(1), 1);
  const double vs = 12, rs = 0.1;
  const double rw[6] = {0.1, 0.2, 0.15, 0.1, 0.3, 0.25};  // rw[k]: wire into node k+2
  const double g[6] = {0.2, 0.1, 0.4, 0, 0.3, 0.6}, i0[6] = {0, -0.5, 0, -1.5, -0.2, 0};  // loads at nodes 2..7
  double alpha[6], beta[6];
  alpha[5] = g[5];
  beta[5] = i0[5];
  for (int k = 4; k >= 0; --k) {
    const double den = 1 + alpha[k + 1] * rw[k + 1];
    alpha[k] = g[k] + alpha[k + 1] / den;
    beta[k] = i0[k] + beta[k + 1] / den;
  }
  const double den0 = 1 + alpha[0] * rw[0];
  const double a1 = alpha[0] / den0, b1 = beta[0] / den0;  // seen by the PSU node
  double v = (vs + rs * b1) / (1 + rs * a1);
  EXPECT_LT(rel(get(*s, "/devices/src", "voltage"), v), 1e-9);
  const char names[6] = {'a', 'b', 'c', 'd', 'e', 'f'};
  for (int k = 0; k < 6; ++k) {
    const double den = 1 + alpha[k] * rw[k];
    v -= rw[k] * (alpha[k] * v - beta[k]) / den;
    EXPECT_LT(rel(get(*s, std::string("/devices/") + names[k], "voltage"), v), 1e-9) << "load " << names[k];
  }
}

// Spin-up emerges from the rotor inertia: first order, tau = J (R + R_series) / (Kt Ke) = 33 ms (and the inertia really is
// the SET armature: a simple dof reads dof_M0, which bind() refreshes with mj_setConst). Euler with the explicit drive term is first order in
// dt/tau = 3%: 5% band on the fraction 1 - 1/e reached after one time constant.
TEST(devices, motor_spin_up_time_constant) {
  auto s = load("motor.xml");
  put_ok(*s, "/devices/m1", "throttle", 1.0);
  const double tau = 3e-5 * (r_at(kTamb) + 0.001) / (ke() * ke());
  const int n1 = static_cast<int>(std::lround(tau / 0.001));
  ASSERT_EQ(s->step(n1), n1);
  const double w1 = get(*s, "/devices/m1", "omega");
  ASSERT_EQ(s->step(3000), 3000);
  const double winf = get(*s, "/devices/m1", "omega");
  const double want = 1 - std::exp(-n1 * 0.001 / tau);
  EXPECT_NEAR(w1 / winf, want, 0.05 * want) << "tau " << tau << " n1 " << n1;
}

// The damping split is exact: (-b qvel + qfrc_applied) on the dof == Kt I_m - tau_f, from the same V_node.
TEST(devices, damping_split_exact) {
  auto s = load("motor.xml");
  put_ok(*s, "/devices/m1", "throttle", 0.5);
  ASSERT_EQ(s->step(4), 4);  // a few ms from rest: speed nonzero, current far from its steady value
  const double t_pre = get(*s, "/devices/m1", "temperature");  // R(T) of the last tick is the one of its start
  ASSERT_EQ(s->step(1), 1);
  const mjModel& m = ms(*s).model();
  const mjData& d = ms(*s).data();
  const int dof = m.jnt_dofadr[mj_name2id(&m, mjOBJ_JOINT, "shaft")];
  const double w = get(*s, "/devices/m1", "omega"), v = get(*s, "/devices/m1", "voltage");
  ASSERT_GT(w, 1.0);
  // Everything below is recomputed here from V_node, omega, T: nothing is read back from the device's current / torque.
  const double kt = ke(), duty = 0.5, im = (duty * v - kt * w) / r_at(t_pre);
  ASSERT_GT(im, 5 * kI0) << "w " << w << " v " << v << " t_pre " << t_pre << " I_m(dev) " << get(*s, "/devices/m1", "current")
                         << " b " << m.dof_damping[dof] << " armature " << m.dof_armature[dof];
  const double tau_f = kt * kI0 * std::tanh(w / 5.0);
  const double want = kt * im - tau_f;
  const double lhs = -m.dof_damping[dof] * w + d.qfrc_applied[dof];
  EXPECT_LT(rel(lhs, want), 1e-12) << "lhs " << lhs << " want " << want;
  EXPECT_LT(rel(get(*s, "/devices/m1", "torque"), want), 1e-12);
  EXPECT_LT(rel(m.dof_damping[dof], kt * kt / r_at(t_pre)), 1e-12);  // b = Kt Ke / R(T)
  EXPECT_DOUBLE_EQ(m.dof_armature[dof], 3e-5);  // SET to the rotor inertia, not added to the MJCF value
}

// Save mid-run / restore / continue == uninterrupted run, bitwise (ESC current limit, battery RC, throttle changes via set()).
TEST(devices, save_restore_bitwise) {
  auto a = load("motor_batt.xml"), b = load("motor_batt.xml");
  run(*a, 0, 3000);
  run(*b, 0, 1500);
  const std::vector<std::byte> snap = b->save();
  const std::uint64_t h_mid = b->state_hash();
  run(*b, 1500, 2200);
  ASSERT_NE(b->state_hash(), h_mid);
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), h_mid);
  run(*b, 1500, 3000);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->save(), b->save());
  EXPECT_FALSE(a->diverged());
  EXPECT_LT(get(*a, "/devices/pack", "soc"), 0.8);  // the battery really discharged
  EXPECT_GT(get(*a, "/devices/m1", "omega"), 100);
}

// The ESC current limit engages at stall and the network stays consistent: battery current = ESC bus current + avionics.
TEST(devices, esc_current_limit) {
  auto s = load("motor_batt.xml");
  put_ok(*s, "/devices/m1", "throttle", 1.0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(ms(*s).devices()->esc.st[0].mode, 2.0);
  EXPECT_DOUBLE_EQ(get(*s, "/devices/m1", "current"), 40.0);
  const double ibus = get(*s, "/devices/m1", "bus_current"), ibatt = get(*s, "/devices/pack", "current");
  EXPECT_LT(rel(ibatt, ibus + 0.5), 1e-9) << "battery " << ibatt << " bus " << ibus;
  EXPECT_LT(get(*s, "/devices/m1", "duty"), 1.0);
  // duty reduced so that d V = I_lim R(T) (w = 0)
  EXPECT_LT(rel(get(*s, "/devices/m1", "duty"), 40.0 * r_at(kTamb) / get(*s, "/devices/m1", "voltage")), 1e-12);
}

// PSU current limit: constant-current mode when a stalled motor would draw more than the limit.
TEST(devices, psu_current_limit) {
  auto s = load("motor_lim.xml");
  put_ok(*s, "/devices/m1", "throttle", 1.0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "/devices/bench", "limited"), 1.0);
  EXPECT_DOUBLE_EQ(get(*s, "/devices/bench", "current"), 20.0);
  EXPECT_LT(rel(get(*s, "/devices/m1", "voltage"), 20.0 * r_at(kTamb)), 1e-9);  // d = 1: V = I R
  // Disabled supply = open circuit.
  put_ok(*s, "/devices/bench", "enable", 0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(get(*s, "/devices/bench", "current"), 0.0);
  EXPECT_LT(std::abs(get(*s, "/devices/m1", "voltage")), 1e-9);
}

// Non-complementary ESC: a phase that would back-feed is open (diode freewheel); complementary: it brakes / regenerates.
TEST(devices, freewheel_and_regen) {
  for (bool comp : {false, true}) {
    auto s = load(comp ? "motor_comp.xml" : "motor.xml");
    put_ok(*s, "/devices/m1", "throttle", 1.0);
    ASSERT_EQ(s->step(1500), 1500);
    put_ok(*s, "/devices/m1", "throttle", 0.1);  // d V << Ke w
    ASSERT_EQ(s->step(1), 1);
    if (comp) {
      EXPECT_LT(get(*s, "/devices/m1", "current"), -1.0);
      EXPECT_LT(get(*s, "/devices/m1", "bus_current"), 0.0);
      EXPECT_LT(get(*s, "/devices/bench", "current"), 0.0);  // flows back into the (sinking) supply
      EXPECT_LT(get(*s, "/devices/m1", "torque"), 0.0);
    } else {
      EXPECT_EQ(get(*s, "/devices/m1", "current"), 0.0);
      EXPECT_EQ(get(*s, "/devices/m1", "bus_current"), 0.0);
      EXPECT_LT(std::abs(get(*s, "/devices/bench", "current")), 1e-9);
      EXPECT_EQ(ms(*s).devices()->esc.st[0].b, 0.0);  // open phase: no back-EMF damping
      EXPECT_LT(get(*s, "/devices/m1", "torque"), 0.0);  // friction only
    }
  }
}

TEST(devices, clone_equal_hash) {
  auto a = load("motor_batt.xml");
  run(*a, 0, 800);
  auto c = a->clone();
  EXPECT_EQ(a->state_hash(), c->state_hash());
  run(*a, 800, 1500);
  run(*c, 800, 1500);
  EXPECT_EQ(a->state_hash(), c->state_hash());
  EXPECT_NE(c->state_hash(), load("motor_batt.xml")->state_hash());
}

// Reloading from mj_saveModel bytes (custom text carried; armature SET, so idempotent) gives the same prims, attrs and hash.
TEST(devices, model_bytes_round_trip) {
  auto fresh = load("motor_batt.xml");
  auto used = load("motor_batt.xml");
  run(*used, 0, 200);  // its model now holds the device-owned dof_damping of a later tick
  for (const std::vector<std::byte>& bytes : {fresh->model_bytes(), used->model_bytes()}) {
    auto r = load_mujoco_bytes(bytes);
    ASSERT_NE(ms(*r).devices(), nullptr);
    const auto pa = fresh->prims(), pb = r->prims();
    ASSERT_EQ(pa.size(), pb.size());
    for (std::size_t i = 0; i < pa.size(); ++i) EXPECT_EQ(pa[i].path, pb[i].path);
    EXPECT_EQ(r->unit("/devices/m1", "temperature"), "degC");
    auto ref = load("motor_batt.xml");
    EXPECT_EQ(r->state_hash(), ref->state_hash());
    run(*r, 0, 1200);
    run(*ref, 0, 1200);
    EXPECT_EQ(r->state_hash(), ref->state_hash());
  }
}

TEST(devices, prims_attrs_and_input_validation) {
  auto s = load("motor_batt.xml");
  std::vector<std::string> dev;
  for (const Prim& p : s->prims())
    if (p.kind == "device") dev.push_back(p.path);
  EXPECT_EQ(dev, (std::vector<std::string>{"/devices/pack", "/devices/harness", "/devices/avionics", "/devices/m1"}));
  EXPECT_EQ(s->unit("/devices/pack", "voltage"), "V");
  EXPECT_EQ(s->unit("/devices/m1", "omega"), "rad/s");
  EXPECT_EQ(s->unit("/devices/m1", "torque"), "N*m");
  EXPECT_EQ(s->unit("/devices/m1", "nonsense"), "");
  EXPECT_EQ(put(*s, "/devices/m1", "throttle", 1.5).code, Result::invalid_argument);
  EXPECT_EQ(put(*s, "/devices/m1", "omega", 1).code, Result::invalid_argument);  // read-only
  EXPECT_EQ(put(*s, "/devices/m1", "bogus", 1).code, Result::not_found);
  std::vector<double> o;
  EXPECT_EQ(s->get("/devices/m1", "omega", "world", o).code, Result::ok);
  EXPECT_EQ(s->get("/devices/m1", "omega", "/world/rotor", o).code, Result::invalid_argument);
  EXPECT_EQ(s->get("/devices/nope", "omega", "", o).code, Result::not_found);
}

// Devices go through the unchanged bus: SetAttr throttle is logged, and replaying the log (model rebuilt from the header
// bytes) reproduces the final state hash. Snapshot / Restore commands are in the mix.
TEST(devices, bus_log_and_replay) {
  Session sess;
  sess.caps = {"read", "control"};
  sess.root = DVSIM_TEST_DATA_DIR;
  auto cmd = [](const std::string& type, const std::string& body) {
    return R"({"id":1,"source":"t","cmd_type":")" + type + R"(","cmd":)" + body + "}";
  };
  auto set_thr = [&](double v) {
    return cmd("SetAttr", R"({"path":"/devices/m1","attr":"throttle","values":[)" + std::to_string(v) + "]}");
  };
  World w(load_mujoco, kDir + "motor_batt.xml");
  w.start_log();
  for (const std::string& c : {set_thr(1.0), cmd("Step", R"({"n":300})"), cmd("Snapshot", "{}"), set_thr(0.4),
                               cmd("Step", R"({"n":400})"), cmd("Restore", R"({"id":1})"), set_thr(0.7),
                               cmd("Step", R"({"n":500})")}) {
    const std::string rep = execute_json(w, c, sess);
    ASSERT_NE(rep.find("\"ok\": true"), std::string::npos) << c << " -> " << rep;
  }
  const std::string q = execute_json(w, cmd("Query", R"({"path":"/devices/pack","attr":"soc"})"), sess);
  EXPECT_NE(q.find("\"unit\": \"1\""), std::string::npos) << q;
  World r = replay(w.log()->bytes(), load_mujoco_bytes, load_mujoco, DVSIM_TEST_DATA_DIR);
  EXPECT_EQ(r.tick(), w.tick());
  EXPECT_EQ(r.state_hash(), w.state_hash());
}

// Spawn / Delete keep device state and re-resolve names; deleting a device's joint fails cleanly (nothing changes).
TEST(devices, spawn_delete_keep_devices) {
  auto s = load("motor_batt.xml");
  run(*s, 0, 400);
  const std::vector<std::byte> before = s->save();
  const std::uint64_t h = s->state_hash();
  const Result bad = s->remove("/world/rotor");
  EXPECT_EQ(bad.code, Result::invalid_argument);
  EXPECT_NE(bad.message.find("shaft"), std::string::npos) << bad.message;
  EXPECT_EQ(s->state_hash(), h);
  EXPECT_EQ(s->save(), before);
  const double pos[3] = {1, 0, 0};
  const Result sp = s->spawn(std::string(DVSIM_TEST_DATA_DIR) + "/pendulum.xml", "/world/pend", pos, {});
  ASSERT_EQ(sp.code, Result::ok) << sp.message;
  const double soc = get(*s, "/devices/pack", "soc"), w = get(*s, "/devices/m1", "omega");
  EXPECT_LT(soc, 0.8);
  run(*s, 400, 800);  // still steps, on the new layout
  EXPECT_LT(get(*s, "/devices/pack", "soc"), soc);
  EXPECT_NE(get(*s, "/devices/m1", "omega"), w);
  EXPECT_FALSE(s->diverged());
  EXPECT_EQ(s->remove("/world/pend").code, Result::ok);
}

TEST(devices, bad_config_fails_load_with_message) {
  struct Case { const char* file; const char* needle; };
  for (const Case& c : {Case{"bad_field.xml", "bogus_field"}, Case{"bad_joint.xml", "no_such_joint"},
                        Case{"bad_r0.xml", "pack_bad"}, Case{"bad_rk4.xml", "RK4"},
                        Case{"bad_rotor.xml", "prop_bad"}, Case{"bad_env2.xml", "air2"}}) {
    try {
      load(c.file);
      ADD_FAILURE() << c.file << " loaded";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(c.needle), std::string::npos) << c.file << ": " << e.what();
      if (std::string(c.file) == "bad_r0.xml") EXPECT_NE(std::string(e.what()).find("r0"), std::string::npos) << e.what();
    }
  }
}

// ======== Stage S2: rotor aero, environment, body drag ========
namespace {
// ISA troposphere written from the definition (T = T0 - L h, p = p0 (T/T0)^(g0 / R L), rho = p / R T).
double isa(double h) {
  const double t = 288.15 - 0.0065 * h;
  return 101325.0 * std::pow(t / 288.15, 9.80665 / (287.05287 * 0.0065)) / (287.05287 * t);
}
// Propeller of the fixtures: D = 0.254 m, J = {0, 0.4, 0.8}, C_T = {0.12, 0.08, 0.02}, C_Q = {0.006, 0.005, 0.002}.
constexpr double kD = 0.254;
double ct_of(double j) { return j <= 0 ? 0.12 : j < 0.4 ? 0.12 + j / 0.4 * (0.08 - 0.12) : 0.08 + (j - 0.4) / 0.4 * (0.02 - 0.08); }
double cq_of(double j) { return j <= 0 ? 0.006 : j < 0.4 ? 0.006 + j / 0.4 * (0.005 - 0.006) : 0.005 + (j - 0.4) / 0.4 * (0.002 - 0.005); }
int body_id(const Solver& s, const char* name) { return mj_name2id(&ms(s).model(), mjOBJ_BODY, name); }
const double* xfrc(const Solver& s, const char* body) { return ms(s).data().xfrc_applied + 6 * body_id(s, body); }
void put_vec(Solver& s, const std::string& prim, const std::string& attr, std::vector<double> v) {
  const Result r = s.set(prim, attr, "", v);
  EXPECT_EQ(r.code, Result::ok) << r.message;
}
}  // namespace

// (c1) Fixed w, rotor alone: T = C_T(0) rho n^2 D^4, Q = C_Q(0) rho n^2 D^5, read from the attrs AND from the xfrc_applied
// entries. Everything is closed-form arithmetic on identical doubles: 1e-9 relative leaves room only for rounding.
// Then with axial speed (non-table J: interpolated C_T, C_Q) and reverse spin (thrust and torque both flip).
TEST(devices, rotor_thrust_and_torque_fixed_speed) {
  const double w = 500, n = w / (2 * kPi), rho = isa(1.0), d4 = std::pow(kD, 4);
  {
    auto s = load("rotor_free.xml");
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", w);
    ASSERT_EQ(s->step(1), 1);
    const double t = 0.12 * rho * n * n * d4, q = 0.006 * rho * n * n * d4 * kD;
    EXPECT_LT(rel(get(*s, "/devices/prop", "thrust"), t), 1e-9);
    EXPECT_LT(rel(get(*s, "/devices/prop", "torque"), q), 1e-9);
    EXPECT_LT(rel(get(*s, "/devices/prop", "rho"), rho), 1e-9);
    EXPECT_EQ(get(*s, "/devices/prop", "omega"), w);
    EXPECT_EQ(get(*s, "/devices/prop", "advance_ratio"), 0.0);
    EXPECT_EQ(get(*s, "/devices/prop", "ge_ratio"), 1.0);
    const double* f = xfrc(*s, "rotor");
    EXPECT_LT(rel(f[2], t), 1e-9);   // thrust along +axis
    EXPECT_LT(rel(f[5], -q), 1e-9);  // aero torque opposes the positive spin
    for (int i : {0, 1, 3, 4}) EXPECT_EQ(f[i], 0.0) << i;
    EXPECT_EQ(s->unit("/devices/prop", "thrust"), "N");
    EXPECT_EQ(s->unit("/devices/prop", "rho"), "kg/m^3");
  }
  {  // climbing at 5 m/s along the thrust axis
    auto s = load("rotor_free.xml");
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", w);
    put_ok(*s, "/world/carrier/lift", "qvel", 5.0);
    ASSERT_EQ(s->step(1), 1);
    const double j = 5.0 / (n * kD);
    ASSERT_GT(j, 0.2);
    ASSERT_LT(j, 0.4);  // strictly between table points
    EXPECT_LT(rel(get(*s, "/devices/prop", "advance_ratio"), j), 1e-9);
    EXPECT_LT(rel(get(*s, "/devices/prop", "thrust"), ct_of(j) * rho * n * n * d4), 1e-9);
    EXPECT_LT(rel(get(*s, "/devices/prop", "torque"), cq_of(j) * rho * n * n * d4 * kD), 1e-9);
    EXPECT_LT(rel(xfrc(*s, "rotor")[2], ct_of(j) * rho * n * n * d4), 1e-9);
  }
  {  // reverse spin: thrust along -axis, torque (about +axis) now positive; J measured in the thrust direction
    auto s = load("rotor_free.xml");
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", -w);
    ASSERT_EQ(s->step(1), 1);
    EXPECT_LT(rel(xfrc(*s, "rotor")[2], -0.12 * rho * n * n * d4), 1e-9);
    EXPECT_LT(rel(xfrc(*s, "rotor")[5], 0.006 * rho * n * n * d4 * kD), 1e-9);
    EXPECT_EQ(get(*s, "/devices/prop", "omega"), -w);
  }
}

// Steady wind (3, -4, 2) m/s on a held rotor: V_rel = -wind. H-force F = -k_h |w| v_perp = -0.02 * 300 * (-3, 4) = (18, -24);
// axial component -2 m/s gives J < 0, clamped to the static C_T. Arithmetic only: 1e-9.
TEST(devices, rotor_in_wind_h_force) {
  auto s = load("rotor_wind.xml");
  const double w = 300, n = w / (2 * kPi);
  put_ok(*s, "/world/rotor/shaft", "qvel", w);
  ASSERT_EQ(s->step(1), 1);
  const double* f = xfrc(*s, "rotor");
  EXPECT_LT(rel(f[0], 18.0), 1e-9);
  EXPECT_LT(rel(f[1], -24.0), 1e-9);
  EXPECT_LT(rel(f[2], 0.12 * isa(2.0) * n * n * std::pow(kD, 4)), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/prop", "advance_ratio"), -2.0 / (n * kD)), 1e-9);
}

// (c2) Reaction on the stator. Stator welded to the world with a torque sensor (= wrench the world applies to everything
// below the site, in the site frame, which is aligned with the world). At steady speed there is no angular acceleration, so
// the world must carry the whole aero torque: sensor_z = +Q (the aero torque on the rotor is -Q, the motor torque is
// internal), and Q = Kt I - tau_f = the shaft torque. Tolerance 1e-4 relative: the residual J dw/dt after 4 s (> 100
// mechanical time constants of ~27 ms; winding creep of 0.003 K/s with c_th = 2000 J/K gives ~2e-6) is far below it.
TEST(devices, rotor_torque_reaction_on_stator) {
  auto s = load("rotor_motor.xml");
  put_ok(*s, "/devices/m1", "throttle", 0.4);
  ASSERT_EQ(s->step(4000), 4000);
  std::vector<double> sensor;
  ASSERT_EQ(s->get("/sensors/reaction", "value", "", sensor).code, Result::ok);
  ASSERT_EQ(sensor.size(), 3u);
  const double q = get(*s, "/devices/prop", "torque"), tau = get(*s, "/devices/m1", "torque");
  ASSERT_GT(get(*s, "/devices/m1", "omega"), 300);
  ASSERT_GT(q, 0.02);
  EXPECT_GT(sensor[2], 0.0);  // sign: the world pushes with +Q on the stator
  EXPECT_LT(rel(sensor[2], q), 1e-4) << sensor[2] << " vs Q " << q;
  EXPECT_LT(rel(tau, q), 1e-4) << "shaft " << tau << " vs Q " << q;
  EXPECT_LT(rel(sensor[2], tau), 1e-4);
  EXPECT_LT(std::abs(sensor[0]) + std::abs(sensor[1]), 1e-6 * q);  // nothing off-axis
}

// (c3) Free-running motor + rotor reaches w* with Kt I_m(w*) - tau_f(w*) = Q(w*). The scalar equation is solved here by
// bisection from the formulas (V_node and winding temperature read from the device); the simulated speed is that of the
// last tick. 0.5%: the same band as the no-load test (residual lag + the 1 ms explicit step of the friction term).
TEST(devices, rotor_motor_equilibrium_speed) {
  auto s = load("rotor_motor.xml");
  const double d = 0.4;
  put_ok(*s, "/devices/m1", "throttle", d);
  ASSERT_EQ(s->step(4000), 4000);
  const double v = get(*s, "/devices/m1", "voltage"), t = get(*s, "/devices/m1", "temperature");
  const double rho = isa(0.5), kt = ke();
  auto f = [&](double w) {
    const double n = w / (2 * kPi);
    return kt * (d * v - kt * w) / r_at(t) - kt * kI0 * std::tanh(w / 5.0) - 0.006 * rho * n * n * std::pow(kD, 5);
  };
  double lo = 0, hi = 2000;
  ASSERT_GT(f(lo), 0);
  ASSERT_LT(f(hi), 0);
  for (int i = 0; i < 200; ++i) (f(0.5 * (lo + hi)) > 0 ? lo : hi) = 0.5 * (lo + hi);
  const double w_star = 0.5 * (lo + hi), w = get(*s, "/devices/m1", "omega");
  EXPECT_GT(w_star, 300);
  EXPECT_LT(rel(w, w_star), 0.005) << "sim " << w << " analytic " << w_star;
}

// (c4) Ground effect: ratio 1 / (1 - (R / 4z)^2), z = ray distance along the downwash (-s * axis) from the CoM. Rotor "hi" is 0.3 m over a plane
// with a group-4 plate 0.14 m below it: the default mask hits the plate, ray_groups = 47 skips group 4 and hits the plane,
// a disabled ray gives 1. Rotor "low" (0.05 m) is clamped at z = R / 2 -> exactly 4/3. 1e-9: closed form on the exact
// ray distance (MuJoCo's plane / box hit is exact to rounding).
TEST(devices, rotor_ground_effect) {
  auto s = load("rotor_ge.xml");
  const double w = 300, n = w / (2 * kPi), rad = kD / 2;
  put_ok(*s, "/world/hi/shaft_hi", "qvel", w);
  put_ok(*s, "/world/low/shaft_low", "qvel", w);
  put_ok(*s, "/world/neg/shaft_neg", "qvel", w);
  ASSERT_EQ(s->step(1), 1);
  auto ge = [&](double z) { const double q = rad / (4 * z); return 1.0 / (1.0 - q * q); };
  // Thrust along -axis (axis -z, dir -1, w > 0 -> thrust up): the ray follows the downwash (+axis = down) and hits the plane.
  EXPECT_LT(rel(get(*s, "/devices/ge_neg", "ge_ratio"), ge(0.2)), 1e-9);
  EXPECT_EQ(get(*s, "/devices/ge_neg_down", "ge_ratio"), 1.0);  // thrust down, downwash up: nothing above
  EXPECT_LT(rel(get(*s, "/devices/ge_all", "ge_ratio"), ge(0.14)), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/ge_no4", "ge_ratio"), ge(0.30)), 1e-9);
  EXPECT_EQ(get(*s, "/devices/ge_off", "ge_ratio"), 1.0);
  EXPECT_LT(rel(get(*s, "/devices/ge_low", "ge_ratio"), 4.0 / 3.0), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/ge_no4", "thrust"), ge(0.30) * 0.12 * isa(0.3) * n * n * std::pow(kD, 4)), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/ge_off", "thrust"), 0.12 * isa(0.3) * n * n * std::pow(kD, 4)), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/ge_low", "thrust"), 4.0 / 3.0 * 0.12 * isa(0.05) * n * n * std::pow(kD, 4)), 1e-9);
  EXPECT_NE(get(*s, "/devices/ge_all", "ge_ratio"), get(*s, "/devices/ge_no4", "ge_ratio"));  // the mask changed the hit
}

// (c5) ISA: standard-atmosphere table values 1.2250 (0 m) and 1.1117 (1000 m: 1.11164 from the exact formula); the table
// is rounded to 5 digits, 1e-4 relative covers that. Valid before the first tick.
TEST(devices, environment_isa_density) {
  auto a = load("env.xml");
  EXPECT_LT(rel(get(*a, "/devices/air", "density"), 1.225), 1e-4);
  ASSERT_EQ(a->step(1), 1);
  EXPECT_LT(rel(get(*a, "/devices/air", "density"), 1.225), 1e-4);
  auto b = load("env_alt.xml");
  EXPECT_LT(rel(get(*b, "/devices/air", "density"), 1.11164), 1e-4);
  ASSERT_EQ(b->step(1), 1);
  EXPECT_LT(rel(get(*b, "/devices/air", "density"), 1.11164), 1e-4);
  EXPECT_EQ(b->unit("/devices/air", "density"), "kg/m^3");
}

// (c5) Body drag F = -1/2 rho(z) Cd A |v_rel| v_rel with v_rel = v - wind, for a free body at 100 m moving at (3, -2, 1) in a
// (4, 1, 0) m/s wind; turbulence off = exactly 0. Closed form: 1e-9. Also the wind input's validation.
TEST(devices, body_drag_formula_and_wind_input) {
  auto s = load("env.xml");
  put_vec(*s, "/devices/air", "wind", {4, 1, 0});
  put_vec(*s, "/world/blob/fj", "qvel", {3, -2, 1, 0, 0, 0});
  ASSERT_EQ(s->step(1), 1);
  const double vr[3] = {-1, -3, 1}, k = -0.5 * isa(100.0) * 1.1 * 0.04 * std::sqrt(11.0);
  std::vector<double> f, wd, tb;
  ASSERT_EQ(s->get("/devices/drag", "force", "", f).code, Result::ok);
  ASSERT_EQ(f.size(), 3u);
  const double* xf = xfrc(*s, "blob");
  for (int i = 0; i < 3; ++i) {
    EXPECT_LT(rel(f[i], k * vr[i]), 1e-9) << i;
    EXPECT_LT(rel(xf[i], k * vr[i]), 1e-9) << i;
  }
  ASSERT_EQ(s->get("/devices/air", "wind", "", wd).code, Result::ok);
  EXPECT_EQ(wd, (std::vector<double>{4, 1, 0}));
  ASSERT_EQ(s->get("/devices/air", "turbulence", "", tb).code, Result::ok);
  EXPECT_EQ(tb, (std::vector<double>{0, 0, 0}));
  EXPECT_EQ(s->unit("/devices/drag", "force"), "N");
  EXPECT_EQ(s->unit("/devices/air", "wind"), "m/s");
  const double two[2] = {1, 2}, bad[3] = {1, NAN, 0};
  EXPECT_EQ(s->set("/devices/air", "wind", "", two).code, Result::invalid_argument);
  EXPECT_EQ(s->set("/devices/air", "wind", "", bad).code, Result::invalid_argument);
  EXPECT_EQ(s->set("/devices/air", "turbulence", "", std::span<const double>(bad)).code, Result::invalid_argument);  // read-only
  EXPECT_EQ(s->set("/devices/drag", "force", "", std::span<const double>(bad)).code, Result::invalid_argument);
  put_vec(*s, "/devices/air", "wind", {0, 0, 0});  // still accepted after the rejected sets
}

// (c5) Dryden turbulence: same seed -> identical sequence (bitwise), different seed -> different, and the stationary
// per-axis std matches MIL-F-8785C: h = 10 ft (low clamp), W20 = 10 m/s: sigma_w = 1, sigma_u = sigma_v = sigma_w / k^0.4,
// k = 0.177 + 0.000823 h_ft. Run: 400 000 ticks = 400 s after a 20 s warm-up; the slowest scale is T_u = L_u / V ~ 1.2 s, so
// ~330 correlation times: the relative standard error of a std estimate is ~ sqrt(T_u / (N dt)) ~ 5%. 20% = 4 sigma.
TEST(devices, environment_dryden_turbulence) {
  auto a = load("env_turb.xml"), b = load("env_turb.xml");
  const auto& ta = ms(*a).devices()->env.st[0].turb;
  const auto& tb = ms(*b).devices()->env.st[0].turb;
  double peak = 0;
  for (int i = 0; i < 2000; ++i) {
    ASSERT_EQ(a->step(1), 1);
    ASSERT_EQ(b->step(1), 1);
    for (int k = 0; k < 3; ++k) {
      ASSERT_EQ(ta[k], tb[k]) << i;
      peak = std::max(peak, std::abs(ta[k]));
    }
  }
  EXPECT_GT(peak, 0.1);
  EXPECT_EQ(a->save(), b->save());
  auto c = load("env_turb_seed6.xml");  // other seed: another sequence
  const auto& tc = ms(*c).devices()->env.st[0].turb;
  ASSERT_EQ(c->step(2000), 2000);
  EXPECT_NE(tc[0], ta[0]);
  EXPECT_NE(tc[2], ta[2]);

  const double kk = 0.177 + 0.000823 * 10.0, sig[3] = {1.0 / std::pow(kk, 0.4), 1.0 / std::pow(kk, 0.4), 1.0};
  const int warm = 20000, n = 400000;
  double s1[3] = {0, 0, 0}, s2[3] = {0, 0, 0};
  for (int i = 0; i < warm + n; ++i) {
    ASSERT_EQ(a->step(1), 1);
    if (i < warm) continue;
    for (int k = 0; k < 3; ++k) {
      s1[k] += ta[k];
      s2[k] += ta[k] * ta[k];
    }
  }
  for (int k = 0; k < 3; ++k) {
    const double mean = s1[k] / n, sd = std::sqrt(s2[k] / n - mean * mean);
    EXPECT_NEAR(sd, sig[k], 0.2 * sig[k]) << "axis " << k << " std " << sd << " sigma " << sig[k];
    EXPECT_LT(std::abs(mean), 0.3 * sig[k]) << "axis " << k;
  }
}

// Rotor + environment (turbulence, wind, ground effect, body drag) on a free airframe: save / restore / continue is bitwise
// identical to the uninterrupted run, including the Dryden filter states and RNG.
TEST(devices, save_restore_bitwise_aero) {
  auto a = load("aero_full.xml"), b = load("aero_full.xml");
  run(*a, 0, 3000);
  run(*b, 0, 1500);
  const std::vector<std::byte> snap = b->save();
  const std::uint64_t h_mid = b->state_hash();
  run(*b, 1500, 2200);
  ASSERT_NE(b->state_hash(), h_mid);
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), h_mid);
  run(*b, 1500, 3000);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->save(), b->save());
  EXPECT_FALSE(a->diverged());
  const auto& env = ms(*a).devices()->env.st[0];
  EXPECT_GT(std::abs(env.turb[0]) + std::abs(env.turb[1]) + std::abs(env.turb[2]), 0.0);  // turbulence really ran
  EXPECT_GT(get(*a, "/devices/prop", "thrust"), 0.0);
  EXPECT_GT(get(*a, "/devices/prop", "ge_ratio"), 0.999);
  EXPECT_NE(get(*a, "/devices/drag", "force"), 0.0);
}

// ======== Stage S3: servo ========
namespace {
// Servo of the fixtures (test values): 2100 KV, R20 4.4, I0 0.05, w_s 50, N 345, eta 0.7, 4096 counts, 100 Hz firmware.
constexpr double kSKv = 2100, kSR20 = 4.4, kSI0 = 0.05, kSWs = 50, kN = 345, kEta = 0.7, kCpr = 4096;
double ske() { return 60.0 / (2.0 * kPi * kSKv); }
double sr_at(double t) { return kSR20 * (1 + kAlpha * (t - 20)); }
double counts_of(double rad) { return std::floor(rad * kCpr / (2 * kPi) + 0.5); }
int jnt_id(const Solver& s, const char* j) { return mj_name2id(&ms(s).model(), mjOBJ_JOINT, j); }
int qadr(const Solver& s, const char* j) { return ms(s).model().jnt_qposadr[jnt_id(s, j)]; }
int dofadr(const Solver& s, const char* j) { return ms(s).model().jnt_dofadr[jnt_id(s, j)]; }
void enable(Solver& s, const std::string& servo, double goal) {
  put_ok(s, servo, "torque_enable", 1);
  put_ok(s, servo, "goal_position", goal);
}
// Two chained servos on a schedule that depends on the tick only (goal changes every tick; sa also ramps at 2 rad/s, sb
// jumps and so engages its 2 A current limit).
void run_servo(Solver& s, int t0, int t1, int* limited = nullptr) {
  for (int t = t0; t < t1; ++t) {
    if (t == 0) {
      enable(s, "/devices/sa", 0);
      enable(s, "/devices/sb", 0);
      put_ok(s, "/devices/sa", "goal_speed", 2.0);
    }
    put_ok(s, "/devices/sa", "goal_position", 0.9 * std::sin(t * 0.002));
    put_ok(s, "/devices/sb", "goal_position", 0.6 * std::sin(t * 0.003 + 1.0));
    ASSERT_EQ(s.step(1), 1);
    if (limited && ms(s).devices()->drv_mode[1] == 2) ++*limited;
  }
}
}  // namespace

// (e1) Step response of a position servo on a light link (zero gravity), goal step of 0.5 rad = 326 counts: settles to
// within 1 count of the goal and stays there for the rest of a 5 s run. Rise (10-90%) and settle times are printed;
// the 2 s settle bound is ~2x the measured 0.9 s (PI loop, closed-loop tau ~ 0.1 s) and guards against a stuck loop.
TEST(devices, servo_step_response) {
  auto s = load("servo_step.xml");
  EXPECT_EQ(get(*s, "/devices/s1", "torque_enable"), 0.0);  // power-up: torque off
  ASSERT_EQ(s->step(50), 50);
  EXPECT_EQ(get(*s, "/devices/s1", "current"), 0.0);  // open phase draws nothing
  EXPECT_LT(std::abs(get(*s, "/devices/bench", "current")), 1e-9);  // (gmin leak only)
  enable(*s, "/devices/s1", 0.5);
  const double goal = counts_of(0.5);
  ASSERT_EQ(goal, 326.0);
  const int n = 5000;
  int t10 = -1, t90 = -1, last_bad = -1;
  double peak = 0;
  for (int t = 0; t < n; ++t) {
    ASSERT_EQ(s->step(1), 1);
    const double c = get(*s, "/devices/s1", "counts");
    if (t10 < 0 && c >= 0.1 * goal) t10 = t;
    if (t90 < 0 && c >= 0.9 * goal) t90 = t;
    if (std::abs(goal - c) > 1) last_bad = t;
    peak = std::max(peak, c);
  }
  std::printf("[servo step] rise 10-90%% %d ms, last |error| > 1 count at %d ms, peak %.0f counts (goal %.0f)\n",
              t90 - t10, last_bad, peak, goal);
  EXPECT_LT(last_bad, 2000);  // settled within 2 s ...
  EXPECT_LT(last_bad, n - 1);  // ... and stayed for the remaining time
  EXPECT_LE(std::abs(goal - get(*s, "/devices/s1", "counts")), 1.0);
  EXPECT_NEAR(get(*s, "/devices/s1", "position"), get(*s, "/devices/s1", "counts") * 2 * kPi / kCpr, 1e-12);
  EXPECT_FALSE(s->diverged());
  EXPECT_GT(get(*s, "/devices/s1", "temperature"), kTamb);  // the winding saw current
  EXPECT_EQ(s->unit("/devices/s1", "goal_position"), "rad");
  EXPECT_EQ(s->unit("/devices/s1", "torque"), "N*m");
  EXPECT_EQ(get(*s, "/devices/s1", "backlash"), 0.0);
}

// (e2) Stall against a hard stop (joint limit at 0.3 rad, goal beyond it, max_torque 800 -> |duty| 0.8, zero
// gravity). Static output torque measured independently: with qacc ~ 0 and no other force on the dof, MuJoCo's
// constraint force balances the motor torque exactly (-qfrc_constraint = tau_out), whatever the limit's softness.
// Expected tau_out = N eta Kt d V / R(T) - N eta tau_f with d = 0.8 as commanded (not read back), V_node and T read
// from the device (T of the tick start). 1%: residual velocity / acceleration of the soft limit and the damping term
// b w (~0 at rest), plus the loss factor's smoothing at the residual speed (k - eta ~ 0.4 |tanh(w_m / w_s)| < 1% for
// |w_out| < 1e-3 rad/s).
TEST(devices, servo_stall_torque_vs_constraint_force) {
  auto s = load("servo_stall.xml");
  enable(*s, "/devices/s1", 1.0);
  ASSERT_EQ(s->step(3000), 3000);
  const double t_pre = get(*s, "/devices/s1", "temperature");
  ASSERT_EQ(s->step(1), 1);
  const mjData& d = ms(*s).data();
  const int dof = dofadr(*s, "out");
  // Soft limit: penetrates a little under the load (~ a0 (1-d)/d / K ~ 8e-3 rad).
  const double q = d.qpos[qadr(*s, "out")];
  EXPECT_GT(q, 0.3);
  EXPECT_LT(q, 0.32);
  const double w_m = ms(*s).devices()->servo.st[0].omega, v = get(*s, "/devices/s1", "voltage");
  std::printf("[servo stall] q %.4f rad (limit 0.3), w_out %.2e rad/s\n", q, w_m / kN);
  EXPECT_LT(std::abs(w_m / kN), 1e-3);  // at rest: no chatter of the gearbox loss factor (see drive_motor)
  const double kt = ske(), dd = 0.8;
  const double want = kN * kEta * (kt * dd * v / sr_at(t_pre) - kt * kSI0 * std::tanh(w_m / kSWs));
  EXPECT_GT(want, 1.0);  // N m: a real stall (~2.4)
  EXPECT_LT(rel(-d.qfrc_constraint[dof], want), 0.01) << "constraint " << -d.qfrc_constraint[dof] << " want " << want;
  EXPECT_LT(rel(get(*s, "/devices/s1", "torque"), want), 0.01) << "attr " << get(*s, "/devices/s1", "torque");
  EXPECT_LT(rel(get(*s, "/devices/s1", "duty"), dd), 1e-12);
  EXPECT_LT(rel(get(*s, "/devices/s1", "current"), dd * v / sr_at(t_pre)), 1e-3);
}

// (e3) The damping split is exact in both regimes: (-b qvel + qfrc_applied) == tau_out, recomputed here from V_node,
// w, T, duty: tau_out = N eta tau_m while the motor delivers power (tau_m w >= 0), N tau_m when back-driven,
// b = k Kt Ke N^2 / R.
TEST(devices, servo_damping_split_exact) {
  for (const bool driving : {true, false}) {
    auto s = load("servo_step.xml");
    if (driving) {
      enable(*s, "/devices/s1", 0.5);
      ASSERT_EQ(s->step(8), 8);  // accelerating at full duty: well below the no-load speed, tau_m > 0
    } else {  // holding 13 counts ahead, but the load drives the shaft at 6 rad/s: the motor brakes it
      enable(*s, "/devices/s1", 0.02);
      ASSERT_EQ(s->step(15), 15);
      put_ok(*s, "/world/link/out", "qvel", 6.0);
    }
    const double t_pre = get(*s, "/devices/s1", "temperature");
    ASSERT_EQ(s->step(1), 1);
    const mjModel& m = ms(*s).model();
    const mjData& d = ms(*s).data();
    const int dof = dofadr(*s, "out");
    const double w_m = ms(*s).devices()->servo.st[0].omega, w = w_m / kN;  // speed at the start of the tick
    const double v = get(*s, "/devices/s1", "voltage"), duty = get(*s, "/devices/s1", "duty");
    const double kt = ske(), r = sr_at(t_pre);
    const double im = (duty * v - kt * kN * w) / r, tau_m = kt * im - kt * kSI0 * std::tanh(w_m / kSWs);
    ASSERT_GT(std::abs(w), 0.5);
    EXPECT_EQ(tau_m * w >= 0, driving) << "tau_m " << tau_m << " w " << w;
    // Loss factor: the sharp rule (eta driving, 1 back-driven) ramped over w_s (see drive_motor); at these speeds
    // (|w_m| > 1000 >> w_s) tanh saturates and it IS the sharp rule.
    const double g = std::tanh(w_m / kSWs) * (tau_m > 0 ? 1.0 : -1.0);
    const double k = g >= 0 ? kEta : kEta + (1 - kEta) * -g, want = kN * k * tau_m;
    ASSERT_NEAR(k, driving ? kEta : 1.0, 1e-15);
    const double lhs = -m.dof_damping[dof] * w + d.qfrc_applied[dof];
    EXPECT_LT(rel(lhs, want), 1e-12) << "driving " << driving << " lhs " << lhs << " want " << want;
    EXPECT_LT(rel(get(*s, "/devices/s1", "torque"), want), 1e-12);
    EXPECT_LT(rel(m.dof_damping[dof], k * kt * kt * kN * kN / r), 1e-12);  // b = k Kt Ke N^2 / R(T)
    EXPECT_LT(rel(get(*s, "/devices/s1", "current"), im), 1e-12);
    EXPECT_DOUBLE_EQ(m.dof_armature[dof], 5e-8 * kN * kN);  // reflected rotor inertia J N^2, SET
  }
}

// (e4) Two servos daisy-chained (PSU -> w1 0.2 ohm -> sa -> w2 0.3 ohm -> sb). sa runs at full duty; sb is turned at
// 6 rad/s by its load against a goal 13 counts ahead (d ~ 0.05), so I_m < 0 and I_bus = d I_m < 0: it regenerates
// into the bus. Node KCL from the attrs: I(w2) = I_bus(sb), I(w1) = I_bus(sa) + I_bus(sb), I(psu) = I(w1) (the
// 1e-12 S gmin leak is ~1e-11 A of ~1 A: 1e-9 relative is rounding plus that leak).
TEST(devices, servo_chain_kcl_and_regen) {
  auto s = load("servo_chain.xml");
  enable(*s, "/devices/sa", 1.0);
  enable(*s, "/devices/sb", 0.02);
  ASSERT_EQ(s->step(15), 15);
  put_ok(*s, "/world/lb/jb", "qvel", 6.0);
  ASSERT_EQ(s->step(1), 1);
  const double ia = get(*s, "/devices/sa", "bus_current"), ib = get(*s, "/devices/sb", "bus_current");
  EXPECT_GT(ia, 0.1);
  EXPECT_LT(ib, 0.0);                                      // regenerating
  EXPECT_LT(get(*s, "/devices/sb", "current"), -1.0);      // phase current reversed
  EXPECT_GT(get(*s, "/devices/sb", "duty"), 0.0);
  EXPECT_LT(get(*s, "/devices/sb", "torque"), 0.0);        // braking
  EXPECT_LT(rel(get(*s, "/devices/w2", "current"), ib), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/w1", "current"), ia + ib), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/bench", "current"), ia + ib), 1e-9);
  EXPECT_LT(get(*s, "/devices/w2", "current"), 0.0);       // flows from sb back towards sa
  EXPECT_LT(get(*s, "/devices/bench", "current"), ia);     // the regen current offsets the draw of sa
  // Pushing into w2: node 3 above node 2.
  EXPECT_GT(get(*s, "/devices/sb", "voltage"), get(*s, "/devices/sa", "voltage"));
  EXPECT_LT(get(*s, "/devices/sa", "voltage"), get(*s, "/devices/bench", "voltage"));
  for (const char* sv : {"/devices/sa", "/devices/sb"})  // each servo's own consistency: I_bus = d I_m
    EXPECT_LT(rel(get(*s, sv, "bus_current"), get(*s, sv, "duty") * get(*s, sv, "current")), 1e-12) << sv;
}

// (e5) Backlash: the gear output is held at its goal (0) while gravity on a 0.5 m offset link mass pushes the link
// against the +b stop (b = 0.05 rad = the MJCF range of "play"): the link moves by b relative to the gear output, the
// encoder (on the gear output) still reads the goal. The stop is a soft MuJoCo limit: the equilibrium penetration is
// r = a0 (1 - d)/d / K (reference weight = true inverse inertia, see bind()), with impedance d = 0.95, constraint
// stiffness K = 1 / (d^2 tc^2) ~ 2770 s^-2 (tc = solref 0.02 s) and the free acceleration of the play dof
// a0 = A tau_g, tau_g = 0.245 N m, A = 1 / J_link + 1 / J_gear = 80 + 168 (the servo cancels tau_g on the gear dof,
// so the link accelerates against the gear side's reflected inertia too): a0 ~ 61 rad/s^2, r ~ 1.2e-3 rad = 2.3% of
// b (measured value printed). Bound: 5% of b = 2.5e-3 rad, about twice the soft-limit penetration. Without the
// backlash joint the link follows the gear output, which is held within 1 count.
TEST(devices, servo_backlash) {
  auto s = load("servo_backlash.xml"), n = load("servo_nobacklash.xml");
  for (Solver* x : {s.get(), n.get()}) {
    enable(*x, "/devices/s1", 0.0);
    ASSERT_EQ(x->step(3000), 3000);
    EXPECT_EQ(get(*x, "/devices/s1", "counts"), 0.0);
    EXPECT_FALSE(x->diverged());
  }
  const double b = 0.05, one_count = 2 * kPi / kCpr;
  EXPECT_EQ(get(*s, "/devices/s1", "backlash"), b);
  std::printf("[servo backlash] play %.5f rad (b %.5f), gear output %.2e rad\n",
              ms(*s).data().qpos[qadr(*s, "play")], b, ms(*s).data().qpos[qadr(*s, "out")]);
  EXPECT_NEAR(ms(*s).data().qpos[qadr(*s, "play")], b, 0.05 * b);
  EXPECT_LT(std::abs(ms(*s).data().qpos[qadr(*s, "out")]), one_count);  // gear output at the goal
  EXPECT_EQ(get(*n, "/devices/s1", "backlash"), 0.0);
  EXPECT_LT(std::abs(ms(*n).data().qpos[qadr(*n, "out")]), one_count);  // link rigid on the output: stays put
}

// (e6) Inputs through the bus: SetAttr goal_position / torque_enable are logged, and replaying the log (model rebuilt
// from the header bytes) reproduces the final state hash, with Snapshot / Restore in the mix. Invalid values are
// rejected and change nothing.
TEST(devices, servo_bus_log_and_replay) {
  Session sess;
  sess.caps = {"read", "control"};
  sess.root = DVSIM_TEST_DATA_DIR;
  auto cmd = [](const std::string& type, const std::string& body) {
    return R"({"id":1,"source":"t","cmd_type":")" + type + R"(","cmd":)" + body + "}";
  };
  auto set_attr = [&](const char* attr, double v) {
    const std::string body = std::string(R"({"path":"/devices/s1","attr":")") + attr + R"(","values":[)";
    return cmd("SetAttr", body + std::to_string(v) + "]}");
  };
  World w(load_mujoco, kDir + "servo_step.xml");
  w.start_log();
  for (const std::string& c :
       {set_attr("torque_enable", 1), set_attr("goal_position", 0.4), cmd("Step", R"({"n":300})"),
        cmd("Snapshot", "{}"), set_attr("goal_position", -0.3), set_attr("goal_speed", 1.5),
        cmd("Step", R"({"n":403})"), cmd("Restore", R"({"id":1})"), set_attr("goal_position", 0.2),
        cmd("Step", R"({"n":500})")}) {
    const std::string rep = execute_json(w, c, sess);
    ASSERT_NE(rep.find("\"ok\": true"), std::string::npos) << c << " -> " << rep;
  }
  World r = replay(w.log()->bytes(), load_mujoco_bytes, load_mujoco, DVSIM_TEST_DATA_DIR);
  EXPECT_EQ(r.tick(), w.tick());
  EXPECT_EQ(r.state_hash(), w.state_hash());
  EXPECT_NE(w.state_hash(), load("servo_step.xml")->state_hash());

  auto s = load("servo_step.xml");
  put_ok(*s, "/devices/s1", "goal_position", 0.25);
  for (const double bad : std::initializer_list<double>{NAN, INFINITY, -INFINITY, 2e6})
    EXPECT_EQ(put(*s, "/devices/s1", "goal_position", bad).code, Result::invalid_argument) << bad;
  for (const double bad : std::initializer_list<double>{0.5, 2.0, -1.0, NAN})
    EXPECT_EQ(put(*s, "/devices/s1", "torque_enable", bad).code, Result::invalid_argument) << bad;
  EXPECT_EQ(put(*s, "/devices/s1", "goal_speed", -1).code, Result::invalid_argument);
  EXPECT_EQ(put(*s, "/devices/s1", "counts", 1).code, Result::invalid_argument);  // read-only
  EXPECT_EQ(get(*s, "/devices/s1", "goal_position"), 0.25);                         // rejected sets wrote nothing
  EXPECT_EQ(get(*s, "/devices/s1", "torque_enable"), 0.0);
}

// Two servos (one current-limited, one with a goal-speed profile) on a shared bus: save mid-run at tick 1503, which is
// not a firmware tick (period 10), restore, continue == uninterrupted run, bitwise. Also clone.
TEST(devices, servo_save_restore_bitwise) {
  auto a = load("servo_chain.xml"), b = load("servo_chain.xml");
  int limited = 0;
  run_servo(*a, 0, 3000, &limited);
  run_servo(*b, 0, 1503);
  ASSERT_NE(ms(*b).devices()->servo.st[0].ticks % 10, 0u);  // between firmware steps
  const std::vector<std::byte> snap = b->save();
  const std::uint64_t h_mid = b->state_hash();
  run_servo(*b, 1503, 2200);
  ASSERT_NE(b->state_hash(), h_mid);
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), h_mid);
  auto c = b->clone();
  EXPECT_EQ(c->state_hash(), h_mid);
  run_servo(*b, 1503, 3000);
  run_servo(*c, 1503, 3000);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->state_hash(), c->state_hash());
  EXPECT_EQ(a->save(), b->save());
  EXPECT_FALSE(a->diverged());
  EXPECT_GT(limited, 0);  // sb's current limit really engaged
  EXPECT_GT(std::abs(get(*a, "/devices/sa", "position")) + std::abs(get(*a, "/devices/sb", "position")), 0.01);
}

// ---- servo runtime registers + firmware protection (N6b); fixtures servo_protect.xml (7.4 V, hard stop at 0.3 rad,
// protection on) and servo_lowv.xml (3.5 V). Alarm bits (STS status byte): 1 voltage, 4 temperature, 32 overload.
namespace {
unsigned alarm_of(const Solver& s) { return static_cast<unsigned>(get(s, "/devices/s1", "alarm")); }
// Steps (at most `max` ticks) until `bit` is set; returns the tick count stepped, or -1.
int step_until_alarm(Solver& s, unsigned bit, int max) {
  for (int t = 1; t <= max; ++t) {
    if (s.step(1) != 1) return -1;
    if (alarm_of(s) & bit) return t;
  }
  return -1;
}
}  // namespace

// Runtime attrs: the duty limit is min(max_torque, torque_limit); at a stall the static output torque follows it (measured
// against MuJoCo's constraint force, as in servo_stall_torque_vs_constraint_force); ranges are enforced; velocity / load
// are read-only views (velocity = output shaft speed, load = duty).
TEST(devices, servo_runtime_registers_set_limits_and_gains) {
  struct Case { double max_torque, torque_limit, duty; };
  for (const Case c : {Case{1000, 500, 0.5}, Case{300, 1000, 0.3}, Case{1000, 1000, 1.0}}) {
    auto s = load("servo_protect.xml");
    put_ok(*s, "/devices/s1", "overload_load", 0);  // protection off: this test measures the stall
    put_ok(*s, "/devices/s1", "max_torque", c.max_torque);
    put_ok(*s, "/devices/s1", "torque_limit", c.torque_limit);
    enable(*s, "/devices/s1", 1.0);
    ASSERT_EQ(s->step(2000), 2000);
    const double t_pre = get(*s, "/devices/s1", "temperature");
    ASSERT_EQ(s->step(1), 1);
    const mjData& d = ms(*s).data();
    const double w_m = ms(*s).devices()->servo.st[0].omega, v = get(*s, "/devices/s1", "voltage");
    const double want = kN * kEta * (ske() * c.duty * v / sr_at(t_pre) - ske() * kSI0 * std::tanh(w_m / kSWs));
    EXPECT_LT(rel(-d.qfrc_constraint[dofadr(*s, "out")], want), 0.02) << c.max_torque << "/" << c.torque_limit;
    EXPECT_LT(rel(get(*s, "/devices/s1", "load"), c.duty), 1e-12);
    EXPECT_EQ(alarm_of(*s), 0u);
  }
  auto s = load("servo_protect.xml");
  EXPECT_EQ(get(*s, "/devices/s1", "p_gain"), 32.0);  // initial values come from the config
  EXPECT_EQ(get(*s, "/devices/s1", "max_torque"), 1000.0);
  EXPECT_EQ(get(*s, "/devices/s1", "torque_limit"), 1000.0);
  EXPECT_EQ(put(*s, "/devices/s1", "torque_limit", 1001).code, Result::invalid_argument);
  EXPECT_EQ(put(*s, "/devices/s1", "velocity", 1).code, Result::invalid_argument);
  EXPECT_EQ(put(*s, "/devices/s1", "alarm", 0).code, Result::invalid_argument);
  put_ok(*s, "/devices/s1", "p_gain", 0);
  put_ok(*s, "/devices/s1", "i_gain", 0);
  put_ok(*s, "/devices/s1", "d_gain", 0);
  enable(*s, "/devices/s1", 0.2);
  ASSERT_EQ(s->step(50), 50);
  EXPECT_EQ(get(*s, "/devices/s1", "duty"), 0.0);  // zero gains: no drive
  put_ok(*s, "/devices/s1", "p_gain", 32);
  ASSERT_EQ(s->step(30), 30);
  const double w = get(*s, "/devices/s1", "velocity");
  EXPECT_GT(w, 0.1);  // rad/s at the output shaft
  EXPECT_DOUBLE_EQ(w * kN, ms(*s).devices()->servo.st[0].omega);
  EXPECT_EQ(s->unit("/devices/s1", "velocity"), "rad/s");
  EXPECT_EQ(s->unit("/devices/s1", "max_torque"), "0.1%");
}

// Undervoltage: a 3.5 V supply is below min_voltage 4.0 V -> bit 1 within two ticks (the first tick has no solved voltage
// yet). The bit is not in the unload mask: the servo keeps driving. It latches while torque is on, and a tick with
// torque off clears it (and only a condition that no longer holds stays cleared).
TEST(devices, servo_protect_undervoltage_alarm_latches_until_torque_cycled) {
  auto s = load("servo_lowv.xml");
  EXPECT_EQ(alarm_of(*s), 0u);
  enable(*s, "/devices/s1", 1.0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 0u);  // nothing solved before tick 1
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 1u);
  ASSERT_EQ(s->step(50), 50);
  EXPECT_GT(std::abs(get(*s, "/devices/s1", "duty")), 0.5);  // alarm only, not released
  EXPECT_LT(get(*s, "/devices/s1", "voltage"), 4.0);
  put_ok(*s, "/devices/s1", "min_voltage", 3.0);  // condition gone ...
  ASSERT_EQ(s->step(5), 5);
  EXPECT_EQ(alarm_of(*s), 1u);  // ... still latched with torque on
  put_ok(*s, "/devices/s1", "torque_enable", 0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 0u);  // cleared by a torque-off tick
  put_ok(*s, "/devices/s1", "min_voltage", 4.0);  // condition back: live even with torque off
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 1u);
  put_ok(*s, "/devices/s1", "max_voltage", 3.0);  // over-voltage side: v > max_voltage
  put_ok(*s, "/devices/s1", "min_voltage", 0);
  ASSERT_EQ(s->step(1), 1);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 1u);
  put_ok(*s, "/devices/s1", "max_voltage", 0);  // both off
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 0u);
  EXPECT_FALSE(s->diverged());
}

// Overheat by sustained stall heating (overload off, max_temp lowered to 30 degC): bit 4 appears the tick the winding
// passes 30 degC, the torque is released (4 is in the unload mask), and the latch outlives the condition until the torque
// is cycled with the winding below max_temp.
TEST(devices, servo_protect_overheat_by_stall_heating) {
  auto s = load("servo_protect.xml");
  put_ok(*s, "/devices/s1", "overload_load", 0);
  put_ok(*s, "/devices/s1", "max_temp", 30);
  enable(*s, "/devices/s1", 1.0);
  const int t = step_until_alarm(*s, 4, 30000);
  ASSERT_GT(t, 0);
  std::printf("[servo protect] overheat alarm at %d ms, T %.2f degC\n", t, get(*s, "/devices/s1", "temperature"));
  EXPECT_EQ(alarm_of(*s), 4u);
  EXPECT_GT(get(*s, "/devices/s1", "temperature"), 30.0);
  EXPECT_GT(t, 1000);  // heating takes seconds (about 1 K/s at stall)
  ASSERT_EQ(s->step(2), 2);
  EXPECT_EQ(get(*s, "/devices/s1", "duty"), 0.0);  // released
  EXPECT_EQ(get(*s, "/devices/s1", "current"), 0.0);
  EXPECT_EQ(get(*s, "/devices/s1", "torque_enable"), 1.0);
  const double t_hot = get(*s, "/devices/s1", "temperature");
  ASSERT_EQ(s->step(1000), 1000);
  EXPECT_LT(get(*s, "/devices/s1", "temperature"), t_hot);  // cooling, still latched and released
  EXPECT_EQ(alarm_of(*s), 4u);
  EXPECT_EQ(get(*s, "/devices/s1", "duty"), 0.0);
  put_ok(*s, "/devices/s1", "max_temp", 20);  // below the (ambient 25 degC) winding: the condition holds
  put_ok(*s, "/devices/s1", "torque_enable", 0);  // torque-off tick: the alarm is raised again at once
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 4u);
  put_ok(*s, "/devices/s1", "max_temp", 70);  // threshold above the winding temperature: a torque-off tick clears
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 0u);
  put_ok(*s, "/devices/s1", "torque_enable", 1);
  ASSERT_EQ(s->step(30), 30);
  EXPECT_GT(get(*s, "/devices/s1", "duty"), 0.5);  // driving again
}

// Overload: stalled against the stop at full duty (> overload_load 0.8) for overload_time (shortened to 0.5 s) -> bit 32
// and release. With unload_mask 0 the alarm is reported and the servo keeps pushing.
TEST(devices, servo_protect_overload_releases_after_protection_time) {
  auto s = load("servo_protect.xml");
  put_ok(*s, "/devices/s1", "overload_time", 0.5);
  enable(*s, "/devices/s1", 1.0);
  const int t = step_until_alarm(*s, 32, 3000);
  ASSERT_GT(t, 0);
  std::printf("[servo protect] overload alarm at %d ms (protection time 500 ms)\n", t);
  EXPECT_GE(t, 500);
  EXPECT_LE(t, 520);  // the duty is above 0.8 from the first firmware step on; the timer counts ticks
  EXPECT_EQ(alarm_of(*s), 32u);
  ASSERT_EQ(s->step(2), 2);
  EXPECT_EQ(get(*s, "/devices/s1", "load"), 0.0);
  EXPECT_EQ(get(*s, "/devices/s1", "current"), 0.0);  // torque released (open phase)
  ASSERT_EQ(s->step(500), 500);
  EXPECT_EQ(alarm_of(*s), 32u);  // latched
  EXPECT_EQ(get(*s, "/devices/s1", "load"), 0.0);
  put_ok(*s, "/devices/s1", "torque_enable", 0);
  ASSERT_EQ(s->step(1), 1);
  EXPECT_EQ(alarm_of(*s), 0u);
  put_ok(*s, "/devices/s1", "torque_enable", 1);
  ASSERT_EQ(s->step(300), 300);
  EXPECT_EQ(alarm_of(*s), 0u);  // timer restarted
  EXPECT_GT(get(*s, "/devices/s1", "load"), 0.8);
  EXPECT_FALSE(s->diverged());

  auto n = load("servo_protect.xml");  // reported only
  put_ok(*n, "/devices/s1", "overload_time", 0.5);
  put_ok(*n, "/devices/s1", "unload_mask", 0);
  enable(*n, "/devices/s1", 1.0);
  ASSERT_GT(step_until_alarm(*n, 32, 3000), 0);
  ASSERT_EQ(n->step(100), 100);
  EXPECT_GT(get(*n, "/devices/s1", "load"), 0.8);

  auto q = load("servo_protect.xml");  // a lower torque limit keeps the duty under the threshold: no overload
  put_ok(*q, "/devices/s1", "overload_time", 0.5);
  put_ok(*q, "/devices/s1", "torque_limit", 700);
  enable(*q, "/devices/s1", 1.0);
  ASSERT_EQ(q->step(1500), 1500);
  EXPECT_EQ(alarm_of(*q), 0u);
}

// Mid-overload save / restore / clone continue bitwise (timer and latches are in the saved state).
TEST(devices, servo_protect_save_restore_mid_overload_bitwise) {
  auto go = [](Solver& s, int t0, int t1) {
    for (int t = t0; t < t1; ++t) {
      if (t == 0) {
        put_ok(s, "/devices/s1", "overload_time", 0.5);
        enable(s, "/devices/s1", 1.0);
      }
      ASSERT_EQ(s.step(1), 1);
    }
  };
  auto a = load("servo_protect.xml"), b = load("servo_protect.xml");
  go(*a, 0, 900);
  go(*b, 0, 303);
  EXPECT_EQ(alarm_of(*b), 0u);
  EXPECT_GT(ms(*b).devices()->servo.st[0].ovl_t, 0.2);  // timer running
  const std::vector<std::byte> snap = b->save();
  const std::uint64_t h_mid = b->state_hash();
  go(*b, 303, 700);
  EXPECT_EQ(alarm_of(*b), 32u);
  ASSERT_NE(b->state_hash(), h_mid);
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), h_mid);
  auto c = b->clone();
  go(*b, 303, 900);
  go(*c, 303, 900);
  EXPECT_EQ(alarm_of(*a), 32u);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->state_hash(), c->state_hash());
  EXPECT_EQ(a->save(), b->save());
}

// d_scale: the first firmware step after enabling has e_prev = 0, so u = (p e + i e) / pid_scale + d e / d_scale; with
// d_scale absent (servo_step.xml, same config otherwise) the single-scale law u = (p + i + d) e / pid_scale, exactly.
TEST(devices, servo_d_scale_separates_the_derivative_scale) {
  const double p = 32, i = 2, d = 32, ps = 8192, ds = 16384, goal = 0.05;
  for (const bool separate : {false, true}) {
    auto s = load(separate ? "servo_dscale.xml" : "servo_step.xml");
    enable(*s, "/devices/s1", goal);
    ASSERT_EQ(s->step(1), 1);
    const auto& st = ms(*s).devices()->servo.st[0];
    const double e = std::floor(goal * 4096 / (2 * M_PI) + 0.5);
    ASSERT_EQ(st.e_prev, e);
    const double want = separate ? (p + i) * e / ps + d * e / ds : (p + i + d) * e / ps;
    EXPECT_LT(rel(st.cmd, want), 1e-12) << separate;
  }
}

TEST(devices, servo_bad_firmware_rate_fails_load) {
  try {
    load("bad_servo_rate.xml");
    ADD_FAILURE() << "loaded";
  } catch (const std::runtime_error& e) {
    const std::string w = e.what();
    EXPECT_NE(w.find("dvsim.devices: device 'servo_bad': "), std::string::npos) << w;
    EXPECT_NE(w.find("firmware_hz"), std::string::npos) << w;
  }
}

// ======== Stage S4: sensors, model constants, all device types together ========
namespace {
std::vector<double> getv(const Solver& s, const std::string& prim, const std::string& attr) {
  std::vector<double> o;
  const Result r = s.get(prim, attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << prim << "." << attr << ": " << r.message;
  return o;
}
double norm3(const std::vector<double>& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
constexpr double kG = 9.81;  // the fixtures' gravity
}  // namespace

// (f1) IMU at rest on a static body: the plate is welded to the world (MuJoCo's own acceleration of it omits gravity,
// the IMU adds it back: a bench IMU reads +g up). 20000 samples at 1 kHz: gyro sigma = 1e-3 sqrt(1000 / 2) = 0.0224
// rad/s, accel sigma = 4e-3 sqrt(500) = 0.0894 m/s^2. Sample mean within 3 sigma / sqrt(N) of the truth (a 3-sigma
// test: the seed is fixed, so it is deterministic), sample std within 10% (the relative standard error of a std over
// N = 20000 is 0.5%, so 10% is 20 sigma). imu_a and imu_b share a seed: bitwise equal streams; imu_c (other seed)
// differs; a second load of the same file gives the same final state bytes.
TEST(devices, imu_at_rest_noise_statistics) {
  auto s = load("imu_rest.xml"), s2 = load("imu_rest.xml");
  const int n = 20000;
  const double sg = 1e-3 * std::sqrt(500.0), sa = 4e-3 * std::sqrt(500.0);
  double sum[6] = {0, 0, 0, 0, 0, 0}, sq[6] = {0, 0, 0, 0, 0, 0};
  int a_ne_b = 0, a_ne_c = 0;
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(s->step(1), 1);
    const auto ga = getv(*s, "/devices/imu_a", "gyro"), aa = getv(*s, "/devices/imu_a", "accel");
    a_ne_b += ga != getv(*s, "/devices/imu_b", "gyro") || aa != getv(*s, "/devices/imu_b", "accel");
    a_ne_c += ga != getv(*s, "/devices/imu_c", "gyro");
    for (int k = 0; k < 3; ++k) {
      sum[k] += ga[k];
      sq[k] += ga[k] * ga[k];
      sum[3 + k] += aa[k];
      sq[3 + k] += aa[k] * aa[k];
    }
  }
  ASSERT_EQ(s2->step(n), n);
  EXPECT_EQ(s->save(), s2->save());
  EXPECT_EQ(a_ne_b, 0);
  EXPECT_GT(a_ne_c, n * 99 / 100);
  const double truth[6] = {0, 0, 0, 0, 0, kG}, sigma[6] = {sg, sg, sg, sa, sa, sa};
  for (int k = 0; k < 6; ++k) {
    const double mean = sum[k] / n, sd = std::sqrt(sq[k] / n - mean * mean);
    EXPECT_LE(std::abs(mean - truth[k]), 3 * sigma[k] / std::sqrt(n)) << "axis " << k << " mean " << mean;
    EXPECT_NEAR(sd, sigma[k], 0.1 * sigma[k]) << "axis " << k;
  }
  EXPECT_EQ(s->unit("/devices/imu_a", "gyro"), "rad/s");
  EXPECT_EQ(s->unit("/devices/imu_a", "accel"), "m/s^2");
}

// (f2) One error term at a time, noise off, on the tilted site (g in the site frame = (0, g sin 30, g cos 30)).
// Quantisation: 8 bits over +-16 -> LSB 0.125; saturation at 6; bias + scale: out = (1 + s) truth + b. Exact
// arithmetic on the same doubles (1e-12 absolute covers the rotation of g into the tilted frame).
TEST(devices, imu_quantisation_saturation_bias_scale) {
  auto s = load("imu_rest.xml");
  ASSERT_EQ(s->step(1), 1);
  const double gy = kG * std::sin(kPi / 6), gz = kG * std::cos(kPi / 6), lsb = 2.0 * 16 / 256;
  const auto q = getv(*s, "/devices/imu_q", "accel");
  EXPECT_NEAR(q[0], 0.0, 1e-12);
  EXPECT_NEAR(q[1], std::round(gy / lsb) * lsb, 1e-12);  // 4.905 / 0.125 = 39.24 -> 4.875
  EXPECT_NEAR(q[2], std::round(gz / lsb) * lsb, 1e-12);  // 8.4957 / 0.125 = 67.97 -> 8.5
  EXPECT_NE(q[1], gy);
  EXPECT_DOUBLE_EQ(q[1], 4.875);
  EXPECT_DOUBLE_EQ(q[2], 8.5);
  const auto sat = getv(*s, "/devices/imu_s", "accel");
  EXPECT_NEAR(sat[1], gy, 1e-12);  // below full scale: untouched
  EXPECT_EQ(sat[2], 6.0);          // 8.4957 clamps
  const auto ba = getv(*s, "/devices/imu_bs", "accel"), bg = getv(*s, "/devices/imu_bs", "gyro");
  EXPECT_NEAR(ba[0], 0.1, 1e-12);
  EXPECT_NEAR(ba[1], gy - 0.2, 1e-12);
  EXPECT_NEAR(ba[2], 0.98 * gz + 0.3, 1e-12);
  EXPECT_NEAR(bg[0], 0.01, 1e-12);  // truth 0: only the bias remains
  EXPECT_NEAR(bg[1], 0.02, 1e-12);
  EXPECT_NEAR(bg[2], 0.03, 1e-12);
}

// (f2) Bias random walk: increments of the bias between samples have std rw_density sqrt(1 / rate): gyro
// 0.01 sqrt(1e-3) = 3.16e-4 rad/s, accel 0.02 sqrt(1e-3) = 6.32e-4 m/s^2 (noise off: the output IS the bias, truth
// 0). 3 axes x 20000 increments: relative standard error of the std 0.4%; 10% band; mean within 3 sigma / sqrt(N).
TEST(devices, imu_bias_random_walk_increments) {
  auto s = load("imu_rest.xml");
  const int n = 20000;
  ASSERT_EQ(s->step(1), 1);  // the first sample: before it the outputs are 0, which is not a step of the walk
  std::vector<double> prev = getv(*s, "/devices/imu_rw", "gyro"), pa = getv(*s, "/devices/imu_rw", "accel");
  double sg1 = 0, sg2 = 0, sa1 = 0, sa2 = 0;
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(s->step(1), 1);
    const auto g = getv(*s, "/devices/imu_rw", "gyro"), a = getv(*s, "/devices/imu_rw", "accel");
    for (int k = 0; k < 3; ++k) {
      sg1 += g[k] - prev[k];
      sg2 += (g[k] - prev[k]) * (g[k] - prev[k]);
      sa1 += a[k] - pa[k];
      sa2 += (a[k] - pa[k]) * (a[k] - pa[k]);
    }
    prev = g;
    pa = a;
  }
  const double m = 3.0 * n, wg = 0.01 * std::sqrt(1e-3), wa = 0.02 * std::sqrt(1e-3);
  EXPECT_NEAR(std::sqrt(sg2 / m), wg, 0.1 * wg);
  EXPECT_NEAR(std::sqrt(sa2 / m), wa, 0.1 * wa);
  EXPECT_LE(std::abs(sg1 / m), 3 * wg / std::sqrt(m));
  EXPECT_LE(std::abs(sa1 / m), 3 * wa / std::sqrt(m));
  EXPECT_GT(norm3(prev), 0.0);  // it really walked
}

// (f3) Known motion: a body spinning at w = 5 rad/s about the vertical hinge, site 0.3 m off the axis, site frame
// turned 90 deg about z. gyro = (0, 0, w); accel = centripetal w^2 r toward the axis = site +y, plus the +g that the
// hinge supports: (0, w^2 r, g) = (0, 7.5, 9.81). With zero damping and no torque w is constant to rounding, so the
// only error is double rounding in the spatial transforms (~1e-15): 1e-9 relative leaves room for nothing else.
TEST(devices, imu_spin_gyro_and_centripetal) {
  auto s = load("imu_spin.xml");
  const double w = 5.0, r = 0.3;
  put_ok(*s, "/world/rotor/spin", "qvel", w);
  int done = 0;
  for (const int target : {1, 50, 333}) {  // different angles of the rotating body
    ASSERT_EQ(s->step(target - done), target - done);
    done = target;
    const auto g = getv(*s, "/devices/imu", "gyro"), a = getv(*s, "/devices/imu", "accel");
    EXPECT_NEAR(g[0], 0.0, 1e-9) << target;
    EXPECT_NEAR(g[1], 0.0, 1e-9) << target;
    EXPECT_LT(rel(g[2], w), 1e-9) << target;
    EXPECT_NEAR(a[0], 0.0, 1e-9 * w * w * r) << target;
    EXPECT_LT(rel(a[1], w * w * r), 1e-9) << target;
    EXPECT_LT(rel(a[2], kG), 1e-9) << target;
  }
}

// (f3) A pendulum that really accelerates, from rest at 1 rad. At each tick the state s_k is read BEFORE the step; the
// sample of that tick must equal the closed form at s_k: body frame x along the swing, z along the rod (up):
// f = (g sin(th) (m L^2 / I - 1), 0, w^2 L + g cos(th)), gyro (0, w, 0), I = m L^2 + 2/5 m r^2. Everything is
// evaluated on the same state as one rigid-body solve: 1e-9.
TEST(devices, imu_pendulum_matches_closed_form) {
  auto s = load("imu_pendulum.xml");
  put_ok(*s, "/world/arm/hinge", "qpos", 1.0);
  const double m = 0.2, len = 0.5, inertia = m * len * len + 0.4 * m * 0.02 * 0.02;
  for (int k = 0; k < 400; ++k) {
    const double th = getv(*s, "/world/arm/hinge", "qpos")[0], w = getv(*s, "/world/arm/hinge", "qvel")[0];
    ASSERT_EQ(s->step(1), 1);
    const auto a = getv(*s, "/devices/imu", "accel"), g = getv(*s, "/devices/imu", "gyro");
    EXPECT_NEAR(a[0], kG * std::sin(th) * (m * len * len / inertia - 1), 1e-9) << k;
    EXPECT_NEAR(a[1], 0.0, 1e-9) << k;
    EXPECT_NEAR(a[2], w * w * len + kG * std::cos(th), 1e-9) << k;
    EXPECT_NEAR(g[1], w, 1e-12) << k;
  }
}

// (f3) The acceleration must use the qvel of the start of the tick (mj_step2 has integrated it by the time the hook
// runs): for a link hanging from a moving link cacc contains cdof_dot * qvel. Reference = MuJoCo's own accelerometer
// / gyro at the same site, which mj_step2 evaluates before integrating (a model with it also takes the IMU's other code
// path: the acceleration data are already there). Both models run the same trajectory: the IMU must agree with the
// sensors, and the model without them, to 1e-12 (same arithmetic; the scale is ~10 m/s^2). Without the saved qvel
// the two differ by ~1e-4 (checked by disabling it once).
TEST(devices, imu_acceleration_uses_start_of_tick_velocity) {
  auto a = load("imu_double.xml"), b = load("imu_double_acc.xml");
  for (Solver* x : {a.get(), b.get()}) {
    put_ok(*x, "/world/arm1/h1", "qpos", 1.2);
    put_ok(*x, "/world/arm1/arm2/h2", "qpos", -0.7);
  }
  double seen = 0;
  for (int k = 0; k < 600; ++k) {
    ASSERT_EQ(a->step(1), 1);
    ASSERT_EQ(b->step(1), 1);
    const auto acc = getv(*b, "/sensors/acc", "value"), gyr = getv(*b, "/sensors/gyro", "value");
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(getv(*a, "/devices/imu", "accel")[i], getv(*b, "/devices/imu", "accel")[i], 1e-12) << k;
      EXPECT_NEAR(getv(*b, "/devices/imu", "accel")[i], acc[i], 1e-12) << k;
      EXPECT_NEAR(getv(*b, "/devices/imu", "gyro")[i], gyr[i], 1e-12) << k;
    }
    seen = std::max(seen, std::abs(getv(*a, "/devices/imu", "accel")[0]));
  }
  EXPECT_GT(seen, 1.0);  // it moved
}

// (f4) Barometer: p = p0 (1 - L h / T0)^(g0 / R L), h = origin_alt 300 + site z 100.5; noise off (1e-9). With bias
// 12.5 Pa and resolution 10 Pa: round((p + 12.5) / 10) 10.
TEST(devices, baro_matches_isa) {
  auto s = load("sens_static.xml");
  ASSERT_EQ(s->step(1), 1);
  const double p = 101325.0 * std::pow(1.0 - 0.0065 * 400.5 / 288.15, 9.80665 / (287.05287 * 0.0065));
  EXPECT_LT(rel(get(*s, "/devices/baro", "pressure"), p), 1e-9);
  EXPECT_NEAR(p, 96605, 50);  // sanity anchor: the ISA table gives 96611 Pa at 400 m (6 Pa per 0.5 m)
  EXPECT_NEAR(get(*s, "/devices/baro_q", "pressure"), std::round((p + 12.5) / 10) * 10, 1e-9);
  EXPECT_EQ(s->unit("/devices/baro", "pressure"), "Pa");
  EXPECT_EQ(put(*s, "/devices/baro", "pressure", 1).code, Result::invalid_argument);  // read-only
}

// (f5) Magnetometer: B_site = R^T B_world for a 40 deg yaw: (Bx c + By s, -Bx s + By c, Bz) (+ bias for mag_b).
// 1e-12 of |B|.
TEST(devices, mag_rotates_into_site_frame) {
  auto s = load("sens_static.xml");
  ASSERT_EQ(s->step(1), 1);
  const double bx = 2.1e-5, by = 0.4e-5, bz = -4.3e-5, c = std::cos(kPi * 40 / 180), sn = std::sin(kPi * 40 / 180);
  const double want[3] = {bx * c + by * sn, -bx * sn + by * c, bz}, bias[3] = {1e-6, -2e-6, 3e-6};
  const double bmag = std::sqrt(bx * bx + by * by + bz * bz);
  const auto m = getv(*s, "/devices/mag", "field"), mb = getv(*s, "/devices/mag_b", "field");
  for (int k = 0; k < 3; ++k) {
    EXPECT_LT(std::abs(m[k] - want[k]) / bmag, 1e-12) << k;
    EXPECT_LT(std::abs(mb[k] - want[k] - bias[k]) / bmag, 1e-12) << k;
  }
  EXPECT_EQ(s->unit("/devices/mag", "field"), "T");
}

namespace {
struct Geo { double lat, lon, alt; };
// Flat earth about (47.3769, 8.5417, 408): east / north in metres via the WGS84 meridian and prime-vertical radii.
Geo flat_earth(double east, double north, double up) {
  const double lat0 = 47.3769, a = 6378137.0, f = 1 / 298.257223563, e2 = f * (2 - f);
  const double phi = lat0 * kPi / 180, w = 1 - e2 * std::sin(phi) * std::sin(phi);
  const double m_rad = a * (1 - e2) / std::pow(w, 1.5), n_rad = a / std::sqrt(w);
  return {lat0 + north / m_rad * 180 / kPi, 8.5417 + east / (n_rad * std::cos(phi)) * 180 / kPi, 408 + up};
}
}  // namespace

// (f6) GNSS at a known offset (30 m E, -20 m N, 100.5 m up), noise off: the flat-earth formula to 1e-9 of the offset
// (in degrees; the rounding of lat itself is 7e-15 deg ~ 4e-11 of this offset). Independent cross-check of the radii:
// the standard series for the length of a degree at latitude phi (metres: 111132.92 - 559.82 cos 2phi + 1.175 cos 4phi
// per degree of latitude, 111412.84 cos phi - 93.5 cos 3phi + 0.118 cos 5phi per degree of longitude; truncation
// ~1e-8 rel).
TEST(devices, gnss_flat_earth_offset) {
  auto s = load("sens_static.xml");
  ASSERT_EQ(s->step(1), 1);
  const Geo want = flat_earth(30, -20, 100.5);
  const double lat = get(*s, "/devices/g_post", "lat"), lon = get(*s, "/devices/g_post", "lon");
  EXPECT_LT(rel(lat - 47.3769, want.lat - 47.3769), 1e-9);
  EXPECT_LT(rel(lon - 8.5417, want.lon - 8.5417), 1e-9);
  EXPECT_LT(rel(get(*s, "/devices/g_post", "alt"), want.alt), 1e-9);
  const double phi = 47.3769 * kPi / 180;
  const double per_lat = 111132.92 - 559.82 * std::cos(2 * phi) + 1.175 * std::cos(4 * phi);
  const double per_lon = 111412.84 * std::cos(phi) - 93.5 * std::cos(3 * phi) + 0.118 * std::cos(5 * phi);
  EXPECT_LT(rel((lat - 47.3769) * per_lat, -20.0), 1e-6);
  EXPECT_LT(rel((lon - 8.5417) * per_lon, 30.0), 1e-6);
  EXPECT_EQ(getv(*s, "/devices/g_post", "vel_ned"), (std::vector<double>{0, 0, 0}));
  EXPECT_EQ(s->unit("/devices/g_post", "lat"), "deg");
  EXPECT_EQ(s->unit("/devices/g_post", "alt"), "m");
  EXPECT_EQ(s->unit("/devices/g_post", "vel_ned"), "m/s");
}

// (f6) GNSS on a moving site (com (3, 4, 5) m/s, yaw 2 rad/s, site 0.5 m ahead): position and NED velocity are the
// closed form at the state s_j a sample describes, p(t) = (3t + .5 cos 2t, 4t + .5 sin 2t, 50 + 5t),
// v_ned = (4 + cos 2t, 3 - sin 2t, -5), t = j dt. Sample timing at 100 Hz (period 10 ticks): after N ticks the newest
// sample is j = floor((N - 1) / 10) taken from the state at the START of tick 10 j (one tick of latency: after tick 1
// the position is still that of tick 0), and it is held until tick 10 (j + 1) completes. g5 (0.05 s = 5 samples)
// outputs sample j - 5, the origin until then. 1e-12 deg absolute (0.1 micrometre; one sample step is ~3e-7 deg),
// 1e-9 m for the altitude and the velocity.
TEST(devices, gnss_timing_hold_and_latency) {
  auto s = load("sens_static.xml");
  put_vec(*s, "/world/mover/fj", "qvel", {3, 4, 5, 0, 0, 2});
  auto pos = [](int tick) {
    const double t = tick * 0.001;
    return flat_earth(3 * t + 0.5 * std::cos(2 * t), 4 * t + 0.5 * std::sin(2 * t), 50 + 5 * t);
  };
  auto check = [&](const char* dev, const Geo& want, double tick, int n) {
    const std::string p = std::string("/devices/") + dev;
    EXPECT_NEAR(get(*s, p, "lat"), want.lat, 1e-12) << dev << " N=" << n;
    EXPECT_NEAR(get(*s, p, "lon"), want.lon, 1e-12) << dev << " N=" << n;
    EXPECT_NEAR(get(*s, p, "alt"), want.alt, 1e-9) << dev << " N=" << n;
    if (tick >= 0) {
      const double t = tick * 0.001, ned[3] = {4 + std::cos(2 * t), 3 - std::sin(2 * t), -5};
      const auto v = getv(*s, p, "vel_ned");
      for (int k = 0; k < 3; ++k) EXPECT_NEAR(v[k], ned[k], 1e-9) << dev << " N=" << n << " axis " << k;
    }
  };
  double alt_prev = -1;
  int changes = 0;
  for (int n = 1; n <= 400; ++n) {
    ASSERT_EQ(s->step(1), 1);
    const int j = (n - 1) / 10;
    check("g0", pos(10 * j), 10 * j, n);
    if (j >= 5) {
      check("g5", pos(10 * (j - 5)), 10 * (j - 5), n);
    } else {
      check("g5", {47.3769, 8.5417, 408}, -1, n);
      EXPECT_EQ(getv(*s, "/devices/g5", "vel_ned"), (std::vector<double>{0, 0, 0}));
    }
    const double alt = get(*s, "/devices/g0", "alt");
    changes += alt != alt_prev;
    alt_prev = alt;
  }
  EXPECT_EQ(changes, 40);  // 400 ticks / 10: the output changes once per sample, i.e. it is held in between
}

// (f6) GNSS Gauss-Markov position error (sens_gnss_gm.xml, 100000 samples of the altitude error, sigma 3 m): the white
// receiver has variance 9 and lag-1 autocorrelation 0; the Gauss-Markov one has the SAME total variance and lag-1
// autocorrelation corr_frac x phi = 0.8 x exp(-0.02) = 0.7842 (the white 20 % does not correlate). The estimator error of
// the variance is ~4 % (effective sample ~1300), so 12 %; of the autocorrelation ~0.02, so 0.05.
TEST(devices, gnss_gauss_markov_error) {
  auto s = load("sens_gnss_gm.xml");
  std::vector<double> w, g;
  for (int i = 0; i < 100000; ++i) {
    ASSERT_EQ(s->step(1), 1);
    w.push_back(get(*s, "/devices/g_white", "alt") - 408);
    g.push_back(get(*s, "/devices/g_gm", "alt") - 408);
  }
  auto stats = [](const std::vector<double>& x) {
    double m = 0;
    for (double v : x) m += v;
    m /= static_cast<double>(x.size());
    double c0 = 0, c1 = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      c0 += (x[i] - m) * (x[i] - m);
      if (i + 1 < x.size()) c1 += (x[i] - m) * (x[i + 1] - m);
    }
    return std::pair{c0 / static_cast<double>(x.size()), c1 / c0};
  };
  const auto [vw, rw] = stats(w);
  const auto [vg, rg] = stats(g);
  EXPECT_NEAR(vw, 9.0, 0.27);
  EXPECT_NEAR(rw, 0.0, 0.02);
  EXPECT_NEAR(vg, 9.0, 1.08);
  EXPECT_NEAR(rg, 0.8 * std::exp(-0.02), 0.05);
}

TEST(devices, sensor_bad_config_fails_load_with_message) {
  struct Case { const char* file; const char* dev; const char* needle; };
  for (const Case& c : {Case{"bad_sensor_rate.xml", "imu_bad", "rate_hz"},
                        Case{"bad_gnss_latency.xml", "gnss_bad", "63"},
                        Case{"bad_site.xml", "baro_bad", "no_such_site"}}) {
    try {
      load(c.file);
      ADD_FAILURE() << c.file << " loaded";
    } catch (const std::runtime_error& e) {
      const std::string w = e.what();
      const std::string prefix = std::string("dvsim.devices: device '") + c.dev + "': ";
      EXPECT_NE(w.find(prefix), std::string::npos) << c.file << ": " << w;
      EXPECT_NE(w.find(c.needle), std::string::npos) << c.file << ": " << w;
    }
  }
}

// (B) The model constants follow the device-set armature (bind() refreshes them once with mj_setConst, for ESC and
// servo joints alike): on a one-dof model dof_invweight0 = 1 / dof_M0 and M0 holds the rotor inertia; and the refresh
// is idempotent: a model rebuilt from its own saved bytes (armature already set) has bitwise the same constants.
TEST(devices, model_constants_follow_device_armature_and_are_idempotent) {
  auto s = load("motor.xml");
  const mjModel& m = ms(*s).model();
  ASSERT_EQ(m.nv, 1);
  EXPECT_GE(m.dof_M0[0], 3e-5);
  EXPECT_LT(rel(m.dof_invweight0[0], 1.0 / m.dof_M0[0]), 1e-12);
  EXPECT_DOUBLE_EQ(m.dof_armature[0], 3e-5);
  for (const char* f : {"motor.xml", "motor_batt.xml", "servo_backlash.xml", "all.xml"}) {
    auto a = load(f);
    auto b = load_mujoco_bytes(a->model_bytes());
    const mjModel &ma = ms(*a).model(), &mb = ms(*b).model();
    ASSERT_EQ(ma.nv, mb.nv) << f;
    for (int i = 0; i < ma.nv; ++i) {
      EXPECT_EQ(ma.dof_M0[i], mb.dof_M0[i]) << f << " dof " << i;
      EXPECT_EQ(ma.dof_invweight0[i], mb.dof_invweight0[i]) << f << " dof " << i;
      EXPECT_EQ(ma.dof_armature[i], mb.dof_armature[i]) << f << " dof " << i;
    }
    EXPECT_EQ(a->state_hash(), b->state_hash()) << f;
  }
}

// ---- all device types in one model (all.xml) ----
namespace {
constexpr double kHover = 0.452;  // hover throttle of all.xml (found with all_free_quad_runs_without_divergence)
// Inputs that depend on the tick only: throttles around hover (a different phase per ESC), a servo goal sweep, wind
// steps.
void run_all(Solver& s, int t0, int t1) {
  for (int t = t0; t < t1; ++t) {
    if (t == 0) put_ok(s, "/devices/gimbal_pan", "torque_enable", 1);
    for (int k = 0; k < 4; ++k)
      put_ok(s, "/devices/esc" + std::to_string(k + 1), "throttle", kHover + 0.01 * std::sin(t * 0.004 + k));
    put_ok(s, "/devices/gimbal_pan", "goal_position", 0.6 * std::sin(t * 0.003));
    if (t % 500 == 0) put_vec(s, "/devices/air", "wind", {2.0 + std::sin(t * 0.001), 1.0, 0.0});
    ASSERT_EQ(s.step(1), 1);
  }
}
}  // namespace

// (g) A balanced free-floating quad (diagonal pairs spin opposite ways, so the reaction torques cancel) at hover
// throttle, with environment turbulence, wind, drag, the servo and all four sensors running, must not diverge for 4 s;
// the single free rotor of S2 diverged at 2.6 s because its reaction torque was unbalanced. The thrust at hover
// balances the weight (1.27 kg) within 3%, so the throttle constant is right, and the quad stays near its start (no
// tumble in 4 s).
TEST(devices, all_free_quad_runs_without_divergence) {
  auto s = load("all.xml");
  put_ok(*s, "/devices/gimbal_pan", "torque_enable", 1);
  for (int t = 0; t < 4000; ++t) {
    for (int k = 1; k <= 4; ++k) put_ok(*s, "/devices/esc" + std::to_string(k), "throttle", kHover);
    ASSERT_EQ(s->step(1), 1) << t;
    ASSERT_FALSE(s->diverged()) << "diverged at tick " << t;
    if (t % 500 == 499) {
      const mjData& d = ms(*s).data();
      std::printf("[all] t %.1f s: z %.3f vz %.3f m/s, body z axis . up %.4f, |w| %.3f rad/s\n", (t + 1) * 0.001,
                  d.qpos[2], d.qvel[2], 1.0 - 2 * (d.qpos[4] * d.qpos[4] + d.qpos[5] * d.qpos[5]),
                  std::sqrt(d.qvel[3] * d.qvel[3] + d.qvel[4] * d.qvel[4] + d.qvel[5] * d.qvel[5]));
    }
  }
  double thrust = 0;  // the attr is signed along the rotor's own axis (-z for rotors 2 and 4): sum magnitudes
  for (int k = 1; k <= 4; ++k) thrust += std::abs(get(*s, "/devices/prop" + std::to_string(k), "thrust"));
  double mass = 0;
  const mjModel& m = ms(*s).model();
  for (int b = 1; b < m.nbody; ++b) mass += m.body_mass[b];
  std::printf("[all] hover: thrust %.3f N vs weight %.3f N, z %.2f m, battery soc %.4f\n", thrust, mass * kG,
              ms(*s).data().qpos[2], get(*s, "/devices/pack", "soc"));
  EXPECT_LT(rel(thrust, mass * kG), 0.03);
  EXPECT_GT(ms(*s).data().qpos[2], 15.0);
  EXPECT_LT(ms(*s).data().qpos[2], 25.0);
  EXPECT_GT(std::abs(getv(*s, "/devices/imu", "accel")[2]), 5.0);  // the IMU reads ~ +g up while hovering
  EXPECT_GT(get(*s, "/devices/baro", "pressure"), 90000.0);
}

// (g) Save mid-run at tick 1503, which is not a sample / firmware tick of any device (IMU 500 Hz: period 2, baro
// 50 Hz: 20, mag 125 Hz: 8, GNSS 10 Hz: 100, servo firmware 100 Hz: 10), restore, continue == uninterrupted run,
// bitwise (hash and save bytes); a clone taken after the restore does the same.
TEST(devices, all_save_restore_clone_bitwise) {
  auto a = load("all.xml"), b = load("all.xml");
  run_all(*a, 0, 3000);
  run_all(*b, 0, 1503);
  const auto& dv = *ms(*b).devices();
  EXPECT_NE(dv.imu.st[0].ticks % 2, 0u);
  EXPECT_NE(dv.baro.st[0].ticks % 20, 0u);
  EXPECT_NE(dv.mag.st[0].ticks % 8, 0u);
  EXPECT_NE(dv.gnss.st[0].ticks % 100, 0u);
  EXPECT_NE(dv.servo.st[0].ticks % 10, 0u);
  const std::vector<std::byte> snap = b->save();
  const std::uint64_t h_mid = b->state_hash();
  run_all(*b, 1503, 2200);
  ASSERT_NE(b->state_hash(), h_mid);
  b->restore(snap);
  EXPECT_EQ(b->state_hash(), h_mid);
  auto c = b->clone();
  EXPECT_EQ(c->state_hash(), h_mid);
  run_all(*b, 1503, 3000);
  run_all(*c, 1503, 3000);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(a->state_hash(), c->state_hash());
  EXPECT_EQ(a->save(), b->save());
  EXPECT_EQ(a->save(), c->save());
  EXPECT_FALSE(a->diverged());
  // every device type really ran
  EXPECT_LT(get(*a, "/devices/pack", "soc"), 0.9);
  EXPECT_GT(get(*a, "/devices/esc1", "omega"), 100);
  EXPECT_GT(get(*a, "/devices/esc2", "omega"), 100);  // positive about its own axis (-z)
  EXPECT_GT(get(*a, "/devices/prop1", "thrust"), 0);
  EXPECT_LT(get(*a, "/devices/prop2", "thrust"), 0);  // signed along its axis: +z overall, as prop1
  EXPECT_NE(getv(*a, "/devices/air", "turbulence"), (std::vector<double>{0, 0, 0}));
  EXPECT_NE(getv(*a, "/devices/drag", "force"), (std::vector<double>{0, 0, 0}));
  EXPECT_NE(get(*a, "/devices/gimbal_pan", "position"), 0.0);
  EXPECT_NE(norm3(getv(*a, "/devices/imu", "gyro")), 0.0);
  EXPECT_GT(get(*a, "/devices/baro", "pressure"), 90000.0);
  EXPECT_GT(norm3(getv(*a, "/devices/mag", "field")), 1e-5);
  EXPECT_NE(get(*a, "/devices/gnss", "lat"), 47.3769);
}

// (g) Through the unchanged bus: SetAttr (throttles, servo goal and enable, wind), Step, Snapshot / Restore are
// logged; the replay (model rebuilt from the header bytes) reproduces the final hash. (all.xml has no PSU: its
// battery is the source.)
TEST(devices, all_bus_log_and_replay) {
  Session sess;
  sess.caps = {"read", "control"};
  sess.root = DVSIM_TEST_DATA_DIR;
  auto cmd = [](const std::string& type, const std::string& body) {
    return R"({"id":1,"source":"t","cmd_type":")" + type + R"(","cmd":)" + body + "}";
  };
  auto set_attr = [&](const std::string& dev, const char* attr, const std::string& values) {
    return cmd("SetAttr", R"({"path":"/devices/)" + dev + R"(","attr":")" + attr + R"(","values":[)" + values + "]}");
  };
  World w(load_mujoco, kDir + "all.xml");
  w.start_log();
  std::vector<std::string> cmds = {set_attr("gimbal_pan", "torque_enable", "1"),
                                   set_attr("gimbal_pan", "goal_position", "0.4"),
                                   set_attr("air", "wind", "3.5, 0.5, 0")};
  for (int k = 1; k <= 4; ++k) cmds.push_back(set_attr("esc" + std::to_string(k), "throttle", "0.46"));
  for (const std::string& c : {cmd("Step", R"({"n":503})"), cmd("Snapshot", "{}"), set_attr("esc1", "throttle", "0.5"),
                               set_attr("gimbal_pan", "goal_position", "-0.3"), set_attr("air", "wind", "0, 4, 0"),
                               cmd("Step", R"({"n":401})"), cmd("Restore", R"({"id":1})"),
                               set_attr("esc3", "throttle", "0.44"), cmd("Step", R"({"n":700})")})
    cmds.push_back(c);
  for (const std::string& c : cmds) {
    const std::string rep = execute_json(w, c, sess);
    ASSERT_NE(rep.find("\"ok\": true"), std::string::npos) << c << " -> " << rep;
  }
  const std::string q = execute_json(w, cmd("Query", R"({"path":"/devices/imu","attr":"accel"})"), sess);
  EXPECT_NE(q.find("\"unit\": \"m/s^2\""), std::string::npos) << q;
  World r = replay(w.log()->bytes(), load_mujoco_bytes, load_mujoco, DVSIM_TEST_DATA_DIR);
  EXPECT_EQ(r.tick(), w.tick());
  EXPECT_EQ(r.state_hash(), w.state_hash());
  EXPECT_NE(w.state_hash(), load("all.xml")->state_hash());
}

// ======== N4b: Reynolds factor of the rotor, speed-dependent motor loss ========
// (r1) re tables: interpolation inside the table, clamping outside, and the diameter scaling x = |n| D^2 (a prop of half
// the diameter at 4x the speed sits at the same x, so it gets the same factor). T = C_T f_T rho n^2 D^4, Q = C_Q f_Q rho n^2 D^5.
TEST(devices, rotor_reynolds_factor_table_and_diameter_scaling) {
  const double rho = isa(1.0), d_small = 0.127;
  struct Pt { double x, ft, fq; };
  // fixture: x = {1, 2, 4}, f_T = {0.9, 1.0, 1.2}, f_Q = {1.0, 1.0, 0.8}
  for (const Pt& p : {Pt{1.5, 0.95, 1.0}, Pt{3.0, 1.1, 0.9}, Pt{0.5, 0.9, 1.0}, Pt{10.0, 1.2, 0.8}}) {
    auto s = load("rotor_re.xml");
    const double n1 = p.x / (kD * kD), n2 = p.x / (d_small * d_small);  // same x on both rotors
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", 2 * kPi * n1);
    put_ok(*s, "/world/carrier/rotor2/shaft2", "qvel", 2 * kPi * n2);
    ASSERT_EQ(s->step(1), 1);
    EXPECT_LT(rel(get(*s, "/devices/prop", "thrust"), 0.1 * p.ft * rho * n1 * n1 * std::pow(kD, 4)), 1e-9) << p.x;
    EXPECT_LT(rel(get(*s, "/devices/prop", "torque"), 0.006 * p.fq * rho * n1 * n1 * std::pow(kD, 5)), 1e-9) << p.x;
    EXPECT_LT(rel(get(*s, "/devices/small", "thrust"), 0.1 * p.ft * rho * n2 * n2 * std::pow(d_small, 4)), 1e-9) << p.x;
    EXPECT_LT(rel(get(*s, "/devices/small", "torque"), 0.006 * p.fq * rho * n2 * n2 * std::pow(d_small, 5)), 1e-9) << p.x;
  }
}

// (r2) off = unchanged: factors identically 1 reproduce the table-free rotor bitwise (thrust, torque, state hash), and a
// rotor without re tables (every older fixture) never enters the factor path.
TEST(devices, rotor_reynolds_unity_matches_off_bitwise) {
  auto a = load("rotor_free.xml"), b = load("rotor_re_unity.xml");
  for (auto* s : {a.get(), b.get()}) {
    put_ok(*s, "/world/carrier/rotor/shaft", "qvel", 431.7);
    put_ok(*s, "/world/carrier/lift", "qvel", 2.5);
    ASSERT_EQ(s->step(1), 1);  // the free light rotor diverges within a few ticks at this speed: one tick is enough
  }
  for (const char* attr : {"thrust", "torque", "omega", "advance_ratio"})
    EXPECT_EQ(get(*a, "/devices/prop", attr), get(*b, "/devices/prop", attr)) << attr;
  EXPECT_EQ(a->state_hash(), b->state_hash());
}

// (m1) pure viscous loss (i0_visc = 1, w0 = 300 rad/s): steady no-load speed from Ke i_m = Ke I0 w / w0 with
// i_m = (V - Ke w) / R(T), i.e. w = V / (Ke + R I0 / w0), V and T read from the device.
TEST(devices, motor_loss_viscous_no_load_speed) {
  auto s = load("motor_loss_visc.xml");
  put_ok(*s, "/devices/m1", "throttle", 1.0);
  ASSERT_EQ(s->step(3000), 3000);
  const double v = get(*s, "/devices/m1", "voltage"), t = get(*s, "/devices/m1", "temperature");
  const double want = v / (ke() + r_at(t) * kI0 / 300.0);
  const double coulomb = (kKv * 2 * kPi / 60) * (v - kI0 * r_at(t));  // what the constant-I0 law would give
  EXPECT_LT(rel(get(*s, "/devices/m1", "omega"), want), 0.005) << want;
  EXPECT_GT(rel(coulomb, want), 0.005);  // the law is really different from the constant-friction one
  EXPECT_FALSE(s->diverged());
}

// (m2) mixed loss (30 % tanh, 50 % w, 20 % w|w| of Kt I0 at w0): torque balance (V - Ke w) / R = I0 [..], bisection.
TEST(devices, motor_loss_mixed_no_load_speed) {
  auto s = load("motor_loss_mix.xml");
  put_ok(*s, "/devices/m1", "throttle", 1.0);
  ASSERT_EQ(s->step(3000), 3000);
  const double v = get(*s, "/devices/m1", "voltage"), r = r_at(get(*s, "/devices/m1", "temperature"));
  auto f = [&](double w) {
    return (v - ke() * w) / r - kI0 * (0.3 * std::tanh(w / 5.0) + 0.5 * w / 900.0 + 0.2 * w * w / (900.0 * 900.0));
  };
  double lo = 0, hi = v / ke();
  for (int i = 0; i < 100; ++i) (f(0.5 * (lo + hi)) > 0 ? lo : hi) = 0.5 * (lo + hi);
  EXPECT_LT(rel(get(*s, "/devices/m1", "omega"), 0.5 * (lo + hi)), 0.005);
  EXPECT_FALSE(s->diverged());
}

TEST(devices, motor_loss_needs_reference_speed) {
  try {
    load("bad_loss.xml");
    ADD_FAILURE() << "loaded";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("m_bad_loss"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("i0_w"), std::string::npos) << e.what();
  }
}

// ======== ESC losses (r_on, sw_loss, i_q): analytic checks, written from the definition ========
namespace {
constexpr double kRon = 0.02, kSw = 0.01, kIq = 0.05;  // motor_esc_loss.xml
}

// (e1) At zero throttle only the quiescent current flows: bus = PSU = i_q, no phase current.
// (e2) Stalled (w = 0) at d = 0.05: I_m = d V / (R(T) + r_on); I_bus = (d + sw) I_m + i_q; PSU current = I_bus (KCL), and the
// input power splits exactly into winding + FET conduction loss, switching loss and quiescent loss.
TEST(devices, esc_loss_idle_and_stall_balance) {
  auto s = load("motor_esc_loss.xml");
  ASSERT_EQ(s->step(1), 1);
  EXPECT_DOUBLE_EQ(get(*s, "/devices/m1", "bus_current"), kIq);
  EXPECT_LT(rel(get(*s, "/devices/bench", "current"), kIq), 1e-9);
  EXPECT_EQ(get(*s, "/devices/m1", "current"), 0.0);

  const double d = 0.05;
  put_ok(*s, "/devices/m1", "throttle", d);
  ASSERT_EQ(s->step(1), 1);
  ASSERT_EQ(get(*s, "/devices/m1", "omega"), 0.0);
  const double v = get(*s, "/devices/m1", "voltage"), im = get(*s, "/devices/m1", "current"), ib = get(*s, "/devices/m1", "bus_current");
  EXPECT_LT(rel(im, d * v / (r_at(kTamb) + kRon)), 1e-12);
  EXPECT_LT(rel(ib, (d + kSw) * im + kIq), 1e-12);
  EXPECT_LT(rel(get(*s, "/devices/bench", "current"), ib), 1e-9);  // PSU current = ESC bus current
  const double p_in = v * ib, p_cu = im * im * r_at(kTamb), p_fet = im * im * kRon, p_sw = kSw * v * im, p_q = kIq * v;
  EXPECT_LT(rel(p_in, p_cu + p_fet + p_sw + p_q), 1e-9) << p_in << " vs " << p_cu + p_fet + p_sw + p_q;
  EXPECT_LT(rel(get(*s, "/devices/m1", "temperature"), kTamb + 0.0), 1e-3);  // r_on heats the ESC, not the winding
}

// (e3) Current limit with losses: the phase current is the limit (10 A), duty = I_lim (R + r_on) / V at w = 0, bus current
// (duty + sw) I_lim + i_q.
TEST(devices, esc_loss_current_limit) {
  auto s = load("motor_esc_loss.xml");
  put_ok(*s, "/devices/m1", "throttle", 1.0);
  ASSERT_EQ(s->step(1), 1);
  const double v = get(*s, "/devices/m1", "voltage"), duty = get(*s, "/devices/m1", "duty");
  EXPECT_EQ(ms(*s).devices()->esc.st[0].mode, 2.0);
  EXPECT_DOUBLE_EQ(get(*s, "/devices/m1", "current"), 10.0);
  EXPECT_LT(rel(duty, 10.0 * (r_at(kTamb) + kRon) / v), 1e-12);
  EXPECT_LT(rel(get(*s, "/devices/m1", "bus_current"), (duty + kSw) * 10.0 + kIq), 1e-12);
  EXPECT_LT(rel(get(*s, "/devices/bench", "current"), get(*s, "/devices/m1", "bus_current")), 1e-9);
}

// (e4) Spinning: the steady state satisfies d V = Ke w + I_m (R + r_on) with I_m Ke = Kt I0 (Coulomb friction, no load), i.e.
// w = (d V - I0 (R + r_on)) / Ke; the loss changes the speed against the ideal stage.
TEST(devices, esc_loss_no_load_speed) {
  auto s = load("motor_esc_loss.xml");
  put_ok(*s, "/devices/m1", "throttle", 0.5);
  ASSERT_EQ(s->step(3000), 3000);
  const double v = get(*s, "/devices/m1", "voltage"), t = get(*s, "/devices/m1", "temperature");
  const double want = (0.5 * v - kI0 * (r_at(t) + kRon)) / ke();
  EXPECT_LT(rel(get(*s, "/devices/m1", "omega"), want), 0.005);
  EXPECT_NEAR(get(*s, "/devices/m1", "current"), kI0, 0.01 * kI0);
  EXPECT_FALSE(s->diverged());
}

// (e5) Zero loss fields are the ideal stage, bit for bit (same throttle schedule as the determinism tests).
TEST(devices, esc_loss_zero_is_ideal) {
  auto a = load("motor.xml"), b = load("motor_esc_zero.xml");
  run(*a, 0, 400);
  run(*b, 0, 400);
  EXPECT_EQ(a->state_hash(), b->state_hash());
  EXPECT_EQ(get(*a, "/devices/m1", "bus_current"), get(*b, "/devices/m1", "bus_current"));
}
