#pragma once
// Devices (milestone N2): deterministic C++ physics models stepped inside MujocoSolver::step, exposed as prims
// /devices/<name>. Data-oriented: per device type one vector of plain config structs and one of POD state structs, free
// functions over them, no class hierarchy. Config: schema/devices.fbs, carried in the MJCF as <custom><text
// name="dvsim.devices" data='{json}'/>. A new device type is a Group plus one line in each of: for_groups(), the slot
// table, parse(), bind(), step()'s phase list, the attribute tables in devices.cpp.
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/solver.h"

namespace devs {

constexpr int kMaxNodes = 128;  // non-ground electrical nodes (ground = node 0, implicit); 32 per vehicle, several vehicles per scene
constexpr double kPi = 3.14159265358979323846;

// xoshiro256** seeded through splitmix64; one stream per device that needs noise. gauss() is our own Box-Muller:
// std::normal_distribution is implementation-defined, which would break bitwise replay across toolchains.
struct Rng {
  std::uint64_t s[4];
  void seed(std::uint64_t x);
  std::uint64_t next();
  double uniform();  // [0, 1), 53 bits
  double gauss();    // N(0,1); two uniforms per sample, no cached spare (so the state is exactly `s`)
};

// ---- config (immutable after parse; resolved MuJoCo ids are filled by bind) ----
struct Meta {
  std::string name;
  std::vector<std::string> parts;  // part ids: provenance for the fidelity report, never used by the physics
};
struct MotorCfg {  // see schema; shared by Esc and Servo
  double kv, r20, i0, inertia, w_s, c_th, r_th, t_amb, alpha;
  double i0_w = 0, i0_visc = 0, i0_quad = 0;  // speed-dependent no-load loss (schema MotorParams); 0 = constant Coulomb
  double ke() const { return 60.0 / (2.0 * kPi * kv); }  // = Kt in SI
  // Motor-shaft loss torque at speed w: Kt i0 [(1 - v - q) tanh(w / w_s) + v w / w0 + q w |w| / w0^2].
  double loss(double w) const {
    double x = (1.0 - i0_visc - i0_quad) * std::tanh(w / w_s);
    if (i0_w > 0) x += i0_visc * w / i0_w + i0_quad * w * std::abs(w) / (i0_w * i0_w);
    return ke() * i0 * x;
  }
  double r_at(double temp_c) const { return r20 * (1.0 + alpha * (temp_c - 20.0)); }
};
struct BatteryCfg {
  Meta meta; int node; std::vector<double> ocv_soc, ocv_v; double capacity_ah, r0, r1, c1, soc0; std::uint64_t seed;
};
struct PsuCfg { Meta meta; int node; double voltage, r_series, i_limit; bool enabled0; };
struct WireCfg { Meta meta; int a, b; double r; };
struct LoadCfg { Meta meta; int node; double g, i0; };
struct EscCfg {
  Meta meta; int node; std::string joint; MotorCfg motor; bool complementary; double i_limit; std::uint64_t seed;
  double r_on = 0, sw_loss = 0, i_q = 0;  // ESC losses (schema Esc); 0 = ideal
  bool bidirectional = false;             // throttle in [-1, 1] (schema Esc.bidirectional)
  int jnt = -1, dof = -1;  // resolved by bind()
};
struct ServoCfg {
  Meta meta; int node; std::string joint; MotorCfg motor; double n, eta, rate_hz, p, i, d, pid_scale, deadband, max_torque;
  double d_scale = 0;  // 0 = D uses pid_scale
  double i_limit; int cpr; std::string backlash_joint;
  double min_voltage, max_voltage, max_temp, overload_load, overload_time; int unload_mask;  // protection, initial values
  int jnt = -1, dof = -1, qadr = -1, period = 0;  // resolved by bind(): firmware period in ticks
  // The joint's own limit fields as compiled, put back when a jam clears (a jam is a zero-width limit, see step()).
  int lim0 = 0; double range0[2] = {0, 0}, solref0[2] = {0.02, 1}, solimp0[5] = {0.9, 0.95, 0.001, 0.5, 2};
};
struct RotorCfg {
  Meta meta; std::string body, joint; double diameter; int dir; std::vector<double> j, ct, cq; double k_h;
  bool ground_effect; unsigned ray_groups;
  bool water = false;                      // marine propeller (schema Rotor.water)
  double rev_thrust = 1.0;                 // thrust factor in reverse spin (schema Rotor.rev_thrust)
  std::vector<double> re_x, re_ft, re_fq;  // optional Reynolds factors over x = |n| D^2 (empty = off)
  std::string stator_body; double swirl_recovery = 0;  // EDF stator: share of the fan's torque handed to stator_body (schema Rotor)
  double rain_kt = 0.10, rain_kq = 0.15;               // rain: thrust loss / torque rise per 23 g/m^3 of LWC (schema Rotor)
  int bid = -1, jnt = -1, dof = -1, stator_bid = -1;  // resolved by bind()
};
struct EnvCfg {
  Meta meta; double origin_alt, wind0[3], w20; std::uint64_t seed; std::string ref_body; double ref_height;
  int bid = -1;  // ref_body, resolved by bind() (-1: fixed ref_height)
};
struct DragCfg { Meta meta; std::string body; double cd, area; int bid = -1; };
// Sensors (S4): each on a site; sid / period (ticks per sample) are resolved by bind().
struct InertialErrCfg { double bias[3], scale[3], noise_density, rw_density, full_scale; int bits; };
struct ImuCfg {
  Meta meta; std::string site; double rate_hz; InertialErrCfg gyro, accel; std::uint64_t seed;
  int sid = -1, period = 0;
};
struct BaroCfg {
  Meta meta; std::string site; double rate_hz, bias, noise_std, resolution; std::uint64_t seed;
  int sid = -1, period = 0;
};
struct MagCfg {
  Meta meta; std::string site; double rate_hz, field[3], bias[3], noise_std; std::uint64_t seed;
  int sid = -1, period = 0;
};
constexpr int kGnssRing = 64;  // samples held for the latency: delay <= kGnssRing - 1
struct GnssCfg {
  Meta meta; std::string site; double rate_hz, lat0, lon0, alt0, horiz_std, vert_std, vel_std, latency_s;
  std::uint64_t seed; int sid = -1, period = 0, delay = 0;  // delay (samples) = round(latency_s rate_hz), set by parse
  double corr_time_s = 0, corr_frac = 0;                    // Gauss-Markov position error (0 = white), see schema Gnss
};

constexpr int kMaxPellets = 16;  // launcher pool
constexpr int kHitRing = 16;     // hits kept (newest overwrite oldest); hit k lives at k % kHitRing
struct LauncherCfg {
  Meta meta; std::string site; double v0, mass, diameter, cd, max_time, v_std, spread_std; unsigned ray_groups;
  std::vector<std::string> visual; std::uint64_t seed;
  int sid = -1, body = -1;     // resolved by bind(): muzzle site, its body (excluded from the hit test)
  std::vector<int> mocap;      // mocap index of each visual body
  double cl_k = 0, spin_tau = 0;  // Magnus slope / spin decay (cuas2); either > 0 gives the launcher a LauncherXState
  int lx = -1;                 // index into Devices::lx, -1 = none (every cuas-lite launcher: state layout and hashes unchanged)
};
struct CameraCfg { Meta meta; std::string site; double fov_v, rate_hz; int width, height; int sid = -1; };

// Phase 5 devices (rocket.cpp, vessel.cpp, rangefinder.cpp).
struct SolidMotorCfg {
  Meta meta; std::string body, site; std::vector<double> t, f, cum;  // curve; cum[k] = integral of the curve up to t[k]
  double prop_mass, total_mass, diameter, length, delay, scale0;
  double i_total = 0;  // cum.back()
  int bid = -1, sid = -1;
};
struct AirframeCfg {
  Meta meta; std::string body, site; double s_ref, cd0, cd_rev, cn_alpha, x_cp, s_plan, cd_cross;
  int bid = -1, sid = -1;
};
struct JetVaneCfg {
  Meta meta; std::string body, site, rotor; double area, exit_area, cn_alpha, cn_max, cd0, jet_factor;
  int bid = -1, sid = -1, rot = -1;  // resolved by bind(): plate body, aero-centre site, index of the rotor device
};
struct WaveCfg { double a, k, w, th, phi; };  // amplitude m, wave number rad/m, angular frequency rad/s, direction rad, phase rad
struct SeaCfg {
  Meta meta; double rho, level, depth, current0[3]; std::vector<WaveCfg> waves;
};
struct Cell { double c[3], v, e[3]; };  // centre (body frame), volume, extents (body axes)
struct FinCfg { double p[3], area, span, cla, cd; };  // skeg / keel: body-frame centre, side area, depth, lift slope /rad, cross-flow Cd
struct VesselCfg {
  Meta meta; std::string body; std::vector<Cell> cells; double volume;  // total hull volume
  std::vector<FinCfg> fins;
  double a[6], dl[6], dq[6];                                            // added mass, linear / quadratic damping
  bool has_a = false, heading_frame = false;  // heading_frame: surge / sway / heave damping and added mass in the yaw-only frame (schema Vessel)
  int bid = -1;
};
struct RainCfg {
  Meta meta; double rate0; std::string body; double catch_area, wet_area, film_depth, film_tau, collect_eff;
  int bid = -1;  // body, resolved by bind() (-1: no film)
};
struct RangefinderCfg {
  Meta meta; std::string site; double rate_hz, fov_h, fov_v, min_r, max_r, s0, s1, s2, res, dropout; std::string proj;
  int n_h, n_v; unsigned ray_groups; std::uint64_t seed;
  bool pinhole = true;
  bool rosette = false;                                     // projection "rosette": rays generated per frame (rangefinder.cpp), dir is empty
  double fov_v_min = 0, fov_v_max = 0, off1 = 0, off2 = 0;  // rosette: elevation limits (rad), R2 sequence offsets in [0, 1) from pattern_seed
  double rain_a = 1.82e-4, rain_b = 0.63, rain_margin_db = 6.0, rain_beam = 0.02, rain_zone = 1.5, rain_det = 0.01, rain_window_db = 1.5;  // schema Rangefinder
  std::vector<double> dir;   // n_v * n_h unit rays in the site frame (row-major), 3 doubles each
  std::vector<double> scale;  // range -> output factor per pixel (pinhole: 1 / |unnormalised ray| = z-depth per metre; spherical: 1)
  double cutoff = 0;  // Euclidean ray length that can still give an output <= max_r
  int sid = -1, body = -1, period = 0;
};
// SP2 devices (qdd.cpp, contact.cpp).
constexpr int kQddRing = 16;  // sampled commands kept for the CAN latency: latency_s / period <= kQddRing - 2
struct QddCfg {
  Meta meta; int node; std::string joint; MotorCfg motor;
  double n, eta, cmd_hz, latency_s, i_limit, kp_max, kd_max, tau_ff_max, timeout_s, kd_timeout, t_max, derate_start;
  int cpr, timeout_mode;
  int jnt = -1, dof = -1, qadr = -1, period = 0, delay = 0;  // resolved by bind(): command period and latency, in ticks
};
struct FootContactCfg {
  Meta meta; std::string geom; int mode; double f_max, noise_std, hys_on, hys_off, debounce_s, rate_hz; std::uint64_t seed;
  int gid = -1, period = 0;  // resolved by bind()
};
// Terrain-aware wind (schema WindField, windfield.cpp). The hfield grids are resolved by bind(): heights in metres (world z), row r (y) x
// column c (x) at index r * ncol + c; h = the surface, hl / hb = h smoothed (Gaussian, sigma ridge_len / 4 ridge_len), not saved (derived).
struct WindFieldCfg {
  Meta meta; std::string hfield_geom; double ground_z, z0, z_ref, h_max, ridge_gain, ridge_len, lee_turb, lee_down;
  int nrow = 0, ncol = 0; double x_lo = 0, y_lo = 0, dx = 1, dy = 1;  // grid origin (corner) and spacing, m; nrow = 0: flat ground
  std::vector<float> h, hl, hb;
};

// ---- state: every mutable quantity, doubles / uint64 only (no padding), saved and hashed byte-for-byte ----
// "Outputs" (last solved values) live here too: they are cheap, deterministic, and make get() a plain field read.
// v, i, ocv: of the last solve (i > 0 = discharging). Fault inputs (F3, 0 = healthy): r_fault = extra series resistance (ohm,
// one weak cell: its R step adds to r0), ocv_offset = EMF added to the pack (V, negative: a dead / shorted cell).
struct BatteryState { double soc, v_rc, v, i, ocv; Rng rng; double r_fault, ocv_offset; };
struct PsuState { double enable, v, i, limited; };              // enable: input (0/1); limited: 1 while current-limited
struct WireState { double i, v; };                              // a -> b current, voltage drop a - b
struct LoadState { double v, i; };
struct EscState {
  double throttle;  // input [0,1]
  double temp;      // winding temperature, degC
  double omega;     // shaft speed used by the last solve, rad/s
  double v, duty, i_m, i_bus, torque, b, drive, mode;  // mode: 0 normal, 1 phase open, 2 current-limited
  Rng rng;
  double fault_open;  // fault input (F3): 1 = output stage / winding open circuit (like zero throttle: no drive, coasts)
};

struct ServoState {
  double goal_position, goal_speed, torque_enable;  // inputs (rad, rad/s, 0/1)
  double setpoint, integ, e_prev, cmd;               // firmware: ramped goal (rad), sum(e), last e (counts), held duty
  double counts, position;                           // encoder reading: counts, quantised angle (rad); every tick
  double temp, omega;                                // winding degC; motor shaft speed (N x output) of the last solve
  double v, duty, i_m, i_bus, torque, b, drive;      // as EscState; duty: effective (after current limit)
  double backlash;                                   // half-range of the backlash joint, rad (0: none); set by bind()
  // Runtime settable (initial values from the config): firmware gains (register units), deadband (counts), duty limits
  // (0..1000, effective limit min(max_torque, torque_limit) / 1000), protection thresholds (0 = off, see servo.cpp).
  double p, i, d, deadband, max_torque, torque_limit;
  double min_voltage, max_voltage, max_temp, overload_load, overload_time, unload_mask;
  double alarm, ovl_t;                               // latched alarm bits (servo.cpp); seconds |duty| has been above overload_load
  double velocity;                                   // output shaft speed of the last tick start, rad/s
  // Fault inputs (F3, 0 = healthy): ext_torque = external torque on the output joint, N m (a load the servo must fight);
  // jam = 1 seizes the joint at the angle it had when set (stiff spring + damper in servo stall); jammed / q_lock: latch.
  double ext_torque, jam, jammed, q_lock;
  // Motor-controller firmware (A4, RoboClaw-style, see servo.cpp): mode 0 = the position firmware above (default, unchanged),
  // 1 = open-loop duty, 2 = velocity PID on the encoder. All inputs settable, all 0 at power-up.
  double mode, goal_duty, duty_accel;                // duty_accel: duty change per second (0 = immediate)
  double goal_velocity, vel_accel, qpps;             // counts/s, counts/s^2 (0 = immediate), counts/s at 100 % duty
  double vel_p, vel_i, vel_d;                        // dimensionless, 1/s, s
  double vsp, vmeas, vraw, vint, ve_prev, counts_prev;  // ramped setpoint, filtered / raw speed (counts/s), sum(e) Ts, last e, last counts
  std::uint64_t ticks;                              // ticks stepped; firmware runs when ticks % period == 0
};
// Phase open (no drive): torque off, or released by a latched alarm that is in the unload mask.
inline bool servo_open(const ServoState& s) {
  return !(s.torque_enable > 0.5) || (static_cast<unsigned>(s.alarm) & static_cast<unsigned>(s.unload_mask)) != 0;
}
// Outputs of the last tick, plus the fault input damage (F3, 0 = healthy): fraction of thrust and torque lost (a chipped prop).
struct RotorState { double omega, thrust, torque, advance_ratio, ge_ratio, rho; double damage; };
struct EnvState {
  double wind[3];  // steady wind, world (input)
  double turb[3];  // Dryden turbulence, world, of the last tick
  double density;  // ISA density at world z = 0
  double xa[3], xb[3];  // Dryden filter states (unit-variance-scaled): u: xa[0]; v, w: first / second stage in [1], [2]
  Rng rng;
};
struct DragState { double force[3]; };
// Sensor outputs are the held sample (see sensors.cpp for the timing); ticks counts ticks, a sample is taken when
// ticks % period == 0 (the first tick samples).
struct ImuState { double gyro[3], accel[3], gyro_bias[3], accel_bias[3]; Rng rng; std::uint64_t ticks; };
// bias_extra (Pa) grows by drift_rate (Pa/s, fault input) per second of samples and stays when the drift stops.
struct BaroState { double pressure; Rng rng; std::uint64_t ticks; double bias_extra, drift_rate; };
struct MagState { double field[3]; Rng rng; std::uint64_t ticks; };
struct GnssState {
  double lat, lon, alt, vel_ned[3];  // output: deg, deg, m, m/s (the sample taken `delay` samples ago)
  double ring[kGnssRing][6];         // lat, lon, alt, vel_ned of the last samples, sample k at ring[k % kGnssRing]
  double gm[3];                      // Gauss-Markov position error north, east, up, m (corr_time_s > 0 only)
  Rng rng;
  std::uint64_t ticks, samples;      // samples: taken so far
  // Fault inputs (F3, 0 = healthy): fix_lost = 1: no fix, the output holds its last value (consumers read fix_lost);
  // jump = position step added to the samples taken from now on, ENU m (east, north, up).
  double fix_lost, jump[3];
};

// Launcher (schema Launcher). Pool arrays are flat (3 doubles per pellet); active: 0 free, 1 in flight. Hit k (k = hit_n - 1 is
// the newest) is in ring slot k % kHitRing: body id (0 = ground / static world), geom id, point, sim time, impact speed, pellet round no.
struct LauncherState {
  double fire;                                  // input 0 / 1; consumed by the next tick
  double rounds, hits, ground, expired, dropped;  // counters: fired, hit a non-world body, hit the world, timed out, refused (pool full)
  double hit_n;                                 // hits + ground contacts recorded so far (ring write count)
  double time;                                  // s stepped by this device (ticks dt)
  double muzzle_pos[3], muzzle_dir[3];          // output: world muzzle point and barrel axis (unit)
  double active[kMaxPellets], round_no[kMaxPellets], age[kMaxPellets];
  double pos[3 * kMaxPellets], vel[3 * kMaxPellets];
  double hit_body[kHitRing], hit_geom[kHitRing], hit_t[kHitRing], hit_v[kHitRing], hit_round[kHitRing];
  double hit_pos[3 * kHitRing];
  Rng rng;
};
// Launcher per-shot inputs (cuas2), a separate group so launchers without cl_k / spin_tau keep their state layout. muzzle_velocity:
// m/s of the next shots (0 = the configured v0); spin_rps: signed spin about the lateral axis site_x cross site_z (negative =
// backspin when site x is up); dir_offset_x / y: rad, deterministic aim error along the site x / y axes. They persist until rewritten.
// omega0: the spin vector (rad/s, world) each pool pellet was launched with.
struct LauncherXCfg { int li; };
struct LauncherXState {
  double muzzle_velocity, spin_rps, dir_offset_x, dir_offset_y;
  double omega0[3 * kMaxPellets];
};
// Camera pose (start of the last tick): world position, optical axis, image-up.
struct CameraState { double pos[3], fwd[3], up[3]; };

// Solid motor (schema SolidMotor). phase: 0 idle, 1 ignition delay, 2 burning, 3 burnt out. t_ign: s since the ignite command was
// taken (0 on its first tick); t_burn: s of the curve done; progress: impulse fraction of the curve delivered, in [0, 1]; mass = loaded
// motor mass now. pos / dir: nozzle point and thrust axis (world, start of the last tick).
struct SolidMotorState {
  double ignite, thrust_scale;  // inputs
  double phase, t_ign, t_burn, impulse, progress, burned, mass, thrust;
  double pos[3], dir[3];
};
// Airframe outputs of the last tick: the aerodynamic force (world, N), the angle of attack (rad, 0 without airflow) and the CP air speed.
struct AirframeState { double force[3], alpha, speed; };
// Jet vane outputs of the last tick: the aerodynamic force (world, N), the angle of attack (rad) and the jet speed at the vane (m/s).
struct JetVaneState { double force[3], alpha, speed; };
// Sea: its own clock (s), the wave amplitude multiplier and the current (inputs; the vertical current is ignored).
// The wave table is a copy of the (constant) config, [amplitude, k, w, direction, phase] per component, so that clients can read the
// exact surface function (dvsim.hydro re-implements it) and it is part of the saved state.
constexpr int kMaxWaves = 64;
struct SeaState {
  double time, wave_scale, current[3], eta0;  // eta0: surface elevation at the world origin (output)
  double n_waves, waves[5 * kMaxWaves];
};
// Vessel outputs of the last tick: buoyancy force (world), submerged volume, wetted fraction, surface elevation at the body origin,
// hydrodynamic damping + added-mass force (world).
struct VesselState { double buoyancy[3], submerged, wetted, eta, hydro[3]; };
// Range sensor: the image itself lives in Devices::range_img (derived output, not saved; a restored run holds the old image until its
// next sample). frames: samples taken; valid: returns in the last one; t: sim time of the last sample.
struct RangefinderState { Rng rng; std::uint64_t ticks; double frames, valid, t; };
// Wind field: its own clock t (s). Input `gust_in` = [amplitude m/s (signed), H m (gradient distance, > 0), direction xyz (any length,
// normalised here), delay s from now, speed m/s]: consumed by the next tick into the first free of kMaxGusts slots (all-zero / H <= 0 = no
// gust; the oldest slot is overwritten when full), `gust_clear` = 1 drops every gust. A slot is [amp, H, ux, uy, uz, t0, c, nx, ny]
// (n = unit horizontal propagation direction). Outputs of the last tick at the reference body (Environment.ref_body, else the origin
// at ref_height): wind (steady + ridge + gust + turbulence), height above ground, ground height, lee factor.
constexpr int kMaxGusts = 8;
struct WindState {
  double t, gust_in[7], gust_clear, n_gusts;  // n_gusts: scheduled so far (slot = n_gusts % kMaxGusts)
  double gust[kMaxGusts][9];
  double wind_ref[3], hag, ground, lee;
};

// Rain (rain.cpp). rate: input (mm/h). Outputs of the last tick: lwc (g/m^3), v_fall (m/s), film_mass (kg; also the film's state),
// force (N, world: the catch force and the film's weight on the body), thrust_loss (mean fractional thrust loss of the air rotors).
struct RainState { double rate, lwc, v_fall, film_mass, force[3], thrust_loss; };

// Qdd (qdd.cpp). Inputs p_des .. enable (settable; every set stamps rx_tick = the tick the frame is received at, for the
// timeout). ring: the sampled commands [p_des, v_des, kp, kd, tau_ff, enable, rx_tick], sample j at ring[j % kQddRing].
// Outputs of the last tick: q, dq (encoder), tau_est (electromagnetic output torque Kt N eta i_q), i_q (phase current),
// temp (winding degC), fault (latched), bus_i (A into the load, + = drawn from the node; injected into the NEXT solve),
// vbus (node voltage of this tick), derate (current-limit factor), timeout (1 while the CAN timeout is active), tau_cmd.
struct QddState {
  double p_des, v_des, kp, kd, tau_ff, enable;
  double rx_tick;
  double q, dq, tau_est, i_q, temp, fault, bus_i, vbus, derate, timeout, tau_cmd;
  double ring[kQddRing][7];
  std::uint64_t ticks;
};
// Foot contact outputs (held between samples). pend: seconds the comparator has differed from `contact` (-1 = it agrees).
struct FootContactState { double force, contact, pend; Rng rng; std::uint64_t ticks; };

template <class C, class S> struct Group {
  using Cfg = C;
  using State = S;
  std::vector<C> cfg;
  std::vector<S> st;
};

struct Devices {
  Group<BatteryCfg, BatteryState> battery;
  Group<PsuCfg, PsuState> psu;
  Group<WireCfg, WireState> wire;
  Group<LoadCfg, LoadState> load;
  Group<EscCfg, EscState> esc;
  Group<RotorCfg, RotorState> rotor;
  Group<EnvCfg, EnvState> env;  // 0 or 1 element
  Group<DragCfg, DragState> drag;
  Group<ServoCfg, ServoState> servo;
  Group<ImuCfg, ImuState> imu;
  Group<BaroCfg, BaroState> baro;
  Group<MagCfg, MagState> mag;
  Group<GnssCfg, GnssState> gnss;
  Group<LauncherCfg, LauncherState> launcher;
  Group<CameraCfg, CameraState> camera;
  Group<SolidMotorCfg, SolidMotorState> motor;
  Group<AirframeCfg, AirframeState> airframe;
  Group<SeaCfg, SeaState> sea;  // 0 or 1 element
  Group<VesselCfg, VesselState> vessel;
  Group<RangefinderCfg, RangefinderState> range;
  Group<JetVaneCfg, JetVaneState> vane;
  Group<WindFieldCfg, WindState> wfield;  // 0 or 1 element; appended after the older groups: their state layouts and hashes are unchanged
  Group<RainCfg, RainState> rain;  // 0 or 1 element; models without rain keep their state layout and hashes
  Group<QddCfg, QddState> qdd;     // SP2 groups come last: models without them keep their state layout and hashes
  Group<FootContactCfg, FootContactState> foot;
  Group<LauncherXCfg, LauncherXState> lx;  // last: empty unless a launcher has cl_k / spin_tau
  int n_nodes = 0;  // highest node id used
  struct Slot {
    enum Type { battery, psu, wire, load, esc, rotor, env, drag, servo, imu, baro, mag, gnss, launcher, camera, motor, airframe, sea,
                vessel, range, vane, wfield, rain, qdd, foot } type;
    int i;
  };
  std::vector<Slot> slots;  // prim order: type by type, config order within a type
  const std::string& name(int slot) const;
  // Scratch of solve_electrical (rebuilt from scratch every tick, never saved). drv_*: ESCs first, then servos.
  std::vector<int> drv_mode;
  std::vector<double> drv_ic;
  std::vector<int> psu_lim;
  std::vector<double> qvel_k, qvel_k1;  // scratch of the IMU acceleration (sensors.cpp): qvel before / after mj_step2
  std::vector<std::vector<double>> range_img;  // the held image of each range sensor (rangefinder.cpp), n_v * n_h, not part of the state
};

std::optional<std::string> config_text(const mjModel* m);  // the "dvsim.devices" custom text, if any
Devices parse(const std::string& json);  // throws std::runtime_error naming the device
// Resolves joint names to ids and applies the model-side effects (armature SET, damping owned). Throws on a missing
// joint, a non-hinge joint, or RK4. Idempotent: safe on a model reloaded from saved bytes and after Spawn / Delete.
void bind(mjModel* m, Devices& dv);
// One tick (replaces mj_step): step1 -> device phases -> damping split -> step2.
void step(mjModel* m, mjData* d, Devices& dv);

// Electrical network + ESC phase decisions for the current state (electrical.cpp). dt advances battery RC / SoC.
void solve_electrical(Devices& dv, double dt);

// Environment + aero (aero.cpp). ISA density at world height z (origin altitude 0 without an environment device);
// air velocity at the scene (steady + turbulence, world; zero without an environment device).
double rho_at(const Devices& dv, double z);
double pressure_at(const Devices& dv, double z);  // ISA static pressure, Pa, same altitude convention
void wind_at(const Devices& dv, double out[3]);
// Wind field (windfield.cpp). Wind at a world point: with a WindField device the full field (boundary layer, ridge flow, gusts, turbulence
// scaled by the lee zone), else exactly wind_at. ground_height: the hfield surface (or the flat ground_z) under (x, y); NaN without a
// WindField. step_wind_field: gust intake + reference outputs, before the rotors; advance_wind_clock: after the force phase.
void wind_at_point(const Devices& dv, const double pos[3], double out[3]);
double ground_height(const Devices& dv, double x, double y);
void bind_wind_field(const mjModel* m, WindFieldCfg& c);  // resolves the hfield grid (throws naming the device)
void step_wind_field(const mjModel* m, const mjData* d, Devices& dv);
void advance_wind_clock(Devices& dv, double dt);
// Phase 2 (servo.cpp): encoder reading of every servo, and the firmware position loop on its own period.
void step_firmware(const mjModel* m, const mjData* d, Devices& dv);
// Phase 5: advances the environment (turbulence), then applies rotor and body-drag forces to d->xfrc_applied.
void step_forces(const mjModel* m, mjData* d, Devices& dv);
// Post-step hook (sensors.cpp): samples the sensors that are due, from the derived data of the tick just stepped.
void sample_sensors(const mjModel* m, mjData* d, Devices& dv);
// Launcher + camera (launcher.cpp), between the forces and mj_step2, from the kinematics of mj_step1: camera / muzzle pose,
// a pending fire command spawns a pellet, every pellet flies one tick (RK4 + swept ray test), mocap visuals follow.
void step_launchers(const mjModel* m, mjData* d, Devices& dv);

// Solid motors and rocket airframes (rocket.cpp). prepare_motors, BEFORE mj_step1: sets mass / inertia / CoM of every motor body (and
// the subtree masses) from the motor state. step_motors / step_airframes, with the other forces, from the kinematics of mj_step1.
void prepare_motors(mjModel* m, Devices& dv);
void step_motors(const mjModel* m, mjData* d, Devices& dv);
void step_airframes(const mjModel* m, mjData* d, Devices& dv);
// Rain (rain.cpp): the rate in effect (0 without a rain device) and, in the force phase, the rain state of this tick (LWC, fall speed,
// film mass, the catch force on the film body). The rotor and rangefinder effects live in aero.cpp / rangefinder.cpp.
double rain_rate(const Devices& dv);
void step_rain(const mjModel* m, mjData* d, Devices& dv, const double wind[3]);
// Jet vanes (vane.cpp): plates in the exhaust of a rotor, after the rotor forces of the same tick.
void step_vanes(const mjModel* m, mjData* d, Devices& dv);

// Water (vessel.cpp). Surface elevation at (x, y) at the sea's clock, 0-level plane without a sea; the current (world, horizontal).
double sea_eta(const Devices& dv, double x, double y);
void sea_current(const Devices& dv, double out[3]);
// The sea surface as unrolled triangles (position xyz, smooth normal xyz per vertex; 6 floats) over the grid xs x ys (world coordinates,
// ascending), at the sea's clock: what the renderer draws as water. Empty without a sea.
std::vector<float> sea_surface_mesh(const Devices& dv, const std::vector<double>& xs, const std::vector<double>& ys);
// Last force phase: buoyancy, hydrodynamic damping and the added-mass correction of every vessel, then the sea clock advances.
void step_vessels(const mjModel* m, mjData* d, Devices& dv);

// Qdd actuators (qdd.cpp), after the electrical solve of the tick (needs the node voltage): command intake, FOC current / voltage
// limits, torque on the joint, thermal state, bus current for the next solve.
void step_qdd(const mjModel* m, mjData* d, Devices& dv);
// Foot contact sensors (contact.cpp): post-step hook, samples the ones that are due from the contact forces of the tick just stepped.
void sample_feet(const mjModel* m, mjData* d, Devices& dv);

// Range sensors (rangefinder.cpp): post-step hook, samples the ones that are due from the kinematics of the tick just stepped.
void sample_ranges(const mjModel* m, mjData* d, Devices& dv);
// Rosette projection: unit direction (site frame, +x = up) of ray number `m` of the sensor's life (frame * rays_per_frame + k).
void rosette_dir(const RangefinderCfg& c, std::uint64_t m, double out[3]);

std::size_t state_bytes(const Devices& dv);
void save_state(const Devices& dv, std::byte* out);
void load_state(Devices& dv, const std::byte* in);

Result get(const Devices& dv, int slot, const std::string& attr, std::vector<double>& out);
Result set(Devices& dv, int slot, const std::string& attr, std::span<const double> values);
std::string unit(const Devices& dv, int slot, const std::string& attr);  // "" if unknown

}  // namespace devs
