// Device framework: config parse / bind, per-tick pipeline, state block, prim attributes. Physics: electrical.cpp, aero.cpp.
#include "mujoco/devices/devices.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "devices_bfbs_generated.h"
#include "devices_generated.h"
#include "flatbuffers/idl.h"

namespace fb = dvsim::dev;  // generated config tables (devices.fbs)

namespace devs {

// ---- RNG ----
namespace {
std::uint64_t splitmix64(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

void Rng::seed(std::uint64_t x) {
  for (std::uint64_t& w : s) w = splitmix64(x);  // splitmix64 never yields an all-zero xoshiro state
}
std::uint64_t Rng::next() {
  const std::uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return r;
}
double Rng::uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
double Rng::gauss() {
  const double u1 = 1.0 - uniform(), u2 = uniform();  // u1 in (0, 1]: log is finite
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

const std::string& Devices::name(int slot) const {
  const Slot& s = slots[static_cast<std::size_t>(slot)];
  switch (s.type) {
    case Slot::battery: return battery.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::psu: return psu.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::wire: return wire.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::load: return load.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::rotor: return rotor.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::env: return env.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::drag: return drag.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::servo: return servo.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::imu: return imu.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::baro: return baro.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::mag: return mag.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::gnss: return gnss.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::launcher: return launcher.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::camera: return camera.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::motor: return motor.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::airframe: return airframe.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::sea: return sea.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::vessel: return vessel.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::range: return range.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::vane: return vane.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::wfield: return wfield.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::rain: return rain.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::qdd: return qdd.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::foot: return foot.cfg[static_cast<std::size_t>(s.i)].meta.name;
    case Slot::esc: break;
  }
  return esc.cfg[static_cast<std::size_t>(s.i)].meta.name;
}

// ---- parse ----
std::optional<std::string> config_text(const mjModel* m) {
  for (int i = 0; i < m->ntext; ++i) {
    const char* n = mj_id2name(m, mjOBJ_TEXT, i);
    if (n && std::string_view(n) == "dvsim.devices") return std::string(m->text_data + m->text_adr[i]);
  }
  return std::nullopt;
}

namespace {
[[noreturn]] void fail(const std::string& dev, const std::string& what) {
  throw std::runtime_error("dvsim.devices: device '" + dev + "': " + what);
}
// `ok` is written as a positive comparison so NaN fails it.
void need(bool ok, const std::string& dev, const char* what) {
  if (!ok) fail(dev, what);
}
Meta meta_of(const flatbuffers::String* name, const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* parts,
             std::set<std::string>& seen) {
  Meta m;
  m.name = name->str();
  if (m.name.empty() || m.name.find('/') != std::string::npos) fail(m.name, "name must be non-empty and contain no '/'");
  if (!seen.insert(m.name).second) fail(m.name, "duplicate device name");
  if (parts)
    for (const flatbuffers::String* p : *parts) m.parts.push_back(p->str());
  return m;
}
void node_ok(int node, const std::string& dev, int lo = 1) {
  if (node < lo || node > kMaxNodes) fail(dev, "node must be in [" + std::to_string(lo) + ", " + std::to_string(kMaxNodes) + "]");
}
MotorCfg motor_of(const fb::MotorParams* mp, const std::string& dev) {  // Esc and Servo
  MotorCfg c{mp->kv(), mp->r20(), mp->i0(), mp->rotor_inertia(), mp->w_s(), mp->c_th(), mp->r_th(), mp->t_amb(),
             mp->alpha()};
  c.i0_w = mp->i0_w(); c.i0_visc = mp->i0_visc(); c.i0_quad = mp->i0_quad();
  need(std::isfinite(c.i0_w) && c.i0_w >= 0 && c.i0_visc >= 0 && c.i0_quad >= 0 && c.i0_visc + c.i0_quad <= 1.0 &&
           (c.i0_visc + c.i0_quad == 0 || c.i0_w > 0),
       dev, "motor i0_visc, i0_quad >= 0 with sum <= 1, and i0_w > 0 when either is set");
  need(c.kv > 0 && c.r20 > 0 && c.inertia > 0 && c.w_s > 0 && c.c_th > 0 && c.r_th > 0, dev,
       "motor kv, r20, rotor_inertia, w_s, c_th, r_th must be > 0");
  need(c.i0 >= 0 && std::isfinite(c.t_amb) && c.alpha >= 0, dev, "motor i0 and alpha must be >= 0, t_amb finite");
  return c;
}
// 3 values or none (= 0), finite.
void vec3_of(const flatbuffers::Vector<double>* v, double out[3], const std::string& dev, const char* what) {
  need(!v || v->size() == 0 || v->size() == 3, dev, what);
  for (int k = 0; k < 3; ++k) {
    out[k] = v && v->size() ? v->Get(static_cast<flatbuffers::uoffset_t>(k)) : 0.0;
    need(std::isfinite(out[k]), dev, what);
  }
}
InertialErrCfg inertial_of(const fb::InertialErr* e, const std::string& dev) {
  InertialErrCfg c{};
  vec3_of(e->bias(), c.bias, dev, "gyro / accel bias needs 3 finite values (or none)");
  vec3_of(e->scale(), c.scale, dev, "gyro / accel scale needs 3 finite values (or none)");
  c.noise_density = e->noise_density(); c.rw_density = e->rw_density();
  c.full_scale = e->full_scale(); c.bits = e->bits();
  need(c.noise_density >= 0 && std::isfinite(c.noise_density) && c.rw_density >= 0 && std::isfinite(c.rw_density), dev,
       "noise_density and rw_density must be finite and >= 0");
  need(c.full_scale >= 0 && std::isfinite(c.full_scale), dev, "full_scale must be >= 0 (0 = no saturation)");
  need(c.bits >= 0 && c.bits <= 32 && (c.bits == 0 || c.full_scale > 0), dev,
       "bits must be in [0, 32] (0 = no quantisation) and needs full_scale > 0");
  return c;
}
void sensor_ok(const std::string& dev, const std::string& site, double rate_hz) {
  need(!site.empty(), dev, "site must be non-empty");
  need(rate_hz > 0 && std::isfinite(rate_hz), dev, "rate_hz must be > 0");
}
void std_ok(const std::string& dev, double sd, const char* what) { need(sd >= 0 && std::isfinite(sd), dev, what); }
// Seed of device slot k: its own `seed` if given, else derived from the top-level seed and the slot index.
std::uint64_t seed_of(std::uint64_t own, std::uint64_t top, std::size_t slot) { return own ? own : top + slot; }
constexpr double kG0 = 9.80665;  // standard gravity: wave dispersion (the sea's g, not the model's)
}  // namespace

Devices parse(const std::string& json) {
  flatbuffers::IDLOptions opts;
  opts.strict_json = true;  // quoted names; skip_unexpected_fields_in_json stays false => an unknown field is an error
  flatbuffers::Parser parser(opts);
  if (!parser.Deserialize(fb::DevicesBinarySchema::data(), fb::DevicesBinarySchema::size()))
    throw std::runtime_error("dvsim.devices: embedded schema failed to load");
  if (!parser.Parse(json.c_str())) throw std::runtime_error("dvsim.devices: " + parser.error_);
  const fb::Devices* root = flatbuffers::GetRoot<fb::Devices>(parser.builder_.GetBufferPointer());

  Devices dv;
  std::set<std::string> seen;
  const std::uint64_t top = root->seed();
  auto slot = [&](Devices::Slot::Type t, int i) { dv.slots.push_back({t, i}); };
  auto use_node = [&](int node) { if (node > dv.n_nodes) dv.n_nodes = node; };

  if (root->batteries())
    for (const fb::Battery* b : *root->batteries()) {
      BatteryCfg c{};
      c.meta = meta_of(b->name(), b->parts(), seen);
      const std::string& n = c.meta.name;
      c.node = b->node();
      node_ok(c.node, n);
      if (b->ocv_soc()) c.ocv_soc.assign(b->ocv_soc()->begin(), b->ocv_soc()->end());
      if (b->ocv_v()) c.ocv_v.assign(b->ocv_v()->begin(), b->ocv_v()->end());
      need(!c.ocv_soc.empty() && c.ocv_soc.size() == c.ocv_v.size(), n, "ocv_soc / ocv_v must be equal-length and non-empty");
      for (std::size_t k = 0; k < c.ocv_soc.size(); ++k) {
        need(c.ocv_soc[k] >= 0 && c.ocv_soc[k] <= 1 && std::isfinite(c.ocv_v[k]), n, "ocv_soc must lie in [0,1], ocv_v finite");
        need(k == 0 || c.ocv_soc[k] > c.ocv_soc[k - 1], n, "ocv_soc must be strictly increasing");
      }
      c.capacity_ah = b->capacity_ah(); c.r0 = b->r0(); c.r1 = b->r1(); c.c1 = b->c1(); c.soc0 = b->soc0();
      need(c.capacity_ah > 0, n, "capacity_ah must be > 0");
      need(c.r0 > 0, n, "r0 must be > 0");
      need(c.r1 >= 0 && (c.r1 == 0 || c.c1 > 0), n, "r1 must be >= 0 and c1 > 0 when r1 > 0");
      need(c.soc0 >= 0 && c.soc0 <= 1, n, "soc0 must lie in [0,1]");
      c.seed = b->seed();
      use_node(c.node);
      slot(Devices::Slot::battery, static_cast<int>(dv.battery.cfg.size()));
      dv.battery.cfg.push_back(std::move(c));
    }
  if (root->psus())
    for (const fb::Psu* p : *root->psus()) {
      PsuCfg c{};
      c.meta = meta_of(p->name(), p->parts(), seen);
      const std::string& n = c.meta.name;
      c.node = p->node();
      node_ok(c.node, n);
      c.voltage = p->voltage(); c.r_series = p->r_series(); c.i_limit = p->i_limit(); c.enabled0 = p->enabled();
      need(c.voltage > 0, n, "voltage must be > 0");
      need(c.r_series > 0, n, "r_series must be > 0");
      need(c.i_limit >= 0, n, "i_limit must be >= 0 (0 = none)");
      use_node(c.node);
      slot(Devices::Slot::psu, static_cast<int>(dv.psu.cfg.size()));
      dv.psu.cfg.push_back(std::move(c));
    }
  if (root->wires())
    for (const fb::Wire* w : *root->wires()) {
      WireCfg c{};
      c.meta = meta_of(w->name(), w->parts(), seen);
      const std::string& n = c.meta.name;
      c.a = w->a(); c.b = w->b();
      node_ok(c.a, n, 0);
      node_ok(c.b, n, 0);
      need(c.a != c.b, n, "a and b must differ");
      if (w->resistance() != 0) {
        need(w->resistance() > 0 && w->awg() == 0 && w->length_m() == 0, n, "give either resistance > 0, or awg + length_m, not both");
        c.r = w->resistance();
      } else {
        need(w->awg() >= 1 && w->awg() <= 60 && w->length_m() > 0 && w->conductors() >= 1, n,
             "need resistance > 0, or awg in [1,60] with length_m > 0 and conductors >= 1");
        // AWG: d = 0.127 mm * 92^((36 - AWG)/39); R = rho_Cu(20 degC) L / (pi d^2 / 4), x conductors for a round trip.
        const double d = 0.127e-3 * std::pow(92.0, (36.0 - w->awg()) / 39.0);
        c.r = 1.724e-8 * w->length_m() * w->conductors() / (kPi * d * d / 4.0);
      }
      use_node(c.a);
      use_node(c.b);
      slot(Devices::Slot::wire, static_cast<int>(dv.wire.cfg.size()));
      dv.wire.cfg.push_back(std::move(c));
    }
  if (root->loads())
    for (const fb::Load* l : *root->loads()) {
      LoadCfg c{};
      c.meta = meta_of(l->name(), l->parts(), seen);
      const std::string& n = c.meta.name;
      c.node = l->node(); c.g = l->g(); c.i0 = l->i0();
      node_ok(c.node, n);
      need(c.g >= 0 && std::isfinite(c.i0), n, "g must be >= 0 and i0 finite");
      use_node(c.node);
      slot(Devices::Slot::load, static_cast<int>(dv.load.cfg.size()));
      dv.load.cfg.push_back(std::move(c));
    }
  if (root->escs())
    for (const fb::Esc* e : *root->escs()) {
      EscCfg c{};
      c.meta = meta_of(e->name(), e->parts(), seen);
      const std::string& n = c.meta.name;
      c.node = e->node();
      node_ok(c.node, n);
      c.joint = e->joint()->str();
      need(!c.joint.empty(), n, "joint must be non-empty");
      c.motor = motor_of(e->motor(), n);
      c.complementary = e->complementary();
      c.i_limit = e->i_limit();
      need(c.i_limit >= 0, n, "i_limit must be >= 0 (0 = none)");
      c.r_on = e->r_on(); c.sw_loss = e->sw_loss(); c.i_q = e->i_q();
      need(c.r_on >= 0 && c.r_on < c.motor.r20 * 10 && std::isfinite(c.r_on), n, "r_on must be in [0, 10 R20)");
      need(c.sw_loss >= 0 && c.sw_loss < 0.5, n, "sw_loss must be in [0, 0.5)");
      need(c.i_q >= 0 && std::isfinite(c.i_q), n, "i_q must be >= 0");
      c.bidirectional = e->bidirectional();
      c.seed = e->seed();
      use_node(c.node);
      slot(Devices::Slot::esc, static_cast<int>(dv.esc.cfg.size()));
      dv.esc.cfg.push_back(std::move(c));
    }
  if (root->rotors())
    for (const fb::Rotor* r : *root->rotors()) {
      RotorCfg c{};
      c.meta = meta_of(r->name(), r->parts(), seen);
      const std::string& n = c.meta.name;
      c.body = r->body()->str();
      c.joint = r->joint()->str();
      need(!c.body.empty() && !c.joint.empty(), n, "body and joint must be non-empty");
      c.diameter = r->diameter();
      need(c.diameter > 0 && std::isfinite(c.diameter), n, "diameter must be > 0");
      c.dir = r->dir();
      need(c.dir == 1 || c.dir == -1, n, "dir must be +1 or -1");
      if (r->j()) c.j.assign(r->j()->begin(), r->j()->end());
      if (r->ct()) c.ct.assign(r->ct()->begin(), r->ct()->end());
      if (r->cq()) c.cq.assign(r->cq()->begin(), r->cq()->end());
      need(!c.j.empty() && c.j.size() == c.ct.size() && c.j.size() == c.cq.size(), n,
           "j / ct / cq must be equal-length and non-empty");
      for (std::size_t k = 0; k < c.j.size(); ++k) {
        need(std::isfinite(c.j[k]) && std::isfinite(c.ct[k]) && std::isfinite(c.cq[k]), n, "j / ct / cq must be finite");
        need(k == 0 || c.j[k] > c.j[k - 1], n, "j must be strictly increasing");
      }
      if (r->re_x()) c.re_x.assign(r->re_x()->begin(), r->re_x()->end());
      if (r->re_ft()) c.re_ft.assign(r->re_ft()->begin(), r->re_ft()->end());
      if (r->re_fq()) c.re_fq.assign(r->re_fq()->begin(), r->re_fq()->end());
      need(c.re_x.size() == c.re_ft.size() && c.re_x.size() == c.re_fq.size(), n,
           "re_x / re_ft / re_fq must be equal-length (or all empty = off)");
      for (std::size_t k = 0; k < c.re_x.size(); ++k) {
        need(std::isfinite(c.re_x[k]) && c.re_x[k] > 0 && std::isfinite(c.re_ft[k]) && c.re_ft[k] > 0 &&
                 std::isfinite(c.re_fq[k]) && c.re_fq[k] > 0,
             n, "re_x, re_ft, re_fq must be finite and > 0");
        need(k == 0 || c.re_x[k] > c.re_x[k - 1], n, "re_x must be strictly increasing");
      }
      c.k_h = r->k_h();
      need(c.k_h >= 0 && std::isfinite(c.k_h), n, "k_h must be >= 0");
      c.ground_effect = r->ground_effect();
      c.ray_groups = r->ray_groups();
      need(c.ray_groups < 64, n, "ray_groups is a 6-bit mask (geom groups 0..5)");
      c.water = r->water();
      if (r->stator_body()) c.stator_body = r->stator_body()->str();
      c.rain_kt = r->rain_kt(); c.rain_kq = r->rain_kq();
      need(c.rain_kt >= 0 && c.rain_kt <= 1 && c.rain_kq >= 0 && std::isfinite(c.rain_kq), n, "rain_kt must be in [0, 1], rain_kq >= 0");
      c.swirl_recovery = r->swirl_recovery();
      need(c.swirl_recovery >= 0 && c.swirl_recovery <= 1, n, "swirl_recovery must be in [0, 1]");
      need(c.swirl_recovery == 0 || !c.stator_body.empty(), n, "swirl_recovery needs a stator_body");
      c.rev_thrust = r->rev_thrust();
      need(c.rev_thrust > 0 && std::isfinite(c.rev_thrust), n, "rev_thrust must be > 0");
      slot(Devices::Slot::rotor, static_cast<int>(dv.rotor.cfg.size()));
      dv.rotor.cfg.push_back(std::move(c));
    }
  if (root->environments())
    for (const fb::Environment* e : *root->environments()) {
      EnvCfg c{};
      c.meta = meta_of(e->name(), e->parts(), seen);
      const std::string& n = c.meta.name;
      need(dv.env.cfg.empty(), n, "only one environment device is allowed");
      c.origin_alt = e->origin_alt();
      need(std::isfinite(c.origin_alt) && c.origin_alt > -1000 && c.origin_alt < 10000, n,
           "origin_alt must be in (-1000, 10000) m");
      need(!e->wind() || e->wind()->size() == 3, n, "wind needs 3 values (or none)");
      for (int k = 0; k < 3 && e->wind(); ++k) {
        c.wind0[k] = e->wind()->Get(static_cast<flatbuffers::uoffset_t>(k));
        need(std::isfinite(c.wind0[k]), n, "wind must be finite");
      }
      c.w20 = e->w20();
      need(c.w20 >= 0 && std::isfinite(c.w20), n, "w20 must be >= 0");
      c.seed = e->seed();
      if (e->ref_body()) c.ref_body = e->ref_body()->str();
      c.ref_height = e->ref_height();
      need(c.ref_height >= 0 && std::isfinite(c.ref_height), n, "ref_height must be >= 0");
      slot(Devices::Slot::env, static_cast<int>(dv.env.cfg.size()));
      dv.env.cfg.push_back(std::move(c));
    }
  if (root->body_drags())
    for (const fb::BodyDrag* b : *root->body_drags()) {
      DragCfg c{};
      c.meta = meta_of(b->name(), b->parts(), seen);
      const std::string& n = c.meta.name;
      c.body = b->body()->str();
      need(!c.body.empty(), n, "body must be non-empty");
      c.cd = b->cd();
      c.area = b->area();
      need(c.cd >= 0 && std::isfinite(c.cd) && c.area >= 0 && std::isfinite(c.area), n, "cd and area must be >= 0");
      slot(Devices::Slot::drag, static_cast<int>(dv.drag.cfg.size()));
      dv.drag.cfg.push_back(std::move(c));
    }
  if (root->servos())
    for (const fb::Servo* sv : *root->servos()) {
      ServoCfg c{};
      c.meta = meta_of(sv->name(), sv->parts(), seen);
      const std::string& n = c.meta.name;
      c.node = sv->node();
      node_ok(c.node, n);
      c.joint = sv->joint()->str();
      need(!c.joint.empty(), n, "joint must be non-empty");
      c.motor = motor_of(sv->motor(), n);
      c.n = sv->gear_ratio(); c.eta = sv->efficiency(); c.cpr = sv->encoder_cpr(); c.rate_hz = sv->firmware_hz();
      need(c.n > 0 && std::isfinite(c.n), n, "gear_ratio must be > 0");
      need(c.eta > 0 && c.eta <= 1, n, "efficiency must be in (0, 1]");
      need(c.cpr > 0, n, "encoder_cpr must be > 0");
      need(c.rate_hz > 0 && std::isfinite(c.rate_hz), n, "firmware_hz must be > 0");
      c.p = sv->p(); c.i = sv->i(); c.d = sv->d(); c.pid_scale = sv->pid_scale(); c.deadband = sv->deadband();
      need(c.p >= 0 && c.i >= 0 && c.d >= 0 && std::isfinite(c.p + c.i + c.d), n, "p, i, d must be finite and >= 0");
      need(c.pid_scale > 0 && std::isfinite(c.pid_scale), n, "pid_scale must be > 0");
      c.d_scale = sv->d_scale();
      need(c.d_scale >= 0 && std::isfinite(c.d_scale), n, "d_scale must be >= 0 (0 = pid_scale)");
      need(c.deadband >= 0 && std::isfinite(c.deadband), n, "deadband must be >= 0 (counts)");
      c.max_torque = sv->max_torque();
      need(c.max_torque > 0 && c.max_torque <= 1000, n, "max_torque must be in (0, 1000]");
      c.i_limit = sv->i_limit();
      need(c.i_limit >= 0 && std::isfinite(c.i_limit), n, "i_limit must be >= 0 (0 = none)");
      if (sv->backlash_joint()) c.backlash_joint = sv->backlash_joint()->str();
      c.min_voltage = sv->min_voltage(); c.max_voltage = sv->max_voltage(); c.max_temp = sv->max_temp();
      c.overload_load = sv->overload_load(); c.overload_time = sv->overload_time(); c.unload_mask = sv->unload_mask();
      need(std::isfinite(c.min_voltage + c.max_voltage + c.max_temp + c.overload_time) && c.min_voltage >= 0 &&
               c.max_voltage >= 0 && c.max_temp >= 0 && c.overload_time >= 0,
           n, "min_voltage, max_voltage, max_temp, overload_time must be finite and >= 0 (0 = off)");
      need(c.overload_load >= 0 && c.overload_load <= 1 && c.unload_mask >= 0 && c.unload_mask <= 255, n,
           "overload_load must be in [0, 1], unload_mask in [0, 255]");
      use_node(c.node);
      slot(Devices::Slot::servo, static_cast<int>(dv.servo.cfg.size()));
      dv.servo.cfg.push_back(std::move(c));
    }
  if (root->imus())
    for (const fb::Imu* s : *root->imus()) {
      ImuCfg c{};
      c.meta = meta_of(s->name(), s->parts(), seen);
      const std::string& n = c.meta.name;
      c.site = s->site()->str(); c.rate_hz = s->rate_hz(); c.seed = s->seed();
      sensor_ok(n, c.site, c.rate_hz);
      c.gyro = inertial_of(s->gyro(), n);
      c.accel = inertial_of(s->accel(), n);
      slot(Devices::Slot::imu, static_cast<int>(dv.imu.cfg.size()));
      dv.imu.cfg.push_back(std::move(c));
    }
  if (root->baros())
    for (const fb::Baro* s : *root->baros()) {
      BaroCfg c{};
      c.meta = meta_of(s->name(), s->parts(), seen);
      const std::string& n = c.meta.name;
      c.site = s->site()->str(); c.rate_hz = s->rate_hz(); c.seed = s->seed();
      sensor_ok(n, c.site, c.rate_hz);
      c.bias = s->bias(); c.noise_std = s->noise_std(); c.resolution = s->resolution();
      need(std::isfinite(c.bias), n, "bias must be finite");
      std_ok(n, c.noise_std, "noise_std must be finite and >= 0");
      std_ok(n, c.resolution, "resolution must be finite and >= 0 (0 = none)");
      slot(Devices::Slot::baro, static_cast<int>(dv.baro.cfg.size()));
      dv.baro.cfg.push_back(std::move(c));
    }
  if (root->mags())
    for (const fb::Mag* s : *root->mags()) {
      MagCfg c{};
      c.meta = meta_of(s->name(), s->parts(), seen);
      const std::string& n = c.meta.name;
      c.site = s->site()->str(); c.rate_hz = s->rate_hz(); c.seed = s->seed();
      sensor_ok(n, c.site, c.rate_hz);
      need(s->earth_field() && s->earth_field()->size() == 3, n, "earth_field needs 3 values (world frame, T)");
      vec3_of(s->earth_field(), c.field, n, "earth_field must be finite");
      vec3_of(s->bias(), c.bias, n, "bias needs 3 finite values (or none)");
      c.noise_std = s->noise_std();
      std_ok(n, c.noise_std, "noise_std must be finite and >= 0");
      slot(Devices::Slot::mag, static_cast<int>(dv.mag.cfg.size()));
      dv.mag.cfg.push_back(std::move(c));
    }
  if (root->gnss_receivers())
    for (const fb::Gnss* s : *root->gnss_receivers()) {
      GnssCfg c{};
      c.meta = meta_of(s->name(), s->parts(), seen);
      const std::string& n = c.meta.name;
      c.site = s->site()->str(); c.rate_hz = s->rate_hz(); c.seed = s->seed();
      sensor_ok(n, c.site, c.rate_hz);
      c.lat0 = s->lat0(); c.lon0 = s->lon0(); c.alt0 = s->alt0();
      need(std::abs(c.lat0) <= 89.0 && std::abs(c.lon0) <= 180.0 && std::isfinite(c.alt0), n,
           "lat0 must be in [-89, 89] deg, lon0 in [-180, 180] deg, alt0 finite");
      c.horiz_std = s->horiz_std(); c.vert_std = s->vert_std(); c.vel_std = s->vel_std(); c.latency_s = s->latency_s();
      std_ok(n, c.horiz_std, "horiz_std must be finite and >= 0");
      std_ok(n, c.vert_std, "vert_std must be finite and >= 0");
      std_ok(n, c.vel_std, "vel_std must be finite and >= 0");
      std_ok(n, c.latency_s, "latency_s must be finite and >= 0");
      c.corr_time_s = s->corr_time_s(); c.corr_frac = s->corr_frac();
      std_ok(n, c.corr_time_s, "corr_time_s must be finite and >= 0");
      need(c.corr_frac >= 0 && c.corr_frac <= 1, n, "corr_frac must be in [0, 1]");
      const double delay = c.latency_s * c.rate_hz;
      need(delay < kGnssRing - 0.5, n, "latency_s * rate_hz must round to at most 63 samples (the ring capacity)");
      c.delay = static_cast<int>(std::lround(delay));
      slot(Devices::Slot::gnss, static_cast<int>(dv.gnss.cfg.size()));
      dv.gnss.cfg.push_back(std::move(c));
    }
  if (root->launchers())
    for (const fb::Launcher* l : *root->launchers()) {
      LauncherCfg c{};
      c.meta = meta_of(l->name(), l->parts(), seen);
      const std::string& n = c.meta.name;
      c.site = l->site()->str();
      need(!c.site.empty(), n, "site must be non-empty");
      c.v0 = l->muzzle_velocity(); c.mass = l->pellet_mass(); c.diameter = l->diameter(); c.cd = l->cd();
      c.max_time = l->max_time(); c.v_std = l->v_std(); c.spread_std = l->spread_std(); c.ray_groups = l->ray_groups();
      c.seed = l->seed();
      need(c.v0 > 0 && std::isfinite(c.v0), n, "muzzle_velocity must be > 0");
      need(c.mass > 0 && std::isfinite(c.mass), n, "pellet_mass must be > 0");
      need(c.diameter > 0 && std::isfinite(c.diameter), n, "diameter must be > 0");
      need(c.cd >= 0 && std::isfinite(c.cd), n, "cd must be >= 0");
      need(c.max_time > 0 && std::isfinite(c.max_time), n, "max_time must be > 0");
      need(c.v_std >= 0 && std::isfinite(c.v_std) && c.spread_std >= 0 && std::isfinite(c.spread_std), n,
           "v_std and spread_std must be >= 0");
      need(c.ray_groups < 64, n, "ray_groups is a 6-bit mask (geom groups 0..5)");
      if (l->visual_bodies())
        for (const flatbuffers::String* b : *l->visual_bodies()) c.visual.push_back(b->str());
      need(c.visual.size() <= static_cast<std::size_t>(kMaxPellets), n, "at most 16 visual_bodies (the pellet pool)");
      c.cl_k = l->cl_k(); c.spin_tau = l->spin_tau();
      need(c.cl_k >= 0 && c.cl_k <= 10 && std::isfinite(c.cl_k), n, "cl_k must be in [0, 10]");
      need(c.spin_tau >= 0 && std::isfinite(c.spin_tau), n, "spin_tau must be >= 0");
      if (c.cl_k > 0 || c.spin_tau > 0) {  // cuas2: own state group, so a launcher without these keeps its state layout and hash
        c.lx = static_cast<int>(dv.lx.cfg.size());
        dv.lx.cfg.push_back({static_cast<int>(dv.launcher.cfg.size())});
      }
      slot(Devices::Slot::launcher, static_cast<int>(dv.launcher.cfg.size()));
      dv.launcher.cfg.push_back(std::move(c));
    }
  if (root->cameras())
    for (const fb::Camera* s : *root->cameras()) {
      CameraCfg c{};
      c.meta = meta_of(s->name(), s->parts(), seen);
      const std::string& n = c.meta.name;
      c.site = s->site()->str();
      need(!c.site.empty(), n, "site must be non-empty");
      c.fov_v = s->fov_v(); c.width = s->width(); c.height = s->height(); c.rate_hz = s->rate_hz();
      need(c.fov_v > 0 && c.fov_v < kPi && std::isfinite(c.fov_v), n, "fov_v must be in (0, pi) rad");
      need(c.width > 0 && c.height > 0 && c.rate_hz > 0 && std::isfinite(c.rate_hz), n, "width, height, rate_hz must be > 0");
      slot(Devices::Slot::camera, static_cast<int>(dv.camera.cfg.size()));
      dv.camera.cfg.push_back(std::move(c));
    }
  if (root->solid_motors())
    for (const fb::SolidMotor* sm : *root->solid_motors()) {
      SolidMotorCfg c{};
      c.meta = meta_of(sm->name(), sm->parts(), seen);
      const std::string& n = c.meta.name;
      c.body = sm->body()->str(); c.site = sm->site()->str();
      need(!c.body.empty() && !c.site.empty(), n, "body and site must be non-empty");
      need(sm->time() && sm->thrust() && sm->time()->size() == sm->thrust()->size() && sm->time()->size() >= 2, n,
           "time / thrust must be equal-length curves of at least 2 points");
      c.t.assign(sm->time()->begin(), sm->time()->end());
      c.f.assign(sm->thrust()->begin(), sm->thrust()->end());
      if (c.t.front() > 0) {  // RASP files leave the (0, 0) start implied
        c.t.insert(c.t.begin(), 0.0);
        c.f.insert(c.f.begin(), 0.0);
      }
      need(c.t.front() == 0, n, "time must start at 0 (or after 0: then (0, 0) is implied)");
      c.cum.assign(c.t.size(), 0.0);
      for (std::size_t k = 0; k < c.t.size(); ++k) {
        need(std::isfinite(c.t[k]) && std::isfinite(c.f[k]) && c.f[k] >= 0, n, "time / thrust must be finite, thrust >= 0");
        need(k == 0 || c.t[k] > c.t[k - 1], n, "time must be strictly increasing");
        if (k) c.cum[k] = c.cum[k - 1] + 0.5 * (c.f[k] + c.f[k - 1]) * (c.t[k] - c.t[k - 1]);
      }
      need(c.f.back() == 0, n, "the last thrust point must be 0 (burnout)");
      c.i_total = c.cum.back();
      need(c.i_total > 0, n, "the curve has no impulse");
      c.prop_mass = sm->propellant_mass(); c.total_mass = sm->total_mass(); c.diameter = sm->diameter(); c.length = sm->length();
      c.delay = sm->ignition_delay(); c.scale0 = sm->thrust_scale();
      need(c.prop_mass > 0 && c.total_mass > c.prop_mass && std::isfinite(c.total_mass), n,
           "propellant_mass must be > 0 and total_mass > propellant_mass");
      need(c.diameter > 0 && c.length > 0 && std::isfinite(c.diameter + c.length), n, "diameter and length must be > 0");
      need(c.delay >= 0 && std::isfinite(c.delay), n, "ignition_delay must be >= 0");
      need(c.scale0 >= 0 && c.scale0 <= 2, n, "thrust_scale must be in [0, 2]");
      slot(Devices::Slot::motor, static_cast<int>(dv.motor.cfg.size()));
      dv.motor.cfg.push_back(std::move(c));
    }
  if (root->airframes())
    for (const fb::Airframe* a : *root->airframes()) {
      AirframeCfg c{};
      c.meta = meta_of(a->name(), a->parts(), seen);
      const std::string& n = c.meta.name;
      c.body = a->body()->str(); c.site = a->site()->str();
      need(!c.body.empty() && !c.site.empty(), n, "body and site must be non-empty");
      c.s_ref = a->s_ref(); c.cd0 = a->cd0(); c.cd_rev = a->cd_rev() > 0 ? a->cd_rev() : a->cd0();
      c.cn_alpha = a->cn_alpha(); c.x_cp = a->x_cp(); c.s_plan = a->s_plan(); c.cd_cross = a->cd_cross();
      need(c.s_ref > 0 && std::isfinite(c.s_ref), n, "s_ref must be > 0");
      need(c.cd0 >= 0 && c.cd_rev >= 0 && c.cn_alpha >= 0 && c.s_plan >= 0 && c.cd_cross >= 0 &&
               std::isfinite(c.cd0 + c.cd_rev + c.cn_alpha + c.s_plan + c.cd_cross + c.x_cp),
           n, "cd0, cd_rev, cn_alpha, s_plan, cd_cross must be finite and >= 0, x_cp finite");
      slot(Devices::Slot::airframe, static_cast<int>(dv.airframe.cfg.size()));
      dv.airframe.cfg.push_back(std::move(c));
    }
  if (root->seas())
    for (const fb::Sea* sea : *root->seas()) {
      SeaCfg c{};
      c.meta = meta_of(sea->name(), sea->parts(), seen);
      const std::string& n = c.meta.name;
      need(dv.sea.cfg.empty(), n, "only one sea device is allowed");
      c.rho = sea->rho(); c.level = sea->level(); c.depth = sea->depth();
      need(c.rho > 0 && std::isfinite(c.rho + c.level) && c.depth >= 0 && std::isfinite(c.depth), n,
           "rho must be > 0, level finite, depth >= 0 (0 = deep water)");
      vec3_of(sea->current(), c.current0, n, "current needs 3 finite values (or none)");
      auto add_wave = [&](double a, double period, double dir, double phase) {
        need(a >= 0 && period > 0 && std::isfinite(a + period + dir + phase), n, "wave amplitude >= 0, period > 0, all finite");
        const double w = 2.0 * kPi / period;
        double k = w * w / kG0;  // deep water
        if (c.depth > 0)         // finite depth: w^2 = g k tanh(k h), Newton from the deep-water k, fixed iteration count
          for (int it = 0; it < 60; ++it) {
            const double th = std::tanh(k * c.depth), ch = std::cosh(k * c.depth);
            k -= (kG0 * k * th - w * w) / (kG0 * th + kG0 * k * c.depth / (ch * ch));
          }
        c.waves.push_back({a, k, w, dir, phase});
      };
      if (sea->waves())
        for (const fb::Wave* w : *sea->waves()) add_wave(w->amplitude(), w->period(), w->direction(), w->phase());
      if (const fb::Jonswap* js = sea->jonswap()) {
        need(js->hs() > 0 && js->tp() > 0 && js->gamma() >= 1 && js->n() >= 1 && js->n() <= kMaxWaves - 16 &&
                 std::isfinite(js->hs() + js->tp() + js->gamma() + js->direction()),
             n, "jonswap: hs > 0, tp > 0, gamma >= 1, n in [1, 48]");
        Rng rng;
        rng.seed(seed_of(js->seed(), top, 7000));
        const double wp = 2.0 * kPi / js->tp(), lo = 0.5 * wp, dw = 3.0 * wp / js->n();
        std::vector<double> ws, ss;
        double m0 = 0;
        for (int i = 0; i < js->n(); ++i) {  // one frequency per bin, uniformly drawn (avoids a periodic surface)
          const double w = lo + (i + rng.uniform()) * dw, sg = w <= wp ? 0.07 : 0.09;
          const double r = std::exp(-(w - wp) * (w - wp) / (2.0 * sg * sg * wp * wp));
          const double s = std::pow(w, -5.0) * std::exp(-1.25 * std::pow(wp / w, 4.0)) * std::pow(js->gamma(), r);
          ws.push_back(w); ss.push_back(s); m0 += s * dw;
        }
        const double norm = js->hs() * js->hs() / 16.0 / m0;  // Hm0 = hs exactly for the synthesised sum
        for (int i = 0; i < js->n(); ++i)
          add_wave(std::sqrt(2.0 * ss[static_cast<std::size_t>(i)] * norm * dw), 2.0 * kPi / ws[static_cast<std::size_t>(i)], js->direction(),
                   2.0 * kPi * rng.uniform());
      }
      need(c.waves.size() <= static_cast<std::size_t>(kMaxWaves), n, "at most 64 wave components");
      slot(Devices::Slot::sea, static_cast<int>(dv.sea.cfg.size()));
      dv.sea.cfg.push_back(std::move(c));
    }
  if (root->vessels())
    for (const fb::Vessel* v : *root->vessels()) {
      VesselCfg c{};
      c.meta = meta_of(v->name(), v->parts(), seen);
      const std::string& n = c.meta.name;
      c.body = v->body()->str();
      need(!c.body.empty(), n, "body must be non-empty");
      need(v->hulls() && v->hulls()->size() >= 1, n, "at least one hull");
      for (const fb::Hull* h : *v->hulls()) {
        const std::string shape = h->shape() ? h->shape()->str() : "box";
        need(shape == "box" || shape == "cylinder", n, "hull shape must be 'box' or 'cylinder'");
        need(h->pos() && h->pos()->size() == 3 && h->size() && h->size()->size() == 3, n, "hull pos and size need 3 values");
        int cells[3] = {8, 4, 4};
        if (h->n() && h->n()->size()) {
          need(h->n()->size() == 3, n, "hull n needs 3 values (or none)");
          for (int k = 0; k < 3; ++k) cells[k] = h->n()->Get(static_cast<flatbuffers::uoffset_t>(k));
        }
        double pos[3], size[3];
        for (int k = 0; k < 3; ++k) {
          pos[k] = h->pos()->Get(static_cast<flatbuffers::uoffset_t>(k));
          size[k] = h->size()->Get(static_cast<flatbuffers::uoffset_t>(k));
          need(std::isfinite(pos[k]) && size[k] > 0 && std::isfinite(size[k]), n, "hull size must be > 0 and finite, pos finite");
          need(cells[k] >= 1 && cells[k] <= 64, n, "hull n must be in [1, 64] per axis");
        }
        const bool cyl = shape == "cylinder";
        const double vol = cyl ? kPi / 4.0 * size[0] * size[1] * size[2] : size[0] * size[1] * size[2];
        const double e[3] = {size[0] / cells[0], size[1] / cells[1], size[2] / cells[2]};
        std::vector<Cell> mine;
        for (int i = 0; i < cells[0]; ++i)
          for (int j = 0; j < cells[1]; ++j)
            for (int k = 0; k < cells[2]; ++k) {
              Cell cell{};
              cell.c[0] = pos[0] + (i + 0.5) * e[0] - 0.5 * size[0];
              cell.c[1] = pos[1] + (j + 0.5) * e[1] - 0.5 * size[1];
              cell.c[2] = pos[2] + (k + 0.5) * e[2] - 0.5 * size[2];
              if (cyl) {
                const double y = (cell.c[1] - pos[1]) / (0.5 * size[1]), z = (cell.c[2] - pos[2]) / (0.5 * size[2]);
                if (y * y + z * z > 1.0) continue;
              }
              cell.v = e[0] * e[1] * e[2];
              std::copy(e, e + 3, cell.e);
              mine.push_back(cell);
            }
        need(!mine.empty(), n, "hull has no cells (n too small for the section)");
        double sum = 0;
        for (const Cell& cell : mine) sum += cell.v;
        for (Cell& cell : mine) cell.v *= vol / sum;  // the cells carry exactly the analytic volume
        c.cells.insert(c.cells.end(), mine.begin(), mine.end());
        c.volume += vol;
      }
      auto six = [&](const flatbuffers::Vector<double>* x, double out[6], const char* what) {
        need(!x || x->size() == 0 || x->size() == 6, n, what);
        for (int k = 0; k < 6; ++k) {
          out[k] = x && x->size() ? x->Get(static_cast<flatbuffers::uoffset_t>(k)) : 0.0;
          need(std::isfinite(out[k]) && out[k] >= 0, n, what);
        }
      };
      six(v->added_mass(), c.a, "added_mass needs 6 finite values >= 0 (or none)");
      six(v->damp_lin(), c.dl, "damp_lin needs 6 finite values >= 0 (or none)");
      six(v->damp_quad(), c.dq, "damp_quad needs 6 finite values >= 0 (or none)");
      for (double x : c.a) c.has_a = c.has_a || x > 0;
      c.heading_frame = v->heading_frame();
      if (v->fins())
        for (const fb::Fin* f : *v->fins()) {
          FinCfg fin{};
          need(f->pos() && f->pos()->size() == 3, n, "fin pos needs 3 values");
          for (int k = 0; k < 3; ++k) fin.p[k] = f->pos()->Get(static_cast<flatbuffers::uoffset_t>(k));
          fin.area = f->area(), fin.span = f->span(), fin.cla = f->cl_alpha(), fin.cd = f->cd_cross();
          need(fin.area > 0 && fin.span > 0 && fin.cla >= 0 && fin.cd >= 0 && std::isfinite(fin.p[0] + fin.p[1] + fin.p[2] + fin.area + fin.span + fin.cla + fin.cd),
               n, "fin area and span must be > 0, cl_alpha and cd_cross >= 0, all finite");
          c.fins.push_back(fin);
        }
      slot(Devices::Slot::vessel, static_cast<int>(dv.vessel.cfg.size()));
      dv.vessel.cfg.push_back(std::move(c));
    }
  if (root->rangefinders())
    for (const fb::Rangefinder* rf : *root->rangefinders()) {
      RangefinderCfg c{};
      c.meta = meta_of(rf->name(), rf->parts(), seen);
      const std::string& n = c.meta.name;
      c.site = rf->site()->str(); c.rate_hz = rf->rate_hz(); c.seed = rf->seed();
      sensor_ok(n, c.site, c.rate_hz);
      c.proj = rf->projection() ? rf->projection()->str() : "pinhole";
      if (c.proj.empty()) c.proj = "pinhole";
      need(c.proj == "pinhole" || c.proj == "spherical" || c.proj == "rosette", n, "projection must be 'pinhole', 'spherical' or 'rosette'");
      c.pinhole = c.proj == "pinhole";
      c.rosette = c.proj == "rosette";
      c.fov_h = rf->fov_h(); c.fov_v = rf->fov_v(); c.n_h = rf->n_h(); c.n_v = rf->n_v();
      if (c.rosette) {  // a 1 x rays_per_frame image; the elevation span replaces fov_v
        c.fov_v_min = rf->fov_v_min(); c.fov_v_max = rf->fov_v_max();
        need(std::isfinite(c.fov_v_min + c.fov_v_max) && c.fov_v_min >= -0.5 * kPi && c.fov_v_max <= 0.5 * kPi && c.fov_v_min < c.fov_v_max, n,
             "rosette: need -pi/2 <= fov_v_min < fov_v_max <= pi/2");
        need(rf->rays_per_frame() >= 1, n, "rosette: rays_per_frame must be >= 1");
        c.n_h = rf->rays_per_frame(); c.n_v = 1; c.fov_v = c.fov_v_max - c.fov_v_min;
        // R2 offsets: pattern_seed shifts the start of the sequence (seed 0 starts at 0.5, 0.5)
        const double phi = 0.6180339887498949;
        c.off1 = std::fmod(0.5 + static_cast<double>(rf->pattern_seed() % 1000003ull) * phi, 1.0);
        c.off2 = std::fmod(0.5 + static_cast<double>(rf->pattern_seed() % 1000003ull) * phi * phi, 1.0);
      }
      need(c.n_h >= 1 && c.n_v >= 1 && static_cast<long long>(c.n_h) * c.n_v <= 4000000, n, "n_h, n_v must be >= 1 (at most 4e6 rays)");
      need(std::isfinite(c.fov_h + c.fov_v) && c.fov_h > 0 && c.fov_v > 0, n, "fov_h, fov_v must be > 0");
      need(c.pinhole ? (c.fov_h < kPi && c.fov_v < kPi) : (c.fov_h <= 2.0 * kPi && c.fov_v <= kPi), n,
           "pinhole: fov_h, fov_v < pi; spherical: fov_h <= 2 pi, fov_v <= pi");
      c.min_r = rf->min_range(); c.max_r = rf->max_range();
      need(c.min_r >= 0 && c.max_r > c.min_r && std::isfinite(c.max_r), n, "need 0 <= min_range < max_range");
      c.s0 = rf->sigma0(); c.s1 = rf->sigma1(); c.s2 = rf->sigma2(); c.res = rf->resolution(); c.dropout = rf->dropout();
      need(c.s0 >= 0 && c.s1 >= 0 && c.s2 >= 0 && c.res >= 0 && std::isfinite(c.s0 + c.s1 + c.s2 + c.res), n,
           "sigma0..2 and resolution must be finite and >= 0");
      need(c.dropout >= 0 && c.dropout <= 1, n, "dropout must be in [0, 1]");
      c.ray_groups = rf->ray_groups();
      need(c.ray_groups < 64, n, "ray_groups is a 6-bit mask (geom groups 0..5)");
      c.rain_a = rf->rain_a(); c.rain_b = rf->rain_b(); c.rain_margin_db = rf->rain_margin_db();
      c.rain_beam = rf->rain_beam(); c.rain_zone = rf->rain_zone(); c.rain_det = rf->rain_det();
      c.rain_window_db = rf->rain_window_db();
      need(c.rain_a >= 0 && c.rain_b >= 0 && c.rain_beam >= 0 && c.rain_zone >= 0 && c.rain_det >= 0 && c.rain_det <= 1 &&
               c.rain_window_db >= 0 && std::isfinite(c.rain_a + c.rain_b + c.rain_margin_db + c.rain_beam + c.rain_zone + c.rain_window_db),
           n, "rain_a, rain_b, rain_beam, rain_zone, rain_window_db >= 0, rain_det in [0, 1], rain_margin_db finite");
      const std::size_t cnt = static_cast<std::size_t>(c.n_h) * static_cast<std::size_t>(c.n_v);
      if (c.rosette) {  // directions are generated per frame (rangefinder.cpp); the range is the output
        c.scale.assign(cnt, 1.0);
        c.cutoff = c.max_r;
        slot(Devices::Slot::range, static_cast<int>(dv.range.cfg.size()));
        dv.range.cfg.push_back(std::move(c));
        continue;
      }
      c.dir.resize(3 * cnt);
      c.scale.resize(cnt);
      const double tu = std::tan(0.5 * c.fov_h), tv = std::tan(0.5 * c.fov_v);
      double smin = 1.0;
      for (int i = 0; i < c.n_v; ++i)
        for (int j = 0; j < c.n_h; ++j) {
          const std::size_t k = static_cast<std::size_t>(i) * static_cast<std::size_t>(c.n_h) + static_cast<std::size_t>(j);
          double x, y, z, sc = 1.0;
          if (c.pinhole) {
            x = tv * (1.0 - 2.0 * (i + 0.5) / c.n_v);
            y = tu * (2.0 * (j + 0.5) / c.n_h - 1.0);
            z = 1.0;
            const double nrm = std::sqrt(x * x + y * y + 1.0);
            x /= nrm; y /= nrm; z /= nrm; sc = z;  // z-depth = range * z
          } else {
            const double az = c.fov_h * ((j + 0.5) / c.n_h - 0.5), el = c.fov_v * (0.5 - (i + 0.5) / c.n_v);
            x = std::sin(el); y = std::cos(el) * std::sin(az); z = std::cos(el) * std::cos(az);
          }
          c.dir[3 * k] = x; c.dir[3 * k + 1] = y; c.dir[3 * k + 2] = z;
          c.scale[k] = sc;
          smin = std::min(smin, sc);
        }
      need(smin > 1e-6, n, "field of view too wide for the pinhole projection");
      c.cutoff = c.max_r / smin;
      slot(Devices::Slot::range, static_cast<int>(dv.range.cfg.size()));
      dv.range.cfg.push_back(std::move(c));
    }
  if (root->jet_vanes())
    for (const fb::JetVane* jv : *root->jet_vanes()) {
      JetVaneCfg c{};
      c.meta = meta_of(jv->name(), jv->parts(), seen);
      const std::string& n = c.meta.name;
      c.body = jv->body()->str(); c.site = jv->site()->str(); c.rotor = jv->rotor()->str();
      need(!c.body.empty() && !c.site.empty() && !c.rotor.empty(), n, "body, site and rotor must be non-empty");
      c.area = jv->area(); c.exit_area = jv->exit_area(); c.cn_alpha = jv->cn_alpha(); c.cn_max = jv->cn_max();
      c.cd0 = jv->cd0(); c.jet_factor = jv->jet_factor();
      need(c.area > 0 && c.exit_area > 0 && c.cn_max > 0 && std::isfinite(c.area + c.exit_area + c.cn_max), n,
           "area, exit_area and cn_max must be finite and > 0");
      need(c.cn_alpha >= 0 && c.cd0 >= 0 && c.jet_factor >= 0 && std::isfinite(c.cn_alpha + c.cd0 + c.jet_factor), n,
           "cn_alpha, cd0 and jet_factor must be finite and >= 0");
      slot(Devices::Slot::vane, static_cast<int>(dv.vane.cfg.size()));
      dv.vane.cfg.push_back(std::move(c));
    }
  if (root->wind_fields())
    for (const fb::WindField* w : *root->wind_fields()) {
      WindFieldCfg c{};
      c.meta = meta_of(w->name(), w->parts(), seen);
      const std::string& n = c.meta.name;
      need(dv.wfield.cfg.empty(), n, "only one wind_field device is allowed");
      need(!dv.env.cfg.empty(), n, "a wind_field needs an environment device");
      if (w->hfield_geom()) c.hfield_geom = w->hfield_geom()->str();
      c.ground_z = w->ground_z(); c.z0 = w->z0(); c.z_ref = w->z_ref(); c.h_max = w->h_max();
      c.ridge_gain = w->ridge_gain(); c.ridge_len = w->ridge_len(); c.lee_turb = w->lee_turb(); c.lee_down = w->lee_down();
      need(std::isfinite(c.ground_z + c.z0 + c.z_ref + c.h_max + c.ridge_gain + c.ridge_len + c.lee_turb + c.lee_down), n,
           "wind_field parameters must be finite");
      need(c.z0 >= 0 && (c.z0 == 0 || c.z_ref > 2.72 * c.z0) && c.h_max > c.z0, n, "need z0 >= 0, z_ref > e z0, h_max > z0");
      need(c.ridge_gain >= 0 && c.ridge_len > 0 && c.lee_turb >= 0 && c.lee_down >= 0, n,
           "need ridge_gain, lee_turb, lee_down >= 0 and ridge_len > 0");
      need(c.ridge_gain == 0 || !c.hfield_geom.empty(), n, "ridge_gain needs an hfield_geom");
      slot(Devices::Slot::wfield, static_cast<int>(dv.wfield.cfg.size()));
      dv.wfield.cfg.push_back(std::move(c));
    }
  if (root->rains())
    for (const fb::Rain* rn : *root->rains()) {
      RainCfg c{};
      c.meta = meta_of(rn->name(), rn->parts(), seen);
      const std::string& n = c.meta.name;
      need(dv.rain.cfg.empty(), n, "only one rain device is allowed");
      c.rate0 = rn->rate();
      need(c.rate0 >= 0 && c.rate0 <= 500, n, "rate must be in [0, 500] mm/h");
      if (rn->body()) c.body = rn->body()->str();
      c.catch_area = rn->catch_area(); c.wet_area = rn->wet_area(); c.film_depth = rn->film_depth();
      c.film_tau = rn->film_tau(); c.collect_eff = rn->collect_eff();
      need(c.catch_area >= 0 && c.wet_area >= 0 && c.film_depth >= 0 && c.film_tau > 0 && c.collect_eff >= 0 && c.collect_eff <= 1 &&
               std::isfinite(c.catch_area + c.wet_area + c.film_depth + c.film_tau),
           n, "catch_area, wet_area, film_depth >= 0, film_tau > 0, collect_eff in [0, 1]");
      slot(Devices::Slot::rain, static_cast<int>(dv.rain.cfg.size()));
      dv.rain.cfg.push_back(std::move(c));
    }
  if (root->qdds())
    for (const fb::Qdd* q : *root->qdds()) {
      QddCfg c{};
      c.meta = meta_of(q->name(), q->parts(), seen);
      const std::string& n = c.meta.name;
      c.node = q->node();
      node_ok(c.node, n);
      c.joint = q->joint()->str();
      need(!c.joint.empty(), n, "joint must be non-empty");
      c.motor = motor_of(q->motor(), n);
      c.n = q->gear_ratio(); c.eta = q->efficiency(); c.cpr = q->encoder_cpr(); c.cmd_hz = q->cmd_hz();
      c.latency_s = q->latency_s(); c.i_limit = q->i_limit();
      c.kp_max = q->kp_max(); c.kd_max = q->kd_max(); c.tau_ff_max = q->tau_ff_max();
      c.timeout_s = q->timeout_s(); c.timeout_mode = q->timeout_mode(); c.kd_timeout = q->kd_timeout();
      c.t_max = q->t_max(); c.derate_start = q->derate_start();
      need(c.n > 0 && std::isfinite(c.n), n, "gear_ratio must be > 0");
      need(c.eta > 0 && c.eta <= 1, n, "efficiency must be in (0, 1]");
      need(c.cpr > 0, n, "encoder_cpr must be > 0");
      need(c.cmd_hz > 0 && std::isfinite(c.cmd_hz), n, "cmd_hz must be > 0");
      need(c.latency_s >= 0 && std::isfinite(c.latency_s), n, "latency_s must be >= 0");
      need(c.i_limit > 0 && std::isfinite(c.i_limit), n, "i_limit must be > 0");
      need(c.kp_max > 0 && c.kd_max > 0 && c.tau_ff_max > 0 && std::isfinite(c.kp_max + c.kd_max + c.tau_ff_max), n,
           "kp_max, kd_max, tau_ff_max must be finite and > 0");
      need(c.timeout_s > 0 && std::isfinite(c.timeout_s), n, "timeout_s must be > 0");
      need(c.timeout_mode == 0 || c.timeout_mode == 1, n, "timeout_mode must be 0 (damping) or 1 (open)");
      need(c.kd_timeout >= 0 && std::isfinite(c.kd_timeout) && (c.timeout_mode == 1 || c.kd_timeout > 0), n,
           "kd_timeout must be finite and > 0 in timeout_mode 0");
      need(std::isfinite(c.derate_start + c.t_max) && c.t_max > c.derate_start && c.derate_start > c.motor.t_amb, n,
           "need t_amb < derate_start < t_max");
      use_node(c.node);
      slot(Devices::Slot::qdd, static_cast<int>(dv.qdd.cfg.size()));
      dv.qdd.cfg.push_back(std::move(c));
    }
  if (root->foot_contacts())
    for (const fb::FootContact* f : *root->foot_contacts()) {
      FootContactCfg c{};
      c.meta = meta_of(f->name(), f->parts(), seen);
      const std::string& n = c.meta.name;
      c.geom = f->geom()->str();
      need(!c.geom.empty(), n, "geom must be non-empty");
      c.mode = f->mode(); c.f_max = f->f_max(); c.noise_std = f->noise_std();
      c.hys_on = f->hysteresis_on(); c.hys_off = f->hysteresis_off(); c.debounce_s = f->debounce_s();
      c.rate_hz = f->rate_hz(); c.seed = f->seed();
      need(c.mode == 0 || c.mode == 1, n, "mode must be 0 (force) or 1 (binary)");
      need(c.f_max > 0 && std::isfinite(c.f_max), n, "f_max must be finite and > 0");
      need(c.noise_std >= 0 && std::isfinite(c.noise_std), n, "noise_std must be finite and >= 0");
      need(std::isfinite(c.hys_on) && c.hys_off >= 0 && c.hys_on >= c.hys_off && c.hys_on < c.f_max, n,
           "need 0 <= hysteresis_off <= hysteresis_on < f_max");
      need(c.debounce_s >= 0 && std::isfinite(c.debounce_s), n, "debounce_s must be finite and >= 0");
      need(c.rate_hz > 0 && std::isfinite(c.rate_hz), n, "rate_hz must be > 0");
      slot(Devices::Slot::foot, static_cast<int>(dv.foot.cfg.size()));
      dv.foot.cfg.push_back(std::move(c));
    }
  if (dv.slots.empty()) throw std::runtime_error("dvsim.devices: no devices declared");

  // Initial state. RNG stream k uses the device's flat slot index (battery, psu, wire, load, esc order).
  std::size_t k = 0;
  for (const BatteryCfg& c : dv.battery.cfg) {
    BatteryState s{};
    s.soc = c.soc0;
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.battery.st.push_back(s);
  }
  for (const PsuCfg& c : dv.psu.cfg) dv.psu.st.push_back(PsuState{c.enabled0 ? 1.0 : 0.0, 0, 0, 0}), ++k;
  k += dv.wire.cfg.size() + dv.load.cfg.size();
  dv.wire.st.assign(dv.wire.cfg.size(), WireState{});
  dv.load.st.assign(dv.load.cfg.size(), LoadState{});
  for (const EscCfg& c : dv.esc.cfg) {
    EscState s{};
    s.temp = c.motor.t_amb;
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.esc.st.push_back(s);
  }
  dv.rotor.st.assign(dv.rotor.cfg.size(), RotorState{0, 0, 0, 0, 1, 0});
  k += dv.rotor.cfg.size();
  for (const EnvCfg& c : dv.env.cfg) {
    EnvState s{};
    std::copy(c.wind0, c.wind0 + 3, s.wind);
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.env.st.push_back(s);
    dv.env.st.back().density = rho_at(dv, 0.0);  // valid before the first tick
  }
  dv.drag.st.assign(dv.drag.cfg.size(), DragState{});
  k += dv.drag.cfg.size();
  for (const ServoCfg& c : dv.servo.cfg) {  // no RNG stream: servos draw no noise
    ServoState s{};
    s.temp = c.motor.t_amb;
    s.p = c.p; s.i = c.i; s.d = c.d; s.deadband = c.deadband; s.max_torque = s.torque_limit = c.max_torque;
    s.min_voltage = c.min_voltage; s.max_voltage = c.max_voltage; s.max_temp = c.max_temp;
    s.overload_load = c.overload_load; s.overload_time = c.overload_time; s.unload_mask = c.unload_mask;
    dv.servo.st.push_back(s);
  }
  k += dv.servo.cfg.size();
  for (const ImuCfg& c : dv.imu.cfg) {
    ImuState s{};
    std::copy(c.gyro.bias, c.gyro.bias + 3, s.gyro_bias);
    std::copy(c.accel.bias, c.accel.bias + 3, s.accel_bias);
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.imu.st.push_back(s);
  }
  for (const BaroCfg& c : dv.baro.cfg) {
    BaroState s{};
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.baro.st.push_back(s);
  }
  for (const MagCfg& c : dv.mag.cfg) {
    MagState s{};
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.mag.st.push_back(s);
  }
  for (const GnssCfg& c : dv.gnss.cfg) {
    GnssState s{};
    s.lat = c.lat0; s.lon = c.lon0; s.alt = c.alt0;  // before the first sample (and the first `delay` ones): the origin
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.gnss.st.push_back(s);
  }
  for (const LauncherCfg& c : dv.launcher.cfg) {
    LauncherState s{};
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.launcher.st.push_back(s);
  }
  dv.camera.st.assign(dv.camera.cfg.size(), CameraState{});
  // Phase 5 devices: their RNG streams (rangefinders only) come after every earlier slot, so the older streams are unchanged.
  for (const SolidMotorCfg& c : dv.motor.cfg) {
    SolidMotorState s{};
    s.thrust_scale = c.scale0;
    s.mass = c.total_mass;
    dv.motor.st.push_back(s);
  }
  dv.airframe.st.assign(dv.airframe.cfg.size(), AirframeState{});
  for (const SeaCfg& c : dv.sea.cfg) {
    SeaState s{};
    s.wave_scale = 1.0;
    std::copy(c.current0, c.current0 + 3, s.current);
    s.n_waves = static_cast<double>(c.waves.size());
    for (std::size_t w = 0; w < c.waves.size(); ++w) {
      const WaveCfg& x = c.waves[w];
      const double row[5] = {x.a, x.k, x.w, x.th, x.phi};
      std::copy(row, row + 5, s.waves + 5 * w);
    }
    dv.sea.st.push_back(s);
  }
  dv.vessel.st.assign(dv.vessel.cfg.size(), VesselState{});
  for (const RangefinderCfg& c : dv.range.cfg) {
    RangefinderState s{};
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.range.st.push_back(s);
    dv.range_img.emplace_back(static_cast<std::size_t>(c.n_h) * static_cast<std::size_t>(c.n_v), 0.0);
  }
  dv.vane.st.assign(dv.vane.cfg.size(), JetVaneState{});
  dv.wfield.st.assign(dv.wfield.cfg.size(), WindState{});
  for (const RainCfg& c : dv.rain.cfg) {
    RainState s{};
    s.rate = c.rate0;
    dv.rain.st.push_back(s);
  }
  for (const QddCfg& c : dv.qdd.cfg) {  // no RNG stream: ideal encoder, no noise
    QddState s{};
    s.temp = c.motor.t_amb;
    s.derate = 1.0;
    dv.qdd.st.push_back(s);
  }
  for (const FootContactCfg& c : dv.foot.cfg) {  // streams after every older slot: their streams are unchanged
    FootContactState s{};
    s.pend = -1.0;
    s.rng.seed(seed_of(c.seed, top, k++));
    dv.foot.st.push_back(s);
  }
  dv.lx.st.assign(dv.lx.cfg.size(), LauncherXState{});
  if (!dv.sea.cfg.empty()) dv.sea.st[0].eta0 = sea_eta(dv, 0.0, 0.0);
  return dv;
}

void bind(mjModel* m, Devices& dv) {
  if (m->opt.integrator == mjINT_RK4)
    throw std::runtime_error("dvsim.devices: devices step with mj_step1/mj_step2, which RK4 does not support; use Euler, "
                             "implicit or implicitfast");
  std::set<int> used;
  auto hinge = [&](const std::string& dev, const std::string& joint) {
    const int j = mj_name2id(m, mjOBJ_JOINT, joint.c_str());
    if (j < 0) fail(dev, "joint '" + joint + "' not found in the model");
    if (m->jnt_type[j] != mjJNT_HINGE) fail(dev, "joint '" + joint + "' is not a hinge");
    return j;
  };
  for (EscCfg& c : dv.esc.cfg) {
    c.jnt = hinge(c.meta.name, c.joint);
    if (!used.insert(c.jnt).second) fail(c.meta.name, "joint '" + c.joint + "' is already driven by another device");
    c.dof = m->jnt_dofadr[c.jnt];
  }
  // Period in ticks of a firmware / sensor rate: 1 / (rate dt) must be an integer >= 1.
  auto period_of = [&](const std::string& dev, const char* key, double rate_hz, const char* what) {
    const double ticks = 1.0 / (rate_hz * m->opt.timestep);
    const int period = ticks < 2e9 ? static_cast<int>(std::lround(ticks)) : 0;  // 0: out of range, fails below
    if (period < 1 || std::abs(ticks - period) > 1e-9 * ticks)
      fail(dev, std::string(key) + " " + std::to_string(rate_hz) + " gives " + std::to_string(ticks) + " ticks per " +
                    what + " at the model timestep; need an integer >= 1");
    return period;
  };
  for (std::size_t i = 0; i < dv.servo.cfg.size(); ++i) {
    ServoCfg& c = dv.servo.cfg[i];
    const std::string& n = c.meta.name;
    c.jnt = hinge(n, c.joint);
    if (!used.insert(c.jnt).second) fail(n, "joint '" + c.joint + "' is already driven by another device");
    c.dof = m->jnt_dofadr[c.jnt];
    c.qadr = m->jnt_qposadr[c.jnt];
    // A model cloned mid-jam carries the jam limit (zero width, which the compiler never produces): take the defaults then.
    // ponytail: a joint that was limited before such a clone loses its range; the clone path would need the config to say so.
    if (m->jnt_limited[c.jnt] && m->jnt_range[2 * c.jnt] != m->jnt_range[2 * c.jnt + 1]) {
      c.lim0 = 1;
      mju_copy(c.range0, m->jnt_range + 2 * c.jnt, 2);
      mju_copy(c.solref0, m->jnt_solref + mjNREF * c.jnt, 2);
      mju_copy(c.solimp0, m->jnt_solimp + mjNIMP * c.jnt, 5);
    }
    c.period = period_of(n, "firmware_hz", c.rate_hz, "firmware step");
    ServoState& s = dv.servo.st[i];
    s.backlash = 0;
    if (!c.backlash_joint.empty()) {  // single source of truth: the MJCF range of the (undriven) backlash hinge
      const int bj = hinge(n, c.backlash_joint);
      if (bj == c.jnt) fail(n, "backlash_joint must differ from joint");
      const double* r = m->jnt_range + 2 * bj;
      if (!m->jnt_limited[bj] || !(r[1] > 0) || r[0] != -r[1])
        fail(n, "backlash_joint '" + c.backlash_joint + "' must be limited with a symmetric range -b..+b, b > 0");
      s.backlash = r[1];
    }
  }
  auto body_id = [&](const std::string& dev, const std::string& body) {
    const int b = mj_name2id(m, mjOBJ_BODY, body.c_str());
    if (b <= 0) fail(dev, "body '" + body + "' not found in the model (or is the world)");
    return b;
  };
  for (RotorCfg& c : dv.rotor.cfg) {
    c.bid = body_id(c.meta.name, c.body);
    c.jnt = mj_name2id(m, mjOBJ_JOINT, c.joint.c_str());
    if (c.jnt < 0) fail(c.meta.name, "joint '" + c.joint + "' not found in the model");
    if (m->jnt_type[c.jnt] != mjJNT_HINGE) fail(c.meta.name, "joint '" + c.joint + "' is not a hinge");
    if (m->jnt_bodyid[c.jnt] != c.bid) fail(c.meta.name, "joint '" + c.joint + "' is not on body '" + c.body + "'");
    c.dof = m->jnt_dofadr[c.jnt];
    c.stator_bid = c.stator_body.empty() ? -1 : body_id(c.meta.name, c.stator_body);
  }
  for (EnvCfg& c : dv.env.cfg) c.bid = c.ref_body.empty() ? -1 : body_id(c.meta.name, c.ref_body);
  for (DragCfg& c : dv.drag.cfg) c.bid = body_id(c.meta.name, c.body);
  auto site_id = [&](const std::string& dev, const std::string& site) {
    const int s = mj_name2id(m, mjOBJ_SITE, site.c_str());
    if (s < 0) fail(dev, "site '" + site + "' not found in the model");
    return s;
  };
  for (ImuCfg& c : dv.imu.cfg) {
    c.sid = site_id(c.meta.name, c.site);
    c.period = period_of(c.meta.name, "rate_hz", c.rate_hz, "sample");
  }
  for (BaroCfg& c : dv.baro.cfg) {
    c.sid = site_id(c.meta.name, c.site);
    c.period = period_of(c.meta.name, "rate_hz", c.rate_hz, "sample");
  }
  for (MagCfg& c : dv.mag.cfg) {
    c.sid = site_id(c.meta.name, c.site);
    c.period = period_of(c.meta.name, "rate_hz", c.rate_hz, "sample");
  }
  for (GnssCfg& c : dv.gnss.cfg) {
    c.sid = site_id(c.meta.name, c.site);
    c.period = period_of(c.meta.name, "rate_hz", c.rate_hz, "sample");
  }
  for (LauncherCfg& c : dv.launcher.cfg) {
    c.sid = site_id(c.meta.name, c.site);
    c.body = m->site_bodyid[c.sid];
    c.mocap.clear();
    for (const std::string& vb : c.visual) {
      const int b = body_id(c.meta.name, vb);
      if (m->body_mocapid[b] < 0) fail(c.meta.name, "visual body '" + vb + "' is not a mocap body");
      c.mocap.push_back(m->body_mocapid[b]);
    }
  }
  for (CameraCfg& c : dv.camera.cfg) c.sid = site_id(c.meta.name, c.site);
  for (SolidMotorCfg& c : dv.motor.cfg) {
    c.bid = body_id(c.meta.name, c.body);
    c.sid = site_id(c.meta.name, c.site);
    if (m->site_bodyid[c.sid] != c.bid) fail(c.meta.name, "site '" + c.site + "' is not on body '" + c.body + "'");
    // The device owns the body's inertia: the model must hold the motor and nothing else (the mass lies between the casing alone
    // and the loaded motor whatever burn state a cloned / reloaded model was saved in).
    const double mb = m->body_mass[c.bid], mcase = c.total_mass - c.prop_mass;
    if (!(mb >= 0.98 * mcase && mb <= 1.02 * c.total_mass))
      fail(c.meta.name, "body '" + c.body + "' has mass " + std::to_string(mb) + " kg, but the motor weighs " + std::to_string(mcase) +
                            " (empty) to " + std::to_string(c.total_mass) + " kg: the motor body must hold the motor only");
    for (int dof = 0; dof < m->nv; ++dof)  // a "simple" dof reads its inertia from dof_M0 (fixed at compile time)
      if (m->dof_bodyid[dof] == c.bid && m->dof_simplenum[dof] != 0)
        fail(c.meta.name, "body '" + c.body + "' is a MuJoCo 'simple' body (a hinge directly under a static body): its inertia cannot "
                          "change; put the motor on a body with a child, or below a moving body");
  }
  for (AirframeCfg& c : dv.airframe.cfg) {
    c.bid = body_id(c.meta.name, c.body);
    c.sid = site_id(c.meta.name, c.site);
    if (m->site_bodyid[c.sid] != c.bid) fail(c.meta.name, "site '" + c.site + "' is not on body '" + c.body + "'");
  }
  for (JetVaneCfg& c : dv.vane.cfg) {
    c.bid = body_id(c.meta.name, c.body);
    c.sid = site_id(c.meta.name, c.site);
    if (m->site_bodyid[c.sid] != c.bid) fail(c.meta.name, "site '" + c.site + "' is not on body '" + c.body + "'");
    c.rot = -1;
    for (std::size_t r = 0; r < dv.rotor.cfg.size(); ++r)
      if (dv.rotor.cfg[r].meta.name == c.rotor) c.rot = static_cast<int>(r);
    if (c.rot < 0) fail(c.meta.name, "rotor '" + c.rotor + "' is not a rotor device");
  }
  for (RainCfg& c : dv.rain.cfg) c.bid = c.body.empty() ? -1 : body_id(c.meta.name, c.body);
  for (VesselCfg& c : dv.vessel.cfg) {
    c.bid = body_id(c.meta.name, c.body);
    if (dv.sea.cfg.empty()) fail(c.meta.name, "a vessel needs a sea device");
  }
  for (WindFieldCfg& c : dv.wfield.cfg) bind_wind_field(m, c);
  for (const RotorCfg& c : dv.rotor.cfg)
    if (c.water && dv.sea.cfg.empty()) fail(c.meta.name, "a water rotor needs a sea device");
  for (RangefinderCfg& c : dv.range.cfg) {
    c.sid = site_id(c.meta.name, c.site);
    c.body = m->site_bodyid[c.sid];
    c.period = period_of(c.meta.name, "rate_hz", c.rate_hz, "sample");
  }
  for (QddCfg& c : dv.qdd.cfg) {
    const std::string& n = c.meta.name;
    c.jnt = hinge(n, c.joint);
    if (!used.insert(c.jnt).second) fail(n, "joint '" + c.joint + "' is already driven by another device");
    c.dof = m->jnt_dofadr[c.jnt];
    c.qadr = m->jnt_qposadr[c.jnt];
    c.period = period_of(n, "cmd_hz", c.cmd_hz, "command sample");
    const double dly = c.latency_s / m->opt.timestep;
    c.delay = static_cast<int>(std::lround(dly));
    if (std::abs(dly - c.delay) > 1e-9 * std::max(1.0, dly))
      fail(n, "latency_s " + std::to_string(c.latency_s) + " gives " + std::to_string(dly) + " ticks at the model timestep; need an integer");
    if (c.delay / c.period > kQddRing - 2)
      fail(n, "latency_s is more than " + std::to_string(kQddRing - 2) + " command periods");
  }
  for (FootContactCfg& c : dv.foot.cfg) {
    c.gid = mj_name2id(m, mjOBJ_GEOM, c.geom.c_str());
    if (c.gid < 0) fail(c.meta.name, "geom '" + c.geom + "' not found in the model");
    c.period = period_of(c.meta.name, "rate_hz", c.rate_hz, "sample");
  }
  // Only after every check passed: model edits. SET (not added), so rebinding a reloaded / recompiled model is
  // idempotent. The rotor inertia of an ESC motor is the dof armature (N = 1); a servo reflects it to the gear output:
  // J_rotor N^2, large next to the load.
  for (const EscCfg& c : dv.esc.cfg) {
    m->dof_armature[c.dof] = c.motor.inertia;
    m->dof_damping[c.dof] = 0;  // owned by the device from here on (MJCF damping on this joint is ignored)
  }
  for (const ServoCfg& c : dv.servo.cfg) {
    m->dof_armature[c.dof] = c.motor.inertia * c.n * c.n;
    m->dof_damping[c.dof] = 0;
  }
  // Precomputed constants hold the MJCF armature: a "simple" dof (one hinge on a body with no other dofs) reads its
  // inertia from dof_M0, and dof_invweight0 is MuJoCo's reference weight of limit / friction constraints (with the
  // stale value a servo's joint limit or backlash stop came out ~60x too soft). Refresh every qpos0-derived constant
  // with mj_setConst on a scratch mjData: a pure function of the model, so idempotent, exactly what compile does.
  for (const QddCfg& c : dv.qdd.cfg) {  // like a servo: rotor inertia reflected to the output; the FOC loop owns the damping (none)
    m->dof_armature[c.dof] = c.motor.inertia * c.n * c.n;
    m->dof_damping[c.dof] = 0;
  }
  if (!dv.esc.cfg.empty() || !dv.servo.cfg.empty() || !dv.qdd.cfg.empty()) {
    mjData* scratch = mj_makeData(m);
    if (!scratch) throw std::runtime_error("dvsim.devices: mj_makeData failed while refreshing model constants");
    mj_setConst(m, scratch);
    mj_deleteData(scratch);
  }
}

// ---- tick ----
namespace {
// Motor stage of one drive on `dof`, shared by the ESC (n = 1, eta = 1) and the servo (n = gear ratio, eta = efficiency).
// Reads s.{omega (motor speed), temp, duty, v, i_m} of this tick's solve, writes s.{b, drive, torque, temp}. Damping split
// (stability): the back-EMF damping goes into dof_damping (integrated implicitly by mj_step2's Euler), only the driving term
// is an explicit applied torque. With tau_m = Kt I_m - tau_f, tau_f = Kt I0 tanh(w_m / w_s), and the gear loss factor k
// (below), decided from the state at tick start:
//   b = k Kt Ke n^2 / R,   applied = k n (Kt d V / R) - k n tau_f,   so  -b w_out + applied = k n tau_m = s.torque  exactly.
// k = eta while the motor delivers power (tau_m w_out >= 0; w = 0 counts as driving), 1 when back-driven. Strictly that
// switch is discontinuous at w = 0, and against a stop (stall, w ~ 0) it chatters: k flips every tick with the sign of the
// noise velocity (measured: +-0.04 rad/s, torque dithering between N eta tau_m and N tau_m). So the back-driven side is
// ramped in over the friction smoothing speed: g = tanh(w_m / w_s) sign(tau_m), k = eta for g >= 0 else
// eta + (1 - eta)(-g). Equal to the sharp rule for |w_m| >> w_s and to the driving value at w = 0. Fidelity: a modelling
// regularisation, not a measured gearbox efficiency curve. The ESC passes eta = 1: k = 1.
// The torque acts on the joint, so its reaction lands on the parent body (stator / housing). Open phase: b = 0, no drive;
// friction (bearings + iron loss) stays. Thermal: explicit Euler, C_th dT/dt = I_m^2 R(T) - (T - T_amb)/R_th, R(T) of the
// tick start.
template <class S>
void drive_motor(const MotorCfg& mc, mjModel* m, mjData* d, int dof, double n, double eta, bool open, S& s, double dt,
                 double r_on = 0.0) {  // r_on: ESC FET resistance in series (mechanical side uses R + r_on, heating R only)
  const double kt = mc.ke(), r = mc.r_at(s.temp), rt = r + r_on;
  const double fric = mc.loss(s.omega);
  const double dq = d->qvel[dof];  // dof_damping was 0 in step1: add the whole -b*qvel to the passive force arrays
  const double tau_m = kt * s.i_m - fric;
  const double g = std::tanh(s.omega / mc.w_s) * (tau_m > 0 ? 1.0 : tau_m < 0 ? -1.0 : 0.0);  // > 0: motor delivers power
  const double k = g >= 0 ? eta : eta + (1.0 - eta) * -g;
  s.b = open ? 0.0 : k * kt * mc.ke() * n * n / rt;
  s.drive = open ? 0.0 : k * n * kt * s.duty * s.v / rt;
  s.torque = k * n * (kt * s.i_m - fric);
  d->qfrc_applied[dof] += s.drive - k * n * fric;
  d->qfrc_passive[dof] -= s.b * dq;
  d->qfrc_damper[dof] -= s.b * dq;
  m->dof_damping[dof] = s.b;  // same b for mj_step2's implicit damping
  s.temp += dt / mc.c_th * (s.i_m * s.i_m * r - (s.temp - mc.t_amb) / mc.r_th);
}
}  // namespace

// Servo jam (F3 servo_stall): a seized gearbox as a zero-width joint limit at the latched angle. A MuJoCo limit is a soft
// constraint solved implicitly, so it is stable at any joint inertia (the old explicit spring 200 N m/rad + damper diverged
// below J ~ 5e-6 kg m^2 at dt 1 ms, e.g. the fold drone's SCS0009 tilt). Its stiffness scales with the effective inertia
// (time constant 2 dt, solimp 0.999..0.9999): the hold has no breakaway torque, it deflects ~ a_load * (1 - d) / k_ref.
static void jam_limits(mjModel* m, const mjData* d, Devices& dv) {
  for (std::size_t i = 0; i < dv.servo.cfg.size(); ++i) {
    const ServoCfg& c = dv.servo.cfg[i];
    ServoState& s = dv.servo.st[i];
    mjtNum *rng = m->jnt_range + 2 * c.jnt, *ref = m->jnt_solref + mjNREF * c.jnt, *imp = m->jnt_solimp + mjNIMP * c.jnt;
    if (s.jam > 0.5) {
      if (!(s.jammed > 0.5)) s.q_lock = d->qpos[c.qadr], s.jammed = 1;
      m->jnt_limited[c.jnt] = 1;
      rng[0] = rng[1] = s.q_lock;
      ref[0] = 2 * m->opt.timestep, ref[1] = 1;
      const double jam_imp[5] = {0.999, 0.9999, 0.001, 0.5, 2};
      mju_copy(imp, jam_imp, 5);
    } else {  // canonical every tick, like dof_damping: a restore / clone leaves the (unsaved) model arrays as they were
      s.jammed = 0;
      m->jnt_limited[c.jnt] = static_cast<mjtByte>(c.lim0);
      mju_copy(rng, c.range0, 2), mju_copy(ref, c.solref0, 2), mju_copy(imp, c.solimp0, 5);
    }
  }
}

// Order of device phases (fixed): sensors-read -> controllers -> electrical solve -> motors -> aero/forces -> mj_step2
// -> post-step hook (sensors sample). Sensors-read: the held samples already sit in the state; nothing consumes them yet.
void step(mjModel* m, mjData* d, Devices& dv) {
  // dof_damping of a driven dof is overwritten every tick: zero it BEFORE step1 so qfrc_passive is computed with a canonical
  // b_old = 0 whatever a restore / clone left in the (unsaved) model array. Restored runs are then bitwise identical.
  for (const EscCfg& c : dv.esc.cfg) m->dof_damping[c.dof] = 0;
  for (const ServoCfg& c : dv.servo.cfg) m->dof_damping[c.dof] = 0;
  jam_limits(m, d, dv);  // before step1: the limit rows are built in mj_fwdPosition
  prepare_motors(m, dv);  // burn state -> mass / inertia of the motor bodies, before step1 builds the composite inertias
  mj_step1(m, d);
  mju_zero(d->qfrc_applied, m->nv);  // devices own the applied forces entirely
  mju_zero(d->xfrc_applied, 6 * m->nbody);
  const double dt = m->opt.timestep;

  // 1. sensors-read (held samples: nothing to do)   2. controllers: servo encoder + firmware.
  step_firmware(m, d, dv);

  // 3. electrical solve, with the shaft speeds of the current state (servo: motor side, N x output).
  for (std::size_t e = 0; e < dv.esc.cfg.size(); ++e) dv.esc.st[e].omega = d->qvel[dv.esc.cfg[e].dof];
  for (std::size_t i = 0; i < dv.servo.cfg.size(); ++i) {
    dv.servo.st[i].velocity = d->qvel[dv.servo.cfg[i].dof];
    dv.servo.st[i].omega = dv.servo.cfg[i].n * dv.servo.st[i].velocity;
  }
  solve_electrical(dv, dt);
  step_qdd(m, d, dv);  // QDD actuators: FOC limits at this tick's bus voltage, torque, thermal, bus current for the next solve

  // 4. motors (drive_motor above).
  for (std::size_t e = 0; e < dv.esc.cfg.size(); ++e)
    drive_motor(dv.esc.cfg[e].motor, m, d, dv.esc.cfg[e].dof, 1.0, 1.0, dv.esc.st[e].mode == 1, dv.esc.st[e], dt,
                dv.esc.cfg[e].r_on);
  for (std::size_t i = 0; i < dv.servo.cfg.size(); ++i) {
    const ServoCfg& c = dv.servo.cfg[i];
    drive_motor(c.motor, m, d, c.dof, c.n, c.eta, servo_open(dv.servo.st[i]), dv.servo.st[i], dt);
    d->qfrc_applied[c.dof] += dv.servo.st[i].ext_torque;  // F3 fault: external torque on the output joint (a jam: jam_limits)
  }

  // 5. aero / forces: environment (turbulence), rotors, body drag.
  step_forces(m, d, dv);
  step_launchers(m, d, dv);  // camera pose, muzzle, pellets (kinematics of step1)
  step_motors(m, d, dv);     // solid rocket motors (thrust at the nozzle sites)
  step_airframes(m, d, dv);  // rocket airframe aerodynamics
  step_vanes(m, d, dv);      // jet vanes in the rotors' exhaust (after the rotor forces of this tick)
  step_vessels(m, d, dv);    // LAST force phase: needs every force of the tick for the added-mass solve; then the sea clock advances
  if (!dv.imu.cfg.empty()) dv.qvel_k.assign(d->qvel, d->qvel + m->nv);  // the IMU acceleration needs s_k's qvel
  mj_step2(m, d);

  // 6. post-step hook: sensors sample the derived data of s_k (sensors.cpp).
  sample_sensors(m, d, dv);
  sample_ranges(m, d, dv);
  sample_feet(m, d, dv);
}

// ---- state block ----
namespace {
template <class D, class F> void for_groups(D& dv, F&& f) {
  f(dv.battery);
  f(dv.psu);
  f(dv.wire);
  f(dv.load);
  f(dv.esc);
  f(dv.rotor);
  f(dv.env);
  f(dv.drag);
  f(dv.servo);
  f(dv.imu);
  f(dv.baro);
  f(dv.mag);
  f(dv.gnss);
  f(dv.launcher);
  f(dv.camera);
  f(dv.motor);
  f(dv.airframe);
  f(dv.sea);
  f(dv.vessel);
  f(dv.range);
  f(dv.vane);
  f(dv.wfield);
  f(dv.rain);
  f(dv.qdd);
  f(dv.foot);
  f(dv.lx);
}
template <class G> constexpr bool pod_ok = std::is_trivially_copyable_v<typename G::State> && sizeof(typename G::State) % 8 == 0;
}  // namespace

std::size_t state_bytes(const Devices& dv) {
  std::size_t n = 0;
  for_groups(dv, [&](const auto& g) {
    static_assert(pod_ok<std::decay_t<decltype(g)>>);
    n += g.st.size() * sizeof(typename std::decay_t<decltype(g)>::State);
  });
  return n;
}
void save_state(const Devices& dv, std::byte* out) {
  for_groups(dv, [&](const auto& g) {
    const std::size_t n = g.st.size() * sizeof(typename std::decay_t<decltype(g)>::State);
    if (n) std::memcpy(out, g.st.data(), n);
    out += n;
  });
}
void load_state(Devices& dv, const std::byte* in) {
  for_groups(dv, [&](auto& g) {
    const std::size_t n = g.st.size() * sizeof(typename std::decay_t<decltype(g)>::State);
    if (n) std::memcpy(g.st.data(), in, n);
    in += n;
  });
}

// ---- prim attributes ----
namespace {
// rw: settable input, valid in [lo, hi] (bin: exactly 0 or 1). SI units; temperature is in degC.
template <class S> struct Attr {
  const char* name; const char* unit; double S::* f;
  bool rw = false; bool bin = false; double lo = 0, hi = 0;
};
constexpr Attr<BatteryState> kBattery[] = {{"voltage", "V", &BatteryState::v}, {"current", "A", &BatteryState::i},
                                           {"soc", "1", &BatteryState::soc}, {"ocv", "V", &BatteryState::ocv},
                                           {"v_rc", "V", &BatteryState::v_rc},
                                           // F3 fault inputs (0 = healthy): see the state struct comments in devices.h
                                           {"r_fault", "ohm", &BatteryState::r_fault, true, false, 0, 10},
                                           {"ocv_offset", "V", &BatteryState::ocv_offset, true, false, -100, 100}};
constexpr Attr<PsuState> kPsu[] = {{"voltage", "V", &PsuState::v}, {"current", "A", &PsuState::i},
                                   {"limited", "1", &PsuState::limited}, {"enable", "1", &PsuState::enable, true, true, 0, 1}};
constexpr Attr<WireState> kWire[] = {{"current", "A", &WireState::i}, {"voltage", "V", &WireState::v}};
constexpr Attr<LoadState> kLoad[] = {{"voltage", "V", &LoadState::v}, {"current", "A", &LoadState::i}};
constexpr Attr<EscState> kEsc[] = {
    {"throttle", "1", &EscState::throttle, true, false, -1, 1}, {"duty", "1", &EscState::duty},
    {"current", "A", &EscState::i_m}, {"bus_current", "A", &EscState::i_bus}, {"voltage", "V", &EscState::v},
    {"omega", "rad/s", &EscState::omega}, {"torque", "N*m", &EscState::torque},
    {"temperature", "degC", &EscState::temp}, {"fault_open", "1", &EscState::fault_open, true, true, 0, 1}};

constexpr Attr<RotorState> kRotor[] = {
    {"omega", "rad/s", &RotorState::omega}, {"thrust", "N", &RotorState::thrust}, {"torque", "N*m", &RotorState::torque},
    {"advance_ratio", "1", &RotorState::advance_ratio}, {"ge_ratio", "1", &RotorState::ge_ratio},
    {"rho", "kg/m^3", &RotorState::rho}, {"damage", "1", &RotorState::damage, true, false, 0, 1}};
constexpr Attr<ServoState> kServo[] = {
    {"goal_position", "rad", &ServoState::goal_position, true, false, -1e6, 1e6},
    {"goal_speed", "rad/s", &ServoState::goal_speed, true, false, 0, 1e4},
    {"torque_enable", "1", &ServoState::torque_enable, true, true, 0, 1},
    {"position", "rad", &ServoState::position}, {"counts", "1", &ServoState::counts},
    {"setpoint", "rad", &ServoState::setpoint}, {"duty", "1", &ServoState::duty},
    {"current", "A", &ServoState::i_m}, {"bus_current", "A", &ServoState::i_bus}, {"voltage", "V", &ServoState::v},
    {"torque", "N*m", &ServoState::torque}, {"temperature", "degC", &ServoState::temp},
    {"backlash", "rad", &ServoState::backlash},
    // Runtime registers (settable; the Feetech port maps its register file onto them) and feedback. max_torque /
    // torque_limit: 0.1 % of full duty, effective limit = the smaller. load: effective duty of the last solve (fraction
    // of full duty, sign = direction, + = drive towards increasing joint angle), i.e. Present_Load / 1000.
    {"p_gain", "1", &ServoState::p, true, false, 0, 1e4}, {"i_gain", "1", &ServoState::i, true, false, 0, 1e4},
    {"d_gain", "1", &ServoState::d, true, false, 0, 1e4}, {"deadband", "counts", &ServoState::deadband, true, false, 0, 1e4},
    {"max_torque", "0.1%", &ServoState::max_torque, true, false, 0, 1000},
    {"torque_limit", "0.1%", &ServoState::torque_limit, true, false, 0, 1000},
    {"min_voltage", "V", &ServoState::min_voltage, true, false, 0, 1e3},
    {"max_voltage", "V", &ServoState::max_voltage, true, false, 0, 1e3},
    {"max_temp", "degC", &ServoState::max_temp, true, false, 0, 1e3},
    {"overload_load", "1", &ServoState::overload_load, true, false, 0, 1},
    {"overload_time", "s", &ServoState::overload_time, true, false, 0, 1e4},
    {"unload_mask", "1", &ServoState::unload_mask, true, false, 0, 255},  // integer bit mask; fraction truncated
    {"velocity", "rad/s", &ServoState::velocity}, {"load", "1", &ServoState::duty}, {"alarm", "1", &ServoState::alarm},
    {"ext_torque", "N*m", &ServoState::ext_torque, true, false, -100, 100}, {"jam", "1", &ServoState::jam, true, true, 0, 1},
    // Motor-controller firmware (A4, servo.cpp): mode 0 position / 1 duty / 2 velocity; counts are encoder counts.
    {"mode", "1", &ServoState::mode, true, false, 0, 2}, {"goal_duty", "1", &ServoState::goal_duty, true, false, -1, 1},
    {"duty_accel", "1/s", &ServoState::duty_accel, true, false, 0, 1e6},
    {"goal_velocity", "counts/s", &ServoState::goal_velocity, true, false, -1e8, 1e8},
    {"vel_accel", "counts/s^2", &ServoState::vel_accel, true, false, 0, 1e10},
    {"qpps", "counts/s", &ServoState::qpps, true, false, 0, 1e8}, {"vel_p", "1", &ServoState::vel_p, true, false, 0, 1e6},
    {"vel_i", "1/s", &ServoState::vel_i, true, false, 0, 1e6}, {"vel_d", "s", &ServoState::vel_d, true, false, 0, 1e6},
    {"speed", "counts/s", &ServoState::vmeas}, {"speed_raw", "counts/s", &ServoState::vraw},
    {"speed_setpoint", "counts/s", &ServoState::vsp}};
constexpr Attr<EnvState> kEnv[] = {{"density", "kg/m^3", &EnvState::density}};
// 3-vector attrs (same fields as Attr, member is a double[3]).
template <class S> struct Attr3 {
  const char* name; const char* unit; double (S::*f)[3];
  bool rw = false; double lo = 0, hi = 0;
};
constexpr Attr3<EnvState> kEnv3[] = {{"wind", "m/s", &EnvState::wind, true, -200, 200},
                                     {"turbulence", "m/s", &EnvState::turb}};
constexpr Attr3<DragState> kDrag3[] = {{"force", "N", &DragState::force}};
// Sensors: the held sample (all read-only).
constexpr Attr3<ImuState> kImu3[] = {{"gyro", "rad/s", &ImuState::gyro}, {"accel", "m/s^2", &ImuState::accel},
                                     // current bias (config + walk): writing it is a bias step (F3 imu_bias_step)
                                     {"gyro_bias", "rad/s", &ImuState::gyro_bias, true, -10, 10},
                                     {"accel_bias", "m/s^2", &ImuState::accel_bias, true, -100, 100}};
constexpr Attr<BaroState> kBaro[] = {{"pressure", "Pa", &BaroState::pressure},
                                     {"bias_extra", "Pa", &BaroState::bias_extra, true, false, -1e5, 1e5},
                                     {"drift_rate", "Pa/s", &BaroState::drift_rate, true, false, -1e3, 1e3}};
constexpr Attr3<MagState> kMag3[] = {{"field", "T", &MagState::field}};
constexpr Attr<GnssState> kGnss[] = {{"lat", "deg", &GnssState::lat}, {"lon", "deg", &GnssState::lon},
                                     {"alt", "m", &GnssState::alt}};
constexpr Attr<GnssState> kGnssIn[] = {{"fix_lost", "1", &GnssState::fix_lost, true, true, 0, 1}};  // F3 gnss_outage
constexpr Attr3<GnssState> kGnss3[] = {{"vel_ned", "m/s", &GnssState::vel_ned},
                                       {"jump", "m", &GnssState::jump, true, -1e5, 1e5}};  // F3 gnss_jump: ENU position step

// Launcher (counter-UAS): scalars and 3-vectors here, the pool / hit-ring arrays in locate(). fire is the only input.
constexpr Attr<LauncherState> kLauncher[] = {
    {"fire", "1", &LauncherState::fire, true, true, 0, 1}, {"rounds", "1", &LauncherState::rounds},
    {"hits", "1", &LauncherState::hits}, {"ground", "1", &LauncherState::ground}, {"expired", "1", &LauncherState::expired},
    {"dropped", "1", &LauncherState::dropped}, {"hit_n", "1", &LauncherState::hit_n}, {"time", "s", &LauncherState::time}};
constexpr Attr3<LauncherState> kLauncher3[] = {{"muzzle_pos", "m", &LauncherState::muzzle_pos},
                                              {"muzzle_dir", "1", &LauncherState::muzzle_dir}};
// Launcher per-shot inputs (cuas2; only on launchers with cl_k / spin_tau): see LauncherXState.
constexpr Attr<LauncherXState> kLauncherX[] = {
    {"muzzle_velocity", "m/s", &LauncherXState::muzzle_velocity, true, false, 0, 200},
    {"spin_rps", "1/s", &LauncherXState::spin_rps, true, false, -2000, 2000},
    {"dir_offset_x", "rad", &LauncherXState::dir_offset_x, true, false, -0.5, 0.5},
    {"dir_offset_y", "rad", &LauncherXState::dir_offset_y, true, false, -0.5, 0.5}};
constexpr Attr3<CameraState> kCamera3[] = {{"pos", "m", &CameraState::pos}, {"fwd", "1", &CameraState::fwd},
                                          {"up", "1", &CameraState::up}};

// Phase 5. Solid motor: `ignite` is the only command (consumed by the next tick); phase 0 idle / 1 delay / 2 burning / 3 burnt out.
constexpr Attr<SolidMotorState> kMotor[] = {
    {"ignite", "1", &SolidMotorState::ignite, true, true, 0, 1}, {"thrust_scale", "1", &SolidMotorState::thrust_scale, true, false, 0, 2},
    {"phase", "1", &SolidMotorState::phase}, {"time", "s", &SolidMotorState::t_ign}, {"burn_time", "s", &SolidMotorState::t_burn},
    {"impulse", "N*s", &SolidMotorState::impulse}, {"progress", "1", &SolidMotorState::progress},
    {"burned_mass", "kg", &SolidMotorState::burned}, {"mass", "kg", &SolidMotorState::mass}, {"thrust", "N", &SolidMotorState::thrust}};
constexpr Attr3<SolidMotorState> kMotor3[] = {{"pos", "m", &SolidMotorState::pos}, {"dir", "1", &SolidMotorState::dir}};
constexpr Attr<AirframeState> kAirframe[] = {{"alpha", "rad", &AirframeState::alpha}, {"speed", "m/s", &AirframeState::speed}};
constexpr Attr3<AirframeState> kAirframe3[] = {{"force", "N", &AirframeState::force}};
constexpr Attr<JetVaneState> kVane[] = {{"alpha", "rad", &JetVaneState::alpha}, {"jet_speed", "m/s", &JetVaneState::speed}};
constexpr Attr3<JetVaneState> kVane3[] = {{"force", "N", &JetVaneState::force}};
constexpr Attr<SeaState> kSea[] = {{"time", "s", &SeaState::time}, {"wave_scale", "1", &SeaState::wave_scale, true, false, 0, 10},
                                   {"eta_origin", "m", &SeaState::eta0}, {"n_waves", "1", &SeaState::n_waves}};
constexpr Attr3<SeaState> kSea3[] = {{"current", "m/s", &SeaState::current, true, -20, 20}};
constexpr Attr<VesselState> kVessel[] = {{"submerged", "m^3", &VesselState::submerged}, {"wetted", "1", &VesselState::wetted},
                                         {"eta", "m", &VesselState::eta}};
constexpr Attr3<VesselState> kVessel3[] = {{"buoyancy", "N", &VesselState::buoyancy}, {"hydro", "N", &VesselState::hydro}};
constexpr Attr<WindState> kWind[] = {{"time", "s", &WindState::t}, {"gust_clear", "1", &WindState::gust_clear, true, true, 0, 1},
                                     {"n_gusts", "1", &WindState::n_gusts}, {"height_agl", "m", &WindState::hag},
                                     {"ground", "m", &WindState::ground}, {"lee", "1", &WindState::lee}};
constexpr Attr3<WindState> kWind3[] = {{"wind_ref", "m/s", &WindState::wind_ref}};
constexpr Attr<RangefinderState> kRange[] = {{"frames", "1", &RangefinderState::frames}, {"valid", "1", &RangefinderState::valid},
                                             {"t", "s", &RangefinderState::t}};

// Rain: `rate` is the input (mm/h, 0 = dry); the rest are outputs of the last tick.
constexpr Attr<RainState> kRain[] = {{"rate", "mm/h", &RainState::rate, true, false, 0, 500}, {"lwc", "g/m^3", &RainState::lwc},
                                     {"v_fall", "m/s", &RainState::v_fall}, {"film_mass", "kg", &RainState::film_mass},
                                     {"thrust_loss", "1", &RainState::thrust_loss}};
constexpr Attr3<RainState> kRain3[] = {{"force", "N", &RainState::force}};

// Qdd (qdd.cpp): MIT-mode command inputs, encoder / torque / thermal outputs. Setting any input stamps the CAN frame time (set()).
constexpr Attr<QddState> kQdd[] = {
    {"p_des", "rad", &QddState::p_des, true, false, -1e6, 1e6}, {"v_des", "rad/s", &QddState::v_des, true, false, -1e4, 1e4},
    {"kp", "N*m/rad", &QddState::kp, true, false, 0, 1e5}, {"kd", "N*m*s/rad", &QddState::kd, true, false, 0, 1e4},
    {"tau_ff", "N*m", &QddState::tau_ff, true, false, -1e4, 1e4}, {"enable", "1", &QddState::enable, true, true, 0, 1},
    {"q", "rad", &QddState::q}, {"dq", "rad/s", &QddState::dq}, {"tau_est", "N*m", &QddState::tau_est},
    {"tau_cmd", "N*m", &QddState::tau_cmd}, {"i_q", "A", &QddState::i_q}, {"temp", "degC", &QddState::temp},
    {"fault", "1", &QddState::fault}, {"bus_i", "A", &QddState::bus_i}, {"voltage", "V", &QddState::vbus},
    {"derate", "1", &QddState::derate}, {"timeout", "1", &QddState::timeout}};
constexpr Attr<FootContactState> kFoot[] = {{"force", "N", &FootContactState::force}, {"contact", "1", &FootContactState::contact}};

template <class P> struct Found { P p; const char* unit; bool rw, bin; double lo, hi; std::size_t n = 1; };
template <class D> auto locate(D& dv, int slot, const std::string& attr) {
  using P = std::conditional_t<std::is_const_v<D>, const double*, double*>;
  Found<P> f{nullptr, "", false, false, 0, 0};
  const Devices::Slot s = dv.slots[static_cast<std::size_t>(slot)];
  auto pick = [&](const auto& tab, auto& states) {
    for (const auto& a : tab)
      if (attr == a.name) f = {&(states[static_cast<std::size_t>(s.i)].*(a.f)), a.unit, a.rw, a.bin, a.lo, a.hi};
  };
  auto pick3 = [&](const auto& tab, auto& states) {
    for (const auto& a : tab)
      if (attr == a.name) f = {(states[static_cast<std::size_t>(s.i)].*(a.f)), a.unit, a.rw, false, a.lo, a.hi, 3};
  };
  switch (s.type) {
    case Devices::Slot::battery: pick(kBattery, dv.battery.st); break;
    case Devices::Slot::psu: pick(kPsu, dv.psu.st); break;
    case Devices::Slot::wire: pick(kWire, dv.wire.st); break;
    case Devices::Slot::load: pick(kLoad, dv.load.st); break;
    case Devices::Slot::esc: pick(kEsc, dv.esc.st); break;
    case Devices::Slot::rotor: pick(kRotor, dv.rotor.st); break;
    case Devices::Slot::env: pick(kEnv, dv.env.st); pick3(kEnv3, dv.env.st); break;
    case Devices::Slot::drag: pick3(kDrag3, dv.drag.st); break;
    case Devices::Slot::servo: pick(kServo, dv.servo.st); break;
    case Devices::Slot::imu: pick3(kImu3, dv.imu.st); break;
    case Devices::Slot::baro: pick(kBaro, dv.baro.st); break;
    case Devices::Slot::mag: pick3(kMag3, dv.mag.st); break;
    case Devices::Slot::gnss: pick(kGnss, dv.gnss.st); pick(kGnssIn, dv.gnss.st); pick3(kGnss3, dv.gnss.st); break;
    case Devices::Slot::launcher: {
      pick(kLauncher, dv.launcher.st);
      pick3(kLauncher3, dv.launcher.st);
      auto& st = dv.launcher.st[static_cast<std::size_t>(s.i)];
      if (const int lx = dv.launcher.cfg[static_cast<std::size_t>(s.i)].lx; lx >= 0)
        for (const auto& a : kLauncherX)
          if (attr == a.name) f = {&(dv.lx.st[static_cast<std::size_t>(lx)].*(a.f)), a.unit, a.rw, a.bin, a.lo, a.hi};
      auto arr = [&](const char* name, const char* unit, auto& a) {
        if (attr == name) f = {a, unit, false, false, 0, 0, std::size(a)};
      };
      arr("active", "1", st.active); arr("round_no", "1", st.round_no); arr("age", "s", st.age);
      arr("pos", "m", st.pos); arr("vel", "m/s", st.vel);
      arr("hit_body", "1", st.hit_body); arr("hit_geom", "1", st.hit_geom); arr("hit_t", "s", st.hit_t);
      arr("hit_speed", "m/s", st.hit_v); arr("hit_round", "1", st.hit_round); arr("hit_pos", "m", st.hit_pos);
      break;
    }
    case Devices::Slot::camera: pick3(kCamera3, dv.camera.st); break;
    case Devices::Slot::motor: pick(kMotor, dv.motor.st); pick3(kMotor3, dv.motor.st); break;
    case Devices::Slot::airframe: pick(kAirframe, dv.airframe.st); pick3(kAirframe3, dv.airframe.st); break;
    case Devices::Slot::sea: {
      pick(kSea, dv.sea.st);
      pick3(kSea3, dv.sea.st);
      auto& st = dv.sea.st[static_cast<std::size_t>(s.i)];
      if (attr == "waves") f = {st.waves, "m,rad/m,rad/s,rad,rad", false, false, 0, 0, static_cast<std::size_t>(5 * st.n_waves)};
      break;
    }
    case Devices::Slot::vessel: pick(kVessel, dv.vessel.st); pick3(kVessel3, dv.vessel.st); break;
    case Devices::Slot::vane: pick(kVane, dv.vane.st); pick3(kVane3, dv.vane.st); break;
    case Devices::Slot::wfield: {
      pick(kWind, dv.wfield.st);
      pick3(kWind3, dv.wfield.st);
      auto& st = dv.wfield.st[static_cast<std::size_t>(s.i)];
      // gust: [amplitude m/s, gradient distance H m, direction x y z, delay s, propagation speed m/s]; consumed by the next tick
      if (attr == "gust") f = {st.gust_in, "m/s,m,1,1,1,s,m/s", true, false, -1e4, 1e4, std::size(st.gust_in)};
      if (attr == "gusts") f = {&st.gust[0][0], "m/s,m,1,1,1,s,m/s,1,1", false, false, 0, 0, 9 * kMaxGusts};
      break;
    }
    case Devices::Slot::rain: pick(kRain, dv.rain.st); pick3(kRain3, dv.rain.st); break;
    case Devices::Slot::qdd: pick(kQdd, dv.qdd.st); break;
    case Devices::Slot::foot: pick(kFoot, dv.foot.st); break;
    case Devices::Slot::range: {
      pick(kRange, dv.range.st);
      auto& img = dv.range_img[static_cast<std::size_t>(s.i)];
      if (attr == "range") f = {img.data(), "m", false, false, 0, 0, img.size()};
      break;
    }
  }
  return f;
}
std::string path_of(const Devices& dv, int slot) { return "/devices/" + dv.name(slot); }
}  // namespace

Result get(const Devices& dv, int slot, const std::string& attr, std::vector<double>& out) {
  const auto f = locate(dv, slot, attr);
  if (!f.p) return {Result::not_found, "unknown attr '" + attr + "' on " + path_of(dv, slot)};
  out.assign(f.p, f.p + f.n);
  return {};
}

Result set(Devices& dv, int slot, const std::string& attr, std::span<const double> values) {
  const auto f = locate(dv, slot, attr);
  if (!f.p) return {Result::not_found, "unknown attr '" + attr + "' on " + path_of(dv, slot)};
  if (!f.rw) return {Result::invalid_argument, "attr '" + attr + "' is read-only"};
  if (values.size() != f.n)
    return {Result::invalid_argument, "attr '" + attr + "' takes " + std::to_string(f.n) + " value(s), got " + std::to_string(values.size())};
  const Devices::Slot sl = dv.slots[static_cast<std::size_t>(slot)];
  const bool forward_only = sl.type == Devices::Slot::esc && attr == "throttle" && !dv.esc.cfg[static_cast<std::size_t>(sl.i)].bidirectional;
  for (const double v : values) {  // validate everything before writing anything
    if (!std::isfinite(v)) return {Result::invalid_argument, "values must be finite"};
    if (forward_only && v < 0) return {Result::invalid_argument, "attr 'throttle' must be in [0, 1] (the ESC is not bidirectional)"};
    if (f.bin ? (v != 0 && v != 1) : (v < f.lo || v > f.hi))
      return {Result::invalid_argument, "attr '" + attr + "' must be " + (f.bin ? "0 or 1" : "in [" + std::to_string(f.lo) + ", " + std::to_string(f.hi) + "]")};
  }
  std::copy(values.begin(), values.end(), f.p);
  if (sl.type == Devices::Slot::qdd)  // every settable attr of a Qdd is a command field: the write is a received CAN frame
    dv.qdd.st[static_cast<std::size_t>(sl.i)].rx_tick = static_cast<double>(dv.qdd.st[static_cast<std::size_t>(sl.i)].ticks);
  return {};
}

std::string unit(const Devices& dv, int slot, const std::string& attr) { return locate(dv, slot, attr).unit; }

}  // namespace devs
