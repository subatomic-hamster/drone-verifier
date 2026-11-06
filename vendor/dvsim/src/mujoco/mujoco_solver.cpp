#include "mujoco/mujoco_solver.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace {
constexpr int kStateSig = mjSTATE_INTEGRATION;

[[noreturn]] void on_error(const char* msg) { throw std::runtime_error(std::string("mujoco: ") + msg); }
// Not throwing: divergence (mjWARN_BADQACC etc.) is detected from d->warning counters in step();
// a throwing handler would abort mid-step instead.
void on_warning(const char* msg) { std::fprintf(stderr, "mujoco warning: %s\n", msg); }
}  // namespace

void install_mujoco_handlers() {  // call_once: Python threads load models concurrently
  static std::once_flag once;
  std::call_once(once, [] {
    mju_user_error = on_error;
    mju_user_warning = on_warning;
  });
}

namespace {
// mjSpec owner (null-safe: a shared_ptr runs its deleter even on a null pointer).
std::shared_ptr<mjSpec> own(mjSpec* s) {
  return {s, [](mjSpec* p) { if (p) mj_deleteSpec(p); }};
}
}  // namespace

// MuJoCo-internal but exported (src/xml/xml_numeric_format.h; thread-local): decimal digits of numbers in saved XML.
// The default (6) is lossy; 17 round-trips doubles exactly. Pinned MuJoCo version, see deps.cmake.
extern "C" {
int _mjPRIVATE__get_xml_precision();
void _mjPRIVATE__set_xml_precision(int precision);
}

std::unique_ptr<Solver> load_mujoco(const std::string& path) {
  install_mujoco_handlers();
  char err[1000] = "";
  std::shared_ptr<mjSpec> spec = own(mj_parseXML(path.c_str(), nullptr, err, sizeof err));  // = mj_loadXML, keeping the spec
  if (!spec) throw std::runtime_error("mj_loadXML failed for '" + path + "': " + err);
  mjModel* m = mj_compile(spec.get(), nullptr);
  if (!m) throw std::runtime_error("mj_loadXML failed for '" + path + "': " + mjs_getError(spec.get()));
  mjs_setDeepCopy(spec.get(), 1);  // attached children are copied into the spec, so they need not outlive it
  mjData* d = mj_makeData(m);  // no thread pool is ever bound: stepping is single-threaded
  if (!d) {
    mj_deleteModel(m);
    throw std::runtime_error("mj_makeData failed for '" + path + "'");
  }
  return std::make_unique<MujocoSolver>(m, d, std::move(spec));
}

// Blob = u64 n | mj_saveModel bytes (n) | spec XML (the rest; may be empty). See replay.h.
std::unique_ptr<Solver> load_mujoco_bytes(std::span<const std::byte> bytes) {
  install_mujoco_handlers();
  std::uint64_t n = 0;
  if (bytes.size() < sizeof n) throw std::runtime_error("model buffer has invalid size");
  std::memcpy(&n, bytes.data(), sizeof n);
  if (n == 0 || n > bytes.size() - sizeof n || n > static_cast<std::uint64_t>(INT_MAX))
    throw std::runtime_error("model buffer has invalid size");
  mjModel* m = mj_loadModelBuffer(bytes.data() + sizeof n, static_cast<int>(n));
  if (!m) throw std::runtime_error("mj_loadModelBuffer failed (corrupt or incompatible model bytes)");
  mjData* d = mj_makeData(m);
  if (!d) {
    mj_deleteModel(m);
    throw std::runtime_error("mj_makeData failed for model buffer");
  }
  // ttsr-ignore: cpp/no-reinterpret-const-cast  (std::byte -> char view of the XML text, same idiom as replay.cpp)
  const auto* xml = reinterpret_cast<const char*>(bytes.data() + sizeof n + n);
  return std::make_unique<MujocoSolver>(m, d, nullptr, std::string(xml, bytes.size() - sizeof n - n));
}

std::vector<std::byte> MujocoSolver::model_bytes() const {
  std::vector<std::byte> b(sizeof(std::uint64_t) + static_cast<std::size_t>(mj_sizeModel(m_)));
  const std::uint64_t n = b.size() - sizeof n;
  std::memcpy(b.data(), &n, sizeof n);
  mj_saveModel(m_, nullptr, b.data() + sizeof n, static_cast<int>(n));
  std::string xml = spec_xml_;  // a spec-less (replayed) solver has not changed it since the header
  if (spec_) {
    // Any failure to write XML only makes Spawn / Delete unreplayable; plain logs are unaffected. The XML is the
    // structure only: SetAttr'd kp/kv come with the header state and are re-synced from the model at each edit.
    xml.clear();
    try {
      std::string buf(1 << 16, '\0');
      char err[1000] = "";
      const int prev = _mjPRIVATE__get_xml_precision();
      _mjPRIVATE__set_xml_precision(17);
      int rc = mj_saveXMLString(spec_.get(), buf.data(), static_cast<int>(buf.size()), err, sizeof err);
      if (rc > 0) {  // buffer too small: rc = required size
        buf.assign(static_cast<std::size_t>(rc) + 1, '\0');
        rc = mj_saveXMLString(spec_.get(), buf.data(), static_cast<int>(buf.size()), err, sizeof err);
      }
      _mjPRIVATE__set_xml_precision(prev);
      if (rc == 0) xml.assign(buf.c_str());
    } catch (const std::exception&) {
    }
  }
  // ttsr-ignore: cpp/no-reinterpret-const-cast  (char -> std::byte view of the XML text, same idiom as replay.cpp)
  const auto* x = reinterpret_cast<const std::byte*>(xml.data());
  b.insert(b.end(), x, x + xml.size());
  return b;
}

MujocoSolver::~MujocoSolver() {
  mj_deleteData(d_);
  mj_deleteModel(m_);
}

// A private, compiled working copy of the spec carrying the model's current parameters (SetAttr'd kp/kv ...). spec_
// itself is never mutated (clones share it). The compile matters: a plain mj_copySpec copy has lost the state
// addresses mj_recompile needs to carry qpos/qvel over (it would silently reset them), and compiling gives them back.
std::shared_ptr<mjSpec> MujocoSolver::working_spec() {
  if (!spec_) {  // replayed solver: the spec comes from the log header's XML
    if (spec_xml_.empty()) throw std::runtime_error("this model's spec is unavailable (the replay log header carries none)");
    char err[1000] = "";
    std::shared_ptr<mjSpec> s = own(mj_parseXMLString(spec_xml_.c_str(), nullptr, err, sizeof err));
    if (!s) throw std::runtime_error(std::string("replay header spec: ") + err);
    mjs_setDeepCopy(s.get(), 1);
    spec_ = std::move(s);
  }
  std::shared_ptr<mjSpec> c = own(mj_copySpec(spec_.get()));
  if (!c) throw std::runtime_error("mj_copySpec failed");
  mjModel* m = mj_compile(c.get(), nullptr);
  if (!m) throw std::runtime_error(std::string("the model spec does not compile: ") + mjs_getError(c.get()));
  const bool same = m->nq == m_->nq && m->nv == m_->nv && m->nu == m_->nu && m->na == m_->na && m->nbody == m_->nbody &&
                    m->njnt == m_->njnt && m->ngeom == m_->ngeom && m->nsensor == m_->nsensor;
  mj_deleteModel(m);
  if (!same) throw std::runtime_error("the model spec does not match the model (replay header?)");
  // The model parameters SetAttr changes (actuator gain / bias: kp, kv; see save()) live only in m_: write them into
  // the spec so the recompile keeps them. (mj_copyBack would not do: it fills compiler-internal copies that the
  // next compile overwrites from the spec.)
  int a = 0;
  for (mjsElement* e = mjs_firstElement(c.get(), mjOBJ_ACTUATOR); e; e = mjs_nextElement(c.get(), e), ++a) {
    if (a >= m_->nu) break;
    mjsActuator* sa = mjs_asActuator(e);
    mju_copy(sa->gainprm, m_->actuator_gainprm + a * mjNGAIN, mjNGAIN);
    mju_copy(sa->biasprm, m_->actuator_biasprm + a * mjNBIAS, mjNBIAS);
  }
  if (a != m_->nu) throw std::runtime_error("the model spec does not match the model (actuators)");
  return c;
}

Result MujocoSolver::restructure(const std::function<Result(mjSpec*)>& edit) {
  std::shared_ptr<mjSpec> cand = working_spec();
  if (Result r = edit(cand.get()); r.code != Result::ok) return r;
  mjModel* m2 = mj_copyModel(nullptr, m_);
  mjData* d2 = mj_copyData(nullptr, m2, d_);
  // Compiles the edited spec into m2 in place and rebuilds d2, carrying qpos/qvel/act/ctrl/time of every element that
  // survives. On failure it has already freed both (so they are not freed here).
  if (mj_recompile(cand.get(), nullptr, m2, d2) != 0)
    return {Result::invalid_argument, std::string("the edited model does not compile: ") + mjs_getError(cand.get())};
  // Devices keep their state; names are re-resolved against the new layout (a spawned model's own dvsim.devices text is
  // ignored: only the root model's text, exact name "dvsim.devices", configures devices). If a device's joint is gone the
  // edit fails here, with the live model, data and devices untouched.
  std::unique_ptr<devs::Devices> dv;
  if (dev_) {
    dv = std::make_unique<devs::Devices>(*dev_);
    try {
      devs::bind(m2, *dv);
    } catch (const std::exception& e) {
      mj_deleteData(d2);
      mj_deleteModel(m2);
      return {Result::invalid_argument, std::string("the edit breaks a device: ") + e.what()};
    }
  }
  dev_ = std::move(dv);
  mj_deleteData(d_);
  mj_deleteModel(m_);
  m_ = m2;
  d_ = d2;
  spec_ = std::move(cand);
  spec_xml_.clear();
  mj_forward(m_, d_);  // derived data (sensors, xpos) of the new layout; warm-start is reset by the recompile
  build_index();
  return {};
}

Result MujocoSolver::spawn(const std::string& model_path, const std::string& at, std::span<const double> pos,
                           std::span<const double> quat) {
  constexpr std::string_view kWorld = "/world/";
  const std::size_t k = at.rfind('/');
  if (at.compare(0, kWorld.size(), kWorld) != 0 || at.size() == kWorld.size() || at.back() == '/')
    return {Result::invalid_argument, "Spawn 'at' must be '<parent body prim>/<name>' under /world, got '" + at + "'"};
  const std::string parent = at.substr(0, k);
  auto p = index_.find(parent);
  if (p == index_.end() || p->second.kind != Ref::body) return {Result::not_found, "no parent body prim '" + parent + "'"};
  if (index_.count(at)) return {Result::invalid_argument, "prim '" + at + "' already exists"};
  if ((!pos.empty() && pos.size() != 3) || (!quat.empty() && quat.size() != 4))
    return {Result::invalid_argument, "Spawn pos takes 3 values and quat 4"};
  for (double v : pos) if (!std::isfinite(v)) return {Result::invalid_argument, "values must be finite"};
  for (double v : quat) if (!std::isfinite(v)) return {Result::invalid_argument, "values must be finite"};
  const char* pname = mj_id2name(m_, mjOBJ_BODY, p->second.id);
  if (!pname || !*pname) return {Result::invalid_argument, "parent body '" + parent + "' has no name to attach under"};
  const std::string parent_name = pname;
  // MuJoCo names are global, so the entity's container body is named by its whole path below /world ("arm2",
  // "pole/arm3"): unique per path, and the children get the prefix "<that>/". build_index() shows only the leaf.
  const std::string name = at.substr(kWorld.size());
  return restructure([&](mjSpec* s) -> Result {
    char err[1000] = "";
    std::shared_ptr<mjSpec> child = own(mj_parseXML(model_path.c_str(), nullptr, err, sizeof err));
    if (!child) return {Result::invalid_argument, "cannot load '" + model_path + "': " + err};
    mjsBody* pb = mjs_findBody(s, parent_name.c_str());
    if (!pb) return {Result::not_found, "no parent body '" + parent_name + "' in the spec"};
    mjsBody* cb = mjs_addBody(pb, nullptr);
    mjs_setName(cb->element, name.c_str());
    std::copy(pos.begin(), pos.end(), cb->pos);
    std::copy(quat.begin(), quat.end(), cb->quat);
    if (!mjs_attach(cb->element, child->element, (name + "/").c_str(), ""))
      return {Result::invalid_argument, "cannot attach '" + model_path + "': " + mjs_getError(s)};
    return {};
  });
}

Result MujocoSolver::remove(const std::string& prim) {
  auto it = index_.find(prim);
  if (it == index_.end()) return {Result::not_found, "unknown prim '" + prim + "'"};
  if (it->second.kind != Ref::body || it->second.id == 0)
    return {Result::invalid_argument, "Delete takes a body prim other than /world, got '" + prim + "'"};
  const char* n = mj_id2name(m_, mjOBJ_BODY, it->second.id);
  if (!n || !*n) return {Result::invalid_argument, "body '" + prim + "' has no name to delete it by"};
  const std::string name = n;
  return restructure([&](mjSpec* s) -> Result {
    mjsBody* b = mjs_findBody(s, name.c_str());
    if (!b || mjs_delete(s, b->element) != 0)
      return {Result::invalid_argument, "cannot delete '" + prim + "': " + (b ? mjs_getError(s) : "not in the spec")};
    return {};
  });
}

int MujocoSolver::step(int n) {
  int done = 0;
  while (done < n && !diverged_) {
    if (dev_) devs::step(m_, d_, *dev_);  // step1 / device phases / step2
    else mj_step(m_, d_);                 // no devices: bit-identical to the pre-N2 behaviour
    ++done;
    if (d_->warning[mjWARN_BADQACC].number > 0 || d_->warning[mjWARN_BADQPOS].number > 0 ||
        d_->warning[mjWARN_BADQVEL].number > 0) {
      diverged_ = true;
    }
    for (int i = 0; i < m_->nq && !diverged_; ++i) diverged_ = !std::isfinite(d_->qpos[i]);
  }
  return done;
}

namespace {
std::string elem_name(const mjModel* m, int type, int id, const char* fallback) {
  const char* n = mj_id2name(m, type, id);
  return n && *n ? std::string(n) : fallback + std::to_string(id);  // unnamed: stable index-based name
}
bool is_world(const std::string& f) { return f.empty() || f == "world"; }
}  // namespace

MujocoSolver::MujocoSolver(mjModel* m, mjData* d, std::shared_ptr<mjSpec> spec, std::string spec_xml)
    : m_(m), d_(d), spec_(std::move(spec)), spec_xml_(std::move(spec_xml)) {
  try {  // every load path (file, bytes, clone) ends here; a bad device config fails the load with its message
    if (const auto js = devs::config_text(m_)) {
      dev_ = std::make_unique<devs::Devices>(devs::parse(*js));
      devs::bind(m_, *dev_);
    }
  } catch (...) {  // the caller handed us ownership of m and d: a throwing constructor never runs the destructor
    mj_deleteData(d_);
    mj_deleteModel(m_);
    throw;
  }
  build_index();
}

void MujocoSolver::build_index() {
  prims_.clear();
  index_.clear();
  std::uint64_t h = 14695981039346656037ull;  // FNV-1a 64 over what a snapshot's layout depends on
  auto mix = [&](std::string_view s) {
    for (char c : s) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    h *= 1099511628211ull;
  };
  auto add = [&](std::string path, const char* kind, Ref::Kind k, int id) {
    mix(path);
    mix(kind);
    index_[path] = {k, id};
    prims_.push_back({std::move(path), kind});
  };
  // A spawned entity's names carry its path as a prefix ("arm2/base"); a body / joint prim shows only the leaf, under
  // its parent: /world/arm2/base. Actuators and sensors keep the full name ("/actuators/arm2/motor").
  auto leaf = [](std::string n) { return n.substr(n.rfind('/') + 1); };
  std::vector<std::string> body_path(static_cast<std::size_t>(m_->nbody));  // parents precede children in body order
  for (int i = 0; i < m_->nbody; ++i) {
    body_path[i] = i == 0 ? "/world" : body_path[m_->body_parentid[i]] + "/" + leaf(elem_name(m_, mjOBJ_BODY, i, "body"));
    add(body_path[i], "body", Ref::body, i);
  }
  for (int j = 0; j < m_->njnt; ++j)
    add(body_path[m_->jnt_bodyid[j]] + "/" + leaf(elem_name(m_, mjOBJ_JOINT, j, "joint")), "joint", Ref::joint, j);
  for (int a = 0; a < m_->nu; ++a)
    add("/actuators/" + elem_name(m_, mjOBJ_ACTUATOR, a, "actuator"), "actuator", Ref::actuator, a);
  for (int s = 0; s < m_->nsensor; ++s)
    add("/sensors/" + elem_name(m_, mjOBJ_SENSOR, s, "sensor"), "sensor", Ref::sensor, s);
  if (dev_)
    for (int i = 0; i < static_cast<int>(dev_->slots.size()); ++i) add("/devices/" + dev_->name(i), "device", Ref::device, i);
  structure_ = h ^ (static_cast<std::uint64_t>(m_->nq) << 40 | static_cast<std::uint64_t>(m_->nv) << 20 | static_cast<std::uint64_t>(m_->na));
}

// A position servo has a fixed gain and an affine bias with biasprm[1] == -gainprm[0] (what <position> compiles to).
// Motors (no bias) and velocity actuators (biasprm[1] == 0) are not servos. kp = 0 is still a servo (and stays one).
bool MujocoSolver::is_servo(int a) const {
  return m_->actuator_gaintype[a] == mjGAIN_FIXED && m_->actuator_biastype[a] == mjBIAS_AFFINE &&
         m_->actuator_biasprm[a * mjNBIAS + 1] == -m_->actuator_gainprm[a * mjNGAIN];
}

// Attribute table for the non-body prims: where the values live in mjData. Body attrs are derived (see get).
bool MujocoSolver::locate(const Ref& r, const std::string& attr, Loc& loc) const {
  switch (r.kind) {
    case Ref::joint: {
      const int type = m_->jnt_type[r.id];
      if (attr == "qpos") {
        loc = {d_->qpos + m_->jnt_qposadr[r.id], type == mjJNT_FREE ? 7 : type == mjJNT_BALL ? 4 : 1, false};
        return true;
      }
      if (attr == "qvel") {
        loc = {d_->qvel + m_->jnt_dofadr[r.id], type == mjJNT_FREE ? 6 : type == mjJNT_BALL ? 3 : 1, false};
        return true;
      }
      return false;
    }
    case Ref::actuator:
      if (attr != "ctrl") return false;
      loc = {d_->ctrl + r.id, 1, false};
      return true;
    case Ref::sensor:
      if (attr != "value") return false;
      loc = {d_->sensordata + m_->sensor_adr[r.id], m_->sensor_dim[r.id], true};
      return true;
    default:
      return false;
  }
}

Result MujocoSolver::get(const std::string& prim, const std::string& attr, const std::string& frame,
                         std::vector<double>& out) const {
  auto it = index_.find(prim);
  if (it == index_.end()) return {Result::not_found, "unknown prim '" + prim + "'"};
  const Ref r = it->second;
  out.clear();
  if (r.kind == Ref::actuator && (attr == "kp" || attr == "kv")) {
    if (!is_world(frame)) return {Result::invalid_argument, "frame only applies to body attributes"};
    if (!is_servo(r.id)) return {Result::invalid_argument, prim + " is not a position servo (no kp/kv)"};
    out.push_back(attr == "kp" ? m_->actuator_gainprm[r.id * mjNGAIN] : -m_->actuator_biasprm[r.id * mjNBIAS + 2]);
    return {};
  }
  if (r.kind == Ref::actuator && attr == "ctrlrange") {  // read-only; (-inf, inf) when the model sets no limit
    if (!is_world(frame)) return {Result::invalid_argument, "frame only applies to body attributes"};
    const double inf = std::numeric_limits<double>::infinity();
    const bool lim = m_->actuator_ctrllimited[r.id];
    out = {lim ? m_->actuator_ctrlrange[2 * r.id] : -inf, lim ? m_->actuator_ctrlrange[2 * r.id + 1] : inf};
    return {};
  }
  if (r.kind == Ref::body) {
    if (attr != "pos" && attr != "quat") return {Result::not_found, "unknown attr '" + attr + "' on body"};
    int fb = -1;  // frame body; -1 = world
    if (!is_world(frame)) {
      auto f = index_.find(frame);
      if (f == index_.end() || f->second.kind != Ref::body)
        return {Result::not_found, "unknown frame '" + frame + "' (need a body prim path or \"world\")"};
      fb = f->second.id;
    }
    mj_kinematics(m_, d_);  // xpos/xquat lag one step behind qpos after mj_step; derived data only, not in the hash
    if (attr == "pos") {
      mjtNum v[3];
      mju_copy3(v, d_->xpos + 3 * r.id);
      if (fb >= 0) {
        mjtNum diff[3];
        mju_sub3(diff, v, d_->xpos + 3 * fb);
        mju_mulMatTVec3(v, d_->xmat + 9 * fb, diff);  // R^T (p - p_b)
      }
      out.assign(v, v + 3);
    } else {
      mjtNum q[4];
      mju_copy4(q, d_->xquat + 4 * r.id);
      if (fb >= 0) {
        mjtNum inv[4];
        mju_negQuat(inv, d_->xquat + 4 * fb);
        mju_mulQuat(q, inv, d_->xquat + 4 * r.id);
      }
      out.assign(q, q + 4);
    }
    return {};
  }
  if (!is_world(frame)) return {Result::invalid_argument, "frame only applies to body attributes"};
  if (r.kind == Ref::device) return devs::get(*dev_, r.id, attr, out);
  Loc loc;
  if (!locate(r, attr, loc)) return {Result::not_found, "unknown attr '" + attr + "' on " + prim};
  out.assign(loc.p, loc.p + loc.n);
  return {};
}

Result MujocoSolver::set(const std::string& prim, const std::string& attr, const std::string& frame,
                         std::span<const double> values) {
  auto it = index_.find(prim);
  if (it == index_.end()) return {Result::not_found, "unknown prim '" + prim + "'"};
  const Ref r = it->second;
  if (r.kind == Ref::body) {
    if (attr == "pos" || attr == "quat") return {Result::invalid_argument, "attr '" + attr + "' is read-only"};
    return {Result::not_found, "unknown attr '" + attr + "' on body"};
  }
  if (!is_world(frame)) return {Result::invalid_argument, "frame only applies to body attributes"};
  if (r.kind == Ref::device) return devs::set(*dev_, r.id, attr, values);
  if (r.kind == Ref::actuator && (attr == "kp" || attr == "kv")) {
    if (!is_servo(r.id)) return {Result::invalid_argument, prim + " is not a position servo (no kp/kv)"};
    if (values.size() != 1 || !std::isfinite(values[0]) || values[0] < 0)
      return {Result::invalid_argument, "attr '" + attr + "' takes one finite value >= 0"};
    // MuJoCo position actuator: force = kp*(ctrl - q) - kv*qdot  =>  gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv.
    if (attr == "kp") {
      m_->actuator_gainprm[r.id * mjNGAIN] = values[0];
      m_->actuator_biasprm[r.id * mjNBIAS + 1] = -values[0];
    } else {
      m_->actuator_biasprm[r.id * mjNBIAS + 2] = -values[0];
    }
    return {};
  }
  Loc loc;
  if (!locate(r, attr, loc)) return {Result::not_found, "unknown attr '" + attr + "' on " + prim};
  if (loc.read_only) return {Result::invalid_argument, "attr '" + attr + "' is read-only"};
  if (static_cast<int>(values.size()) != loc.n)
    return {Result::invalid_argument, "attr '" + attr + "' takes " + std::to_string(loc.n) + " values, got " +
                                          std::to_string(values.size())};
  for (double v : values)
    if (!std::isfinite(v)) return {Result::invalid_argument, "values must be finite"};
  std::memcpy(loc.p, values.data(), values.size_bytes());
  if (r.kind == Ref::joint) mj_forward(m_, d_);  // refresh derived state after a state edit
  return {};
}

// SI units, one table. ctrl is "1": the raw actuator input; its physical unit depends on gear/transmission.
// Mixed-unit attrs (free-joint qpos/qvel) and unlisted sensor types give "".
std::string MujocoSolver::unit(const std::string& prim, const std::string& attr) const {
  auto it = index_.find(prim);
  if (it == index_.end()) return "";
  const Ref r = it->second;
  switch (r.kind) {
    case Ref::device:
      return devs::unit(*dev_, r.id, attr);
    case Ref::body:
      return attr == "pos" ? "m" : attr == "quat" ? "1" : "";
    case Ref::joint: {
      const int t = m_->jnt_type[r.id];
      if (attr == "qpos") return t == mjJNT_HINGE ? "rad" : t == mjJNT_SLIDE ? "m" : t == mjJNT_BALL ? "1" : "";
      if (attr == "qvel") return t == mjJNT_SLIDE ? "m/s" : t == mjJNT_FREE ? "" : "rad/s";
      return "";
    }
    case Ref::actuator: {
      if (attr == "ctrl" || attr == "ctrlrange") return "1";
      if ((attr != "kp" && attr != "kv") || !is_servo(r.id)) return "";
      // Gains are per ctrl unit (ctrl is the target joint position) only for a unit-gear joint transmission.
      if (m_->actuator_trntype[r.id] != mjTRN_JOINT || m_->actuator_gear[r.id * 6] != 1) return "";
      const int t = m_->jnt_type[m_->actuator_trnid[r.id * 2]];
      if (t == mjJNT_SLIDE) return attr == "kp" ? "N/m" : "N*s/m";
      return t == mjJNT_HINGE ? (attr == "kp" ? "N*m/rad" : "N*m*s/rad") : "";
    }
    case Ref::sensor: {
      if (attr != "value") return "";
      const bool slide = (m_->sensor_type[r.id] == mjSENS_JOINTPOS || m_->sensor_type[r.id] == mjSENS_JOINTVEL) &&
                         m_->jnt_type[m_->sensor_objid[r.id]] == mjJNT_SLIDE;
      switch (m_->sensor_type[r.id]) {
        case mjSENS_JOINTPOS: return slide ? "m" : "rad";
        case mjSENS_JOINTVEL: return slide ? "m/s" : "rad/s";
        case mjSENS_FRAMEPOS: return "m";
        case mjSENS_FRAMELINVEL: case mjSENS_VELOCIMETER: return "m/s";
        case mjSENS_FRAMEANGVEL: case mjSENS_GYRO: return "rad/s";
        case mjSENS_ACCELEROMETER: return "m/s^2";
        case mjSENS_FORCE: case mjSENS_TOUCH: return "N";
        case mjSENS_TORQUE: return "N*m";
        default: return "";
      }
    }
  }
  return "";
}

// Snapshot = mjSTATE_INTEGRATION followed by every model parameter a SetAttr can change (today: the actuator
// gain/bias arrays holding kp/kv). Model parameters are not in mjSTATE_INTEGRATION, so without this block
// restore() would silently keep a later kp. Add any new settable model attribute here too. state_hash() covers it.
// Then the device state block (devs::save_state; empty without devices), then the structure id. dof_damping of driven
// dofs is NOT saved: step() zeroes it before every use, so it never carries state between ticks.
std::vector<std::byte> MujocoSolver::save() const {
  const int ns = mj_stateSize(m_, kStateSig), np = m_->nu * (mjNGAIN + mjNBIAS);
  std::vector<mjtNum> s(static_cast<std::size_t>(ns + np));
  mj_getState(m_, d_, s.data(), kStateSig);
  mju_copy(s.data() + ns, m_->actuator_gainprm, m_->nu * mjNGAIN);
  mju_copy(s.data() + ns + m_->nu * mjNGAIN, m_->actuator_biasprm, m_->nu * mjNBIAS);
  const std::size_t body = s.size() * sizeof(mjtNum), nd = dev_ ? devs::state_bytes(*dev_) : 0;
  std::vector<std::byte> out(body + nd + sizeof structure_);  // + device state block + the structure id (Spawn / Delete)
  std::memcpy(out.data(), s.data(), body);
  if (dev_) devs::save_state(*dev_, out.data() + body);
  std::memcpy(out.data() + body + nd, &structure_, sizeof structure_);
  return out;
}

void MujocoSolver::restore(std::span<const std::byte> bytes) {
  const int ns = mj_stateSize(m_, kStateSig);
  const std::size_t body = sizeof(mjtNum) * static_cast<std::size_t>(ns + m_->nu * (mjNGAIN + mjNBIAS));
  const std::size_t nd = dev_ ? devs::state_bytes(*dev_) : 0;
  std::uint64_t structure = 0;
  if (bytes.size() == body + nd + sizeof structure) std::memcpy(&structure, bytes.data() + body + nd, sizeof structure);
  if (bytes.size() != body + nd + sizeof structure || structure != structure_)
    throw std::runtime_error(
        "MujocoSolver::restore: snapshot does not fit the current model structure (taken on another model, or before a "
        "Spawn / Delete)");
  std::vector<mjtNum> s(body / sizeof(mjtNum));  // copy: span bytes may be unaligned
  std::memcpy(s.data(), bytes.data(), body);
  mju_copy(m_->actuator_gainprm, s.data() + ns, m_->nu * mjNGAIN);
  mju_copy(m_->actuator_biasprm, s.data() + ns + m_->nu * mjNGAIN, m_->nu * mjNBIAS);
  if (dev_) devs::load_state(*dev_, bytes.data() + body);
  mj_setState(m_, d_, s.data(), kStateSig);
  mj_forward(m_, d_);
  // step() reads these sticky counters: a restored healthy state must not inherit the divergence it replaced.
  for (mjWarningStat& w : d_->warning) w.number = 0;
  diverged_ = false;
}

namespace {
// Grid lines along one axis of the water mesh: 1.5 m steps over the middle (|u| <= 0.55 half), growing by 25 % per step out to the plane's edge.
std::vector<double> water_axis(double half) {
  std::vector<double> u{0.0};
  const double fine = 0.55 * half;
  for (double step = 1.5; u.back() < half;) {
    u.push_back(std::min(u.back() + step, half));
    if (u.back() > fine) step *= 1.25;
  }
  std::vector<double> out;
  for (auto it = u.rbegin(); it != u.rend(); ++it) out.push_back(-*it);
  for (std::size_t i = 1; i < u.size(); ++i) out.push_back(u[i]);
  return out;
}

// The heightfield `id` as unrolled triangles (position xyz, smooth normal xyz) in the geom frame. Cell (r, c) is split along the
// (c, r) - (c + 1, r + 1) diagonal; vertex normals from central differences of the grid.
std::vector<float> hfield_mesh(const mjModel* m, int id) {
  const int nr = m->hfield_nrow[id], nc = m->hfield_ncol[id];
  const double* sz = m->hfield_size + 4 * id;
  const float* data = m->hfield_data + m->hfield_adr[id];
  const double dx = 2 * sz[0] / (nc - 1), dy = 2 * sz[1] / (nr - 1);
  // MuJoCo's data row 0 is the +y edge (image convention); our row r is at y = -sy + r dy.
  auto z = [&](int r, int c) { return static_cast<double>(data[(nr - 1 - std::clamp(r, 0, nr - 1)) * nc + std::clamp(c, 0, nc - 1)]) * sz[2]; };
  auto vert = [&](std::vector<float>& o, int r, int c) {
    const double nx = -(z(r, c + 1) - z(r, c - 1)) / ((std::min(c + 1, nc - 1) - std::max(c - 1, 0)) * dx);
    const double ny = -(z(r + 1, c) - z(r - 1, c)) / ((std::min(r + 1, nr - 1) - std::max(r - 1, 0)) * dy);
    const double l = std::sqrt(nx * nx + ny * ny + 1);
    o.insert(o.end(), {static_cast<float>(-sz[0] + c * dx), static_cast<float>(-sz[1] + r * dy), static_cast<float>(z(r, c)),
                       static_cast<float>(nx / l), static_cast<float>(ny / l), static_cast<float>(1 / l)});
  };
  std::vector<float> o;
  o.reserve(static_cast<std::size_t>(nr - 1) * static_cast<std::size_t>(nc - 1) * 36);
  for (int r = 0; r + 1 < nr; ++r)
    for (int c = 0; c + 1 < nc; ++c) {
      vert(o, r, c); vert(o, r, c + 1); vert(o, r + 1, c + 1);
      vert(o, r, c); vert(o, r + 1, c + 1); vert(o, r + 1, c);
    }
  return o;
}
}  // namespace

Weather MujocoSolver::weather() const {
  Weather w;
  if (!dev_ || dev_->rain.st.empty()) return w;
  w.rain = dev_->rain.st[0].rate;
  w.time = d_->time;
  devs::wind_at(*dev_, w.wind);
  return w;
}

std::vector<GeomVisual> MujocoSolver::visuals() const {
  std::vector<GeomVisual> out(static_cast<std::size_t>(m_->ngeom));
  for (int g = 0; g < m_->ngeom; ++g) {
    GeomVisual& v = out[static_cast<std::size_t>(g)];
    switch (m_->geom_type[g]) {
      case mjGEOM_PLANE: v.type = GeomType::plane; break;
      case mjGEOM_SPHERE: v.type = GeomType::sphere; break;
      case mjGEOM_CAPSULE: v.type = GeomType::capsule; break;
      case mjGEOM_ELLIPSOID: v.type = GeomType::ellipsoid; break;
      case mjGEOM_CYLINDER: v.type = GeomType::cylinder; break;
      case mjGEOM_BOX: v.type = GeomType::box; break;
      case mjGEOM_MESH: v.type = GeomType::mesh; break;
      case mjGEOM_HFIELD: v.type = GeomType::hfield; break;
      default: v.type = GeomType::other; break;
    }
    for (int k = 0; k < 3; ++k) v.size[k] = m_->geom_size[3 * g + k];
    for (int k = 0; k < 4; ++k) v.rgba[k] = m_->geom_rgba[4 * g + k];
    v.group = m_->geom_group[g];
    if (const int mat = m_->geom_matid[g]; mat >= 0) {
      // As MuJoCo's own renderer: the material colour unless the geom set its own (anything but the 0.5 0.5 0.5 1 default).
      const bool geom_default = v.rgba[0] == 0.5f && v.rgba[1] == 0.5f && v.rgba[2] == 0.5f && v.rgba[3] == 1.0f;
      if (geom_default)
        for (int k = 0; k < 4; ++k) v.rgba[k] = m_->mat_rgba[4 * mat + k];
      v.specular = m_->mat_specular[mat];
      v.shininess = m_->mat_shininess[mat];
      v.metallic = m_->mat_metallic[mat];
      v.roughness = m_->mat_roughness[mat];
    }
    if (v.type == GeomType::plane && dev_ && !dev_->sea.cfg.empty() && m_->geom_bodyid[g] == 0 && m_->geom_size[3 * g] > 0 &&
        m_->geom_size[3 * g + 1] > 0 && !m_->geom_pos[3 * g] && !m_->geom_pos[3 * g + 1] && !m_->geom_pos[3 * g + 2]) {
      const char* name = mj_id2name(m_, mjOBJ_GEOM, g);
      if (name && std::strcmp(name, "water") == 0) {
        // a plane named "water" at the world origin in a model with a Sea is the sea surface: the wave field sampled now, over the plane's extent
        v.type = GeomType::mesh;
        v.group = GeomVisual::kWaterGroup;
        v.mesh = devs::sea_surface_mesh(*dev_, water_axis(m_->geom_size[3 * g]), water_axis(m_->geom_size[3 * g + 1]));
        continue;
      }
    }
    if (v.type == GeomType::hfield) {
      v.mesh = hfield_mesh(m_, m_->geom_dataid[g]);
      continue;
    }
    if (v.type != GeomType::mesh) continue;
    // mesh_face / mesh_facenormal index locally from the mesh's vertex / normal address.
    const int id = m_->geom_dataid[g], fa = m_->mesh_faceadr[id], fn = m_->mesh_facenum[id];
    const int va = m_->mesh_vertadr[id], na = m_->mesh_normaladr[id];
    v.mesh.reserve(static_cast<std::size_t>(fn) * 18);
    for (int f = 0; f < fn; ++f) {
      const float* p[3];
      for (int k = 0; k < 3; ++k) p[k] = m_->mesh_vert + 3 * (va + m_->mesh_face[3 * (fa + f) + k]);
      float fnorm[3];  // flat face normal from the winding
      const float e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
      const float e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
      fnorm[0] = e1[1] * e2[2] - e1[2] * e2[1];
      fnorm[1] = e1[2] * e2[0] - e1[0] * e2[2];
      fnorm[2] = e1[0] * e2[1] - e1[1] * e2[0];
      const float len = std::sqrt(fnorm[0] * fnorm[0] + fnorm[1] * fnorm[1] + fnorm[2] * fnorm[2]);
      for (float& c : fnorm) c = len > 0 ? c / len : 0;
      for (int k = 0; k < 3; ++k) {
        const float* n = m_->mesh_normal + 3 * (na + m_->mesh_facenormal[3 * (fa + f) + k]);
        // MuJoCo's own renderer (render_context.c) keeps the stored (smooth) normal only when it is within ~37 degrees
        // of the face normal, else shades the face flat; mirror that.
        const bool smooth = n[0] * fnorm[0] + n[1] * fnorm[1] + n[2] * fnorm[2] >= 0.8f;
        const float* use = smooth ? n : fnorm;
        v.mesh.insert(v.mesh.end(), {p[k][0], p[k][1], p[k][2], use[0], use[1], use[2]});
      }
    }
  }
  return out;
}

std::vector<GeomPose> MujocoSolver::poses() const {
  mj_kinematics(m_, d_);  // geom_xpos/xmat lag one step behind qpos after mj_step; derived data only
  std::vector<GeomPose> out(static_cast<std::size_t>(m_->ngeom));
  for (int g = 0; g < m_->ngeom; ++g) {
    GeomPose& p = out[static_cast<std::size_t>(g)];
    for (int k = 0; k < 3; ++k) p.pos[k] = d_->geom_xpos[3 * g + k];
    for (int k = 0; k < 9; ++k) p.rot[k] = d_->geom_xmat[9 * g + k];
  }
  return out;
}

// Spatial tendons through sites (>= 2 site wraps, visible rgba) are the design builder's wires: the renderer draws them
// as tubes. Pulleys / geom wraps are not tubes (their wrap points are not kept here).
std::vector<TubeVisual> MujocoSolver::collect_tubes(std::vector<std::pair<std::string, std::string>>* ends) const {
  std::vector<TubeVisual> out;
  if (m_->ntendon == 0) return out;
  mj_kinematics(m_, d_);  // site_xpos
  for (int t = 0; t < m_->ntendon; ++t) {
    if (m_->tendon_rgba[4 * t + 3] <= 0) continue;
    TubeVisual v;
    for (int k = m_->tendon_adr[t]; k < m_->tendon_adr[t] + m_->tendon_num[t]; ++k)
      if (m_->wrap_type[k] == mjWRAP_SITE)
        for (int c = 0; c < 3; ++c) v.pts.push_back(d_->site_xpos[3 * m_->wrap_objid[k] + c]);
    if (v.pts.size() < 6) continue;
    if (ends) {  // the design builder names a cable "wire:<i>:<c>|<end a>|<end b>"
      const std::string_view s(m_->names + m_->name_tendonadr[t]);
      const std::size_t p = s.find('|'), q = p == std::string_view::npos ? p : s.find('|', p + 1);
      if (q != std::string_view::npos) ends->emplace_back(std::string(s.substr(p + 1, q - p - 1)), std::string(s.substr(q + 1)));
      else ends->emplace_back();
    }
    v.radius = m_->tendon_width[t];  // MuJoCo's tendon "width" is the rendered cross-section RADIUS
    v.length = m_->tendon_lengthspring[2 * t];
    for (int c = 0; c < 4; ++c) v.rgba[c] = m_->tendon_rgba[4 * t + c];
    out.push_back(std::move(v));
  }
  return out;
}

std::vector<double> MujocoSolver::raycast(std::span<const double> origin, std::span<const double> directions) const {
  if (origin.size() != 3 || directions.size() % 3 != 0) throw std::invalid_argument("raycast: need origin[3] and directions[3n]");
  const int n = static_cast<int>(directions.size() / 3);
  std::vector<mjtNum> dirs(directions.begin(), directions.end());
  for (int i = 0; i < n; ++i) {
    mjtNum* d = dirs.data() + 3 * i;
    const mjtNum len = mju_norm3(d);
    if (!(len > 0) || !std::isfinite(len)) throw std::invalid_argument("raycast: direction " + std::to_string(i) + " is zero or not finite");
    mju_scl3(d, d, 1 / len);
  }
  std::vector<double> dist(static_cast<std::size_t>(n));
  std::vector<int> geomid(static_cast<std::size_t>(n));
  if (n == 0) return dist;
  mj_kinematics(m_, d_);  // ray tests read geom_xpos/xmat
  mj_multiRay(m_, d_, origin.data(), dirs.data(), nullptr, 1, -1, geomid.data(), dist.data(), nullptr, n, mjMAXVAL);
  return dist;
}

std::unique_ptr<Solver> MujocoSolver::clone() const {
  mjModel* m = mj_copyModel(nullptr, m_);
  mjData* d = mj_copyData(nullptr, m, d_);
  auto c = std::make_unique<MujocoSolver>(m, d, spec_, spec_xml_);  // the spec is shared: edits copy it
  c->diverged_ = diverged_;
  if (dev_) c->dev_ = std::make_unique<devs::Devices>(*dev_);  // the ctor re-parsed the config; this carries the live state
  return c;
}

std::uint64_t MujocoSolver::state_hash() const {
  std::vector<std::byte> s = save();
  std::uint64_t h = 14695981039346656037ull;  // FNV-1a 64
  for (std::byte b : s) h = (h ^ static_cast<std::uint64_t>(b)) * 1099511628211ull;
  return h;
}
