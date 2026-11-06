// The command bus: World::execute (binary FlatBuffers in/out) and the strict JSON edge.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

#include "core/monitor.h"
#include "core/world.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/reflection.h"
#include "dvsim_generated.h"

namespace fs = std::filesystem;

namespace {
// Plain-C++ outcome of RunTest (converted to dvsim.TestResult when the reply is built). Ticks are relative to
// the test start; `start_tick` converts to world ticks.
struct TestOut {
  bool diverged = false, failed = false;
  std::uint64_t start_tick = 0, end_tick = 0, fail_rel = 0, fail_hash = 0;
  double dt = 0;
  std::vector<MonitorResult> monitors;
  std::string fail_monitor;
  std::vector<std::byte> fail_snapshot;
  std::vector<std::string> trace_cols;
  std::vector<double> trace;
  bool pass() const {
    return !diverged && std::all_of(monitors.begin(), monitors.end(), [](const MonitorResult& m) { return m.pass; });
  }
};

// One sweep run: the parameter values applied and the verdict (error != "" => the run itself failed).
struct SweepRowOut {
  std::vector<double> params;
  bool pass = false, diverged = false;
  std::string error;
  std::uint64_t start_tick = 0;
  double dt = 0;
  std::vector<MonitorResult> monitors;
};
struct SweepOut {
  std::vector<std::string> names;
  std::vector<SweepRowOut> rows;
};

struct Out {
  dvsim::Err err = dvsim::Err::none;
  std::string message, unit;
  std::vector<double> values;
  std::vector<std::string> prims;
  std::uint64_t snapshot_id = 0;
  std::optional<TestOut> test;
  std::optional<SweepOut> sweep;
  std::optional<Image> image;
  std::vector<double> distances;
  bool timed_out = false;  // err == limit_exceeded because the wall-clock deadline passed (not a length limit)
};

using Clock = std::chrono::steady_clock;
constexpr Clock::time_point kNever = Clock::time_point::max();
Clock::time_point deadline_of(const Session& s) {
  return s.max_wall_s > 0 ? Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s.max_wall_s))
                          : kNever;
}

// A sweep parameter value applied after the spec's own settings (scalar attr).
struct Override { std::string path, attr; double value; };

Out fail(dvsim::Err e, std::string msg) {
  Out o;
  o.err = e;
  o.message = std::move(msg);
  return o;
}

Out from(const Result& r) {
  if (r.code == Result::ok) return {};
  return fail(r.code == Result::not_found ? dvsim::Err::not_found : dvsim::Err::invalid_argument, r.message);
}

std::string str(const flatbuffers::String* s) { return s ? s->str() : std::string(); }

// RunTest: validate everything up front, apply settings, then step one tick at a time feeding the monitors.
// Runs to the end even after a violation (full stats); stops early only if the solver diverges. `over` (sweep
// parameters) is applied right after spec.settings. Past `deadline` it returns timed_out (world left mid-test).
Out run_test(World& w, const dvsim::TestSpec& spec, const Session& session, Clock::time_point deadline,
             const std::vector<Override>& over = {}) {
  Solver& sv = w.solver();
  const double dt = sv.dt(), dur = spec.duration();
  const bool has_dur = dur > 0, has_ticks = spec.ticks() > 0;
  if (has_dur == has_ticks) return fail(dvsim::Err::invalid_argument, "TestSpec needs exactly one of duration (> 0 s) or ticks (> 0)");
  const double want = has_ticks ? spec.ticks() : std::round(dur / dt);
  if (!(want <= session.max_steps_per_call))
    return fail(dvsim::Err::limit_exceeded, "test length exceeds max_steps_per_call=" + std::to_string(session.max_steps_per_call));
  if (want < 1) return fail(dvsim::Err::invalid_argument, "duration is shorter than one tick");
  const std::uint64_t n_ticks = static_cast<std::uint64_t>(want);
  const std::uint64_t every = static_cast<std::uint64_t>(std::max(1, spec.trace_every()));
  if (!spec.monitors() || spec.monitors()->size() == 0) return fail(dvsim::Err::invalid_argument, "TestSpec has no monitors");

  if (spec.settings())
    for (const dvsim::Setting* s : *spec.settings()) {
      std::vector<double> v;
      if (s->values()) v.assign(s->values()->begin(), s->values()->end());
      Out o = from(sv.set(str(s->path()), str(s->attr()), str(s->frame()), v));
      if (o.err != dvsim::Err::none) {
        o.message = "setting " + str(s->path()) + "." + str(s->attr()) + ": " + o.message;
        return o;
      }
    }
  for (const Override& p : over) {
    Out o = from(sv.set(p.path, p.attr, "", std::vector<double>{p.value}));
    if (o.err != dvsim::Err::none) {
      o.message = "param " + p.path + "." + p.attr + "=" + std::to_string(p.value) + ": " + o.message;
      return o;
    }
  }

  struct Sig { std::string path, attr, frame; std::size_t index; };
  std::vector<Sig> sigs;
  std::map<std::string, std::size_t> col_of;  // distinct signals share a trace column
  std::vector<std::size_t> col;               // per monitor
  std::vector<Monitor> mons;
  TestOut t;
  t.dt = dt;
  t.start_tick = w.tick();
  t.trace_cols = {"tick", "t"};
  for (const dvsim::Monitor* m : *spec.monitors()) {
    const std::string name = str(m->name());
    auto bad = [&](const std::string& why) { return fail(dvsim::Err::invalid_argument, "monitor '" + name + "': " + why); };
    const dvsim::Signal* sg = m->signal();
    if (!sg) return bad("missing signal");
    if (!(m->lo() <= m->hi())) return bad("need lo <= hi");
    if (!(m->from() >= 0 && m->to() >= m->from())) return bad("need 0 <= from <= to");
    std::vector<double> v;
    Out o = from(sv.get(str(sg->path()), str(sg->attr()), str(sg->frame()), v));
    if (o.err != dvsim::Err::none) return bad(o.message);
    if (sg->index() < 0 || static_cast<std::size_t>(sg->index()) >= v.size())
      return bad("signal index " + std::to_string(sg->index()) + " out of range (attr has " + std::to_string(v.size()) + " values)");
    const std::string cname = str(sg->path()) + "." + str(sg->attr()) + "[" + std::to_string(sg->index()) + "]" +
                              (str(sg->frame()).empty() ? "" : "@" + str(sg->frame()));
    auto [it, fresh] = col_of.try_emplace(cname, sigs.size());
    if (fresh) {
      sigs.push_back({str(sg->path()), str(sg->attr()), str(sg->frame()), static_cast<std::size_t>(sg->index())});
      t.trace_cols.push_back(cname);
    }
    col.push_back(it->second);
    const double ft = std::ceil(m->from() / dt - 1e-9), tt = std::floor(m->to() / dt + 1e-9);  // seconds -> ticks
    mons.emplace_back(name, m->op() == dvsim::MonitorOp::always, m->lo(), m->hi(), static_cast<std::uint64_t>(ft),
                      tt >= 1.8e19 ? std::numeric_limits<std::uint64_t>::max() : static_cast<std::uint64_t>(tt));
  }

  std::vector<double> cur(sigs.size()), tmp;
  for (std::uint64_t k = 0;; ++k) {
    if (k > 0) {
      if (k % 64 == 0 && deadline != kNever && Clock::now() >= deadline) {
        Out o = fail(dvsim::Err::limit_exceeded, "wall time exceeded max_wall_s=" + std::to_string(session.max_wall_s));
        o.timed_out = true;
        return o;
      }
      w.step(1);
      if (w.diverged()) {
        t.diverged = true;
        break;
      }
    }
    for (std::size_t i = 0; i < sigs.size(); ++i) {
      const Result r = sv.get(sigs[i].path, sigs[i].attr, sigs[i].frame, tmp);
      if (r.code != Result::ok) return fail(dvsim::Err::solver_error, "signal read failed at tick " + std::to_string(k) + ": " + r.message);
      cur[i] = tmp[sigs[i].index];
    }
    for (std::size_t i = 0; i < mons.size(); ++i)
      if (mons[i].sample(k, cur[col[i]]) && !t.failed) {  // earliest violation overall (ties: first monitor)
        t.failed = true;
        t.fail_rel = k;
        t.fail_monitor = mons[i].result().name;
        t.fail_hash = w.state_hash();
        t.fail_snapshot = w.save();
      }
    if (k % every == 0 || k == n_ticks) {
      t.trace.push_back(static_cast<double>(w.tick()));
      t.trace.push_back(static_cast<double>(k) * dt);
      t.trace.insert(t.trace.end(), cur.begin(), cur.end());
    }
    if (k == n_ticks) break;
  }
  t.end_tick = w.tick();
  for (const Monitor& m : mons) t.monitors.push_back(m.result());
  Out o;
  o.test = std::move(t);
  return o;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dvsim::MonitorResult>>> build_monitors(
    flatbuffers::FlatBufferBuilder& fbb, const std::vector<MonitorResult>& rs, std::uint64_t start_tick, double dt) {
  std::vector<flatbuffers::Offset<dvsim::MonitorResult>> ms;
  for (const MonitorResult& r : rs)
    ms.push_back(dvsim::CreateMonitorResult(fbb, fbb.CreateString(r.name), r.pass, r.hit, r.hit ? start_tick + r.hit_tick : 0,
                                           r.hit ? static_cast<double>(r.hit_tick) * dt : 0, r.hit_value, r.samples,
                                           r.min_value, r.max_value, r.rms, r.final_value));
  return fbb.CreateVector(ms);
}

flatbuffers::Offset<dvsim::SweepResult> build_sweep(flatbuffers::FlatBufferBuilder& fbb, const SweepOut& s) {
  std::vector<flatbuffers::Offset<dvsim::SweepRow>> rows;
  for (std::size_t i = 0; i < s.rows.size(); ++i) {
    const SweepRowOut& r = s.rows[i];
    auto params = fbb.CreateVector(r.params);
    auto error = fbb.CreateString(r.error);
    auto monitors = build_monitors(fbb, r.monitors, r.start_tick, r.dt);
    rows.push_back(dvsim::CreateSweepRow(fbb, i, params, r.pass, r.diverged, error, monitors));
  }
  auto names = fbb.CreateVectorOfStrings(s.names);
  auto rv = fbb.CreateVector(rows);
  return dvsim::CreateSweepResult(fbb, names, rv);
}

flatbuffers::Offset<dvsim::TestResult> build_result(flatbuffers::FlatBufferBuilder& fbb, const TestOut& t) {
  auto monitors = build_monitors(fbb, t.monitors, t.start_tick, t.dt);
  auto fail_monitor = fbb.CreateString(t.fail_monitor);
  // ttsr-ignore: cpp/no-reinterpret-const-cast  (std::byte -> uint8_t byte view, same idiom as replay.cpp)
  auto snap = fbb.CreateVector(reinterpret_cast<const std::uint8_t*>(t.fail_snapshot.data()), t.fail_snapshot.size());
  auto cols = fbb.CreateVectorOfStrings(t.trace_cols);
  auto trace = fbb.CreateVector(t.trace);
  return dvsim::CreateTestResult(fbb, t.pass(), t.diverged, t.start_tick, t.end_tick, t.dt, monitors, fail_monitor,
                                t.failed ? t.start_tick + t.fail_rel : 0,
                                static_cast<double>(t.fail_rel) * t.dt, t.fail_hash, snap, cols, trace);
}

// Sweep: every run is run_test on its own clone of the base state. Workers pull run indices from an atomic counter and
// write into rows[index], and a run's parameters depend only on its index (grid position / RNG seeded by seed + index),
// so the rows are identical for any thread count. A run's failure is its row's `error`; only the wall deadline aborts
// the whole sweep (no partial rows). `live` is cloned, never modified.
Out run_sweep(const World& live, const std::vector<std::byte>* snap, const dvsim::Sweep& sw, const Session& session) {
  const dvsim::TestSpec* spec = sw.spec();
  if (!spec) return fail(dvsim::Err::invalid_argument, "Sweep needs a spec");
  if (!sw.params() || sw.params()->size() == 0) return fail(dvsim::Err::invalid_argument, "Sweep needs at least one param");
  const bool grid = sw.mode() == dvsim::SweepMode::grid;
  const auto& ps = *sw.params();
  World base = live.clone();
  if (snap) base.restore(*snap);

  SweepOut out;
  double total = grid ? 1 : sw.n();  // double: the product of grid sizes cannot overflow
  std::vector<double> tmp;
  for (const dvsim::SweepParam* p : ps) {
    const std::string name = str(p->path()) + "." + str(p->attr());
    Out o = from(base.solver().get(str(p->path()), str(p->attr()), "", tmp));
    if (o.err != dvsim::Err::none) return fail(o.err, "param " + name + ": " + o.message);
    if (tmp.size() != 1) return fail(dvsim::Err::invalid_argument, "param " + name + " is not a scalar attribute");
    if (grid) {
      if (!p->values() || p->values()->size() == 0) return fail(dvsim::Err::invalid_argument, "param " + name + ": grid needs values");
      total *= static_cast<double>(p->values()->size());
    } else if (!(std::isfinite(p->lo()) && std::isfinite(p->hi()) && p->lo() <= p->hi())) {
      return fail(dvsim::Err::invalid_argument, "param " + name + ": random needs finite lo <= hi");
    }
    out.names.push_back(name);
  }
  if (!grid && sw.n() < 1) return fail(dvsim::Err::invalid_argument, "random sweep needs n >= 1");
  if (total > session.max_sweep_runs)
    return fail(dvsim::Err::limit_exceeded, "sweep has " + std::to_string(total) + " runs, exceeds max_sweep_runs=" +
                                               std::to_string(session.max_sweep_runs));
  const std::size_t runs = static_cast<std::size_t>(total);
  out.rows.resize(runs);

  const Clock::time_point deadline = deadline_of(session);
  std::atomic<std::size_t> next{0};
  std::atomic<bool> timed_out{false};
  auto work = [&] {
    for (std::size_t i = next.fetch_add(1); i < runs && !timed_out; i = next.fetch_add(1)) {
      SweepRowOut& row = out.rows[i];
      std::vector<Override> over;
      std::mt19937_64 rng(sw.seed() + i);  // random mode: stream depends on the run index alone
      std::size_t rest = i;
      over.resize(ps.size());
      for (std::size_t k = ps.size(); k-- > 0;) {  // grid: last param varies fastest
        const dvsim::SweepParam* p = ps[static_cast<flatbuffers::uoffset_t>(k)];
        double v;
        if (grid) {
          const std::size_t sz = p->values()->size();
          v = (*p->values())[static_cast<flatbuffers::uoffset_t>(rest % sz)];
          rest /= sz;
        } else {
          v = 0;  // drawn below in param order
        }
        over[k] = {str(p->path()), str(p->attr()), v};
      }
      if (!grid)
        for (std::size_t k = 0; k < ps.size(); ++k) {
          const dvsim::SweepParam* p = ps[static_cast<flatbuffers::uoffset_t>(k)];
          over[k].value = p->lo() + (p->hi() - p->lo()) * (static_cast<double>(rng() >> 11) * 0x1.0p-53);
        }
      for (const Override& o : over) row.params.push_back(o.value);
      try {
        World w = base.clone();
        Out o = run_test(w, *spec, session, deadline, over);
        if (o.timed_out) {
          timed_out = true;
        } else if (o.err != dvsim::Err::none) {
          row.error = o.message;
        } else {
          row.pass = o.test->pass();
          row.diverged = o.test->diverged;
          row.start_tick = o.test->start_tick;
          row.dt = o.test->dt;
          row.monitors = std::move(o.test->monitors);
        }
      } catch (const std::exception& e) {  // e.g. a MuJoCo error (throwing handler): this run only
        row.error = e.what();
      }
    }
  };
  // sw.threads() and the run count are the limits; the host's core count is the ceiling (hfbox is dedicated: the old "<= 6" house rule is lifted).
  const std::size_t nthreads = std::min<std::size_t>({static_cast<std::size_t>(std::max(1, sw.threads())),
                                                      std::max<std::size_t>(1, std::thread::hardware_concurrency()), runs});
  std::vector<std::thread> pool;
  for (std::size_t t = 1; t < nthreads; ++t) pool.emplace_back(work);
  work();
  for (std::thread& t : pool) t.join();
  if (timed_out)
    return fail(dvsim::Err::limit_exceeded, "sweep exceeded max_wall_s=" + std::to_string(session.max_wall_s));
  Out o;
  o.sweep = std::move(out);
  return o;
}

// weakly_canonical resolves symlinks/".." in the existing prefix; then require the result under root.
bool confined(const fs::path& root, const std::string& path, std::string& resolved) {
  if (root.empty()) return false;
  std::error_code ec;
  const fs::path r = fs::weakly_canonical(root, ec);
  if (ec) return false;
  const fs::path p = fs::weakly_canonical(r / path, ec);  // absolute `path` replaces r
  if (ec) return false;
  const fs::path rel = p.lexically_relative(r);
  if (rel.empty() || *rel.begin() == "..") return false;
  resolved = p.string();
  return true;
}
}  // namespace

// Required capability of a command table (e.g. "LoadModel"), read from the `cap` attribute in the embedded .bfbs.
// Empty string = unknown command or attribute missing (caller fails closed).
std::string command_cap(std::string_view command) {
  static const reflection::Schema* schema = reflection::GetSchema(dvsim::EnvelopeBinarySchema::data());
  const std::string name = "dvsim." + std::string(command);
  const reflection::Object* obj = schema->objects()->LookupByKey(name.c_str());
  const reflection::KeyValue* kv =
      obj && obj->attributes() ? obj->attributes()->LookupByKey("cap") : nullptr;  // cap via reflection
  return kv && kv->value() ? kv->value()->str() : std::string();
}

void World::start_log() {
  if (!snapshots_.empty() || next_snapshot_ != 1) throw std::logic_error("start_log: World already has snapshots");
  log_ = std::make_unique<ReplayLog>(solver_->version(), solver_->model_bytes(), save());
  log_->set_final_tick(tick_);
}

flatbuffers::DetachedBuffer World::execute(std::span<const std::uint8_t> bytes, const Session& session) {
  const std::uint64_t tick0 = tick_;  // applied_tick of a logged record
  std::uint64_t id = 0;
  std::string cap;  // stays empty unless the envelope verified and names a known command
  bool swept = false;
  Out out = [&]() -> Out {
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!dvsim::VerifyEnvelopeBuffer(verifier)) return fail(dvsim::Err::invalid_argument, "malformed envelope buffer");
    const dvsim::Envelope& env = *dvsim::GetEnvelope(bytes.data());
    id = env.id();
    const dvsim::Command type = env.cmd_type();
    if (type == dvsim::Command::NONE) return fail(dvsim::Err::invalid_argument, "envelope has no command");
    const std::string cmd = dvsim::EnumNameCommand(type);
    cap = command_cap(cmd);
    swept = type == dvsim::Command::Sweep;
    if (cap.empty() || !session.has(cap))
      return fail(dvsim::Err::permission_denied,
                  cmd + " requires capability '" + cap + "' which this session lacks");
    try {
      switch (type) {
        case dvsim::Command::Query: {
          const dvsim::Query* q = env.cmd_as_Query();
          Out o;
          const std::string path = str(q->path()), attr = str(q->attr());
          if (path.empty()) {
            for (const Prim& p : solver_->prims()) o.prims.push_back(p.path);
            return o;
          }
          if (attr.empty()) return fail(dvsim::Err::invalid_argument, "Query needs an attr");
          std::vector<double> v;
          o = from(solver_->get(path, attr, str(q->frame()), v));
          if (o.err == dvsim::Err::none) {
            o.values = std::move(v);
            o.unit = solver_->unit(path, attr);
          }
          return o;
        }
        case dvsim::Command::SetAttr: {
          const dvsim::SetAttr* s = env.cmd_as_SetAttr();
          std::vector<double> v;
          if (s->values()) v.assign(s->values()->begin(), s->values()->end());
          Out o = from(solver_->set(str(s->path()), str(s->attr()), str(s->frame()), v));
          if (o.err == dvsim::Err::none) o.unit = solver_->unit(str(s->path()), str(s->attr()));
          return o;
        }
        case dvsim::Command::Step: {
          const int n = env.cmd_as_Step()->n();
          if (n < 1) return fail(dvsim::Err::invalid_argument, "Step n must be >= 1");
          if (n > session.max_steps_per_call)
            return fail(dvsim::Err::limit_exceeded,
                        "Step n=" + std::to_string(n) + " exceeds max_steps_per_call=" +
                            std::to_string(session.max_steps_per_call));
          step(n);
          if (diverged()) return fail(dvsim::Err::solver_error, "solver diverged");
          return {};
        }
        case dvsim::Command::Snapshot: {
          Out o;
          o.snapshot_id = next_snapshot_++;
          snapshots_[o.snapshot_id] = save();
          return o;
        }
        case dvsim::Command::Restore: {
          auto it = snapshots_.find(env.cmd_as_Restore()->id());
          if (it == snapshots_.end()) return fail(dvsim::Err::not_found, "no such snapshot id");
          restore(it->second);
          return {};
        }
        case dvsim::Command::RunTest: {
          const dvsim::TestSpec* spec = env.cmd_as_RunTest()->spec();
          if (!spec) return fail(dvsim::Err::invalid_argument, "RunTest needs a spec");
          // On a wall-time timeout, roll back so the refused (hence unlogged) command leaves no trace.
          std::vector<std::byte> before;
          if (session.max_wall_s > 0) before = save();
          Out o = run_test(*this, *spec, session, deadline_of(session));
          if (o.timed_out) restore(before);
          return o;
        }
        case dvsim::Command::Sweep: {
          const dvsim::Sweep* sw = env.cmd_as_Sweep();
          const std::vector<std::byte>* snap = nullptr;
          if (sw->base() != 0) {
            auto it = snapshots_.find(sw->base());
            if (it == snapshots_.end()) return fail(dvsim::Err::not_found, "no such snapshot id");
            snap = &it->second;
          }
          return run_sweep(*this, snap, *sw, session);
        }
        case dvsim::Command::Run: {  // pacing itself belongs to Host; here: validate only
          const double rate = env.cmd_as_Run()->rate();
          if (!(rate >= 0)) return fail(dvsim::Err::invalid_argument, "Run rate must be >= 0");
          return {};
        }
        case dvsim::Command::Pause:
          return {};
        case dvsim::Command::Render: {
          const dvsim::Render* r = env.cmd_as_Render();
          Camera cam;
          auto vec3 = [](const flatbuffers::Vector<double>* v, double* dst) {
            if (!v || v->size() != 3) return false;
            for (flatbuffers::uoffset_t i = 0; i < 3; ++i) {
              dst[i] = (*v)[i];
              if (!std::isfinite(dst[i])) return false;
            }
            return true;
          };
          if (!vec3(r->pos(), cam.pos) || !vec3(r->lookat(), cam.lookat))
            return fail(dvsim::Err::invalid_argument, "Render needs pos and lookat: 3 finite doubles each");
          const double dx = cam.lookat[0] - cam.pos[0], dy = cam.lookat[1] - cam.pos[1], dz = cam.lookat[2] - cam.pos[2];
          if (!(dx * dx + dy * dy + dz * dz > 0)) return fail(dvsim::Err::invalid_argument, "Render: pos and lookat coincide");
          cam.fovy = r->fovy();
          if (!(cam.fovy > 0 && cam.fovy < 180)) return fail(dvsim::Err::invalid_argument, "Render fovy must be in (0, 180) degrees");
          constexpr int kMaxSide = 4096;
          if (r->width() < 1 || r->height() < 1 || r->width() > kMaxSide || r->height() > kMaxSide)
            return fail(dvsim::Err::invalid_argument, "Render width/height must be in [1, " + std::to_string(kMaxSide) + "]");
          if (!render_) return fail(dvsim::Err::solver_error, "no renderer installed");
          Out o;
          o.image = render_(*solver_, cam, r->width(), r->height());  // throws (-> solver_error) if there is no GPU
          return o;
        }
        case dvsim::Command::Raycast: {
          const dvsim::Raycast* r = env.cmd_as_Raycast();
          constexpr std::size_t kMaxRays = 1u << 20;
          if (!r->origin() || r->origin()->size() != 3 || (r->directions() && r->directions()->size() % 3 != 0))
            return fail(dvsim::Err::invalid_argument, "Raycast needs origin (3 doubles) and directions (3n doubles)");
          const std::size_t n = r->directions() ? r->directions()->size() / 3 : 0;
          if (n == 0) return fail(dvsim::Err::invalid_argument, "Raycast needs at least one direction");
          if (n > kMaxRays) return fail(dvsim::Err::limit_exceeded, "Raycast has more than " + std::to_string(kMaxRays) + " rays");
          Out o;
          try {
            o.distances = solver_->raycast({r->origin()->data(), 3}, {r->directions()->data(), r->directions()->size()});
          } catch (const std::invalid_argument& e) {
            return fail(dvsim::Err::invalid_argument, e.what());
          }
          return o;
        }
        case dvsim::Command::LoadModel: {
          std::string resolved;
          const std::string path = str(env.cmd_as_LoadModel()->path());
          if (!confined(session.root, path, resolved))
            return fail(dvsim::Err::permission_denied, "LoadModel path '" + path + "' is outside the session root");
          solver_ = load_(resolved);  // a throwing loader leaves the old solver in place
          tick_ = 0;
          snapshots_.clear();  // old snapshots do not fit the new model
          return {};
        }
        case dvsim::Command::Spawn: {
          const dvsim::Spawn* s = env.cmd_as_Spawn();
          std::string resolved;
          const std::string path = str(s->path());
          if (!confined(session.root, path, resolved))
            return fail(dvsim::Err::permission_denied, "Spawn path '" + path + "' is outside the session root");
          std::vector<double> pos, quat;
          if (s->pos()) pos.assign(s->pos()->begin(), s->pos()->end());
          if (s->quat()) quat.assign(s->quat()->begin(), s->quat()->end());
          return from(solver_->spawn(resolved, str(s->at()), pos, quat));  // snapshots stay; restore rejects stale ones
        }
        case dvsim::Command::Delete:
          return from(solver_->remove(str(env.cmd_as_Delete()->path())));
        default:
          return fail(dvsim::Err::invalid_argument, "unsupported command " + cmd);
      }
    } catch (const std::exception& e) {
      return fail(dvsim::Err::solver_error, e.what());
    }
  }();

  // A Sweep changes no state (it runs on clones), so it is not logged: replaying it would only burn compute.
  if (log_ && !cap.empty() && cap != "read" && !swept && out.err != dvsim::Err::permission_denied &&
      out.err != dvsim::Err::limit_exceeded)  // refused at the gate: no state change, not logged
    log_->append(tick0, static_cast<std::uint8_t>(out.err), bytes);

  if (log_) log_->set_final_tick(tick_);

  flatbuffers::FlatBufferBuilder fbb;
  auto msg = fbb.CreateString(out.message), unit = fbb.CreateString(out.unit);
  auto values = fbb.CreateVector(out.values);
  auto prims = fbb.CreateVectorOfStrings(out.prims);
  flatbuffers::Offset<dvsim::TestResult> result;
  if (out.test) result = build_result(fbb, *out.test);
  flatbuffers::Offset<dvsim::SweepResult> sweep;
  if (out.sweep) sweep = build_sweep(fbb, *out.sweep);
  flatbuffers::Offset<flatbuffers::Vector<std::uint8_t>> image;
  flatbuffers::Offset<flatbuffers::Vector<float>> depth;
  if (out.image) {
    image = fbb.CreateVector(out.image->rgba);
    depth = fbb.CreateVector(out.image->depth);
  }
  auto distances = fbb.CreateVector(out.distances);
  dvsim::ReplyBuilder rb(fbb);
  rb.add_id(id);
  rb.add_ok(out.err == dvsim::Err::none);
  rb.add_error(out.err);
  rb.add_message(msg);
  rb.add_values(values);
  rb.add_unit(unit);
  rb.add_prims(prims);
  rb.add_tick(tick_);
  rb.add_snapshot_id(out.snapshot_id);
  if (out.test) rb.add_result(result);
  if (out.sweep) rb.add_sweep(sweep);
  if (out.image) {
    rb.add_width(out.image->width);
    rb.add_height(out.image->height);
    rb.add_image(image);
    rb.add_depth(depth);
  }
  if (!out.distances.empty()) rb.add_distances(distances);
  fbb.Finish(rb.Finish());
  return fbb.Release();
}

std::string execute_json(World& w, std::string_view json, const Session& session) {
  return execute_json([&](std::span<const std::uint8_t> env) { return w.execute(env, session); },
                      [&] { return w.tick(); }, json);
}

std::string execute_json(const EnvelopeExecutor& exec, const std::function<std::uint64_t()>& tick, std::string_view json) {
  flatbuffers::IDLOptions opts;
  opts.strict_json = true;                   // quoted names; skip_unexpected_fields_in_json stays false => unknown field is an error
  opts.output_default_scalars_in_json = true;  // print ok:false, tick:0, ...
  opts.indent_step = -1;                     // one line
  flatbuffers::Parser parser(opts);
  if (!parser.Deserialize(dvsim::EnvelopeBinarySchema::data(), dvsim::EnvelopeBinarySchema::size()))
    return R"({"ok":false,"error":"solver_error","message":"embedded schema failed to load"})";

  flatbuffers::DetachedBuffer reply;
  if (parser.Parse(std::string(json).c_str())) {
    reply = exec({parser.builder_.GetBufferPointer(), parser.builder_.GetSize()});
  } else {  // parser.error_ names the offending field for unknown fields
    flatbuffers::FlatBufferBuilder fbb;
    auto msg = fbb.CreateString(parser.error_);
    dvsim::ReplyBuilder rb(fbb);
    rb.add_error(dvsim::Err::invalid_argument);
    rb.add_message(msg);
    rb.add_tick(tick());
    fbb.Finish(rb.Finish());
    reply = fbb.Release();
  }
  parser.SetRootType("dvsim.Reply");
  std::string text;
  if (const char* err = flatbuffers::GenText(parser, reply.data(), &text))
    return std::string(R"({"ok":false,"error":"solver_error","message":")") + err + "\"}";
  return text;
}
