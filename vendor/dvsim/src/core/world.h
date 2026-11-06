#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/render.h"
#include "core/replay.h"
#include "core/solver.h"
#include "flatbuffers/flatbuffers.h"

struct Session {
  std::vector<std::string> caps;      // subset of "read", "control", "destructive"
  int max_steps_per_call = 100000;
  int max_sweep_runs = 1000;          // Sweep with more runs: limit_exceeded before any work
  // Wall-clock budget of one RunTest / Sweep (0 = none). On expiry the command fails with limit_exceeded and leaves
  // no trace: RunTest rolls the world back to its pre-test state, Sweep returns no partial rows. Results of a
  // command that finishes in time never depend on it.
  double max_wall_s = 0;
  std::filesystem::path root;         // LoadModel paths are confined under this
  bool has(std::string_view cap) const {
    for (const std::string& c : caps) if (c == cap) return true;
    return false;
  }
};

// Single-threaded. Owns one Solver and the tick clock.
class World {
 public:
  using Loader = std::function<std::unique_ptr<Solver>(const std::string& path)>;

  World(const Loader& load, const std::string& path) : load_(load), solver_(load(path)) {}

  void step(int n = 1) {
    tick_ += static_cast<std::uint64_t>(solver_->step(n));
    if (log_) log_->set_final_tick(tick_);  // ticks advanced outside commands (Host Run) still reach the log footer
  }
  std::uint64_t tick() const { return tick_; }
  bool diverged() const { return solver_->diverged(); }
  Solver& solver() { return *solver_; }
  const Solver& solver() const { return *solver_; }

  std::vector<std::byte> save() const;                    // tick (8 bytes) + solver bytes
  void restore(std::span<const std::byte> bytes);         // throws on malformed input
  World clone() const {
    World c(load_, solver_->clone(), tick_);
    c.render_ = render_;  // renders are pure reads; the hook serialises its own GPU use
    return c;
  }

  // Installs the renderer used by Render commands (none: Render fails with solver_error "no renderer installed").
  void set_render_hook(RenderHook h) { render_ = std::move(h); }
  std::uint64_t state_hash() const { return solver_->state_hash(); }

  World(Loader load, std::unique_ptr<Solver> s, std::uint64_t tick = 0)
      : load_(std::move(load)), solver_(std::move(s)), tick_(tick) {}

  // The bus: verifies the raw dvsim.Envelope bytes (trust boundary), checks the command's `cap`
  // (schema reflection) against the session, runs it, returns a finished dvsim.Reply buffer.
  // Never throws for command-level failures. Records non-read commands when a log is active.
  flatbuffers::DetachedBuffer execute(std::span<const std::uint8_t> envelope, const Session& session);

  // Begin recording. Header captures model + current state, so call it on a fresh World (no
  // snapshots yet: their ids would not exist on replay). Throws std::logic_error otherwise.
  void start_log();
  const ReplayLog* log() const { return log_.get(); }

 private:
  Loader load_;
  RenderHook render_;
  std::unique_ptr<ReplayLog> log_;
  std::unique_ptr<Solver> solver_;
  std::uint64_t tick_ = 0;
  std::map<std::uint64_t, std::vector<std::byte>> snapshots_;
  std::uint64_t next_snapshot_ = 1;
};

// The `cap` attribute ("read" | "control" | "destructive") of a command table by name; "" if unknown.
std::string command_cap(std::string_view command);

// JSON edge (agent-facing): strict parse of a dvsim.Envelope, execute, print the dvsim.Reply.
// Malformed JSON / unknown fields yield a Reply with ok:false (message carries the parser text).
std::string execute_json(World& w, std::string_view json, const Session& session);

// Same, with the executor supplied (a Host routes it to its sim thread). `tick` is read only to fill the reply to
// unparsable JSON.
using EnvelopeExecutor = std::function<flatbuffers::DetachedBuffer(std::span<const std::uint8_t>)>;
std::string execute_json(const EnvelopeExecutor& exec, const std::function<std::uint64_t()>& tick, std::string_view json);

// Rebuilds a Solver from Solver::model_bytes().
using BytesLoader = std::function<std::unique_ptr<Solver>(std::span<const std::byte>)>;

// Replays a log (untrusted input) into a fresh World and returns it. Verifies magic, format version,
// schema hash and solver version, runs flatbuffers::Verifier on every record, steps up to each
// record's applied_tick (ticks that advanced outside commands, e.g. Host Run), executes it with an
// all-caps session whose `root` confines LoadModel paths, and checks the reply error equals the
// logged one. `record` re-records into the returned World's log. Throws std::runtime_error with a
// message naming the failing record.
World replay(std::span<const std::byte> log, const BytesLoader& from_bytes, const World::Loader& load,
             const std::filesystem::path& root, bool record = false);
