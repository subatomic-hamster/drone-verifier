#pragma once
// Replay log. Little-endian, no timestamps, nothing nondeterministic:
//   magic[8] "DVSIMLOG\0" | u32 format_version | u64 schema_hash (FNV-1a of the embedded .bfbs)
//   | u32 n + solver version string | u64 n + model bytes (Solver::model_bytes)
//   | u64 n + initial state (World::save: tick + solver state)
//   then records: u64 applied_tick | u8 expected_err | u32 n + dvsim.Envelope bytes,
//   then a 24-byte footer: u64 record count | u64 final tick | u64 FNV-1a of every preceding byte.
// final_tick is the live World's tick when the log was read: replay steps to it after the last record, so
// ticks a Host advanced after the final command (real-time Run) are reproduced.
// Replay checks the checksum (and count) before any model byte reaches a solver loader, so truncation
// (even at a record boundary) and bit rot anywhere are detected. Not a MAC: no protection from forgery.
// A record is written for every non-read command that reached dispatch, whatever its outcome
// (a failed Step may still have advanced a diverging solver); commands refused by the capability or
// step-limit gate change nothing and are not logged. expected_err (dvsim.Err of the live reply) is
// re-checked on replay, so any divergence in behaviour is caught at the record that caused it.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// v4: Spawn / Delete commands. The model blob is opaque to the core; the MuJoCo solver's blob is now
// u64 n + mj_saveModel bytes, then the model spec as MJCF XML (17-digit numbers; rest of the blob; may be empty).
// mj_saveModel carries no mjSpec, which a replayed Spawn / Delete needs to attach / remove bodies, so replay rebuilds
// the spec from the XML on the first structural command (a log without one never touches it).
constexpr std::uint32_t kReplayFormatVersion = 4;
constexpr std::size_t kReplayFooterSize = 24;

std::uint64_t schema_hash();  // FNV-1a 64 over the embedded dvsim.bfbs
std::uint64_t fnv1a64(std::span<const std::byte> data);

class ReplayLog {
 public:
  ReplayLog(std::string_view solver_version, std::span<const std::byte> model, std::span<const std::byte> state);
  void append(std::uint64_t applied_tick, std::uint8_t expected_err, std::span<const std::uint8_t> envelope);
  void set_final_tick(std::uint64_t t) { final_tick_ = t; }  // World keeps this current
  std::vector<std::byte> bytes() const;  // complete log: body + footer (a copy)
  void write(const std::filesystem::path& p) const;  // throws std::runtime_error on I/O failure

 private:
  std::vector<std::byte> buf_;  // body, no footer
  std::uint64_t count_ = 0, final_tick_ = 0;
};
