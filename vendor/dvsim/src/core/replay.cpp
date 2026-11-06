// Replay log writer and replayer. Format is documented in replay.h.
#include "core/replay.h"

#include <bit>
#include <climits>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "core/world.h"
#include "flatbuffers/flatbuffers.h"
#include "dvsim_generated.h"

static_assert(std::endian::native == std::endian::little, "replay log format is little-endian");

namespace {
constexpr char kMagic[8] = {'N', 'A', 'P', 'L', 'L', 'O', 'G', '\0'};

template <class T>
void put(std::vector<std::byte>& b, T v) {
  const auto* p = reinterpret_cast<const std::byte*>(&v);
  b.insert(b.end(), p, p + sizeof v);
}
void put_bytes(std::vector<std::byte>& b, const void* p, std::size_t n) {
  const auto* q = static_cast<const std::byte*>(p);
  b.insert(b.end(), q, q + n);
}

// Bounds-checked cursor over untrusted bytes; every failure throws with the field name.
struct Reader {
  std::span<const std::byte> d;
  std::size_t p = 0;
  std::size_t left() const { return d.size() - p; }
  template <class T>
  T num(const char* what) {
    if (left() < sizeof(T)) throw std::runtime_error(std::string("truncated log: missing ") + what);
    T v;
    std::memcpy(&v, d.data() + p, sizeof v);
    p += sizeof v;
    return v;
  }
  std::span<const std::byte> take(std::uint64_t n, const char* what) {
    if (n > left()) throw std::runtime_error(std::string("truncated log: ") + what + " claims " + std::to_string(n) +
                                             " bytes but only " + std::to_string(left()) + " remain");
    auto s = d.subspan(p, static_cast<std::size_t>(n));
    p += static_cast<std::size_t>(n);
    return s;
  }
};
}  // namespace

std::uint64_t schema_hash() {
  std::uint64_t h = 14695981039346656037ull;
  const std::uint8_t* s = dvsim::EnvelopeBinarySchema::data();
  for (std::size_t i = 0; i < dvsim::EnvelopeBinarySchema::size(); ++i) h = (h ^ s[i]) * 1099511628211ull;
  return h;
}

std::uint64_t fnv1a64(std::span<const std::byte> d) {
  std::uint64_t h = 14695981039346656037ull;
  for (const std::byte b : d) h = (h ^ static_cast<std::uint8_t>(b)) * 1099511628211ull;
  return h;
}

std::vector<std::byte> ReplayLog::bytes() const {
  std::vector<std::byte> b = buf_;
  put<std::uint64_t>(b, count_);
  put<std::uint64_t>(b, final_tick_);
  put<std::uint64_t>(b, fnv1a64(b));
  return b;
}

ReplayLog::ReplayLog(std::string_view version, std::span<const std::byte> model, std::span<const std::byte> state) {
  put_bytes(buf_, kMagic, sizeof kMagic);
  put<std::uint32_t>(buf_, kReplayFormatVersion);
  put<std::uint64_t>(buf_, schema_hash());
  put<std::uint32_t>(buf_, static_cast<std::uint32_t>(version.size()));
  put_bytes(buf_, version.data(), version.size());
  put<std::uint64_t>(buf_, model.size());
  put_bytes(buf_, model.data(), model.size());
  put<std::uint64_t>(buf_, state.size());
  put_bytes(buf_, state.data(), state.size());
}

void ReplayLog::append(std::uint64_t tick, std::uint8_t err, std::span<const std::uint8_t> env) {
  put<std::uint64_t>(buf_, tick);
  put<std::uint8_t>(buf_, err);
  put<std::uint32_t>(buf_, static_cast<std::uint32_t>(env.size()));
  put_bytes(buf_, env.data(), env.size());
  ++count_;
}

void ReplayLog::write(const std::filesystem::path& p) const {
  const std::vector<std::byte> b = bytes();
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  if (!f) throw std::runtime_error("cannot write log '" + p.string() + "'");
}

World replay(std::span<const std::byte> log, const BytesLoader& from_bytes, const World::Loader& load,
             const std::filesystem::path& root, bool record) {
  Reader r{log};
  if (r.left() < sizeof kMagic || std::memcmp(r.d.data(), kMagic, sizeof kMagic) != 0)
    throw std::runtime_error("not a dvsim replay log (bad magic)");
  r.p = sizeof kMagic;
  const auto fmt = r.num<std::uint32_t>("format version");
  if (fmt != kReplayFormatVersion)
    throw std::runtime_error("unsupported log format version " + std::to_string(fmt) + " (this build reads " +
                             std::to_string(kReplayFormatVersion) + ")");
  const auto hash = r.num<std::uint64_t>("schema hash");
  if (hash != schema_hash()) throw std::runtime_error("log was written with a different command schema (schema hash mismatch)");
  // Footer first: nothing below (model bytes included) is trusted until the checksum over all of it passes.
  std::uint64_t count = 0, final_tick = 0, sum = 0;
  if (r.left() >= kReplayFooterSize) {
    std::memcpy(&count, log.data() + log.size() - 24, 8);
    std::memcpy(&final_tick, log.data() + log.size() - 16, 8);
    std::memcpy(&sum, log.data() + log.size() - 8, 8);
  }
  if (r.left() < kReplayFooterSize || fnv1a64(log.first(log.size() - 8)) != sum)
    throw std::runtime_error("truncated or corrupt log: checksum mismatch (missing footer, truncated, or damaged)");
  r.d = log.first(log.size() - kReplayFooterSize);
  const auto vlen = r.num<std::uint32_t>("solver version length");
  const auto vbytes = r.take(vlen, "solver version");
  const std::string version(reinterpret_cast<const char*>(vbytes.data()), vbytes.size());
  const auto model = r.take(r.num<std::uint64_t>("model length"), "model bytes");
  const auto state = r.take(r.num<std::uint64_t>("state length"), "initial state");

  World w(load, from_bytes(model));
  if (w.solver().version() != version)
    throw std::runtime_error("solver version mismatch: log has '" + version + "', this build has '" +
                             w.solver().version() + "'");
  w.restore(state);
  if (record) w.start_log();

  Session s;
  s.caps = {"read", "control", "destructive"};
  s.max_steps_per_call = INT_MAX;
  s.root = root;
  // Steps through ticks advanced outside commands (Host Run) up to `target`.
  auto advance = [&](std::uint64_t target, const std::string& rec) {
    if (w.tick() > target)
      throw std::runtime_error(rec + "tick " + std::to_string(target) + " is behind world tick " + std::to_string(w.tick()));
    while (w.tick() < target) {
      const std::uint64_t before = w.tick(), gap = target - before;
      w.step(static_cast<int>(gap < 1000000000 ? gap : 1000000000));
      if (w.tick() == before)
        throw std::runtime_error(rec + "solver diverged before tick " + std::to_string(target));
    }
  };
  std::uint64_t i = 0;
  for (; r.left() > 0; ++i) {
    const std::string rec = "record " + std::to_string(i) + ": ";
    const auto applied = r.num<std::uint64_t>("applied_tick");
    const auto want = r.num<std::uint8_t>("expected_err");
    const auto n = r.num<std::uint32_t>("envelope length");
    const auto src = r.take(n, "envelope");
    std::vector<std::uint8_t> env(n);  // heap copy: Verifier wants an aligned buffer
    if (n) std::memcpy(env.data(), src.data(), n);
    flatbuffers::Verifier v(env.data(), env.size());
    if (!dvsim::VerifyEnvelopeBuffer(v)) throw std::runtime_error(rec + "corrupt envelope (flatbuffers verification failed)");

    advance(applied, rec);
    const flatbuffers::DetachedBuffer reply = w.execute(env, s);
    const dvsim::Reply* rep = flatbuffers::GetRoot<dvsim::Reply>(reply.data());  // our own output: trusted
    const auto got = static_cast<std::uint8_t>(rep->error());
    if (got != want)
      throw std::runtime_error(rec + "replay diverged from the live run: error " + dvsim::EnumNameErr(rep->error()) +
                               ", log expected " + dvsim::EnumNameErr(static_cast<dvsim::Err>(want)) + " (" +
                               rep->message()->str() + ")");
  }
  if (i != count)
    throw std::runtime_error("log footer claims " + std::to_string(count) + " records, found " + std::to_string(i));
  advance(final_tick, "final tick: ");
  return w;
}
