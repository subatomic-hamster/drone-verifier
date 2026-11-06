#pragma once
// dvsim shared-memory lockstep: the block layout, the futex helpers and a header-only client. No dependencies beyond
// the C++20 standard library and POSIX/Linux, so third-party code (a C++ controller, ...) can include just this file.
// The server side is src/core/shm.{h,cpp}; it includes this header, so the layout has one definition.
//
// Segment (POSIX shm_open name like "/dvsim-xyz", mode 0600, same user only):
//   Header (80 bytes) | double ctrl[nu] (client -> server) | double obs[nobs] (server -> client)
// One round: the client writes cmd, n_steps and ctrl[], then publishes req_seq = seq (release store). The server
// (acquire load) applies ctrl to the actuators in order through the command bus (logged, so the server's replay log
// reproduces the run), advances n_steps ticks (0 = observe only), fills status/tick/state_hash/obs[], then publishes
// resp_seq = seq (release store). Waiting is FUTEX_WAIT/FUTEX_WAKE on those two words WITHOUT the PRIVATE flag: private
// futexes key on (mm, address) and never wake a waiter in another process. (libstdc++'s std::atomic::wait uses the
// private flag, so it is not used here.) A lost wake-up can only delay a waiter until its timeout, never corrupt data.
// Every wait has a timeout; the segment is a trust boundary and attach() validates it before use.
// status: 0 ok; 1..5 = dvsim.Err of the failing command (invalid_argument, not_found, permission_denied, limit_exceeded,
// solver_error: the world diverged); kBadRequest = n_steps above the server's limit or unknown cmd.
// Single client at a time. Linux only (futex); macOS/Windows need another wait primitive (follow-up).
#ifndef __linux__
#error "dvsim shm lockstep is Linux-only for now"
#endif
#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

namespace dvsim::shm {

constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kStep = 0, kQuit = 1;  // Header::cmd
constexpr std::uint32_t kBadRequest = 100;     // Header::status
constexpr std::uint32_t kAny = 0xffffffffu;    // Client: accept whatever nu / nobs the server has
constexpr std::uint64_t kMaxSegment = 1u << 30;

struct Header {
  char magic[8];  // "DVSIMSHM\0"
  std::uint32_t version, header_size, nu, nobs;
  std::uint64_t total_size;  // header + 8 * (nu + nobs)
  std::atomic<std::uint32_t> ready;     // server: 1 once everything above is initialised
  std::atomic<std::uint32_t> attached;  // client: 1 while a client holds the segment
  std::atomic<std::uint32_t> closed;    // server: 1 once it stopped serving
  std::atomic<std::uint32_t> req_seq;   // client -> server: round number, written last
  std::uint32_t cmd, n_steps;
  std::atomic<std::uint32_t> resp_seq;  // server -> client: echo of req_seq, written last (also bumped on close)
  std::uint32_t status;
  std::uint64_t tick, state_hash;
};
static_assert(sizeof(Header) == 80 && alignof(Header) == 8);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free, "needs address-free atomics across processes");
inline constexpr char kMagic[8] = {'N', 'A', 'P', 'L', 'S', 'H', 'M', 0};

using Clock = std::chrono::steady_clock;
inline Clock::time_point after(double seconds) {
  return Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

// Sleeps while a == old; true once it differs, false at the deadline. Shared (non-private) futex.
inline bool wait_ne(std::atomic<std::uint32_t>& a, std::uint32_t old, Clock::time_point deadline) {
  for (;;) {
    if (a.load(std::memory_order_acquire) != old) return true;
    const auto left = deadline - Clock::now();
    if (left <= Clock::duration::zero()) return false;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(left).count();
    const timespec ts{static_cast<time_t>(ns / 1000000000), static_cast<long>(ns % 1000000000)};
    syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&a), FUTEX_WAIT, old, &ts, nullptr, 0);  // EAGAIN/EINTR: re-check
  }
}
inline void wake(std::atomic<std::uint32_t>& a) {
  syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&a), FUTEX_WAKE, 1, nullptr, nullptr, 0);
}
inline std::byte* ctrl_base(Header* h) { return reinterpret_cast<std::byte*>(h) + sizeof(Header); }

class Client {
 public:
  struct Reply {
    std::uint32_t status;
    std::uint64_t tick, state_hash;
  };

  // Attaches to the segment `name` (waits up to timeout_s for the server to create and initialise it). Throws
  // std::runtime_error on a missing/foreign/malformed segment (magic, version, sizes), a nu/nobs different from the
  // expected ones, a segment that already has a client, or timeout.
  explicit Client(const std::string& name, std::uint32_t expect_nu = kAny, std::uint32_t expect_nobs = kAny,
                  double timeout_s = 10) {
    const auto deadline = after(timeout_s);
    const auto fail = [&](const std::string& m) { return std::runtime_error("dvsim shm '" + name + "': " + m); };
    int fd = -1;
    while ((fd = shm_open(name.c_str(), O_RDWR, 0)) < 0) {
      if (errno != ENOENT || Clock::now() >= deadline) throw fail(std::string("shm_open: ") + std::strerror(errno));
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    struct stat st{};
    while (fstat(fd, &st) == 0 && st.st_size == 0 && Clock::now() < deadline)  // server between create and ftruncate
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    size_ = static_cast<std::uint64_t>(st.st_size);
    if (size_ < sizeof(Header) || size_ > kMaxSegment) {
      close(fd);
      throw fail("segment size " + std::to_string(size_) + " is not a dvsim block");
    }
    void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) throw fail(std::string("mmap: ") + std::strerror(errno));
    h_ = static_cast<Header*>(p);
    try {
      while (h_->ready.load(std::memory_order_acquire) != 1) {
        if (Clock::now() >= deadline) throw fail("server not ready (timeout)");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (std::memcmp(h_->magic, kMagic, sizeof kMagic) != 0) throw fail("bad magic");
      if (h_->version != kVersion)
        throw fail("layout version " + std::to_string(h_->version) + ", client speaks " + std::to_string(kVersion));
      const std::uint64_t nu = h_->nu, nobs = h_->nobs;
      if (h_->header_size != sizeof(Header) || nu > (kMaxSegment >> 3) || nobs > (kMaxSegment >> 3) ||
          h_->total_size != size_ || size_ != sizeof(Header) + 8 * (nu + nobs))
        throw fail("inconsistent header sizes");
      if ((expect_nu != kAny && expect_nu != nu) || (expect_nobs != kAny && expect_nobs != nobs))
        throw fail("layout mismatch: server nu=" + std::to_string(nu) + " nobs=" + std::to_string(nobs) +
                   ", expected nu=" + std::to_string(expect_nu) + " nobs=" + std::to_string(expect_nobs));
      if (h_->attached.exchange(1) != 0) throw fail("already has a client");
    } catch (...) {
      munmap(h_, size_);
      throw;
    }
    seq_ = h_->req_seq.load(std::memory_order_acquire);
  }
  ~Client() {
    h_->attached.store(0);
    munmap(h_, size_);
  }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  std::uint32_t nu() const { return h_->nu; }
  std::uint32_t nobs() const { return h_->nobs; }
  std::span<double> ctrl() { return {reinterpret_cast<double*>(ctrl_base(h_)), h_->nu}; }  // fill before step()
  std::span<const double> obs() const {  // valid after step()
    return {reinterpret_cast<const double*>(ctrl_base(h_)) + h_->nu, h_->nobs};
  }

  // One lockstep round: apply ctrl(), advance n ticks (0 = just apply and observe), wait for the server. Throws
  // std::runtime_error on timeout or if the server closed; after that the client must be discarded.
  Reply step(std::uint32_t n, double timeout_s = 10) { return round(kStep, n, timeout_s); }
  // Tells the server this client is done (its serve() returns).
  void close_session(double timeout_s = 10) { round(kQuit, 0, timeout_s); }

 private:
  Reply round(std::uint32_t cmd, std::uint32_t n, double timeout_s) {
    const auto deadline = after(timeout_s);
    h_->cmd = cmd;
    h_->n_steps = n;
    h_->req_seq.store(++seq_, std::memory_order_release);
    wake(h_->req_seq);
    for (;;) {
      const std::uint32_t r = h_->resp_seq.load(std::memory_order_acquire);
      if (r == seq_) break;
      if (h_->closed.load(std::memory_order_acquire) != 0) throw std::runtime_error("dvsim shm: server closed");
      if (!wait_ne(h_->resp_seq, r, deadline)) throw std::runtime_error("dvsim shm: timeout waiting for the server");
    }
    return {h_->status, h_->tick, h_->state_hash};
  }

  Header* h_ = nullptr;
  std::uint64_t size_ = 0;
  std::uint32_t seq_ = 0;
};

}  // namespace dvsim::shm
