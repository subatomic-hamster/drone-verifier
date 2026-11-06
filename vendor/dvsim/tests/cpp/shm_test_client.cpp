// Child process of test_shm.cpp: drives a lockstep run through the header-only client ONLY (no dvsim/mujoco/flatbuffers).
// usage: shm_test_client <name> <seed> <rounds> [die_after]   die_after: SIGKILL itself after that many rounds.
// Prints "tick=<n> hash=<hex> obs0=<%.17g> maxlat_us=<n>" on success.
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "dvsim/shm_client.hpp"
#include "shm_ctrl.h"

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  try {
    dvsim::shm::Client c(argv[1]);
    const std::uint64_t seed = std::strtoull(argv[2], nullptr, 10), rounds = std::strtoull(argv[3], nullptr, 10);
    const std::uint64_t die_after = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : ~0ull;
    dvsim::shm::Client::Reply r{};
    double max_us = 0;
    for (std::uint64_t k = 0; k < rounds; ++k) {
      if (k == die_after) std::raise(SIGKILL);
      for (std::uint32_t i = 0; i < c.nu(); ++i) c.ctrl()[i] = shm_ctrl_at(seed, k, static_cast<int>(i));
      const auto t0 = dvsim::shm::Clock::now();
      r = c.step(1);
      max_us = std::max(max_us, std::chrono::duration<double, std::micro>(dvsim::shm::Clock::now() - t0).count());
      if (r.status != 0) return 3;
    }
    c.close_session();
    std::printf("tick=%llu hash=%016llx obs0=%.17g maxlat_us=%.0f\n", static_cast<unsigned long long>(r.tick),
                static_cast<unsigned long long>(r.state_hash), c.obs()[0], max_us);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "shm_test_client: %s\n", e.what());
    return 1;
  }
}
