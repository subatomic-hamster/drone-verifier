#pragma once
// Deterministic ctrl sequence shared by test_shm.cpp (server + in-process reference) and shm_test_client.cpp (child).
#include <cstdint>

// splitmix64 of (seed, round, i) mapped to [-5, 5] (the pendulum motor's ctrlrange).
inline double shm_ctrl_at(std::uint64_t seed, std::uint64_t round, int i) {
  std::uint64_t z = seed * 0x9e3779b97f4a7c15ull + round * 0xbf58476d1ce4e5b9ull + static_cast<std::uint64_t>(i) + 1;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) / 9007199254740992.0 * 10.0 - 5.0;
}
