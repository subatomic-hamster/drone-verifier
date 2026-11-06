#pragma once
// Server side of the shm lockstep block (layout, protocol and the client: include/dvsim/shm_client.hpp).
// Runs on the thread that owns the World (World is single-threaded); every ctrl change and Step goes through the
// command bus (World::execute, source "shm"), so a started replay log reproduces the run.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/world.h"
#include "dvsim/shm_client.hpp"

class ShmServer {
 public:
  using Obs = std::vector<std::pair<std::string, std::string>>;  // (prim, attr) reads, concatenated into obs[]

  // Creates the segment `name` ("/dvsim-xyz"; mode 0600, exclusive: a leftover one is an error), ctrl[i] = actuator i
  // (Solver::prims() order). `session` supplies the capabilities (needs "control") and step limit. Requests above
  // max_steps ticks are refused with status kBadRequest. Throws std::runtime_error. `w` must outlive the server.
  ShmServer(World& w, const Session& session, const std::string& name, Obs obs, std::uint32_t max_steps = 100000);
  ~ShmServer();  // tells a waiting client the server closed, unmaps and shm_unlink()s the segment
  ShmServer(const ShmServer&) = delete;
  ShmServer& operator=(const ShmServer&) = delete;

  // Serves rounds until the client sends quit; returns the number of step rounds. Throws std::runtime_error if no
  // request arrives within idle_timeout_s (client died / never came): the caller then unwinds and the destructor
  // cleans up.
  std::uint64_t serve(double idle_timeout_s);

 private:
  void round(std::uint32_t n);

  World& w_;
  Session session_;
  std::string name_;
  Obs obs_;
  std::uint32_t max_steps_;
  std::vector<std::string> actuators_;
  dvsim::shm::Header* h_ = nullptr;
  std::uint64_t size_ = 0;
  std::uint32_t served_ = 0;  // req_seq of the last answered request (a fresh segment starts at 0)
};
