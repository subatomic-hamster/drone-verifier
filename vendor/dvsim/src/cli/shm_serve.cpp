// dvsim-shm-serve <model> --name /dvsim-xyz [--timeout s] [--log out]
// Serves a shm lockstep block (include/dvsim/shm_client.hpp) on a World of <model>: ctrl[] = the actuators, obs[] = qpos
// and qvel of every joint. Exits when the client sends quit (rc 0) or after --timeout idle seconds (rc 1).
#include <cinttypes>
#include <cstdio>
#include <exception>
#include <string>

#include "core/shm.h"
#include "mujoco/mujoco_solver.h"

namespace {
int usage() {
  std::fprintf(stderr, "usage: dvsim-shm-serve <model> --name /dvsim-xyz [--timeout s] [--log out]\n");
  return 2;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string model = argv[1];
  std::string name, log;
  double timeout = 30;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) return usage();
    if (a == "--name") name = argv[++i];
    else if (a == "--log") log = argv[++i];
    else if (a == "--timeout") timeout = std::stod(argv[++i]);
    else return usage();
  }
  if (name.empty()) return usage();
  try {
    World w(load_mujoco, model);
    if (!log.empty()) w.start_log();
    Session s;
    s.caps = {"read", "control"};
    ShmServer::Obs obs;
    for (const Prim& p : w.solver().prims())
      if (p.kind == "joint") obs.insert(obs.end(), {{p.path, "qpos"}, {p.path, "qvel"}});
    ShmServer server(w, s, name, obs);
    std::printf("ready %s\n", name.c_str());
    std::fflush(stdout);
    const std::uint64_t rounds = server.serve(timeout);
    if (!log.empty()) w.log()->write(log);
    std::printf("rounds=%" PRIu64 " tick=%" PRIu64 " state_hash=%016" PRIx64 "\n", rounds, w.tick(), w.state_hash());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dvsim-shm-serve: error: %s\n", e.what());
    return 1;
  }
}
