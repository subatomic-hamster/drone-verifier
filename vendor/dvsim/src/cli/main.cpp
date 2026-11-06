// dvsim-replay <log> [--record <out>] [--root <dir>]   (the Python CLI owns the name `dvsim`)
// Replays a replay log into a fresh World; prints the final tick and state hash.
#include <cinttypes>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>

#include "core/world.h"
#include "mujoco/mujoco_solver.h"

namespace {
int usage() {
  std::fprintf(stderr, "usage: dvsim-replay <log> [--record <out>] [--root <dir>]\n");
  return 2;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  std::string log_path = argv[1], record_path, root;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if ((a == "--record" || a == "--root") && i + 1 < argc) (a == "--record" ? record_path : root) = argv[++i];
    else return usage();
  }
  try {
    std::ifstream f(log_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open '" + log_path + "'");
    const std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) bytes[i] = static_cast<std::byte>(raw[i]);
    World w = replay(bytes, load_mujoco_bytes, load_mujoco, root, !record_path.empty());
    if (!record_path.empty()) w.log()->write(record_path);
    std::printf("tick=%" PRIu64 " state_hash=%016" PRIx64 "\n", w.tick(), w.state_hash());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dvsim-replay: error: %s\n", e.what());
    return 1;
  }
}
