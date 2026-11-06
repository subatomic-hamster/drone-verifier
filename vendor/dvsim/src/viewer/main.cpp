// dvsim-view <model> [--frames N] [--script "FRAME ACTION ARGS; ..."] [--log OUT] [--root DIR] [--design REPORT.json] [--offscreen WxH]
// A model with <name>.report.json beside it (from `dvsim design build`) also gets the Design window (--design: other path).
// Opens the model (MJCF/URDF) in the viewer. --frames N: leave after N frames; --script: UI-equivalent commands issued at
// given frames (see viewer/viewer.h); --log: write the session's replay log on exit (dvsim-replay reproduces it).
// Prints the final tick and state hash, in dvsim-replay's format.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

#include "core/host.h"
#include "mujoco/mujoco_solver.h"
#include "viewer/viewer.h"

namespace {
int usage() {
  std::fprintf(stderr, "usage: dvsim-view <model> [--frames N] [--script \"FRAME ACTION ARGS; ...\"] [--log OUT] [--root DIR] [--design REPORT.json] [--offscreen WxH]\n");
  return 2;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string model = argv[1];
  ViewerOptions opts;
  std::string log_path, root;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) return usage();
    const std::string v = argv[++i];
    if (a == "--frames") opts.frames = std::atoi(v.c_str());
    else if (a == "--script") opts.script = v;
    else if (a == "--log") log_path = v;
    else if (a == "--root") root = v;
    else if (a == "--design") opts.design_report = v;
    else if (a == "--offscreen") {  // WxH: no window / X server; frames render on the best Vulkan device (see ViewerOptions::offscreen)
      opts.offscreen = true;
      if (std::sscanf(v.c_str(), "%dx%d", &opts.width, &opts.height) != 2 || opts.width < 1 || opts.height < 1) return usage();
    }
    else return usage();
  }
  if (opts.design_report.empty()) {  // <name>.xml next to <name>.report.json = built by `dvsim design build`
    const auto rep = std::filesystem::path(model).replace_extension(".report.json");
    if (std::filesystem::is_regular_file(rep)) opts.design_report = rep.string();
  }
  try {
    parse_script(opts.script);  // a bad script fails before anything else happens
    Session session;
    session.caps = {"read", "control"};
    session.root = root.empty() ? std::filesystem::path(model).parent_path() : std::filesystem::path(root);
    Host host(World(load_mujoco, model), session);
    try {
      run_viewer(host, opts);
    } catch (...) {
      host.stop();
      throw;
    }
    host.stop();
    if (!log_path.empty()) host.world().log()->write(log_path);
    std::printf("tick=%" PRIu64 " state_hash=%016" PRIx64 "\n", host.world().tick(), host.world().state_hash());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dvsim-view: error: %s\n", e.what());
    return 1;
  }
}
