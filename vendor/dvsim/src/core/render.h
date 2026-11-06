#pragma once
// Rendering types shared by the bus and the renderer. dvsim_core knows nothing about Vulkan: the app (Python
// binding, dvsim_render's make_render_hook) installs a RenderHook into the World; Render commands call it.
#include <cstdint>
#include <functional>
#include <vector>

#include "core/solver.h"

// World +z is up. The camera looks from `pos` at `lookat`; fovy is the vertical field of view in degrees.
struct Camera {
  double pos[3] = {0, 0, 0};
  double lookat[3] = {1, 0, 0};
  double fovy = 45;
};

// Row 0 is the top row. rgba: width*height*4 bytes. depth: width*height floats = linear depth in metres along
// the optical axis (not ray length); +inf where nothing was hit.
struct Image {
  int width = 0, height = 0;
  std::vector<std::uint8_t> rgba;
  std::vector<float> depth;
};

// May throw std::runtime_error (e.g. no Vulkan device); the bus turns that into a solver_error reply.
using RenderHook = std::function<Image(const Solver&, const Camera&, int width, int height)>;
