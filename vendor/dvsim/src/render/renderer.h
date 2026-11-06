#pragma once
// Headless Vulkan 1.3 renderer (dynamic rendering, no window system, no swapchain): procedural sky, sun + hemisphere
// light with Blinn-Phong/Fresnel specular from the MuJoCo material, one PCF sun shadow map fitted to the scene, 4x MSAA,
// sRGB output, optional CAD-style crease edges, cables as tubes. Offscreen RGBA8 colour + an exact linear-depth (R32F)
// pass, CPU readback.
//
// Precision: poses arrive in double. Per geom the renderer subtracts the camera position IN DOUBLE, and everything
// downstream (view, light, fog, sky) works on those camera-relative floats -- so a scene (and camera) 1e5 m from the
// origin renders like one at the origin.
//
// Swapchain reuse (M10): everything that draws is Impl::record(), which takes attachment views and a command buffer
// and knows nothing about readback; a viewer adds a swapchain image view as colour target and skips the readback.
#include <memory>
#include <span>
#include <string>

#include "core/render.h"
#include "core/solver.h"

class WindowRenderer;  // window_renderer.h: shares Impl (the draw path) with the headless renderer

// How the scene looks. "outdoor": Waterloo-like sky (warm bright horizon, clouds, sun glare), dark-grey ground fading into
// the haze. "studio": neutral gradient backdrop, light floor with a fading metre grid (the design / build view).
struct RenderStyle {
  enum class Sky { outdoor, studio } sky = Sky::outdoor;
  bool shadows = true;  // the sun shadow map
  bool edges = false;   // dark crease edges (CAD look)
  bool tubes = true;    // draw cables (TubeVisual)
  float wire_px = 0;    // > 0: cables are never thinner than this many pixels on screen (the data keeps their true diameter); 0: ~1.4 px floor
  // Weather override: rain rate in mm/h (< 0: take it from the scene's rain device, via Solver::weather()) and overcast 0..1
  // (< 0: derived from the rain rate). 0 / 0 = the clear look. Rain: overcast sky, dim sun, denser haze, wet ground, streaks.
  float rain = -1, overcast = -1;
};
// "studio,edges,noshadows,nowires,boldwires,rain=25,overcast=0.6" (comma separated; "outdoor" "shadows" "noedges" "wires" "plainwires"
// are the opposites) applied on top of `base`; "boldwires" = cables at least 4 px wide on screen.
// Unknown words throw std::invalid_argument.
RenderStyle parse_render_style(const std::string& words, RenderStyle base = {});
// Process-wide default every Renderer / WindowRenderer starts with and re-reads per frame unless it was given its own
// (set_style). DVSIM_RENDER_STYLE in the environment is applied on top of it.
void set_default_render_style(const RenderStyle& s);
RenderStyle default_render_style();

class Renderer {
  friend class WindowRenderer;

 public:
  // Picks the first Vulkan 1.3 device with dynamic rendering (the loader honours VK_DRIVER_FILES). Throws
  // std::runtime_error with a plain-language reason when there is no loader, driver or suitable device.
  Renderer();
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  const std::string& device_name() const;

  // Tessellates (primitives, once) and uploads the scene; draw() then needs one pose per visual.
  void set_scene(std::span<const GeomVisual> visuals);
  void set_tubes(std::span<const TubeVisual> tubes);  // the cables of the next draw() (copied)
  void set_style(const RenderStyle& s);               // overrides the process default
  void set_weather(const Weather& w);                 // the rain look of the next draw() (render(solver) sets it from the solver)
  // Throws std::invalid_argument for bad sizes / a pose count that differs from the scene.
  Image draw(std::span<const GeomPose> poses, const Camera& cam, int width, int height);
  // Convenience: re-uploads the scene only if solver.visuals() changed since the last call; takes the cables from it.
  Image render(const Solver& solver, const Camera& cam, int width, int height);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  std::uint64_t scene_hash_ = 0;
};

// Hook for World::set_render_hook: one process-wide Renderer, created on the first render (so merely installing the
// hook never touches Vulkan), serialised by a mutex. A failed creation throws every time until a device exists.
RenderHook make_render_hook();
