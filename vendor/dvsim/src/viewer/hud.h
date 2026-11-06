#pragma once
// A data-driven HUD for the viewer's ViewerOptions::overlay hook. The producer (python: dvsim.hud / dvsim.cuas.hud, or any native
// code) hands over a JSON document -- screen-space boxes, texts, lines, images, and world-anchored labels / polylines /
// markers -- and the viewer draws the latest one each frame on top of the 3D scene. Nothing here knows about turrets: a rocket,
// a boat or an obstacle-avoiding drone shows its own panels with the same primitives. Document (every number is a JSON number):
//
//   { "seq": N, "tick": T?, "camera": {"target":[x,y,z], "yaw":rad, "pitch":rad, "dist":m}?,   // camera: drives the orbit view
//     "items": [ item, ... ] }
//
// Items are drawn in order; positions are in window pixels relative to the viewport (or the enclosing group) and measured
// from the anchor corner of the item. Common fields:
//   "at":[x,y]   position;   "a":"tl|t|tr|l|c|r|bl|b|br"   which point of the item sits at `at` (default "tl")
//   "f":1        `at` and `size` are fractions of the enclosing frame instead of pixels;   "m":[dx,dy]  a pixel nudge applied after
//   colours: [r,g,b] or [r,g,b,a], 0..255 each
//   "text"    s, size (px, default 14), col, font ("sans" | "mono"), bg (colour; a padded box behind the text), pad (px, 6)
//   "kv"      rows:[[label, value],...], size (13), col (values), key_col (labels), bg, pad (10), gap (26): label / value rows in a box
//   "tabs"    items:[str,...], active:i, s (a status line below), size, col, dim (inactive tabs), bg, pad, gap: a row of text tabs, the
//             active one underlined
//   "rect"    size:[w,h], fill, line, lw (line width), r (corner radius); size[1] = 0 with "ar" (aspect w/h) derives the height
//   "line"    pts:[[x,y],...] (screen), col, lw
//   "image"   path (PNG/JPG file), size:[w,h] (or w + "ar"), uv:[u0,v0,u1,v1] (a crop), id (texture slot: one texture per id, replaced
//             when the path changes; default "img")
//   "group"   size:[w,h], space:[sw,sh] -- children use their own pixel space (e.g. a camera frame's 1280 x 720), mapped onto the
//             rectangle, clipped to it; fill, line, lw, items:[...]. Text sizes and line widths stay in screen pixels.
//   "wtext"   w:[x,y,z] (world, z up), the text fields above; `off`:[dx,dy] screen offset; default anchor "b" (the text sits above the
//             point). Hidden when the point is behind the camera or outside the viewport.
//   "wline"   pts:[[x,y,z],...] (world), col, lw
//   "wring"   w:[x,y,z], r (px), col, lw -- a marker circle
//   "wdot"    w:[x,y,z], r (px), col -- a filled disc
// The camera that projects world points is the one the scene was rendered with (OverlayCtx); `Overlay::labels()` lists where each
// world text landed (tests compare them to analytic projections).
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "viewer/viewer.h"

namespace hud {

struct Label {
  std::string text;
  float x = 0, y = 0;  // window pixels of the projected world point (before `off`)
  bool visible = false;
};

class Overlay {
 public:
  Overlay();
  ~Overlay();
  Overlay(const Overlay&) = delete;
  Overlay& operator=(const Overlay&) = delete;

  // Thread-safe. Throws std::runtime_error naming the problem when the document is not valid JSON / not an object. The new
  // document replaces the previous one at the start of the next frame.
  void set(const std::string& json);
  // Viewer thread: draw the current document (what ViewerOptions::overlay calls).
  void draw(const OverlayCtx& ctx);
  // Camera requested by the current document (ViewerOptions::camera); false when it has none.
  bool camera(CameraCtl& c);

  // Progress, for producers that step a paused sim in lockstep with the window (recording): the seq of the document drawn most
  // recently, how many consecutive frames drew exactly this seq at the document's `tick` (when it has one), and the frame count.
  struct Progress {
    std::int64_t seq = -1;
    std::uint64_t tick = 0;
    int stable = 0;
    std::uint64_t frames = 0;
    std::string error;  // first draw error ("" none): a bad item is skipped, the HUD stays up
  };
  Progress progress() const;
  std::vector<Label> labels() const;  // world texts of the last drawn frame

  // Frees the image textures (viewer thread, ImGui context current): the viewer calls it before it shuts ImGui down.
  void release();
  // Headless: draws the current document once into a throwaway ImGui context (no window, no GPU) with the camera / window of `ctx`.
  // Not while a viewer is running in this process (ImGui's current context is global).
  struct DryRun {
    std::vector<Label> labels;
    int vertices = 0;  // vertices the document added to the draw list
    std::string error;
  };
  DryRun dry_run(const OverlayCtx& ctx);

  std::function<void(const OverlayCtx&)> callback() {
    return [this](const OverlayCtx& c) { draw(c); };
  }
  std::function<bool(CameraCtl&)> camera_driver() {
    return [this](CameraCtl& c) { return camera(c); };
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace hud
