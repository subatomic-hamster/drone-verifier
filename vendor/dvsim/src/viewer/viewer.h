#pragma once
// The viewer: a GLFW window with a Vulkan swapchain (WindowRenderer) and a Dear ImGui overlay (scene tree, inspector with
// SI units, pause/run/step/rate). It runs on the calling thread -- GLFW needs the main thread on macOS -- while the Host's
// own thread runs the simulation. The viewer never touches the World: it draws Host::take_frame() snapshots, and every
// UI action is an ordinary bus command (dvsim.Envelope with source "viewer") sent through Host::submit(), so it is
// capability-checked, logged and replayable like any other client's.
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/host.h"

// What an overlay callback (a HUD, e.g. the counter-UAS display) gets once per frame, after the 3D scene and before the panels:
// draw_list = the ImGui background draw list (cast to ImDrawList*), the viewport rectangle (window pixels; the part of the window
// the panels leave free), and the camera the scene was rendered with (world +z up; fovy degrees).
struct OverlayCtx {
  void* draw_list = nullptr;
  float x = 0, y = 0, w = 0, h = 0;
  double cam_pos[3] = {0, 0, 0}, cam_look[3] = {1, 0, 0}, cam_fovy = 45;
  double window_w = 0, window_h = 0;  // the full window: project with it (renderer aspect), draw inside x/y/w/h
  std::uint64_t tick = 0;
  double sim_time = 0;
};

// The viewer's orbit camera (target, yaw / pitch in rad, distance in m; position = target + dist * (cos p cos y, cos p sin y, sin p)).
struct CameraCtl {
  double target[3] = {0, 0, 0}, yaw = 0, pitch = 0, dist = 1;
};

struct ViewerOptions {
  int frames = 0;                          // > 0: leave after this many frames (smoke tests); 0: until the window closes
  std::string script;                       // UI-equivalent commands issued at given frames, see parse_script
  std::string design_report;                // optional <name>.report.json of a `dvsim design build`: shows the Design window
  // Optional editing backend (python: dvsim.design.viewer.Backend): JSON request in, JSON reply out (see viewer/design.h
  // `request`). The viewer holds no design logic: every Design-window edit is one such request, answered by the same
  // dvsim.design.Design API the MCP tools use; the viewer then loads the rebuilt model (LoadModel on the bus, source
  // "viewer", elevated: the person at the window is the approval) keeping the camera. Called on worker threads, one at a
  // time; blocking is fine.
  std::function<std::string(const std::string&)> design_api;
  // Optional per-frame overlay hook (HUD): called on the viewer thread inside the ImGui frame.
  std::function<void(const OverlayCtx&)> overlay;
  // Optional camera driver (HUD chase / presets): called each frame with the current orbit; returning true replaces it.
  std::function<bool(CameraCtl&)> camera;
  std::function<void()> on_close;  // called on the viewer thread just before ImGui is shut down (free what the overlay registered)
  bool clean = false;  // SIMULATE with no panes, top bar or status bar: the 3D view fills the window (HUD / recording)
  int width = 1280, height = 720;
  const std::atomic<bool>* stop = nullptr;  // optional: another thread sets it to close the viewer
  // No window, no X server, no swapchain: width x height frames (scene + ImGui) render into an off-screen image on the best
  // Vulkan device (discrete > integrated > cpu; DVSIM_VK_DEVICE=<name substring> overrides), capped at 60 fps, and are read
  // back to RGBA (viewer_latest_frame). DVSIM_OFFSCREEN=1 in the environment turns it on without code changes.
  bool offscreen = false;
};

// Off-screen mode: copies the newest frame into *rgba (width*height*4, row 0 = top) and returns its sequence number
// (counts frames since the process started; 0 = none yet). Safe from any thread while run_viewer runs.
std::uint64_t viewer_latest_frame(std::vector<std::uint8_t>* rgba, int* width, int* height);

// Throws std::runtime_error with a plain-language reason: no display, no Vulkan loader or device, window creation
// failed. Returns when the window is closed, `frames` frames were drawn, or *stop is set; commands still in flight
// are answered before it returns.
void run_viewer(Host& host, const ViewerOptions& opts);

// Script syntax: "FRAME ACTION [ARGS...]; ..." with ACTION one of
//   run RATE | pause | step N | setattr PATH ATTR V [V...]   (bus commands, see script_envelope)
//   select PATH                                              (no command: shows that prim in the inspector)
//   inst ID                                                  (no command: a Design-window click on that instance)
//   Design-window editing (needs ViewerOptions::design_api; each is the button of the same name, queued in order):
//     dsuggest TARGET [CATEGORY]   library browser: parts for port 'inst.port' (or swap candidates for 'inst')
//     dsnap PART TO [ID]           "snap here": add PART to port 'inst.port' / instance TO, best placement
//     dcycle N                     preview placement alternative N (0 = the chosen one) of the last snap
//     dswap ID PART                swap, regenerating the device bindings (auto)
//     dremove ID | dautowire PORT [TO] | dautowire_all | dundo | dredo | dsave [PATH] | dsim hover|hold
//     dfinding N                   click DRC finding N (selects the instances involved)
//     dsync                        re-read the design (what a polled change from another client does)
//   UI-only actions (no bus command; mode / selection / camera are window state, search types into the library box):
//     mode build|library|wiring|simulate|program | uisel ID[,ID...] | search TEXT | uicam top|front|side|iso | uiframe | uikeys [on|off] | crumb DEPTH
//     (crumb = a click on that breadcrumb of the top bar: back to that open document; a dirty child asks to save first)
//     (uikeys = the controls popover of the top bar; the window reports what it shows back with the backend op ui.report)
//     (every one of these is also a backend op the window shows through `ui_poll`: python/dvsim/design/uiops.py)
// executed (in order) when that frame is reached, exactly like the matching UI control would.
struct ScriptCmd {
  int frame = 0;
  std::string action;
  std::vector<std::string> args;
};
std::vector<ScriptCmd> parse_script(const std::string& script);  // throws std::runtime_error naming the bad entry
// The dvsim.Envelope bytes of one scripted command (what the viewer submits); throws std::runtime_error on bad arguments.
std::vector<std::uint8_t> script_envelope(const ScriptCmd& cmd, const std::string& source = "viewer");
