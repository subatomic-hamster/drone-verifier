#pragma once
// The viewer's look, after the CAD / EDA tools it sits next to (Fusion 360, Onshape, KiCad): one restrained dark-grey palette of
// opaque docked panels, a single blue accent, thin hairlines, a fixed type scale. Embedded fonts: Roboto Medium (UI) and Cousine
// (code / logs), both Apache-2.0 from Dear ImGui's misc/fonts (cmake/deps.cmake). Icons are drawn with ImDrawList primitives
// (theme::icon): no icon font, nothing to license.
#include <imgui.h>

namespace theme {
// palette
constexpr ImVec4 kBar{0.105f, 0.110f, 0.121f, 1.f};      // application bar, status bar
constexpr ImVec4 kBg{0.125f, 0.131f, 0.145f, 1.f};       // page / sheet background (library, program, wiring)
constexpr ImVec4 kPanel{0.158f, 0.166f, 0.182f, 1.f};    // docked side panels and the ribbon
constexpr ImVec4 kRaised{0.215f, 0.226f, 0.247f, 1.f};   // fields, cards
constexpr ImVec4 kLine{1.f, 1.f, 1.f, 0.10f};
constexpr ImVec4 kText{0.90f, 0.91f, 0.93f, 1.f};
constexpr ImVec4 kDim{0.60f, 0.62f, 0.66f, 1.f};
constexpr ImVec4 kAccent{0.231f, 0.545f, 0.965f, 1.f};   // the one accent: CAD blue
constexpr ImVec4 kOk{0.40f, 0.80f, 0.52f, 1.f};
constexpr ImVec4 kWarn{0.98f, 0.75f, 0.28f, 1.f};
constexpr ImVec4 kErr{0.95f, 0.40f, 0.40f, 1.f};
// type scale (px)
constexpr float kFontBody = 14.f, kFontSmall = 12.f, kFontCaption = 11.f, kFontTitle = 16.f;
// chrome heights (px): application bar, ribbon, status bar
constexpr float kAppBar = 34.f, kRibbon = 68.f, kStatusBar = 24.f;

ImU32 u32(ImVec4 c, float alpha = 1.f);
// Fonts + style. Call once after ImGui::CreateContext(). The mono font is theme::mono() (nullptr before init).
void init();
ImFont* mono();
// Section label: dim, small, uppercase (the caller passes the text already in capitals).
void caption(const char* text);
// A row of text tabs on the current window's draw list starting at the cursor; the active one gets the accent underline.
// Returns the index clicked this frame or -1.
int tabs(const char* const* labels, int n, int active, float gap = 26.f);
// Small rounded count pill ("3 ERR"); returns its width.
float pill(ImDrawList* dl, ImVec2 pos, const char* text, ImVec4 col);
// A flat button in the HUD style; accent = filled accent.
bool button(const char* label, bool accent = false, ImVec2 size = {0, 0});

// ---- icons ----------------------------------------------------------------------------------------------------------------
enum class Icon {
  Select, Move, Rotate, Align, Mate, Colour, Measure, Undo, Redo, Wire, Pin, Auto, Check, Save, Eye, EyeOff, Fit, Home, Orbit, Pan,
  Zoom, Search, Mirror, Copy, Trash, Play, Pause, Step, Cube, Add, Swap, Clear, Doc, Net, Keys,
};
// Draw `ic` centred at `c`, about `s` px square, stroke `col`.
void icon(ImDrawList* dl, Icon ic, ImVec2 c, float s, ImU32 col);
// A square icon-only button (viewport navigation, tree eyes); `on` = accent tint. Draws on the current window.
bool icon_button(const char* id, Icon ic, float size, bool on = false, bool enabled = true, const char* tip = nullptr);

// The ribbon: groups of icon-over-label buttons with the group name underneath, hairlines between groups. Use inside a window
// that spans the ribbon rectangle.
struct Ribbon {
  ImDrawList* dl = nullptr;
  ImVec2 org{0, 0};  // top-left of the ribbon on screen
  float h = kRibbon, x = 0;
  float group_x0 = 0;
  const char* group_name = nullptr;
  void begin(ImVec2 origin, float height = kRibbon);
  void group(const char* name);  // closes the previous group, starts the next
  bool button(Icon ic, const char* label, bool on = false, bool enabled = true, const char* tip = nullptr);
  void end();
};
}  // namespace theme
