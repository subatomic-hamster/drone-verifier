#include "viewer/theme.h"

#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

extern unsigned char kUiFont[], kUiMono[];  // generated (CMakeLists.txt); non-const because ImGui takes void*, never written
extern const std::size_t kUiFontSize, kUiMonoSize;

namespace theme {
namespace {
ImFont* g_mono = nullptr;
}
ImU32 u32(ImVec4 c, float alpha) { c.w *= alpha; return ImGui::ColorConvertFloat4ToU32(c); }
ImFont* mono() { return g_mono; }

void init() {
  ImGuiIO& io = ImGui::GetIO();
  ImFontConfig cfg;
  cfg.FontDataOwnedByAtlas = false;  // the arrays are static
  io.Fonts->AddFontFromMemoryTTF(kUiFont, static_cast<int>(kUiFontSize), kFontBody, &cfg);
  g_mono = io.Fonts->AddFontFromMemoryTTF(kUiMono, static_cast<int>(kUiMonoSize), 13.f, &cfg);

  ImGuiStyle& s = ImGui::GetStyle();
  s.WindowRounding = 0;
  s.ChildRounding = 2;
  s.FrameRounding = 3;
  s.PopupRounding = 4;
  s.GrabRounding = 2;
  s.ScrollbarRounding = 3;
  s.TabRounding = 0;
  s.WindowBorderSize = 0;
  s.ChildBorderSize = 0;
  s.FrameBorderSize = 0;
  s.PopupBorderSize = 1;
  s.WindowPadding = {10, 8};
  s.FramePadding = {8, 4};
  s.ItemSpacing = {8, 5};
  s.ItemInnerSpacing = {6, 4};
  s.IndentSpacing = 14;
  s.ScrollbarSize = 8;
  s.GrabMinSize = 10;
  s.SeparatorTextBorderSize = 1;
  ImVec4* c = s.Colors;
  c[ImGuiCol_Text] = kText;
  c[ImGuiCol_TextDisabled] = kDim;
  c[ImGuiCol_WindowBg] = kPanel;
  c[ImGuiCol_ChildBg] = {0, 0, 0, 0};
  c[ImGuiCol_PopupBg] = {0.135f, 0.142f, 0.157f, 0.99f};
  c[ImGuiCol_Border] = kLine;
  c[ImGuiCol_FrameBg] = kRaised;
  c[ImGuiCol_FrameBgHovered] = {0.255f, 0.268f, 0.292f, 1.f};
  c[ImGuiCol_FrameBgActive] = {0.285f, 0.300f, 0.325f, 1.f};
  c[ImGuiCol_TitleBg] = kPanel;
  c[ImGuiCol_TitleBgActive] = kPanel;
  c[ImGuiCol_MenuBarBg] = kPanel;
  c[ImGuiCol_ScrollbarBg] = {0, 0, 0, 0};
  c[ImGuiCol_ScrollbarGrab] = {1, 1, 1, 0.18f};
  c[ImGuiCol_ScrollbarGrabHovered] = {1, 1, 1, 0.30f};
  c[ImGuiCol_ScrollbarGrabActive] = {1, 1, 1, 0.40f};
  c[ImGuiCol_CheckMark] = kAccent;
  c[ImGuiCol_SliderGrab] = kAccent;
  c[ImGuiCol_SliderGrabActive] = {0.36f, 0.64f, 1.f, 1.f};
  c[ImGuiCol_Button] = {1, 1, 1, 0.08f};
  c[ImGuiCol_ButtonHovered] = {1, 1, 1, 0.15f};
  c[ImGuiCol_ButtonActive] = {1, 1, 1, 0.23f};
  c[ImGuiCol_Header] = {1, 1, 1, 0.06f};
  c[ImGuiCol_HeaderHovered] = {1, 1, 1, 0.10f};
  c[ImGuiCol_HeaderActive] = {kAccent.x, kAccent.y, kAccent.z, 0.35f};
  c[ImGuiCol_Separator] = kLine;
  c[ImGuiCol_SeparatorHovered] = kAccent;
  c[ImGuiCol_SeparatorActive] = kAccent;
  c[ImGuiCol_ResizeGrip] = {0, 0, 0, 0};
  c[ImGuiCol_Tab] = {0, 0, 0, 0};
  c[ImGuiCol_TabHovered] = {1, 1, 1, 0.10f};
  c[ImGuiCol_TabSelected] = {1, 1, 1, 0.06f};
  c[ImGuiCol_TableHeaderBg] = {0.135f, 0.142f, 0.157f, 1.f};
  c[ImGuiCol_TableBorderStrong] = kLine;
  c[ImGuiCol_TableBorderLight] = {1, 1, 1, 0.05f};
  c[ImGuiCol_TableRowBg] = {0, 0, 0, 0};
  c[ImGuiCol_TableRowBgAlt] = {1, 1, 1, 0.025f};
  c[ImGuiCol_TextSelectedBg] = {kAccent.x, kAccent.y, kAccent.z, 0.40f};
  c[ImGuiCol_DragDropTarget] = kAccent;
  c[ImGuiCol_NavHighlight] = {0, 0, 0, 0};
}

void caption(const char* text) {
  ImGui::PushStyleColor(ImGuiCol_Text, kDim);
  ImGui::PushFont(nullptr, kFontCaption);
  ImGui::TextUnformatted(text);
  ImGui::PopFont();
  ImGui::PopStyleColor();
}

int tabs(const char* const* labels, int n, int active, float gap) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  int clicked = -1;
  ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::PushFont(nullptr, 13.f);
  const float h = ImGui::GetFontSize() + 12;
  for (int i = 0; i < n; ++i) {
    const ImVec2 ts = ImGui::CalcTextSize(labels[i]);
    ImGui::SetCursorScreenPos(p);
    ImGui::PushID(i);
    if (ImGui::InvisibleButton("##tab", {ts.x + 4, h})) clicked = i;
    const bool hov = ImGui::IsItemHovered();
    ImGui::PopID();
    const ImVec4 col = i == active ? kText : hov ? ImVec4{0.85f, 0.85f, 0.87f, 1} : kDim;
    dl->AddText({p.x + 2, p.y + 6}, u32(col), labels[i]);
    if (i == active) dl->AddRectFilled({p.x, p.y + h - 2}, {p.x + ts.x + 4, p.y + h}, u32(kAccent));
    p.x += ts.x + 4 + gap;
  }
  ImGui::PopFont();
  ImGui::SetCursorScreenPos({p.x, p.y});
  return clicked;
}

float pill(ImDrawList* dl, ImVec2 pos, const char* text, ImVec4 col) {
  ImGui::PushFont(nullptr, 12.f);
  const ImVec2 ts = ImGui::CalcTextSize(text);
  dl->AddRectFilled(pos, {pos.x + ts.x + 14, pos.y + ts.y + 6}, u32({0, 0, 0, 0.55f}), 3);
  dl->AddRect(pos, {pos.x + ts.x + 14, pos.y + ts.y + 6}, u32(col, 0.55f), 3);
  dl->AddText({pos.x + 7, pos.y + 3}, u32(col), text);
  ImGui::PopFont();
  return ts.x + 14;
}

bool button(const char* label, bool accent, ImVec2 size) {
  if (accent) {
    ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.36f, 0.64f, 1.f, 1.f});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.17f, 0.45f, 0.86f, 1.f});
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.f, 1.f, 1.f, 1.f});
  }
  const bool r = ImGui::Button(label, size);
  if (accent) ImGui::PopStyleColor(4);
  return r;
}

// ---- icons -----------------------------------------------------------------------------------------------------------------
void icon(ImDrawList* dl, Icon ic, ImVec2 c, float s, ImU32 col) {
  const float k = s / 24.f, th = std::max(1.3f, 1.6f * k);
  const auto P = [&](float x, float y) { return ImVec2{c.x + (x - 12) * k, c.y + (y - 12) * k}; };
  const auto L = [&](float x0, float y0, float x1, float y1) { dl->AddLine(P(x0, y0), P(x1, y1), col, th); };
  const auto poly = [&](std::initializer_list<ImVec2> pts, bool closed) {
    ImVec2 v[16];
    int n = 0;
    for (const ImVec2& p : pts) v[n++] = P(p.x, p.y);
    dl->AddPolyline(v, n, col, closed ? ImDrawFlags_Closed : ImDrawFlags_None, th);
  };
  const auto dot = [&](float x, float y, float r, bool fill) {
    if (fill) dl->AddCircleFilled(P(x, y), r * k, col, 12);
    else dl->AddCircle(P(x, y), r * k, col, 20, th);
  };
  const auto rect = [&](float x0, float y0, float x1, float y1, bool fill) {
    if (fill) dl->AddRectFilled(P(x0, y0), P(x1, y1), col, 1.5f * k);
    else dl->AddRect(P(x0, y0), P(x1, y1), col, 1.5f * k, 0, th);
  };
  const auto arrows4 = [&] {
    L(12, 3, 12, 21), L(3, 12, 21, 12);
    poly({{9, 6}, {12, 3}, {15, 6}}, false), poly({{9, 18}, {12, 21}, {15, 18}}, false);
    poly({{6, 9}, {3, 12}, {6, 15}}, false), poly({{18, 9}, {21, 12}, {18, 15}}, false);
  };
  const auto eye = [&] {
    dl->AddBezierCubic(P(2, 12), P(7, 4.5f), P(17, 4.5f), P(22, 12), col, th);
    dl->AddBezierCubic(P(2, 12), P(7, 19.5f), P(17, 19.5f), P(22, 12), col, th);
    dot(12, 12, 3.2f, false);
  };
  switch (ic) {
    case Icon::Select: poly({{6, 3}, {6, 19}, {10.5f, 15}, {13.5f, 21}, {16, 19.8f}, {13, 14}, {19, 14}}, true); break;
    case Icon::Move: arrows4(); break;
    case Icon::Pan: arrows4(); dot(12, 12, 2.2f, true); break;
    case Icon::Rotate:
      dl->PathArcTo(c, 7.5f * k, 0.9f, 5.4f, 20);
      dl->PathStroke(col, 0, th);
      poly({{19.5f, 6}, {19.5f, 11.5f}, {14, 11.5f}}, false);
      break;
    case Icon::Align:
      L(4, 3, 4, 21);
      rect(7, 6, 19, 10, true), rect(7, 14, 14, 18, true);
      break;
    case Icon::Mate: rect(3, 8, 14, 18, false), rect(10, 6, 21, 16, false); break;
    case Icon::Colour:
      poly({{12, 3}, {6.5f, 11}, {6.5f, 15.5f}, {9, 19.5f}, {15, 19.5f}, {17.5f, 15.5f}, {17.5f, 11}}, true);
      dl->AddCircleFilled(P(12, 15), 2.6f * k, col, 10);
      break;
    case Icon::Measure:
      poly({{3, 16}, {8, 21}, {21, 8}, {16, 3}}, true);
      L(7.5f, 12.5f, 10.5f, 15.5f), L(11.5f, 8.5f, 13.5f, 10.5f), L(15, 5, 18, 8);
      break;
    case Icon::Undo:
    case Icon::Redo: {
      const float m = ic == Icon::Undo ? 1.f : -1.f, o = ic == Icon::Undo ? 0.f : 24.f;
      const auto X = [&](float x) { return o + m * x; };
      dl->AddBezierCubic(P(X(5), 11), P(X(13), 6), P(X(20), 10), P(X(19), 18), col, th);
      poly({{X(9), 6}, {X(5), 11}, {X(10.5f), 14.5f}}, false);
      break;
    }
    case Icon::Wire:
      poly({{4, 17}, {10, 17}, {10, 7}, {20, 7}}, false);
      dot(4, 17, 2.2f, true), dot(20, 7, 2.2f, true);
      break;
    case Icon::Pin:
      dot(5, 7, 1.8f, true), dot(5, 12, 1.8f, true), dot(5, 17, 1.8f, true);
      poly({{5, 12}, {13, 12}, {16, 7}, {21, 7}}, false);
      dot(21, 7, 1.6f, true);
      break;
    case Icon::Auto: poly({{13.5f, 3}, {5.5f, 13.5f}, {11, 13.5f}, {10, 21}, {18.5f, 10}, {13, 10}}, true); break;
    case Icon::Check:
      dot(12, 12, 9, false);
      poly({{7.5f, 12.5f}, {10.8f, 15.8f}, {16.5f, 8.8f}}, false);
      break;
    case Icon::Save:
      poly({{4, 4}, {17, 4}, {20, 7}, {20, 20}, {4, 20}}, true);
      rect(8, 4, 15, 9, false), rect(7, 13, 17, 20, false);
      break;
    case Icon::Eye: eye(); break;
    case Icon::EyeOff: eye(); L(4, 20, 20, 4); break;
    case Icon::Fit:
      poly({{4, 9}, {4, 4}, {9, 4}}, false), poly({{15, 4}, {20, 4}, {20, 9}}, false);
      poly({{4, 15}, {4, 20}, {9, 20}}, false), poly({{15, 20}, {20, 20}, {20, 15}}, false);
      rect(9, 9, 15, 15, false);
      break;
    case Icon::Home:
    case Icon::Cube:
      poly({{12, 3}, {20, 7.5f}, {20, 16.5f}, {12, 21}, {4, 16.5f}, {4, 7.5f}}, true);
      L(12, 12, 12, 21), L(12, 12, 4, 7.5f), L(12, 12, 20, 7.5f);
      break;
    case Icon::Orbit: {
      ImVec2 v[28];
      for (int i = 0; i < 28; ++i) {
        const float a = 6.2831853f * static_cast<float>(i) / 28.f, x = 9.5f * std::cos(a), y = 4.8f * std::sin(a);
        v[i] = P(12 + x * 0.87f - y * 0.5f, 12 + x * 0.5f + y * 0.87f);
      }
      dl->AddPolyline(v, 28, col, ImDrawFlags_Closed, th);
      dot(12, 12, 2.4f, true);
      break;
    }
    case Icon::Zoom:
    case Icon::Search:
      dot(10, 10, 6, false);
      L(14.5f, 14.5f, 20.5f, 20.5f);
      break;
    case Icon::Mirror:
      L(12, 3, 12, 21);
      poly({{9, 7}, {3, 17}, {9, 17}}, true), poly({{15, 7}, {21, 17}, {15, 17}}, true);
      break;
    case Icon::Copy: rect(8, 8, 20, 20, false), poly({{16, 8}, {16, 4}, {4, 4}, {4, 16}, {8, 16}}, false); break;
    case Icon::Trash:
      L(4.5f, 7, 19.5f, 7);
      poly({{9, 7}, {9, 4}, {15, 4}, {15, 7}}, false);
      poly({{6.5f, 7}, {7.5f, 20}, {16.5f, 20}, {17.5f, 7}}, false);
      break;
    case Icon::Play: dl->AddTriangleFilled(P(8, 5), P(8, 19), P(19, 12), col); break;
    case Icon::Pause: rect(7, 5, 11, 19, true), rect(13, 5, 17, 19, true); break;
    case Icon::Step:
      dl->AddTriangleFilled(P(5, 5), P(5, 19), P(15, 12), col);
      rect(17, 5, 20, 19, true);
      break;
    case Icon::Add: L(12, 5, 12, 19), L(5, 12, 19, 12); break;
    case Icon::Clear: L(6, 6, 18, 18), L(18, 6, 6, 18); break;
    case Icon::Swap:
      L(4, 8, 20, 8), poly({{16, 4}, {20, 8}, {16, 12}}, false);
      L(20, 16, 4, 16), poly({{8, 12}, {4, 16}, {8, 20}}, false);
      break;
    case Icon::Doc: poly({{6, 3}, {14, 3}, {19, 8}, {19, 21}, {6, 21}}, true), poly({{14, 3}, {14, 8}, {19, 8}}, false); break;
    case Icon::Net:
      L(5, 6, 12, 12), L(5, 18, 12, 12), L(12, 12, 19, 12);
      dot(5, 6, 2.2f, true), dot(5, 18, 2.2f, true), dot(19, 12, 2.2f, true), dot(12, 12, 1.8f, false);
      break;
    case Icon::Keys:  // a keyboard: outline, a row of keys, the space bar
      rect(2, 6, 22, 18, false);
      dot(6, 10, 1.1f, true), dot(10, 10, 1.1f, true), dot(14, 10, 1.1f, true), dot(18, 10, 1.1f, true);
      L(7, 14.5f, 17, 14.5f);
      break;
  }
}

bool icon_button(const char* id, Icon ic, float size, bool on, bool enabled, const char* tip) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::BeginDisabled(!enabled);
  const bool hit = ImGui::InvisibleButton(id, {size, size});
  ImGui::EndDisabled();
  const bool hov = ImGui::IsItemHovered() && enabled;
  ImDrawList* dl = ImGui::GetWindowDrawList();
  if (on) dl->AddRectFilled(p, {p.x + size, p.y + size}, u32(kAccent, 0.28f), 3);
  else if (hov) dl->AddRectFilled(p, {p.x + size, p.y + size}, u32({1, 1, 1, 0.10f}), 3);
  icon(dl, ic, {p.x + size / 2, p.y + size / 2}, size * 0.62f, u32(on ? kAccent : kText, enabled ? (hov || on ? 1.f : 0.78f) : 0.3f));
  if (tip && hov) ImGui::SetTooltip("%s", tip);
  return hit && enabled;
}

// ---- ribbon ----------------------------------------------------------------------------------------------------------------
void Ribbon::begin(ImVec2 origin, float height) {
  dl = ImGui::GetWindowDrawList();
  org = origin;
  h = height;
  x = origin.x + 10;
  group_x0 = x;
  group_name = nullptr;
}

void Ribbon::group(const char* name) {
  if (group_name) {  // close the previous: its name centred under the buttons, a hairline after it
    ImGui::PushFont(nullptr, 10.5f);
    const float tw = ImGui::CalcTextSize(group_name).x;
    if (tw + 4 > x - group_x0) x = group_x0 + tw + 4;  // a group is at least as wide as its name
    const float w = x - group_x0;
    dl->AddText({group_x0 + (w - tw) * 0.5f, org.y + h - 15}, u32(kDim, 0.8f), group_name);
    ImGui::PopFont();
    if (name) dl->AddLine({x + 3, org.y + 8}, {x + 3, org.y + h - 8}, u32(kLine));
    x += 8;
  }
  group_x0 = x;
  group_name = name;
}

bool Ribbon::button(Icon ic, const char* label, bool on, bool enabled, const char* tip) {
  ImGui::PushFont(nullptr, 11.5f);
  const float tw = ImGui::CalcTextSize(label).x, w = std::max(44.f, tw + 12);
  const ImVec2 p{x, org.y + 5}, sz{w, h - 24};
  ImGui::SetCursorScreenPos(p);
  ImGui::PushID(label);
  ImGui::BeginDisabled(!enabled);
  const bool hit = ImGui::InvisibleButton("##rb", sz);
  ImGui::EndDisabled();
  ImGui::PopID();
  const bool hov = ImGui::IsItemHovered() && enabled;
  if (on) dl->AddRectFilled(p, {p.x + sz.x, p.y + sz.y}, u32(kAccent, 0.26f), 4);
  else if (hov) dl->AddRectFilled(p, {p.x + sz.x, p.y + sz.y}, u32({1, 1, 1, 0.09f}), 4);
  if (on) dl->AddRect(p, {p.x + sz.x, p.y + sz.y}, u32(kAccent, 0.7f), 4);
  const float a = enabled ? 1.f : 0.32f;
  icon(dl, ic, {p.x + w / 2, p.y + 15}, 22, u32(on ? ImVec4{0.55f, 0.76f, 1.f, 1.f} : kText, a));
  dl->AddText({p.x + (w - tw) / 2, p.y + sz.y - 15}, u32(on ? kText : kDim, a < 1 ? a : (hov || on ? 1.f : 0.95f)), label);
  ImGui::PopFont();
  if (tip && hov) ImGui::SetTooltip("%s", tip);
  x += w + 2;
  return hit && enabled;
}

void Ribbon::end() {
  group(nullptr);  // flush the last group's name
  dl->AddLine({org.x, org.y + h - 0.5f}, {org.x + ImGui::GetWindowWidth(), org.y + h - 0.5f}, u32(kLine));
}

}  // namespace theme
