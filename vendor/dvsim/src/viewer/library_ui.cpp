#include "viewer/library_ui.h"

#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO_WRITE
#include <stb_image.h>

#include "viewer/theme.h"

namespace libui {
namespace {
using theme::u32;

ImVec4 fidelity_color(const std::string& f) {
  if (f == "datasheet") return {0.45f, 0.85f, 0.55f, 1};
  if (f == "literature") return {0.40f, 0.82f, 0.82f, 1};
  if (f == "measured") return {0.45f, 0.65f, 1.f, 1};
  if (f == "derived") return {1.f, 0.78f, 0.30f, 1};
  return {1.f, 0.50f, 0.35f, 1};  // estimated
}
std::string top(const std::string& cat) { return cat.substr(0, cat.find('.')); }
ImVec4 cat_color(const std::string& cat) {
  unsigned h = 2166136261u;
  for (char c : top(cat)) h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
  float r, g, b;
  ImGui::ColorConvertHSVtoRGB(static_cast<float>(h % 360) / 360.f, 0.38f, 0.30f, r, g, b);
  return {r, g, b, 1};
}
void placeholder(ImDrawList* dl, ImVec2 a, ImVec2 b, const std::string& cat) {
  const ImVec4 c = cat_color(cat);
  dl->AddRectFilledMultiColor(a, b, u32(c, 0.65f), u32(c, 0.65f), u32(c, 0.25f), u32(c, 0.25f));
  const ImVec2 m{(a.x + b.x) / 2, (a.y + b.y) / 2};
  const float r = std::min(b.x - a.x, b.y - a.y) * 0.24f;
  const ImU32 line = u32({1, 1, 1, 0.55f});
  const std::string t = top(cat);
  if (t == "motor" || t == "prop" || t == "wheel" || t == "actuator") {
    dl->AddCircle(m, r, line, 32, 2.f);
    dl->AddCircle(m, r * 0.35f, line, 16, 2.f);
  } else if (t == "fastener") {
    ImVec2 p[6];
    for (int i = 0; i < 6; ++i) p[i] = {m.x + r * std::cos(i * 1.0472f), m.y + r * std::sin(i * 1.0472f)};
    dl->AddPolyline(p, 6, line, ImDrawFlags_Closed, 2.f);
  } else if (t == "wire") {
    for (int i = 0; i < 7; ++i)
      dl->AddLine({m.x - r + i * r / 3.5f, m.y + ((i & 1) ? -r * 0.4f : r * 0.4f)}, {m.x - r + (i + 1) * r / 3.5f, m.y + ((i & 1) ? r * 0.4f : -r * 0.4f)}, line, 2.f);
  } else {
    dl->AddRect({m.x - r, m.y - r * 0.7f}, {m.x + r, m.y + r * 0.7f}, line, t == "battery" ? 6.f : 2.f, 0, 2.f);
    if (t == "pcb" || t == "esc" || t == "sensor")
      for (int i = 0; i < 4; ++i) dl->AddCircleFilled({m.x + (i % 2 ? r : -r) * 0.7f, m.y + (i / 2 ? r : -r) * 0.45f}, 2.2f, line);
  }
}
}  // namespace

std::string Model::filters_json() const {
  std::string o = "{";
  for (const auto& [k, f] : filters)
    if (f.on) {
      char b[96];
      std::snprintf(b, sizeof b, "%s\"%s\":[%.9g,%.9g]", o.size() > 1 ? "," : "", k.c_str(), static_cast<double>(f.lo), static_cast<double>(f.hi));
      o += b;
    }
  return o + "}";
}

Thumbs::~Thumbs() {
  for (auto& [p, t] : tex_) {
    if (!t) continue;
    ImGui::UnregisterUserTexture(t);
    IM_DELETE(t);
  }
}
ImTextureRef Thumbs::get(const std::string& path) {
  auto it = tex_.find(path);
  if (it == tex_.end()) {
    ImTextureData* t = nullptr;
    int w = 0, h = 0, n = 0;
    if (unsigned char* px = stbi_load(path.c_str(), &w, &h, &n, 4)) {
      t = IM_NEW(ImTextureData)();
      t->Create(ImTextureFormat_RGBA32, w, h);
      std::memcpy(t->GetPixels(), px, static_cast<std::size_t>(w) * h * 4);
      stbi_image_free(px);
      ImGui::RegisterUserTexture(t);
    }
    it = tex_.emplace(path, t).first;
  }
  return it->second ? it->second->GetTexRef() : ImTextureRef();
}

void thumb_rect(Thumbs& th, const Hit& h, ImVec2 a, ImVec2 b) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  if (!h.thumb.empty()) {
    const ImTextureRef r = th.get(h.thumb);
    if (r._TexData || r._TexID != 0) {
      dl->AddRectFilled(a, b, u32({0.10f, 0.10f, 0.115f, 1}), 2);
      const float s = std::min(b.x - a.x, b.y - a.y);  // square thumbnail, centred
      const ImVec2 c{(a.x + b.x) / 2, (a.y + b.y) / 2};
      dl->AddImage(r, {c.x - s / 2, c.y - s / 2}, {c.x + s / 2, c.y + s / 2});
      return;
    }
  }
  placeholder(dl, a, b, h.category);
}

void tick_typing(Model& m, double now, Actions&) {
  if (m.type_target.empty() || now < m.type_next) return;
  const std::size_t have = std::strlen(m.query);
  if (have < m.type_target.size() && have + 1 < sizeof m.query) {
    m.query[have] = m.type_target[have];
    m.query[have + 1] = 0;
    m.dirty = true;
    m.type_next = now + m.type_ms / 1000.0;
  }
  if (std::strlen(m.query) >= m.type_target.size()) m.type_target.clear();
}

namespace {
bool card(Model& m, Thumbs& th, const Hit& h, ImVec2 size, bool selected, Actions& a) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::PushID(h.id.c_str());
  ImGui::InvisibleButton("##card", size);
  const bool hov = ImGui::IsItemHovered();
  const bool clicked = ImGui::IsItemClicked();
  if (hov && ImGui::IsMouseDoubleClicked(0)) a.open_page = h.id;
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceNoPreviewTooltip)) {
    ImGui::SetDragDropPayload(h.mine ? "DVSIM_MYPART" : "DVSIM_PART", h.id.c_str(), h.id.size() + 1);
    ImGui::EndDragDropSource();
  }
  ImGui::PopID();
  const ImVec2 q{p.x + size.x, p.y + size.y};
  dl->AddRectFilled(p, q, u32(theme::kRaised), 3);
  const float ts = size.y > 100 ? size.x - 12 : size.y - 12;
  if (size.y > 100) thumb_rect(th, h, {p.x + 6, p.y + 6}, {q.x - 6, p.y + 6 + ts * 0.72f});
  else thumb_rect(th, h, {p.x + 6, p.y + 6}, {p.x + 6 + ts, q.y - 6});
  const float tx = size.y > 100 ? p.x + 8 : p.x + ts + 14, ty = size.y > 100 ? p.y + 6 + ts * 0.72f + 6 : p.y + 8;
  dl->PushClipRect({tx, ty}, {q.x - 6, q.y - 2}, true);
  ImGui::PushFont(nullptr, 13.f);
  dl->AddText({tx, ty}, u32(theme::kText), h.name.c_str());
  ImGui::PopFont();
  ImGui::PushFont(nullptr, 11.5f);
  dl->AddText({tx, ty + 17}, u32(theme::kDim), (h.vendor.empty() ? h.category : h.vendor).c_str());
  if (size.y <= 100) dl->AddText({tx, ty + 32}, u32(theme::kDim, 0.7f), h.category.c_str());
  ImGui::PopFont();
  dl->PopClipRect();
  if (selected) dl->AddRect(p, q, u32(theme::kAccent), 3, 0, 1.5f);
  else if (hov) dl->AddRect(p, q, u32({1, 1, 1, 0.25f}), 3);
  m.card_at[h.id] = {(p.x + q.x) / 2, (p.y + q.y) / 2};
  if (hov && !ImGui::IsMouseDragging(0)) ImGui::SetTooltip("%s\n%s   %s%s%s", h.name.c_str(), h.id.c_str(), h.category.c_str(),
                                                          h.has_mass ? "   " : "", h.has_mass ? (std::to_string(h.mass * 1000).substr(0, 7) + " g").c_str() : "");
  return clicked;
}

void search_box(Model& m, float width, const char* hint) {
  ImGui::SetNextItemWidth(width);
  if (ImGui::InputTextWithHint("##q", hint, m.query, sizeof m.query)) m.dirty = true;
}

void filters(Model& m) {
  static const char* const kOrder[] = {"mass", "voltage", "kv", "capacity", "current", "diameter", "length", "width"};
  theme::caption("FILTERS");
  bool any = false;
  for (const char* k : kOrder) {
    const auto it = m.ranges.find(k);
    if (it == m.ranges.end() || it->second.hi <= it->second.lo) continue;
    any = true;
    Filter& f = m.filters[k];
    ImGui::PushID(k);
    if (ImGui::Checkbox(k, &f.on)) {
      if (f.on && f.hi <= f.lo) { f.lo = static_cast<float>(it->second.lo); f.hi = static_cast<float>(it->second.hi); }
      m.dirty = true;
    }
    if (!it->second.unit.empty()) { ImGui::SameLine(); ImGui::TextDisabled("[%s]", it->second.unit.c_str()); }
    if (f.on) {
      ImGui::SetNextItemWidth(-1);
      const float lo = static_cast<float>(it->second.lo), hi = static_cast<float>(it->second.hi);
      if (ImGui::DragFloatRange2("##r", &f.lo, &f.hi, (hi - lo) / 300.f, lo, hi, "%.4g", "%.4g")) m.dirty = true;
    }
    ImGui::PopID();
  }
  if (!any) ImGui::TextDisabled("(no numeric params in this category)");
}

void categories(Model& m) {
  theme::caption("CATEGORY");
  const auto row = [&](const std::string& label, const std::string& path, int count, int depth) {
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10.f * depth);
    const bool sel = m.category == path;
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s  %d##%s", label.c_str(), count, path.c_str());
    if (ImGui::Selectable(buf, sel)) {
      m.category = path;
      m.dirty = true;
    }
  };
  int total = 0;
  for (const Cat& c : m.cats)
    if (c.depth == 0) total += c.count;
  row("All parts", "", total, 0);
  row("MY PARTS", kMyParts, m.my_count, 0);  // the workspace's own sub-assemblies (files with an [exports] table)
  for (const Cat& c : m.cats) {
    // a sub-category shows once its top-level category is chosen, down to one level below the chosen one
    const int cur = m.category.empty() ? -1 : static_cast<int>(std::count(m.category.begin(), m.category.end(), '.'));
    const bool open = c.depth == 0 || (cur >= 0 && c.path.rfind(top(m.category) + ".", 0) == 0 && c.depth <= cur + 1);
    if (!open) continue;
    row(c.path.substr(c.path.rfind('.') == std::string::npos ? 0 : c.path.rfind('.') + 1), c.path, c.count, c.depth);
  }
}

void detail_mine(const MyPart& p, Actions& a) {
  ImGui::PushFont(nullptr, 17.f);
  ImGui::TextWrapped("%s", p.name.c_str());
  ImGui::PopFont();
  ImGui::TextDisabled("my part   %s", p.path.c_str());
  ImGui::Spacing();
  if (theme::button("ADD", true)) a.add_mine = p.path;
  ImGui::TextDisabled("or drag the card into the 3D view");
  ImGui::Spacing();
  theme::caption("EXPORTS");
  ImGui::TextWrapped("%s", p.exports.c_str());
  ImGui::Spacing();
  theme::caption("MASS / COST");
  if (p.has_mass) ImGui::Text("%.1f g", p.mass * 1000);
  else ImGui::TextDisabled("mass: not built");
  if (p.has_cost) ImGui::Text("$%.2f", p.cost);
  else ImGui::TextDisabled("cost: no price data");
}

void detail(Model& m, Thumbs& th, const std::string& selection, int targets, Actions& a) {
  if (const auto mp = m.mine.find(m.selected); mp != m.mine.end()) return detail_mine(mp->second, a);
  const auto it = m.infos.find(m.selected);
  if (m.selected.empty() || it == m.infos.end()) {
    ImGui::TextDisabled("%s", m.selected.empty() ? "select a part" : "loading ...");
    return;
  }
  const Info& i = it->second;
  Hit h;
  h.id = i.id; h.category = i.category; h.thumb = i.thumb;
  const float w = ImGui::GetContentRegionAvail().x;
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::Dummy({w, w * 0.62f});
  thumb_rect(th, h, p, {p.x + w, p.y + w * 0.62f});
  ImGui::PushFont(nullptr, 17.f);
  ImGui::TextWrapped("%s", i.name.c_str());
  ImGui::PopFont();
  ImGui::TextDisabled("%s   %s", i.id.c_str(), i.category.c_str());
  if (!i.vendor.empty()) ImGui::TextDisabled("%s%s%s", i.vendor.c_str(), i.part_number.empty() ? "" : "   P/N ", i.part_number.c_str());
  ImGui::Spacing();
  if (theme::button(selection.empty() ? "ADD" : ("ADD TO " + selection).c_str(), true)) a.add = i.id;
  ImGui::SameLine();
  ImGui::BeginDisabled(selection.empty());
  if (theme::button("SWAP")) a.swap = i.id;
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("replace the selected instance with this part (device bindings refreshed)");
  ImGui::TextDisabled("or drag the card into the 3D view");
  ImGui::Spacing();
  theme::caption("WHERE IT SNAPS");
  if (targets < 0) ImGui::TextDisabled("...");
  else if (targets == 0) ImGui::TextDisabled("no open compatible port in this design: it is placed freely");
  else ImGui::Text("%d open compatible port%s in this design", targets, targets == 1 ? "" : "s");
  ImGui::Spacing();
  theme::caption("PORTS");
  for (const PortRow& p2 : i.ports) {
    ImGui::Text("%s", p2.name.c_str());
    ImGui::SameLine(110);
    ImGui::TextDisabled("%s %s %s", p2.kind.c_str(), p2.type.c_str(), p2.gender.c_str());
  }
  ImGui::Spacing();
  theme::caption("PARAMS");
  for (const ParamRow& pr : i.params) {
    ImGui::PushID(pr.name.c_str());
    ImGui::Text("%s", pr.name.c_str());
    ImGui::SameLine(150);
    ImGui::Text("%s %s", pr.value.c_str(), pr.unit.c_str());
    ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 78);
    ImGui::TextColored(fidelity_color(pr.fidelity), "%s", pr.fidelity.c_str());
    if (ImGui::IsItemHovered() && (!pr.quote.empty() || !pr.url.empty())) {
      ImGui::BeginTooltip();
      ImGui::PushTextWrapPos(380);
      ImGui::TextColored(fidelity_color(pr.fidelity), "%s", pr.fidelity.c_str());
      if (!pr.src.empty()) ImGui::TextDisabled("source %s  %s", pr.src.c_str(), pr.url.c_str());
      if (!pr.quote.empty()) ImGui::TextWrapped("\"%s\"", pr.quote.c_str());
      ImGui::PopTextWrapPos();
      ImGui::EndTooltip();
    }
    ImGui::PopID();
  }
}
}  // namespace

void page(Model& m, Thumbs& th, ImVec2 pos, ImVec2 size, const std::string& selection, int targets, Actions& a) {
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::kBg);
  ImGui::Begin("##library_page", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
  ImGui::PopStyleColor();
  const float lw = 214, rw = std::min(300.f, size.x * 0.3f);
  ImGui::BeginChild("filters", {lw, 0});
  categories(m);
  ImGui::Spacing();
  filters(m);
  ImGui::EndChild();
  ImGui::SameLine();
  ImGui::BeginChild("grid", {size.x - lw - rw - 36, 0});
  ImGui::PushFont(nullptr, 15.f);
  search_box(m, -1, "SEARCH  id, name, vendor, category, params (e.g. 2216 kv920)");
  ImGui::PopFont();
  ImGui::TextDisabled("%d part%s", m.total, m.total == 1 ? "" : "s");
  ImGui::BeginChild("cards", {0, 0});
  if (m.hits.empty() && !m.dirty) {  // empty state: say why and how to get parts back
    const ImVec2 a0 = ImGui::GetContentRegionAvail();
    ImGui::SetCursorPos({a0.x / 2 - 150, a0.y / 3});
    ImGui::BeginGroup();
    ImGui::PushFont(nullptr, 17.f);
    ImGui::TextUnformatted("No parts match");
    ImGui::PopFont();
    ImGui::TextDisabled("Try fewer words, another category or wider filters.");
    ImGui::Spacing();
    if (theme::button("CLEAR SEARCH AND FILTERS")) {
      m.query[0] = 0;
      m.category.clear();
      for (auto& kv : m.filters) kv.second.on = false;
      m.dirty = true;
    }
    ImGui::EndGroup();
  }
  const float cw = 132, ch = 130, gap = 8;
  const int cols = std::max(1, static_cast<int>((ImGui::GetContentRegionAvail().x + gap) / (cw + gap)));
  int k = 0;
  for (const Hit& h : m.hits) {
    if (k % cols) ImGui::SameLine(0, gap);
    if (card(m, th, h, {cw, ch}, h.id == m.selected, a)) {
      m.selected = h.id;
      a.info = h.id;
    }
    ++k;
  }
  ImGui::EndChild();
  ImGui::EndChild();
  ImGui::SameLine();
  ImGui::BeginChild("detail", {0, 0});
  detail(m, th, selection, targets, a);
  ImGui::EndChild();
  ImGui::End();
}

void sidebar(Model& m, Thumbs& th, ImVec2 pos, ImVec2 size, Actions& a) {
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::Begin("##library_side", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
  theme::caption("COMPONENTS");
  search_box(m, -1, "search parts");
  // category chips: the top-level categories
  ImGui::PushFont(nullptr, 12.f);
  float x = 0;
  const float avail = ImGui::GetContentRegionAvail().x;
  const auto chip = [&](const std::string& label, const std::string& path) {
    const float w = ImGui::CalcTextSize(label.c_str()).x + 12;
    if (x > 0 && x + w > avail) x = 0;
    else if (x > 0) ImGui::SameLine(0, 4);
    const bool on = m.category == path;
    ImGui::PushStyleColor(ImGuiCol_Button, on ? ImVec4{theme::kAccent.x, theme::kAccent.y, theme::kAccent.z, 0.30f} : ImVec4{1, 1, 1, 0.06f});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {5, 1});
    if (ImGui::SmallButton((label + "##chip" + path).c_str())) {
      m.category = on ? "" : path;
      m.dirty = true;
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    x += w + 4;
  };
  int shown = 0;
  for (const Cat& c : m.cats)
    if (c.depth == 0 && c.count >= 5 && shown++ < 14) chip(c.path, c.path);
  ImGui::PopFont();
  ImGui::Spacing();
  ImGui::TextDisabled("%d parts   drag into the 3D view", m.total);
  ImGui::BeginChild("list", {0, 0});
  if (m.hits.empty() && !m.dirty) ImGui::TextDisabled("No parts match. Clear the search or pick another category.");
  for (const Hit& h : m.hits)
    if (card(m, th, h, {ImGui::GetContentRegionAvail().x, 58}, h.id == m.selected, a)) {
      m.selected = h.id;
      a.info = h.id;
    }
  ImGui::EndChild();
  ImGui::End();
}

}  // namespace libui
