#include "viewer/wire2d.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "viewer/theme.h"

namespace wire2d {
namespace {
constexpr float kHeadH = 30, kPad = 10, kRowH = 22;  // mirror python/dvsim/design/wire2d.py
constexpr float kPinR = 6, kSideW = 220, kStripW = 22;  // the side panel of the flattened view, open / folded up to a strip
constexpr double kFade = 0.45, kCardPop = 0.35, kWireAnim = 0.8, kGlide = 0.6;
constexpr int kAwg[] = {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};

ImU32 kind_col(const std::string& k, int a = 255) {
  if (k == "power") return IM_COL32(236, 92, 62, a);
  if (k == "signal") return IM_COL32(82, 152, 242, a);
  if (k == "ground") return IM_COL32(168, 142, 110, a);
  if (k == "nc") return IM_COL32(96, 100, 110, a);
  return IM_COL32(150, 150, 160, a);
}
ImU32 flag_col(const std::string& f, int a = 255) { return f == "error" ? IM_COL32(255, 64, 64, a) : IM_COL32(255, 190, 40, a); }
ImU32 group_col(const std::string& g) {
  static const ImU32 pal[] = {IM_COL32(58, 86, 120, 255), IM_COL32(96, 70, 120, 255), IM_COL32(60, 110, 92, 255), IM_COL32(124, 90, 56, 255),
                              IM_COL32(110, 62, 70, 255), IM_COL32(72, 100, 128, 255), IM_COL32(88, 100, 60, 255), IM_COL32(84, 84, 100, 255)};
  std::size_t h = 1469598103u;
  for (char c : g) h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
  return pal[h % 8];
}
double ease_out(double t) {
  t = std::clamp(t, 0.0, 1.0);
  return 1 - std::pow(1 - t, 3);
}
double ease_back(double t) {  // overshoots a little: a card "pops"
  t = std::clamp(t, 0.0, 1.0);
  const double c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * std::pow(t - 1, 3) + c1 * std::pow(t - 1, 2);
}
std::string str_args(std::initializer_list<std::pair<const char*, std::string>> kv) {
  std::string o = "{";
  for (const auto& [k, v] : kv) o += (o.size() > 1 ? "," : "") + design::quote(k) + ":" + v;
  return o + "}";
}
struct Bez {
  ImVec2 p[4];
  ImVec2 at(float t) const {
    const float u = 1 - t;
    const float a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return {a * p[0].x + b * p[1].x + c * p[2].x + d * p[3].x, a * p[0].y + b * p[1].y + c * p[2].y + d * p[3].y};
  }
};
Bez make_bez(ImVec2 a, bool a_right, ImVec2 b, bool b_right, float zoom) {
  const float k = std::max(50.f * zoom, std::abs(b.x - a.x) * 0.5f);
  return {{a, {a.x + (a_right ? k : -k), a.y}, {b.x + (b_right ? k : -k), b.y}, b}};
}
float dist_to_poly(const std::vector<ImVec2>& pts, ImVec2 m) {
  float best = 1e9f;
  for (std::size_t i = 1; i < pts.size(); ++i) {
    const ImVec2 prev = pts[i - 1], q = pts[i];
    const float vx = q.x - prev.x, vy = q.y - prev.y, wx = m.x - prev.x, wy = m.y - prev.y;
    const float len2 = vx * vx + vy * vy, t = len2 > 0 ? std::clamp((vx * wx + vy * wy) / len2, 0.f, 1.f) : 0.f;
    const float dx = prev.x + t * vx - m.x, dy = prev.y + t * vy - m.y;
    best = std::min(best, std::sqrt(dx * dx + dy * dy));
  }
  return best;
}
// Corners of an orthogonal polyline rounded with radius <= r (a quadratic curve through each corner), flattened.
std::vector<ImVec2> round_path(const std::vector<ImVec2>& p, float r) {
  std::vector<ImVec2> o;
  if (p.size() < 3) return p;
  o.push_back(p[0]);
  for (std::size_t i = 1; i + 1 < p.size(); ++i) {
    const ImVec2 a = p[i - 1], b = p[i], c = p[i + 1];
    const float la = std::hypot(b.x - a.x, b.y - a.y), lc = std::hypot(c.x - b.x, c.y - b.y);
    const float k = std::min({r, la / 2, lc / 2});
    if (k < 1) {
      o.push_back(b);
      continue;
    }
    const ImVec2 s{b.x + (a.x - b.x) / la * k, b.y + (a.y - b.y) / la * k}, e{b.x + (c.x - b.x) / lc * k, b.y + (c.y - b.y) / lc * k};
    for (int j = 0; j <= 6; ++j) {
      const float t = j / 6.f, u = 1 - t;
      o.push_back({u * u * s.x + 2 * u * t * b.x + t * t * e.x, u * u * s.y + 2 * u * t * b.y + t * t * e.y});
    }
  }
  o.push_back(p.back());
  return o;
}
float poly_len(const std::vector<ImVec2>& p) {
  float l = 0;
  for (std::size_t i = 1; i < p.size(); ++i) l += std::hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
  return l;
}
// The first `f` (0..1) of the polyline by length.
std::vector<ImVec2> prefix(const std::vector<ImVec2>& p, float f) {
  if (f >= 1 || p.size() < 2) return p;
  const float want = poly_len(p) * f;
  std::vector<ImVec2> o{p[0]};
  float acc = 0;
  for (std::size_t i = 1; i < p.size(); ++i) {
    const float l = std::hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
    if (acc + l >= want) {
      const float t = l > 0 ? (want - acc) / l : 0;
      o.push_back({p[i - 1].x + (p[i].x - p[i - 1].x) * t, p[i - 1].y + (p[i].y - p[i - 1].y) * t});
      return o;
    }
    acc += l;
    o.push_back(p[i]);
  }
  return o;
}
// `t` cut to fit `maxw` px at font size `size`, with ".." for what is gone (the full text is in the hover tooltip / the part file).
std::string elide(ImFont* f, float size, const std::string& t, float maxw) {
  if (f->CalcTextSizeA(size, 1e9f, 0, t.c_str()).x <= maxw) return t;
  std::string o = t;
  while (!o.empty() && f->CalcTextSizeA(size, 1e9f, 0, (o + "..").c_str()).x > maxw) o.pop_back();
  return o.empty() ? o : o + "..";
}
}  // namespace

float Canvas::side_w() const { return side_open_ ? kSideW : kStripW; }

// A wire disappears with either of its components (a hidden part's wires are not drawn, labels and junction dots included).
bool Canvas::wire_hidden(const Wire& w) const {
  if (!hid_ || hid_->empty()) return false;
  if (hidden(w.from.substr(0, w.from.find('.'))) || hidden(w.to.substr(0, w.to.find('.')))) return true;
  for (const Strand& s : w.strands)
    if (hidden(s.from.substr(0, s.from.find('.'))) || hidden(s.to.substr(0, s.to.find('.')))) return true;
  return false;
}

const Card* Canvas::card(const std::string& id) const {
  for (const Card& c : cards_)
    if (c.id == id) return &c;
  return nullptr;
}

bool Canvas::pin_pos(const Card& c, const Pin& p, float& x, float& y) const {
  x = ox_ + (c.x + (p.side == "right" ? c.w : 0)) * zoom_;
  y = oy_ + (c.y + kHeadH + kPad * 0.5f + (static_cast<float>(p.row) + 0.5f) * kRowH) * zoom_;
  return true;
}

// The colour class of one conductor: its from-pin's kind (ground / power / signal / nc), else the wire's.
std::string Canvas::strand_kind(const Wire& w, const Strand& s) const {
  const std::size_t dot = s.from.find('.');
  if (const Card* c = card(s.from.substr(0, dot)))
    for (const Pin& p : c->pins)
      if (p.name == s.from.substr(dot + 1) && !p.conn.empty()) return p.kind == "nc" ? w.kind : p.kind;
  return w.kind;
}

std::vector<ImVec2> Canvas::strand_path(const Strand& s) const {
  const bool stale = drag_.kind == Drag::on_card && (s.from.rfind(drag_.card + ".", 0) == 0 || s.to.rfind(drag_.card + ".", 0) == 0);
  std::vector<ImVec2> pts;
  if (!stale && s.route.size() >= 2) {
    for (const ImVec2& p : s.route) pts.push_back({ox_ + p.x * zoom_, oy_ + p.y * zoom_});
    return pts;
  }
  ImVec2 a[2];
  bool right[2] = {false, false}, ok[2] = {false, false};
  const std::string* refs[2] = {&s.from, &s.to};
  for (int k = 0; k < 2; ++k) {
    const std::size_t dot = refs[k]->find('.');  // "card.TERMINAL": a pin terminal is itself "CONN.PIN"
    const Card* c = card(refs[k]->substr(0, dot));
    if (!c) continue;
    for (const Pin& p : c->pins)
      if (p.name == refs[k]->substr(dot + 1)) pin_pos(*c, p, a[k].x, a[k].y), right[k] = p.side == "right", ok[k] = true;
  }
  if (!ok[0] || !ok[1]) return pts;
  const float st = 16 * zoom_;
  const ImVec2 s0{a[0].x + (right[0] ? st : -st), a[0].y}, s1{a[1].x + (right[1] ? st : -st), a[1].y};
  const float mx = (s0.x + s1.x) / 2;
  pts = {a[0], s0, {mx, s0.y}, {mx, s1.y}, s1, a[1]};
  return pts;
}

std::vector<ImVec2> Canvas::wire_path(const Wire& w, ImVec2* label_c, bool* has_label) const {
  *has_label = !w.label.empty();
  if (w.strands.empty()) return {};
  std::vector<ImVec2> pts = strand_path(w.strands[0]);
  const Strand& s = w.strands[0];
  const bool stale = drag_.kind == Drag::on_card && (s.from.rfind(drag_.card + ".", 0) == 0 || s.to.rfind(drag_.card + ".", 0) == 0);
  if (!stale && s.route.size() >= 2) *label_c = {ox_ + w.label_at.x * zoom_, oy_ + w.label_at.y * zoom_};
  else if (pts.size() >= 4) *label_c = {pts[2].x, (pts[2].y + pts[3].y) / 2};
  return pts;
}

void Canvas::toast(const std::string& s, bool bad) {
  toast_ = s;
  toast_bad_ = bad;
  toast_t_ = ImGui::GetTime();
}

void Canvas::poll(int flat_rev) {
  if (flat_rev != seen_rev_) {
    seen_rev_ = flat_rev;
    send_("flatten.state", "{}", true);
  }
}

Card Canvas::parse_card(const design::Json& c) {
  Card n;
  n.id = c.s("id"), n.part = c.s("part"), n.group = c.s("group");
  n.x = static_cast<float>(c.n("x")), n.y = static_cast<float>(c.n("y")), n.w = static_cast<float>(c.n("w", 200)), n.h = static_cast<float>(c.n("h", 60));
  for (const design::Json& p : c.at("pins").arr) {
    Pin pin{p.s("name"), p.s("kind"), p.s("type"), p.s("gender"), p.s("role"), p.s("side"), static_cast<int>(p.n("row")),
            p.at("wired").num != 0, p.s("label"), p.s("conn"), p.s("tip"), p.s("flag"), {}, p.s("port"), {}};
    if (pin.label.empty()) pin.label = pin.name;
    for (const design::Json& f : p.at("flags").arr) pin.flag_msgs.push_back(f.s("code") + ": " + f.s("message"));
    for (const design::Json& l : p.at("connects").arr) pin.connects.push_back({l.s("component"), l.s("port"), l.s("pin")});
    n.pins.push_back(std::move(pin));
  }
  for (const design::Json& k : c.at("connectors").arr)
    n.conns.push_back({k.s("id"), k.s("name"), k.s("type"), k.s("side"), static_cast<int>(k.n("row")), static_cast<int>(k.n("nrows", 1))});
  return n;
}

void Canvas::apply(const design::Json& f) {
  const double now = ImGui::GetTime();
  const bool was = active_;
  active_ = f.at("active").num != 0;
  seen_rev_ = static_cast<int>(f.n("rev"));
  std::vector<Card> next;
  int idx = 0;
  for (const design::Json& c : f.at("cards").arr) {
    Card n = parse_card(c);
    const Card* old = card(n.id);
    n.born = old ? old->born : was ? now + 0.18 + 0.06 * idx++ : -1e9;  // the first state arrives laid out: it fades in, nothing pops
    if (old && drag_.kind == Drag::on_card && drag_.card == n.id) n.x = old->x, n.y = old->y;  // being dragged: the local position wins
    n.tx = n.x, n.ty = n.y, n.x0 = n.x, n.y0 = n.y;
    if (old && active_ && was && !(drag_.kind == Drag::on_card && drag_.card == n.id) && std::abs(old->tx - n.tx) + std::abs(old->ty - n.ty) > 1.f) {
      n.x0 = old->x, n.y0 = old->y;  // the layout moved this card: glide from where it is drawn now
      layout_t_ = now;
    }
    next.push_back(std::move(n));
  }
  std::set<std::string> known;
  for (const Wire& w : wires_) known.insert(w.name);
  std::vector<Wire> wnext;
  for (const design::Json& w : f.at("wires").arr) {
    Wire n;
    n.name = w.s("name"), n.from = w.s("from"), n.to = w.s("to"), n.part = w.s("part"), n.kind = w.s("kind"), n.flag = w.s("flag");
    n.awg = w.n("awg"), n.length = w.n("length"), n.conductors = static_cast<int>(w.n("conductors", 1));
    for (const design::Json& f : w.at("flags").arr) n.flag_msgs.push_back(f.s("code") + ": " + f.s("message"));
    for (const design::Json& sj : w.at("strands").arr) {
      Strand s;
      s.from = sj.s("from"), s.to = sj.s("to"), s.flag = sj.s("flag");
      for (const design::Json& p : sj.at("route").arr)
        if (p.arr.size() >= 2) s.route.push_back({static_cast<float>(p.arr[0].num), static_cast<float>(p.arr[1].num)});
      n.strands.push_back(std::move(s));
    }
    if (const design::Json& l = w.at("label"); l.kind == design::Json::Obj)
      n.label = l.s("text"), n.label_at = {static_cast<float>(l.n("x")), static_cast<float>(l.n("y"))}, n.label_size = {static_cast<float>(l.n("w")), static_cast<float>(l.n("h"))};
    if (known.count(n.name)) {
      for (const Wire& o : wires_)
        if (o.name == n.name) n.born = o.born;
    } else if (was || !next.empty()) {  // a new wire: its drag animation, queued behind the ones already playing
      new_wire_slot_ = std::max(new_wire_slot_, now);
      n.born = new_wire_slot_;
      new_wire_slot_ += kWireAnim + 0.2;
    }
    wnext.push_back(std::move(n));
  }
  if (!was && active_) fit_pending_ = true;
  if (!was) {  // the first state: wires that already exist are not "drawn"
    for (Wire& w : wnext) w.born = -1e9;
    new_wire_slot_ = 0;
  }
  auto moved = [&](const std::vector<Card>& o) {  // the backend re-laid-out the cards (a hand move already updated the local ones)
    if (o.size() != next.size()) return false;
    for (size_t i = 0; i < o.size(); ++i)
      if (std::abs(o[i].x - next[i].x) + std::abs(o[i].y - next[i].y) + std::abs(o[i].w - next[i].w) > 1.f) return true;
    return false;
  };
  if (was && active_ && drag_.kind == Drag::none && moved(cards_)) fit_pending_ = true;
  cards_ = std::move(next);
  wires_ = std::move(wnext);
  hidden_wires_ = static_cast<int>(f.n("hidden_wires"));
  if (!sel_wire_.empty() && std::none_of(wires_.begin(), wires_.end(), [&](const Wire& w) { return w.name == sel_wire_; })) sel_wire_.clear();
}

void Canvas::on_reply(const std::string& tool, const design::Json& r) {
  if (tool == "flatten.preview") {  // the picker's preview: cards only, never a toast
    preview_.clear();
    if (r.at("ok").num != 0)
      for (const design::Json& c : r.at("result").at("cards").arr) preview_.push_back(parse_card(c));
    preview_sig_ = preview_asked_;
    return;
  }
  if (r.at("ok").num == 0) {
    toast(tool.substr(tool.find('.') + 1) + ": " + r.s("error"), true);
    return;
  }
  if (r.at("flatten").kind == design::Json::Obj) apply(r.at("flatten"));
  const design::Json& res = r.at("result");
  if (tool == "flatten.wire" && !res.at("wires").arr.empty()) {
    const design::Json& drc = res.at("drc");
    const int n_new = static_cast<int>(drc.at("new").arr.size());
    toast("wired " + res.at("wires").arr[0].str + (n_new ? "  (" + std::to_string(n_new) + " new DRC finding(s))" : ""), false);
  } else if (tool == "flatten.unwire") {
    toast("wire removed", false);
  } else if (tool == "flatten.set_wire") {
    toast("wire updated", false);
  }
}

void Canvas::fit() {
  if (cards_.empty()) return;
  float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
  for (const Card& c : cards_) {
    if (hidden(c.id)) continue;
    x0 = std::min(x0, c.tx), y0 = std::min(y0, c.ty), x1 = std::max(x1, c.tx + c.w), y1 = std::max(y1, c.ty + c.h);
  }
  if (x1 < x0) return;
  const float lg = legend_open_ ? 190.f : 0.f;  // the key sits top-left: the sheet starts right of it
  const float aw = sz_.x - side_w() - lg - 20, ah = sz_.y - 40;
  zoom_ = std::clamp(std::min(aw / (x1 - x0 + 80), ah / (y1 - y0 + 60)), 0.15f, 1.4f);  // (below ~0.5 the labels go connector-level, see canvas_ui)
  ox_ = org_.x + lg + 10 + (aw - (x1 - x0) * zoom_) * 0.5f - x0 * zoom_;
  oy_ = org_.y + 20 + (ah - (y1 - y0) * zoom_) * 0.5f - y0 * zoom_;
}

// The picked components / terminals as the args of flatten.begin and flatten.preview.
std::string Canvas::pick_args() const {
  std::string ids, ports;
  for (const auto& [id, ps] : pick_) {
    ids += (ids.empty() ? "" : ",") + design::quote(id);
    std::string l;
    for (const std::string& p : ps) l += (l.empty() ? "" : ",") + design::quote(p);
    ports += (ports.empty() ? "" : ",") + design::quote(id) + ":[" + l + "]";
  }
  return "{\"instances\":[" + ids + "],\"ports\":{" + ports + "}}";
}
void Canvas::begin_flatten() { send_("flatten.begin", pick_args(), false); }

// The ribbon's WIRE: with nothing picked yet, the BUILD selection (all its power / signal terminals) is flattened at once.
void Canvas::begin_selected(const design::State& st, const std::string& selected) {
  if (active_) return;
  if (pick_.empty() && !selected.empty()) {
    const auto it = st.ports.find(selected);
    if (it != st.ports.end())
      for (const design::Port& p : it->second)
        if (p.kind == "power" || p.kind == "signal") pick_[selected].insert(p.name);
    pick_[selected];
  }
  if (pick_.empty()) toast("pick components below (or select one in BUILD), then WIRE", true);
  else begin_flatten();
}

void Canvas::pick(const std::vector<std::string>& ids, const design::State& st) {
  pick_.clear();
  for (const std::string& id : ids) {
    pick_[id];
    if (const auto it = st.ports.find(id); it != st.ports.end())
      for (const design::Port& p : it->second)
        if (p.kind == "power" || p.kind == "signal") pick_[id].insert(p.name);
  }
}

// PIN WIRE: zoom the sheet to 100 % about its centre so every pin label of a connector can be read and hit.
void Canvas::zoom_to_pins() { view(1.0, false); }

// ui.wire_view: fit the sheet, or zoom to `zoom` (1 = 100 %) about the middle of the canvas.
void Canvas::view(double zoom, bool fit_it, double cx, double cy, double seconds) {
  const float z0 = zoom_, ox0 = ox_, oy0 = oy_;
  const float sx = org_.x + (sz_.x - side_w()) / 2, sy = org_.y + sz_.y / 2;  // the middle of the sheet (screen)
  if (fit_it) {
    fit();
  } else {
    const float z = zoom > 0 ? std::clamp(static_cast<float>(zoom), 0.15f, 3.5f) : z0;
    const float mx = cx >= kKeep ? (sx - ox0) / z0 : static_cast<float>(cx);  // canvas point that ends in the middle
    const float my = cy >= kKeep ? (sy - oy0) / z0 : static_cast<float>(cy);
    zoom_ = z, ox_ = sx - mx * z, oy_ = sy - my * z;
  }
  if (seconds > 0) {
    va_ = {ImGui::GetTime(), seconds, z0, ox0, oy0, zoom_, ox_, oy_};
    zoom_ = z0, ox_ = ox0, oy_ = oy0;
  }
}

// ---- 3D mode: choose components and terminals ----------------------------------------------------------------------------
// One card in screen space (see CardView): body, header, connector boxes and the pins with their labels.
void Canvas::draw_card(ImDrawList* dl, const Card& c, const CardView& v) const {
  ImFont* font = ImGui::GetFont();
  const float fs = ImGui::GetFontSize(), zoom = v.zoom, s = v.s;
  const float cx = v.ox + (c.x + c.w / 2) * zoom, cy = v.oy + (c.y + c.h / 2) * zoom;  // popping cards scale about their centre
  const ImVec2 p0{cx - c.w * zoom * s / 2, cy - c.h * zoom * s / 2}, p1{cx + c.w * zoom * s / 2, cy + c.h * zoom * s / 2};
  const int al = static_cast<int>(255 * std::min(1.f, s) * v.fade);
  dl->AddRectFilled({p0.x + 3, p0.y + 4}, {p1.x + 3, p1.y + 4}, IM_COL32(0, 0, 0, al / 3), 7 * zoom);
  dl->AddRectFilled(p0, p1, IM_COL32(28, 33, 41, al), 7 * zoom);
  const ImU32 gc = group_col(c.group);
  dl->AddRectFilled(p0, {p1.x, p0.y + kHeadH * zoom * s}, (gc & 0x00FFFFFF) | (static_cast<ImU32>(al) << 24), 7 * zoom, ImDrawFlags_RoundCornersTop);
  dl->AddRect(p0, p1, IM_COL32(120, 130, 148, al), 7 * zoom, 0, 1.2f);
  // level of detail: the label sizes follow the zoom but never drop below a legible floor. Pin labels: all of them (lod 2), only the
  // wired / flagged ones (lod 1) or none (lod 0, the connector titles carry the card); the header moves above the card when it is
  // too short to hold its text. The preview (bare) has no labels at all, only the component's name.
  const float pin_fs = fs * 0.92f * zoom * s;
  const int lod = v.bare ? -1 : pin_fs >= 8.5f ? 2 : kRowH * zoom * s >= 10.f ? 1 : 0;
  if (s > 0.6f) {
    const float hh = kHeadH * zoom * s, hs = std::max(fs * zoom * s, v.bare ? 12.5f : 11.5f), ps = std::max(fs * 0.78f * zoom * s, 9.5f);
    const float room = p1.x - p0.x - 16 * zoom;  // the header text never leaves the card
    if (!v.bare && hh >= hs + ps + 3) {
      dl->AddText(font, hs, {p0.x + 8 * zoom, p0.y + 2 * zoom}, IM_COL32(244, 246, 250, al), elide(font, hs, c.id, room).c_str());
      dl->AddText(font, ps, {p0.x + 8 * zoom, p0.y + 3 * zoom + hs}, IM_COL32(190, 198, 212, al), elide(font, ps, c.part, room).c_str());
    } else if (hh >= hs + 3) {
      dl->AddText(font, hs, {p0.x + 8 * zoom, p0.y + (hh - hs) / 2}, IM_COL32(244, 246, 250, al), elide(font, hs, c.id, room).c_str());
    } else {  // above the card, on the sheet
      const std::string t = v.bare ? c.id : c.id + "   " + c.part;
      dl->AddText(font, hs, {p0.x + 2, p0.y - hs - 3}, IM_COL32(236, 239, 244, al), t.c_str());
    }
  }
  // connector boxes: a rounded panel behind the pins of one connector, titled with its id (a two-row header spans the card)
  const auto sy = [&](float row) { return cy + (v.oy + (c.y + kHeadH + kPad * 0.5f + row * kRowH) * zoom - cy) * s; };
  const auto sx = [&](float xx) { return cx + (v.ox + (c.x + xx) * zoom - cx) * s; };
  for (const ConnBox& k : c.conns) {
    const float xl = k.side == "right" ? c.w / 2 + 3 : 5, xr = k.side == "left" ? c.w / 2 - 3 : c.w - 5;
    const ImVec2 b0{sx(xl), sy(static_cast<float>(k.row)) + 1}, b1{sx(xr), sy(static_cast<float>(k.row + k.nrows)) - 1};
    dl->AddRectFilled(b0, b1, IM_COL32(36, 42, 53, al), 5 * zoom * s);
    dl->AddRect(b0, b1, IM_COL32(86, 96, 116, al), 5 * zoom * s, 0, 1.f);
    if (s > 0.6f && !v.bare) {
      const float tfs = std::max(fs * 0.74f * zoom * s, lod == 0 ? 10.5f : 9.f);  // connector titles carry the card when the pins are unlabelled
      const std::string title = elide(font, tfs, k.id + "  " + k.name, b1.x - b0.x - 14 * zoom);  // (a long name is cut to its box)
      const ImVec2 tsz = font->CalcTextSizeA(tfs, 1e9f, 0, title.c_str());
      const float tx = k.side == "right" ? b1.x - tsz.x - 8 * zoom : k.side == "left" ? b0.x + 8 * zoom : (b0.x + b1.x - tsz.x) / 2;
      const float rowh = kRowH * zoom * s;
      if (lod > 0 && rowh >= tsz.y - 2) {  // the title sits in the connector's first row (a little tight is fine: pins start in the next one)
        dl->PushClipRect(b0, {b1.x, b0.y + std::max(rowh, tsz.y)}, true);
        dl->AddText(font, tfs, {tx, b0.y + (rowh - tsz.y) / 2}, IM_COL32(150, 162, 184, al), title.c_str());
        dl->PopClipRect();
      } else if (rowh >= tsz.y + 2) {  // the title sits in the connector's first row
        dl->PushClipRect(b0, {b1.x, b0.y + rowh}, true);
        dl->AddText(font, tfs, {tx, b0.y + (rowh - tsz.y) / 2}, IM_COL32(150, 162, 184, al), title.c_str());
        dl->PopClipRect();
      } else if (b1.y - b0.y >= tsz.y * 2 + 4) {  // a tall box with tiny rows (zoomed out): the title in the middle, over a plate
        const float cxm = (b0.x + b1.x - tsz.x) / 2, cym = (b0.y + b1.y - tsz.y) / 2;
        dl->AddRectFilled({cxm - 4, cym - 2}, {cxm + tsz.x + 4, cym + tsz.y + 2}, IM_COL32(36, 42, 53, al), 3);
        dl->AddText(font, tfs, {cxm, cym}, IM_COL32(190, 200, 220, al), title.c_str());
      } else {  // a short box: the title above it
        dl->AddText(font, tfs, {b0.x, b0.y - tsz.y - 2}, IM_COL32(150, 162, 184, al), title.c_str());
      }
    }
  }
  for (const Pin& p : c.pins) {
    float x = v.ox + (c.x + (p.side == "right" ? c.w : 0)) * zoom;
    float y = v.oy + (c.y + kHeadH + kPad * 0.5f + (static_cast<float>(p.row) + 0.5f) * kRowH) * zoom;
    x = cx + (x - cx) * s, y = cy + (y - cy) * s;
    const bool right = p.side == "right";
    const bool target = drag_.kind == Drag::on_pin && v.hover == &p && !(drag_.card == c.id && drag_.pin == p.name);
    const bool compat = drag_.kind == Drag::on_pin && !(drag_.card == c.id && drag_.pin == p.name);
    const float r = std::max(kPinR * zoom * s, v.bare ? 2.f : 0.f) * (target ? 1.5f : 1.f);
    if (compat) dl->AddCircle({x, y}, r + 4 * zoom, IM_COL32(255, 255, 255, target ? 255 : 90), 0, 1.5f);
    if (!p.flag.empty() && !v.bare) dl->AddCircle({x, y}, r + 3.5f * zoom, flag_col(p.flag, al), 0, 2.5f);
    const bool wired = v.bare || (v.ends ? v.ends->count(c.id + "." + p.name) > 0 : p.wired);  // (a hidden component's wires do not count)
    if (wired) {  // (the preview shows every pin filled in its colour)
      dl->AddCircleFilled({x, y}, r, kind_col(p.kind, al));
      if (v.ends)
        if (const auto e = v.ends->find(c.id + "." + p.name); e != v.ends->end() && e->second >= 2) dl->AddCircleFilled({x, y}, r * 0.5f, IM_COL32(244, 247, 252, al));  // junction
    } else {
      dl->AddCircleFilled({x, y}, r, IM_COL32(28, 33, 41, al));
      dl->AddCircle({x, y}, r, kind_col(p.kind, al), 0, 2);
    }
    if (s > 0.6f && !v.bare && (lod == 2 || (lod == 1 && (wired || !p.flag.empty())))) {
      const bool dim = p.kind == "nc" || (!p.conn.empty() && !wired);  // unwired connector pins recede: the wired ones read first
      const ImU32 tcol = dim ? IM_COL32(150, 158, 174, al) : IM_COL32(236, 239, 244, al);
      const float lfs = std::max(fs * 0.92f * zoom, 9.f);
      const float mid = (p0.x + p1.x) / 2, room = right ? x - r - 6 * zoom - (mid + 4) : mid - 4 - (x + r + 6 * zoom);  // a label stays on its half of the card
      const std::string label = elide(font, lfs, p.label, room);
      const ImVec2 ts = font->CalcTextSizeA(lfs, 1e9f, 0, label.c_str());
      const float tx = right ? x - r - 6 * zoom - ts.x : x + r + 6 * zoom;
      if (!label.empty()) dl->AddText(font, lfs, {tx, y - ts.y / 2}, tcol, label.c_str());
      if (p.conn.empty() && lod == 2 && label == p.label) {  // a port terminal also names its connector type
        const ImVec2 ty = font->CalcTextSizeA(fs * 0.72f * zoom, 1e9f, 0, p.type.c_str());
        const float tx2 = right ? tx - ty.x - 8 * zoom : tx + ts.x + 8 * zoom;
        if (std::abs(tx2 - (right ? p0.x : p1.x)) > 12 && tx2 > p0.x && tx2 + ty.x < p1.x)
          dl->AddText(font, fs * 0.72f * zoom, {tx2, y - ty.y / 2}, IM_COL32(140, 150, 168, al), p.type.c_str());
      }
    }
  }
}

// The picked components as cards, packed in rows into `size` at the largest zoom (<= 1) at which they all fit. Read-only.
void Canvas::preview_ui(ImVec2 at, ImVec2 size) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(at, {at.x + size.x, at.y + size.y}, IM_COL32(12, 14, 19, 255), 6);
  dl->AddRect(at, {at.x + size.x, at.y + size.y}, IM_COL32(120, 130, 148, 70), 6);
  if (pick_.empty() || preview_.empty()) return;
  constexpr float kEdge = 18, kGapX = 22, kGapY = 34;  // screen px: the name of a small card sits in the gap above it
  const ImVec2 room{size.x - 2 * kEdge, size.y - 2 * kEdge};
  std::vector<ImVec2> pos(preview_.size());
  ImVec2 used{0, 0};
  const auto pack = [&](float z) {
    float x = 0, y = kGapY - 14, row_h = 0;
    used = {0, 0};
    for (std::size_t i = 0; i < preview_.size(); ++i) {
      const float w = preview_[i].w * z, h = preview_[i].h * z;
      if (w > room.x) return false;
      if (x > 0 && x + w > room.x) y += row_h + kGapY, x = 0, row_h = 0;
      pos[i] = {x, y};
      x += w + kGapX, row_h = std::max(row_h, h);
      used = {std::max(used.x, x - kGapX), y + row_h};
    }
    return used.y <= room.y;
  };
  float z = 1.f;
  while (z > 0.06f && !pack(z)) z *= 0.94f;
  pack(z);
  const ImVec2 org{at.x + kEdge + (room.x - used.x) / 2, at.y + kEdge + std::max(0.f, (room.y - used.y) / 2)};
  dl->PushClipRect(at, {at.x + size.x, at.y + size.y}, true);
  for (std::size_t i = 0; i < preview_.size(); ++i) {
    const Card& c = preview_[i];  // drawn with its own (x, y) cancelled: origin = where the pack put it
    draw_card(dl, c, {org.x + pos[i].x - c.x * z, org.y + pos[i].y - c.y * z, z, 1.f, true, nullptr, nullptr});
  }
  dl->PopClipRect();
}

// The picker: "Wire Components", the filter and the component / terminal list under it; the picked components as cards on the right.
void Canvas::select_ui(const design::State& st, const std::string& selected) {
  const float col_w = std::min(400.f, sz_.x * 0.42f), top = 22.f;
  ImGui::SetCursorPos({28, top});
  ImGui::BeginGroup();
  ImGui::PushFont(nullptr, 20.f);
  ImGui::TextUnformatted("Wire Components");
  ImGui::PopFont();
  ImGui::Spacing();
  ImGui::SetNextItemWidth(col_w);
  ImGui::InputTextWithHint("##filter", "filter components", filter_, sizeof filter_);
  ImGui::Spacing();
  const float foot = 40, auto_h = autowire_open_ && autowire_ui ? 170.f : 0.f;  // the action row under the list; AUTOWIRE (folded) above it
  ImGui::BeginChild("pick", {col_w, sz_.y - ImGui::GetCursorPosY() - foot - auto_h - 12});
  const std::string f = filter_;
  for (const design::Instance& i : st.report.instances) {
    auto pit = st.ports.find(i.id);
    if (pit == st.ports.end()) continue;
    std::vector<const design::Port*> wp;
    for (const design::Port& p : pit->second)
      if (p.kind == "power" || p.kind == "signal") wp.push_back(&p);
    if (wp.empty() || (!f.empty() && i.id.find(f) == std::string::npos && i.part.find(f) == std::string::npos)) continue;
    ImGui::PushID(i.id.c_str());
    bool on = pick_.count(i.id) != 0;
    if (ImGui::Checkbox("##on", &on)) {
      if (on)
        for (const design::Port* p : wp) pick_[i.id].insert(p->name);
      else pick_.erase(i.id);
    }
    ImGui::SameLine();
    const bool open = ImGui::TreeNodeEx(i.id.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth, "%s  (%zu)", i.id.c_str(), wp.size());
    if (open) {
      for (const design::Port* p : wp) {
        bool pon = pick_.count(i.id) && pick_[i.id].count(p->name);
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kind_col(p->kind)));
        const std::string label = p->name + "  " + p->type;
        if (ImGui::Checkbox(label.c_str(), &pon)) {
          if (pon) pick_[i.id].insert(p->name);
          else if (pick_.count(i.id)) {
            pick_[i.id].erase(p->name);
            if (pick_[i.id].empty()) pick_.erase(i.id);
          }
        }
        ImGui::PopStyleColor();
      }
      ImGui::TreePop();
    }
    ImGui::PopID();
  }
  ImGui::EndChild();
  if (auto_h > 0) {
    ImGui::BeginChild("autowire", {col_w, auto_h}, ImGuiChildFlags_Borders);
    autowire_ui();
    ImGui::EndChild();
  }
  if (theme::button("ADD SELECTED") && !selected.empty()) {
    auto it = st.ports.find(selected);
    if (it != st.ports.end())
      for (const design::Port& p : it->second)
        if (p.kind == "power" || p.kind == "signal") pick_[selected].insert(p.name);
    pick_[selected];
  }
  ImGui::SetItemTooltip("the instance selected in BUILD (all its power / signal terminals)");
  ImGui::SameLine();
  if (theme::button("CLEAR")) pick_.clear();
  ImGui::SameLine();
  ImGui::BeginDisabled(pick_.empty());
  if (theme::button("FLATTEN", true)) begin_flatten();
  ImGui::EndDisabled();
  if (autowire_ui) {
    ImGui::SameLine();
    const std::string al = std::string("AUTOWIRE") + (autowire_n ? " (" + std::to_string(autowire_n) + ")" : "") + (autowire_open_ ? " v" : " >");
    if (theme::button(al.c_str(), autowire_open_)) autowire_open_ = !autowire_open_;
    ImGui::SetItemTooltip("wire open ports automatically; the ambiguous ones are listed for you to pick");
  }
  ImGui::EndGroup();
  // the picked components, as the canvas will draw them
  std::string sig;
  for (const auto& [id, ps] : pick_) {
    sig += id + ":";
    for (const std::string& p : ps) sig += p + ",";
    sig += ";";
  }
  if (sig != preview_asked_) {
    preview_asked_ = sig;
    if (pick_.empty()) preview_.clear(), preview_sig_ = sig;
    else send_("flatten.preview", pick_args(), true);
  }
  const float px = 28 + col_w + 28;
  if (sz_.x - px > 120) preview_ui({org_.x + px, org_.y + top}, {sz_.x - px - 24, sz_.y - top - 24});
}

// ---- flattened: the canvas ------------------------------------------------------------------------------------------------
void Canvas::canvas_ui() {
  ImGuiIO& io = ImGui::GetIO();
  const double now = ImGui::GetTime();
  const float a = static_cast<float>(alpha_);
  // The canvas fills the pane left of the side panel: org_ / sz_ (screen). Everything below draws in screen coordinates.
  const ImVec2 cs{sz_.x - side_w(), sz_.y};
  ImDrawList* dl = ImGui::GetWindowDrawList();
  ImGui::SetCursorScreenPos(org_);
  ImGui::InvisibleButton("##bg", cs, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
  const ImVec2 mp = ImGui::GetIO().MousePos;
  const bool over_key = mp.x >= legend_rect_.x && mp.x <= legend_rect_.z && mp.y >= legend_rect_.y && mp.y <= legend_rect_.w;  // (last frame's key)
  const bool bg_hover = ImGui::IsItemHovered() && !over_key;
  dl->PushClipRect(org_, {org_.x + cs.x, org_.y + cs.y}, true);
  // backdrop; the grid of the canvas fades in with the transition
  // the sheet: very dark, a thin grid (minor 20 units while it is still more than a few px apart, major every 100), like an EDA schematic
  dl->AddRectFilled(org_, {org_.x + cs.x, org_.y + cs.y}, IM_COL32(12, 14, 19, 255));
  const auto grid = [&](float step, int alpha) {
    const float st = step * zoom_;
    for (float g = org_.x + std::fmod(ox_ - org_.x, st); g < org_.x + cs.x; g += st) dl->AddLine({g, org_.y}, {g, org_.y + cs.y}, IM_COL32(150, 165, 200, static_cast<int>(alpha * a)));
    for (float g = org_.y + std::fmod(oy_ - org_.y, st); g < org_.y + cs.y; g += st) dl->AddLine({org_.x, g}, {org_.x + cs.x, g}, IM_COL32(150, 165, 200, static_cast<int>(alpha * a)));
  };
  if (20 * zoom_ >= 8) grid(20, 9);
  grid(100, 22);
  const float intro = static_cast<float>(ease_out(alpha_));
  ImFont* font = ImGui::GetFont();
  const float fs = ImGui::GetFontSize();

  // glide the cards to their new layout; the wires (routed for the final layout) fade in once they have arrived
  const double glide = std::clamp((now - layout_t_) / kGlide, 0.0, 1.0);
  const float gk = static_cast<float>(ease_out(glide));
  if (drag_.kind != Drag::on_card)
    for (Card& c : cards_) c.x = c.x0 + (c.tx - c.x0) * gk, c.y = c.y0 + (c.ty - c.y0) * gk;
  const float wire_fade = static_cast<float>(std::clamp((glide - 0.6) / 0.4, 0.0, 1.0));

  // pan / zoom
  if (bg_hover && io.MouseWheel != 0 && drag_.kind == Drag::none) {
    const float z0 = zoom_;
    zoom_ = std::clamp(zoom_ * std::pow(1.1f, io.MouseWheel), 0.15f, 3.5f);
    ox_ = io.MousePos.x - (io.MousePos.x - ox_) * zoom_ / z0;
    oy_ = io.MousePos.y - (io.MousePos.y - oy_) * zoom_ / z0;
  }
  if (bg_hover && ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0)) ox_ += io.MouseDelta.x, oy_ += io.MouseDelta.y;

  // hit tests: pin under the mouse, wire under the mouse
  const Card* hc = nullptr;
  const Pin* hp = nullptr;
  float hit_d2 = 12 * 12 * zoom_ * zoom_ + 16;  // the nearest pin within reach (a 2 x 20 header packs pins 22 units apart)
  for (const Card& c : cards_)
    for (const Pin& p : c.pins) {
      if (hidden(c.id) || over_key) break;
      float x, y;
      pin_pos(c, p, x, y);
      const float d2 = (io.MousePos.x - x) * (io.MousePos.x - x) + (io.MousePos.y - y) * (io.MousePos.y - y);
      if (d2 < hit_d2) hit_d2 = d2, hc = &c, hp = &p;
    }
  auto pos_of = [&](const std::string& ref, ImVec2& out, bool& right) {
    const std::size_t dot = ref.find('.');
    const Card* c = card(ref.substr(0, dot));
    if (!c) return false;
    for (const Pin& p : c->pins)
      if (p.name == ref.substr(dot + 1)) {
        pin_pos(*c, p, out.x, out.y);
        right = p.side == "right";
        return true;
      }
    return false;
  };
  if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) move_t_ = now;
  const bool tips = tip_mode_ == 1 || (tip_mode_ == 0 && now - move_t_ < 10.0);  // a still pointer (recordings) hovers nothing
  hover_wire_.clear();
  if (tips && !hp && !over_key && drag_.kind == Drag::none) {
    float best = 7;
    for (const Wire& w : wires_)
      for (const Strand& s : w.strands) {
        if (wire_hidden(w)) break;
        const float d = dist_to_poly(strand_path(s), io.MousePos);
        if (d < best) best = d, hover_wire_ = w.name;
      }
  }

  // wires (under the cards): orthogonal routes from the backend, rounded corners, an arrow into the target pin
  struct Label {
    const Wire* w;
    ImVec2 at;
    bool sel;
    std::vector<ImVec2> path;
  };
  std::vector<Label> labels;
  for (const Wire& w : wires_) {
    if (wire_hidden(w)) continue;  // a wire goes with either of its components
    ImVec2 lc{0, 0};
    bool has_label = false;
    const std::vector<ImVec2> lpath = wire_path(w, &lc, &has_label);
    const double t = (now - w.born) / kWireAnim;
    const bool drawing = t >= 0 && t < 1;
    if (t < 0 || wire_fade <= 0.f) continue;  // queued behind another wire's animation / the cards are still gliding
    const float a = static_cast<float>(alpha_) * wire_fade;
    const float f = drawing ? static_cast<float>(ease_out(t)) : 1.f;
    const bool sel = w.name == sel_wire_, hov = w.name == hover_wire_;
    const float th = (sel ? 4.5f : hov ? 4.f : (w.strands.size() > 3 ? 1.8f : 2.6f)) * std::max(0.75f, zoom_);
    for (std::size_t si = 0; si < w.strands.size(); ++si) {  // one line per conductor, pin to pin
      const std::vector<ImVec2> raw = strand_path(w.strands[si]);
      if (raw.size() < 2) continue;
      const std::vector<ImVec2> path = round_path(raw, 9 * zoom_);
      const std::vector<ImVec2> pts = prefix(path, f);
      const std::string& fl = w.strands[si].flag.empty() ? w.flag : w.strands[si].flag;
      const std::string sk = strand_kind(w, w.strands[si]);  // a ground conductor is drawn in the ground colour
      if (sel) dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), IM_COL32(255, 255, 255, static_cast<int>(200 * a)), 0, th + 3);
      if (!fl.empty()) dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), flag_col(fl, static_cast<int>(150 * a)), 0, th + 4);
      dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), kind_col(sk, static_cast<int>(240 * a)), 0, th);
      if (drawing) {  // the cursor that "drags" the wire
        const ImVec2 tip = pts.back();
        dl->AddCircleFilled(tip, 7 * zoom_, IM_COL32(255, 255, 255, 230));
        dl->AddCircle(tip, 11 * zoom_, kind_col(sk, 220), 0, 2);
      } else {
        const ImVec2 e = path.back(), q = path[path.size() - 2];
        const float l = std::max(1e-3f, std::hypot(e.x - q.x, e.y - q.y)), ux = (e.x - q.x) / l, uy = (e.y - q.y) / l, s = 5.5f * std::max(0.75f, zoom_);
        const ImVec2 base{e.x - ux * (6 * zoom_ + 1), e.y - uy * (6 * zoom_ + 1)};  // stops short of the pin dot
        dl->AddTriangleFilled({base.x + ux * s, base.y + uy * s}, {base.x - uy * s * 0.7f, base.y + ux * s * 0.7f}, {base.x + uy * s * 0.7f, base.y - ux * s * 0.7f},
                              kind_col(sk, static_cast<int>(240 * a)));
      }
    }
    if (!drawing && has_label) labels.push_back({&w, lc, sel, round_path(lpath, 9 * zoom_)});
  }
  // gauge labels on top of every wire. The backend placed them clear of each other for its own text size; at the size drawn here two can
  // still touch (or sit on a card), so each is moved along its wire to the first spot that is free, else left off (the wire's tooltip has it).
  std::vector<ImVec4> taken;
  for (const Card& c : cards_)
    if (!hidden(c.id)) taken.push_back({ox_ + c.x * zoom_ - 2, oy_ + c.y * zoom_ - 2, ox_ + (c.x + c.w) * zoom_ + 2, oy_ + (c.y + c.h) * zoom_ + 2});
  for (const Label& l : labels) {
    const float lfs = fs * 0.8f * std::max(0.8f, zoom_);
    const ImVec2 ts = font->CalcTextSizeA(lfs, 1e9f, 0, l.w->label.c_str());
    const auto box = [&](ImVec2 c) { return ImVec4{c.x - ts.x / 2 - 5, c.y - ts.y / 2 - 2, c.x + ts.x / 2 + 5, c.y + ts.y / 2 + 2}; };
    const auto free_at = [&](ImVec2 c) {
      const ImVec4 b = box(c);
      for (const ImVec4& r : taken)
        if (b.x < r.z && b.z > r.x && b.y < r.w && b.w > r.y) return false;
      return true;
    };
    ImVec2 c = l.at;
    bool ok = free_at(c);
    for (float f : {0.5f, 0.35f, 0.65f, 0.2f, 0.8f, 0.12f, 0.88f}) {
      if (ok || l.path.size() < 2) break;
      c = prefix(l.path, f).back();
      ok = free_at(c);
    }
    if (!ok) continue;
    const ImVec4 b = box(c);
    taken.push_back(b);
    dl->AddRectFilled({b.x, b.y}, {b.z, b.w}, IM_COL32(14, 17, 22, 235), 4);
    dl->AddRect({b.x, b.y}, {b.z, b.w}, kind_col(l.w->kind, 150), 4, 0, 1.f);
    dl->AddText(font, lfs, {c.x - ts.x / 2, c.y - ts.y / 2}, IM_COL32(230, 232, 238, 255), l.w->label.c_str());
  }

  // cards
  std::map<std::string, int> ends;  // conductors ending on each pin: 2+ get a junction dot, as in a schematic
  for (const Wire& w : wires_)
    if (!wire_hidden(w))
      for (const Strand& st : w.strands) ++ends[st.from], ++ends[st.to];
  const Card* head_hit = nullptr;
  for (const Card& c : cards_) {
    if (hidden(c.id)) continue;
    const float s = static_cast<float>(ease_back((now - c.born) / kCardPop));
    if (now < c.born || s <= 0.01f) continue;
    draw_card(dl, c, {ox_, oy_, zoom_, s, false, &ends, hp, intro});
    const float cx = ox_ + (c.x + c.w / 2) * zoom_, cy = oy_ + (c.y + c.h / 2) * zoom_;  // (popping cards scale about their centre)
    const ImVec2 p0{cx - c.w * zoom_ * s / 2, cy - c.h * zoom_ * s / 2}, p1{cx + c.w * zoom_ * s / 2, cy + c.h * zoom_ * s / 2};
    if (io.MousePos.x >= p0.x && io.MousePos.x <= p1.x && io.MousePos.y >= p0.y && io.MousePos.y <= p0.y + kHeadH * zoom_ * s) head_hit = &c;
  }

  // interaction
  if (drag_.kind == Drag::none && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    if (hp && bg_hover) drag_ = {Drag::on_pin, hc->id, hp->name, 0, 0};
    else if (head_hit && bg_hover) {
      drag_ = {Drag::on_card, head_hit->id, "", (io.MousePos.x - ox_) / zoom_ - head_hit->x, (io.MousePos.y - oy_) / zoom_ - head_hit->y};
    } else if (bg_hover) {
      sel_wire_ = hover_wire_;
    }
  }
  if (drag_.kind == Drag::on_card) {
    for (Card& c : cards_)
      if (c.id == drag_.card) c.x = c.tx = c.x0 = (io.MousePos.x - ox_) / zoom_ - drag_.dx, c.y = c.ty = c.y0 = (io.MousePos.y - oy_) / zoom_ - drag_.dy;
  }
  if (drag_.kind == Drag::on_pin) {  // the rubber band
    ImVec2 p0;
    bool r0 = false;
    if (pos_of(drag_.card + "." + drag_.pin, p0, r0)) {
      const Bez b = make_bez(p0, r0, io.MousePos, !r0, zoom_);
      dl->AddBezierCubic(b.p[0], b.p[1], b.p[2], b.p[3], IM_COL32(255, 255, 255, 230), 2.5f * zoom_);
    }
  }
  if (drag_.kind != Drag::none && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    if (drag_.kind == Drag::on_pin && hp && !(hc->id == drag_.card && hp->name == drag_.pin)) {
      send_("flatten.wire", str_args({{"from", design::quote(drag_.card + "." + drag_.pin)}, {"to", design::quote(hc->id + "." + hp->name)}}), false);
    } else if (drag_.kind == Drag::on_card) {
      for (const Card& c : cards_)
        if (c.id == drag_.card)
          send_("flatten.move_card", str_args({{"id", design::quote(c.id)}, {"x", std::to_string(c.x)}, {"y", std::to_string(c.y)}}), true);
    }
    drag_ = {};
  }
  if (!sel_wire_.empty() && ImGui::IsKeyPressed(ImGuiKey_Delete) && !io.WantTextInput)
    send_("flatten.unwire", str_args({{"name", design::quote(sel_wire_)}}), false);
  if (tips && hp && drag_.kind == Drag::none) {
    ImGui::BeginTooltip();
    ImGui::Text("%s", hc->id.c_str());  // component, port, pin (the connector pin's own name, else just the port)
    ImGui::SameLine();
    ImGui::TextDisabled("%s%s%s", hp->port.empty() ? hp->name.c_str() : hp->port.c_str(), hp->conn.empty() ? "" : "  ", hp->conn.empty() ? "" : hp->name.c_str());
    if (!hp->tip.empty()) {  // a connector pin: its functions, alternates, ratings (the part file's pinout)
      ImGui::TextDisabled("%s  %s", hp->type.c_str(), hp->gender.c_str());
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kind_col(hp->kind)));
      ImGui::TextUnformatted(hp->tip.c_str());
      ImGui::PopStyleColor();
    } else {
      ImGui::TextDisabled("%s  %s  %s%s%s", hp->kind.c_str(), hp->type.c_str(), hp->gender.c_str(), hp->role.empty() ? "" : "  ", hp->role.c_str());
    }
    for (const std::string& m : hp->flag_msgs) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(flag_col(hp->flag)));
      ImGui::TextWrapped("%s", m.c_str());
      ImGui::PopStyleColor();
    }
    if (!hp->connects.empty()) {  // what its wires lead to
      ImGui::Separator();
      ImGui::TextDisabled("connects to");
      for (const Link& l : hp->connects) ImGui::Text("%s  %s%s%s", l.component.c_str(), l.port.c_str(), l.pin.empty() ? "" : "  ", l.pin.c_str());
    }
    ImGui::EndTooltip();
  }

  if (!hover_wire_.empty() && !hp && drag_.kind == Drag::none) {
    for (const Wire& w : wires_)
      if (w.name == hover_wire_) {
        ImGui::BeginTooltip();
        ImGui::Text("%s", w.name.c_str());
        ImGui::TextDisabled("%s -> %s", w.from.c_str(), w.to.c_str());
        ImGui::TextDisabled("%s%s", w.part.c_str(), w.awg > 0 ? (",  " + std::to_string(static_cast<int>(w.awg)) + " AWG").c_str() : "");
        if (w.strands.size() > 1) ImGui::TextDisabled("%zu conductors", w.strands.size());
        for (const std::string& m : w.flag_msgs) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(flag_col(w.flag)));
          ImGui::TextWrapped("%s", m.c_str());
          ImGui::PopStyleColor();
        }
        ImGui::EndTooltip();
      }
  }
  legend_rect_ = legend(dl, org_, a);

  const float x0 = org_.x, y0 = org_.y;  // the toast (a refusal / confirmation of the last action)
  const double age = now - toast_t_;
  if (age < 7 && !toast_.empty()) {
    const ImVec2 ts = font->CalcTextSizeA(fs, 1e9f, 0, toast_.c_str());
    const float y = y0 + cs.y - 44;
    const int al = static_cast<int>(255 * std::min(1.0, (7 - age)) * a);
    dl->AddRectFilled({x0 + 14, y - 6}, {x0 + 14 + std::min(ts.x, cs.x - 40) + 20, y + ts.y + 8}, toast_bad_ ? IM_COL32(120, 30, 30, al) : IM_COL32(24, 80, 52, al), 5);
    dl->PushClipRect({x0 + 14, y - 6}, {x0 + cs.x - 10, y + ts.y + 8}, true);
    dl->AddText({x0 + 24, y}, IM_COL32(255, 255, 255, al), toast_.c_str());
    dl->PopClipRect();
  }
  dl->PopClipRect();
}

// Top-left key of the canvas: the symbols (wire kinds, pin states, the flow arrow, the gauge label). A click on the title folds it
// up. (The canvas's mouse bindings are in the top bar's controls popover, viewer.cpp.)
ImVec4 Canvas::legend(ImDrawList* dl, ImVec2 org, float a) {
  ImFont* font = ImGui::GetFont();
  const ImGuiIO& io = ImGui::GetIO();
  const float fs = ImGui::GetFontSize() * 0.8f, w = 196, x = org.x + 12, head = 26;
  float y = org.y + 12;
  const auto al = [&](int v) { return static_cast<int>(v * a); };
  // frame of the panel: background, title, fold chevron; returns the y of the first row's centre (-1 when folded)
  const auto frame = [&](const char* title, float body_h) {
    const bool open = legend_open_;
    const float h = open ? head + body_h : head;
    dl->AddRectFilled({x, y}, {x + w, y + h}, IM_COL32(8, 10, 14, al(225)), 6);
    dl->AddRect({x, y}, {x + w, y + h}, IM_COL32(120, 130, 148, al(90)), 6);
    dl->AddText(font, fs * 0.92f, {x + 10, y + 7}, IM_COL32(150, 160, 178, al(255)), title);
    const ImVec2 c{x + w - 14, y + head / 2};
    if (open) dl->AddTriangleFilled({c.x - 4, c.y - 2}, {c.x + 4, c.y - 2}, {c.x, c.y + 3}, IM_COL32(150, 160, 178, al(255)));
    else dl->AddTriangleFilled({c.x - 2, c.y - 4}, {c.x - 2, c.y + 4}, {c.x + 3, c.y}, IM_COL32(150, 160, 178, al(255)));
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.MousePos.x >= x && io.MousePos.x <= x + w && io.MousePos.y >= y && io.MousePos.y <= y + head)
      legend_open_ = !open;
    const float top = y;
    y += h + 8;
    return open ? top + head + 10 : -1.f;  // y of the first row's centre
  };
  const auto row = [&](float yy, const char* text) { dl->AddText(font, fs, {x + 52, yy - fs / 2}, IM_COL32(220, 224, 232, al(255)), text); };
  float yy = frame("LEGEND", 122);
  if (yy >= 0) {
  dl->AddLine({x + 10, yy}, {x + 42, yy}, kind_col("power", al(240)), 2.6f), row(yy, "power wire");
  yy += 18;
  dl->AddLine({x + 10, yy}, {x + 42, yy}, kind_col("signal", al(240)), 2.6f), row(yy, "signal wire");
  yy += 18;
  dl->AddCircleFilled({x + 20, yy}, 5, IM_COL32(150, 150, 160, al(255)));
  dl->AddCircle({x + 36, yy}, 5, IM_COL32(150, 150, 160, al(255)), 0, 2);
  row(yy, "wired / open pin");
  yy += 18;
  dl->AddTriangleFilled({x + 42, yy}, {x + 34, yy - 4}, {x + 34, yy + 4}, IM_COL32(150, 150, 160, al(255)));
  dl->AddLine({x + 10, yy}, {x + 34, yy}, IM_COL32(150, 150, 160, al(255)), 2.f);
  row(yy, "flow: source to load");
  yy += 18;
  dl->AddCircleFilled({x + 20, yy}, 5, kind_col("ground", al(255)));
  dl->AddCircleFilled({x + 36, yy}, 5, kind_col("nc", al(255)));
  row(yy, "ground / no-connect pin");
  yy += 18;
  dl->AddCircle({x + 28, yy}, 8, flag_col("error", al(255)), 0, 2.5f);
  dl->AddCircleFilled({x + 28, yy}, 4, kind_col("signal", al(255)));
  row(yy, "pin DRC flag");
  }
  return {x, org.y + 12, x + w, y - 8};
}

// The right panel: view buttons, the selected wire, the wire list. Folds up to a strip with just the toggle.
void Canvas::side_ui() {
  const float sw = side_w();
  const ImVec2 p{org_.x + sz_.x - sw, org_.y};
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(p, {p.x + sw, p.y + sz_.y}, theme::u32(theme::kPanel));
  dl->AddLine(p, {p.x, p.y + sz_.y}, theme::u32(theme::kLine));
  ImGui::SetCursorScreenPos({p.x + (side_open_ ? 6.f : 0.f), p.y + 8});
  if (ImGui::InvisibleButton("##sidetoggle", {22, 26})) side_open_ = !side_open_;
  ImGui::SetItemTooltip("%s", side_open_ ? "fold the panel up" : "open the panel");
  const ImVec2 c{p.x + (side_open_ ? 17.f : 11.f), p.y + 21};
  const ImU32 tc = theme::u32(ImGui::IsItemHovered() ? theme::kText : theme::kDim);
  if (side_open_) dl->AddTriangleFilled({c.x - 3, c.y - 5}, {c.x - 3, c.y + 5}, {c.x + 3, c.y}, tc);  // points at the edge it folds into
  else dl->AddTriangleFilled({c.x + 3, c.y - 5}, {c.x + 3, c.y + 5}, {c.x - 3, c.y}, tc);
  if (!side_open_) return;
  ImGui::SetCursorScreenPos({p.x + 10, p.y + 44});
  ImGui::BeginChild("wiring tools", {sw - 20, sz_.y - 54});
  if (theme::button("3D")) send_("flatten.end", "{}", false);
  ImGui::SetItemTooltip("back to the 3D view");
  ImGui::SameLine();
  if (theme::button("LAYOUT")) {
    send_("flatten.layout", "{}", false);
    fit_pending_ = true;
  }
  ImGui::SetItemTooltip("auto layout: signal flow left to right");
  ImGui::SameLine();
  if (theme::button("FIT")) fit();
  ImGui::SetItemTooltip("fit the sheet in the window");
  ImGui::Spacing();
  ImGui::Separator();
  const Wire* sel = nullptr;
  for (const Wire& w : wires_)
    if (w.name == sel_wire_ && !wire_hidden(w)) sel = &w;
  if (sel) {
    ImGui::Text("wire  %s", sel->name.c_str());
    ImGui::TextDisabled("%s -> %s", sel->from.c_str(), sel->to.c_str());
    ImGui::TextDisabled("%s, %d conductor(s)%s", sel->part.c_str(), sel->conductors, sel->length > 0 ? "" : "");
    if (sel->length > 0) ImGui::TextDisabled("length %.3f m", sel->length);
    if (sel->strands.size() > 1) ImGui::TextDisabled("%zu conductors drawn pin to pin", sel->strands.size());
    for (const std::string& m : sel->flag_msgs) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(flag_col(sel->flag)));
      ImGui::TextWrapped("%s", m.c_str());
      ImGui::PopStyleColor();
    }
    ImGui::SetNextItemWidth(90);
    const std::string cur = std::to_string(kAwg[awg_idx_]) + " AWG";
    if (ImGui::BeginCombo("gauge", cur.c_str())) {
      for (int i = 0; i < 12; ++i)
        if (ImGui::Selectable((std::to_string(kAwg[i]) + " AWG").c_str(), i == awg_idx_)) awg_idx_ = i;
      ImGui::EndCombo();
    }
    ImGui::SetNextItemWidth(90);
    ImGui::InputText("length m", len_buf_, sizeof len_buf_);
    if (ImGui::Button("apply")) {
      std::string args = "{\"name\":" + design::quote(sel->name) + ",\"awg\":" + std::to_string(kAwg[awg_idx_]);
      if (len_buf_[0]) args += std::string(",\"length\":") + len_buf_;
      send_("flatten.set_wire", args + "}", false);
    }
    ImGui::SameLine();
    if (ImGui::Button("delete wire")) send_("flatten.unwire", str_args({{"name", design::quote(sel->name)}}), false);
  }
  ImGui::Separator();
  theme::caption("WIRES");
  ImGui::BeginChild("wl", {0, 0});
  for (const Wire& w : wires_) {
    if (wire_hidden(w)) continue;
    std::string row = w.name + (w.awg > 0 ? "  " + std::to_string(static_cast<int>(w.awg)) + "AWG" : "");
    if (ImGui::Selectable(row.c_str(), w.name == sel_wire_)) sel_wire_ = w.name;
  }
  ImGui::EndChild();
  ImGui::EndChild();
}

void Canvas::page(ImVec2 size, const design::State& st, const std::string& selected, const std::set<std::string>& hidden) {
  org_ = ImGui::GetCursorScreenPos(), sz_ = size;
  hid_ = &hidden;
  const double dt = std::min(0.1, static_cast<double>(ImGui::GetIO().DeltaTime));
  alpha_ = std::clamp(alpha_ + (active_ ? dt : -dt) / kFade, 0.0, 1.0);
  if (fit_pending_ && !cards_.empty()) {
    fit();
    fit_pending_ = false;
  }
  if (va_.t0 >= 0) {  // ui.wire_view with seconds: smoothstep glide; the zoom runs in log space so it feels even
    const float u = std::clamp(static_cast<float>((ImGui::GetTime() - va_.t0) / va_.dur), 0.f, 1.f), e = u * u * (3 - 2 * u);
    zoom_ = std::exp(std::log(va_.z0) + (std::log(va_.z1) - std::log(va_.z0)) * e);
    ox_ = va_.ox0 + (va_.ox1 - va_.ox0) * e, oy_ = va_.oy0 + (va_.oy1 - va_.oy0) * e;
    if (u >= 1) va_.t0 = -1;
  }
  if (!active_ && alpha_ <= 0) {
    select_ui(st, selected);
    return;
  }
  canvas_ui();
  if (active_) side_ui();
}

}  // namespace wire2d
