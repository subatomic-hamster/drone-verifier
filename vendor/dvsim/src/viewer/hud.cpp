#include "viewer/hud.h"

#include <imgui.h>
#include <imgui_internal.h>  // (Register / UnregisterUserTexture)
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "viewer/build3d.h"
#include "viewer/design.h"
#include "viewer/theme.h"

namespace hud {
namespace {
using design::Json;

struct Tex {
  std::string path;
  ImTextureData* t = nullptr;
  int w = 0, h = 0;
};
struct Rect {
  float x, y, w, h;  // the enclosing frame in window pixels
};
struct Frame {
  Rect r;
  float kx = 1, ky = 1;  // child units -> window pixels (groups with a `space`)
  float ox = 0, oy = 0;  // window position of child (0, 0)
};

ImU32 col(const Json& j, ImU32 dflt) {
  if (j.kind != Json::Arr || j.arr.size() < 3) return dflt;
  auto c = [&](std::size_t i) { return static_cast<int>(std::clamp(j.arr[i].num, 0.0, 255.0)); };
  return IM_COL32(c(0), c(1), c(2), j.arr.size() > 3 ? c(3) : 255);
}
bool vec(const Json& j, double* v, std::size_t n) {
  if (j.kind != Json::Arr || j.arr.size() < n) return false;
  for (std::size_t i = 0; i < n; ++i) v[i] = j.arr[i].num;
  return true;
}
void anchor(const std::string& a, float& ax, float& ay) {
  ax = a.find('r') != std::string::npos ? 1.f : (a == "t" || a == "b" || a == "c") ? 0.5f : 0.f;
  ay = a.find('b') != std::string::npos ? 1.f : (a == "l" || a == "r" || a == "c") ? 0.5f : 0.f;
}
}  // namespace

struct Overlay::Impl {
  std::mutex mu;
  std::shared_ptr<const Json> next;  // set by any thread
  std::shared_ptr<const Json> doc;   // the one being drawn
  std::int64_t drawn_seq = -1;
  std::uint64_t drawn_tick = 0, frames = 0;
  int stable = 0;
  std::string error;
  std::vector<Label> labels;
  std::vector<Label> labels_now;
  std::map<std::string, Tex> tex;

  // ---- per-frame drawing state
  ImDrawList* dl = nullptr;
  b3::Cam cam;
  OverlayCtx ctx;

  ImGuiContext* tex_ctx = nullptr;  // the context the textures are registered with

  // Unregisters the textures when their context is still current, otherwise only frees the memory (ImGui is gone).
  void drop_textures() {
    const bool live = tex_ctx && ImGui::GetCurrentContext() == tex_ctx;
    for (auto& [id, t] : tex)
      if (t.t) {
        if (live) ImGui::UnregisterUserTexture(t.t);
        IM_DELETE(t.t);
      }
    tex.clear();
  }
  ~Impl() { drop_textures(); }

  // The last two decoded files: the camera PiP and its zoomed crop are the same PNG in two texture slots.
  struct Img {
    std::string path;
    int w = 0, h = 0;
    std::vector<unsigned char> px;
  };
  std::vector<Img> decoded;
  const Img* decode(const std::string& path) {
    for (const Img& i : decoded)
      if (i.path == path) return &i;
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!px) return nullptr;
    if (decoded.size() >= 2) decoded.erase(decoded.begin());
    decoded.push_back({path, w, h, std::vector<unsigned char>(px, px + static_cast<std::size_t>(w) * h * 4)});
    stbi_image_free(px);
    return &decoded.back();
  }

  ImTextureRef texture(const std::string& id, const std::string& path, float& iw, float& ih) {
    if (tex_ctx != ImGui::GetCurrentContext()) {  // a new viewer run: the old textures belonged to the previous context
      drop_textures();
      tex_ctx = ImGui::GetCurrentContext();
    }
    Tex& s = tex[id];
    if (s.path != path) {
      s.path = path;
      const Img* img = decode(path);
      if (!img) {
        s.w = 0;  // show nothing rather than the previous frame
        s.path.clear();  // (and retry next frame: a live run may not have written the file yet)
        return {};
      }
      const int w = img->w, h = img->h;
      const unsigned char* px = img->px.data();
      if (!s.t || s.w != w || s.h != h) {  // (re)create: a new size is a new texture
        if (s.t) {
          ImGui::UnregisterUserTexture(s.t);
          IM_DELETE(s.t);
        }
        s.t = IM_NEW(ImTextureData)();
        s.t->Create(ImTextureFormat_RGBA32, w, h);
        ImGui::RegisterUserTexture(s.t);
        std::memcpy(s.t->GetPixels(), px, static_cast<std::size_t>(w) * h * 4);
      } else {
        std::memcpy(s.t->GetPixels(), px, static_cast<std::size_t>(w) * h * 4);
        if (s.t->Status == ImTextureStatus_OK) {  // (a texture still waiting for its first upload just carries the new pixels)
          const ImTextureRect all{0, 0, static_cast<unsigned short>(w), static_cast<unsigned short>(h)};
          s.t->UpdateRect = all;
          s.t->Updates.resize(0);
          s.t->Updates.push_back(all);
          s.t->SetStatus(ImTextureStatus_WantUpdates);
        }
      }
      s.w = w, s.h = h;
    }
    iw = static_cast<float>(s.w), ih = static_cast<float>(s.h);
    return s.t && s.w ? s.t->GetTexRef() : ImTextureRef();
  }

  // position / size of an item inside frame `fr`
  static void place(const Json& it, const Frame& fr, float w, float h, float& x, float& y) {
    double at[2] = {0, 0};
    vec(it.at("at"), at, 2);
    const bool f = it.n("f") != 0;
    float px = f ? static_cast<float>(at[0]) * fr.r.w : static_cast<float>(at[0]);
    float py = f ? static_cast<float>(at[1]) * fr.r.h : static_cast<float>(at[1]);
    float ax, ay;
    anchor(it.s("a", "tl"), ax, ay);
    double m[2] = {0, 0};
    vec(it.at("m"), m, 2);  // pixel nudge after the position is resolved
    x = fr.ox + px * fr.kx - ax * w + static_cast<float>(m[0]);
    y = fr.oy + py * fr.ky - ay * h + static_cast<float>(m[1]);
  }
  static void size_of(const Json& it, const Frame& fr, float& w, float& h) {
    double sz[2] = {0, 0};
    vec(it.at("size"), sz, 2);
    const bool f = it.n("f") != 0;
    w = static_cast<float>(sz[0]) * (f ? fr.r.w : 1.f) * fr.kx;
    h = static_cast<float>(sz[1]) * (f ? fr.r.h : 1.f) * fr.ky;
    const double ar = it.n("ar");
    if (h <= 0 && ar > 0) h = w / static_cast<float>(ar);
  }

  ImFont* font_of(const Json& it) {
    if (it.s("font") == "mono" && theme::mono()) return theme::mono();
    return ImGui::GetFont();
  }
  // Draws text with its anchor at (px, py) (window pixels) with the optional padded background; returns nothing.
  void text_at(const Json& it, float px, float py, const char* dflt_anchor) {
    const std::string s = it.s("s");
    if (s.empty()) return;
    const float size = static_cast<float>(it.n("size", 14));
    ImFont* font = font_of(it);
    const ImVec2 ts = font->CalcTextSizeA(size, 1e9f, 0.f, s.c_str());
    float ax, ay;
    anchor(it.s("a", dflt_anchor), ax, ay);
    const bool bg = it.at("bg").kind == Json::Arr;
    const float pad = bg ? static_cast<float>(it.n("pad", 6)) : 0.f;
    const float w = ts.x + 2 * pad, h = ts.y + 2 * pad;
    const float x = px - ax * w, y = py - ay * h;
    if (bg) dl->AddRectFilled({x, y}, {x + w, y + h}, col(it.at("bg"), IM_COL32(0, 0, 0, 200)), static_cast<float>(it.n("r", 0)));
    dl->AddText(font, size, {x + pad, y + pad}, col(it.at("col"), IM_COL32_WHITE), s.c_str());
  }

  void item(const Json& it, const Frame& fr) {
    const std::string k = it.s("k");
    if (k == "text") {
      float x, y;
      place(it, fr, 0, 0, x, y);  // zero size: (x, y) is the anchor point itself
      text_at(it, x, y, "tl");
    } else if (k == "kv") {
      const float size = static_cast<float>(it.n("size", 13)), pad = static_cast<float>(it.n("pad", 10)), gap = static_cast<float>(it.n("gap", 26));
      ImFont* font = font_of(it);
      const float lh = size * 1.4f;
      float kw = 0, vw = 0;
      for (const Json& r : it.at("rows").arr) {
        if (r.arr.size() < 2) continue;
        kw = std::max(kw, font->CalcTextSizeA(size, 1e9f, 0.f, r.arr[0].str.c_str()).x);
        vw = std::max(vw, font->CalcTextSizeA(size, 1e9f, 0.f, r.arr[1].str.c_str()).x);
      }
      const float w = 2 * pad + kw + gap + vw, h = 2 * pad - (lh - size) + lh * static_cast<float>(it.at("rows").arr.size());
      float x, y;
      place(it, fr, w, h, x, y);
      dl->AddRectFilled({x, y}, {x + w, y + h}, col(it.at("bg"), IM_COL32(0, 0, 0, 205)));
      float ty = y + pad - (lh - size) / 2;
      for (const Json& r : it.at("rows").arr) {
        if (r.arr.size() < 2) continue;
        dl->AddText(font, size, {x + pad, ty}, col(it.at("key_col"), IM_COL32(160, 165, 175, 255)), r.arr[0].str.c_str());
        const float tw = font->CalcTextSizeA(size, 1e9f, 0.f, r.arr[1].str.c_str()).x;
        dl->AddText(font, size, {x + w - pad - tw, ty}, col(it.at("col"), IM_COL32_WHITE), r.arr[1].str.c_str());
        ty += lh;
      }
    } else if (k == "tabs") {
      const float size = static_cast<float>(it.n("size", 13)), pad = static_cast<float>(it.n("pad", 10)), gap = static_cast<float>(it.n("gap", 22));
      ImFont* font = font_of(it);
      const std::string status = it.s("s");
      const auto& tabs = it.at("items").arr;
      std::vector<float> tw;
      float row = 0;
      for (const Json& t : tabs) tw.push_back(font->CalcTextSizeA(size, 1e9f, 0.f, t.str.c_str()).x), row += tw.back() + (tw.size() > 1 ? gap : 0.f);
      const float sw = status.empty() ? 0.f : font->CalcTextSizeA(size, 1e9f, 0.f, status.c_str()).x;
      const float w = 2 * pad + std::max(row, sw), h = 2 * pad + size + (status.empty() ? 0.f : size * 0.8f + size);
      float x, y;
      place(it, fr, w, h, x, y);
      dl->AddRectFilled({x, y}, {x + w, y + h}, col(it.at("bg"), IM_COL32(0, 0, 0, 205)));
      const int active = static_cast<int>(it.n("active", -1));
      float tx = x + pad;
      for (std::size_t i = 0; i < tabs.size(); ++i) {
        const bool on = static_cast<int>(i) == active;
        dl->AddText(font, size, {tx, y + pad}, on ? col(it.at("col"), IM_COL32_WHITE) : col(it.at("dim"), IM_COL32(160, 165, 175, 255)), tabs[i].str.c_str());
        if (on) dl->AddLine({tx, y + pad + size + 4.f}, {tx + tw[i], y + pad + size + 4.f}, col(it.at("col"), IM_COL32_WHITE), 1.5f);
        tx += tw[i] + gap;
      }
      if (!status.empty()) dl->AddText(font, size, {x + pad, y + pad + size * 1.8f}, col(it.at("col"), IM_COL32_WHITE), status.c_str());
    } else if (k == "rect") {
      float w, h, x, y;
      size_of(it, fr, w, h);
      place(it, fr, w, h, x, y);
      const float r = static_cast<float>(it.n("r", 0));
      if (it.at("fill").kind == Json::Arr) dl->AddRectFilled({x, y}, {x + w, y + h}, col(it.at("fill"), 0), r);
      if (it.at("line").kind == Json::Arr)
        dl->AddRect({x, y}, {x + w, y + h}, col(it.at("line"), IM_COL32_WHITE), r, 0, static_cast<float>(it.n("lw", 1)));
    } else if (k == "line") {
      std::vector<ImVec2> pts;
      for (const Json& p : it.at("pts").arr) {
        double v[2];
        if (!vec(p, v, 2)) continue;
        pts.push_back({fr.ox + static_cast<float>(v[0]) * fr.kx, fr.oy + static_cast<float>(v[1]) * fr.ky});
      }
      if (pts.size() >= 2)
        dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), col(it.at("col"), IM_COL32_WHITE), 0, static_cast<float>(it.n("lw", 1)));
    } else if (k == "image") {
      float w, h, x, y, iw = 0, ih = 0;
      size_of(it, fr, w, h);
      const ImTextureRef tr = texture(it.s("id", "img"), it.s("path"), iw, ih);
      if (!tr._TexData && tr._TexID == 0) return;
      if (h <= 0 && iw > 0) h = w * ih / iw;
      place(it, fr, w, h, x, y);
      double uv[4] = {0, 0, 1, 1};
      vec(it.at("uv"), uv, 4);
      dl->AddImage(tr, {x, y}, {x + w, y + h}, {static_cast<float>(uv[0]), static_cast<float>(uv[1])},
                   {static_cast<float>(uv[2]), static_cast<float>(uv[3])}, col(it.at("tint"), IM_COL32_WHITE));
    } else if (k == "group") {
      float w, h, x, y;
      size_of(it, fr, w, h);
      place(it, fr, w, h, x, y);
      double sp[2] = {w, h};
      vec(it.at("space"), sp, 2);
      if (it.at("fill").kind == Json::Arr) dl->AddRectFilled({x, y}, {x + w, y + h}, col(it.at("fill"), 0));
      Frame g;
      g.r = {x, y, w, h};
      g.kx = sp[0] > 0 ? w / static_cast<float>(sp[0]) : 1.f;
      g.ky = sp[1] > 0 ? h / static_cast<float>(sp[1]) : 1.f;
      g.ox = x, g.oy = y;
      dl->PushClipRect({x, y}, {x + w, y + h}, true);
      Frame child = g;
      child.r = {x, y, static_cast<float>(sp[0]), static_cast<float>(sp[1])};  // fractions inside a group are of its own space
      for (const Json& c : it.at("items").arr) guarded(c, child);
      dl->PopClipRect();
      if (it.at("line").kind == Json::Arr)
        dl->AddRect({x, y}, {x + w, y + h}, col(it.at("line"), IM_COL32_WHITE), 0, 0, static_cast<float>(it.n("lw", 1)));
    } else if (k == "wtext" || k == "wline" || k == "wring" || k == "wdot") {
      world_item(it, k, fr);
    }
  }

  bool project(const double* w, ImVec2& out) const { return cam.project({w[0], w[1], w[2]}, out); }
  void world_item(const Json& it, const std::string& k, const Frame& fr) {
    if (k == "wline") {
      std::vector<ImVec2> pts;
      const ImU32 c = col(it.at("col"), IM_COL32_WHITE);
      const float lw = static_cast<float>(it.n("lw", 1));
      auto flush = [&] {
        if (pts.size() >= 2) dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), c, 0, lw);
        pts.clear();
      };
      for (const Json& p : it.at("pts").arr) {
        double v[3];
        ImVec2 s;
        if (!vec(p, v, 3) || !project(v, s)) {
          flush();  // behind the camera: break the polyline
          continue;
        }
        pts.push_back(s);
      }
      flush();
      return;
    }
    double w[3];
    ImVec2 s;
    if (!vec(it.at("w"), w, 3)) throw std::runtime_error(k + ": 'w' must be [x, y, z]");
    const bool vis = project(w, s) && s.x >= fr.r.x && s.x <= fr.r.x + fr.r.w && s.y >= fr.r.y && s.y <= fr.r.y + fr.r.h;
    if (k == "wtext") {
      labels_now.push_back({it.s("s"), s.x, s.y, vis});
      if (!vis) return;
      double off[2] = {0, 0};
      vec(it.at("off"), off, 2);
      text_at(it, s.x + static_cast<float>(off[0]), s.y + static_cast<float>(off[1]), "b");
    } else if (vis && k == "wring") {
      dl->AddCircle(s, static_cast<float>(it.n("r", 8)), col(it.at("col"), IM_COL32_WHITE), 24, static_cast<float>(it.n("lw", 1.5)));
    } else if (vis) {
      dl->AddCircleFilled(s, static_cast<float>(it.n("r", 3)), col(it.at("col"), IM_COL32_WHITE), 16);
    }
  }

  void guarded(const Json& it, const Frame& fr) {
    try {
      item(it, fr);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> g(mu);
      if (error.empty()) error = e.what();
    }
  }
};

Overlay::Overlay() : p_(std::make_unique<Impl>()) {}
Overlay::~Overlay() = default;

void Overlay::set(const std::string& json) {
  auto doc = std::make_shared<Json>(design::Parser(json).parse());
  if (doc->kind != Json::Obj) throw std::runtime_error("hud: the document must be a JSON object");
  std::lock_guard<std::mutex> g(p_->mu);
  p_->next = std::move(doc);
}

bool Overlay::camera(CameraCtl& c) {
  std::shared_ptr<const Json> d;
  {
    std::lock_guard<std::mutex> g(p_->mu);
    if (p_->next) p_->doc = std::move(p_->next), p_->next.reset();  // adopt here too: the camera is asked before the draw
    d = p_->doc;
  }
  if (!d) return false;
  const Json& cj = d->at("camera");
  double t[3];
  if (cj.kind != Json::Obj || !vec(cj.at("target"), t, 3)) return false;
  for (int k = 0; k < 3; ++k) c.target[k] = t[k];
  c.yaw = cj.n("yaw", c.yaw), c.pitch = cj.n("pitch", c.pitch), c.dist = cj.n("dist", c.dist);
  return true;
}

void Overlay::draw(const OverlayCtx& ctx) {
  Impl& s = *p_;
  std::shared_ptr<const Json> d;
  {
    std::lock_guard<std::mutex> g(s.mu);
    if (s.next) s.doc = std::move(s.next), s.next.reset();
    d = s.doc;
  }
  ++s.frames;
  if (!d || !ctx.draw_list) return;
  s.dl = static_cast<ImDrawList*>(ctx.draw_list);
  s.ctx = ctx;
  Camera c;
  for (int k = 0; k < 3; ++k) c.pos[k] = ctx.cam_pos[k], c.lookat[k] = ctx.cam_look[k];
  c.fovy = ctx.cam_fovy;
  s.cam = b3::Cam::make(c, ctx.window_w, ctx.window_h);
  s.labels_now.clear();
  Frame root;
  root.r = {ctx.x, ctx.y, ctx.w, ctx.h};
  root.ox = ctx.x, root.oy = ctx.y;
  s.dl->PushClipRect({ctx.x, ctx.y}, {ctx.x + ctx.w, ctx.y + ctx.h}, true);
  for (const Json& it : d->at("items").arr) s.guarded(it, root);
  s.dl->PopClipRect();

  const std::int64_t seq = static_cast<std::int64_t>(d->n("seq", -1));
  const bool has_tick = d->at("tick").kind == Json::Num;
  const std::uint64_t tick = has_tick ? static_cast<std::uint64_t>(d->n("tick")) : ctx.tick;
  std::lock_guard<std::mutex> g(s.mu);
  s.stable = ctx.tick != tick ? 0 : seq == s.drawn_seq ? s.stable + 1 : 1;
  s.drawn_seq = seq;
  s.drawn_tick = ctx.tick;
  s.labels = s.labels_now;
}

void Overlay::release() { p_->drop_textures(); }

Overlay::DryRun Overlay::dry_run(const OverlayCtx& in) {
  ImGuiContext* prev = ImGui::GetCurrentContext();
  ImGuiContext* c = ImGui::CreateContext();
  ImGui::SetCurrentContext(c);
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
  theme::init();
  io.DisplaySize = {static_cast<float>(in.window_w), static_cast<float>(in.window_h)};
  io.DeltaTime = 1.f / 60;
  ImGui::NewFrame();
  OverlayCtx o = in;
  ImDrawList* dl = ImGui::GetBackgroundDrawList();
  o.draw_list = dl;
  const int v0 = dl->VtxBuffer.Size;
  draw(o);
  Overlay::DryRun r;
  r.vertices = dl->VtxBuffer.Size - v0;
  r.labels = labels();
  r.error = progress().error;
  ImGui::EndFrame();
  release();
  ImGui::DestroyContext(c);
  ImGui::SetCurrentContext(prev);
  return r;
}

Overlay::Progress Overlay::progress() const {
  std::lock_guard<std::mutex> g(p_->mu);
  return {p_->drawn_seq, p_->drawn_tick, p_->stable, p_->frames, p_->error};
}
std::vector<Label> Overlay::labels() const {
  std::lock_guard<std::mutex> g(p_->mu);
  return p_->labels;
}

}  // namespace hud
