#include "viewer/build3d.h"

#include <algorithm>

namespace b3 {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr int kEdges[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7}, {7, 6}, {6, 4}, {0, 4}, {1, 5}, {3, 7}, {2, 6}};
constexpr int kFaces[6][4] = {{0, 1, 3, 2}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 3, 7, 5}};

void box_corners(const Box& b, V3 out[8]) {
  for (int i = 0; i < 8; ++i) {
    const V3 l{(i & 1) ? b.h.x : -b.h.x, (i & 2) ? b.h.y : -b.h.y, (i & 4) ? b.h.z : -b.h.z};
    out[i] = b.c + b.R * l;
  }
}
double seg_dist(ImVec2 p, ImVec2 a, ImVec2 b) {
  const double vx = b.x - a.x, vy = b.y - a.y, wx = p.x - a.x, wy = p.y - a.y;
  const double l2 = vx * vx + vy * vy;
  const double t = l2 > 0 ? std::clamp((vx * wx + vy * wy) / l2, 0.0, 1.0) : 0;
  return std::hypot(wx - t * vx, wy - t * vy);
}
// Orthonormal basis (u, v) of the plane normal to e.
void plane_basis(V3 e, V3& u, V3& v) {
  u = unit(cross(e, std::abs(e.z) < 0.9 ? V3{0, 0, 1} : V3{1, 0, 0}));
  v = cross(e, u);
}
double axis_param(const Ray& r, V3 c, V3 e) {  // t of the point c + e t on the axis closest to the ray
  const V3 w0 = c - r.o;
  const double b = dot(e, r.d), cc = dot(r.d, r.d), d = dot(e, w0), ee = dot(r.d, w0), den = cc - b * b;
  return std::abs(den) < 1e-12 ? 0 : (b * ee - cc * d) / den;
}
ImU32 with_alpha(ImU32 c, float a) {
  const ImU32 al = static_cast<ImU32>(std::clamp(a, 0.f, 1.f) * ((c >> IM_COL32_A_SHIFT) & 0xFF));
  return (c & ~IM_COL32_A_MASK) | (al << IM_COL32_A_SHIFT);
}
}  // namespace

Mat3 rot_axis(V3 a, double th) {
  a = unit(a);
  const double c = std::cos(th), s = std::sin(th), k = 1 - c;
  Mat3 r;
  const double m[9] = {c + a.x * a.x * k, a.x * a.y * k - a.z * s, a.x * a.z * k + a.y * s,
                       a.y * a.x * k + a.z * s, c + a.y * a.y * k, a.y * a.z * k - a.x * s,
                       a.z * a.x * k - a.y * s, a.z * a.y * k + a.x * s, c + a.z * a.z * k};
  std::copy(m, m + 9, r.m);
  return r;
}
Mat3 rpy_to_mat(V3 v) { return rot_axis({0, 0, 1}, v.z) * rot_axis({0, 1, 0}, v.y) * rot_axis({1, 0, 0}, v.x); }
V3 mat_to_rpy(const Mat3& R) {
  const double sp = -R.m[6], p = std::asin(std::clamp(sp, -1.0, 1.0));
  if (std::abs(sp) > 1 - 1e-9) return {0, p, std::atan2(-R.m[1], R.m[4])};
  return {std::atan2(R.m[7], R.m[8]), p, std::atan2(R.m[3], R.m[0])};
}

Cam Cam::make(const Camera& c, double w, double h) {
  Cam v;
  v.pos = {c.pos[0], c.pos[1], c.pos[2]};
  v.f = unit(V3{c.lookat[0], c.lookat[1], c.lookat[2]} - v.pos);
  const V3 up0 = std::abs(v.f.z) > 0.9999 ? V3{0, 1, 0} : V3{0, 0, 1};  // as the renderer (render/renderer.cpp make_view)
  v.r = unit(cross(v.f, up0));
  v.u = cross(v.r, v.f);
  v.t = 1 / std::tan(c.fovy * kPi / 360);
  v.sx = v.t * h / w;
  v.w = w;
  v.h = h;
  return v;
}
bool Cam::project(V3 p, ImVec2& out) const {
  const V3 rel = p - pos;
  const double z = dot(f, rel);
  if (z < 0.02) return false;
  out.x = static_cast<float>((sx * dot(r, rel) / z * 0.5 + 0.5) * w);
  out.y = static_cast<float>((-t * dot(u, rel) / z * 0.5 + 0.5) * h);
  return true;
}
Ray Cam::ray(ImVec2 px) const {
  const double nx = px.x / w * 2 - 1, ny = px.y / h * 2 - 1;
  return {pos, unit(f + r * (nx / sx) + u * (-ny / t))};
}
double Cam::px_per_m(V3 p) const { return 0.5 * h * t / std::max(0.02, dot(f, p - pos)); }

bool ray_box(const Ray& ray, const Box& b, double& tout, V3* normal) {
  const Mat3 Rt = b.R.T();
  const V3 o = Rt * (ray.o - b.c), d = Rt * ray.d;
  double t0 = 0, t1 = 1e300;
  int face = -1;
  double sign = 1;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(d[i]) < 1e-12) {
      if (std::abs(o[i]) > b.h[i]) return false;
      continue;
    }
    double a = (-b.h[i] - o[i]) / d[i], c = (b.h[i] - o[i]) / d[i];
    double sg = -1;
    if (a > c) { std::swap(a, c); sg = 1; }
    if (a > t0) { t0 = a; face = i; sign = sg; }
    t1 = std::min(t1, c);
    if (t0 > t1) return false;
  }
  tout = t0;
  if (normal) *normal = face >= 0 ? b.R.col(face) * sign : V3{0, 0, 1};
  return true;
}
Hit pick(const Cam& cam, ImVec2 px, const Scene& scene, const std::string& skip, const std::set<std::string>* hidden) {
  const Ray r = cam.ray(px);
  Hit best;
  for (const auto& [id, g] : scene) {
    if (id == skip || (hidden && hidden->count(id))) continue;
    for (const Box& b : g.boxes) {
      double t;
      V3 n;
      if (ray_box(r, b, t, &n) && t > 0 && t < best.t) best = {id, t, r.o + r.d * t, n};
    }
  }
  return best;
}
bool ray_plane(const Ray& r, V3 p0, V3 n, V3& hit) {
  const double den = dot(n, r.d);
  if (std::abs(den) < 1e-9) return false;
  const double t = dot(n, p0 - r.o) / den;
  if (t < 0) return false;
  hit = r.o + r.d * t;
  return true;
}

V3 snap_translate(const Scene& scene, const InstGeo& mv, V3 delta, const bool mask[3], double thresh, const Settings& s,
                  std::vector<Guide>& guides) {
  V3 d = delta, lo = mv.lo + d, hi = mv.hi + d;
  bool snapped[3] = {false, false, false};
  struct Pick { double e = 0; double value = 0; const InstGeo* other = nullptr; } best[3];
  if (s.edges)
    for (int ax = 0; ax < 3; ++ax) {
      if (!mask[ax]) continue;
      const double mf[3] = {lo[ax], 0.5 * (lo[ax] + hi[ax]), hi[ax]};
      Pick b;
      for (const auto& [id, o] : scene) {
        if (id == mv.id || o.boxes.empty()) continue;
        const double of[3] = {o.lo[ax], 0.5 * (o.lo[ax] + o.hi[ax]), o.hi[ax]};
        for (double m : mf)
          for (double v : of)
            if (std::abs(v - m) <= thresh && (!b.other || std::abs(v - m) < std::abs(b.e))) b = {v - m, v, &o};
      }
      if (b.other) {
        d[ax] += b.e;
        lo[ax] += b.e;
        hi[ax] += b.e;
        snapped[ax] = true;
        best[ax] = b;
      }
    }
  if (s.ports && !mv.ports.empty()) {
    double bd = thresh * 1.6;
    V3 fix;
    const InstGeo* bo = nullptr;
    V3 at;
    for (const auto& [id, o] : scene) {
      if (id == mv.id) continue;
      for (const Port& op : o.ports)
        for (const Port& mp : mv.ports) {
          V3 diff = op.pos - (mp.pos + d);
          V3 in, out;
          for (int ax = 0; ax < 3; ++ax) (mask[ax] ? in : out)[ax] = diff[ax];
          const double dist = std::max(len(in), len(out));
          if (dist < bd) { bd = dist; fix = in; bo = &o; at = op.pos; }
        }
    }
    if (bo) {
      d = d + fix;
      lo = lo + fix;
      hi = hi + fix;
      for (int ax = 0; ax < 3; ++ax) snapped[ax] = snapped[ax] || (mask[ax] && fix[ax] != 0);
      guides.push_back({0, at, at, true});
    }
  }
  if (s.grid > 0)
    for (int ax = 0; ax < 3; ++ax)
      if (mask[ax] && !snapped[ax]) d[ax] += std::round((mv.pos[ax] + d[ax]) / s.grid) * s.grid - (mv.pos[ax] + d[ax]);
  for (int ax = 0; ax < 3; ++ax) {
    if (!best[ax].other || !s.guides) continue;
    V3 a = (mv.lo + mv.hi) * 0.5 + d, b = (best[ax].other->lo + best[ax].other->hi) * 0.5;
    a[ax] = b[ax] = best[ax].value;
    guides.push_back({ax, a, b, false});
  }
  return d;
}

ImU32 axis_color(int axis, float alpha) {
  static const ImU32 c[3] = {IM_COL32(255, 82, 82, 255), IM_COL32(94, 220, 110, 255), IM_COL32(86, 148, 255, 255)};
  return with_alpha(c[axis], alpha);
}

void draw_box(ImDrawList* dl, const Cam& cam, const Box& b, ImU32 col, float th) {
  V3 c[8];
  box_corners(b, c);
  ImVec2 p[8];
  bool ok[8];
  for (int i = 0; i < 8; ++i) ok[i] = cam.project(c[i], p[i]);
  for (const auto& e : kEdges)  // a soft dark halo under each edge keeps the outline crisp on the light studio floor
    if (ok[e[0]] && ok[e[1]]) dl->AddLine(p[e[0]], p[e[1]], IM_COL32(6, 8, 12, 70), th + 2.4f);
  for (const auto& e : kEdges)
    if (ok[e[0]] && ok[e[1]]) dl->AddLine(p[e[0]], p[e[1]], col, th);
}
void draw_brackets(ImDrawList* dl, const Cam& cam, V3 lo, V3 hi, ImU32 col, float th) {
  const V3 ext = hi - lo;
  for (int i = 0; i < 8; ++i) {
    const V3 c{(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z};
    ImVec2 pc;
    if (!cam.project(c, pc)) continue;
    for (int ax = 0; ax < 3; ++ax) {
      V3 e = c;
      e[ax] += ((i >> ax) & 1 ? -1 : 1) * ext[ax] * 0.22;
      ImVec2 pe;
      if (cam.project(e, pe)) {
        dl->AddLine(pc, pe, IM_COL32(6, 8, 12, 80), th + 2.4f);
        dl->AddLine(pc, pe, col, th);
      }
    }
  }
}
void draw_ring(ImDrawList* dl, const Cam& cam, V3 c, V3 axis, double radius, ImU32 col, float th) {
  V3 u, v;
  plane_basis(axis, u, v);
  ImVec2 prev;
  bool have = false;
  for (int i = 0; i <= 64; ++i) {
    const double a = 2 * kPi * i / 64;
    ImVec2 p;
    const bool ok = cam.project(c + (u * std::cos(a) + v * std::sin(a)) * radius, p);
    if (ok && have) dl->AddLine(prev, p, col, th);
    prev = p;
    have = ok;
  }
}
void draw_guides(ImDrawList* dl, const Cam& cam, const std::vector<Guide>& gs, float alpha) {
  for (const Guide& g : gs) {
    ImVec2 pa, pb;
    if (!cam.project(g.a, pa) || !cam.project(g.b, pb)) continue;
    if (g.port) {
      dl->AddCircle(pb, 9, IM_COL32(255, 255, 255, static_cast<int>(230 * alpha)), 24, 1.5f);
      dl->AddCircleFilled(pb, 2.5f, IM_COL32(255, 255, 255, static_cast<int>(230 * alpha)));
      continue;
    }
    const ImU32 col = axis_color(g.axis, alpha);
    const double dx = pb.x - pa.x, dy = pb.y - pa.y, l = std::max(1.0, std::hypot(dx, dy));
    const ImVec2 ea{static_cast<float>(pa.x - dx / l * 12), static_cast<float>(pa.y - dy / l * 12)};
    const ImVec2 eb{static_cast<float>(pb.x + dx / l * 12), static_cast<float>(pb.y + dy / l * 12)};
    dl->AddLine(ea, eb, col, 1.5f);
    dl->AddCircleFilled(pa, 3.f, col);
    dl->AddCircleFilled(pb, 3.f, col);
  }
}
void draw_ghost(ImDrawList* dl, const Cam& cam, const std::vector<Box>& local, V3 pos, const Mat3& R, ImU32 col) {
  for (const Box& lb : local) {
    const Box b{pos + R * lb.c, R * lb.R, lb.h};
    V3 c[8];
    box_corners(b, c);
    ImVec2 p[8];
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok = cam.project(c[i], p[i]) && ok;
    if (!ok) continue;
    const ImU32 fill = with_alpha(col, 0.14f);
    for (const auto& f : kFaces) {
      ImVec2 q[4] = {p[f[0]], p[f[1]], p[f[2]], p[f[3]]};
      dl->AddConvexPolyFilled(q, 4, fill);
    }
    draw_box(dl, cam, b, col, 1.5f);
  }
}
Box moved_box(const Box& b, V3 pivot, const Mat3& Rd, V3 shift) { return {pivot + Rd * (b.c - pivot) + shift, Rd * b.R, b.h}; }

// ---- gizmo ----------------------------------------------------------------------------------------------------------------
GizmoOut Gizmo::update(const Cam& cam, const Scene& scene, const InstGeo& tg, const Settings& st, bool ok, bool invert, ImVec4 clip) {
  GizmoOut out;
  if (mode == Off) { drag_ = -1; return out; }
  ImDrawList* dl = ImGui::GetBackgroundDrawList();
  dl->PushClipRect({clip.x, clip.y}, {clip.z, clip.w}, true);
  const ImGuiIO& io = ImGui::GetIO();
  const ImVec2 m = io.MousePos;
  const V3 C = tg.pos;
  ImVec2 cs;
  if (!cam.project(C, cs)) { dl->PopClipRect(); return out; }
  const double L = 96.0 / cam.px_per_m(C);  // handles are 96 px long whatever the zoom
  const double ring = L * 0.78;

  // hit testing
  if (drag_ < 0) {
    hover_ = -1;
    if (ok) {
      double best = 9;
      if (mode == Move) {
        if (std::hypot(m.x - cs.x, m.y - cs.y) < 11) { hover_ = 3; best = 0; }
        for (int i = 0; i < 3 && best > 0; ++i) {
          ImVec2 e;
          if (!cam.project(C + axis_vec(i) * L, e)) continue;
          const double dd = seg_dist(m, cs, e);
          if (dd < best) { best = dd; hover_ = i; }
        }
      } else {
        for (int i = 0; i < 3; ++i) {
          V3 u, v;
          plane_basis(axis_vec(i), u, v);
          ImVec2 prev;
          bool have = false;
          for (int k = 0; k <= 64; ++k) {
            const double a = 2 * kPi * k / 64;
            ImVec2 p;
            const bool okp = cam.project(C + (u * std::cos(a) + v * std::sin(a)) * ring, p);
            if (okp && have) {
              const double dd = seg_dist(m, prev, p);
              if (dd < best) { best = dd; hover_ = i; }
            }
            prev = p;
            have = okp;
          }
        }
      }
    }
    if (hover_ >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      drag_ = hover_;
      moved_ = 0;
      const Ray r = cam.ray(m);
      if (mode == Move) {
        if (drag_ < 3) t0_ = axis_param(r, C, axis_vec(drag_));
        else if (!ray_plane(r, C, {0, 0, 1}, p0_)) drag_ = -1;
      } else {
        V3 hit, u, v;
        if (!ray_plane(r, C, axis_vec(drag_), hit)) drag_ = -1;
        else {
          plane_basis(axis_vec(drag_), u, v);
          a0_ = std::atan2(dot(hit - C, v), dot(hit - C, u));
        }
      }
    }
  }
  out.hot = hover_ >= 0 || drag_ >= 0;

  Mat3 Rd;
  V3 shift;
  if (drag_ >= 0) {
    const Ray r = cam.ray(m);
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      out.dragging = true;
    } else {
      out.released = moved_ > 1e-6;
      out.dragging = false;
    }
    const bool snap_on = !invert;
    Settings s = st;
    if (!snap_on) { s.grid = 0; s.edges = false; s.ports = false; s.angle = 0; }
    if (mode == Move) {
      V3 d;
      bool mask[3] = {false, false, false};
      if (drag_ < 3) {
        d = axis_vec(drag_) * (axis_param(r, C, axis_vec(drag_)) - t0_);
        mask[drag_] = true;
      } else {
        V3 hit;
        if (ray_plane(r, C, {0, 0, 1}, hit)) d = hit - p0_;
        d.z = 0;
        mask[0] = mask[1] = true;
      }
      d = snap_translate(scene, tg, d, mask, 8.0 / cam.px_per_m(C), s, out.guides);
      out.delta = d;
      shift = d;
      moved_ = len(d);
    } else {
      V3 hit, u, v;
      double ang = 0;
      if (ray_plane(r, C, axis_vec(drag_), hit)) {
        plane_basis(axis_vec(drag_), u, v);
        ang = std::atan2(dot(hit - C, v), dot(hit - C, u)) - a0_;
        ang = std::remainder(ang, 2 * kPi);
        if (s.angle > 0) ang = std::round(ang * 180 / kPi / s.angle) * s.angle * kPi / 180;
      }
      out.rotate = true;
      out.axis = axis_vec(drag_);
      out.angle = ang * 180 / kPi;
      Rd = rot_axis(out.axis, ang);
      moved_ = std::abs(out.angle);
    }
    if (!out.dragging) drag_ = -1;
    // preview: the part's boxes at the new pose
    for (const Box& b : tg.boxes) draw_box(dl, cam, moved_box(b, C, Rd, shift), IM_COL32(255, 255, 255, 200), 1.25f);
  }

  // handles at the (previewed) origin
  const V3 Cd = C + shift;
  ImVec2 cp;
  if (cam.project(Cd, cp)) {
    const int act = drag_ >= 0 ? drag_ : hover_;
    if (mode == Move) {
      for (int i = 0; i < 3; ++i) {
        ImVec2 e;
        if (!cam.project(Cd + axis_vec(i) * L, e)) continue;
        const bool on = act == i;
        const ImU32 col = axis_color(i, on || act < 0 ? 1.f : 0.55f);
        dl->AddLine(cp, e, col, on ? 4.f : 2.5f);
        const double dx = e.x - cp.x, dy = e.y - cp.y, l = std::max(1.0, std::hypot(dx, dy));
        const ImVec2 n{static_cast<float>(-dy / l), static_cast<float>(dx / l)}, ux{static_cast<float>(dx / l), static_cast<float>(dy / l)};
        dl->AddTriangleFilled({e.x + ux.x * 12, e.y + ux.y * 12}, {e.x + n.x * 5.5f, e.y + n.y * 5.5f}, {e.x - n.x * 5.5f, e.y - n.y * 5.5f}, col);
        if (drag_ == i) {  // the full travel line while dragging
          ImVec2 a, b2;
          if (cam.project(C - axis_vec(i) * 50, a) && cam.project(C + axis_vec(i) * 50, b2)) dl->AddLine(a, b2, axis_color(i, 0.25f), 1.f);
        }
      }
      dl->AddRectFilled({cp.x - 6, cp.y - 6}, {cp.x + 6, cp.y + 6}, act == 3 ? IM_COL32(255, 255, 255, 255) : IM_COL32(255, 255, 255, 150), 2);
    } else {
      for (int i = 0; i < 3; ++i) {
        const bool on = act == i;
        draw_ring(dl, cam, Cd, axis_vec(i), ring, axis_color(i, on || act < 0 ? 1.f : 0.45f), on ? 3.5f : 2.f);
      }
      if (drag_ >= 0) {
        V3 u, v;
        plane_basis(axis_vec(drag_), u, v);
        const double a = a0_ + out.angle * kPi / 180;
        ImVec2 p0, p1;
        if (cam.project(C + (u * std::cos(a0_) + v * std::sin(a0_)) * ring, p0)) dl->AddLine(cp, p0, IM_COL32(255, 255, 255, 90), 1.f);
        if (cam.project(C + (u * std::cos(a) + v * std::sin(a)) * ring, p1)) {
          dl->AddLine(cp, p1, IM_COL32(255, 255, 255, 230), 1.5f);
          char buf[24];
          std::snprintf(buf, sizeof buf, "%+.0f deg", out.angle);
          dl->AddText({p1.x + 10, p1.y - 8}, IM_COL32(255, 255, 255, 240), buf);
        }
      }
    }
  }
  if (out.dragging && out.guides.size() && st.guides) draw_guides(dl, cam, out.guides, 1.f);
  if (out.dragging && mode == Move && moved_ > 0) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%+.1f %+.1f %+.1f mm", out.delta.x * 1000, out.delta.y * 1000, out.delta.z * 1000);
    dl->AddText({m.x + 16, m.y + 14}, IM_COL32(255, 255, 255, 235), buf);
  }
  dl->PopClipRect();
  return out;
}

}  // namespace b3
