#include "render/renderer.h"

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION 1003000
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "geom_spv.h"
#include "render/window_renderer.h"

namespace {
constexpr VkFormat kColorFmt = VK_FORMAT_R8G8B8A8_UNORM;  // headless colour target; a window uses its swapchain format
constexpr VkFormat kLinFmt = VK_FORMAT_R32_SFLOAT;  // linear depth in metres
constexpr VkFormat kDepthFmt = VK_FORMAT_D32_SFLOAT;
constexpr int kShadowSize = 2048;
constexpr double kNear = 0.05, kFar = 1000;  // metres; depth testing only (the linear-depth output is exact)
constexpr double kFarTerrain = 5000;          // scenes with a heightfield (mountains are seen from kilometres); others keep kFar bit for bit
constexpr double kPi = 3.14159265358979323846;
constexpr double kUnboundedPlane = 500;  // half extent drawn for a plane of size <= 0 (inside kFar)
constexpr float kEdgePx = 1.6f;          // crease-edge line width in pixels

void check(VkResult r, const char* what) {
  if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan: ") + what + " failed (VkResult " + std::to_string(r) + ")");
}

// ---- CPU tessellation: interleaved position xyz + normal xyz in the geom frame, size baked in -------------------------
using Verts = std::vector<float>;
void vert(Verts& o, double px, double py, double pz, double nx, double ny, double nz) {
  for (double d : {px, py, pz, nx, ny, nz}) o.push_back(static_cast<float>(d));
}
struct Pn { double p[3], n[3]; };
void quad(Verts& o, const Pn& a, const Pn& b, const Pn& c, const Pn& d) {
  for (const Pn* q : {&a, &b, &c, &a, &c, &d}) vert(o, q->p[0], q->p[1], q->p[2], q->n[0], q->n[1], q->n[2]);
}

// Surface of revolution about local z. A strip is a polyline of (rho, z, n_rho, n_z) profile points.
struct Prof { double rho, z, nr, nz; };
void revolve(Verts& o, const std::vector<Prof>& strip, int segs) {
  auto at = [&](const Prof& q, int j) {
    const double phi = 2 * kPi * j / segs, c = std::cos(phi), s = std::sin(phi);
    return Pn{{q.rho * c, q.rho * s, q.z}, {q.nr * c, q.nr * s, q.nz}};
  };
  for (std::size_t i = 0; i + 1 < strip.size(); ++i)
    for (int j = 0; j < segs; ++j) quad(o, at(strip[i], j), at(strip[i], j + 1), at(strip[i + 1], j + 1), at(strip[i + 1], j));
}

// Capsule along z with hemispherical caps; radius r, half length h (h = 0: sphere).
std::vector<Prof> capsule_profile(double r, double h, int rings) {
  std::vector<Prof> s;
  for (int i = 0; i <= rings; ++i) {  // bottom pole -> equator
    const double th = -kPi / 2 + (kPi / 2) * i / rings;
    s.push_back({r * std::cos(th), -h + r * std::sin(th), std::cos(th), std::sin(th)});
  }
  for (int i = 0; i <= rings; ++i) {  // equator -> top pole
    const double th = (kPi / 2) * i / rings;
    s.push_back({r * std::cos(th), h + r * std::sin(th), std::cos(th), std::sin(th)});
  }
  return s;
}

Verts tessellate(const GeomVisual& g) {
  Verts o;
  const double a = g.size[0], b = g.size[1], c = g.size[2];
  constexpr int kSegs = 32, kRings = 8;
  switch (g.type) {
    case GeomType::sphere: revolve(o, capsule_profile(a, 0, kRings), kSegs); break;
    case GeomType::capsule: revolve(o, capsule_profile(a, b, kRings), kSegs); break;
    case GeomType::ellipsoid: {  // unit sphere scaled; normal of x^2/a^2+.. = 1 is (x/a^2, ..) = (nx/a, ..)
      Verts s;
      revolve(s, capsule_profile(1, 0, kRings), kSegs);
      for (std::size_t i = 0; i < s.size(); i += 6) {
        const double sc[3] = {a, b, c};
        double n[3] = {s[i + 3] / sc[0], s[i + 4] / sc[1], s[i + 5] / sc[2]};
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        vert(o, s[i] * a, s[i + 1] * b, s[i + 2] * c, n[0] / len, n[1] / len, n[2] / len);
      }
      break;
    }
    case GeomType::cylinder:
      revolve(o, {{0, -b, 0, -1}, {a, -b, 0, -1}}, kSegs);
      revolve(o, {{a, -b, 1, 0}, {a, b, 1, 0}}, kSegs);
      revolve(o, {{a, b, 0, 1}, {0, b, 0, 1}}, kSegs);
      break;
    case GeomType::box: {
      const double h[3] = {a, b, c};
      for (int ax = 0; ax < 3; ++ax)
        for (double sgn : {-1.0, 1.0}) {
          const int u = (ax + 1) % 3, v = (ax + 2) % 3;
          auto corner = [&](double su, double sv) {
            Pn q{};
            q.p[ax] = sgn * h[ax];
            q.p[u] = su * h[u];
            q.p[v] = sv * h[v];
            q.n[ax] = sgn;
            return q;
          };
          quad(o, corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1));
        }
      break;
    }
    case GeomType::plane: {
      const double hx = a > 0 ? a : kUnboundedPlane, hy = b > 0 ? b : kUnboundedPlane;
      const Pn q0{{-hx, -hy, 0}, {0, 0, 1}}, q1{{hx, -hy, 0}, {0, 0, 1}}, q2{{hx, hy, 0}, {0, 0, 1}}, q3{{-hx, hy, 0}, {0, 0, 1}};
      quad(o, q0, q1, q2, q3);
      break;
    }
    case GeomType::mesh:
    case GeomType::hfield: o = g.mesh; break;
    case GeomType::other: break;
  }
  return o;
}

// Radius of a sphere about the geom origin that holds the tessellation (the shadow-map fit).
double bound_radius(const Verts& v) {
  double r2 = 0;
  for (std::size_t i = 0; i + 2 < v.size(); i += 6) r2 = std::max(r2, double(v[i]) * v[i] + double(v[i + 1]) * v[i + 1] + double(v[i + 2]) * v[i + 2]);
  return std::sqrt(r2);
}

// CAD-style edges: segments where two triangles meet at more than ~37 degrees, plus open boundaries. Each segment becomes 6
// vertices (a screen-space quad, see vsEdge in geom.slang) carrying both end points: position = this end, "normal" = the other.
Verts crease_edges(const Verts& tris) {
  struct Acc { int count = 0; bool crease = false; double n[3]; float a[3], b[3]; };
  std::map<std::array<long long, 6>, Acc> acc;
  auto q = [](float x) { return static_cast<long long>(std::llround(double(x) * 1e6)); };
  for (std::size_t t = 0; t + 17 < tris.size(); t += 18) {
    const float* v[3] = {&tris[t], &tris[t + 6], &tris[t + 12]};
    double e1[3], e2[3], n[3];
    for (int k = 0; k < 3; ++k) {
      e1[k] = v[1][k] - v[0][k];
      e2[k] = v[2][k] - v[0][k];
    }
    n[0] = e1[1] * e2[2] - e1[2] * e2[1];
    n[1] = e1[2] * e2[0] - e1[0] * e2[2];
    n[2] = e1[0] * e2[1] - e1[1] * e2[0];
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len < 1e-12) continue;  // degenerate (sphere poles)
    for (double& c : n) c /= len;
    for (int e = 0; e < 3; ++e) {
      const float *p = v[e], *s = v[(e + 1) % 3];
      std::array<long long, 6> ka{q(p[0]), q(p[1]), q(p[2]), q(s[0]), q(s[1]), q(s[2])}, kb{ka[3], ka[4], ka[5], ka[0], ka[1], ka[2]};
      if (ka == kb) continue;
      Acc& r = acc[std::min(ka, kb)];
      if (r.count++ == 0) {
        std::copy(n, n + 3, r.n);
        std::copy(p, p + 3, r.a);
        std::copy(s, s + 3, r.b);
      } else if (r.n[0] * n[0] + r.n[1] * n[1] + r.n[2] * n[2] < 0.8) {
        r.crease = true;
      }
    }
  }
  Verts o;
  for (const auto& [k, r] : acc) {
    if (r.count != 1 && !r.crease) continue;
    for (int c = 0; c < 6; ++c) {
      o.insert(o.end(), r.a, r.a + 3);
      o.insert(o.end(), r.b, r.b + 3);
    }
  }
  return o;
}

std::uint64_t hash_visuals(std::span<const GeomVisual> v) {  // FNV-1a 64 over everything tessellation depends on
  std::uint64_t h = 14695981039346656037ull;
  auto mix = [&](const void* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(p)[i]) * 1099511628211ull;
  };
  for (const GeomVisual& g : v) {
    mix(&g.type, sizeof g.type);
    mix(g.size, sizeof g.size);
    mix(g.rgba, sizeof g.rgba);
    mix(&g.group, sizeof g.group);
    mix(&g.specular, 4 * sizeof(float));  // specular, shininess, metallic, roughness
    if (g.water()) {  // the sea mesh changes every frame and is rewritten in place (Renderer::render): only its size is part of the scene
      const std::size_t n = g.mesh.size();
      mix(&n, sizeof n);
    } else if (g.type == GeomType::hfield) {  // static and large (100k+ triangles): the size and every 61st float stand for it
      const std::size_t n = g.mesh.size();
      mix(&n, sizeof n);
      for (std::size_t i = 0; i < n; i += 61) mix(&g.mesh[i], sizeof(float));
    } else {
      mix(g.mesh.data(), g.mesh.size() * sizeof(float));
    }
  }
  return h;
}

// ---- camera-relative maths (double) ------------------------------------------------------------------------------------
struct V3 { double x, y, z; };
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double norm(V3 a) { return std::sqrt(dot(a, a)); }
V3 unit(V3 a) {
  const double l = norm(a);
  return {a.x / l, a.y / l, a.z / l};
}

// Cable -> triangles (camera-relative floats). The polyline sags (gravity, a parabola per segment) by its slack:
// length L, chord D -> L ~ D + 8 h^2 / (3 D), so h = sqrt(3 D slack / 8) (capped at 30 % of D).
void build_tube(Verts& o, const TubeVisual& t, V3 cam, double px_world, double min_px) {
  std::vector<V3> p;
  for (std::size_t i = 0; i + 2 < t.pts.size(); i += 3) p.push_back({t.pts[i], t.pts[i + 1], t.pts[i + 2]});
  if (p.size() < 2) return;
  double path = 0;
  for (std::size_t i = 0; i + 1 < p.size(); ++i) path += norm(p[i + 1] - p[i]);
  if (path < 1e-6) return;
  const double slack = std::max(0.0, t.length - path);
  constexpr int kSeg = 12, kSides = 8;
  std::vector<V3> q;
  for (std::size_t i = 0; i + 1 < p.size(); ++i) {
    const V3 d = p[i + 1] - p[i];
    const double D = norm(d);
    if (D < 1e-9) continue;
    const V3 s = d * (1 / D);
    const V3 down{0, 0, -1};
    V3 perp = down - s * dot(down, s);
    if (norm(perp) < 0.2) perp = cross(s, std::abs(s.x) < 0.9 ? V3{1, 0, 0} : V3{0, 1, 0});  // near-vertical: any side, deterministic
    perp = unit(perp);
    const double h = std::min(std::sqrt(3 * D * slack * (D / path) / 8), 0.3 * D);
    for (int j = 0; j < kSeg; ++j) {
      const double u = double(j) / kSeg;
      q.push_back(p[i] + d * u + perp * (h * 4 * u * (1 - u)));
    }
  }
  q.push_back(p.back());
  if (q.size() < 2) return;
  const std::size_t n = q.size();
  std::vector<V3> tan(n), nrm(n), rel(n);
  for (std::size_t i = 0; i < n; ++i) {
    tan[i] = unit(q[std::min(i + 1, n - 1)] - q[i > 0 ? i - 1 : 0]);
    rel[i] = q[i] - cam;
  }
  V3 nn = cross(tan[0], std::abs(tan[0].z) < 0.9 ? V3{0, 0, 1} : V3{1, 0, 0});
  for (std::size_t i = 0; i < n; ++i) {  // parallel transport
    nn = nn - tan[i] * dot(nn, tan[i]);
    nn = unit(nn);
    nrm[i] = nn;
  }
  auto ring = [&](std::size_t i, int k) {
    const double r = std::max(t.radius, std::max(0.7, 0.5 * min_px) * px_world * norm(rel[i]));  // never thinner than ~1.4 px (min_px when set)
    const double a = 2 * kPi * k / kSides;
    const V3 b = cross(tan[i], nrm[i]), d = nrm[i] * std::cos(a) + b * std::sin(a);
    return Pn{{rel[i].x + r * d.x, rel[i].y + r * d.y, rel[i].z + r * d.z}, {d.x, d.y, d.z}};
  };
  for (std::size_t i = 0; i + 1 < n; ++i)
    for (int k = 0; k < kSides; ++k) quad(o, ring(i, k), ring(i, k + 1), ring(i + 1, k + 1), ring(i + 1, k));
  for (int end = 0; end < 2; ++end) {  // flat caps
    const std::size_t i = end ? n - 1 : 0;
    const double sgn = end ? 1 : -1;
    const Pn c{{rel[i].x, rel[i].y, rel[i].z}, {sgn * tan[i].x, sgn * tan[i].y, sgn * tan[i].z}};
    for (int k = 0; k < kSides; ++k) {
      Pn a = ring(i, k), b = ring(i, k + 1);
      a.n[0] = c.n[0], a.n[1] = c.n[1], a.n[2] = c.n[2];
      b.n[0] = c.n[0], b.n[1] = c.n[1], b.n[2] = c.n[2];
      for (const Pn* v : end ? std::array<const Pn*, 3>{&c, &a, &b} : std::array<const Pn*, 3>{&c, &b, &a})
        vert(o, v->p[0], v->p[1], v->p[2], v->n[0], v->n[1], v->n[2]);
    }
  }
}

// ---- style ---------------------------------------------------------------------------------------------------------------
std::mutex g_style_mu;
RenderStyle g_default_style;

struct Palette {
  float top[3], mid[3], hor[3], ground[3], ground_col[3], sun[3];  // sRGB
  float sun_dir[3], glare, haze, ambient;
};
Palette palette(RenderStyle::Sky s) {
  if (s == RenderStyle::Sky::studio)
    return {{0.60f, 0.64f, 0.70f}, {0.74f, 0.77f, 0.81f}, {0.86f, 0.88f, 0.90f}, {0.55f, 0.56f, 0.58f}, {0.44f, 0.45f, 0.47f},
            {1.0f, 1.0f, 1.0f},    {0.45f, 0.30f, 0.84f},   0.0f,                0.010f,                0.78f};
  return {{0.16f, 0.38f, 0.78f}, {0.52f, 0.70f, 0.92f}, {1.00f, 0.86f, 0.66f}, {0.36f, 0.34f, 0.31f}, {0.27f, 0.27f, 0.28f},
          {1.0f, 0.95f, 0.85f},  {0.45f, 0.30f, 0.84f}, 1.0f,                  0.0045f,               0.60f};
}
void lin3(float* o, const float* srgb, float k = 1) {
  for (int i = 0; i < 3; ++i) o[i] = static_cast<float>(std::pow(srgb[i], 2.2) * k);
  o[3] = 1;
}

struct View {
  V3 pos, r, u, f;  // camera position and basis rows: right, up, forward
  double sx, sy;    // clip scale
};
View make_view(const Camera& c, int w, int h) {
  View v;
  v.pos = {c.pos[0], c.pos[1], c.pos[2]};
  v.f = unit(V3{c.lookat[0], c.lookat[1], c.lookat[2]} - v.pos);
  const V3 up0 = std::abs(v.f.z) > 0.9999 ? V3{0, 1, 0} : V3{0, 0, 1};  // straight up/down: image top = +y
  v.r = unit(cross(v.f, up0));
  v.u = cross(v.r, v.f);
  const double t = 1 / std::tan(c.fovy * kPi / 360);
  v.sx = t * h / w;
  v.sy = -t;  // Vulkan clip y points down
  return v;
}

// std140 mirrors of Scene / Draw in geom.slang (float4 members only).
struct GpuScene {
  float view_r[4], view_u[4], view_f[4], proj[4], lcol[4][4], sun_dir[4], sun_col[4], sky_top[4], sky_mid[4], sky_hor[4], ground[4],
      ground_col[4], fog[4], vp[4], camf[4], wx[4], rainv[4], rcam[4];
};
struct GpuDraw {
  float rot[12], trel[4], color[4], mat[4], flags[4];
};
struct Push { std::uint32_t draw, pad[3]; };
}  // namespace

RenderStyle parse_render_style(const std::string& words, RenderStyle s) {
  std::istringstream in(words);
  for (std::string w; std::getline(in, w, ',');) {
    if (w.empty()) continue;
    if (w == "studio") s.sky = RenderStyle::Sky::studio;
    else if (w == "outdoor") s.sky = RenderStyle::Sky::outdoor;
    else if (w == "edges") s.edges = true;
    else if (w == "noedges") s.edges = false;
    else if (w == "shadows") s.shadows = true;
    else if (w == "noshadows") s.shadows = false;
    else if (w == "wires") s.tubes = true;
    else if (w == "nowires") s.tubes = false;
    else if (w == "boldwires") s.wire_px = 4.f;
    else if (w == "plainwires") s.wire_px = 0.f;
    else if (w == "rain=auto") s.rain = -1;
    else if (w == "overcast=auto") s.overcast = -1;
    else if (w.rfind("rain=", 0) == 0 || w.rfind("overcast=", 0) == 0) {
      const std::size_t eq = w.find('=');
      std::size_t used = 0;
      float v = 0;
      try {
        v = std::stof(w.substr(eq + 1), &used);
      } catch (const std::exception&) {
        used = 0;
      }
      if (used == 0 || used != w.size() - eq - 1 || !std::isfinite(v) || v < 0 || (w[0] == 'o' && v > 1) || v > 500)
        throw std::invalid_argument("render style: bad value in '" + w + "' (rain=<mm/h 0..500>, overcast=<0..1>)");
      (w[0] == 'r' ? s.rain : s.overcast) = v;
    } else throw std::invalid_argument("render style: unknown word '" + w + "' (studio outdoor edges noedges shadows noshadows wires nowires boldwires plainwires rain=<mm/h> overcast=<0..1>)");
  }
  return s;
}
void set_default_render_style(const RenderStyle& s) {
  std::lock_guard<std::mutex> l(g_style_mu);
  g_default_style = s;
}
RenderStyle default_render_style() {
  RenderStyle s;
  {
    std::lock_guard<std::mutex> l(g_style_mu);
    s = g_default_style;
  }
  if (const char* e = std::getenv("DVSIM_RENDER_STYLE"))
    try {
      s = parse_render_style(e, s);
    } catch (const std::invalid_argument&) {  // a typo in the environment must not kill the renderer
    }
  return s;
}

struct Renderer::Impl {
  struct Draw {
    std::uint32_t first = 0, count = 0, efirst = 0, ecount = 0;
    float rgba[4] = {}, mat[4] = {};
    double radius = 0;
    bool visible = false, plane = false, water = false;  // water: the sea mesh (rewritten every frame by update_water)
    bool terrain = false;  // heightfield: slope / height shaded, no crease edges, outside the shadow-map fit
    float zlo = 0, zhi = 1;  // terrain: local z range of the mesh
  };
  struct Target {
    int w = 0, h = 0;
    VkImage ms_c = nullptr, ms_d = nullptr, color = nullptr, lin = nullptr, depth1 = nullptr;
    VmaAllocation ms_c_a = nullptr, ms_d_a = nullptr, color_a = nullptr, lin_a = nullptr, depth1_a = nullptr;
    VkImageView ms_c_v = nullptr, ms_d_v = nullptr, color_v = nullptr, lin_v = nullptr, depth1_v = nullptr;
    VkBuffer rb_color = nullptr, rb_lin = nullptr;  // host-visible readback
    VmaAllocation rb_color_a = nullptr, rb_lin_a = nullptr;
    void *color_ptr = nullptr, *lin_ptr = nullptr;
  };
  struct HostBuf {
    VkBuffer buf = nullptr;
    VmaAllocation alloc = nullptr;
    void* ptr = nullptr;
    std::size_t cap = 0;  // bytes
  };

  VkInstance inst = nullptr;
  VkPhysicalDevice pd = nullptr;
  VkDevice dev = nullptr;
  VkQueue queue = nullptr;
  std::uint32_t qfam = 0;
  VmaAllocator vma = nullptr;
  VkCommandPool pool = nullptr;
  VkCommandBuffer cb = nullptr;
  VkFence fence = nullptr;
  VkShaderModule shader = nullptr;
  VkDescriptorSetLayout dsl = nullptr;
  VkDescriptorPool dpool = nullptr;
  VkDescriptorSet dset = nullptr;
  VkSampler shadow_smp = nullptr;
  VkPipelineLayout layout = nullptr;
  VkPipeline pipe_scene = nullptr, pipe_sky = nullptr, pipe_edge = nullptr, pipe_shadow = nullptr, pipe_lin = nullptr, pipe_rain = nullptr;
  VkImage sh_img = nullptr;
  VmaAllocation sh_a = nullptr;
  VkImageView sh_view = nullptr;
  HostBuf vbuf, tbuf, ubo, ssbo;  // geometry (static), cables (per frame), Scene, Draw[]
  std::string name;
  std::vector<Draw> draws;
  Target tgt;
  VkFormat color_fmt = kColorFmt;
  VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_1_BIT;

  RenderStyle style;
  bool has_style = false;
  Weather weather;               // from the solver / viewer frame; the style's rain= / overcast= words override it
  std::uint32_t n_streaks = 0;   // rain streak instances of the frame being recorded (0 = none drawn)
  std::vector<TubeVisual> tubes_in;
  struct TubeDraw { std::uint32_t first, count, draw; };
  std::vector<TubeDraw> tube_draws;
  std::uint32_t ndraw_total = 0;

  // Window mode (WindowRenderer): surface + swapchain, one frame in flight.
  const WindowRenderer::Surface* win = nullptr;
  VkSurfaceKHR surface = nullptr;
  VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  VkSwapchainKHR swapchain = nullptr;
  VkExtent2D sc_extent{};
  int req_w = 0, req_h = 0;  // framebuffer size the swapchain was last built for
  bool sc_stale = false;
  std::vector<VkImage> sc_images;
  std::vector<VkImageView> sc_views;
  std::vector<VkSemaphore> sc_done;  // render finished, per swapchain image
  VkSemaphore acquire_sem = nullptr;

  bool sc = false;  // swapchain mode (a window that presents); a window with Surface::offscreen renders into tgt and reads back instead

  explicit Impl(const WindowRenderer::Surface* window = nullptr) : win(window), sc(window && !window->offscreen) {
    if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("no Vulkan loader (libvulkan) found on this machine");
    try {
      init();
    } catch (...) {
      destroy();
      throw;
    }
  }
  ~Impl() { destroy(); }

  void init() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "dvsim";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (sc) {
      ici.enabledExtensionCount = static_cast<std::uint32_t>(win->instance_extensions.size());
      ici.ppEnabledExtensionNames = win->instance_extensions.data();
    }
    const VkResult ir = vkCreateInstance(&ici, nullptr, &inst);
    if (ir == VK_ERROR_INCOMPATIBLE_DRIVER)
      throw std::runtime_error("no Vulkan device: no usable Vulkan driver (ICD) found (check VK_DRIVER_FILES)");
    if (ir == VK_ERROR_EXTENSION_NOT_PRESENT)
      throw std::runtime_error("no Vulkan device: the driver lacks the window-system extensions needed to show a window");
    check(ir, "vkCreateInstance");
    volkLoadInstance(inst);
    if (sc) surface = win->create(inst);

    std::uint32_t n = 0;
    check(vkEnumeratePhysicalDevices(inst, &n, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> pds(n);
    check(vkEnumeratePhysicalDevices(inst, &n, pds.data()), "vkEnumeratePhysicalDevices");
    int best = -1;
    const auto lower = [](std::string s) {
      for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      return s;
    };
    const char* env_dev = std::getenv("DVSIM_VK_DEVICE");
    const std::string want = env_dev ? lower(env_dev) : "";
    const char* cpu_env = std::getenv("DVSIM_RENDER_ALLOW_CPU");  // software rasterisers (llvmpipe / lavapipe) only when asked for
    const bool allow_cpu = cpu_env && *cpu_env && std::string(cpu_env) != "0";
    bool refused_cpu = false;
    for (VkPhysicalDevice cand : pds) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(cand, &props);
      if (props.apiVersion < VK_API_VERSION_1_3) continue;
      VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
      VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
      vkGetPhysicalDeviceFeatures2(cand, &f2);
      if (!f13.dynamicRendering || !f13.synchronization2) continue;
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(cand, kDepthFmt, &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) continue;
      std::uint32_t nq = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(cand, &nq, nullptr);
      std::vector<VkQueueFamilyProperties> qs(nq);
      vkGetPhysicalDeviceQueueFamilyProperties(cand, &nq, qs.data());
      if (sc && !can_present(cand)) continue;
      std::uint32_t q = nq;  // first graphics queue (that can present, in swapchain mode)
      for (std::uint32_t i = 0; i < nq && q == nq; ++i) {
        VkBool32 present = !sc;
        if (sc && vkGetPhysicalDeviceSurfaceSupportKHR(cand, i, surface, &present) != VK_SUCCESS) present = VK_FALSE;
        if ((qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) q = i;
      }
      if (q == nq) continue;
      // Swapchain mode takes the first device that can present. Headless / off-screen: discrete > integrated > virtual > cpu
      // (the first of equals), and DVSIM_VK_DEVICE=<substring of deviceName, any case> restricts the choice.
      int rank = 0;
      if (!sc) {
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !allow_cpu) {
          refused_cpu = true;
          continue;
        }
        if (!want.empty() && lower(props.deviceName).find(want) == std::string::npos) continue;
        rank = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 4
               : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
               : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2
               : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? 0
                                                                           : 1;
      }
      if (pd && rank <= best) continue;
      best = rank;
      pd = cand;
      qfam = q;
      name = props.deviceName;
      const VkSampleCountFlags ok = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
      msaa = (ok & VK_SAMPLE_COUNT_4_BIT) ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;
      if (sc) break;
    }
    if (!pd)
      throw std::runtime_error("no Vulkan device: none of the " + std::to_string(n) + " physical device(s) supports Vulkan 1.3 with dynamic rendering" +
                               (sc ? " and presenting to this window (an X server without DRI3 cannot present from a hardware GPU)" : "") +
                               (!sc && !want.empty() ? " and matches DVSIM_VK_DEVICE=" + want : "") +
                               (refused_cpu ? " (a CPU renderer such as llvmpipe was refused: set DVSIM_RENDER_ALLOW_CPU=1 to use it)" : ""));
    if (!win) {  // headless Renderer: say once which device renders (the viewer prints its own line)
      static std::once_flag logged;
      std::call_once(logged, [&] { std::fprintf(stderr, "dvsim: rendering on %s\n", name.c_str()); });
    }

    const float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    const char* swapchain_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    if (sc) {
      dci.enabledExtensionCount = 1;
      dci.ppEnabledExtensionNames = &swapchain_ext;
    }
    check(vkCreateDevice(pd, &dci, nullptr, &dev), "vkCreateDevice");
    volkLoadDevice(dev);
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    VmaVulkanFunctions fns{};
    fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = pd;
    aci.device = dev;
    aci.instance = inst;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.pVulkanFunctions = &fns;
    check(vmaCreateAllocator(&aci, &vma), "vmaCreateAllocator");

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = qfam;
    check(vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(dev, &cai, &cb), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(dev, &fci, nullptr, &fence), "vkCreateFence");
    if (sc) {
      VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      check(vkCreateSemaphore(dev, &semi, nullptr, &acquire_sem), "vkCreateSemaphore");
      std::uint32_t nf = 0;
      check(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &nf, nullptr), "vkGetPhysicalDeviceSurfaceFormatsKHR");
      std::vector<VkSurfaceFormatKHR> fmts(nf);
      check(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &nf, fmts.data()), "vkGetPhysicalDeviceSurfaceFormatsKHR");
      auto find = [&](VkFormat f) { return std::find_if(fmts.begin(), fmts.end(), [f](const auto& s) { return s.format == f; }); };
      // UNORM first: ImGui does no sRGB conversion (the scene shader encodes sRGB itself). can_present() guaranteed a format.
      auto it = find(VK_FORMAT_B8G8R8A8_UNORM);
      if (it == fmts.end()) it = find(VK_FORMAT_R8G8B8A8_UNORM);
      const VkSurfaceFormatKHR pick = it != fmts.end() ? *it : fmts.at(0);
      color_fmt = pick.format;
      color_space = pick.colorSpace;
    }
    make_resources();
    make_pipelines();
  }

  // Window mode: the device must offer the swapchain extension and at least one surface format.
  bool can_present(VkPhysicalDevice cand) const {
    std::uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(cand, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(cand, nullptr, &n, exts.data());
    bool has = false;
    for (const VkExtensionProperties& e : exts) has = has || std::string(e.extensionName) == VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    std::uint32_t nf = 0;
    return has && vkGetPhysicalDeviceSurfaceFormatsKHR(cand, surface, &nf, nullptr) == VK_SUCCESS && nf > 0;
  }

  // ---- buffers / images -------------------------------------------------------------------------------------------------
  void host_buffer(HostBuf& b, std::size_t bytes, VkBufferUsageFlags usage) {
    if (b.buf) vmaDestroyBuffer(vma, b.buf, b.alloc);
    b = {};
    b.cap = std::max<std::size_t>(bytes, 64);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = b.cap;
    bi.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info;
    check(vmaCreateBuffer(vma, &bi, &ai, &b.buf, &b.alloc, &info), "vmaCreateBuffer(host)");
    b.ptr = info.pMappedData;
  }
  void fill(HostBuf& b, const void* data, std::size_t bytes) {
    if (bytes) std::memcpy(b.ptr, data, bytes);
    check(vmaFlushAllocation(vma, b.alloc, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");
  }

  void image(int w, int h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkSampleCountFlagBits samples, VkImage& img,
             VmaAllocation& alloc, VkImageView& view) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = fmt;
    ii.extent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = samples;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    check(vmaCreateImage(vma, &ii, &ai, &img, &alloc, nullptr), "vmaCreateImage");
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    check(vkCreateImageView(dev, &vi, nullptr, &view), "vkCreateImageView");
  }

  void readback_buffer(std::size_t bytes, VkBuffer& buf, VmaAllocation& alloc, void*& ptr) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info;
    check(vmaCreateBuffer(vma, &bi, &ai, &buf, &alloc, &info), "vmaCreateBuffer(readback)");
    ptr = info.pMappedData;
  }

  // Descriptor set (Scene UBO, Draw[] SSBO, shadow map + comparison sampler), shadow map, constant buffers.
  void make_resources() {
    VkDescriptorSetLayoutBinding b[4] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 4;
    li.pBindings = b;
    check(vkCreateDescriptorSetLayout(dev, &li, nullptr, &dsl), "vkCreateDescriptorSetLayout");
    const VkDescriptorPoolSize sizes[4] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                           {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1;
    pi.poolSizeCount = 4;
    pi.pPoolSizes = sizes;
    check(vkCreateDescriptorPool(dev, &pi, nullptr, &dpool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dsl;
    check(vkAllocateDescriptorSets(dev, &ai, &dset), "vkAllocateDescriptorSets");

    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pd, kDepthFmt, &fp);
    const bool lin = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;  // linear + compare = hardware 2x2 PCF
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;  // outside the map: lit
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    check(vkCreateSampler(dev, &si, nullptr, &shadow_smp), "vkCreateSampler");
    image(kShadowSize, kShadowSize, kDepthFmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
          VK_IMAGE_ASPECT_DEPTH_BIT, VK_SAMPLE_COUNT_1_BIT, sh_img, sh_a, sh_view);

    host_buffer(ubo, sizeof(GpuScene), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    host_buffer(ssbo, 256 * sizeof(GpuDraw), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    host_buffer(tbuf, 1 << 16, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    host_buffer(vbuf, 1 << 10, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    write_descriptors();
  }

  void write_descriptors() {
    const VkDescriptorBufferInfo ub{ubo.buf, 0, sizeof(GpuScene)}, sb{ssbo.buf, 0, VK_WHOLE_SIZE};
    const VkDescriptorImageInfo im{nullptr, sh_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}, sm{shadow_smp, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet w[4]{};
    for (int i = 0; i < 4; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = dset;
      w[i].dstBinding = static_cast<std::uint32_t>(i);
      w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &ub;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &sb;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[2].pImageInfo = &im;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[3].pImageInfo = &sm;
    vkUpdateDescriptorSets(dev, 4, w, 0, nullptr);
  }

  VkPipeline gfx(const char* vs, const char* fs, bool vertex_in, VkSampleCountFlagBits samples, std::vector<VkFormat> colors, bool dtest,
                 bool dwrite, VkCompareOp op, bool bias, bool blend = false) {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, shader, vs, nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, shader, fs, nullptr}};
    VkVertexInputBindingDescription bind{0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[2] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                                  {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 3 * sizeof(float)}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (vertex_in) {
      vi.vertexBindingDescriptionCount = 1;
      vi.pVertexBindingDescriptions = &bind;
      vi.vertexAttributeDescriptionCount = 2;
      vi.pVertexAttributeDescriptions = attrs;
    }
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;  // planes and meshes are seen from both sides
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1;
    if (bias) {
      rs.depthBiasEnable = VK_TRUE;
      rs.depthBiasConstantFactor = 2.0f;
      rs.depthBiasSlopeFactor = 2.5f;
    }
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = samples;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = dtest;
    ds.depthWriteEnable = dwrite;
    ds.depthCompareOp = op;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (blend) {  // straight alpha over the colour already drawn
      cba.blendEnable = VK_TRUE;
      cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      cba.colorBlendOp = VK_BLEND_OP_ADD;
      cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      cba.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    std::vector<VkPipelineColorBlendAttachmentState> cbas(colors.size(), cba);
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = static_cast<std::uint32_t>(cbas.size());
    cbs.pAttachments = cbas.data();
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dsi.dynamicStateCount = 2;
    dsi.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
    rci.pColorAttachmentFormats = colors.data();
    rci.depthAttachmentFormat = kDepthFmt;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
    gpi.stageCount = fs ? 2 : 1;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cbs;
    gpi.pDynamicState = &dsi;
    gpi.layout = layout;
    VkPipeline p = nullptr;
    check(vkCreateGraphicsPipelines(dev, nullptr, 1, &gpi, nullptr, &p), "vkCreateGraphicsPipelines");
    return p;
  }

  void make_pipelines() {
    std::vector<std::uint32_t> code(sizeof kSpv_geom / 4);
    std::memcpy(code.data(), kSpv_geom, code.size() * 4);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size() * 4;
    smi.pCode = code.data();
    check(vkCreateShaderModule(dev, &smi, nullptr, &shader), "vkCreateShaderModule");

    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    check(vkCreatePipelineLayout(dev, &pli, nullptr, &layout), "vkCreatePipelineLayout");

    pipe_scene = gfx("vsMain", "fsMain", true, msaa, {color_fmt}, true, true, VK_COMPARE_OP_LESS, false);
    pipe_sky = gfx("vsSky", "fsSky", false, msaa, {color_fmt}, false, false, VK_COMPARE_OP_ALWAYS, false);
    pipe_edge = gfx("vsEdge", "fsEdge", true, msaa, {color_fmt}, true, false, VK_COMPARE_OP_LESS_OR_EQUAL, false);
    pipe_shadow = gfx("vsShadow", nullptr, true, VK_SAMPLE_COUNT_1_BIT, {}, true, true, VK_COMPARE_OP_LESS, true);
    pipe_lin = gfx("vsMain", "fsLin", true, VK_SAMPLE_COUNT_1_BIT, {kLinFmt}, true, true, VK_COMPARE_OP_LESS, false);
    pipe_rain = gfx("vsRain", "fsRain", false, msaa, {color_fmt}, true, false, VK_COMPARE_OP_LESS, false, true);
  }

  void destroy_target() {
    if (!tgt.w) return;
    for (VkImageView v : {tgt.ms_c_v, tgt.ms_d_v, tgt.color_v, tgt.lin_v, tgt.depth1_v})
      if (v) vkDestroyImageView(dev, v, nullptr);
    auto kill = [&](VkImage i, VmaAllocation a) {
      if (i) vmaDestroyImage(vma, i, a);
    };
    kill(tgt.ms_c, tgt.ms_c_a);
    kill(tgt.ms_d, tgt.ms_d_a);
    kill(tgt.color, tgt.color_a);
    kill(tgt.lin, tgt.lin_a);
    kill(tgt.depth1, tgt.depth1_a);
    if (tgt.rb_color) vmaDestroyBuffer(vma, tgt.rb_color, tgt.rb_color_a);
    if (tgt.rb_lin) vmaDestroyBuffer(vma, tgt.rb_lin, tgt.rb_lin_a);
    tgt = {};
  }

  // readback = false (window mode): the colour target is the swapchain image; only the multisampled colour + depth are
  // allocated. Headless: also the resolved colour, the exact linear-depth pair and the readback buffers.
  void ensure_target(int w, int h, bool readback = true) {
    if (tgt.w == w && tgt.h == h) return;
    destroy_target();
    Target t;
    try {
      constexpr auto kCA = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, kSrc = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      if (msaa != VK_SAMPLE_COUNT_1_BIT)
        image(w, h, color_fmt, kCA, VK_IMAGE_ASPECT_COLOR_BIT, msaa, t.ms_c, t.ms_c_a, t.ms_c_v);
      image(w, h, kDepthFmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, msaa, t.ms_d, t.ms_d_a, t.ms_d_v);
      if (readback) {
        image(w, h, color_fmt, kCA | kSrc, VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, t.color, t.color_a, t.color_v);
        image(w, h, kLinFmt, kCA | kSrc, VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, t.lin, t.lin_a, t.lin_v);
        image(w, h, kDepthFmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_SAMPLE_COUNT_1_BIT, t.depth1,
              t.depth1_a, t.depth1_v);
        readback_buffer(static_cast<std::size_t>(w) * h * 4, t.rb_color, t.rb_color_a, t.color_ptr);
        readback_buffer(static_cast<std::size_t>(w) * h * 4, t.rb_lin, t.rb_lin_a, t.lin_ptr);
      }
    } catch (...) {
      tgt = t;  // free whatever was created
      tgt.w = w;
      destroy_target();
      throw;
    }
    t.w = w;
    t.h = h;
    tgt = t;
  }

  void transition(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 src_stage,
                  VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
  }

  // ---- per-frame data -----------------------------------------------------------------------------------------------------
  RenderStyle effective_style() const { return has_style ? style : default_render_style(); }
  RenderStyle cur;  // the style of the frame being recorded

  // Fills the Scene UBO, the Draw[] SSBO and the cable vertices for one frame (all camera-relative).
  void prepare(std::span<const GeomPose> poses, const Camera& cam, int w, int h) {
    cur = effective_style();
    const View v = make_view(cam, w, h);
    const double zfar = std::any_of(draws.begin(), draws.end(), [](const Draw& d) { return d.terrain; }) ? kFarTerrain : kFar;
    const double A = zfar / (zfar - kNear), B = -zfar * kNear / (zfar - kNear);
    Palette pal = palette(cur.sky);
    // Weather (outdoor sky only; rain 0 and overcast 0 leave every value above untouched, bit for bit). Overcast follows the rain rate
    // unless the style says otherwise: o = 1 - exp(-R / 5) (0.18 at 1 mm/h, 0.99 at 25). The look is art-directed, not radiometric:
    // grey sky stops, sun x (1 - 0.85 o) with no glare, ambient x (1 - 0.2 o), wet (darker) ground, haze density + 0.0016 sqrt(R) per m.
    const double rain = cur.sky == RenderStyle::Sky::studio ? 0.0 : (cur.rain >= 0 ? cur.rain : weather.rain);
    const float oc = cur.sky == RenderStyle::Sky::studio ? 0.0f
                     : cur.overcast >= 0                 ? cur.overcast
                                                         : (rain > 0 ? static_cast<float>(1.0 - std::exp(-rain / 5.0)) : 0.0f);
    if (oc > 0 || rain > 0) {
      constexpr float grey_top[3] = {0.40f, 0.43f, 0.47f}, grey_mid[3] = {0.52f, 0.55f, 0.58f}, grey_hor[3] = {0.64f, 0.66f, 0.68f};
      for (int i = 0; i < 3; ++i) {
        pal.top[i] += (grey_top[i] - pal.top[i]) * oc;
        pal.mid[i] += (grey_mid[i] - pal.mid[i]) * oc;
        pal.hor[i] += (grey_hor[i] - pal.hor[i]) * oc;
        pal.sun[i] *= 1.0f - 0.85f * oc;
        pal.ground[i] *= 1.0f - 0.3f * oc;
        pal.ground_col[i] *= 1.0f - 0.3f * oc;
      }
      pal.glare *= 1.0f - oc;
      pal.ambient *= 1.0f - 0.2f * oc;
      pal.haze += 0.0016f * static_cast<float>(std::sqrt(rain));
    }

    // sun shadow map: an orthographic box fitted to the bounding sphere of everything that casts (planes do not)
    V3 lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
    bool any = false;
    for (std::size_t i = 0; i < draws.size(); ++i) {
      if (!draws[i].visible || draws[i].plane || draws[i].water || draws[i].terrain) continue;
      const V3 p{poses[i].pos[0], poses[i].pos[1], poses[i].pos[2]};
      const double r = draws[i].radius;
      lo = {std::min(lo.x, p.x - r), std::min(lo.y, p.y - r), std::min(lo.z, p.z - r)};
      hi = {std::max(hi.x, p.x + r), std::max(hi.y, p.y + r), std::max(hi.z, p.z + r)};
      any = true;
    }
    const V3 centre = any ? (lo + hi) * 0.5 : v.pos;
    const double R = any ? std::max(0.05, norm(hi - lo) * 0.5) * 1.02 : 1.0;
    const V3 L = unit(V3{pal.sun_dir[0], pal.sun_dir[1], pal.sun_dir[2]});
    const V3 Lx = unit(cross(std::abs(L.z) < 0.99 ? V3{0, 0, 1} : V3{1, 0, 0}, L)), Ly = cross(L, Lx);
    const V3 rows[3] = {Lx * (1 / R), Ly * (1 / R), L * (-1 / (2 * R))};  // light clip = rows . (p - centre) (+ 0.5 in z)
    const V3 off = v.pos - centre;

    GpuScene s{};
    auto f3 = [](float* o, const V3& a, float w4 = 0) {
      o[0] = static_cast<float>(a.x), o[1] = static_cast<float>(a.y), o[2] = static_cast<float>(a.z), o[3] = w4;
    };
    f3(s.view_r, v.r), f3(s.view_u, v.u), f3(s.view_f, v.f);
    s.proj[0] = static_cast<float>(v.sx), s.proj[1] = static_cast<float>(v.sy), s.proj[2] = static_cast<float>(A), s.proj[3] = static_cast<float>(B);
    for (int j = 0; j < 3; ++j) {
      const double col[3] = {j == 0 ? rows[0].x : j == 1 ? rows[0].y : rows[0].z, j == 0 ? rows[1].x : j == 1 ? rows[1].y : rows[1].z,
                             j == 0 ? rows[2].x : j == 1 ? rows[2].y : rows[2].z};
      for (int k = 0; k < 3; ++k) s.lcol[j][k] = static_cast<float>(col[k]);
    }
    s.lcol[3][0] = static_cast<float>(dot(rows[0], off));
    s.lcol[3][1] = static_cast<float>(dot(rows[1], off));
    s.lcol[3][2] = static_cast<float>(dot(rows[2], off) + 0.5);
    f3(s.sun_dir, L, pal.glare);
    lin3(s.sun_col, pal.sun, cur.sky == RenderStyle::Sky::studio ? 0.9f : 1.0f);
    lin3(s.sky_top, pal.top), lin3(s.sky_mid, pal.mid), lin3(s.sky_hor, pal.hor), lin3(s.ground, pal.ground), lin3(s.ground_col, pal.ground_col);
    s.fog[0] = pal.haze;
    s.fog[1] = cur.sky == RenderStyle::Sky::studio ? 1.0f : 0.0f;
    s.fog[2] = kEdgePx;
    s.fog[3] = pal.ambient;
    s.vp[0] = static_cast<float>(w), s.vp[1] = static_cast<float>(h), s.vp[2] = cur.shadows ? 1.0f : 0.0f;
    s.vp[3] = static_cast<float>(2 * R / kShadowSize);
    s.camf[0] = static_cast<float>(std::fmod(cam.pos[0], 10.0)), s.camf[1] = static_cast<float>(std::fmod(cam.pos[1], 10.0));
    s.camf[2] = static_cast<float>(cam.pos[2]), s.camf[3] = 1.0f / kShadowSize;
    // Rain streaks: Marshall-Palmer fall speed (docs/models.md "Rain": v = R / LWC, LWC = 0.072 R^0.88 g/m^3), drops drift with the wind.
    n_streaks = 0;
    s.wx[0] = oc;
    if (rain > 0) {
      constexpr double kBox = 48.0;  // edge of the world-fixed periodic box the streaks live in, m
      const double vf = rain / 3600.0 / (0.072 * std::pow(rain, 0.88) * 1e-3);
      n_streaks = static_cast<std::uint32_t>(14000.0 * (1.0 - std::exp(-rain / 30.0)));
      s.wx[1] = static_cast<float>(0.2 + 0.3 * std::min(1.0, rain / 25.0)), s.wx[2] = static_cast<float>(0.06 * std::hypot(vf, std::hypot(weather.wind[0], weather.wind[1])));
      s.wx[3] = static_cast<float>(std::fmod(weather.time, 1000.0));
      s.rainv[0] = static_cast<float>(weather.wind[0]), s.rainv[1] = static_cast<float>(weather.wind[1]);
      s.rainv[2] = static_cast<float>(weather.wind[2] - vf), s.rainv[3] = static_cast<float>(kBox);
      for (int i = 0; i < 3; ++i) s.rcam[i] = static_cast<float>(std::fmod(cam.pos[i], kBox));
    }
    fill(ubo, &s, sizeof s);

    // draws: geoms, then cables
    std::vector<GpuDraw> gd(draws.size());
    for (std::size_t i = 0; i < draws.size(); ++i) {
      const GeomPose& g = poses[i];
      GpuDraw& d = gd[i];
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) d.rot[4 * j + k] = static_cast<float>(g.rot[3 * k + j]);
      d.trel[0] = static_cast<float>(g.pos[0] - cam.pos[0]);  // the double-precision camera-relative step
      d.trel[1] = static_cast<float>(g.pos[1] - cam.pos[1]);
      d.trel[2] = static_cast<float>(g.pos[2] - cam.pos[2]);
      std::memcpy(d.color, draws[i].rgba, sizeof d.color);
      std::memcpy(d.mat, draws[i].mat, sizeof d.mat);
      d.flags[0] = draws[i].plane ? 1.0f : 0.0f;
      d.flags[1] = draws[i].water ? 1.0f : draws[i].terrain ? 2.0f : 0.0f;
      d.flags[2] = draws[i].zlo, d.flags[3] = draws[i].zhi;
    }
    Verts tv;
    tube_draws.clear();
    if (cur.tubes) {
      const double px = 2 * std::tan(cam.fovy * kPi / 360) / h;
      for (const TubeVisual& t : tubes_in) {
        const std::size_t before = tv.size();
        build_tube(tv, t, v.pos, px, cur.wire_px);
        if (tv.size() == before) continue;
        GpuDraw d{};
        d.rot[0] = d.rot[5] = d.rot[10] = 1;
        std::memcpy(d.color, t.rgba, sizeof d.color);
        d.mat[0] = 0.6f, d.mat[1] = 0.25f, d.mat[2] = -1, d.mat[3] = -1;
        tube_draws.push_back({static_cast<std::uint32_t>(before / 6), static_cast<std::uint32_t>((tv.size() - before) / 6),
                              static_cast<std::uint32_t>(gd.size())});
        gd.push_back(d);
      }
    }
    if (gd.size() * sizeof(GpuDraw) > ssbo.cap) {
      host_buffer(ssbo, gd.size() * sizeof(GpuDraw) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      write_descriptors();
    }
    fill(ssbo, gd.data(), gd.size() * sizeof(GpuDraw));
    if (tv.size() * sizeof(float) > tbuf.cap) host_buffer(tbuf, tv.size() * sizeof(float) * 2, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    fill(tbuf, tv.data(), tv.size() * sizeof(float));
  }

  void set_vp(int w, int h) {
    const VkViewport vp{0, 0, static_cast<float>(w), static_cast<float>(h), 0, 1};
    const VkRect2D sc{{0, 0}, {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
  }
  void push(std::uint32_t i) {
    const Push p{i, {0, 0, 0}};
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof p, &p);
  }
  void bind(VkPipeline p) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &dset, 0, nullptr);
  }
  // Geometry (+ cables) with the currently bound pipeline.
  void draw_geoms(bool planes, bool with_tubes) {
    const VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vbuf.buf, &off);
    for (std::size_t i = 0; i < draws.size(); ++i) {
      if (!draws[i].visible || (!planes && (draws[i].plane || draws[i].water))) continue;  // planes and water cast no shadow
      push(static_cast<std::uint32_t>(i));
      vkCmdDraw(cb, draws[i].count, 1, draws[i].first, 0);
    }
    if (!with_tubes || tube_draws.empty()) return;
    vkCmdBindVertexBuffers(cb, 0, 1, &tbuf.buf, &off);
    for (const TubeDraw& t : tube_draws) {
      push(t.draw);
      vkCmdDraw(cb, t.count, 1, t.first, 0);
    }
  }

  constexpr static auto kColorOut = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  constexpr static auto kDepthTest = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
  constexpr static auto kColorW = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  constexpr static auto kDepthRW = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  void record_shadow() {
    transition(sh_img, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
               VK_PIPELINE_STAGE_2_NONE, 0, kDepthTest, kDepthRW);
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageView = sh_view;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    da.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {kShadowSize, kShadowSize}};
    ri.layerCount = 1;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cb, &ri);
    if (cur.shadows) {
      bind(pipe_shadow);
      set_vp(kShadowSize, kShadowSize);
      draw_geoms(/*planes=*/false, /*with_tubes=*/false);
    }
    vkCmdEndRendering(cb);
    transition(sh_img, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }

  // The lit scene into `resolve_v` (via the multisampled targets when msaa > 1). Colour/depth images are already in their
  // attachment layouts.
  void record_main(VkImageView resolve_v, int w, int h) {
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;  // the sky covers every pixel
    if (msaa != VK_SAMPLE_COUNT_1_BIT) {
      ca.imageView = tgt.ms_c_v;
      ca.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      ca.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
      ca.resolveImageView = resolve_v;
      ca.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else {
      ca.imageView = resolve_v;
      ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageView = tgt.ms_d_v;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    da.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cb, &ri);
    set_vp(w, h);
    bind(pipe_sky);
    vkCmdDraw(cb, 3, 1, 0, 0);
    bind(pipe_scene);
    draw_geoms(true, true);
    if (cur.edges) {
      bind(pipe_edge);
      const VkDeviceSize off = 0;
      vkCmdBindVertexBuffers(cb, 0, 1, &vbuf.buf, &off);
      for (std::size_t i = 0; i < draws.size(); ++i) {
        if (!draws[i].visible || draws[i].ecount == 0) continue;
        push(static_cast<std::uint32_t>(i));
        vkCmdDraw(cb, draws[i].ecount, 1, draws[i].efirst, 0);
      }
    }
    if (n_streaks > 0) {  // rain: instanced streak quads over the lit scene (depth-tested, no depth write)
      bind(pipe_rain);
      vkCmdDraw(cb, 6, n_streaks, 0, 0);
    }
    vkCmdEndRendering(cb);
  }

  void record_lin(int w, int h) {
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = tgt.lin_v;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color.float32[0] = std::numeric_limits<float>::infinity();  // no hit
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageView = tgt.depth1_v;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    da.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cb, &ri);
    bind(pipe_lin);
    set_vp(w, h);
    draw_geoms(true, true);
    vkCmdEndRendering(cb);
  }

  Image draw(std::span<const GeomPose> poses, const Camera& cam, int w, int h) {
    if (w < 1 || h < 1 || w > 16384 || h > 16384) throw std::invalid_argument("render size out of range");
    if (poses.size() != draws.size())
      throw std::invalid_argument("draw: " + std::to_string(poses.size()) + " poses for a scene of " + std::to_string(draws.size()) + " geoms");
    const double d2 = dot(V3{cam.lookat[0], cam.lookat[1], cam.lookat[2]} - V3{cam.pos[0], cam.pos[1], cam.pos[2]},
                          V3{cam.lookat[0], cam.lookat[1], cam.lookat[2]} - V3{cam.pos[0], cam.pos[1], cam.pos[2]});
    if (!(d2 > 0) || !(cam.fovy > 0 && cam.fovy < 180)) throw std::invalid_argument("bad camera");
    ensure_target(w, h);
    prepare(poses, cam, w, h);

    check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    for (VkImage img : {tgt.ms_c, tgt.color, tgt.lin})
      if (img)
        transition(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_2_NONE, 0, kColorOut, kColorW);
    for (VkImage img : {tgt.ms_d, tgt.depth1})
      transition(img, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_NONE, 0, kDepthTest, kDepthRW);

    record_shadow();
    record_main(tgt.color_v, w, h);
    record_lin(w, h);

    for (VkImage img : {tgt.color, tgt.lin})
      transition(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 kColorOut, kColorW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
    vkCmdCopyImageToBuffer(cb, tgt.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tgt.rb_color, 1, &region);
    vkCmdCopyImageToBuffer(cb, tgt.lin, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tgt.rb_lin, 1, &region);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
    check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    check(vkResetFences(dev, 1, &fence), "vkResetFences");
    check(vkQueueSubmit2(queue, 1, &si, fence), "vkQueueSubmit2");
    check(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    check(vmaInvalidateAllocation(vma, tgt.rb_color_a, 0, VK_WHOLE_SIZE), "vmaInvalidateAllocation");
    check(vmaInvalidateAllocation(vma, tgt.rb_lin_a, 0, VK_WHOLE_SIZE), "vmaInvalidateAllocation");

    Image out;
    out.width = w;
    out.height = h;
    const std::size_t n = static_cast<std::size_t>(w) * h;
    out.rgba.resize(n * 4);
    out.depth.resize(n);
    std::memcpy(out.rgba.data(), tgt.color_ptr, n * 4);
    std::memcpy(out.depth.data(), tgt.lin_ptr, n * 4);
    return out;
  }

  // Overwrites the vertices of draw i (the water mesh) in place; a mesh of another size than the uploaded one is ignored.
  void update_water(std::size_t i, const std::vector<float>& mesh) {
    if (i >= draws.size() || !draws[i].water || mesh.empty() || draws[i].count * 6ull != mesh.size()) return;
    const VkDeviceSize off = static_cast<VkDeviceSize>(draws[i].first) * 6 * sizeof(float), bytes = mesh.size() * sizeof(float);
    std::memcpy(static_cast<char*>(vbuf.ptr) + off, mesh.data(), bytes);
    check(vmaFlushAllocation(vma, vbuf.alloc, off, bytes), "vmaFlushAllocation");
  }

  void set_scene(std::span<const GeomVisual> visuals) {
    Verts all;
    std::vector<Draw> next(visuals.size());
    for (std::size_t i = 0; i < visuals.size(); ++i) {
      const GeomVisual& g = visuals[i];
      Draw& d = next[i];
      Verts v = tessellate(g);
      d.first = static_cast<std::uint32_t>(all.size() / 6);
      d.count = static_cast<std::uint32_t>(v.size() / 6);
      std::memcpy(d.rgba, g.rgba, sizeof d.rgba);
      d.mat[0] = g.specular, d.mat[1] = g.shininess, d.mat[2] = g.metallic, d.mat[3] = g.roughness;
      d.visible = d.count > 0 && g.group != 3 && g.rgba[3] > 0;  // group 3: hidden collision geoms (4 = design-builder vehicle geoms: shown)
      d.plane = g.type == GeomType::plane;
      d.water = g.water();
      d.terrain = g.type == GeomType::hfield;
      if (d.terrain && !v.empty()) {
        d.zlo = d.zhi = v[2];
        for (std::size_t k = 2; k < v.size(); k += 6) d.zlo = std::min(d.zlo, v[k]), d.zhi = std::max(d.zhi, v[k]);
        d.mat[0] = 0.08f, d.mat[1] = 0.1f, d.mat[2] = d.mat[3] = -1;  // a matte surface, whatever the geom's default material
      }
      d.radius = bound_radius(v);
      all.insert(all.end(), v.begin(), v.end());
      if (d.visible && !d.plane && !d.water && !d.terrain) {
        const Verts e = crease_edges(v);
        d.efirst = static_cast<std::uint32_t>(all.size() / 6);
        d.ecount = static_cast<std::uint32_t>(e.size() / 6);
        all.insert(all.end(), e.begin(), e.end());
      }
    }
    host_buffer(vbuf, all.size() * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    fill(vbuf, all.data(), all.size() * sizeof(float));
    draws = std::move(next);
  }

  // ---- window mode ----------------------------------------------------------------------------------------------------
  void destroy_swapchain_views() {
    for (VkImageView v : sc_views) vkDestroyImageView(dev, v, nullptr);
    for (VkSemaphore s : sc_done) vkDestroySemaphore(dev, s, nullptr);
    sc_views.clear();
    sc_done.clear();
    sc_images.clear();
  }

  // (Re)builds the swapchain for a (w, h) framebuffer. false = the surface has zero extent (minimised).
  bool make_swapchain(int w, int h) {
    vkDeviceWaitIdle(dev);
    VkSurfaceCapabilitiesKHR caps;
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFFu)  // the surface leaves it to us
      ext = {std::clamp<std::uint32_t>(static_cast<std::uint32_t>(w), caps.minImageExtent.width, caps.maxImageExtent.width),
             std::clamp<std::uint32_t>(static_cast<std::uint32_t>(h), caps.minImageExtent.height, caps.maxImageExtent.height)};
    if (ext.width == 0 || ext.height == 0) return false;
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = std::max(2u, caps.minImageCount);
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = color_fmt;
    ci.imageColorSpace = color_space;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                                                                           : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // the one mode every driver has
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain;
    VkSwapchainKHR next = nullptr;
    check(vkCreateSwapchainKHR(dev, &ci, nullptr, &next), "vkCreateSwapchainKHR");
    destroy_swapchain_views();
    if (swapchain) vkDestroySwapchainKHR(dev, swapchain, nullptr);
    swapchain = next;
    sc_extent = ext;
    std::uint32_t n = 0;
    check(vkGetSwapchainImagesKHR(dev, swapchain, &n, nullptr), "vkGetSwapchainImagesKHR");
    sc_images.resize(n);
    check(vkGetSwapchainImagesKHR(dev, swapchain, &n, sc_images.data()), "vkGetSwapchainImagesKHR");
    for (VkImage img : sc_images) {
      VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vi.image = img;
      vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vi.format = color_fmt;
      vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      sc_views.emplace_back();
      check(vkCreateImageView(dev, &vi, nullptr, &sc_views.back()), "vkCreateImageView");
      VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      sc_done.emplace_back();
      check(vkCreateSemaphore(dev, &si, nullptr, &sc_done.back()), "vkCreateSemaphore");
    }
    return true;
  }

  // Off-screen viewer frame: scene (MSAA, resolved into tgt.color) + overlay on top, copied to `rgba` (w*h*4, RGBA8).
  // Same passes as window_frame; no swapchain, no semaphores.
  void offscreen_frame(std::span<const GeomPose> poses, const Camera& cam, int w, int h,
                       const std::function<void(VkCommandBuffer)>& overlay, std::vector<std::uint8_t>& rgba) {
    ensure_target(w, h);
    prepare(poses, cam, w, h);
    check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    for (VkImage img : {tgt.color, tgt.ms_c})
      if (img)
        transition(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_2_NONE, 0, kColorOut, kColorW);
    transition(tgt.ms_d, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
               VK_PIPELINE_STAGE_2_NONE, 0, kDepthTest, kDepthRW);
    record_shadow();
    record_main(tgt.color_v, w, h);
    if (overlay) {
      VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      mb.srcStageMask = kColorOut;
      mb.srcAccessMask = kColorW;
      mb.dstStageMask = kColorOut;
      mb.dstAccessMask = kColorW | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
      VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      di.memoryBarrierCount = 1;
      di.pMemoryBarriers = &mb;
      vkCmdPipelineBarrier2(cb, &di);
      VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      ca.imageView = tgt.color_v;
      ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
      ri.renderArea = {{0, 0}, {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)}};
      ri.layerCount = 1;
      ri.colorAttachmentCount = 1;
      ri.pColorAttachments = &ca;
      vkCmdBeginRendering(cb, &ri);
      overlay(cb);
      vkCmdEndRendering(cb);
    }
    transition(tgt.color, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               kColorOut, kColorW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
    vkCmdCopyImageToBuffer(cb, tgt.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tgt.rb_color, 1, &region);
    VkMemoryBarrier2 hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    hb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    hb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    hb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo hd{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    hd.memoryBarrierCount = 1;
    hd.pMemoryBarriers = &hb;
    vkCmdPipelineBarrier2(cb, &hd);
    check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    check(vkResetFences(dev, 1, &fence), "vkResetFences");
    check(vkQueueSubmit2(queue, 1, &si, fence), "vkQueueSubmit2");
    check(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    check(vmaInvalidateAllocation(vma, tgt.rb_color_a, 0, VK_WHOLE_SIZE), "vmaInvalidateAllocation");
    rgba.resize(static_cast<std::size_t>(w) * h * 4);
    std::memcpy(rgba.data(), tgt.color_ptr, rgba.size());
  }

  bool window_frame(std::span<const GeomPose> poses, const Camera& cam, int w, int h,
                    const std::function<void(VkCommandBuffer)>& overlay, std::vector<std::uint8_t>* rgba) {
    if (poses.size() != draws.size())
      throw std::invalid_argument("frame: " + std::to_string(poses.size()) + " poses for a scene of " + std::to_string(draws.size()) + " geoms");
    const V3 look{cam.lookat[0] - cam.pos[0], cam.lookat[1] - cam.pos[1], cam.lookat[2] - cam.pos[2]};
    if (!(dot(look, look) > 0) || !(cam.fovy > 0 && cam.fovy < 180)) throw std::invalid_argument("bad camera");
    if (w < 1 || h < 1) return false;
    if (!sc) {
      if (!rgba) throw std::invalid_argument("frame: an off-screen WindowRenderer needs an output buffer");
      if (w > 16384 || h > 16384) throw std::invalid_argument("render size out of range");
      offscreen_frame(poses, cam, w, h, overlay, *rgba);
      return true;
    }
    if (!swapchain || sc_stale || w != req_w || h != req_h) {
      sc_stale = false;
      req_w = w;
      req_h = h;
      if (!make_swapchain(w, h)) return false;
    }
    std::uint32_t idx = 0;
    const VkResult ar = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, acquire_sem, nullptr, &idx);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
      sc_stale = true;
      return false;
    }
    if (ar != VK_SUBOPTIMAL_KHR) check(ar, "vkAcquireNextImageKHR");
    const int fw = static_cast<int>(sc_extent.width), fh = static_cast<int>(sc_extent.height);
    ensure_target(fw, fh, /*readback=*/false);
    prepare(poses, cam, fw, fh);

    check(vkResetFences(dev, 1, &fence), "vkResetFences");
    check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    for (VkImage img : {sc_images[idx], tgt.ms_c})
      if (img)
        transition(img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_2_NONE, 0, kColorOut, kColorW);
    transition(tgt.ms_d, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
               VK_PIPELINE_STAGE_2_NONE, 0, kDepthTest, kDepthRW);

    record_shadow();
    record_main(sc_views[idx], fw, fh);

    if (overlay) {  // second pass on the same colour image: scene writes -> overlay blend reads/writes
      VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      mb.srcStageMask = kColorOut;
      mb.srcAccessMask = kColorW;
      mb.dstStageMask = kColorOut;
      mb.dstAccessMask = kColorW | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
      VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      di.memoryBarrierCount = 1;
      di.pMemoryBarriers = &mb;
      vkCmdPipelineBarrier2(cb, &di);
      VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      ca.imageView = sc_views[idx];
      ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
      ri.renderArea = {{0, 0}, sc_extent};
      ri.layerCount = 1;
      ri.colorAttachmentCount = 1;
      ri.pColorAttachments = &ca;
      vkCmdBeginRendering(cb, &ri);
      overlay(cb);
      vkCmdEndRendering(cb);
    }
    transition(sc_images[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
               kColorOut, kColorW, VK_PIPELINE_STAGE_2_NONE, 0);
    check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = acquire_sem;
    wait.stageMask = kColorOut;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = sc_done[idx];
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos = &wait;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &signal;
    check(vkQueueSubmit2(queue, 1, &si, fence), "vkQueueSubmit2");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &sc_done[idx];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &idx;
    const VkResult pr = vkQueuePresentKHR(queue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) sc_stale = true;  // rebuilt on the next frame
    else check(pr, "vkQueuePresentKHR");
    check(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");  // one frame in flight
    return true;
  }

  void destroy() noexcept {
    if (dev) vkDeviceWaitIdle(dev);
    if (dev) {
      destroy_swapchain_views();
      if (swapchain) vkDestroySwapchainKHR(dev, swapchain, nullptr);
      if (acquire_sem) vkDestroySemaphore(dev, acquire_sem, nullptr);
    }
    swapchain = nullptr;
    acquire_sem = nullptr;
    if (dev && sh_view) vkDestroyImageView(dev, sh_view, nullptr);
    if (vma) {
      destroy_target();
      for (HostBuf* b : {&vbuf, &tbuf, &ubo, &ssbo})
        if (b->buf) vmaDestroyBuffer(vma, b->buf, b->alloc);
      if (sh_img) vmaDestroyImage(vma, sh_img, sh_a);
    }
    if (dev) {
      for (VkPipeline p : {pipe_scene, pipe_sky, pipe_edge, pipe_shadow, pipe_lin, pipe_rain})
        if (p) vkDestroyPipeline(dev, p, nullptr);
      if (layout) vkDestroyPipelineLayout(dev, layout, nullptr);
      if (shadow_smp) vkDestroySampler(dev, shadow_smp, nullptr);
      if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
      if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
      if (shader) vkDestroyShaderModule(dev, shader, nullptr);
      if (fence) vkDestroyFence(dev, fence, nullptr);
      if (pool) vkDestroyCommandPool(dev, pool, nullptr);
    }
    if (vma) vmaDestroyAllocator(vma);
    if (dev) vkDestroyDevice(dev, nullptr);
    if (surface) vkDestroySurfaceKHR(inst, surface, nullptr);  // the window must still exist
    surface = nullptr;
    if (inst) vkDestroyInstance(inst, nullptr);
    vbuf = tbuf = ubo = ssbo = {};
    sh_img = nullptr;
    sh_view = nullptr;
    pipe_scene = pipe_sky = pipe_edge = pipe_shadow = pipe_lin = pipe_rain = nullptr;
    layout = nullptr;
    shadow_smp = nullptr;
    dpool = nullptr;
    dsl = nullptr;
    shader = nullptr;
    fence = nullptr;
    pool = nullptr;
    vma = nullptr;
    dev = nullptr;
    inst = nullptr;
  }
};

Renderer::Renderer() : p_(std::make_unique<Impl>()) {}
Renderer::~Renderer() = default;
const std::string& Renderer::device_name() const { return p_->name; }

void Renderer::set_scene(std::span<const GeomVisual> visuals) {
  p_->set_scene(visuals);
  scene_hash_ = hash_visuals(visuals);
}
void Renderer::set_tubes(std::span<const TubeVisual> tubes) { p_->tubes_in.assign(tubes.begin(), tubes.end()); }
void Renderer::set_style(const RenderStyle& s) {
  p_->style = s;
  p_->has_style = true;
}
void Renderer::set_weather(const Weather& w) { p_->weather = w; }

Image Renderer::draw(std::span<const GeomPose> poses, const Camera& cam, int width, int height) {
  return p_->draw(poses, cam, width, height);
}

// ponytail: visuals() is rebuilt and hashed per render (mesh models: a copy of every mesh); cache a model generation
// counter in the Solver seam if that ever shows up in a profile.
Image Renderer::render(const Solver& s, const Camera& cam, int width, int height) {
  const std::vector<GeomVisual> v = s.visuals();
  if (hash_visuals(v) != scene_hash_ || p_->draws.size() != v.size()) set_scene(v);
  for (std::size_t i = 0; i < v.size(); ++i)
    if (v[i].water()) p_->update_water(i, v[i].mesh);
  p_->tubes_in = s.tubes();
  p_->weather = s.weather();
  const std::vector<GeomPose> poses = s.poses();
  return p_->draw(poses, cam, width, height);
}

WindowRenderer::WindowRenderer(const Surface& surface) : p_(std::make_unique<Renderer::Impl>(&surface)) {}
WindowRenderer::~WindowRenderer() = default;
const std::string& WindowRenderer::device_name() const { return p_->name; }
WindowRenderer::Handles WindowRenderer::handles() const {
  return {p_->inst, p_->pd, p_->dev, p_->queue, p_->qfam, p_->color_fmt};
}
void WindowRenderer::set_scene(std::span<const GeomVisual> visuals) { p_->set_scene(visuals); }
void WindowRenderer::set_tubes(std::span<const TubeVisual> tubes) { p_->tubes_in.assign(tubes.begin(), tubes.end()); }
void WindowRenderer::set_weather(const Weather& w) { p_->weather = w; }
void WindowRenderer::set_style(const RenderStyle& s) {
  p_->style = s;
  p_->has_style = true;
}
bool WindowRenderer::frame(std::span<const GeomPose> poses, const Camera& cam, int width, int height,
                           const std::function<void(VkCommandBuffer)>& overlay, std::vector<std::uint8_t>* rgba) {
  return p_->window_frame(poses, cam, width, height, overlay, rgba);
}

RenderHook make_render_hook() {
  return [](const Solver& s, const Camera& cam, int w, int h) {
    static std::mutex mu;
    static Renderer* r = nullptr;  // leaked on purpose: tearing a GPU driver down during static destruction is fragile
    std::lock_guard<std::mutex> lock(mu);
    if (!r) r = new Renderer();  // ttsr-ignore: cpp/no-raw-new  (deliberately never freed, see above)
    return r->render(s, cam, w, h);
  };
}
