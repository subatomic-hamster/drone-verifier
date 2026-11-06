// Terrain: MuJoCo heightfield contact / rays / visuals, and the terrain-aware wind (src/mujoco/devices/windfield.cpp).
// Expectations are closed forms: a flat-pad contact height, the 1 - cos gust integral and peak time, the potential-flow signs of a
// Gaussian ridge. Models are generated per test into a temp file.
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "mujoco/mujoco_solver.h"

namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<double> getv(const Solver& s, const std::string& prim, const std::string& attr) {
  std::vector<double> o;
  const Result r = s.get(prim, attr, "", o);
  EXPECT_EQ(r.code, Result::ok) << prim << "." << attr << ": " << r.message;
  return o;
}
double get(const Solver& s, const std::string& prim, const std::string& attr) {
  const auto o = getv(s, prim, attr);
  return o.empty() ? NAN : o[0];
}
std::unique_ptr<Solver> load_text(const std::string& xml) {
  static int n = 0;
  const std::string path = (std::filesystem::temp_directory_path() / ("dvsim_tr_" + std::to_string(::getpid()) + "_" + std::to_string(n++) + ".xml")).string();
  std::ofstream(path) << xml;
  auto s = load_mujoco(path);
  std::remove(path.c_str());
  return s;
}

// Terrain profile h(x) (metres, 0..H, min and max both present so any normalisation is the identity) over a grid of nc x nr samples,
// half extents (rx, ry); uniform in y.
struct Terr {
  int nc = 81, nr = 11;
  double rx = 200, ry = 100, H = 40, sigma = 40, base = 5, z = 0;
  bool ridge = true;  // false: flat (one 1 mm bump in a far corner, MuJoCo wants a non-empty range)
  double profile(double x) const { return ridge ? H * std::exp(-x * x / (2 * sigma * sigma)) : 0.0; }
  std::string xml_asset() const {
    std::ostringstream e;
    e.precision(12);
    const double elev = ridge ? H : 0.001;
    for (int r = 0; r < nr; ++r)
      for (int c = 0; c < nc; ++c) {
        const double x = -rx + 2 * rx * c / (nc - 1);
        double v = ridge ? profile(x) / elev : (r == 0 && c == 0 ? 1.0 : 0.0);
        if (ridge && c == 0) v = 0;  // exact minimum
        e << v << " ";
      }
    std::ostringstream a;
    a << "<hfield name=\"t\" nrow=\"" << nr << "\" ncol=\"" << nc << "\" size=\"" << rx << " " << ry << " " << elev << " " << base
      << "\" elevation=\"" << e.str() << "\"/>";
    return a.str();
  }
};

// A tiny free body "b" (no contacts) at (x, 0, z) in a gravity-free world with the terrain, an environment (wind (wx, 0, 0), ref_body b)
// and an optional wind_field JSON body.
std::string wind_xml(const Terr& t, double x, double z, double wx, const std::string& wf, double w20 = 0) {
  std::ostringstream o;
  o << "<mujoco model=\"wf\"><option timestep=\"0.001\" gravity=\"0 0 0\"/><asset>" << t.xml_asset() << "</asset><worldbody>"
    << "<geom name=\"terr\" type=\"hfield\" hfield=\"t\" pos=\"0 0 " << t.z << "\"/>"
    << "<body name=\"b\" pos=\"" << x << " 0 " << z << "\"><freejoint/><geom type=\"sphere\" size=\"0.05\" mass=\"1\" contype=\"0\" conaffinity=\"0\"/></body>"
    << "</worldbody><custom><text name=\"dvsim.devices\" data='{\"environments\": [{\"name\": \"air\", \"wind\": [" << wx
    << ", 0, 0], \"w20\": " << w20 << ", \"seed\": 3, \"ref_body\": \"b\"}], \"body_drags\": [{\"name\": \"drag\", \"body\": \"b\", \"cd\": 1, \"area\": 0.5}]" << wf
    << "}'/></custom></mujoco>";
  return o.str();
}
std::string wf_json(const std::string& body) {
  return ", \"wind_fields\": [{\"name\": \"wf\"" + (body.empty() ? "" : ", " + body) + "}]";
}
std::vector<double> wind_ref(Solver& s) {
  s.step(1);
  return getv(s, "/devices/wf", "wind_ref");
}
}  // namespace

// A box dropped on a flat pad of the heightfield (the ridge top is flat enough at sigma = 1e6): it rests on the surface, and a ray straight
// down sees the same distance (mj_ray hits hfields, which is what rangefinders and ground-effect rays use).
TEST(terrain, box_rests_on_the_hfield_and_rays_see_it) {
  Terr t;
  t.sigma = 1e6;  // profile is H everywhere: a flat table at z = 40 (the left column is 0 only at the edge, far from the box)
  t.z = 3.0;
  std::ostringstream o;
  o << "<mujoco model=\"drop\"><option timestep=\"0.002\"/><asset>" << t.xml_asset() << "</asset><worldbody>"
    << "<geom name=\"terr\" type=\"hfield\" hfield=\"t\" pos=\"0 0 " << t.z << "\" contype=\"1\" conaffinity=\"1\"/>"
    << "<body name=\"b\" pos=\"0 0 " << (t.z + t.H + 0.5) << "\"><freejoint/><geom type=\"box\" size=\"0.1 0.1 0.05\" mass=\"1\" contype=\"1\" conaffinity=\"1\"/></body>"
    << "</worldbody></mujoco>";
  auto s = load_text(o.str());
  s->step(1500);
  const auto p = getv(*s, "/world/b", "pos");
  ASSERT_EQ(p.size(), 3u);
  EXPECT_NEAR(p[2], t.z + t.H + 0.05, 0.003) << "box centre rests half a height above the surface";
  EXPECT_NEAR(p[0], 0.0, 0.01);

  // ray from 7 m above straight down, all groups
  const double from[3] = {50.0, 20.0, t.z + t.H + 7.0}, dir[3] = {0, 0, -1};
  const auto dist = s->raycast(from, dir);
  ASSERT_EQ(dist.size(), 1u);
  EXPECT_NEAR(dist[0], 7.0, 0.001);
}

// The visual of an hfield geom is the triangulated surface: 2 (n-1)^2 triangles, z within [0, elevation].
TEST(terrain, visual_is_a_triangle_mesh) {
  Terr t;
  auto s = load_text(wind_xml(t, 0, 10, 0, ""));
  const std::vector<GeomVisual> v = s->visuals();
  const GeomVisual* h = nullptr;
  for (const GeomVisual& g : v)
    if (g.type == GeomType::hfield) h = &g;
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(h->mesh.size(), static_cast<std::size_t>((t.nc - 1) * (t.nr - 1) * 2 * 3 * 6));
  float zmax = 0;
  for (std::size_t i = 2; i < h->mesh.size(); i += 6) zmax = std::max(zmax, h->mesh[i]);
  EXPECT_NEAR(zmax, t.H, 1e-3);
}

// Flat terrain, no profile, no ridge: the field is the old uniform wind (+ turbulence), exactly, at any point.
TEST(terrain, flat_field_equals_uniform_wind) {
  Terr t;
  t.ridge = false;
  for (double x : {-120.0, 0.0, 77.0}) {
    auto a = load_text(wind_xml(t, x, 12.5, 7.0, "", 4.0));  // uniform: no wind_field device
    auto b = load_text(wind_xml(t, x, 12.5, 7.0, wf_json("\"hfield_geom\": \"terr\""), 4.0));
    auto c = load_text(wind_xml(t, x, 12.5, 7.0, wf_json("\"ground_z\": 0"), 4.0));  // flat ground without an hfield geom
    for (int k = 0; k < 400; ++k) {
      a->step(1); b->step(1); c->step(1);
      const auto fa = getv(*a, "/devices/drag", "force"), fb = getv(*b, "/devices/drag", "force"), fc = getv(*c, "/devices/drag", "force");
      for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(fa[i], fb[i]) << "x=" << x << " k=" << k;
        ASSERT_EQ(fa[i], fc[i]);
      }
    }
  }
}

// Log law: W (ln(h/z0) / ln(zref/z0)) at h = 2, 10, 100 m above flat ground.
TEST(terrain, log_boundary_layer) {
  Terr t;
  t.ridge = false;
  for (double h : {2.0, 10.0, 100.0}) {
    auto s = load_text(wind_xml(t, 0, h, 8.0, wf_json("\"ground_z\": 0, \"z0\": 0.03, \"z_ref\": 10")));
    const auto w = wind_ref(*s);
    EXPECT_NEAR(w[0], 8.0 * std::log(h / 0.03) / std::log(10 / 0.03), 1e-9) << h;
    EXPECT_NEAR(get(*s, "/devices/wf", "height_agl"), h, 0.01);
  }
}

// Gaussian ridge h = H exp(-x^2 / 2 s^2), wind +x: updraft on the windward slope, downdraft on the lee slope, speed-up on the crest.
// Linear theory with the smoothing length L << s: w = U exp(-h/L) dh/dx, u' = U exp(-h/L) (h_L - h_B) / L.
TEST(terrain, ridge_updraft_windward_downdraft_lee) {
  Terr t;
  const std::string wf = wf_json("\"hfield_geom\": \"terr\", \"ridge_gain\": 1, \"ridge_len\": 20, \"lee_down\": 0, \"lee_turb\": 1");
  const double U = 10.0, hag = 5.0;
  auto at = [&](double x) {
    auto s = load_text(wind_xml(t, x, t.profile(x) + hag, U, wf));
    return wind_ref(*s);
  };
  const double xw = -40.0, xl = 40.0;
  const auto ww = at(xw), wl = at(xl), wc = at(0.0), wfar = at(-190.0);
  const double slope = -xw / (t.sigma * t.sigma) * t.profile(xw);  // dh/dx at xw (positive)
  EXPECT_GT(ww[2], 0.5) << "windward lift";
  EXPECT_LT(wl[2], -0.5) << "lee sink";
  EXPECT_NEAR(ww[2], -wl[2], 0.15 * ww[2]) << "antisymmetric";
  // order of magnitude of the closed form (the 20 m smoothing flattens the slope a little: within 40 %)
  EXPECT_NEAR(ww[2], U * std::exp(-hag / 20) * slope, 0.4 * U * slope);
  EXPECT_GT(wc[0], U * 1.05) << "crest speed-up";
  EXPECT_NEAR(wc[2], 0.0, 0.05) << "no lift at the crest";
  EXPECT_NEAR(wfar[2], 0.0, 0.05) << "no lift far from the ridge";
  EXPECT_NEAR(wfar[0], U, 0.15 * U);
  // the ridge flow decays with height: 100 m above the crest it is a few per cent
  auto hi = load_text(wind_xml(t, 0, t.profile(0) + 100, U, wf));
  EXPECT_NEAR(wind_ref(*hi)[0], U, 0.02 * U);
  // lee separation: a downdraft and more turbulence just behind the steep face
  const std::string wl2 = wf_json("\"hfield_geom\": \"terr\", \"ridge_gain\": 1, \"ridge_len\": 20, \"lee_down\": 0.3");
  auto s1 = load_text(wind_xml(t, xl, t.profile(xl) + hag, U, wl2));
  EXPECT_LT(wind_ref(*s1)[2], wl[2] - 0.5);
  EXPECT_GT(get(*s1, "/devices/wf", "lee"), 0.3);
  auto s2 = load_text(wind_xml(t, xw, t.profile(xw) + hag, U, wl2));
  s2->step(1);
  EXPECT_LT(get(*s2, "/devices/wf", "lee"), 0.01) << "no separation on the windward side";
}

// A uniform 1 - cos gust: integral over its life = A H / c with c = max(|wind|, 2) (here 2 m/s), peak A at s = H.
TEST(terrain, gust_shape) {
  Terr t;
  t.ridge = false;
  auto s = load_text(wind_xml(t, 0, 10, 0, wf_json("\"ground_z\": 0")));
  const double A = 6.0, H = 30.0, c = 2.0, T = 2 * H / c;
  const std::vector<double> g{A, H, 1, 0, 0, 1.0, 0};  // from t = 1 s, +x, uniform in space
  ASSERT_EQ(s->set("/devices/wf", "gust", "", g).code, Result::ok);
  double integral = 0, peak = 0, tpeak = 0;
  for (int k = 0; k < static_cast<int>((T + 3) / 0.001); ++k) {
    s->step(1);
    const double w = getv(*s, "/devices/wf", "wind_ref")[0], tt = get(*s, "/devices/wf", "time");
    integral += w * 0.001;
    if (w > peak) peak = w, tpeak = tt;
  }
  EXPECT_NEAR(integral, A * H / c, 0.003 * A * H / c);
  EXPECT_NEAR(peak, A, 1e-3);
  EXPECT_NEAR(tpeak, 1.0 + H / c, 0.01);
  EXPECT_EQ(get(*s, "/devices/wf", "n_gusts"), 1.0);
  // validation: wrong arity and out-of-range values are rejected
  EXPECT_EQ(s->set("/devices/wf", "gust", "", std::vector<double>{1, 2, 3}).code, Result::invalid_argument);
  EXPECT_EQ(s->set("/devices/wf", "gust_clear", "", std::vector<double>{1}).code, Result::ok);
  s->step(1);
  EXPECT_NEAR(getv(*s, "/devices/wf", "wind_ref")[0], 0.0, 1e-12);
}

// A gust front moving at c = 20 m/s along +x passes a body at x = 100 m, 0 s delay: the peak arrives at t = (100 + c H / c ...) i.e.
// s = c t - x = H, so t = (x + H) / c.
TEST(terrain, gust_front_travels) {
  Terr t;
  t.ridge = false;
  const double A = 5.0, H = 40.0, c = 20.0, x = 100.0;
  auto s = load_text(wind_xml(t, x, 10, 0, wf_json("\"ground_z\": 0")));
  ASSERT_EQ(s->set("/devices/wf", "gust", "", std::vector<double>{A, H, 0, 0, 1, 0, c}).code, Result::ok);  // an updraft gust
  double peak = 0, tpeak = 0, w_before = 1;
  for (int k = 0; k < 12000; ++k) {
    s->step(1);
    const double w = getv(*s, "/devices/wf", "wind_ref")[2], tt = get(*s, "/devices/wf", "time");
    if (tt < 0.9 * x / c) w_before = std::min(w_before, std::abs(w));
    if (w > peak) peak = w, tpeak = tt;
  }
  EXPECT_NEAR(peak, A, 1e-3);
  EXPECT_NEAR(tpeak, (x + H) / c, 0.01);
  EXPECT_EQ(w_before, 0.0) << "calm before the front";
}

// Save / restore of the wind field state (the gust table and the clock) is bitwise, and replays continue identically.
TEST(terrain, state_restores) {
  Terr t;
  const std::string wf = wf_json("\"hfield_geom\": \"terr\", \"ridge_gain\": 1, \"ridge_len\": 30, \"z0\": 0.03");
  auto a = load_text(wind_xml(t, -30, 25, 9, wf, 3.0));
  a->set("/devices/wf", "gust", "", std::vector<double>{3, 20, 1, 1, 0, 0.5, 5});
  a->step(300);
  const auto blob = a->save();
  auto b = a->clone();
  a->step(500);
  b->restore(blob);
  b->step(500);
  EXPECT_EQ(a->state_hash(), b->state_hash());
}
