// Terrain-aware wind (schema WindField). The uniform wind of the Environment becomes a field evaluated at the point where the air is
// used (rotor hub, drag body CoM). Everything is a simple, documented model, NOT CFD; estimates are marked.
//
//   wind(p) = W_h(p) + ridge(p) + gust(p, t) + g_lee(p) turbulence
//
//  1. Boundary layer (neutral log law, Environment.wind = W at z_ref above ground): W_h = W (ln(h / z0) / ln(z_ref / z0)) on the
//     horizontal components, h = height above the ground under p (hfield surface, bilinear, else the flat ground_z), clamped to
//     [2 z0, h_max]. z0 = 0: W_h = W. Roughness lengths: Wieringa 1992 / WMO-No. 8 (short grass 0.03 m, snow / bare rock 0.001-0.01 m).
//  2. Ridge flow (ESTIMATE; linear potential flow over low hills, the surface-perturbation result of Jackson & Hunt 1975 as simplified
//     in Hunt, Leibovich & Richards 1988: for terrain h = a cos(kx) in a wind U along x, u' = U k a cos(kx) e^{-kz},
//     w' = U dh/dx e^{-kz}, i.e. a speed-up over crests, slowing in valleys, lift on windward and sink on lee slopes, decaying over
//     a height 1/k). A general terrain is represented by h_L (Gaussian-smoothed over ridge_len L) and h_B (over 4 L): with
//     u = wind_xy / |wind_xy|, U = |W_h,xy| and d = exp(-h / L):
//         u' = gain U d clamp((h_L - h_B) / L, -0.5, 1)   along u,        w' = gain U d clamp(u . grad h_L, -1, 1).
//     (A single decay height for all scales; nonlinear effects beyond slope ~0.3 are clamped, not modelled.)
//  3. Lee separation (ESTIMATE; Hunt's "lee-slope" separation sets in near a mean slope of ~0.3 (~17 deg); we ramp from 0.15 to 0.4):
//     the lee factor f = max over k = 0..3 of ramp(-u . grad h_L at p - k L u) (1 - k / 4) e^{-h / 2L} adds a downdraft -lee_down U f and
//     multiplies the Dryden turbulence by 1 + (lee_turb - 1) f (rotor / separated-wake turbulence).
//  4. Discrete gusts (FAR 25.341(a) / MIL-F-8785C 1 - cos shape): a gust of amplitude A, gradient distance H, direction u_g is
//         g(s) = u_g (A / 2) (1 - cos(pi s / H)),  0 <= s <= 2 H,  else 0   (peak A at s = H, total length 2 H)
//     with s = c (t - t0) - n . (x, y) for a front moving at c > 0 along the unit horizontal direction n (n = horizontal part of u_g, else
//     of the wind, else +x: the gust front reaches the world origin at t0 and a point x metres downstream x / c later), or, for
//     speed 0, uniform in space with s = max(|W_xy|, 2) (t - t0) (Taylor's frozen-turbulence convention, 2 m/s floor as the Dryden
//     advection). Several gusts add; u_g may have a vertical part (an updraft gust).
//  Turbulence: the Environment's Dryden vector, with the height h above LOCAL ground (aero.cpp) and the lee gain of 3.
// Everything is evaluated from the config and the saved clock only; the smoothed grids are derived at bind().
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mujoco/devices/devices.h"

namespace devs {
namespace {
constexpr double kVFloor = 2.0;  // m/s, advection floor of the uniform gust (as the Dryden filter)

// Separable Gaussian smoothing (sigma in metres), edges clamped; returns a copy for sigma < 0.3 cells.
std::vector<float> blur(const std::vector<float>& src, int nrow, int ncol, double sigma, double dx, double dy) {
  std::vector<float> a = src, b = src;
  for (int pass = 0; pass < 2; ++pass) {
    const double sc = sigma / (pass == 0 ? dx : dy);  // sigma in cells along this axis
    if (sc < 0.3) continue;
    const int rad = static_cast<int>(std::ceil(3.0 * sc));
    std::vector<double> w(static_cast<std::size_t>(2 * rad + 1));
    double sum = 0;
    for (int k = -rad; k <= rad; ++k) sum += w[static_cast<std::size_t>(k + rad)] = std::exp(-0.5 * k * k / (sc * sc));
    for (double& x : w) x /= sum;
    for (int r = 0; r < nrow; ++r)
      for (int c = 0; c < ncol; ++c) {
        double acc = 0;
        for (int k = -rad; k <= rad; ++k) {
          const int rr = pass == 1 ? std::clamp(r + k, 0, nrow - 1) : r, cc = pass == 0 ? std::clamp(c + k, 0, ncol - 1) : c;
          acc += w[static_cast<std::size_t>(k + rad)] * a[static_cast<std::size_t>(rr) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(cc)];
        }
        b[static_cast<std::size_t>(r) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(c)] = static_cast<float>(acc);
      }
    a = b;
  }
  return a;
}

double bil(const std::vector<float>& g, const WindFieldCfg& c, double x, double y) {
  const double u = std::clamp((x - c.x_lo) / c.dx, 0.0, c.ncol - 1.0), v = std::clamp((y - c.y_lo) / c.dy, 0.0, c.nrow - 1.0);
  const int i = std::min(static_cast<int>(u), c.ncol - 2), j = std::min(static_cast<int>(v), c.nrow - 2);
  const double fu = u - i, fv = v - j;
  const std::size_t n = static_cast<std::size_t>(c.ncol), k = static_cast<std::size_t>(j) * n + static_cast<std::size_t>(i);
  return (1 - fv) * ((1 - fu) * g[k] + fu * g[k + 1]) + fv * ((1 - fu) * g[k + n] + fu * g[k + n + 1]);
}

double ground_of(const WindFieldCfg& c, double x, double y) { return c.nrow ? bil(c.h, c, x, y) : c.ground_z; }

// Slope of the smoothed terrain along the unit horizontal direction (ux, uy) at (x, y).
double slope_along(const WindFieldCfg& c, double x, double y, double ux, double uy, double* gx = nullptr, double* gy = nullptr) {
  const double e = std::max(c.dx, c.dy);
  const double sx = (bil(c.hl, c, x + e, y) - bil(c.hl, c, x - e, y)) / (2 * e), sy = (bil(c.hl, c, x, y + e) - bil(c.hl, c, x, y - e)) / (2 * e);
  if (gx) *gx = sx;
  if (gy) *gy = sy;
  return ux * sx + uy * sy;
}

double lee_of(const WindFieldCfg& c, double x, double y, double hag, double ux, double uy) {
  if (!c.nrow || c.ridge_gain <= 0) return 0.0;
  double f = 0;
  for (int k = 0; k < 4; ++k) {
    const double s = -slope_along(c, x - k * c.ridge_len * ux, y - k * c.ridge_len * uy, ux, uy);
    f = std::max(f, std::clamp((s - 0.15) / 0.25, 0.0, 1.0) * (1.0 - 0.25 * k));
  }
  return f * std::exp(-hag / (2 * c.ridge_len));
}
}  // namespace

void bind_wind_field(const mjModel* m, WindFieldCfg& c) {
  c.nrow = c.ncol = 0;
  c.h.clear(); c.hl.clear(); c.hb.clear();
  if (c.hfield_geom.empty()) return;
  const std::string who = "dvsim.devices: device '" + c.meta.name + "': ";
  const int g = mj_name2id(m, mjOBJ_GEOM, c.hfield_geom.c_str());
  if (g < 0) throw std::runtime_error(who + "geom '" + c.hfield_geom + "' not found in the model");
  if (m->geom_type[g] != mjGEOM_HFIELD) throw std::runtime_error(who + "geom '" + c.hfield_geom + "' is not an hfield");
  const double* q = m->geom_quat + 4 * g;
  if (m->geom_bodyid[g] != 0 || std::abs(q[0] - 1.0) > 1e-9)
    throw std::runtime_error(who + "hfield '" + c.hfield_geom + "' must be a static (world body), unrotated geom");
  const int id = m->geom_dataid[g];
  c.nrow = m->hfield_nrow[id];
  c.ncol = m->hfield_ncol[id];
  if (c.nrow < 2 || c.ncol < 2) throw std::runtime_error(who + "hfield needs at least 2 x 2 samples");
  const double* sz = m->hfield_size + 4 * id;
  const double* p = m->geom_pos + 3 * g;
  c.x_lo = p[0] - sz[0]; c.dx = 2 * sz[0] / (c.ncol - 1);
  c.y_lo = p[1] - sz[1]; c.dy = 2 * sz[1] / (c.nrow - 1);
  const float* data = m->hfield_data + m->hfield_adr[id];
  c.h.resize(static_cast<std::size_t>(c.nrow) * static_cast<std::size_t>(c.ncol));
  // MuJoCo's data row 0 is the +y edge (image convention): our row r is at y_lo + r dy, so rows are read from the end.
  for (int r = 0; r < c.nrow; ++r)
    for (int col = 0; col < c.ncol; ++col)
      c.h[static_cast<std::size_t>(r) * static_cast<std::size_t>(c.ncol) + static_cast<std::size_t>(col)] = static_cast<float>(
          p[2] + static_cast<double>(data[static_cast<std::size_t>(c.nrow - 1 - r) * static_cast<std::size_t>(c.ncol) + static_cast<std::size_t>(col)]) * sz[2]);
  c.hl = blur(c.h, c.nrow, c.ncol, c.ridge_len, c.dx, c.dy);
  c.hb = blur(c.h, c.nrow, c.ncol, 4 * c.ridge_len, c.dx, c.dy);
}

double ground_height(const Devices& dv, double x, double y) {
  return dv.wfield.cfg.empty() ? std::nan("") : ground_of(dv.wfield.cfg[0], x, y);
}

void wind_at_point(const Devices& dv, const double pos[3], double out[3]) {
  if (dv.wfield.cfg.empty() || dv.env.st.empty()) return wind_at(dv, out);
  const WindFieldCfg& c = dv.wfield.cfg[0];
  const WindState& ws = dv.wfield.st[0];
  const EnvState& e = dv.env.st[0];
  const double ground = ground_of(c, pos[0], pos[1]), hag = std::max(pos[2] - ground, 0.0);
  const double wh = std::hypot(e.wind[0], e.wind[1]);
  const double ux = wh > 1e-9 ? e.wind[0] / wh : 1.0, uy = wh > 1e-9 ? e.wind[1] / wh : 0.0;
  double prof = 1.0;
  if (c.z0 > 0) prof = std::log(std::clamp(hag, 2 * c.z0, c.h_max) / c.z0) / std::log(c.z_ref / c.z0);
  out[0] = e.wind[0] * prof; out[1] = e.wind[1] * prof; out[2] = e.wind[2];
  double lee = 0;
  if (c.nrow && c.ridge_gain > 0 && wh > 1e-9) {
    const double U = wh * prof, d = std::exp(-hag / c.ridge_len);
    const double du = c.ridge_gain * U * d * std::clamp((bil(c.hl, c, pos[0], pos[1]) - bil(c.hb, c, pos[0], pos[1])) / c.ridge_len, -0.5, 1.0);
    const double w = c.ridge_gain * U * d * std::clamp(slope_along(c, pos[0], pos[1], ux, uy), -1.0, 1.0);
    lee = lee_of(c, pos[0], pos[1], hag, ux, uy);
    out[0] += du * ux; out[1] += du * uy;
    out[2] += w - c.lee_down * U * lee;
  }
  for (int k = 0; k < kMaxGusts; ++k) {
    const double* g = ws.gust[k];
    if (g[0] == 0) continue;
    double s;
    if (g[6] > 0) s = g[6] * (ws.t - g[5]) - (pos[0] * g[7] + pos[1] * g[8]);
    else s = std::max(wh, kVFloor) * (ws.t - g[5]);
    if (s < 0 || s > 2 * g[1]) continue;
    const double a = 0.5 * g[0] * (1.0 - std::cos(kPi * s / g[1]));
    for (int i = 0; i < 3; ++i) out[i] += a * g[2 + i];
  }
  const double gain = 1.0 + (c.lee_turb - 1.0) * lee;
  for (int i = 0; i < 3; ++i) out[i] += e.turb[i] * gain;
}

void step_wind_field(const mjModel*, const mjData* d, Devices& dv) {
  if (dv.wfield.cfg.empty() || dv.env.cfg.empty()) return;
  WindState& s = dv.wfield.st[0];
  if (s.gust_clear > 0.5) {
    for (auto& g : s.gust) std::fill(g, g + 9, 0.0);
    s.gust_clear = 0;
  }
  const double* gi = s.gust_in;
  const double len = std::sqrt(gi[2] * gi[2] + gi[3] * gi[3] + gi[4] * gi[4]);
  if (gi[0] != 0 && gi[1] > 0 && len > 0) {
    double* g = s.gust[static_cast<int>(s.n_gusts) % kMaxGusts];
    s.n_gusts += 1;
    const EnvState& e = dv.env.st[0];
    g[0] = gi[0]; g[1] = gi[1];
    for (int i = 0; i < 3; ++i) g[2 + i] = gi[2 + i] / len;
    g[5] = s.t + std::max(gi[5], 0.0);
    g[6] = std::max(gi[6], 0.0);
    double nx = g[2], ny = g[3], nl = std::hypot(nx, ny);
    if (nl < 1e-9) { nx = e.wind[0]; ny = e.wind[1]; nl = std::hypot(nx, ny); }
    g[7] = nl > 1e-9 ? nx / nl : 1.0; g[8] = nl > 1e-9 ? ny / nl : 0.0;
  }
  for (double& x : s.gust_in) x = 0;
  const EnvCfg& ec = dv.env.cfg[0];
  const double pos[3] = {ec.bid >= 0 ? d->xpos[3 * ec.bid] : 0.0, ec.bid >= 0 ? d->xpos[3 * ec.bid + 1] : 0.0,
                         ec.bid >= 0 ? d->xpos[3 * ec.bid + 2] : ec.ref_height};
  const WindFieldCfg& c = dv.wfield.cfg[0];
  s.ground = ground_of(c, pos[0], pos[1]);
  s.hag = std::max(pos[2] - s.ground, 0.0);
  wind_at_point(dv, pos, s.wind_ref);
  const double wh = std::hypot(dv.env.st[0].wind[0], dv.env.st[0].wind[1]);
  s.lee = wh > 1e-9 ? lee_of(c, pos[0], pos[1], s.hag, dv.env.st[0].wind[0] / wh, dv.env.st[0].wind[1] / wh) : 0.0;
}

void advance_wind_clock(Devices& dv, double dt) {
  if (!dv.wfield.st.empty()) dv.wfield.st[0].t += dt;
}

}  // namespace devs
