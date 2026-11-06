// Range sensors: depth camera, ToF, lidar (phase 5). Post-step hook of the device tick (after mj_step2), like the other sensors: the
// sample describes the state at the START of the sampling tick (xpos, site pose of s_k) and is visible from its end on; the first
// tick samples (ticks % period == 0). Between samples the image is held.
//
// Geometry: mj_multiRay from the site origin along the precomputed unit rays rotated by the site frame, against the geoms of
// ray_groups, the sensor's own body excluded, cutoff max_range. mj_ray skips fully transparent geoms (alpha 0) and has no notion of
// contype / conaffinity, so what the sensor sees is the set of geoms in the mask, visible or not collidable. Deterministic: a pure
// function of the pose, the geometry and the device RNG.
//
// Projection "rosette" (Livox Mid-360-like, SP2): the directions of a frame are generated here from the global ray index (frames x
// rays_per_frame + k) with the R2 low-discrepancy sequence, so successive frames fill the field of view without repeating; everything
// below the direction generation is shared with the other projections. docs/models.md "Rosette scan pattern".
//
// Fidelity (docs/models.md): ideal pinhole / angular grid, no lens distortion, no rolling shutter, no motion blur (the pose of the
// sampling tick), a ray hits the nearest surface of any material (no specular / transparent / dark-surface dropouts, no multipath, no
// edge flying pixels, no stereo occlusion shadow), range-dependent Gaussian noise sigma0 + sigma1 r + sigma2 r^2 per valid pixel,
// uncorrelated between pixels and frames (a real stereo camera's error is spatially correlated and surface dependent).
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {

void rosette_dir(const RangefinderCfg& c, std::uint64_t k, double out[3]) {  // R2 sequence (Roberts 2018): u_j = frac(off_j + m / g^j)
  const double m = static_cast<double>(k);
  const double u1 = c.off1 + 0.7548776662466927 * m, u2 = c.off2 + 0.5698402909980532 * m;  // 1/g, 1/g^2, g = plastic number
  const double az = c.fov_h * (u1 - std::floor(u1) - 0.5), el = c.fov_v_min + (c.fov_v_max - c.fov_v_min) * (u2 - std::floor(u2));
  out[0] = std::sin(el); out[1] = std::cos(el) * std::sin(az); out[2] = std::cos(el) * std::cos(az);
}

void sample_ranges(const mjModel* m, mjData* d, Devices& dv) {
  if (dv.range.cfg.empty()) return;
  dv.range_img.resize(dv.range.cfg.size());
  for (std::size_t i = 0; i < dv.range.cfg.size(); ++i) {
    const RangefinderCfg& c = dv.range.cfg[i];
    RangefinderState& s = dv.range.st[i];
    std::vector<double>& img = dv.range_img[i];
    const std::size_t n = static_cast<std::size_t>(c.n_h) * static_cast<std::size_t>(c.n_v);
    if (img.size() != n) img.assign(n, 0.0);
    if (s.ticks++ % static_cast<std::uint64_t>(c.period) != 0) continue;
    const double* R = d->site_xmat + 9 * c.sid;
    const double* o = d->site_xpos + 3 * c.sid;
    std::vector<double> dirs(3 * n), dist(n);
    std::vector<int> geom(n);
    const std::uint64_t base = static_cast<std::uint64_t>(s.frames) * n;  // rosette: first ray index of this frame
    for (std::size_t k = 0; k < n; ++k) {
      double loc[3];
      if (c.rosette) {  // R2 sequence (Roberts 2018): u_j = frac(off_j + m / g^j), g = plastic number; spherical frame, +x = up
        rosette_dir(c, base + k, loc);
      } else {
        loc[0] = c.dir[3 * k]; loc[1] = c.dir[3 * k + 1]; loc[2] = c.dir[3 * k + 2];
      }
      for (int r = 0; r < 3; ++r)
        dirs[3 * k + static_cast<std::size_t>(r)] = R[3 * r] * loc[0] + R[3 * r + 1] * loc[1] + R[3 * r + 2] * loc[2];
    }
    mjtByte groups[mjNGROUP] = {0};
    for (int g = 0; g < 6; ++g) groups[g] = (c.ray_groups >> g) & 1;
    mj_multiRay(m, d, o, dirs.data(), groups, 1, c.body, geom.data(), dist.data(), nullptr, static_cast<int>(n), c.cutoff);
    // Rain (rain.cpp, docs/models.md "Rain"): none of this runs, and no RNG is drawn, at rate 0.
    const double rr = rain_rate(dv);
    const double alpha = rr > 0 ? c.rain_a * std::pow(rr, c.rain_b) : 0.0;  // extinction, 1/m
    const double zone = std::max(c.rain_zone, c.min_r);
    const double drops = rr > 0 && c.rain_det > 0 ? 1951.0 * std::pow(rr, 0.21) : 0.0;  // Marshall-Palmer drops per m^3
    const double p_false = 1.0 - std::exp(-drops * 0.25 * kPi * c.rain_beam * c.rain_beam * (zone - c.min_r) * c.rain_det);
    auto det = [](double margin_db) { return 1.0 / (1.0 + std::exp(-margin_db / 1.5)); };
    double valid = 0;
    for (std::size_t k = 0; k < n; ++k) {
      double v = 0.0;
      bool drop_rain = false;
      if (dist[k] >= 0) {
        double r = dist[k];
        double sg = c.s0 + c.s1 * r + c.s2 * r * r;
        if (alpha > 0) {
          const double rm = std::max(std::min(r, c.max_r), 1e-3);
          const double loss_db = 8.686 * alpha * rm + c.rain_window_db * std::sqrt(rr / 25.0), margin = c.rain_margin_db + 20.0 * std::log10(c.max_r / rm);
          sg *= std::pow(10.0, loss_db / 20.0);
          drop_rain = s.rng.uniform() >= det(margin - loss_db) / det(margin);
        }
        if (sg > 0) r += sg * s.rng.gauss();
        if (c.res > 0) r = std::round(r / c.res) * c.res;
        const bool drop = c.dropout > 0 && s.rng.uniform() < c.dropout;
        v = r * c.scale[k];
        if (drop || drop_rain || !(v >= c.min_r && v <= c.max_r)) v = 0.0;
      }
      if (p_false > 0 && s.rng.uniform() < p_false) {  // a drop in the near beam returns before the real surface
        const double rf = c.min_r + (zone - c.min_r) * s.rng.uniform();
        if (dist[k] < 0 || rf < dist[k]) v = rf * c.scale[k];
      }
      img[k] = v;
      valid += v > 0 ? 1.0 : 0.0;
    }
    s.frames += 1;
    s.valid = valid;
    s.t = d->time;
  }
}

}  // namespace devs
