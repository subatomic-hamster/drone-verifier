// Pellet launcher + camera pose (counter-UAS demo). Generic small-bore pneumatic launcher (air-rifle class): point-mass
// ballistics with quadratic drag, nothing else is modelled.
//
//  - Spawn: `fire` = 1 puts a pellet at the muzzle site with v = (v0 + v_std g) * axis + site velocity (world, mj_objectVelocity).
//    The axis is the site +z, tilted by spread_std g_x, spread_std g_y about the site x / y axes when spread_std > 0. A zero
//    sigma draws nothing from the RNG.
//  - Flight: a = g - k |v - w| (v - w), k = rho(z) cd A / (2 m), RK4 over the tick (w and rho held at the tick start: the error
//    is far below the drag-model error). rho = ISA at the pellet altitude, w = environment wind + turbulence (device phase order:
//    after the environment has advanced). Fidelity: constant cd (a real diabolo's drag rises toward the transonic range at the
//    top of the muzzle-speed band and a pellet tumbles/spins: neither is modelled); no spin drift, no pellet-pellet
//    interaction; Magnus only for launchers with cl_k / spin_tau (below); the muzzle gas blast and recoil are not modelled
//    (a 0.5 g pellet at 200 m/s is 0.1 N s).
//  - Magnus (cuas2, launchers with cl_k > 0 or spin_tau > 0; otherwise none of this runs and the output is bit-identical): the
//    per-shot inputs muzzle_velocity (0 = v0), spin_rps, dir_offset_x / y (state group Devices::lx) are read at `fire`. The
//    offsets tilt the axis deterministically before the spread. The pellet keeps omega0 = 2 pi spin_rps (site_x cross site_z);
//    omega(t) = omega0 exp(-t / spin_tau) (no decay for tau 0), held over the tick like rho. a_M = 0.5 rho A C_L |v_rel|^2 / m
//    * (w^ cross v^_rel), C_L = min(cl_k S, 0.25), S = |omega| r / |v_rel|. With spin 0 the term is skipped.
//  - Hit test: the tick's segment p0 -> p1 is cast with mj_ray against the geoms of `ray_groups`, using the poses of the tick
//    start (a target that moves 2 cm per tick is not swept: error <= its speed x dt). All bodies but the barrel's own are
//    hit. A segment is never missed by tunnelling however fast the pellet is (the cast is exact along the whole segment).
//  - End of flight: first hit (recorded in the ring; body 0 = the world = ground), age > max_time. Pellets are removed at once.
//  - Camera: pose outputs only.
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {
namespace {
constexpr double kParkZ = -1000.0;  // mocap bodies of free pool slots

inline void axis_of(const mjData* d, int sid, int col, double out[3]) {
  for (int r = 0; r < 3; ++r) out[r] = d->site_xmat[9 * sid + 3 * r + col];
}

struct Env {  // what a pellet flies through, held over one tick
  double g[3], w[3];
};
constexpr double kClMax = 0.25;
struct Mag {  // Magnus term of one pellet over one tick (on = false: not applied)
  bool on = false;
  double w[3] = {0, 0, 0};  // spin vector, rad/s
  double rho_am = 0;        // 0.5 rho A / m
  double r = 0, cl_k = 0;   // ball radius, lift slope
};

// dv/dt for velocity v (relative-air drag), k = rho cd A / (2 m)
inline void accel(const Env& e, const Mag& mg, double k, const double v[3], double a[3]) {
  const double r[3] = {v[0] - e.w[0], v[1] - e.w[1], v[2] - e.w[2]};
  const double vn = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  const double s = k * vn;
  for (int i = 0; i < 3; ++i) a[i] = e.g[i] - s * r[i];
  if (!mg.on) return;
  const double wn = std::sqrt(mg.w[0] * mg.w[0] + mg.w[1] * mg.w[1] + mg.w[2] * mg.w[2]);
  if (vn < 1e-9 || wn < 1e-12) return;
  const double cl = std::fmin(mg.cl_k * wn * mg.r / vn, kClMax);
  const double c = mg.rho_am * cl * vn / wn;  // x (w cross v_rel) = |w||v| sin(angle) (w^ cross v^)
  a[0] += c * (mg.w[1] * r[2] - mg.w[2] * r[1]);
  a[1] += c * (mg.w[2] * r[0] - mg.w[0] * r[2]);
  a[2] += c * (mg.w[0] * r[1] - mg.w[1] * r[0]);
}

// One RK4 step of (p, v) over h.
void rk4(const Env& e, const Mag& mg, double k, double h, double p[3], double v[3]) {
  double k1v[3], k2v[3], k3v[3], k4v[3], v2[3], v3[3], v4[3];
  accel(e, mg, k, v, k1v);
  for (int i = 0; i < 3; ++i) v2[i] = v[i] + 0.5 * h * k1v[i];
  accel(e, mg, k, v2, k2v);
  for (int i = 0; i < 3; ++i) v3[i] = v[i] + 0.5 * h * k2v[i];
  accel(e, mg, k, v3, k3v);
  for (int i = 0; i < 3; ++i) v4[i] = v[i] + h * k3v[i];
  accel(e, mg, k, v4, k4v);
  for (int i = 0; i < 3; ++i) {
    p[i] += h / 6.0 * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i]);
    v[i] += h / 6.0 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]);
  }
}
}  // namespace

void step_launchers(const mjModel* m, mjData* d, Devices& dv) {
  const double dt = m->opt.timestep;
  for (std::size_t i = 0; i < dv.camera.cfg.size(); ++i) {
    const CameraCfg& c = dv.camera.cfg[i];
    CameraState& s = dv.camera.st[i];
    for (int k = 0; k < 3; ++k) s.pos[k] = d->site_xpos[3 * c.sid + k];
    axis_of(d, c.sid, 2, s.fwd);
    axis_of(d, c.sid, 0, s.up);
  }
  if (dv.launcher.cfg.empty()) return;
  Env e{{m->opt.gravity[0], m->opt.gravity[1], m->opt.gravity[2]}, {0, 0, 0}};
  wind_at(dv, e.w);
  for (std::size_t li = 0; li < dv.launcher.cfg.size(); ++li) {
    const LauncherCfg& c = dv.launcher.cfg[li];
    LauncherState& s = dv.launcher.st[li];
    const double area = kPi * c.diameter * c.diameter / 4.0;
    mjtByte groups[mjNGROUP] = {0};
    for (int g = 0; g < 6; ++g) groups[g] = (c.ray_groups >> g) & 1;

    for (int k = 0; k < 3; ++k) s.muzzle_pos[k] = d->site_xpos[3 * c.sid + k];
    axis_of(d, c.sid, 2, s.muzzle_dir);

    if (s.fire > 0.5) {
      s.fire = 0;
      int slot = -1;
      for (int p = 0; p < kMaxPellets && slot < 0; ++p)
        if (s.active[p] < 0.5) slot = p;
      if (slot < 0) {
        s.dropped += 1;
      } else {
        double dir[3] = {s.muzzle_dir[0], s.muzzle_dir[1], s.muzzle_dir[2]};
        double v_set = c.v0;
        if (c.lx >= 0) {  // cuas2 per-shot inputs
          LauncherXState& x = dv.lx.st[static_cast<std::size_t>(c.lx)];
          if (x.muzzle_velocity > 0) v_set = x.muzzle_velocity;
          if (x.dir_offset_x != 0 || x.dir_offset_y != 0) {
            double ax[3], ay[3];
            axis_of(d, c.sid, 0, ax);
            axis_of(d, c.sid, 1, ay);
            double n = 0;
            for (int k = 0; k < 3; ++k) n += std::pow(dir[k] + x.dir_offset_x * ax[k] + x.dir_offset_y * ay[k], 2);
            n = std::sqrt(n);
            for (int k = 0; k < 3; ++k) dir[k] = (dir[k] + x.dir_offset_x * ax[k] + x.dir_offset_y * ay[k]) / n;
          }
          double ax[3], az[3];  // lateral spin axis = site x cross site z (the head's +y when site x is up)
          axis_of(d, c.sid, 0, ax);
          axis_of(d, c.sid, 2, az);
          const double lat[3] = {ax[1] * az[2] - ax[2] * az[1], ax[2] * az[0] - ax[0] * az[2], ax[0] * az[1] - ax[1] * az[0]};
          for (int k = 0; k < 3; ++k) x.omega0[3 * slot + k] = 2.0 * kPi * x.spin_rps * lat[k];
        }
        if (c.spread_std > 0) {
          double ax[3], ay[3];
          axis_of(d, c.sid, 0, ax);
          axis_of(d, c.sid, 1, ay);
          const double gx = c.spread_std * s.rng.gauss(), gy = c.spread_std * s.rng.gauss();
          double n = 0;
          for (int k = 0; k < 3; ++k) n += std::pow(dir[k] + gx * ax[k] + gy * ay[k], 2);
          n = std::sqrt(n);
          for (int k = 0; k < 3; ++k) dir[k] = (dir[k] + gx * ax[k] + gy * ay[k]) / n;
        }
        const double speed = v_set + (c.v_std > 0 ? c.v_std * s.rng.gauss() : 0.0);
        double sv[6];
        mj_objectVelocity(m, d, mjOBJ_SITE, c.sid, sv, 0);  // [rot, lin] in the world frame
        s.rounds += 1;
        s.active[slot] = 1;
        s.round_no[slot] = s.rounds;
        s.age[slot] = 0;
        for (int k = 0; k < 3; ++k) {
          s.pos[3 * slot + k] = s.muzzle_pos[k];
          s.vel[3 * slot + k] = speed * dir[k] + sv[3 + k];
        }
      }
    }

    for (int p = 0; p < kMaxPellets; ++p) {
      if (s.active[p] < 0.5) continue;
      double* pos = &s.pos[3 * p];
      double* vel = &s.vel[3 * p];
      const double p0[3] = {pos[0], pos[1], pos[2]};
      const double k = rho_at(dv, p0[2]) * c.cd * area / (2.0 * c.mass);
      Mag mg;
      if (c.lx >= 0) {
        const double* w0 = &dv.lx.st[static_cast<std::size_t>(c.lx)].omega0[3 * p];
        if (w0[0] != 0 || w0[1] != 0 || w0[2] != 0) {
          const double decay = c.spin_tau > 0 ? std::exp(-s.age[p] / c.spin_tau) : 1.0;
          mg.on = true;
          for (int j = 0; j < 3; ++j) mg.w[j] = w0[j] * decay;
          mg.rho_am = 0.5 * rho_at(dv, p0[2]) * area / c.mass;
          mg.r = c.diameter / 2;
          mg.cl_k = c.cl_k;
        }
      }
      rk4(e, mg, k, dt, pos, vel);
      s.age[p] += dt;
      double seg[3] = {pos[0] - p0[0], pos[1] - p0[1], pos[2] - p0[2]};
      const double len = std::sqrt(seg[0] * seg[0] + seg[1] * seg[1] + seg[2] * seg[2]);
      bool done = false;
      if (len > 0) {
        for (double& x : seg) x /= len;
        int geom = -1;
        const double dist = mj_ray(m, d, p0, seg, groups, 1, c.body, &geom, nullptr);
        if (dist >= 0 && dist <= len) {
          const int idx = static_cast<int>(static_cast<long long>(s.hit_n) % kHitRing);
          const int body = geom >= 0 ? m->geom_bodyid[geom] : 0;
          s.hit_body[idx] = body;
          s.hit_geom[idx] = geom;
          s.hit_t[idx] = s.time + dt;
          s.hit_v[idx] = std::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
          s.hit_round[idx] = s.round_no[p];
          for (int j = 0; j < 3; ++j) s.hit_pos[3 * idx + j] = p0[j] + dist * seg[j];
          s.hit_n += 1;
          (body == 0 ? s.ground : s.hits) += 1;
          done = true;
        }
      }
      if (!done && s.age[p] > c.max_time) {
        s.expired += 1;
        done = true;
      }
      if (done) {
        s.active[p] = 0;
        for (int j = 0; j < 3; ++j) pos[j] = vel[j] = 0;
      }
    }
    s.time += dt;
    for (std::size_t v = 0; v < c.mocap.size(); ++v) {
      double* mp = d->mocap_pos + 3 * c.mocap[v];
      if (s.active[v] > 0.5) {
        for (int j = 0; j < 3; ++j) mp[j] = s.pos[3 * v + j];
      } else {
        mp[0] = mp[1] = 0, mp[2] = kParkZ;
      }
    }
  }
}

}  // namespace devs
