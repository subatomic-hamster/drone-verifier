// Water: the Sea (surface elevation, current) and Vessels (buoyancy of sampled hulls, Fossen damping and added mass). Last force
// phase of the device tick, from the kinematics of mj_step1. Fidelity notes live in docs/models.md.
//
// Buoyancy. Each hull cell (volume v, extents e along the body axes) is a small box; its submerged fraction is the linear ramp
// f = clamp(1/2 + (eta(x, y, t) - z_c) / e_z, 0, 1) over its VERTICAL extent in the world, e_z = |R_20| e_x + |R_21| e_y + |R_22| e_z
// (the extent of a rotated box along world z), and it pushes up with rho g v f at its centre. For a prismatic hull the ramp is exact
// at any draft (the displaced volume is piecewise linear in the draft and exact at the cell faces) and a heeled hull gets the
// restoring moment of a column-wise integration of the waterplane (second moment of area to O(1/n^2) of the cells across).
//
// Added mass. Fossen's M_A dv/dt is an acceleration-dependent force, which cannot be applied lagged by a tick when M_A is a sizeable
// fraction of the hull mass (the iteration a+ = (F - M_A a)/m diverges for M_A > m). It is solved exactly against the multibody
// instead: with J (6 x nv, the CoM Jacobian of the vessel body), A the world-frame added-mass matrix, a0 = M^-1 f the unconstrained
// acceleration under every force of the tick so far, the accelerations of the added-mass system satisfy (M + J^T A J) qacc = f,
// so the vessel body's own acceleration is y = J qacc = (I + A J M^-1 J^T)^-1 J a0 and the force to apply is -A y (a 6 x 6 solve).
// Constraint forces (a hull on the seabed) and actuators are not in f; J-dot qvel is dropped (a CoM Jacobian rate term).
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {

double sea_eta(const Devices& dv, double x, double y) {
  if (dv.sea.cfg.empty()) return 0.0;
  const SeaCfg& c = dv.sea.cfg[0];
  const SeaState& s = dv.sea.st[0];
  double e = 0.0;
  for (const WaveCfg& w : c.waves) e += w.a * std::cos(w.k * (x * std::cos(w.th) + y * std::sin(w.th)) - w.w * s.time + w.phi);
  return c.level + s.wave_scale * e;
}

void sea_current(const Devices& dv, double out[3]) {
  for (int k = 0; k < 3; ++k) out[k] = dv.sea.st.empty() || k == 2 ? 0.0 : dv.sea.st[0].current[k];
}

std::vector<float> sea_surface_mesh(const Devices& dv, const std::vector<double>& xs, const std::vector<double>& ys) {
  std::vector<float> out;
  if (dv.sea.cfg.empty() || xs.size() < 2 || ys.size() < 2) return out;
  const SeaCfg& c = dv.sea.cfg[0];
  const SeaState& s = dv.sea.st[0];
  const std::size_t nx = xs.size(), ny = ys.size();
  struct Node { float p[3], n[3]; };
  std::vector<Node> g(nx * ny);
  for (std::size_t j = 0; j < ny; ++j)
    for (std::size_t i = 0; i < nx; ++i) {
      double e = 0, ex = 0, ey = 0;  // elevation and its slopes: d/dx cos(phase) = -k cos(th) sin(phase)
      for (const WaveCfg& w : c.waves) {
        const double ph = w.k * (xs[i] * std::cos(w.th) + ys[j] * std::sin(w.th)) - w.w * s.time + w.phi;
        e += w.a * std::cos(ph);
        const double ds = -w.a * w.k * std::sin(ph);
        ex += ds * std::cos(w.th), ey += ds * std::sin(w.th);
      }
      ex *= s.wave_scale, ey *= s.wave_scale;
      const double inv = 1.0 / std::sqrt(1.0 + ex * ex + ey * ey);
      g[j * nx + i] = {{static_cast<float>(xs[i]), static_cast<float>(ys[j]), static_cast<float>(c.level + s.wave_scale * e)},
                       {static_cast<float>(-ex * inv), static_cast<float>(-ey * inv), static_cast<float>(inv)}};
    }
  out.reserve((nx - 1) * (ny - 1) * 36);
  auto put = [&](const Node& n) { out.insert(out.end(), {n.p[0], n.p[1], n.p[2], n.n[0], n.n[1], n.n[2]}); };
  for (std::size_t j = 0; j + 1 < ny; ++j)
    for (std::size_t i = 0; i + 1 < nx; ++i) {
      const Node &a = g[j * nx + i], &b = g[j * nx + i + 1], &cc = g[(j + 1) * nx + i + 1], &d = g[(j + 1) * nx + i];
      put(a), put(b), put(cc), put(a), put(cc), put(d);  // counter-clockwise seen from above
    }
  return out;
}

namespace {
// Solve S y = b for a 6 x 6 system, Gaussian elimination with partial pivoting (ties keep the lowest row): deterministic.
void solve6(double S[6][6], double b[6]) {
  for (int c = 0; c < 6; ++c) {
    int p = c;
    for (int r = c + 1; r < 6; ++r)
      if (std::abs(S[r][c]) > std::abs(S[p][c])) p = r;
    if (p != c) {
      for (int k = 0; k < 6; ++k) std::swap(S[c][k], S[p][k]);
      std::swap(b[c], b[p]);
    }
    for (int r = c + 1; r < 6; ++r) {
      const double f = S[r][c] / S[c][c];
      if (f == 0.0) continue;
      for (int k = c; k < 6; ++k) S[r][k] -= f * S[c][k];
      b[r] -= f * b[c];
    }
  }
  for (int r = 5; r >= 0; --r) {
    double x = b[r];
    for (int k = r + 1; k < 6; ++k) x -= S[r][k] * b[k];
    b[r] = x / S[r][r];
  }
}
// R diag(a) R^T for a rotation R (row-major 3 x 3).
void rot_diag(const double* R, const double a[3], double out[3][3]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i][j] = R[3 * i] * a[0] * R[3 * j] + R[3 * i + 1] * a[1] * R[3 * j + 1] + R[3 * i + 2] * a[2] * R[3 * j + 2];
}
}  // namespace

void step_vessels(const mjModel* m, mjData* d, Devices& dv) {
  const double dt = m->opt.timestep;
  if (!dv.sea.cfg.empty() && !dv.vessel.cfg.empty()) {
    const SeaCfg& sc = dv.sea.cfg[0];
    const double grav = -m->opt.gravity[2], cur[3] = {dv.sea.st[0].current[0], dv.sea.st[0].current[1], 0.0};
    for (std::size_t vi = 0; vi < dv.vessel.cfg.size(); ++vi) {
      const VesselCfg& c = dv.vessel.cfg[vi];
      VesselState& s = dv.vessel.st[vi];
      const double* R = d->xmat + 9 * c.bid;
      const double* p0 = d->xpos + 3 * c.bid;
      const double* com = d->xipos + 3 * c.bid;
      double force[3] = {0, 0, 0}, torque[3] = {0, 0, 0}, sub = 0.0, fb = 0.0;
      for (const Cell& cell : c.cells) {
        double p[3];
        for (int i = 0; i < 3; ++i) p[i] = p0[i] + R[3 * i] * cell.c[0] + R[3 * i + 1] * cell.c[1] + R[3 * i + 2] * cell.c[2];
        const double ez = std::abs(R[6]) * cell.e[0] + std::abs(R[7]) * cell.e[1] + std::abs(R[8]) * cell.e[2];
        const double f = std::clamp(0.5 + (sea_eta(dv, p[0], p[1]) - p[2]) / ez, 0.0, 1.0);
        if (f <= 0.0) continue;
        sub += cell.v * f;
        const double up = sc.rho * grav * cell.v * f;
        fb += up;
        torque[0] += (p[1] - com[1]) * up;  // r x (0, 0, up)
        torque[1] += -(p[0] - com[0]) * up;
      }
      force[2] = fb;
      // Wetted fraction: the displacement relative to the design one (the vessel floating at rest displaces its own mass), capped at 1:
      // the hydrodynamic coefficients are those at the design waterline, and fade out as the hull comes out of the water.
      const double fw = std::min(1.0, sub * sc.rho / m->body_subtreemass[c.bid]);
      s.buoyancy[0] = 0, s.buoyancy[1] = 0, s.buoyancy[2] = fb;
      s.submerged = sub;
      s.wetted = fw;
      s.eta = sea_eta(dv, p0[0], p0[1]);

      // Fossen terms in the body frame (x forward, y left, z up), velocities relative to the water.
      double vel[6];
      mj_objectVelocity(m, d, mjOBJ_BODY, c.bid, vel, 0);  // rot:lin about the CoM, world axes
      double vb[3], wb[3];
      const double vr[3] = {vel[3] - cur[0], vel[4] - cur[1], vel[5] - cur[2]};
      for (int i = 0; i < 3; ++i) {
        vb[i] = R[i] * vr[0] + R[3 + i] * vr[1] + R[6 + i] * vr[2];  // R^T v
        wb[i] = R[i] * vel[0] + R[3 + i] * vel[1] + R[6 + i] * vel[2];
      }
      const double nu[6] = {vb[0], vb[1], vb[2], wb[0], wb[1], wb[2]};
      double tau[6];
      for (int i = 0; i < 6; ++i) tau[i] = -fw * (c.dl[i] * nu[i] + c.dq[i] * std::abs(nu[i]) * nu[i]);
      // Heading frame (schema Vessel.heading_frame): surge / sway / heave damping and added mass about the yaw-only axes (x along the body
      // x axis projected on the horizontal, z up). A body-frame heave force of a pitched hull has a horizontal part that, with the pitch
      // and heave of a hull riding waves in phase, averages to a net surge force.
      const double hn = std::hypot(R[0], R[3]);
      const bool hf = c.heading_frame && hn > 1e-3;
      const double Rh[9] = {R[0] / hn, -R[3] / hn, 0, R[3] / hn, R[0] / hn, 0, 0, 0, 1};  // columns: heading, left, up
      double tauh[3] = {0, 0, 0};
      if (hf) {
        const double nh[3] = {Rh[0] * vr[0] + Rh[3] * vr[1], Rh[1] * vr[0] + Rh[4] * vr[1], vr[2]};  // Rh^T v_r
        for (int i = 0; i < 3; ++i) {
          tauh[i] = -fw * (c.dl[i] * nh[i] + c.dq[i] * std::abs(nh[i]) * nh[i]);
          tau[i] = 0;
        }
      }
      if (c.has_a) {  // Munk terms of the 3-DOF Coriolis matrix of the added mass: [a_v v r, -a_u u r, (a_u - a_v) u v]
        tau[0] += fw * c.a[1] * nu[1] * nu[5];
        tau[1] += -fw * c.a[0] * nu[0] * nu[5];
        tau[5] += fw * (c.a[0] - c.a[1]) * nu[0] * nu[1];
      }
      double fh[3], th[3];
      for (int i = 0; i < 3; ++i) {
        fh[i] = R[3 * i] * tau[0] + R[3 * i + 1] * tau[1] + R[3 * i + 2] * tau[2];  // R tau
        th[i] = R[3 * i] * tau[3] + R[3 * i + 1] * tau[4] + R[3 * i + 2] * tau[5];
        if (hf) fh[i] += Rh[3 * i] * tauh[0] + Rh[3 * i + 1] * tauh[1] + Rh[3 * i + 2] * tauh[2];
      }
      for (const FinCfg& f : c.fins) {  // skegs: sideways lift + cross-flow drag at the fin centre (schema Vessel.fins)
        double pf[3], r[3], vf[3];
        for (int i = 0; i < 3; ++i) pf[i] = p0[i] + R[3 * i] * f.p[0] + R[3 * i + 1] * f.p[1] + R[3 * i + 2] * f.p[2];
        for (int i = 0; i < 3; ++i) r[i] = pf[i] - com[i];
        vf[0] = vel[3] - cur[0] + vel[1] * r[2] - vel[2] * r[1];  // v_com - current + omega x r (world)
        vf[1] = vel[4] - cur[1] + vel[2] * r[0] - vel[0] * r[2];
        vf[2] = vel[5] - cur[2] + vel[0] * r[1] - vel[1] * r[0];
        const double u = R[0] * vf[0] + R[3] * vf[1] + R[6] * vf[2], w = R[1] * vf[0] + R[4] * vf[1] + R[7] * vf[2];  // body x, y
        const double imm = std::clamp(0.5 + (sea_eta(dv, pf[0], pf[1]) - pf[2]) / f.span, 0.0, 1.0);
        const double fy = -imm * 0.5 * sc.rho * f.area * (f.cla * std::abs(u) * w + f.cd * std::abs(w) * w);
        const double fw3[3] = {fy * R[1], fy * R[4], fy * R[7]};  // along the body y axis, world components
        for (int i = 0; i < 3; ++i) fh[i] += fw3[i];
        th[0] += r[1] * fw3[2] - r[2] * fw3[1];
        th[1] += r[2] * fw3[0] - r[0] * fw3[2];
        th[2] += r[0] * fw3[1] - r[1] * fw3[0];
      }
      double* xf = d->xfrc_applied + 6 * c.bid;
      for (int i = 0; i < 3; ++i) {
        xf[i] += force[i] + fh[i];
        xf[3 + i] += torque[i] + th[i];
        s.hydro[i] = fh[i];
      }

      if (c.has_a && fw > 0.0) {  // added mass, exact against the multibody (see the header comment)
        const int nv = m->nv;
        std::vector<double> f(d->qfrc_applied, d->qfrc_applied + nv);
        for (int i = 0; i < nv; ++i) f[static_cast<std::size_t>(i)] += d->qfrc_passive[i] - d->qfrc_bias[i];
        for (int b = 1; b < m->nbody; ++b) {  // xfrc_applied (force at the CoM, torque about it) -> generalised forces
          const double* x6 = d->xfrc_applied + 6 * b;
          if (x6[0] != 0 || x6[1] != 0 || x6[2] != 0 || x6[3] != 0 || x6[4] != 0 || x6[5] != 0)
            mj_applyFT(m, d, x6, x6 + 3, d->xipos + 3 * b, b, f.data());
        }
        std::vector<double> a0(static_cast<std::size_t>(nv)), jac(static_cast<std::size_t>(6 * nv)), x(static_cast<std::size_t>(6 * nv));
        mj_solveM(m, d, a0.data(), f.data(), 1);
        mj_jacBodyCom(m, d, jac.data(), jac.data() + 3 * nv, c.bid);
        mj_solveM(m, d, x.data(), jac.data(), 6);
        double z[6] = {0, 0, 0, 0, 0, 0}, W[6][6];
        for (int i = 0; i < 6; ++i) {
          for (int k = 0; k < nv; ++k) z[i] += jac[static_cast<std::size_t>(i * nv + k)] * a0[static_cast<std::size_t>(k)];
          for (int j = 0; j < 6; ++j) {
            double w = 0;
            for (int k = 0; k < nv; ++k) w += jac[static_cast<std::size_t>(i * nv + k)] * x[static_cast<std::size_t>(j * nv + k)];
            W[i][j] = w;
          }
        }
        double A[6][6] = {}, At[3][3], Ar[3][3];
        rot_diag(hf ? Rh : R, c.a, At);
        rot_diag(R, c.a + 3, Ar);
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) A[i][j] = fw * At[i][j], A[3 + i][3 + j] = fw * Ar[i][j];
        double S[6][6], y[6];
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j) {
            double v = i == j ? 1.0 : 0.0;
            for (int k = 0; k < 6; ++k) v += W[i][k] * A[k][j];  // (I + W A) y = z; A W differs from W A once the inertia has products
            S[i][j] = v;
          }
          y[i] = z[i];
        }
        solve6(S, y);
        for (int i = 0; i < 6; ++i) {
          double fc = 0;
          for (int k = 0; k < 6; ++k) fc -= A[i][k] * y[k];
          xf[i] += fc;
          if (i < 3) s.hydro[i] += fc;
        }
      }
    }
  }
  for (std::size_t i = 0; i < dv.sea.cfg.size(); ++i) {
    dv.sea.st[i].time += dt;
    dv.sea.st[i].eta0 = sea_eta(dv, 0.0, 0.0);
  }
}

}  // namespace devs
