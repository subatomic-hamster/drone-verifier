// Jet vanes: flat plates hinged in the exhaust of a rotor (schema JetVane). Phase 5 of the device tick, after the rotor forces of the
// same tick (it reads the rotor's thrust of this tick). Fidelity notes live in docs/models.md.
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {

void step_vanes(const mjModel* m, mjData* d, Devices& dv) {
  for (std::size_t i = 0; i < dv.vane.cfg.size(); ++i) {
    const JetVaneCfg& c = dv.vane.cfg[i];
    JetVaneState& s = dv.vane.st[i];
    const RotorCfg& rc = dv.rotor.cfg[static_cast<std::size_t>(c.rot)];
    const RotorState& rs = dv.rotor.st[static_cast<std::size_t>(c.rot)];
    s.force[0] = s.force[1] = s.force[2] = s.alpha = s.speed = 0;
    if (!(std::abs(rs.thrust) > 1e-9)) continue;  // no exhaust
    const double* ax = d->xaxis + 3 * rc.jnt;
    const double sg = rs.thrust > 0 ? -1.0 : 1.0;  // the exhaust leaves opposite the thrust vector
    const double e[3] = {sg * ax[0], sg * ax[1], sg * ax[2]};
    const double* R = d->site_xmat + 9 * c.sid;
    double ch[3] = {R[2], R[5], R[8]}, nm[3] = {R[0], R[3], R[6]};  // site z (chord) and x (plate normal), world
    double ec = e[0] * ch[0] + e[1] * ch[1] + e[2] * ch[2];
    if (ec < 0) {  // symmetric plate: turn the chord to point downstream, the normal with it
      for (int k = 0; k < 3; ++k) ch[k] = -ch[k], nm[k] = -nm[k];
      ec = -ec;
    }
    const double en = e[0] * nm[0] + e[1] * nm[1] + e[2] * nm[2];
    const double ve = c.jet_factor * std::sqrt(std::abs(rs.thrust) / (rs.rho * c.exit_area));
    const double q = 0.5 * rs.rho * ve * ve;
    const double alpha = std::atan2(en, ec), sa = std::sin(alpha);
    const double cn = c.cn_max * std::tanh(c.cn_alpha * sa / c.cn_max);
    double f[3], rr[3], tq[3];
    for (int k = 0; k < 3; ++k) {
      f[k] = q * c.area * (cn * nm[k] + c.cd0 * ch[k]);
      rr[k] = d->site_xpos[3 * c.sid + k] - d->xipos[3 * c.bid + k];
      s.force[k] = f[k];
    }
    mju_cross(tq, rr, f);
    for (int k = 0; k < 3; ++k) {
      d->xfrc_applied[6 * c.bid + k] += f[k];
      d->xfrc_applied[6 * c.bid + 3 + k] += tq[k];
    }
    s.alpha = alpha;
    s.speed = ve;
  }
}

}  // namespace devs
