// FootContact: normal-force sensor / contact switch on one geom. Post-step hook (after mj_step2), like the other sensors.
//
// Fidelity notes (docs/models.md "FootContact"):
//  - force = sum over ALL contacts that involve the geom (any partner: rock boxes, hfield, ...) of the solved normal contact
//    force (mj_contactForce, component 0 of the contact frame; MuJoCo's convention is >= 0 for a pushing contact). Contacts the
//    solver gave no constraint row (excluded / disabled) count 0. The force belongs to the state s_k the tick started from,
//    like every other sensor sample.
//  - The reading is force + noise_std N(0, 1) (own RNG, one draw per sample, none when noise_std = 0), clipped to [0, f_max].
//    No bandwidth limit, no temperature or creep drift, no cross-axis (shear) sensitivity: a real strain-gauge foot also
//    reads a tangential load. Held between samples (rate_hz).
//  - `contact`: comparator on the reading, "on" above hysteresis_on, "off" at or below hysteresis_off, otherwise unchanged. A
//    change must persist for debounce_s of samples (the first differing sample starts the clock, the output follows once the
//    elapsed time since then reaches debounce_s). debounce_s = 0: the output follows at the first differing sample.
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {

void sample_feet(const mjModel* m, mjData* d, Devices& dv) {
  for (std::size_t i = 0; i < dv.foot.cfg.size(); ++i) {
    const FootContactCfg& c = dv.foot.cfg[i];
    FootContactState& s = dv.foot.st[i];
    if (s.ticks++ % static_cast<std::uint64_t>(c.period) != 0) continue;
    double sum = 0.0;
    for (int k = 0; k < d->ncon; ++k) {
      const mjContact& ct = d->contact[k];
      if (ct.geom1 != c.gid && ct.geom2 != c.gid) continue;
      mjtNum f6[6];
      mj_contactForce(m, d, k, f6);
      sum += f6[0];
    }
    if (c.noise_std > 0) sum += c.noise_std * s.rng.gauss();
    s.force = std::clamp(sum, 0.0, c.f_max);
    const bool now = s.contact > 0.5, want = now ? s.force > c.hys_off : s.force > c.hys_on;
    if (want == now) {
      s.pend = -1.0;
    } else {
      s.pend = s.pend < 0 ? 0.0 : s.pend + c.period * m->opt.timestep;
      if (s.pend >= c.debounce_s - 1e-12) {
        s.contact = want ? 1.0 : 0.0;
        s.pend = -1.0;
      }
    }
  }
}

}  // namespace devs
