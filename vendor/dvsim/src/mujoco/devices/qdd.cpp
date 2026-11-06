// Qdd: quasi-direct-drive actuator (planetary gearbox + FOC-driven BLDC) in MIT mode over CAN. Runs after the electrical
// solve of the tick (it needs the node voltage), before the force phase. Averaged model, no PWM, no phase currents.
//
// Fidelity notes (docs/models.md "Qdd"):
//  - CAN: the host's inputs (p_des, v_des, kp, kd, tau_ff, enable) are sampled every cmd_hz period (tick % period == 0) into a
//    ring and become effective latency_s later (integer ticks). Before the first sample is effective the driver is disabled.
//    A set() of any input stamps rx_tick (= a received frame); the frame is received latency_s after that. No frame for more
//    than timeout_s while enabled: the driver's safe behaviour (timeout_mode 0: damping tau = -kd_timeout dq, 1: open).
//  - Encoder: q is the output angle quantised to encoder_cpr counts (floor(x + 0.5)); dq is the ideal joint velocity (a real
//    driver differentiates / filters; no noise, no delay beyond the sample-and-hold of the tick).
//  - Control law (MIT mode): tau_cmd = clamp(kp, 0, kp_max) (p_des - q) + clamp(kd, 0, kd_max) (v_des - dq)
//    + clamp(tau_ff, +-tau_ff_max). The torque loop is ideal (a current loop much faster than the tick): the q-axis current is
//    set to the commanded value in the same tick, then limited.
//  - Current: i_q = tau_cmd / (Kt N k), k = eta driving, 1 when back-driven (smoothed over w_s like drive_motor in devices.cpp).
//    Limits, all applied to i_q (DC-equivalent winding, Kt = Ke = 60/(2 pi kv), R(T) line-to-line as for the ESC motor):
//      |i_q| <= i_limit x derate(T),   |R(T) i_q + Ke w| <= V_bus   (w = N dq, motor shaft)
//    The voltage bound is the linear SVPWM limit (max phase amplitude V_bus/sqrt(3), i.e. line-to-line peak V_bus, which is the
//    convention of the datasheet Kv / Ke / R): torque falls linearly to 0 at the no-load speed V_bus / Ke. No field weakening.
//    Beyond the no-load speed (overspeed) the feasible set is empty: i_q is the voltage bound, i.e. a regenerative brake.
//  - Output torque on the joint: tau_est = Kt N k i_q (electromagnetic, through the gearbox), minus the motor loss torque
//    N loss(w) (MotorParams i0 model) which is applied whether or not the driver is enabled.
//  - Thermal: C_th dT/dt = i_q^2 R(T) - (T - T_amb)/R_th, explicit Euler. derate = 1 below derate_start, linear to 0 at t_max;
//    fault latches at T >= t_max (torque off) and clears once enable = 0 and T < t_max.
//  - Bus: i_bus = (R i_q + Ke w) i_q / V_bus: the electrical power of the DC-equivalent winding over the node voltage (lossless
//    inverter, no quiescent draw; negative = regeneration into the node). It is injected as a constant sink into the NEXT
//    tick's electrical solve (one tick lag).
#include <algorithm>
#include <cmath>

#include "mujoco/devices/devices.h"

namespace devs {

void step_qdd(const mjModel* m, mjData* d, Devices& dv) {
  const double dt = m->opt.timestep;
  for (std::size_t i = 0; i < dv.qdd.cfg.size(); ++i) {
    const QddCfg& c = dv.qdd.cfg[i];
    QddState& s = dv.qdd.st[i];
    const MotorCfg& mc = c.motor;
    const std::uint64_t t = s.ticks, per = static_cast<std::uint64_t>(c.period), dly = static_cast<std::uint64_t>(c.delay);

    // CAN: sample the host inputs, then read the sample that has crossed the latency.
    if (t % per == 0) {
      double* r = s.ring[(t / per) % kQddRing];
      r[0] = s.p_des; r[1] = s.v_des; r[2] = s.kp; r[3] = s.kd; r[4] = s.tau_ff; r[5] = s.enable; r[6] = s.rx_tick;
    }
    const bool have = t >= dly;
    double cmd[7] = {0, 0, 0, 0, 0, 0, 0};
    if (have) std::copy_n(s.ring[((t - dly) / per) % kQddRing], 7, cmd);

    const double per_rad = c.cpr / (2.0 * kPi);
    s.q = std::floor(d->qpos[c.qadr] * per_rad + 0.5) / per_rad;
    s.dq = d->qvel[c.dof];
    const double w = c.n * s.dq;  // motor shaft speed
    const double kt = mc.ke(), r = mc.r_at(s.temp);

    const bool en = have && cmd[5] > 0.5;
    if (s.temp >= c.t_max) s.fault = 1.0;
    else if (!en) s.fault = 0.0;
    s.derate = std::clamp((c.t_max - s.temp) / (c.t_max - c.derate_start), 0.0, 1.0);
    // frame age at the driver: received rx_tick + latency
    const double age = static_cast<double>(t) - (cmd[6] + static_cast<double>(c.delay));
    s.timeout = en && age * dt > c.timeout_s ? 1.0 : 0.0;

    bool open = !en || s.fault > 0.5;
    double tau_cmd = 0.0;
    if (!open) {
      if (s.timeout > 0.5) {
        if (c.timeout_mode == 1) open = true;
        else tau_cmd = -c.kd_timeout * s.dq;
      } else {
        tau_cmd = std::clamp(cmd[2], 0.0, c.kp_max) * (cmd[0] - s.q) + std::clamp(cmd[3], 0.0, c.kd_max) * (cmd[1] - s.dq) +
                  std::clamp(cmd[4], -c.tau_ff_max, c.tau_ff_max);
      }
    }
    // gearbox factor: eta while the motor delivers power, 1 when back-driven, ramped in over the friction speed w_s
    auto kfac = [&](double sign) {
      const double g = std::tanh(w / mc.w_s) * sign;
      return g >= 0 ? c.eta : c.eta + (1.0 - c.eta) * -g;
    };
    auto sgn = [](double x) { return x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0; };
    double iq = 0.0;
    if (!open) {
      const double ilim = c.i_limit * s.derate, vlim = std::max(s.vbus, 0.0);
      iq = tau_cmd / (kt * c.n * kfac(sgn(tau_cmd)));
      const double lo = std::max(-ilim, (-vlim - kt * w) / r), hi = std::min(ilim, (vlim - kt * w) / r);
      if (lo <= hi) iq = std::clamp(iq, lo, hi);
      else iq = std::clamp(w > 0 ? (vlim - kt * w) / r : (-vlim - kt * w) / r, -ilim, ilim);  // overspeed: regenerative bound
    }
    s.tau_cmd = tau_cmd;
    s.i_q = iq;
    s.tau_est = iq == 0.0 ? 0.0 : kt * c.n * kfac(sgn(iq)) * iq;
    s.bus_i = !open && s.vbus > 1e-6 ? (r * iq + kt * w) * iq / s.vbus : 0.0;
    s.temp += dt / mc.c_th * (iq * iq * r - (s.temp - mc.t_amb) / mc.r_th);
    d->qfrc_applied[c.dof] += s.tau_est - c.n * mc.loss(w);
    ++s.ticks;
  }
}

}  // namespace devs
