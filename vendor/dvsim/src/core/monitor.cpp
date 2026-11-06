#include "core/monitor.h"

#include <algorithm>
#include <cmath>

bool Monitor::sample(std::uint64_t tick, double v) {
  if (tick < from_ || tick > to_) return false;
  min_ = n_ == 0 ? v : std::min(min_, v);
  max_ = n_ == 0 ? v : std::max(max_, v);
  sumsq_ += v * v;
  last_ = v;
  ++n_;
  const bool inside = v >= lo_ && v <= hi_;
  if (hit_ || inside == always_) return false;  // a hit is a violation (always) or a satisfaction (eventually)
  hit_ = true;
  hit_tick_ = tick;
  hit_value_ = v;
  return always_;
}

MonitorResult Monitor::result() const {
  MonitorResult r;
  r.name = name_;
  r.hit = hit_;
  r.pass = always_ ? !hit_ : hit_;
  r.hit_tick = hit_tick_;
  r.hit_value = hit_value_;
  r.samples = n_;
  if (n_) {
    r.min_value = min_;
    r.max_value = max_;
    r.rms = std::sqrt(sumsq_ / static_cast<double>(n_));
    r.final_value = last_;
  }
  return r;
}
