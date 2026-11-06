#pragma once
// Online pass/fail monitor over one scalar signal. Pure logic (no solver, no schema types) so a sweep can
// reuse it per run. Feed it one sample per tick, in order; samples outside [from_tick, to_tick] are ignored.
#include <cstdint>
#include <string>
#include <utility>

struct MonitorResult {
  std::string name;
  bool pass = false;
  bool hit = false;  // always: first violation seen (=> fail); eventually: first satisfaction seen (=> pass)
  std::uint64_t hit_tick = 0;
  double hit_value = 0;
  std::uint64_t samples = 0;  // in-window samples; the stats below are 0 when this is 0
  double min_value = 0, max_value = 0, rms = 0, final_value = 0;
};

class Monitor {
 public:
  Monitor(std::string name, bool always, double lo, double hi, std::uint64_t from_tick, std::uint64_t to_tick)
      : name_(std::move(name)), always_(always), lo_(lo), hi_(hi), from_(from_tick), to_(to_tick) {}

  // Returns true iff this sample is the first violation of an `always` monitor. NaN lies outside any band.
  bool sample(std::uint64_t tick, double v);
  MonitorResult result() const;

 private:
  std::string name_;
  bool always_;
  double lo_, hi_;
  std::uint64_t from_, to_;
  bool hit_ = false;
  std::uint64_t hit_tick_ = 0, n_ = 0;
  double hit_value_ = 0, min_ = 0, max_ = 0, sumsq_ = 0, last_ = 0;
};
