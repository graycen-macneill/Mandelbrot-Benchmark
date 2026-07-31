// timing.cpp — clock granularity probe.

#include "timing.hpp"

namespace mb {

double MeasuredClockGranularityNanos() {
  static const double cached = [] {
    // Spin until the clock ticks, repeatedly, and keep the smallest non-zero
    // delta observed. 256 trials is plenty to hit the true granularity.
    double smallest = 1e18;
    for (int trial = 0; trial < 256; ++trial) {
      const Timestamp start = Now();
      Timestamp end = start;
      while (end == start) {
        end = Now();
      }
      const double delta = ElapsedNanos(start, end);
      if (delta > 0.0 && delta < smallest) {
        smallest = delta;
      }
    }
    return smallest == 1e18 ? kClockTickNanos : smallest;
  }();
  return cached;
}

}  // namespace mb
