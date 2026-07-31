// performance_grade.cpp — the tier table and its calibration.

#include "performance_grade.hpp"

#include <cmath>

namespace mb {
namespace {

struct TierDef {
  double lower_mips;  // inclusive lower bound, millions of iterations/second
  const char* label;
  const char* blurb;
  float r, g, b;
};

// Calibration reference points, all measured single-threaded on an Apple M5 Pro
// (4-wide NEON) at -O3 unless noted:
//
//     ~150-180 MIter/s   scalar or NEON compiled at -O0 (a Debug build)
//     ~550-590 MIter/s   scalar C++, release
//    ~1720-2242 MIter/s  NEON intrinsics and hand-written AArch64 asm, release
//
// Headroom above that is deliberate and reachable, not decorative: 8-wide AVX2
// roughly doubles the per-core rate, and the row-band kernel signature is built
// for multithreading, which on this 15-core part would put ~33,000 MIter/s in
// range. "Godly" is meant to be earned by actually finishing the optimisation
// work, not handed out for owning a fast laptop.
constexpr TierDef kTiers[kGradeTierCount] = {
    // lower   label                blurb                                                       r     g     b
    {    0.0, "Absolute Garbage", "Something is broken. Nothing should be this slow.",       0.85f, 0.15f, 0.15f},
    {   80.0, "Potato",           "Almost certainly an unoptimised Debug build.",            0.90f, 0.30f, 0.15f},
    {  200.0, "Awful",            "Optimisation is off, or the code fights the CPU.",        0.92f, 0.42f, 0.15f},
    {  350.0, "Bad",              "It runs, but wastes most of the processor.",              0.93f, 0.55f, 0.18f},
    {  500.0, "Poor",             "What a plain one-pixel-at-a-time loop gets.",             0.94f, 0.70f, 0.20f},
    {  700.0, "Mediocre",         "A little vectorisation, or a narrow SIMD path.",          0.90f, 0.82f, 0.25f},
    {  950.0, "Decent",           "Real SIMD is working now.",                               0.78f, 0.85f, 0.30f},
    { 1300.0, "Good",             "Solid throughput for a single CPU core.",                 0.55f, 0.85f, 0.35f},
    { 1800.0, "Very Good",        "Near the limit of one core at this vector width.",        0.35f, 0.88f, 0.45f},
    { 2400.0, "Great",            "Excellent use of a single core.",                         0.30f, 0.90f, 0.65f},
    { 3100.0, "Amazing",          "Wider vectors, or unusually good scheduling.",            0.30f, 0.85f, 0.90f},
    { 4200.0, "Elite",            "Beyond what one narrow-SIMD core can reach.",             0.45f, 0.70f, 1.00f},
    { 6000.0, "Legendary",        "Wide vectors plus genuine multi-core work.",              0.75f, 0.60f, 1.00f},
    { 9000.0, "Godly",            "Every core, every lane, nothing wasted.",                 1.00f, 0.85f, 0.35f},
};

// Meter endpoints for the log-scaled bar. Below the low end the bar reads empty,
// above the high end it reads full.
constexpr double kMeterLowMips = 80.0;
constexpr double kMeterHighMips = 9000.0;

}  // namespace

PerformanceGrade GradeIterationRate(double mega_iters_per_sec) {
  PerformanceGrade out;

  if (!(mega_iters_per_sec > 0.0)) {
    // Covers 0, negatives and NaN (NaN fails every comparison).
    const TierDef& bottom = kTiers[0];
    out.tier = GradeTier::AbsoluteGarbage;
    out.label = bottom.label;
    out.blurb = "Waiting for the first samples.";
    out.r = 0.55f;
    out.g = 0.55f;
    out.b = 0.55f;
    out.meter = 0.0f;
    return out;
  }

  int index = 0;
  for (int i = kGradeTierCount - 1; i >= 0; --i) {
    if (mega_iters_per_sec >= kTiers[i].lower_mips) {
      index = i;
      break;
    }
  }

  const TierDef& t = kTiers[index];
  out.tier = static_cast<GradeTier>(index);
  out.label = t.label;
  out.blurb = t.blurb;
  out.r = t.r;
  out.g = t.g;
  out.b = t.b;

  // Log scale: the range spans more than two orders of magnitude, so a linear
  // bar would sit almost empty for every realistic single-core result.
  const double lo = std::log(kMeterLowMips);
  const double hi = std::log(kMeterHighMips);
  double fill = (std::log(mega_iters_per_sec) - lo) / (hi - lo);
  if (fill < 0.0) {
    fill = 0.0;
  }
  if (fill > 1.0) {
    fill = 1.0;
  }
  out.meter = static_cast<float>(fill);

  return out;
}

void GradeTierRange(GradeTier tier, double& lower_mips, double& upper_mips) {
  const int index = static_cast<int>(tier);
  if (index < 0 || index >= kGradeTierCount) {
    lower_mips = 0.0;
    upper_mips = 0.0;
    return;
  }
  lower_mips = kTiers[index].lower_mips;
  upper_mips = (index + 1 < kGradeTierCount) ? kTiers[index + 1].lower_mips : 0.0;
}

const char* GradeTierLabel(GradeTier tier) {
  const int index = static_cast<int>(tier);
  if (index < 0 || index >= kGradeTierCount) {
    return "";
  }
  return kTiers[index].label;
}

const char* GradeTierBlurb(GradeTier tier) {
  const int index = static_cast<int>(tier);
  if (index < 0 || index >= kGradeTierCount) {
    return "";
  }
  return kTiers[index].blurb;
}

MachineUtilisation DescribeUtilisation(double measured_mips, int physical_cores,
                                       int threads_used) {
  MachineUtilisation u;
  if (physical_cores < 1) {
    physical_cores = 1;
  }
  if (threads_used < 1) {
    threads_used = 1;
  }
  if (threads_used > physical_cores) {
    // More threads than physical cores is legal (SMT, oversubscription) but for
    // reporting purposes the chip cannot be more than fully used.
    threads_used = physical_cores;
  }

  u.cores_total = physical_cores;
  u.cores_used = threads_used;
  u.percent_of_chip =
      100.0 * static_cast<double>(threads_used) / static_cast<double>(physical_cores);
  u.using_whole_chip = (threads_used >= physical_cores);

  // Once every core is in use the measurement speaks for itself; extrapolating
  // would only add error. Below that, project from the observed per-core rate.
  if (measured_mips > 0.0 && !u.using_whole_chip) {
    const double per_core = measured_mips / static_cast<double>(threads_used);
    u.projected_all_core_mips = per_core * static_cast<double>(physical_cores);
    u.projection_valid = true;
  }
  return u;
}

SimpleRating RateConsistency(double cv_percent) {
  // Coefficient of variation: standard deviation as a percentage of the mean.
  // Low means the timings are repeatable and a difference between kernels can
  // be trusted; high means the scheduler is interfering.
  if (!(cv_percent >= 0.0)) {
    return {"unknown", 0.55f, 0.55f, 0.55f};
  }
  if (cv_percent < 1.0) {
    return {"Rock solid", 0.35f, 0.88f, 0.45f};
  }
  if (cv_percent < 3.0) {
    return {"Excellent", 0.55f, 0.85f, 0.35f};
  }
  if (cv_percent < 6.0) {
    return {"Good", 0.85f, 0.85f, 0.30f};
  }
  if (cv_percent < 12.0) {
    return {"Noisy", 0.93f, 0.60f, 0.20f};
  }
  return {"Very noisy", 0.90f, 0.30f, 0.20f};
}

SimpleRating RateSmoothness(double fps) {
  if (!(fps > 0.0)) {
    return {"unknown", 0.55f, 0.55f, 0.55f};
  }
  if (fps >= 90.0) {
    return {"Buttery", 0.35f, 0.88f, 0.45f};
  }
  if (fps >= 55.0) {
    return {"Smooth", 0.55f, 0.85f, 0.35f};
  }
  if (fps >= 30.0) {
    return {"Playable", 0.85f, 0.85f, 0.30f};
  }
  if (fps >= 15.0) {
    return {"Choppy", 0.93f, 0.60f, 0.20f};
  }
  return {"Slideshow", 0.90f, 0.30f, 0.20f};
}

}  // namespace mb
