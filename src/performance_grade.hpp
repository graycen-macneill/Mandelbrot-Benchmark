// performance_grade.hpp — plain-language performance ratings for the simple panel.
//
// The dashboard proper reports trimmed means, standard deviations and confidence
// intervals, which is the right output for someone reading a benchmark. This is
// the other half: a single word that anyone can read at a glance.
//
// The grade is driven by **pixel-iterations per second**, not pixels per second
// or frames per second, because only the former is workload-independent.
// Measured on the development machine, the same three kernels across five very
// different views:
//
//     metric        scalar range        NEON range      spread
//     Mpx/s          2.9 - 1181         9.3 - 3088      ~400x   <- useless
//     MIter/s         542 - 590         1720 - 2242     ~1.3x   <- stable
//
// A rating built on Mpx/s would call the same kernel "Godly" on a zoomed-out
// low-iteration view and "Absolute Garbage" on a deep boundary zoom, while the
// hardware did exactly the same work. Iterations/second does not have that
// defect, so it is what the tiers below are cut against.
#pragma once

#include <cstddef>

namespace mb {

// Ordered worst to best.
enum class GradeTier : int {
  AbsoluteGarbage = 0,
  Potato,
  Awful,
  Bad,
  Poor,
  Mediocre,
  Decent,
  Good,
  VeryGood,
  Great,
  Amazing,
  Elite,
  Legendary,
  Godly,
  kCount,
};

constexpr int kGradeTierCount = static_cast<int>(GradeTier::kCount);

struct PerformanceGrade {
  GradeTier tier = GradeTier::AbsoluteGarbage;
  const char* label = "";
  const char* blurb = "";  // one line of plain-language context
  float r = 1.0f;          // display colour, 0..1
  float g = 1.0f;
  float b = 1.0f;
  float meter = 0.0f;      // 0..1 bar fill, log-scaled across the whole range
};

// mega_iters_per_sec: millions of Mandelbrot inner-loop iterations per second.
// Pass 0 or a negative value for "no data yet" and you get the bottom tier with
// a zero meter.
PerformanceGrade GradeIterationRate(double mega_iters_per_sec);

// Lower and upper bounds of a tier in MIter/s, for building a scale legend.
// The top tier reports an upper bound of 0.0 meaning "unbounded".
void GradeTierRange(GradeTier tier, double& lower_mips, double& upper_mips);

const char* GradeTierLabel(GradeTier tier);

// One short plain-English sentence describing what a tier means. Available for
// every tier, not just the active one, so the scale can be shown as a legend.
const char* GradeTierBlurb(GradeTier tier);

// --- Whole-machine context -------------------------------------------------
// The grade above rates one kernel on one core. On a multi-core machine that is
// a small fraction of the hardware, and saying "Very Good" without that context
// invites reading it as a verdict on the computer. This supplies the context.
struct MachineUtilisation {
  int cores_used = 1;
  int cores_total = 1;
  double percent_of_chip = 100.0;
  bool using_whole_chip = false;  // cores_used >= cores_total

  // Linear projection of what all physical cores could reach, extrapolated from
  // the per-core rate. An ESTIMATE, not a measurement: real scaling is sublinear
  // because of shared cache, memory bandwidth, power limits and heterogeneous
  // core speeds. Callers must label it as such.
  //
  // Deliberately NOT offered once cores_used == cores_total: at that point the
  // measured figure *is* the machine's throughput and a projection would be
  // strictly worse information.
  double projected_all_core_mips = 0.0;
  bool projection_valid = false;
};

// measured_mips is the total throughput achieved by `threads_used` threads
// working together, i.e. what the profiler actually observed.
MachineUtilisation DescribeUtilisation(double measured_mips, int physical_cores,
                                       int threads_used);

// --- Secondary, simpler ratings --------------------------------------------

struct SimpleRating {
  const char* label = "";
  float r = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
};

// Run-to-run consistency, from the coefficient of variation (percent).
SimpleRating RateConsistency(double cv_percent);

// Perceived smoothness, from frames per second.
SimpleRating RateSmoothness(double fps);

}  // namespace mb
