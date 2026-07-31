// profiler.hpp — Component 4: the Statistical Profiler.
//
// Design notes on the statistics, since "measure it a few times and average" is
// the usual failure mode here:
//
//  * Sample size. A single timing is a sample of size one from a distribution
//    whose tail is shaped by the OS scheduler, not by the kernel under test.
//    We collect n >= kCltMinSamples (30) before reporting anything as
//    trustworthy, which is the conventional threshold at which the sampling
//    distribution of the mean is treated as approximately normal under the
//    Central Limit Theorem.
//
//  * Trimming. Execution-time distributions are right-skewed: a preempted run
//    can be 10x the median, but nothing can run faster than the hardware
//    allows. The mean is therefore dragged upward by scheduler noise while the
//    left tail is essentially hard-bounded. We drop the slowest 5% of samples
//    and report both the raw and trimmed statistics, so the reader can see how
//    much of the mean was noise. This is an upper-tail trim, not a symmetric
//    trimmed mean — the fast tail is signal, not outliers.
//
//  * Dispersion. Standard deviation is reported on the raw samples (n-1
//    denominator, i.e. the unbiased sample variance) alongside the coefficient
//    of variation, which is what actually tells you whether a difference
//    between two kernels is meaningful at this sample size.
//
// Allocation: the sample buffer and the sort scratch are both fixed-capacity
// members. push() is O(1) and compute() sorts in place via std::sort (introsort,
// no allocation). Nothing here touches the heap after construction.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "arch.hpp"

namespace mb {

// Conventional CLT rule-of-thumb threshold for treating the sampling
// distribution of the mean as approximately normal.
constexpr std::uint32_t kCltMinSamples = 30;

// Fraction of the slowest samples discarded before computing trimmed metrics.
constexpr double kDefaultTrimFraction = 0.05;

struct LatencyStats {
  std::uint32_t n = 0;         // samples in the window
  std::uint32_t n_trimmed = 0; // samples surviving the upper-tail trim
  std::uint32_t n_dropped = 0; // samples discarded as upper-tail outliers

  // Raw window
  double mean_ns = 0.0;
  double stddev_ns = 0.0;  // unbiased sample stddev (n-1)
  double min_ns = 0.0;
  double max_ns = 0.0;
  double median_ns = 0.0;
  double p95_ns = 0.0;
  double p99_ns = 0.0;

  // After dropping the slowest kTrimFraction
  double trimmed_mean_ns = 0.0;
  double trimmed_stddev_ns = 0.0;

  // Inferential
  double sem_ns = 0.0;   // standard error of the mean, stddev / sqrt(n)
  double ci95_ns = 0.0;  // half-width of the 95% CI, 1.96 * SEM
  double cv_pct = 0.0;   // coefficient of variation, 100 * stddev / mean

  bool clt_satisfied = false;  // n >= kCltMinSamples

  // Convenience conversions for display.
  double mean_us() const { return mean_ns / 1000.0; }
  double trimmed_mean_us() const { return trimmed_mean_ns / 1000.0; }
  double trimmed_mean_ms() const { return trimmed_mean_ns / 1'000'000.0; }
};

// Fixed-capacity rolling sample window.
//
// Not thread-safe: compute() mutates an internal scratch buffer. Intended for
// single-threaded use from the render loop, which is where it lives.
template <std::size_t Capacity>
class LatencyProfiler {
 public:
  static_assert(Capacity >= 2, "need at least two samples to compute a variance");

  static constexpr std::size_t kCapacity = Capacity;

  void Reset() noexcept {
    head_ = 0;
    size_ = 0;
    total_pushed_ = 0;
  }

  // O(1), allocation-free. Overwrites the oldest sample once full.
  void Push(double nanoseconds) noexcept {
    samples_[head_] = nanoseconds;
    head_ = (head_ + 1) % Capacity;
    if (size_ < Capacity) {
      ++size_;
    }
    ++total_pushed_;
  }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  bool full() const noexcept { return size_ == Capacity; }
  std::uint64_t total_pushed() const noexcept { return total_pushed_; }

  // Computes the full statistics block. Allocation-free: sorts a member scratch
  // buffer rather than a temporary.
  LatencyStats Compute(double trim_fraction = kDefaultTrimFraction) const noexcept {
    LatencyStats out;
    if (size_ == 0) {
      return out;
    }

    const std::size_t n = size_;
    out.n = static_cast<std::uint32_t>(n);
    out.clt_satisfied = out.n >= kCltMinSamples;

    // Copy the live window into scratch and sort ascending.
    for (std::size_t i = 0; i < n; ++i) {
      scratch_[i] = samples_[i];
    }
    std::sort(scratch_.begin(), scratch_.begin() + static_cast<std::ptrdiff_t>(n));

    out.min_ns = scratch_[0];
    out.max_ns = scratch_[n - 1];
    out.median_ns = Percentile(n, 0.50);
    out.p95_ns = Percentile(n, 0.95);
    out.p99_ns = Percentile(n, 0.99);

    // Raw mean and unbiased sample standard deviation.
    out.mean_ns = Mean(0, n);
    out.stddev_ns = StdDev(0, n, out.mean_ns);

    if (out.mean_ns > 0.0) {
      out.cv_pct = 100.0 * out.stddev_ns / out.mean_ns;
    }
    if (n >= 2) {
      out.sem_ns = out.stddev_ns / std::sqrt(static_cast<double>(n));
      out.ci95_ns = 1.959964 * out.sem_ns;  // normal approximation
    }

    // Upper-tail trim: drop the slowest ceil(trim_fraction * n), always keeping
    // at least two samples so the trimmed variance stays defined.
    std::size_t drop = 0;
    if (trim_fraction > 0.0 && n >= 3) {
      drop = static_cast<std::size_t>(
          std::ceil(trim_fraction * static_cast<double>(n)));
      const std::size_t max_drop = n - 2;
      if (drop > max_drop) {
        drop = max_drop;
      }
    }
    const std::size_t kept = n - drop;
    out.n_dropped = static_cast<std::uint32_t>(drop);
    out.n_trimmed = static_cast<std::uint32_t>(kept);
    out.trimmed_mean_ns = Mean(0, kept);
    out.trimmed_stddev_ns = StdDev(0, kept, out.trimmed_mean_ns);

    return out;
  }

 private:
  // Linear-interpolated percentile over the sorted scratch buffer.
  double Percentile(std::size_t n, double q) const noexcept {
    if (n == 1) {
      return scratch_[0];
    }
    const double pos = q * static_cast<double>(n - 1);
    const double lo = std::floor(pos);
    const std::size_t i = static_cast<std::size_t>(lo);
    if (i + 1 >= n) {
      return scratch_[n - 1];
    }
    const double frac = pos - lo;
    return scratch_[i] + frac * (scratch_[i + 1] - scratch_[i]);
  }

  double Mean(std::size_t begin, std::size_t end) const noexcept {
    if (end <= begin) {
      return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      sum += scratch_[i];
    }
    return sum / static_cast<double>(end - begin);
  }

  // Two-pass variance: numerically stable, and the cost is irrelevant here
  // because it runs once per frame over a few hundred doubles, never inside
  // the measured region.
  double StdDev(std::size_t begin, std::size_t end, double mean) const noexcept {
    const std::size_t count = end - begin;
    if (count < 2) {
      return 0.0;
    }
    double acc = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      const double d = scratch_[i] - mean;
      acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(count - 1));
  }

  alignas(MB_CACHE_ALIGN) mutable std::array<double, Capacity> samples_{};
  alignas(MB_CACHE_ALIGN) mutable std::array<double, Capacity> scratch_{};
  std::size_t head_ = 0;
  std::size_t size_ = 0;
  std::uint64_t total_pushed_ = 0;
};

}  // namespace mb
