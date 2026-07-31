// verify_kernels.cpp — headless correctness + statistics self-test.
//
// Deliberately free of SDL / ImGui / hwinfo dependencies so it can be built with
// a single clang++ invocation (see scripts/selftest.sh) and run in CI or over
// SSH. It answers three questions:
//
//   1. Do all compiled kernels produce bit-identical output to the scalar
//      reference? (A fast wrong kernel is worthless.)
//   2. Does the profiler compute the statistics it claims to, including the
//      upper-tail trim, checked against hand-computed values?
//   3. What is the actual measured speedup on this machine?
//
// Exit status is 0 only if every check passes.

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "../src/alloc_tracker.hpp"
#include "../src/arch.hpp"
#include "../src/cpu_features.hpp"
#include "../src/kernel_api.hpp"
#include "../src/kernel_registry.hpp"
#include "../src/performance_grade.hpp"
#include "../src/profiler.hpp"
#include "../src/render_target.hpp"
#include "../src/thread_pool.hpp"
#include "../src/timing.hpp"

namespace {

int g_failures = 0;

void Check(bool condition, const char* what) {
  std::printf("  [%s] %s\n", condition ? " ok " : "FAIL", what);
  if (!condition) {
    ++g_failures;
  }
}

bool NearlyEqual(double a, double b, double tolerance) {
  return std::fabs(a - b) <= tolerance;
}

mb::KernelParams MakeParams(int width, int height, int max_iter, double center_x,
                            double center_y, double span_x) {
  mb::KernelParams p{};
  const double dx = span_x / static_cast<double>(width);
  p.dx = static_cast<float>(dx);
  p.dy = static_cast<float>(dx);  // square pixels
  p.cx0 = static_cast<float>(center_x - span_x * 0.5);
  p.cy0 = static_cast<float>(center_y - dx * static_cast<double>(height) * 0.5);
  p.escape_r2 = 4.0f;
  p.width = width;
  p.vec_width = mb::PaddedWidth(width);
  p.height = height;
  p.max_iter = max_iter;
  p.stride = mb::kRenderStride;
  return p;
}

std::uint64_t Checksum(const std::uint16_t* buf, const mb::KernelParams& p) {
  // FNV-1a over the visible region only.
  std::uint64_t h = 1469598103934665603ull;
  for (int y = 0; y < p.height; ++y) {
    const std::uint16_t* row = buf + static_cast<std::size_t>(y) * p.stride;
    for (int x = 0; x < p.width; ++x) {
      h ^= row[x];
      h *= 1099511628211ull;
    }
  }
  return h;
}

// -------------------------------------------------------------------------
// 1. Kernel agreement
// -------------------------------------------------------------------------
void TestKernelAgreement() {
  std::printf("\n=== Kernel correctness (bit-exactness vs scalar C++) ===\n");

  mb::RenderTarget& t = mb::Target();
  const mb::KernelRegistry& reg = mb::Kernels();

  // Three views: the full set (mixed escape depths), a deep zoom near the
  // boundary (long dependency chains, many lanes surviving), and a region
  // entirely outside the set (immediate escape, exercises the early-out).
  struct View {
    const char* name;
    int w, h, iter;
    double cx, cy, span;
  };
  const View views[] = {
      {"full set        ", 512, 288, 256, -0.6, 0.0, 3.2},
      {"boundary zoom   ", 256, 144, 2048, -0.743643887, 0.131825904, 0.00005},
      {"outside the set ", 256, 144, 128, 3.0, 3.0, 0.5},
      {"iter cap = 0    ", 64, 32, 0, -0.6, 0.0, 3.2},
      {"non-mult-of-8 w ", 253, 61, 300, -0.6, 0.0, 3.2},
  };

  for (const View& v : views) {
    const mb::KernelParams p = MakeParams(v.w, v.h, v.iter, v.cx, v.cy, v.span);

    // Reference: scalar into t.reference.
    std::memset(t.reference, 0xCD, sizeof(std::uint16_t) * mb::kIterationBufferElems);
    mandelbrot_scalar_cpp(t.reference, &p, 0, p.height);
    const std::uint64_t ref_sum = Checksum(t.reference, p);

    for (int k = 0; k < reg.count; ++k) {
      const mb::KernelDesc& d = reg[k];
      if (!d.available || d.fn == &mandelbrot_scalar_cpp) {
        continue;
      }

      // Poison the buffer first so a kernel that silently writes nothing fails.
      std::memset(t.iterations, 0xCD, sizeof(std::uint16_t) * mb::kIterationBufferElems);
      d.fn(t.iterations, &p, 0, p.height);

      const mb::VerifyResult r =
          mb::CompareBuffers(t.iterations, t.reference, p.width, p.height, p.stride);
      const std::uint64_t sum = Checksum(t.iterations, p);

      char label[256];
      std::snprintf(label, sizeof(label),
                    "%s | %-26s mismatches=%llu maxdiff=%d checksum=%s",
                    v.name, d.name, static_cast<unsigned long long>(r.mismatched_pixels),
                    r.max_abs_difference, sum == ref_sum ? "match" : "DIFFER");
      Check(r.exact() && sum == ref_sum, label);
    }
  }

  // Row-range split must equal a single whole-image call: this is the property
  // that makes the [row_begin, row_end) signature safe to thread later.
  {
    const mb::KernelParams p = MakeParams(256, 128, 512, -0.6, 0.0, 3.2);
    mandelbrot_scalar_cpp(t.reference, &p, 0, p.height);

    for (int k = 0; k < reg.count; ++k) {
      const mb::KernelDesc& d = reg[k];
      if (!d.available) {
        continue;
      }
      std::memset(t.iterations, 0xCD, sizeof(std::uint16_t) * mb::kIterationBufferElems);
      const int mid = p.height / 3;  // deliberately not a power of two
      d.fn(t.iterations, &p, 0, mid);
      d.fn(t.iterations, &p, mid, p.height);

      const mb::VerifyResult r =
          mb::CompareBuffers(t.iterations, t.reference, p.width, p.height, p.stride);
      char label[192];
      std::snprintf(label, sizeof(label), "banded call equals whole-image call | %-26s",
                    d.name);
      Check(r.exact(), label);
    }
  }

  // Spot-check known points so a kernel that agrees with a *broken* reference
  // still gets caught.
  {
    const mb::KernelParams p = MakeParams(64, 64, 500, 0.0, 0.0, 0.001);
    mandelbrot_scalar_cpp(t.reference, &p, 0, p.height);
    // c ~ 0 is deep inside the set: every pixel must hit the iteration cap.
    bool all_interior = true;
    for (int y = 0; y < p.height && all_interior; ++y) {
      for (int x = 0; x < p.width; ++x) {
        if (t.reference[static_cast<std::size_t>(y) * p.stride + x] != 500) {
          all_interior = false;
          break;
        }
      }
    }
    Check(all_interior, "origin neighbourhood saturates at max_iter (interior)");
  }
  {
    const mb::KernelParams p = MakeParams(8, 8, 500, 10.0, 10.0, 0.001);
    mandelbrot_scalar_cpp(t.reference, &p, 0, p.height);
    // The escape test runs on the *entry* magnitude, and |z_0| = 0 always
    // passes it, so the first step is always counted and z_1 = c. |c| ~ 14 then
    // fails the test at step 2, giving a count of exactly 1. This is the
    // standard escape-time convention; the point of pinning it here is that all
    // five kernels must agree on it.
    Check(t.reference[0] == 1, "far exterior escapes with iteration count 1");
  }
}

// -------------------------------------------------------------------------
// 2. Profiler statistics
// -------------------------------------------------------------------------
void TestProfiler() {
  std::printf("\n=== Statistical profiler ===\n");

  mb::LatencyProfiler<64> prof;
  for (int i = 1; i <= 10; ++i) {
    prof.Push(static_cast<double>(i));
  }
  const mb::LatencyStats s = prof.Compute(0.05);

  // Hand-computed for the sample {1..10}:
  //   mean            = 5.5
  //   sample stddev   = sqrt(82.5 / 9)  = 3.0276504
  //   median          = 5.5
  //   p95 (linear)    = 9.55
  //   trim 5% of n=10 -> ceil(0.5) = 1 dropped, leaving {1..9}:
  //   trimmed mean    = 5.0
  //   trimmed stddev  = sqrt(60 / 8)    = 2.7386128
  Check(s.n == 10, "n == 10");
  Check(NearlyEqual(s.mean_ns, 5.5, 1e-12), "mean == 5.5");
  Check(NearlyEqual(s.stddev_ns, 3.0276503541, 1e-9), "sample stddev == sqrt(82.5/9)");
  Check(NearlyEqual(s.min_ns, 1.0, 1e-12), "min == 1");
  Check(NearlyEqual(s.max_ns, 10.0, 1e-12), "max == 10");
  Check(NearlyEqual(s.median_ns, 5.5, 1e-12), "median == 5.5");
  Check(NearlyEqual(s.p95_ns, 9.55, 1e-12), "p95 == 9.55 (linear interpolation)");
  Check(s.n_dropped == 1, "upper-tail trim drops ceil(0.05 * 10) == 1");
  Check(s.n_trimmed == 9, "9 samples survive the trim");
  Check(NearlyEqual(s.trimmed_mean_ns, 5.0, 1e-12), "trimmed mean == 5.0");
  Check(NearlyEqual(s.trimmed_stddev_ns, 2.7386127875, 1e-9),
        "trimmed stddev == sqrt(60/8)");
  Check(NearlyEqual(s.sem_ns, 3.0276503541 / std::sqrt(10.0), 1e-9),
        "SEM == stddev / sqrt(n)");
  Check(!s.clt_satisfied, "n=10 correctly reported as below the CLT threshold");

  // The trim must actually suppress a scheduler-noise outlier.
  mb::LatencyProfiler<64> spiky;
  for (int i = 0; i < 39; ++i) {
    spiky.Push(100.0);
  }
  spiky.Push(100000.0);  // one preempted run
  const mb::LatencyStats t = spiky.Compute(0.05);
  Check(t.n == 40, "spiky sample n == 40");
  Check(t.clt_satisfied, "n=40 satisfies the CLT threshold");
  Check(t.mean_ns > 2000.0, "raw mean is dragged upward by the outlier");
  Check(NearlyEqual(t.trimmed_mean_ns, 100.0, 1e-9),
        "trimmed mean recovers the true 100 ns hardware throughput");
  Check(NearlyEqual(t.trimmed_stddev_ns, 0.0, 1e-9), "trimmed stddev collapses to 0");

  // Ring buffer must overwrite oldest, not grow or wrap incorrectly.
  mb::LatencyProfiler<8> ring;
  for (int i = 0; i < 100; ++i) {
    ring.Push(static_cast<double>(i));
  }
  const mb::LatencyStats rs = ring.Compute(0.0);
  Check(ring.size() == 8, "ring window saturates at capacity");
  Check(ring.total_pushed() == 100, "total_pushed keeps counting past capacity");
  Check(NearlyEqual(rs.min_ns, 92.0, 1e-12), "ring retains only the newest 8 samples");
  Check(NearlyEqual(rs.max_ns, 99.0, 1e-12), "ring max is the most recent sample");

  // Degenerate inputs must not produce NaN.
  mb::LatencyProfiler<8> single;
  single.Push(42.0);
  const mb::LatencyStats ss = single.Compute();
  Check(NearlyEqual(ss.mean_ns, 42.0, 1e-12), "n=1 mean is the sample itself");
  Check(ss.stddev_ns == 0.0, "n=1 stddev is 0, not NaN");
  Check(ss.n_dropped == 0, "n=1 trims nothing");
}

// -------------------------------------------------------------------------
// 3. Colour mapping
// -------------------------------------------------------------------------
void TestColourize() {
  std::printf("\n=== Palette / colourisation ===\n");

  mb::RenderTarget& t = mb::Target();
  constexpr int kMaxIter = 256;
  const mb::KernelParams p = MakeParams(256, 144, kMaxIter, -0.6, 0.0, 3.2);

  mb::EnsurePaletteLut(t, mb::Palette::Ember, kMaxIter);
  Check(t.lut_max_iter == kMaxIter, "LUT records the iteration cap it was built for");

  // Points in the set must be black; the LUT must be opaque everywhere.
  Check((t.palette_lut[kMaxIter] & 0x00FFFFFFu) == 0u, "interior colour is black");
  bool all_opaque = true;
  for (int i = 0; i <= kMaxIter; ++i) {
    if ((t.palette_lut[i] >> 24) != 0xFFu) {
      all_opaque = false;
      break;
    }
  }
  Check(all_opaque, "every LUT entry is fully opaque (alpha 255)");

  // Rebuilding with identical arguments must be a no-op, and switching palette
  // must actually change the table.
  const std::uint32_t before = t.palette_lut[kMaxIter / 2];
  mb::EnsurePaletteLut(t, mb::Palette::Ember, kMaxIter);
  Check(t.palette_lut[kMaxIter / 2] == before, "rebuild with same args is stable");
  mb::EnsurePaletteLut(t, mb::Palette::Ice, kMaxIter);
  Check(t.palette_lut[kMaxIter / 2] != before, "switching palette changes the table");
  mb::EnsurePaletteLut(t, mb::Palette::Ember, kMaxIter);

  // Colourise a real render and confirm the output is a genuine image rather
  // than a uniform fill (which is what a broken LUT index would produce).
  mandelbrot_scalar_cpp(t.iterations, &p, 0, p.height);
  std::memset(t.pixels, 0, sizeof(std::uint32_t) * mb::kIterationBufferElems);
  mb::Colourize(t, p.width, p.height);

  std::uint64_t distinct_estimate = 0;
  std::uint32_t first = t.pixels[0];
  bool any_black = false;
  bool any_nonblack = false;
  for (int y = 0; y < p.height; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * mb::kRenderStride;
    for (int x = 0; x < p.width; ++x) {
      const std::uint32_t px = t.pixels[row + static_cast<std::size_t>(x)];
      if (px != first) {
        ++distinct_estimate;
      }
      if ((px & 0x00FFFFFFu) == 0u) {
        any_black = true;
      } else {
        any_nonblack = true;
      }
    }
  }
  Check(distinct_estimate > 0, "colourised image is not a uniform fill");
  Check(any_black && any_nonblack, "image contains both interior and exterior pixels");

  // Colourise must not write outside the requested region.
  const std::size_t past_end =
      static_cast<std::size_t>(p.height) * mb::kRenderStride;
  Check(t.pixels[past_end] == 0u, "Colourize does not write past the last row");
  Check(t.pixels[static_cast<std::size_t>(p.width)] == 0u,
        "Colourize does not write past the visible width");
}

// -------------------------------------------------------------------------
// 4. Zero heap allocation on the hot path
// -------------------------------------------------------------------------
// This is the constraint most likely to rot silently, so it is asserted here
// rather than only being displayed in the GUI.
void TestNoHotPathAllocation() {
  std::printf("\n=== Zero heap allocation on the hot path ===\n");

  mb::RenderTarget& t = mb::Target();
  const mb::KernelRegistry& reg = mb::Kernels();
  const mb::KernelParams p = MakeParams(256, 144, 128, -0.6, 0.0, 3.2);

  // Warm everything that legitimately allocates once, before measuring.
  mb::EnsurePaletteLut(t, mb::Palette::Ember, p.max_iter);
  mb::MeasuredClockGranularityNanos();
  mb::LatencyProfiler<256> prof;
  prof.Push(1.0);
  (void)prof.Compute();

  // Confirm the operator new replacement is actually linked in before trusting
  // a zero reading. Counting startup allocations would not work: this binary
  // uses only static buffers and stdio, so it legitimately reaches this point
  // having never called operator new. Provoke one instead.
  {
    const std::uint64_t allocs_before = mb::HeapAllocCount();
    const std::uint64_t frees_before = mb::HeapFreeCount();
    int* probe = new int[64];
    mb::DoNotOptimize(probe);
    delete[] probe;
    Check(mb::HeapAllocCount() == allocs_before + 1,
          "allocation counter is wired up (a deliberate new[] was observed)");
    Check(mb::HeapFreeCount() == frees_before + 1,
          "deallocation counter is wired up (the matching delete[] was observed)");
  }

  for (int k = 0; k < reg.count; ++k) {
    if (!reg[k].available) {
      continue;
    }
    const mb::AllocationScope scope;
    for (int i = 0; i < 8; ++i) {
      const mb::Timestamp t0 = mb::Now();
      reg[k].fn(t.iterations, &p, 0, p.height);
      prof.Push(mb::ElapsedNanos(t0, mb::Now()));
    }
    mb::Colourize(t, p.width, p.height);
    const mb::LatencyStats s = prof.Compute();
    mb::DoNotOptimize(s);

    char label[192];
    std::snprintf(label, sizeof(label),
                  "%-26s kernel + profiler + colourise allocated %llu times", reg[k].name,
                  static_cast<unsigned long long>(scope.allocations()));
    Check(scope.clean(), label);
  }

  // Regression test for a real false positive: AllocationScope originally read
  // process-wide counters, so allocations made by SDL's Cocoa/display thread
  // during the measured region were attributed to the kernel. The GUI reported
  // "12 allocations in measured region -- regression!" while the kernel in fact
  // allocated nothing. The scope is now per-thread; this test fails if it ever
  // goes back to process-wide counting.
  {
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> noise_allocs{0};

    std::thread noisy([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        // Deliberately heap-allocate on another thread, defeating any
        // small-object optimisation.
        auto* block = new std::uint64_t[64];
        mb::DoNotOptimize(block);
        delete[] block;
        noise_allocs.fetch_add(1, std::memory_order_relaxed);
      }
    });

    // Wait until the other thread is definitely allocating.
    while (noise_allocs.load(std::memory_order_relaxed) < 1000) {
    }

    const std::uint64_t process_before = mb::HeapAllocCount();
    std::uint64_t observed = 0;
    {
      const mb::AllocationScope scope;
      for (int i = 0; i < 4; ++i) {
        reg[0].fn(t.iterations, &p, 0, p.height);
      }
      observed = scope.allocations();
    }
    const std::uint64_t process_delta = mb::HeapAllocCount() - process_before;

    stop.store(true, std::memory_order_relaxed);
    noisy.join();

    char label[224];
    std::snprintf(label, sizeof(label),
                  "scope ignores another thread's allocations (this thread %llu, "
                  "process-wide %llu)",
                  static_cast<unsigned long long>(observed),
                  static_cast<unsigned long long>(process_delta));
    Check(observed == 0, label);
    Check(process_delta > 0,
          "process-wide counter still sees the other thread (proves the test is real)");
  }
}

// -------------------------------------------------------------------------
// 5. Performance grading
// -------------------------------------------------------------------------
void TestPerformanceGrade() {
  std::printf("\n=== Performance grade ===\n");

  // Monotonic: a faster rate must never grade lower than a slower one.
  bool monotonic = true;
  int last_tier = -1;
  for (double mips = 1.0; mips < 40000.0; mips *= 1.07) {
    const int tier = static_cast<int>(mb::GradeIterationRate(mips).tier);
    if (tier < last_tier) {
      monotonic = false;
      break;
    }
    last_tier = tier;
  }
  Check(monotonic, "grade never decreases as throughput increases");

  // Endpoints and degenerate inputs.
  Check(mb::GradeIterationRate(1.0).tier == mb::GradeTier::AbsoluteGarbage,
        "1 MIter/s grades as Absolute Garbage");
  Check(mb::GradeIterationRate(1e9).tier == mb::GradeTier::Godly,
        "absurdly high throughput grades as Godly");
  Check(mb::GradeIterationRate(0.0).meter == 0.0f, "no data yields an empty meter");
  Check(mb::GradeIterationRate(-5.0).meter == 0.0f, "negative input is handled");
  Check(mb::GradeIterationRate(std::nan("")).meter == 0.0f, "NaN input is handled");

  // Meter must stay in range and be monotonic too.
  bool meter_in_range = true;
  for (double mips = 0.5; mips < 100000.0; mips *= 1.5) {
    const float m = mb::GradeIterationRate(mips).meter;
    if (!(m >= 0.0f && m <= 1.0f)) {
      meter_in_range = false;
      break;
    }
  }
  Check(meter_in_range, "meter fill stays within [0, 1]");

  // Every tier must have a non-empty label and a distinct one.
  bool labels_ok = true;
  for (int i = 0; i < mb::kGradeTierCount; ++i) {
    const char* a = mb::GradeTierLabel(static_cast<mb::GradeTier>(i));
    if (a == nullptr || a[0] == '\0') {
      labels_ok = false;
      break;
    }
    for (int j = i + 1; j < mb::kGradeTierCount; ++j) {
      if (std::strcmp(a, mb::GradeTierLabel(static_cast<mb::GradeTier>(j))) == 0) {
        labels_ok = false;
        break;
      }
    }
  }
  Check(labels_ok, "all 14 tier labels are present and unique");

  // Every tier needs its own description, since the scale legend shows all of
  // them side by side and a duplicate or empty one would be visible.
  bool blurbs_ok = true;
  for (int i = 0; i < mb::kGradeTierCount; ++i) {
    const char* a = mb::GradeTierBlurb(static_cast<mb::GradeTier>(i));
    if (a == nullptr || a[0] == '\0') {
      blurbs_ok = false;
      break;
    }
    for (int j = i + 1; j < mb::kGradeTierCount; ++j) {
      if (std::strcmp(a, mb::GradeTierBlurb(static_cast<mb::GradeTier>(j))) == 0) {
        blurbs_ok = false;
        break;
      }
    }
  }
  Check(blurbs_ok, "all 14 tier descriptions are present and unique");
  Check(std::strcmp(mb::GradeTierLabel(mb::GradeTier::AbsoluteGarbage),
                    "Absolute Garbage") == 0,
        "bottom tier is 'Absolute Garbage'");
  Check(std::strcmp(mb::GradeTierLabel(mb::GradeTier::Godly), "Godly") == 0,
        "top tier is 'Godly'");

  // Tier ranges must tile the axis without gaps or overlaps.
  bool ranges_tile = true;
  for (int i = 0; i + 1 < mb::kGradeTierCount; ++i) {
    double lo_a = 0.0;
    double hi_a = 0.0;
    double lo_b = 0.0;
    double hi_b = 0.0;
    mb::GradeTierRange(static_cast<mb::GradeTier>(i), lo_a, hi_a);
    mb::GradeTierRange(static_cast<mb::GradeTier>(i + 1), lo_b, hi_b);
    if (hi_a != lo_b || !(lo_a < lo_b)) {
      ranges_tile = false;
      break;
    }
  }
  Check(ranges_tile, "tier ranges tile the axis with no gaps or overlaps");

  // The calibration claims in performance_grade.cpp: a real release-mode scalar
  // kernel should land in the low-middle of the scale, and a working SIMD kernel
  // clearly above it. If someone retunes the thresholds and breaks this, the
  // grade stops meaning what the comments say it means.
  const mb::GradeTier scalar_tier = mb::GradeIterationRate(570.0).tier;
  const mb::GradeTier simd_tier = mb::GradeIterationRate(2100.0).tier;
  const mb::GradeTier debug_tier = mb::GradeIterationRate(170.0).tier;
  Check(scalar_tier == mb::GradeTier::Poor,
        "measured release scalar rate (570) grades as Poor");
  Check(simd_tier == mb::GradeTier::VeryGood,
        "measured release NEON rate (2100) grades as Very Good");
  Check(debug_tier == mb::GradeTier::Potato,
        "measured -O0 debug rate (170) grades as Potato");
  Check(static_cast<int>(simd_tier) > static_cast<int>(scalar_tier) + 2,
        "SIMD grades clearly above scalar, not just one notch");

  // Secondary ratings.
  Check(std::strcmp(mb::RateConsistency(0.4).label, "Rock solid") == 0,
        "CV 0.4% is Rock solid");
  Check(std::strcmp(mb::RateConsistency(25.0).label, "Very noisy") == 0,
        "CV 25% is Very noisy");
  Check(std::strcmp(mb::RateSmoothness(120.0).label, "Buttery") == 0, "120 FPS is Buttery");
  Check(std::strcmp(mb::RateSmoothness(8.0).label, "Slideshow") == 0, "8 FPS is a Slideshow");
  Check(std::strcmp(mb::RateSmoothness(0.0).label, "unknown") == 0, "0 FPS is unknown");

  // Whole-machine context. The projection must be linear in core count and must
  // never silently claim to be a measurement.
  {
    const mb::MachineUtilisation u = mb::DescribeUtilisation(2174.0, 15, 1);
    Check(u.cores_used == 1, "utilisation reports one core in use (single-threaded)");
    Check(u.cores_total == 15, "utilisation reports the real core count");
    Check(std::fabs(u.percent_of_chip - 100.0 / 15.0) < 1e-9,
          "percent of chip is 1/cores");
    Check(u.projection_valid, "projection is available when a rate is known");
    Check(std::fabs(u.projected_all_core_mips - 2174.0 * 15.0) < 1e-6,
          "all-core projection is linear in core count");

    const mb::MachineUtilisation none = mb::DescribeUtilisation(0.0, 15, 1);
    Check(!none.projection_valid, "no rate means no projection is offered");

    // Degenerate core counts must not divide by zero or report 0 cores.
    const mb::MachineUtilisation single = mb::DescribeUtilisation(500.0, 0, 1);
    Check(single.cores_total == 1, "a reported core count of 0 is clamped to 1");
    Check(std::fabs(single.percent_of_chip - 100.0) < 1e-9,
          "one core means 100% of the chip");

    // With every core in use the measurement is the answer, so no projection
    // should be offered -- extrapolating would only add error.
    const mb::MachineUtilisation all = mb::DescribeUtilisation(30000.0, 15, 15);
    Check(all.using_whole_chip, "15 of 15 threads is reported as the whole chip");
    Check(!all.projection_valid, "no projection is offered once all cores are in use");
    Check(std::fabs(all.percent_of_chip - 100.0) < 1e-9, "all cores is 100%");

    // Partial parallelism must project from the per-core rate, not the total.
    const mb::MachineUtilisation half = mb::DescribeUtilisation(8000.0, 16, 4);
    Check(half.projection_valid, "partial parallelism still projects");
    Check(std::fabs(half.projected_all_core_mips - 32000.0) < 1e-6,
          "projection extrapolates from per-core rate (8000/4 * 16)");

    // Oversubscription must not report more than 100% of the chip.
    const mb::MachineUtilisation over = mb::DescribeUtilisation(100.0, 8, 32);
    Check(over.cores_used == 8, "thread count above core count is clamped for reporting");
    Check(std::fabs(over.percent_of_chip - 100.0) < 1e-9, "never exceeds 100% of chip");
  }
}

// -------------------------------------------------------------------------
// 6. Iteration accounting
// -------------------------------------------------------------------------
void TestIterationSum() {
  std::printf("\n=== Iteration accounting ===\n");

  mb::RenderTarget& t = mb::Target();

  // A view entirely inside the set: every pixel must hit the cap exactly, so
  // the total is analytically known.
  {
    const mb::KernelParams p = MakeParams(64, 32, 100, 0.0, 0.0, 0.0005);
    mandelbrot_scalar_cpp(t.iterations, &p, 0, p.height);
    const std::uint64_t expected =
        static_cast<std::uint64_t>(p.width) * p.height * p.max_iter;
    const std::uint64_t actual = mb::SumIterations(t, p.width, p.height);
    char label[160];
    std::snprintf(label, sizeof(label), "interior sum == w*h*max_iter (%llu)",
                  static_cast<unsigned long long>(expected));
    Check(actual == expected, label);
  }

  // Summing must respect the visible width, not the padded stride: a stale
  // padding column must not inflate the throughput figure.
  {
    const mb::KernelParams p = MakeParams(250, 40, 64, -0.6, 0.0, 3.2);
    mandelbrot_scalar_cpp(t.iterations, &p, 0, p.height);
    const std::uint64_t visible = mb::SumIterations(t, p.width, p.height);
    const std::uint64_t padded = mb::SumIterations(t, p.vec_width, p.height);
    Check(p.vec_width > p.width, "test view really is padded");
    Check(visible <= padded, "visible-region sum ignores the padding columns");
  }

  // Zero-size regions must not read anything or trap.
  Check(mb::SumIterations(t, 0, 0) == 0, "empty region sums to zero");
}

// -------------------------------------------------------------------------
// 7. Multi-threaded rendering
// -------------------------------------------------------------------------
// Parallelism that changes the output is not an optimisation, so the headline
// requirement is bit-exactness against the single-threaded result for every
// kernel at every thread count.
void TestThreadedRendering() {
  std::printf("\n=== Multi-core rendering ===\n");

  mb::RenderTarget& t = mb::Target();
  const mb::KernelRegistry& reg = mb::Kernels();
  mb::RenderThreadPool& pool = mb::Pool();

  const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
  const int max_threads = hw_threads > 1 ? hw_threads : 2;
  Check(pool.Start(max_threads), "pool starts");
  Check(pool.max_thread_count() >= 2, "pool reports at least 2 participants");
  std::printf("  pool participants: %d\n", pool.max_thread_count());

  // A view with strongly uneven per-row cost, which is where a naive static
  // split would both imbalance and (if the bands were wrong) corrupt output.
  const mb::KernelParams p = MakeParams(320, 180, 512, -0.6, 0.0, 3.2);

  for (int k = 0; k < reg.count; ++k) {
    const mb::KernelDesc& d = reg[k];
    if (!d.available) {
      continue;
    }

    // Single-threaded reference via the pool's fast path.
    std::memset(t.reference, 0xCD, sizeof(std::uint16_t) * mb::kIterationBufferElems);
    pool.RenderFrame(d.fn, t.reference, &p, 1);

    const int counts[] = {2, 3, 5, 8, pool.max_thread_count(),
                          pool.max_thread_count() + 7};
    for (int n : counts) {
      if (n < 2) {
        continue;
      }
      std::memset(t.iterations, 0xCD, sizeof(std::uint16_t) * mb::kIterationBufferElems);
      pool.RenderFrame(d.fn, t.iterations, &p, n);

      const mb::VerifyResult r =
          mb::CompareBuffers(t.iterations, t.reference, p.width, p.height, p.stride);

      char label[224];
      std::snprintf(label, sizeof(label),
                    "%-26s %2d threads -> bit-identical (%llu differ)", d.name, n,
                    static_cast<unsigned long long>(r.mismatched_pixels));
      Check(r.exact(), label);
    }
  }

  // Thread counts above the pool size must clamp, not overrun the tally array.
  pool.RenderFrame(reg[0].fn, t.iterations, &p, 10000);
  Check(pool.last_thread_count() <= pool.max_thread_count(),
        "excessive thread count is clamped to the pool size");

  // Every row must be claimed exactly once: the tallies have to sum to height.
  {
    pool.RenderFrame(reg[0].fn, t.iterations, &p, pool.max_thread_count());
    int total = 0;
    for (int i = 0; i < pool.last_thread_count(); ++i) {
      total += pool.rows_for_participant(i);
    }
    char label[160];
    std::snprintf(label, sizeof(label), "rows claimed sum to height (%d of %d)", total,
                  p.height);
    Check(total == p.height, label);

    int lo = 0;
    int hi = 0;
    pool.LastRowSpread(lo, hi);
    std::printf("  row spread across %d threads: %d-%d\n", pool.last_thread_count(), lo,
                hi);
    Check(hi >= lo, "row spread is well-ordered");
  }

  // Degenerate geometry must not hang or split.
  {
    mb::KernelParams tiny = MakeParams(64, 1, 64, -0.6, 0.0, 3.2);
    pool.RenderFrame(reg[0].fn, t.iterations, &tiny, 8);
    Check(pool.last_thread_count() == 1, "single-row image is not split across threads");

    mb::KernelParams empty = MakeParams(64, 0, 64, -0.6, 0.0, 3.2);
    pool.RenderFrame(reg[0].fn, t.iterations, &empty, 8);
    Check(true, "zero-height image does not hang");
  }

  // Null arguments must be rejected rather than dereferenced.
  pool.RenderFrame(nullptr, t.iterations, &p, 4);
  pool.RenderFrame(reg[0].fn, nullptr, &p, 4);
  pool.RenderFrame(reg[0].fn, t.iterations, nullptr, 4);
  Check(true, "null arguments are rejected without crashing");

  // Repeated frames must stay correct: this is what catches a generation-counter
  // or completion-counter race that only shows up after many cycles.
  {
    bool all_exact = true;
    for (int frame = 0; frame < 200 && all_exact; ++frame) {
      const int n = 2 + (frame % (pool.max_thread_count() - 1));
      std::memset(t.iterations, 0, sizeof(std::uint16_t) * mb::kIterationBufferElems);
      pool.RenderFrame(reg[0].fn, t.iterations, &p, n);
      const mb::VerifyResult r =
          mb::CompareBuffers(t.iterations, t.reference, p.width, p.height, p.stride);
      if (!r.exact()) {
        all_exact = false;
      }
    }
    Check(all_exact, "200 consecutive frames at varying thread counts stay exact");
  }

  // Speedup, reported rather than asserted: the actual figure depends on core
  // count, thermal headroom and what else the machine is doing, so a threshold
  // would be a flaky test. Correctness above is what is asserted.
  {
    const mb::KernelParams bench = MakeParams(640, 360, 512, -0.6, 0.0, 3.2);
    const int fastest = reg.count - 1;  // the assembly kernel
    if (reg[fastest].available) {
      std::printf("  %-14s %10s %10s %8s\n", "threads", "ms", "MIter/s", "speedup");
      double base_ms = 0.0;
      for (int n : {1, 2, 4, 8, pool.max_thread_count()}) {
        if (n > pool.max_thread_count()) {
          continue;
        }
        for (int i = 0; i < 3; ++i) {
          pool.RenderFrame(reg[fastest].fn, t.iterations, &bench, n);
        }
        mb::LatencyProfiler<64> prof;
        for (int i = 0; i < 20; ++i) {
          const mb::Timestamp t0 = mb::Now();
          pool.RenderFrame(reg[fastest].fn, t.iterations, &bench, n);
          prof.Push(mb::ElapsedNanos(t0, mb::Now()));
        }
        const mb::LatencyStats s = prof.Compute();
        const std::uint64_t iters = mb::SumIterations(t, bench.width, bench.height);
        const double ms = s.trimmed_mean_ns / 1e6;
        if (n == 1) {
          base_ms = ms;
        }
        std::printf("  %-14d %10.2f %10.0f %7.2fx\n", n, ms,
                    static_cast<double>(iters) / (s.trimmed_mean_ns / 1e9) / 1e6,
                    base_ms / ms);
      }
    }
  }

  pool.Stop();
  Check(!pool.started(), "pool stops cleanly");
}

// -------------------------------------------------------------------------
// 8. Timing / throughput
// -------------------------------------------------------------------------
void TestThroughput() {
  std::printf("\n=== Measured throughput (%s) ===\n", MB_ARCH_NAME);

  mb::RenderTarget& t = mb::Target();
  const mb::KernelRegistry& reg = mb::Kernels();
  const mb::KernelParams p = MakeParams(640, 360, 512, -0.6, 0.0, 3.2);

  constexpr int kWarmup = 8;
  constexpr int kSamples = 64;

  double scalar_trimmed = 0.0;

  std::printf("  %-28s %12s %12s %10s %9s %8s\n", "kernel", "trimmed us", "stddev us",
              "cv %", "Mpx/s", "speedup");

  for (int k = 0; k < reg.count; ++k) {
    const mb::KernelDesc& d = reg[k];
    if (!d.available) {
      std::printf("  %-28s  unavailable: %s\n", d.name, d.unavailable_reason);
      continue;
    }

    for (int i = 0; i < kWarmup; ++i) {
      d.fn(t.iterations, &p, 0, p.height);
    }

    mb::LatencyProfiler<256> prof;
    for (int i = 0; i < kSamples; ++i) {
      const mb::Timestamp t0 = mb::Now();
      d.fn(t.iterations, &p, 0, p.height);
      prof.Push(mb::ElapsedNanos(t0, mb::Now()));
    }

    const mb::LatencyStats s = prof.Compute();
    if (k == 0) {
      scalar_trimmed = s.trimmed_mean_ns;
    }
    const double pixels = static_cast<double>(p.width) * static_cast<double>(p.height);
    const double mpxs = pixels / s.trimmed_mean_ns * 1000.0;  // px/ns -> Mpx/s
    const double speedup = s.trimmed_mean_ns > 0.0 ? scalar_trimmed / s.trimmed_mean_ns : 0.0;

    std::printf("  %-28s %12.2f %12.2f %10.2f %9.1f %7.2fx\n", d.name,
                s.trimmed_mean_ns / 1000.0, s.trimmed_stddev_ns / 1000.0, s.cv_pct, mpxs,
                speedup);

    Check(s.clt_satisfied, "sample count satisfies the CLT threshold");
    Check(s.trimmed_mean_ns > 0.0, "trimmed mean is positive");
  }
}

}  // namespace

int main() {
  const mb::CpuFeatures& cpu = mb::DetectCpuFeatures();
  std::printf("MandelbrotBenchmark self-test\n");
  std::printf("  arch            : %s\n", MB_ARCH_NAME);
  std::printf("  native lanes    : %d\n", mb::kNativeVectorLanes);
  std::printf("  cache line      : %zu bytes\n", mb::kCacheLineBytes);
#if MB_ARCH_X86
  std::printf("  avx=%d avx2=%d fma=%d osxsave=%d ymm_state=%d\n", cpu.avx, cpu.avx2, cpu.fma,
              cpu.osxsave, cpu.ymm_state);
#else
  std::printf("  neon            : %d\n", cpu.neon);
#endif
  std::printf("  buffers         : %.1f MiB static\n",
              static_cast<double>(sizeof(mb::RenderTarget)) / (1024.0 * 1024.0));

  TestKernelAgreement();
  TestProfiler();
  TestColourize();
  TestNoHotPathAllocation();
  TestPerformanceGrade();
  TestIterationSum();
  TestThreadedRendering();
  TestThroughput();

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL CHECKS PASSED" : "FAILURES",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
