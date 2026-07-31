// timing.hpp — nanosecond timing primitives for the profiler.
//
// The spec calls for std::chrono::high_resolution_clock, which is used here
// directly. One caveat worth surfacing rather than hiding: high_resolution_clock
// is permitted to be an alias for system_clock, which is *not* monotonic and can
// step backwards when NTP adjusts the wall clock. On libc++ (macOS) and on
// libstdc++ it aliases steady_clock, so this is fine in practice — but the
// dashboard reports is_steady so a reader can tell whether the numbers are
// trustworthy on their platform instead of assuming.
#pragma once

#include <chrono>
#include <cstdint>

namespace mb {

using Clock = std::chrono::high_resolution_clock;
using Timestamp = Clock::time_point;

// True when the clock is guaranteed monotonic. Surfaced in the UI.
constexpr bool kClockIsSteady = Clock::is_steady;

// Tick period of the clock type, in nanoseconds. This is the *representable*
// resolution, not necessarily the achievable one.
constexpr double kClockTickNanos =
    static_cast<double>(Clock::period::num) * 1e9 / static_cast<double>(Clock::period::den);

inline Timestamp Now() noexcept { return Clock::now(); }

inline double ElapsedNanos(Timestamp begin, Timestamp end) noexcept {
  return std::chrono::duration<double, std::nano>(end - begin).count();
}

// Empirically smallest non-zero delta the clock reports, i.e. the real
// granularity including the cost of the call itself. Measured once, lazily.
double MeasuredClockGranularityNanos();

// Keeps the optimiser from discarding a computation whose result is unused.
// The kernels all write through a pointer to external memory so none of them
// are actually at risk, but benchmark scaffolding should never rely on that.
//
// Two overloads: the mutable one marks the value as read *and written* (so the
// compiler cannot assume anything about it afterwards), the const one only marks
// it as read. Without the const overload, `DoNotOptimize(some_const_ref)` fails
// to compile with "invalid lvalue in asm output".
template <typename T>
inline void DoNotOptimize(T& value) noexcept {
#if defined(__clang__) || defined(__GNUC__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  volatile T sink = value;
  (void)sink;
#endif
}

template <typename T>
inline void DoNotOptimize(const T& value) noexcept {
#if defined(__clang__) || defined(__GNUC__)
  asm volatile("" : : "r,m"(value) : "memory");
#else
  volatile T sink = value;
  (void)sink;
#endif
}

inline void CompilerBarrier() noexcept {
#if defined(__clang__) || defined(__GNUC__)
  asm volatile("" : : : "memory");
#endif
}

}  // namespace mb
