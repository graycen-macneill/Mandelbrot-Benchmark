// alloc_tracker.hpp — a tripwire for the "zero heap allocation on hot paths"
// constraint.
//
// The constraint is easy to state and easy to violate by accident (a stray
// std::string, a std::function, a container that grows). Rather than asserting
// compliance in a comment, this counts every global operator new / delete in
// the process and lets the render loop measure the delta across the measured
// region. The dashboard shows that delta live, so a regression is visible
// immediately instead of at the next profiling session.
//
// Counting is unconditional but costs one relaxed atomic increment per
// allocation, which never occurs inside the measured region by construction.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mb {

// --- Process-wide totals (all threads) -------------------------------------
// Monotonic count of global operator new calls since process start.
std::uint64_t HeapAllocCount() noexcept;
// Monotonic count of global operator delete calls since process start.
std::uint64_t HeapFreeCount() noexcept;
// Total bytes requested through global operator new since process start.
std::uint64_t HeapBytesRequested() noexcept;

// --- Per-thread counts -----------------------------------------------------
// Same events, attributed to the calling thread only.
std::uint64_t ThreadAllocCount() noexcept;
std::uint64_t ThreadFreeCount() noexcept;

// Scoped delta measurement for the *calling thread*.
//
// This deliberately uses the per-thread counters, not the process-wide ones.
// With process-wide counting the scope also caught allocations made by SDL's
// Cocoa/display thread while the kernel was running, which reported a false
// "regression" of ~12 allocations per frame even though the kernel itself
// allocates nothing (the single-threaded self-test measured exactly 0 for the
// same kernels). Attributing another thread's activity to the measured region
// makes the tripwire useless, so the scope is now thread-scoped and the
// process-wide totals remain available separately for diagnostics.
class AllocationScope {
 public:
  AllocationScope() noexcept
      : allocs_at_entry_(ThreadAllocCount()), frees_at_entry_(ThreadFreeCount()) {}

  std::uint64_t allocations() const noexcept { return ThreadAllocCount() - allocs_at_entry_; }
  std::uint64_t frees() const noexcept { return ThreadFreeCount() - frees_at_entry_; }
  bool clean() const noexcept { return allocations() == 0; }

 private:
  std::uint64_t allocs_at_entry_;
  std::uint64_t frees_at_entry_;
};

}  // namespace mb
