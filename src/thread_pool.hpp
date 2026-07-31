// thread_pool.hpp — multi-core frame rendering.
//
// This is what the `[row_begin, row_end)` parameters on every kernel were for.
// Row bands are disjoint, so threads never write the same memory and no locking
// is needed on the output buffer.
//
// Design notes, in rough order of how much they matter:
//
//  * Dynamic row stealing, not static bands. Mandelbrot rows differ enormously
//    in cost: a row through the middle of the set runs to max_iter on most
//    pixels, while a row above it escapes almost immediately. Splitting the
//    image into N equal contiguous bands therefore leaves most threads finished
//    and idle while one grinds through the centre. Instead workers pull one row
//    at a time from a shared atomic counter, which self-balances regardless of
//    where the expensive rows are — and also absorbs heterogeneous cores, where
//    a thread on a slower core simply completes fewer rows.
//
//  * The calling thread participates. For N threads we wake N-1 workers and the
//    caller takes rows too. This makes thread_count == 1 genuinely free: no
//    wakeups, no atomics, no synchronisation at all, just a direct kernel call.
//    Single-threaded numbers stay bit-identical and cost-identical to having no
//    pool, which matters because every prior measurement was taken that way.
//
//  * No allocation per frame. Threads are spawned once, park between frames and
//    are woken by generation counter. The job descriptor and all counters are
//    fixed-size members. Nothing here touches the heap after Start().
//
//  * Spin, then park. In continuous rendering the next frame arrives almost
//    immediately, and a condition-variable round trip costs more than a short
//    spin. Workers spin on the generation counter first and only park if no work
//    appears, so an idle or paused app still drops to zero CPU.
//
//  * Cache-line isolation. The atomic row counter, the completion counter and
//    each thread's row tally are separately aligned. These are precisely the
//    "thread-shared variables" the alignment requirement is about: without it
//    the row counter and the done counter would share a line and every steal
//    would invalidate it on every core.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

#include "arch.hpp"
#include "kernel_api.hpp"

namespace mb {

constexpr int kMaxRenderThreads = 64;

// Hint to the CPU that we are in a spin-wait loop.
inline void CpuRelax() noexcept {
#if MB_ARCH_X86
  __builtin_ia32_pause();
#elif MB_ARCH_ARM
  __asm__ __volatile__("yield" ::: "memory");
#else
  // Nothing portable to emit; the compiler barrier below is enough to stop the
  // loop being hoisted.
  __asm__ __volatile__("" ::: "memory");
#endif
}

class RenderThreadPool {
 public:
  RenderThreadPool() = default;
  ~RenderThreadPool();

  RenderThreadPool(const RenderThreadPool&) = delete;
  RenderThreadPool& operator=(const RenderThreadPool&) = delete;

  // Spawns (max_thread_count - 1) workers. The caller is the remaining
  // participant. Returns false if already started or the count is invalid.
  bool Start(int max_thread_count);
  void Stop();

  bool started() const { return started_; }
  int max_thread_count() const { return worker_count_ + 1; }

  // Renders rows [0, params->height) with `thread_count` participants total,
  // including the calling thread. Blocks until the whole frame is complete.
  //
  // thread_count is clamped to [1, max_thread_count()].
  void RenderFrame(KernelFn fn, std::uint16_t* out, const KernelParams* params,
                   int thread_count);

  // --- load-balance diagnostics for the last frame -------------------------
  int last_thread_count() const { return last_thread_count_; }
  int rows_for_participant(int index) const;
  // Fewest and most rows any participant handled last frame. A large spread on a
  // static split would indicate imbalance; with row stealing these stay close.
  void LastRowSpread(int& min_rows, int& max_rows) const;

 private:
  void WorkerLoop(int worker_index);
  void DrainRows(int participant);

  struct alignas(MB_CACHE_ALIGN) Job {
    KernelFn fn = nullptr;
    std::uint16_t* out = nullptr;
    const KernelParams* params = nullptr;
    int height = 0;
    int thread_count = 1;
  };

  // Each of these is written by every core; they must not share a cache line.
  alignas(MB_CACHE_ALIGN) Job job_{};
  alignas(MB_CACHE_ALIGN) std::atomic<int> next_row_{0};
  alignas(MB_CACHE_ALIGN) std::atomic<int> workers_done_{0};
  alignas(MB_CACHE_ALIGN) std::atomic<std::uint64_t> generation_{0};
  alignas(MB_CACHE_ALIGN) std::atomic<bool> stop_{false};

  struct alignas(MB_CACHE_ALIGN) RowTally {
    std::atomic<int> rows{0};
  };
  RowTally row_tally_[kMaxRenderThreads]{};

  std::thread threads_[kMaxRenderThreads]{};
  int worker_count_ = 0;
  bool started_ = false;
  int last_thread_count_ = 1;

  std::mutex mutex_;
  std::condition_variable wake_cv_;
  std::condition_variable done_cv_;
};

// Process-wide pool, started once from main().
RenderThreadPool& Pool();

}  // namespace mb
