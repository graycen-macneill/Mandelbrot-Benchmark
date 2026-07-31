// thread_pool.cpp — implementation of multi-core frame rendering.

#include "thread_pool.hpp"

namespace mb {
namespace {

// How long a worker spins on the generation counter before parking. Sized so a
// back-to-back frame is caught by the spin (cheap) while a genuinely idle app
// still parks and drops to zero CPU.
constexpr int kSpinRelaxIterations = 4096;

}  // namespace

RenderThreadPool::~RenderThreadPool() { Stop(); }

bool RenderThreadPool::Start(int max_thread_count) {
  if (started_) {
    return false;
  }
  if (max_thread_count < 1) {
    max_thread_count = 1;
  }
  if (max_thread_count > kMaxRenderThreads) {
    max_thread_count = kMaxRenderThreads;
  }

  worker_count_ = max_thread_count - 1;  // the caller is the other participant
  stop_.store(false, std::memory_order_relaxed);
  generation_.store(0, std::memory_order_relaxed);

  for (int i = 0; i < worker_count_; ++i) {
    threads_[i] = std::thread(&RenderThreadPool::WorkerLoop, this, i);
  }
  started_ = true;
  return true;
}

void RenderThreadPool::Stop() {
  if (!started_) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_.store(true, std::memory_order_release);
    // Bump the generation so a worker parked on the predicate re-evaluates.
    generation_.fetch_add(1, std::memory_order_release);
  }
  wake_cv_.notify_all();

  for (int i = 0; i < worker_count_; ++i) {
    if (threads_[i].joinable()) {
      threads_[i].join();
    }
  }
  worker_count_ = 0;
  started_ = false;
}

// Pull rows until the frame is exhausted. One row at a time: a row is thousands
// of iterations of work, so the atomic increment is far too cheap to be worth
// batching, and single-row granularity gives the best possible balance.
void RenderThreadPool::DrainRows(int participant) {
  const Job job = job_;  // stable for the duration of a generation
  int rows_done = 0;

  for (;;) {
    const int y = next_row_.fetch_add(1, std::memory_order_relaxed);
    if (y >= job.height) {
      break;
    }
    job.fn(job.out, job.params, y, y + 1);
    ++rows_done;
  }

  if (participant >= 0 && participant < kMaxRenderThreads) {
    row_tally_[participant].rows.store(rows_done, std::memory_order_relaxed);
  }
}

void RenderThreadPool::WorkerLoop(int worker_index) {
  const int participant = worker_index + 1;  // participant 0 is the caller
  std::uint64_t seen = generation_.load(std::memory_order_acquire);

  for (;;) {
    bool have_work = false;

    // Phase 1: spin. Catches the common case where the next frame is already
    // queued by the time this worker finishes the previous one.
    for (int spin = 0; spin < kSpinRelaxIterations; ++spin) {
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
      if (generation_.load(std::memory_order_acquire) != seen) {
        have_work = true;
        break;
      }
      CpuRelax();
    }

    // Phase 2: park. Costs a futex round trip but uses no CPU while idle.
    if (!have_work) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_cv_.wait(lock, [&] {
        return stop_.load(std::memory_order_acquire) ||
               generation_.load(std::memory_order_acquire) != seen;
      });
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
    }

    // Acquire here pairs with the release store in RenderFrame, making the job_
    // fields written before it visible.
    seen = generation_.load(std::memory_order_acquire);

    // Not needed for this frame: the caller asked for fewer threads than exist.
    if (participant >= job_.thread_count) {
      continue;
    }

    DrainRows(participant);

    // The counter is bumped while holding the mutex so the waiting caller cannot
    // be between evaluating its predicate and sleeping when this happens.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      workers_done_.fetch_add(1, std::memory_order_release);
    }
    done_cv_.notify_one();
  }
}

void RenderThreadPool::RenderFrame(KernelFn fn, std::uint16_t* out,
                                   const KernelParams* params, int thread_count) {
  if (fn == nullptr || out == nullptr || params == nullptr) {
    return;
  }
  const int height = params->height;

  if (thread_count < 1) {
    thread_count = 1;
  }
  if (thread_count > max_thread_count()) {
    thread_count = max_thread_count();
  }
  if (height <= 1) {
    thread_count = 1;  // nothing to split
  }
  last_thread_count_ = thread_count;

  for (int i = 0; i < thread_count && i < kMaxRenderThreads; ++i) {
    row_tally_[i].rows.store(0, std::memory_order_relaxed);
  }

  // Single-threaded fast path: no atomics, no wakeups, no mutex. One direct call,
  // exactly as if this pool did not exist. This keeps thread_count == 1 a true
  // baseline rather than "one thread plus synchronisation overhead".
  if (thread_count == 1 || !started_) {
    fn(out, params, 0, height);
    if (height > 0) {
      row_tally_[0].rows.store(height, std::memory_order_relaxed);
    }
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    job_.fn = fn;
    job_.out = out;
    job_.params = params;
    job_.height = height;
    job_.thread_count = thread_count;
    next_row_.store(0, std::memory_order_relaxed);
    workers_done_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
  }
  wake_cv_.notify_all();

  // The caller is participant 0 and pulls rows like everyone else, so it is never
  // just idling while the workers do the work.
  DrainRows(0);

  const int expected_workers = thread_count - 1;
  std::unique_lock<std::mutex> lock(mutex_);
  done_cv_.wait(lock, [&] {
    return workers_done_.load(std::memory_order_acquire) >= expected_workers;
  });
}

int RenderThreadPool::rows_for_participant(int index) const {
  if (index < 0 || index >= kMaxRenderThreads) {
    return 0;
  }
  return row_tally_[index].rows.load(std::memory_order_relaxed);
}

void RenderThreadPool::LastRowSpread(int& min_rows, int& max_rows) const {
  min_rows = 0;
  max_rows = 0;
  const int n = last_thread_count_;
  if (n <= 0) {
    return;
  }
  min_rows = rows_for_participant(0);
  max_rows = min_rows;
  for (int i = 1; i < n && i < kMaxRenderThreads; ++i) {
    const int r = rows_for_participant(i);
    if (r < min_rows) {
      min_rows = r;
    }
    if (r > max_rows) {
      max_rows = r;
    }
  }
}

RenderThreadPool& Pool() {
  static RenderThreadPool instance;
  return instance;
}

}  // namespace mb
