// alloc_tracker.cpp — global operator new/delete replacements that count traffic.
//
// All overloads are replaced, including the sized and aligned C++17 forms. Any
// gap would let an allocation slip past the counter and quietly weaken the
// tripwire, so the full set is here even though the render loop should trigger
// none of them.
//
// Memory always comes from malloc / posix_memalign and is always released with
// free, so the replacements stay interchangeable with the platform allocator
// used by SDL and ImGui.

#include "alloc_tracker.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

#include "arch.hpp"

namespace mb {
namespace {

// Cache-line isolated: the counters are written on every allocation and we do
// not want them sharing a line with anything else the allocator touches.
struct alignas(MB_CACHE_ALIGN) Counters {
  std::atomic<std::uint64_t> allocs{0};
  std::atomic<std::uint64_t> frees{0};
  std::atomic<std::uint64_t> bytes{0};
};

Counters& Get() noexcept {
  // Function-local static with constant initialisation: usable from the very
  // first allocation, including any that happen before main().
  static Counters counters;
  return counters;
}

// Per-thread counts, so a scoped measurement on the render thread is not
// polluted by SDL's Cocoa/display thread. These are plain integers with
// constant initialisation: no dynamic init, no guard variable, and critically
// no allocation on first touch — which would recurse into operator new.
thread_local std::uint64_t tls_allocs = 0;
thread_local std::uint64_t tls_frees = 0;

void CountAllocation(std::size_t size) noexcept {
  Counters& c = Get();
  c.allocs.fetch_add(1, std::memory_order_relaxed);
  c.bytes.fetch_add(size, std::memory_order_relaxed);
  ++tls_allocs;
}

void* Allocate(std::size_t size) {
  CountAllocation(size);
  // malloc(0) may return nullptr, which operator new must not do.
  return std::malloc(size != 0 ? size : 1);
}

void* AllocateAligned(std::size_t size, std::size_t alignment) {
  CountAllocation(size);

  // posix_memalign requires a power-of-two multiple of sizeof(void*).
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }
  void* p = nullptr;
  if (posix_memalign(&p, alignment, size != 0 ? size : 1) != 0) {
    return nullptr;
  }
  return p;
}

void Release(void* p) noexcept {
  if (p == nullptr) {
    return;
  }
  Get().frees.fetch_add(1, std::memory_order_relaxed);
  ++tls_frees;
  std::free(p);
}

}  // namespace

std::uint64_t HeapAllocCount() noexcept {
  return Get().allocs.load(std::memory_order_relaxed);
}

std::uint64_t HeapFreeCount() noexcept {
  return Get().frees.load(std::memory_order_relaxed);
}

std::uint64_t HeapBytesRequested() noexcept {
  return Get().bytes.load(std::memory_order_relaxed);
}

std::uint64_t ThreadAllocCount() noexcept { return tls_allocs; }

std::uint64_t ThreadFreeCount() noexcept { return tls_frees; }

}  // namespace mb

// ---------------------------------------------------------------------------
// Replaceable global allocation functions
// ---------------------------------------------------------------------------

void* operator new(std::size_t size) {
  void* p = mb::Allocate(size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void* operator new[](std::size_t size) {
  void* p = mb::Allocate(size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return mb::Allocate(size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return mb::Allocate(size);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  void* p = mb::AllocateAligned(size, static_cast<std::size_t>(alignment));
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  void* p = mb::AllocateAligned(size, static_cast<std::size_t>(alignment));
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void* operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  return mb::AllocateAligned(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  return mb::AllocateAligned(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* p) noexcept { mb::Release(p); }
void operator delete[](void* p) noexcept { mb::Release(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { mb::Release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { mb::Release(p); }
void operator delete(void* p, std::size_t) noexcept { mb::Release(p); }
void operator delete[](void* p, std::size_t) noexcept { mb::Release(p); }
void operator delete(void* p, std::align_val_t) noexcept { mb::Release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { mb::Release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { mb::Release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { mb::Release(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
  mb::Release(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
  mb::Release(p);
}
