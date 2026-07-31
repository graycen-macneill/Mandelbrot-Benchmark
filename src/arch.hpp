// arch.hpp — compile-time architecture / ISA detection and cache-line constants.
//
// The benchmark suite ships two sibling SIMD families:
//   x86-64 : AVX2 intrinsics + hand-rolled NASM  (8 lanes / 256-bit)
//   arm64  : NEON intrinsics + hand-rolled AArch64 asm (4 lanes / 128-bit)
//
// Exactly one family is compiled in; the scalar kernel is always present.
#pragma once

#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#define MB_ARCH_X86 1
#define MB_ARCH_ARM 0
#define MB_ARCH_NAME "x86-64"
#elif defined(__aarch64__) || defined(_M_ARM64)
#define MB_ARCH_X86 0
#define MB_ARCH_ARM 1
#define MB_ARCH_NAME "arm64 (AArch64)"
#else
#define MB_ARCH_X86 0
#define MB_ARCH_ARM 0
#define MB_ARCH_NAME "unknown"
#endif

namespace mb {

// The spec mandates alignas(64). Apple Silicon actually reports a 128-byte
// cache line (hw.cachelinesize == 128), so aligning to 64 there would still
// allow two "isolated" objects to share one physical line and false-share.
// We align to the true destructive-interference size, which is >= 64 on every
// target and therefore always satisfies the 64-byte requirement.
#if MB_ARCH_ARM
constexpr std::size_t kCacheLineBytes = 128;
#else
constexpr std::size_t kCacheLineBytes = 64;
#endif

// Widest SIMD batch any compiled kernel consumes per step. Row widths are
// padded up to this so no kernel needs a scalar tail loop.
constexpr int kMaxVectorLanes = 8;

// Lanes actually processed per inner-loop step by this build's SIMD kernels.
#if MB_ARCH_X86
constexpr int kNativeVectorLanes = 8;
#elif MB_ARCH_ARM
constexpr int kNativeVectorLanes = 4;
#else
constexpr int kNativeVectorLanes = 1;
#endif

}  // namespace mb

// MB_CACHE_ALIGN — usable in alignas() where a constexpr variable is awkward.
#if MB_ARCH_ARM
#define MB_CACHE_ALIGN 128
#else
#define MB_CACHE_ALIGN 64
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MB_NOINLINE __attribute__((noinline))
#define MB_HOT __attribute__((hot))
#else
#define MB_NOINLINE
#define MB_HOT
#endif
