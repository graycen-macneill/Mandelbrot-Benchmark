// kernel_api.hpp — the single ABI contract shared by C++, intrinsics and assembly.
//
// Every kernel — scalar C++, AVX2 intrinsics, NASM, NEON intrinsics, AArch64 asm —
// implements exactly this signature:
//
//     void kernel(uint16_t* out, const KernelParams* p, int32_t row_begin, int32_t row_end);
//
// Under the System V AMD64 ABI that lands as:  rdi=out, rsi=p, edx=row_begin, ecx=row_end
// Under AAPCS64 as:                            x0 =out, x1 =p, w2 =row_begin, w3 =row_end
//
// The [row_begin, row_end) split is how RenderThreadPool hands the same kernels
// disjoint horizontal bands across threads, with no signature change. Bands never
// overlap, so no kernel needs any synchronisation on the output buffer.
#pragma once

#include <cstddef>
#include <cstdint>

#include "arch.hpp"

namespace mb {

// ---------------------------------------------------------------------------
// KernelParams
// ---------------------------------------------------------------------------
// Exactly one cache line, POD, trivially copyable. The assembly kernels index
// this by hard-coded byte offset, so the static_asserts below are load-bearing:
// they are the only thing keeping the .asm/.S files honest if a field moves.
struct alignas(64) KernelParams {
  float cx0;         // +0   complex-plane real part at pixel x == 0
  float cy0;         // +4   complex-plane imag part at pixel y == 0
  float dx;          // +8   real-axis step per pixel
  float dy;          // +12  imag-axis step per pixel
  float escape_r2;   // +16  escape radius squared (4.0f)
  int32_t vec_width; // +20  padded width, always a multiple of kMaxVectorLanes
  int32_t height;    // +24  image height in pixels
  int32_t max_iter;  // +28  iteration cap, <= 65535 so results fit in uint16
  int32_t stride;    // +32  uint16 elements between consecutive rows
  int32_t width;     // +36  visible width (<= vec_width); kernels ignore this
  int32_t reserved[6]; // +40..+63
};

static_assert(sizeof(KernelParams) == 64, "assembly kernels assume a 64-byte KernelParams");
static_assert(alignof(KernelParams) >= 64, "KernelParams must be cache-line aligned");
static_assert(offsetof(KernelParams, cx0) == 0, "asm offset P_CX0");
static_assert(offsetof(KernelParams, cy0) == 4, "asm offset P_CY0");
static_assert(offsetof(KernelParams, dx) == 8, "asm offset P_DX");
static_assert(offsetof(KernelParams, dy) == 12, "asm offset P_DY");
static_assert(offsetof(KernelParams, escape_r2) == 16, "asm offset P_ESCAPE_R2");
static_assert(offsetof(KernelParams, vec_width) == 20, "asm offset P_VEC_WIDTH");
static_assert(offsetof(KernelParams, height) == 24, "asm offset P_HEIGHT");
static_assert(offsetof(KernelParams, max_iter) == 28, "asm offset P_MAX_ITER");
static_assert(offsetof(KernelParams, stride) == 32, "asm offset P_STRIDE");
static_assert(offsetof(KernelParams, width) == 36, "asm offset P_WIDTH");

// Iteration counts are stored as uint16, so this is the hard ceiling.
constexpr int32_t kMaxIterationCap = 65535;

using KernelFn = void (*)(uint16_t* out, const KernelParams* p, int32_t row_begin,
                          int32_t row_end);

}  // namespace mb

// ---------------------------------------------------------------------------
// Kernel entry points
// ---------------------------------------------------------------------------
// extern "C" throughout: the assembly kernels have no notion of C++ mangling,
// and keeping the C++ kernels on the same linkage makes the registry uniform.
extern "C" {

// Kernel A — portable scalar C++ reference. Defines the bit-exact ground truth.
void mandelbrot_scalar_cpp(std::uint16_t* out, const mb::KernelParams* p,
                           std::int32_t row_begin, std::int32_t row_end);

#if MB_ARCH_X86
// Kernel B — AVX2 compiler intrinsics, 8 pixels per inner step.
void mandelbrot_avx2_intrinsics(std::uint16_t* out, const mb::KernelParams* p,
                                std::int32_t row_begin, std::int32_t row_end);

// Kernel C — hand-rolled NASM, System V AMD64 ABI, no stack frame.
void mandelbrot_avx2_nasm(std::uint16_t* out, const mb::KernelParams* p,
                          std::int32_t row_begin, std::int32_t row_end);
#endif

#if MB_ARCH_ARM
// Kernel B' — NEON compiler intrinsics, 4 pixels per inner step.
void mandelbrot_neon_intrinsics(std::uint16_t* out, const mb::KernelParams* p,
                                std::int32_t row_begin, std::int32_t row_end);

// Kernel C' — hand-rolled AArch64 asm, AAPCS64, leaf function, no stack frame.
void mandelbrot_neon_asm(std::uint16_t* out, const mb::KernelParams* p,
                         std::int32_t row_begin, std::int32_t row_end);
#endif

}  // extern "C"
