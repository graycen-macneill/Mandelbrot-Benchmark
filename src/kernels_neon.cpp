// kernels_neon.cpp — Kernel B': NEON compiler intrinsics, 4 pixels per inner step.
//
// The arm64 sibling of kernels_avx2.cpp. Structurally line-for-line identical,
// at half the vector width (128-bit / 4 lanes instead of 256-bit / 8 lanes), so
// the intrinsics-vs-assembly comparison isolates codegen quality on both
// architectures rather than comparing two different algorithms.
//
// NEON (ASIMD) is architecturally mandatory in AArch64, so unlike AVX2 there is
// no runtime feature gate to clear.

#include "arch.hpp"

#if MB_ARCH_ARM

#include <arm_neon.h>

#include <cstdint>

#include "kernel_api.hpp"

// Under -ffast-math clang contracts an explicit vmulq_f32 + vaddq_f32 pair into
// a single FMLA — being written as intrinsics does not protect them. That
// changes the rounding of both the coordinate setup and |z|^2, and made this
// kernel disagree with the scalar reference on ~15,000 pixels of a boundary
// zoom before it was fixed.
//
// Reassociation is the second, independent hazard: -ffast-math lets the
// compiler rewrite cx0 + (x + lane) * dx into (cx0 + x*dx) + lane*dx, hoisting
// the invariant part out of the loop. Algebraically identical, different
// rounding — measured at 14,925 mismatched pixels on a boundary zoom.
//
// These pragmas state the intent but do not by themselves suppress either
// transform; the build passes -ffp-contract=off -fno-associative-math, which
// does. Both are kept so a build that forgets the flags is documented as wrong.
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#elif defined(__GNUC__)
#pragma GCC optimize("no-fast-math", "no-associative-math")
#endif

extern "C" void MB_HOT mandelbrot_neon_intrinsics(std::uint16_t* out, const mb::KernelParams* p,
                                                  std::int32_t row_begin, std::int32_t row_end) {
  const float32x4_t v_cx0 = vdupq_n_f32(p->cx0);
  const float32x4_t v_cy0 = vdupq_n_f32(p->cy0);
  const float32x4_t v_dx = vdupq_n_f32(p->dx);
  const float32x4_t v_dy = vdupq_n_f32(p->dy);
  const float32x4_t v_escape = vdupq_n_f32(p->escape_r2);

  static const float kLaneInit[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  const float32x4_t v_lanes = vld1q_f32(kLaneInit);

  const std::int32_t vec_width = p->vec_width;
  const std::int32_t max_iter = p->max_iter;
  const std::ptrdiff_t stride = p->stride;

  for (std::int32_t y = row_begin; y < row_end; ++y) {
    const float32x4_t v_y = vdupq_n_f32(static_cast<float>(y));
    const float32x4_t v_cy = vaddq_f32(vmulq_f32(v_y, v_dy), v_cy0);

    std::uint16_t* row = out + static_cast<std::ptrdiff_t>(y) * stride;

    for (std::int32_t x = 0; x < vec_width; x += 4) {
      // (x + lane) * dx + cx0 — same rounding order as the scalar reference.
      const float32x4_t v_px = vaddq_f32(vdupq_n_f32(static_cast<float>(x)), v_lanes);
      const float32x4_t v_cx = vaddq_f32(vmulq_f32(v_px, v_dx), v_cx0);

      float32x4_t zx = vdupq_n_f32(0.0f);
      float32x4_t zy = vdupq_n_f32(0.0f);
      int32x4_t iter = vdupq_n_s32(0);

      for (std::int32_t n = 0; n < max_iter; ++n) {
        const float32x4_t zx2 = vmulq_f32(zx, zx);
        const float32x4_t zy2 = vmulq_f32(zy, zy);
        const float32x4_t mag = vaddq_f32(zx2, zy2);

        // vcleq_f32 is a quiet compare: escaped lanes holding inf/NaN yield
        // false and latch, exactly like _CMP_LE_OQ on x86.
        const uint32x4_t mask = vcleq_f32(mag, v_escape);
        if (vmaxvq_u32(mask) == 0) {
          break;  // every lane in this batch has escaped
        }

        // mask lanes are 0 or 0xFFFFFFFF (== -1 reinterpreted signed).
        iter = vsubq_s32(iter, vreinterpretq_s32_u32(mask));

        float32x4_t t = vmulq_f32(zx, zy);
        t = vaddq_f32(t, t);
        zy = vaddq_f32(t, v_cy);
        zx = vaddq_f32(vsubq_f32(zx2, zy2), v_cx);
      }

      // Narrow 4 x int32 -> 4 x uint16. Counts are <= max_iter <= 65535, so
      // the truncating narrow is lossless.
      vst1_u16(row + x, vmovn_u32(vreinterpretq_u32_s32(iter)));
    }
  }
}

#endif  // MB_ARCH_ARM
