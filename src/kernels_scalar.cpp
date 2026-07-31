// kernels_scalar.cpp — Kernel A: portable single-precision scalar reference.
//
// This kernel defines the ground truth. The SIMD and assembly kernels are
// required to reproduce its output bit-for-bit, which is what makes the
// "Verify" pass in the UI meaningful rather than decorative.
//
// Bit-exactness requires three things to match across all kernels:
//
//   1. Coordinate synthesis. c is always  (cx0 + (float)x * dx,  cy0 + (float)y * dy),
//      computed as one multiply then one add, never fused and never strength-
//      reduced into a running accumulator (which would drift).
//   2. Iteration algebra. zx' = (zx*zx - zy*zy) + cx and zy' = (2*zx*zy) + cy,
//      in that operation order.
//   3. Escape test. The count is the number of steps whose *entry* magnitude
//      satisfies zx^2 + zy^2 <= escape_r2, capped at max_iter.
//
// Note on -ffast-math (mandated by the spec): it implies -ffp-contract=fast,
// which breaks (2) by fusing the multiply and add of |z|^2 into a single
// fmadd/fmla. The pragmas below express the intent, but they are NOT sufficient
// on their own — contraction is also a backend decision driven by the
// function-level fast-math attributes, and clang still emitted 2 fmadd
// instructions here with the pragmas in place. The build therefore passes
// -ffp-contract=off explicitly; see CMakeLists.txt. Without that flag this
// kernel disagrees with the hand-written assembly on hundreds of boundary
// pixels, which is exactly the bug the self-test caught.

#include <cstdint>

#include "arch.hpp"
#include "kernel_api.hpp"

#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#elif defined(__GNUC__)
#pragma GCC optimize("no-fast-math", "no-associative-math")
#endif

extern "C" void MB_HOT mandelbrot_scalar_cpp(std::uint16_t* out, const mb::KernelParams* p,
                                             std::int32_t row_begin, std::int32_t row_end) {
  const float cx0 = p->cx0;
  const float cy0 = p->cy0;
  const float dx = p->dx;
  const float dy = p->dy;
  const float escape_r2 = p->escape_r2;
  const std::int32_t vec_width = p->vec_width;
  const std::int32_t max_iter = p->max_iter;
  const std::ptrdiff_t stride = p->stride;

  for (std::int32_t y = row_begin; y < row_end; ++y) {
    const float cy = cy0 + static_cast<float>(y) * dy;
    std::uint16_t* row = out + static_cast<std::ptrdiff_t>(y) * stride;

    for (std::int32_t x = 0; x < vec_width; ++x) {
      const float cx = cx0 + static_cast<float>(x) * dx;

      float zx = 0.0f;
      float zy = 0.0f;
      std::int32_t iter = 0;

      while (iter < max_iter) {
        const float zx2 = zx * zx;
        const float zy2 = zy * zy;
        // Escape test on the *entry* magnitude, matching the SIMD mask.
        if (zx2 + zy2 > escape_r2) {
          break;
        }
        ++iter;

        // zy must be updated before zx is overwritten.
        const float t = zx * zy;
        zy = (t + t) + cy;
        zx = (zx2 - zy2) + cx;
      }

      row[x] = static_cast<std::uint16_t>(iter);
    }
  }
}
