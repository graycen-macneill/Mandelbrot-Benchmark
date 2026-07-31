// kernels_avx2.cpp — Kernel B: AVX2 compiler intrinsics, 8 pixels per inner step.
//
// Compiled only on x86-64, and only ever *entered* after cpu_features.cpp has
// confirmed AVX2 via CPUID + XGETBV. The translation unit itself is built with
// -mavx2 (see CMakeLists.txt) so it links even on a host whose -march=native
// lacks AVX2.
//
// Deliberately FMA-free. The recurrence needs zx^2 and zy^2 as separate values
// for the escape test *and* the real update, so there is no multiply-add pair
// left to contract — an FMA form would cost the same instruction count while
// adding a CPUID requirement and breaking bit-exactness with the scalar
// reference. Same reason the NASM sibling is pure AVX2.

#include "arch.hpp"

#if MB_ARCH_X86

#include <immintrin.h>

#include <cstdint>

#include "kernel_api.hpp"

// Under -ffast-math the compiler contracts an explicit _mm256_mul_ps +
// _mm256_add_ps pair into a single VFMADD — being written as intrinsics does not
// protect them. That changes the rounding of both the coordinate setup and
// |z|^2, breaking bit-exactness with the scalar reference, and quietly
// introduces the FMA3 requirement this kernel claims not to have.
//
// Reassociation is the second, independent hazard: -ffast-math lets the
// compiler rewrite cx0 + (x + lane) * dx into (cx0 + x*dx) + lane*dx, hoisting
// the invariant part out of the loop. Algebraically identical, different
// rounding.
//
// These pragmas state the intent but do not by themselves suppress either
// transform; the build passes -ffp-contract=off -fno-associative-math, which
// does. The arm64 sibling of this kernel was verified to need exactly the same
// treatment, at 14,925 mismatched pixels on a boundary zoom.
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#elif defined(__GNUC__)
#pragma GCC optimize("no-fast-math", "no-associative-math")
#endif

extern "C" void MB_HOT mandelbrot_avx2_intrinsics(std::uint16_t* out, const mb::KernelParams* p,
                                                  std::int32_t row_begin, std::int32_t row_end) {
  const __m256 v_cx0 = _mm256_set1_ps(p->cx0);
  const __m256 v_dx = _mm256_set1_ps(p->dx);
  const __m256 v_dy = _mm256_set1_ps(p->dy);
  const __m256 v_cy0 = _mm256_set1_ps(p->cy0);
  const __m256 v_escape = _mm256_set1_ps(p->escape_r2);
  const __m256 v_lanes = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

  const std::int32_t vec_width = p->vec_width;
  const std::int32_t max_iter = p->max_iter;
  const std::ptrdiff_t stride = p->stride;

  for (std::int32_t y = row_begin; y < row_end; ++y) {
    // cy = cy0 + (float)y * dy, broadcast. One multiply, one add — matches scalar.
    const __m256 v_y = _mm256_set1_ps(static_cast<float>(y));
    const __m256 v_cy = _mm256_add_ps(_mm256_mul_ps(v_y, v_dy), v_cy0);

    std::uint16_t* row = out + static_cast<std::ptrdiff_t>(y) * stride;

    for (std::int32_t x = 0; x < vec_width; x += 8) {
      // cx = cx0 + ((float)x + lane) * dx. Forming (x + lane) *before* the
      // multiply is what keeps this bit-identical to the scalar kernel's
      // cx0 + (float)(x + lane) * dx — folding cx0 + x*dx out of the loop
      // and adding lane*dx afterwards would round differently.
      const __m256 v_px = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), v_lanes);
      const __m256 v_cx = _mm256_add_ps(_mm256_mul_ps(v_px, v_dx), v_cx0);

      __m256 zx = _mm256_setzero_ps();
      __m256 zy = _mm256_setzero_ps();
      __m256i iter = _mm256_setzero_si256();

      for (std::int32_t n = 0; n < max_iter; ++n) {
        const __m256 zx2 = _mm256_mul_ps(zx, zx);
        const __m256 zy2 = _mm256_mul_ps(zy, zy);
        const __m256 mag = _mm256_add_ps(zx2, zy2);

        // _CMP_LE_OQ: ordered and non-signalling. Lanes that already escaped
        // may hold +inf or NaN; both compare false and stay false forever, so
        // the mask latches without any explicit "already done" bookkeeping.
        const __m256 mask = _mm256_cmp_ps(mag, v_escape, _CMP_LE_OQ);
        if (_mm256_movemask_ps(mask) == 0) {
          break;  // every lane in this batch has escaped
        }

        // mask lanes are 0 or 0xFFFFFFFF (== -1), so subtracting increments
        // exactly the still-live lanes.
        iter = _mm256_sub_epi32(iter, _mm256_castps_si256(mask));

        __m256 t = _mm256_mul_ps(zx, zy);
        t = _mm256_add_ps(t, t);
        zy = _mm256_add_ps(t, v_cy);
        zx = _mm256_add_ps(_mm256_sub_ps(zx2, zy2), v_cx);
      }

      // Pack 8 x int32 -> 8 x uint16 in lane order.
      //   packus:  [i0 i1 i2 i3 | i0 i1 i2 i3 || i4 i5 i6 i7 | i4 i5 i6 i7]
      //   qwords:  [   q0      |     q1      ||     q2      |     q3      ]
      // so selecting (q0, q2) into the low 128 bits yields i0..i7 in order.
      __m256i packed = _mm256_packus_epi32(iter, iter);
      packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(0, 0, 2, 0));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(row + x), _mm256_castsi256_si128(packed));
    }
  }
}

#endif  // MB_ARCH_X86
