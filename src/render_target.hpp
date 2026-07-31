// render_target.hpp — statically pre-allocated pixel buffers and colour mapping.
//
// Every buffer the render loop touches lives here, sized for the largest
// supported resolution and allocated exactly once in BSS. Nothing is ever
// resized: changing the render resolution only changes how much of each buffer
// is used, so the hot path never allocates, reallocates or frees.
//
// All buffers are aligned to the destructive-interference size (>= 64 bytes),
// which satisfies the mandated alignas(64) and additionally guarantees that a
// 256-bit vmovdqu / 128-bit str never straddles a cache line at a row start.
#pragma once

#include <cstddef>
#include <cstdint>

#include "arch.hpp"
#include "kernel_api.hpp"

namespace mb {

// Largest render resolution the statically allocated buffers can hold.
constexpr int kMaxRenderWidth = 1920;
constexpr int kMaxRenderHeight = 1080;

// Row stride in uint16 elements, padded up to the widest SIMD batch so no
// kernel needs a scalar tail loop. 1920 is already a multiple of 8.
constexpr int kRenderStride =
    ((kMaxRenderWidth + kMaxVectorLanes - 1) / kMaxVectorLanes) * kMaxVectorLanes;

constexpr std::size_t kIterationBufferElems =
    static_cast<std::size_t>(kRenderStride) * kMaxRenderHeight;

// Rounds a width up to the SIMD batch size. Kernels iterate to this bound.
constexpr int PaddedWidth(int width) {
  return ((width + kMaxVectorLanes - 1) / kMaxVectorLanes) * kMaxVectorLanes;
}

enum class Palette : int {
  Ember = 0,
  Ice,
  Spectrum,
  Grayscale,
  kCount,
};

const char* PaletteName(Palette p);

// ---------------------------------------------------------------------------
// RenderTarget — the pre-allocated buffer set.
// ---------------------------------------------------------------------------
struct RenderTarget {
  // Iteration counts written by the kernels.
  alignas(MB_CACHE_ALIGN) std::uint16_t iterations[kIterationBufferElems];

  // Reference output used by the correctness check, so verification does not
  // allocate either.
  alignas(MB_CACHE_ALIGN) std::uint16_t reference[kIterationBufferElems];

  // Packed pixels handed to SDL. Format is SDL_PIXELFORMAT_ABGR8888, i.e.
  // 0xAABBGGRR when read as a native little-endian uint32.
  alignas(MB_CACHE_ALIGN) std::uint32_t pixels[kIterationBufferElems];

  // iteration count -> ABGR8888, rebuilt only when the palette or max_iter
  // changes (never inside the render loop).
  alignas(MB_CACHE_ALIGN) std::uint32_t palette_lut[kMaxIterationCap + 1];

  int lut_max_iter = -1;
  Palette lut_palette = Palette::Ember;
};

// The single instance. Function-local static so initialisation order is defined.
RenderTarget& Target();

// Rebuilds palette_lut if (palette, max_iter) differs from what it holds.
// Cheap no-op when nothing changed.
void EnsurePaletteLut(RenderTarget& t, Palette palette, int max_iter);

// Maps iterations -> pixels for the given region. Pure LUT gather; no branches
// on the palette, no allocation.
void Colourize(RenderTarget& t, int width, int height);

// Counts pixels where `a` and `b` differ and reports the largest absolute
// difference. Used by the UI's kernel verification pass.
struct VerifyResult {
  std::uint64_t mismatched_pixels = 0;
  int max_abs_difference = 0;
  std::uint64_t compared_pixels = 0;
  bool exact() const { return mismatched_pixels == 0; }
};

VerifyResult CompareBuffers(const std::uint16_t* a, const std::uint16_t* b, int width,
                            int height, int stride);

// Total inner-loop iterations actually performed across the visible region.
//
// This is the workload-independent measure of how much work a frame cost.
// Pixels/second is not: the same kernel rendering a zoomed-out view at
// max_iter 64 moves an order of magnitude more pixels per second than it does
// on a deep boundary zoom at max_iter 2048, while the hardware is doing the
// same thing. Iterations/second stays flat across both, so it is what the
// performance grade is built on.
//
// Called outside the timed region, so it never contributes to a latency sample.
std::uint64_t SumIterations(const RenderTarget& t, int width, int height);

}  // namespace mb
