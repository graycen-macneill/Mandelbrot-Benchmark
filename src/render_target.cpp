// render_target.cpp — buffer instance, palette generation and colour mapping.

#include "render_target.hpp"

#include <cmath>
#include <cstdlib>

namespace mb {
namespace {

std::uint32_t PackAbgr(int r, int g, int b) {
  const auto clamp8 = [](int v) -> std::uint32_t {
    if (v < 0) return 0u;
    if (v > 255) return 255u;
    return static_cast<std::uint32_t>(v);
  };
  // SDL_PIXELFORMAT_ABGR8888 read as a little-endian uint32 is 0xAABBGGRR.
  return (0xFFu << 24) | (clamp8(b) << 16) | (clamp8(g) << 8) | clamp8(r);
}

// Piecewise-linear gradient through a small set of stops. t in [0, 1].
struct Stop {
  double t;
  int r, g, b;
};

std::uint32_t SampleGradient(const Stop* stops, int count, double t) {
  if (t <= stops[0].t) {
    return PackAbgr(stops[0].r, stops[0].g, stops[0].b);
  }
  for (int i = 1; i < count; ++i) {
    if (t <= stops[i].t) {
      const Stop& a = stops[i - 1];
      const Stop& b = stops[i];
      const double span = b.t - a.t;
      const double f = span > 0.0 ? (t - a.t) / span : 0.0;
      return PackAbgr(static_cast<int>(a.r + f * (b.r - a.r) + 0.5),
                      static_cast<int>(a.g + f * (b.g - a.g) + 0.5),
                      static_cast<int>(a.b + f * (b.b - a.b) + 0.5));
    }
  }
  const Stop& last = stops[count - 1];
  return PackAbgr(last.r, last.g, last.b);
}

constexpr Stop kEmber[] = {
    {0.00, 0, 0, 4},      {0.15, 32, 12, 74},   {0.35, 120, 28, 109},
    {0.55, 205, 60, 79},  {0.75, 246, 133, 41}, {0.90, 252, 214, 106},
    {1.00, 255, 255, 235},
};

constexpr Stop kIce[] = {
    {0.00, 2, 4, 24},     {0.20, 12, 44, 92},    {0.45, 26, 108, 160},
    {0.70, 88, 180, 208}, {0.88, 176, 226, 233}, {1.00, 245, 253, 255},
};

constexpr Stop kSpectrum[] = {
    {0.00, 8, 0, 32},     {0.16, 60, 12, 140},  {0.33, 20, 90, 200},
    {0.50, 20, 170, 130}, {0.66, 190, 200, 40}, {0.83, 230, 90, 40},
    {1.00, 255, 235, 220},
};

constexpr Stop kGray[] = {
    {0.00, 0, 0, 0},
    {1.00, 255, 255, 255},
};

}  // namespace

const char* PaletteName(Palette p) {
  switch (p) {
    case Palette::Ember:
      return "Ember";
    case Palette::Ice:
      return "Ice";
    case Palette::Spectrum:
      return "Spectrum";
    case Palette::Grayscale:
      return "Grayscale";
    default:
      return "?";
  }
}

RenderTarget& Target() {
  // ~17 MiB of BSS. Deliberately a single static instance rather than a heap
  // allocation so "pre-allocated" is a property of the binary, not of a
  // startup code path that could be skipped.
  static RenderTarget instance;
  return instance;
}

void EnsurePaletteLut(RenderTarget& t, Palette palette, int max_iter) {
  if (max_iter < 1) {
    max_iter = 1;
  }
  if (max_iter > kMaxIterationCap) {
    max_iter = kMaxIterationCap;
  }
  if (t.lut_max_iter == max_iter && t.lut_palette == palette) {
    return;  // already current
  }

  const Stop* stops = kEmber;
  int stop_count = static_cast<int>(sizeof(kEmber) / sizeof(kEmber[0]));
  switch (palette) {
    case Palette::Ice:
      stops = kIce;
      stop_count = static_cast<int>(sizeof(kIce) / sizeof(kIce[0]));
      break;
    case Palette::Spectrum:
      stops = kSpectrum;
      stop_count = static_cast<int>(sizeof(kSpectrum) / sizeof(kSpectrum[0]));
      break;
    case Palette::Grayscale:
      stops = kGray;
      stop_count = static_cast<int>(sizeof(kGray) / sizeof(kGray[0]));
      break;
    case Palette::Ember:
    default:
      break;
  }

  // Points that never escaped belong to the set: paint them black.
  t.palette_lut[max_iter] = PackAbgr(0, 0, 0);

  for (int i = 0; i < max_iter; ++i) {
    // sqrt compresses the low end, where escape counts cluster, so the
    // gradient does not collapse into a single band near the boundary.
    const double linear = static_cast<double>(i) / static_cast<double>(max_iter);
    t.palette_lut[i] = SampleGradient(stops, stop_count, std::sqrt(linear));
  }

  // Anything above max_iter is unreachable but kept well-defined so a stale
  // buffer can never index uninitialised memory.
  for (int i = max_iter + 1; i <= kMaxIterationCap; ++i) {
    t.palette_lut[i] = PackAbgr(0, 0, 0);
  }

  t.lut_max_iter = max_iter;
  t.lut_palette = palette;
}

void Colourize(RenderTarget& t, int width, int height) {
  const std::uint32_t* lut = t.palette_lut;
  for (int y = 0; y < height; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * kRenderStride;
    const std::uint16_t* src = t.iterations + row;
    std::uint32_t* dst = t.pixels + row;
    for (int x = 0; x < width; ++x) {
      dst[x] = lut[src[x]];
    }
  }
}

std::uint64_t SumIterations(const RenderTarget& t, int width, int height) {
  std::uint64_t total = 0;
  for (int y = 0; y < height; ++y) {
    const std::uint16_t* row = t.iterations + static_cast<std::size_t>(y) * kRenderStride;
    for (int x = 0; x < width; ++x) {
      total += row[x];
    }
  }
  return total;
}

VerifyResult CompareBuffers(const std::uint16_t* a, const std::uint16_t* b, int width,
                            int height, int stride) {
  VerifyResult r;
  for (int y = 0; y < height; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
    for (int x = 0; x < width; ++x) {
      const int lhs = a[row + static_cast<std::size_t>(x)];
      const int rhs = b[row + static_cast<std::size_t>(x)];
      if (lhs != rhs) {
        ++r.mismatched_pixels;
        const int d = std::abs(lhs - rhs);
        if (d > r.max_abs_difference) {
          r.max_abs_difference = d;
        }
      }
    }
    r.compared_pixels += static_cast<std::uint64_t>(width);
  }
  return r;
}

}  // namespace mb
