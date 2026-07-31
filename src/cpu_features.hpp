// cpu_features.hpp — runtime ISA detection gating the SIMD kernels.
#pragma once

namespace mb {

struct CpuFeatures {
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool osxsave = false;   // OS has enabled XGETBV
  bool ymm_state = false; // OS actually preserves YMM across context switches
  bool neon = false;      // ASIMD

  // True when this build's hand-rolled assembly kernel is safe to enter.
  bool simd_kernels_usable = false;
};

// Cheap and idempotent; the result is cached after the first call.
const CpuFeatures& DetectCpuFeatures();

}  // namespace mb
