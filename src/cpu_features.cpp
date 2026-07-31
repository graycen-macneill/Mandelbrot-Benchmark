// cpu_features.cpp — runtime ISA detection.
//
// On x86-64 it is not enough to see the AVX2 CPUID bit: the OS must also have
// enabled and be preserving the upper YMM halves across context switches.
// Skipping the OSXSAVE/XGETBV half of the check is the classic way to ship a
// binary that SIGILLs on old kernels and hypervisors, so both are checked here
// before either AVX2 kernel is allowed to run.

#include "cpu_features.hpp"

#include "arch.hpp"

#if MB_ARCH_X86
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace mb {
namespace {

#if MB_ARCH_X86

bool CpuId(unsigned leaf, unsigned subleaf, unsigned regs[4]) {
#if defined(_MSC_VER)
  int out[4];
  __cpuidex(out, static_cast<int>(leaf), static_cast<int>(subleaf));
  regs[0] = static_cast<unsigned>(out[0]);
  regs[1] = static_cast<unsigned>(out[1]);
  regs[2] = static_cast<unsigned>(out[2]);
  regs[3] = static_cast<unsigned>(out[3]);
  return true;
#else
  return __get_cpuid_count(leaf, subleaf, &regs[0], &regs[1], &regs[2], &regs[3]) != 0;
#endif
}

unsigned long long ReadXcr0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned int eax = 0;
  unsigned int edx = 0;
  // XGETBV with ECX=0. Written as raw bytes so no -mxsave is required.
  __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
}

CpuFeatures Detect() {
  CpuFeatures f;

  unsigned r[4] = {0, 0, 0, 0};
  if (!CpuId(1, 0, r)) {
    return f;
  }
  const unsigned ecx1 = r[2];
  f.osxsave = (ecx1 & (1u << 27)) != 0;
  f.avx = (ecx1 & (1u << 28)) != 0;
  f.fma = (ecx1 & (1u << 12)) != 0;

  if (f.osxsave) {
    // Bit 1 = XMM state, bit 2 = YMM state. Both must be enabled by the OS.
    const unsigned long long xcr0 = ReadXcr0();
    f.ymm_state = (xcr0 & 0x6ull) == 0x6ull;
  }

  unsigned r7[4] = {0, 0, 0, 0};
  if (CpuId(7, 0, r7)) {
    f.avx2 = (r7[1] & (1u << 5)) != 0;
  }

  // Both AVX2 kernels are pure AVX2 (no FMA), so FMA is reported but not required.
  f.simd_kernels_usable = f.avx && f.avx2 && f.osxsave && f.ymm_state;
  return f;
}

#elif MB_ARCH_ARM

CpuFeatures Detect() {
  CpuFeatures f;
  // ASIMD/NEON is architecturally mandatory in AArch64 — there is no AArch64
  // core without it, so there is nothing to probe.
  f.neon = true;
  f.simd_kernels_usable = true;
  return f;
}

#else

CpuFeatures Detect() { return CpuFeatures{}; }

#endif

}  // namespace

const CpuFeatures& DetectCpuFeatures() {
  static const CpuFeatures cached = Detect();
  return cached;
}

}  // namespace mb
