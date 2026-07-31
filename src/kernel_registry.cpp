// kernel_registry.cpp — assemble the kernel list for this build + this CPU.

#include "kernel_registry.hpp"

#include "arch.hpp"
#include "cpu_features.hpp"

namespace mb {
namespace {

KernelRegistry Build() {
  KernelRegistry r;
  const CpuFeatures& cpu = DetectCpuFeatures();

  // --- Kernel A: always present, always the reference.
  {
    KernelDesc d;
    d.name = "Scalar C++";
    d.isa = "Scalar";
    d.impl = "C++";
    d.detail = "Single-precision scalar loop. Reference output for correctness.";
    d.fn = &mandelbrot_scalar_cpp;
    d.lanes = 1;
    d.available = true;
    r.kernels[static_cast<std::size_t>(r.count++)] = d;
  }

#if MB_ARCH_X86
  {
    KernelDesc d;
    d.name = "C++ AVX2 Intrinsics";
    d.isa = "AVX2";
    d.impl = "Intrinsics";
    d.detail = "_mm256_* intrinsics, 8 pixels per 256-bit step.";
    d.fn = &mandelbrot_avx2_intrinsics;
    d.lanes = 8;
    d.available = cpu.simd_kernels_usable;
    d.unavailable_reason =
        cpu.avx2 ? "OS is not preserving YMM state (OSXSAVE/XGETBV check failed)"
                 : "CPU does not report AVX2 (CPUID leaf 7, EBX bit 5)";
    r.kernels[static_cast<std::size_t>(r.count++)] = d;
  }
  {
    KernelDesc d;
    d.name = "NASM Assembly (AVX2)";
    d.isa = "AVX2";
    d.impl = "NASM";
    d.detail = "Hand-rolled x86-64 NASM, System V ABI, no stack frame.";
    d.fn = &mandelbrot_avx2_nasm;
    d.lanes = 8;
    d.available = cpu.simd_kernels_usable;
    d.unavailable_reason =
        cpu.avx2 ? "OS is not preserving YMM state (OSXSAVE/XGETBV check failed)"
                 : "CPU does not report AVX2 (CPUID leaf 7, EBX bit 5)";
    r.kernels[static_cast<std::size_t>(r.count++)] = d;
  }
#endif

#if MB_ARCH_ARM
  {
    KernelDesc d;
    d.name = "C++ NEON Intrinsics";
    d.isa = "NEON";
    d.impl = "Intrinsics";
    d.detail = "ARM NEON intrinsics, 4 pixels per 128-bit step.";
    d.fn = &mandelbrot_neon_intrinsics;
    d.lanes = 4;
    d.available = cpu.neon;
    d.unavailable_reason = "NEON/ASIMD not reported";
    r.kernels[static_cast<std::size_t>(r.count++)] = d;
  }
  {
    KernelDesc d;
    d.name = "AArch64 Assembly (NEON)";
    d.isa = "NEON";
    d.impl = "AArch64 asm";
    d.detail = "Hand-rolled AArch64 NEON, AAPCS64 leaf, no stack frame.";
    d.fn = &mandelbrot_neon_asm;
    d.lanes = 4;
    d.available = cpu.neon;
    d.unavailable_reason = "NEON/ASIMD not reported";
    r.kernels[static_cast<std::size_t>(r.count++)] = d;
  }
#endif

  r.default_index = 0;
  for (int i = 0; i < r.count; ++i) {
    if (r.kernels[static_cast<std::size_t>(i)].available) {
      r.default_index = i;
      break;
    }
  }
  return r;
}

}  // namespace

const KernelRegistry& Kernels() {
  static const KernelRegistry cached = Build();
  return cached;
}

}  // namespace mb
