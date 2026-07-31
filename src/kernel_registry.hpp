// kernel_registry.hpp — the set of kernels this binary can actually run.
//
// Built once at startup from compile-time architecture plus runtime CPUID, so
// the UI dropdown only ever offers kernels that are safe to enter. Unavailable
// kernels are kept in the list with a reason string rather than hidden, because
// "why is there no NASM option on this machine" is the first question a reader
// will have.
#pragma once

#include <array>

#include "kernel_api.hpp"

namespace mb {

constexpr int kMaxKernels = 4;

struct KernelDesc {
  const char* name = "";     // dropdown label
  const char* isa = "";      // "Scalar", "AVX2", "NEON"
  const char* impl = "";     // "C++", "Intrinsics", "NASM", "AArch64 asm"
  const char* detail = "";   // one-line explanation for the UI
  KernelFn fn = nullptr;
  int lanes = 1;             // pixels per inner-loop step
  bool available = false;
  const char* unavailable_reason = "";
};

struct KernelRegistry {
  std::array<KernelDesc, kMaxKernels> kernels{};
  int count = 0;
  int default_index = 0;  // first available kernel

  const KernelDesc& operator[](int i) const { return kernels[static_cast<std::size_t>(i)]; }
};

const KernelRegistry& Kernels();

}  // namespace mb
