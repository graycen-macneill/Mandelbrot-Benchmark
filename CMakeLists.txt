# ===========================================================================
#  MandelbrotBenchmark — C++ / NASM SIMD benchmarking suite
# ===========================================================================
#
#  Configure and build:
#      cmake -B build -DCMAKE_BUILD_TYPE=Release
#      cmake --build build -j
#      ./build/MandelbrotBenchmark
#
#  Useful options:
#      -DMB_USE_HWINFO=OFF        use the native sysctl/procfs provider instead
#      -DMB_USE_SYSTEM_SDL2=ON    link a system SDL2 rather than fetching one
#      -DMB_BUILD_TESTS=OFF       skip the headless self-test target
#      -DMB_NATIVE_ARCH=OFF       drop -march=native (for distributable builds)
#
#  Architecture: this project builds a per-architecture SIMD + assembly kernel
#  pair. On x86-64 that is AVX2 intrinsics plus a NASM kernel; on arm64 it is
#  NEON intrinsics plus an AArch64 assembly kernel. Universal / fat binaries are
#  rejected because a single translation unit cannot serve both.
# ===========================================================================

cmake_minimum_required(VERSION 3.21)

project(MandelbrotBenchmark
  VERSION 1.0.0
  DESCRIPTION "Scalar vs SIMD-intrinsics vs hand-rolled-assembly Mandelbrot benchmark"
  LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)

# std::thread in the render pool.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Build type
# ---------------------------------------------------------------------------
# A benchmark built without optimisation measures nothing useful, so default to
# Release. For GCC/Clang, Release supplies the mandated -O3 and -DNDEBUG.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
  message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(MB_USE_HWINFO      "Use the hwinfo library for hardware interrogation" ON)
option(MB_USE_SYSTEM_SDL2 "Link a system-provided SDL2 instead of fetching"  OFF)
option(MB_BUILD_TESTS     "Build the headless kernel/statistics self-test"    ON)
option(MB_NATIVE_ARCH     "Compile with -march=native / -mcpu=native"         ON)

# ---------------------------------------------------------------------------
# Target architecture
# ---------------------------------------------------------------------------
set(MB_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")

if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  list(LENGTH CMAKE_OSX_ARCHITECTURES _mb_arch_count)
  if(_mb_arch_count GREATER 1)
    message(FATAL_ERROR
      "CMAKE_OSX_ARCHITECTURES lists ${_mb_arch_count} architectures. This "
      "project cannot build a universal binary: the AVX2/NASM and NEON/AArch64 "
      "kernels are mutually exclusive within one translation unit. Configure a "
      "single architecture at a time, e.g. -DCMAKE_OSX_ARCHITECTURES=arm64.")
  endif()
  set(MB_TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
endif()

set(MB_X86 OFF)
set(MB_ARM OFF)

if(MB_TARGET_ARCH MATCHES "^([Xx]86_64|AMD64|amd64)$")
  set(MB_X86 ON)
elseif(MB_TARGET_ARCH MATCHES "^([Aa][Rr][Mm]64|aarch64|AARCH64)$")
  set(MB_ARM ON)
else()
  message(WARNING
    "Unrecognised target architecture '${MB_TARGET_ARCH}'. Only the scalar C++ "
    "kernel will be available; no SIMD or assembly kernel will be built.")
endif()

# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------
# x86-64 -> NASM for mandelbrot_avx2.asm.
# arm64  -> the C compiler's integrated assembler for mandelbrot_neon.S, which
#           is preprocessed (.S) so it can switch between Mach-O and ELF idioms.
#           NASM is x86-only and cannot assemble AArch64.
if(MB_X86)
  enable_language(ASM_NASM)
  if(NOT CMAKE_ASM_NASM_COMPILER)
    message(FATAL_ERROR
      "NASM was not found but is required on x86-64.\n"
      "  macOS : brew install nasm\n"
      "  Debian: sudo apt install nasm\n"
      "  Fedora: sudo dnf install nasm")
  endif()
  message(STATUS "NASM: ${CMAKE_ASM_NASM_COMPILER} "
                 "(object format: ${CMAKE_ASM_NASM_OBJECT_FORMAT})")
elseif(MB_ARM)
  enable_language(ASM)
  message(STATUS "Assembler: ${CMAKE_ASM_COMPILER} (integrated, AArch64)")
endif()

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------
add_library(mb_flags INTERFACE)

if(MSVC)
  message(FATAL_ERROR
    "MSVC is not supported: the assembly kernels target the System V AMD64 and "
    "AAPCS64 ABIs, and the FP flags below are GCC/Clang spellings. Use clang-cl "
    "with care, or build under MinGW/WSL.")
endif()

# --- Architecture tuning (spec: -march=native)
if(MB_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=native" MB_HAS_MARCH_NATIVE)
  check_cxx_compiler_flag("-mcpu=native" MB_HAS_MCPU_NATIVE)

  if(MB_X86 AND MB_HAS_MARCH_NATIVE)
    target_compile_options(mb_flags INTERFACE -march=native)
    set(MB_ARCH_FLAG "-march=native")
  elseif(MB_ARM AND MB_HAS_MCPU_NATIVE)
    # -mcpu=native is the correct spelling on AArch64; it selects both the
    # architecture revision and the microarchitectural tuning model.
    target_compile_options(mb_flags INTERFACE -mcpu=native)
    set(MB_ARCH_FLAG "-mcpu=native")
  elseif(MB_HAS_MARCH_NATIVE)
    target_compile_options(mb_flags INTERFACE -march=native)
    set(MB_ARCH_FLAG "-march=native")
  else()
    set(MB_ARCH_FLAG "none (compiler rejected both -march=native and -mcpu=native)")
  endif()
else()
  set(MB_ARCH_FLAG "disabled by -DMB_NATIVE_ARCH=OFF")
endif()

# --- Floating point (spec: -ffast-math, plus two corrections)
#
# -ffast-math alone makes the kernels disagree with each other, in two
# independent ways that were both caught by tests/verify_kernels.cpp:
#
#   1. It implies -ffp-contract=fast. The backend then fuses mul+add into
#      fmadd / fmla / vfmadd *regardless of any source-level pragma* — the
#      "scalar" reference was measured emitting 2 fmadd instructions with
#      `#pragma clang fp contract(off)` in force. Since the hand-written
#      assembly kernels contain no FMA, this alone breaks bit-exactness.
#
#   2. It permits reassociation, which rewrites cx0 + (x + lane)*dx into
#      (cx0 + x*dx) + lane*dx so the invariant part can be hoisted out of the
#      loop. Algebraically identical, different rounding: 14,925 mismatched
#      pixels on a boundary zoom with contraction already disabled.
#
# Both are therefore switched back off. Everything else -ffast-math enables
# (finite-math-only, no-signed-zeros, no-math-errno, reciprocal/approx forms) is
# retained and is harmless here — the kernels contain no division and no
# transcendentals.
foreach(_mb_fp_flag -ffast-math -ffp-contract=off -fno-associative-math)
  string(MAKE_C_IDENTIFIER "MB_HAS${_mb_fp_flag}" _mb_fp_var)
  check_cxx_compiler_flag("${_mb_fp_flag}" ${_mb_fp_var})
  if(${_mb_fp_var})
    target_compile_options(mb_flags INTERFACE ${_mb_fp_flag})
  else()
    message(WARNING "Compiler rejected ${_mb_fp_flag}; cross-kernel "
                    "bit-exactness may not hold.")
  endif()
endforeach()

# Clang warns that -ffp-contract=off overrides part of -ffast-math. That is the
# intent, so silence just that diagnostic.
check_cxx_compiler_flag("-Wno-overriding-option" MB_HAS_WNO_OVERRIDING)
if(MB_HAS_WNO_OVERRIDING)
  target_compile_options(mb_flags INTERFACE -Wno-overriding-option)
endif()

target_compile_options(mb_flags INTERFACE -Wall -Wextra)

# NDEBUG comes from the Release configuration; assert it for non-Release builds
# too only where the spec requires it, without clobbering Debug's -O0.
target_compile_definitions(mb_flags INTERFACE $<$<CONFIG:Release>:NDEBUG>)

# Surface the build type actually used to compile mb_kernels inside the binary,
# so the dashboard can flag it live rather than silently rendering a latency
# number that mostly measures the missing -O3. See the Debug warning above.
target_compile_definitions(mb_flags INTERFACE
  MB_KERNEL_BUILD_TYPE="$<CONFIG>"
  $<$<CONFIG:Debug>:MB_KERNEL_BUILD_IS_DEBUG=1>)

if(MB_USE_HWINFO)
  target_compile_definitions(mb_flags INTERFACE MB_USE_HWINFO=1)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
include(Dependencies)

# ---------------------------------------------------------------------------
# mb_kernels — the measured code, isolated in its own target
# ---------------------------------------------------------------------------
# Do NOT add src/main.cpp here. It belongs to the MandelbrotBenchmark executable
# only, for two reasons: this library is deliberately free of SDL2/ImGui include
# paths so the measured code stays independent of the presentation layer, and
# mb_selftest links mb_kernels while defining its own main().
set(MB_KERNEL_SOURCES
  src/kernels_scalar.cpp
  src/cpu_features.cpp
  src/kernel_registry.cpp
  src/render_target.cpp
  src/timing.cpp
  src/performance_grade.cpp
  src/thread_pool.cpp
)

if(MB_X86)
  list(APPEND MB_KERNEL_SOURCES
    src/kernels_avx2.cpp
    src/asm/mandelbrot_avx2.asm   # linked directly into the target
  )
elseif(MB_ARM)
  list(APPEND MB_KERNEL_SOURCES
    src/kernels_neon.cpp
    src/asm/mandelbrot_neon.S
  )
endif()

add_library(mb_kernels STATIC ${MB_KERNEL_SOURCES})
target_include_directories(mb_kernels PUBLIC src)
target_link_libraries(mb_kernels PUBLIC mb_flags Threads::Threads)

# A benchmark kernel built at -O0 does not measure the hardware, it measures the
# compiler's decision not to optimize -- and the effect is not small. Measured on
# this project: at -O0 the NEON intrinsics kernel is ~11x slower than at -O3 and
# ends up ~10x slower than the hand-written assembly kernel it should be within
# 1% of at -O3, because -O0 skips inlining and loop-invariant hoisting for the
# C++ kernels while the .S/.asm kernel is raw machine code the compiler never
# touches either way. CLion in particular creates a Debug CMake profile by
# default (a separate cmake-build-debug/ tree, CMAKE_BUILD_TYPE=Debug) alongside
# the one you configured from the terminal, and it is easy to end up running
# that one without noticing. This warning exists so that trap is loud instead of
# silent.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "")
  message(WARNING
    "mb_kernels is being configured as a Debug build (CMAKE_BUILD_TYPE="
    "'${CMAKE_BUILD_TYPE}'). Any latency numbers this build produces are "
    "meaningless as a hardware comparison -- they mostly measure missing "
    "-O3, not the kernels. If you are in CLion: Settings > Build, Execution, "
    "Deployment > CMake, add or select a Release profile, and rebuild from "
    "that profile before trusting any benchmark output.")
endif()

if(MB_X86)
  # The AVX2 translation unit needs -mavx2 explicitly so it still builds on a
  # host whose -march=native does not imply AVX2. Entry is gated at runtime by
  # CPUID + XGETBV (see cpu_features.cpp), so this never produces a SIGILL.
  set_source_files_properties(src/kernels_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2")

  # Emit DWARF for the assembly so the kernel is steppable in a debugger and
  # shows up correctly in perf / Instruments.
  if(CMAKE_ASM_NASM_OBJECT_FORMAT STREQUAL "elf64")
    string(APPEND CMAKE_ASM_NASM_FLAGS " -g -F dwarf")
  elseif(CMAKE_ASM_NASM_OBJECT_FORMAT STREQUAL "macho64")
    string(APPEND CMAKE_ASM_NASM_FLAGS " -g")
  endif()
endif()

if(MB_ARM)
  # .S is preprocessed; make sure the assembler is invoked through the C driver
  # so the #if defined(__APPLE__) blocks resolve.
  set_source_files_properties(src/asm/mandelbrot_neon.S
    PROPERTIES LANGUAGE ASM COMPILE_OPTIONS "-x;assembler-with-cpp")
endif()

# ---------------------------------------------------------------------------
# MandelbrotBenchmark — the GUI application
# ---------------------------------------------------------------------------
add_executable(MandelbrotBenchmark
  src/main.cpp
  src/hardware_info.cpp
  src/alloc_tracker.cpp
)

target_link_libraries(MandelbrotBenchmark PRIVATE
  mb_kernels
  mb_imgui
  ${MB_SDL2_TARGET}
)

if(MB_USE_HWINFO)
  target_link_libraries(MandelbrotBenchmark PRIVATE ${MB_HWINFO_TARGET})
endif()

# SDL2 on some platforms wants SDL2main to provide the entry point shim.
if(TARGET SDL2::SDL2main)
  target_link_libraries(MandelbrotBenchmark PRIVATE SDL2::SDL2main)
endif()

if(APPLE)
  # sysctlbyname lives in libSystem, already linked; nothing extra needed.
elseif(UNIX)
  # Linux needs libm and libdl for hwinfo / SDL static links.
  target_link_libraries(MandelbrotBenchmark PRIVATE m dl pthread)
endif()

# ---------------------------------------------------------------------------
# mb_selftest — headless correctness + statistics verification
# ---------------------------------------------------------------------------
if(MB_BUILD_TESTS)
  enable_testing()

  add_executable(mb_selftest
    tests/verify_kernels.cpp
    src/alloc_tracker.cpp
  )
  target_link_libraries(mb_selftest PRIVATE mb_kernels)

  add_test(NAME kernels_and_statistics COMMAND mb_selftest)
  set_tests_properties(kernels_and_statistics PROPERTIES TIMEOUT 300)
endif()

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==== MandelbrotBenchmark configuration ====")
message(STATUS "  build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  target arch     : ${MB_TARGET_ARCH}")
message(STATUS "  compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  arch flag       : ${MB_ARCH_FLAG}")
message(STATUS "  fp flags        : -ffast-math -ffp-contract=off -fno-associative-math")
if(MB_X86)
  message(STATUS "  kernels         : Scalar C++, AVX2 intrinsics, NASM (AVX2)")
elseif(MB_ARM)
  message(STATUS "  kernels         : Scalar C++, NEON intrinsics, AArch64 asm (NEON)")
else()
  message(STATUS "  kernels         : Scalar C++ only")
endif()
message(STATUS "  hardware info   : ${MB_USE_HWINFO}")
message(STATUS "  self-test       : ${MB_BUILD_TESTS}")
message(STATUS "===========================================")
message(STATUS "")
