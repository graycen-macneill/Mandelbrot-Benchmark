#!/usr/bin/env bash
# selftest.sh — build and run the headless kernel/statistics self-test.
#
# Intentionally does not need CMake, NASM, SDL2, ImGui or hwinfo: it compiles the
# kernels, the profiler and the buffers with one compiler invocation. Useful for
# CI, for bisecting a kernel regression, and for checking out a machine you only
# have SSH access to.
#
# Usage:  ./scripts/selftest.sh [-O0|-O1|-O2|-O3]

set -euo pipefail

cd "$(dirname "$0")/.."

OPT="${1:--O3}"
OUT="${TMPDIR:-/tmp}/mb_selftest"

CXX="${CXX:-clang++}"

# Match the release flags the real build uses. See CMakeLists.txt for the full
# rationale; briefly, -ffast-math alone breaks cross-kernel bit-exactness in two
# independent ways and both have to be switched back off:
#
#   -ffp-contract=off        -ffast-math implies -ffp-contract=fast, and the
#                            backend fuses mul+add into fmadd/fmla regardless of
#                            any source-level pragma. Measured: 2 fmadd in the
#                            "scalar" reference.
#   -fno-associative-math    -ffast-math permits reassociation, which reorders
#                            the coordinate arithmetic in the SIMD kernels.
#                            Measured: 14,925 mismatched pixels on a boundary
#                            zoom with contraction already disabled.
FLAGS=(-std=c++20 "$OPT" -ffast-math -ffp-contract=off -fno-associative-math -DNDEBUG
       -Wall -Wextra)

# -march=native on x86, -mcpu=native on Apple/ARM toolchains.
ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
  x86_64|amd64)
    FLAGS+=(-march=native)
    ASM_SRC=""            # NASM is driven by CMake, not by this script
    ;;
  arm64|aarch64)
    if "$CXX" -mcpu=native -xc++ -fsyntax-only /dev/null 2>/dev/null; then
      FLAGS+=(-mcpu=native)
    fi
    ASM_SRC="src/asm/mandelbrot_neon.S"
    ;;
  *)
    ASM_SRC=""
    ;;
esac

SOURCES=(
  tests/verify_kernels.cpp
  src/kernels_scalar.cpp
  src/kernels_avx2.cpp
  src/kernels_neon.cpp
  src/cpu_features.cpp
  src/kernel_registry.cpp
  src/render_target.cpp
  src/timing.cpp
  src/alloc_tracker.cpp
  src/performance_grade.cpp
  src/thread_pool.cpp
)

if [[ -n "$ASM_SRC" ]]; then
  SOURCES+=("$ASM_SRC")
fi

echo "==> compiling self-test ($ARCH_RAW, $OPT)"
# The AVX2 translation unit needs -mavx2 explicitly so it builds even where
# -march=native does not imply AVX2. Harmless on non-x86 (the file is empty there).
if [[ "$ARCH_RAW" == "x86_64" || "$ARCH_RAW" == "amd64" ]]; then
  "$CXX" "${FLAGS[@]}" -mavx2 -c src/kernels_avx2.cpp -o "${OUT}_avx2.o"
  SOURCES=("${SOURCES[@]/src\/kernels_avx2.cpp/${OUT}_avx2.o}")
fi

"$CXX" "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT"

echo "==> running"
"$OUT"
