#!/usr/bin/env bash
# bootstrap_build.sh — build the full GUI app WITHOUT CMake.
#
# Why this exists: the documented build is `cmake -B build && cmake --build build`,
# which fetches SDL2 / Dear ImGui / hwinfo for you. If CMake is not installed and
# you would rather not install it, this script does the same job with nothing but
# a compiler, curl, tar and make:
#
#   1. downloads Dear ImGui and SDL2 into ./.deps  (skipped if already there)
#   2. builds SDL2 as a static library              (skipped if already built)
#   3. compiles Dear ImGui + every project source
#   4. links ./build-manual/MandelbrotBenchmark
#
# It uses the same compiler flags as CMakeLists.txt, including the two FP
# corrections that keep the kernels bit-exact. hwinfo is NOT fetched here; the
# build uses the native sysctl/procfs hardware provider instead, which reports
# the same fields. Pass --with-hwinfo to fetch and link hwinfo as well.
#
# Everything lands in ./.deps and ./build-manual. To undo: rm -rf .deps build-manual
#
# Usage:
#   ./scripts/bootstrap_build.sh [--with-hwinfo] [--jobs N] [--clean]

set -euo pipefail

# macOS ships bash 3.2, where "${arr[@]}" on an *empty* array is an "unbound
# variable" error under `set -u`. Every possibly-empty array below is expanded
# through this guard, which is correct on both bash 3.2 and 5.x.
#   "${arr[@]+"${arr[@]}"}"  ->  expands to nothing when arr is empty

cd "$(dirname "$0")/.."
ROOT="$PWD"
DEPS="$ROOT/.deps"
OUT="$ROOT/build-manual"

IMGUI_TAG="v1.91.5"
SDL_TAG="release-2.30.9"
HWINFO_TAG="v1.0.0"

WITH_HWINFO=0
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-hwinfo) WITH_HWINFO=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --clean) rm -rf "$DEPS" "$OUT"; echo "removed .deps and build-manual"; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

CXX="${CXX:-clang++}"
CC="${CC:-clang}"
command -v "$CXX" >/dev/null || { echo "error: $CXX not found" >&2; exit 1; }
command -v curl  >/dev/null || { echo "error: curl not found" >&2; exit 1; }

ARCH="$(uname -m)"
mkdir -p "$DEPS" "$OUT/obj"

# ---------------------------------------------------------------------------
# Architecture-specific sources and flags
# ---------------------------------------------------------------------------
ARCH_FLAGS=()
ASM_OBJ=""
case "$ARCH" in
  arm64|aarch64)
    if "$CXX" -mcpu=native -xc++ -fsyntax-only /dev/null 2>/dev/null; then
      ARCH_FLAGS+=(-mcpu=native)
    fi
    KERNEL_SRC=(src/kernels_neon.cpp)
    ASM_SRC="src/asm/mandelbrot_neon.S"
    ;;
  x86_64|amd64)
    ARCH_FLAGS+=(-march=native)
    KERNEL_SRC=(src/kernels_avx2.cpp)
    ASM_SRC="src/asm/mandelbrot_avx2.asm"
    if ! command -v nasm >/dev/null; then
      echo "error: NASM is required on x86-64 to assemble $ASM_SRC" >&2
      echo "       macOS: brew install nasm   Debian: sudo apt install nasm" >&2
      exit 1
    fi
    ;;
  *)
    echo "warning: unrecognised architecture '$ARCH'; building scalar kernel only" >&2
    KERNEL_SRC=()
    ASM_SRC=""
    ;;
esac

# Same FP flags as CMakeLists.txt. -ffp-contract=off and -fno-associative-math are
# required for cross-kernel bit-exactness, not optional polish — see the comment
# block in CMakeLists.txt.
FLAGS=(-std=c++20 -O3 -ffast-math -ffp-contract=off -fno-associative-math
       -Wno-overriding-option -DNDEBUG -Wall -Wextra
       ${ARCH_FLAGS[@]+"${ARCH_FLAGS[@]}"})

# std::thread needs -pthread on glibc toolchains; harmless/implicit on macOS.
if [[ "$(uname -s)" != "Darwin" ]]; then
  FLAGS+=(-pthread)
  LINK_EXTRA=(-pthread)
else
  LINK_EXTRA=()
fi

# ---------------------------------------------------------------------------
# 1. Dear ImGui
# ---------------------------------------------------------------------------
IMGUI="$DEPS/imgui"
if [[ ! -d "$IMGUI" ]]; then
  echo "==> fetching Dear ImGui $IMGUI_TAG"
  curl -fsSL "https://codeload.github.com/ocornut/imgui/tar.gz/refs/tags/$IMGUI_TAG" \
    -o "$DEPS/imgui.tgz"
  tar xzf "$DEPS/imgui.tgz" -C "$DEPS"
  mv "$DEPS/imgui-${IMGUI_TAG#v}" "$IMGUI"
  rm -f "$DEPS/imgui.tgz"
else
  echo "==> Dear ImGui already present"
fi

# ---------------------------------------------------------------------------
# 2. SDL2 (static)
# ---------------------------------------------------------------------------
SDL_SRC="$DEPS/sdl2-src"
SDL_INST="$DEPS/sdl2"
if [[ ! -f "$SDL_INST/lib/libSDL2.a" ]]; then
  if [[ ! -d "$SDL_SRC" ]]; then
    echo "==> fetching SDL2 $SDL_TAG"
    curl -fsSL "https://codeload.github.com/libsdl-org/SDL/tar.gz/refs/tags/$SDL_TAG" \
      -o "$DEPS/sdl2.tgz"
    tar xzf "$DEPS/sdl2.tgz" -C "$DEPS"
    mv "$DEPS/SDL-$SDL_TAG" "$SDL_SRC"
    rm -f "$DEPS/sdl2.tgz"
  fi
  echo "==> building SDL2 static (this takes a couple of minutes, once)"
  (
    cd "$SDL_SRC"
    ./configure --prefix="$SDL_INST" --disable-shared --enable-static \
      > "$DEPS/sdl2-configure.log" 2>&1
    make -j"$JOBS" > "$DEPS/sdl2-make.log" 2>&1
    make install   > "$DEPS/sdl2-install.log" 2>&1
  ) || { echo "SDL2 build failed; see $DEPS/sdl2-*.log" >&2; exit 1; }
else
  echo "==> SDL2 already built"
fi

SDL_CFLAGS="-I$SDL_INST/include/SDL2"
SDL_LIBS="$("$SDL_INST/bin/sdl2-config" --static-libs)"

# ---------------------------------------------------------------------------
# 3. hwinfo (optional)
# ---------------------------------------------------------------------------
if [[ $WITH_HWINFO -eq 1 ]]; then
  echo "==> --with-hwinfo requested, but hwinfo ships a CMake-only build." >&2
  echo "    Use the CMake build for hwinfo support:" >&2
  echo "      cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
  echo "    Continuing with the native hardware provider instead." >&2
fi

# ---------------------------------------------------------------------------
# 4. Compile Dear ImGui
# ---------------------------------------------------------------------------
# Deliberately NOT built with -ffast-math: only the measured code should carry it.
echo "==> compiling Dear ImGui"
IMGUI_UNITS=(imgui imgui_draw imgui_tables imgui_widgets imgui_demo)
for u in "${IMGUI_UNITS[@]}"; do
  [[ "$OUT/obj/$u.o" -nt "$IMGUI/$u.cpp" ]] && continue
  "$CXX" -std=c++17 -O2 -I"$IMGUI" -I"$IMGUI/backends" $SDL_CFLAGS \
    -c "$IMGUI/$u.cpp" -o "$OUT/obj/$u.o"
done
for u in imgui_impl_sdl2 imgui_impl_sdlrenderer2; do
  [[ "$OUT/obj/$u.o" -nt "$IMGUI/backends/$u.cpp" ]] && continue
  "$CXX" -std=c++17 -O2 -I"$IMGUI" -I"$IMGUI/backends" $SDL_CFLAGS \
    -c "$IMGUI/backends/$u.cpp" -o "$OUT/obj/$u.o"
done

# ---------------------------------------------------------------------------
# 5. Compile the project
# ---------------------------------------------------------------------------
echo "==> compiling MandelbrotBenchmark ($ARCH)"
SOURCES=(
  src/main.cpp
  src/hardware_info.cpp
  src/alloc_tracker.cpp
  src/kernels_scalar.cpp
  src/cpu_features.cpp
  src/kernel_registry.cpp
  src/render_target.cpp
  src/timing.cpp
  src/performance_grade.cpp
  src/thread_pool.cpp
  ${KERNEL_SRC[@]+"${KERNEL_SRC[@]}"}
)

OBJS=()
for src in "${SOURCES[@]}"; do
  obj="$OUT/obj/$(basename "${src%.cpp}").o"
  OBJS+=("$obj")
  extra=()
  # The AVX2 unit needs -mavx2 explicitly; runtime entry is CPUID-gated.
  if [[ "$src" == "src/kernels_avx2.cpp" ]]; then
    extra+=(-mavx2)
  fi
  "$CXX" "${FLAGS[@]}" ${extra[@]+"${extra[@]}"} -Isrc -I"$IMGUI" \
    -I"$IMGUI/backends" $SDL_CFLAGS -c "$src" -o "$obj"
done

# Assembly kernel
if [[ -n "$ASM_SRC" ]]; then
  echo "==> assembling $ASM_SRC"
  ASM_OBJ="$OUT/obj/kernel_asm.o"
  case "$ARCH" in
    arm64|aarch64) "$CC" -c "$ASM_SRC" -o "$ASM_OBJ" ;;
    x86_64|amd64)
      fmt="elf64"
      [[ "$(uname -s)" == "Darwin" ]] && fmt="macho64"
      nasm -f "$fmt" -g "$ASM_SRC" -o "$ASM_OBJ"
      ;;
  esac
  OBJS+=("$ASM_OBJ")
fi

# ---------------------------------------------------------------------------
# 6. Link
# ---------------------------------------------------------------------------
echo "==> linking"
IMGUI_OBJS=()
for u in "${IMGUI_UNITS[@]}" imgui_impl_sdl2 imgui_impl_sdlrenderer2; do
  IMGUI_OBJS+=("$OUT/obj/$u.o")
done

# shellcheck disable=SC2086
"$CXX" "${OBJS[@]}" "${IMGUI_OBJS[@]}" ${LINK_EXTRA[@]+"${LINK_EXTRA[@]}"} $SDL_LIBS -o "$OUT/MandelbrotBenchmark"

echo
echo "Built: $OUT/MandelbrotBenchmark"
echo "Run:   ./build-manual/MandelbrotBenchmark"
