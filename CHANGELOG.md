# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-07-31

Initial release.

### Kernels

- **Scalar C++** reference kernel, single-precision, defining the bit-exact
  ground truth all other kernels are checked against.
- **x86-64:** AVX2 intrinsics kernel (8 pixels per step) and a hand-written
  NASM kernel (`src/asm/mandelbrot_avx2.asm`) following the System V AMD64 ABI.
- **arm64:** NEON intrinsics kernel (4 pixels per step) and a hand-written
  AArch64 kernel (`src/asm/mandelbrot_neon.S`) following AAPCS64.
- Both assembly kernels are true leaf functions: no stack frame, no prologue,
  no frame pointer, every register used is caller-saved.
- Entry to the AVX2 kernels is gated at runtime on CPUID leaf 7 plus
  OSXSAVE/XGETBV, so the binary cannot SIGILL on a CPU without AVX2 or on an OS
  that is not preserving YMM state.
- Shared ABI contract in `kernel_api.hpp` with a `static_assert` per struct
  field, guarding the byte offsets the assembly indexes by hand.

### Correctness

- Every kernel is verified **bit-identical** to the scalar reference across five
  views, including a deep boundary zoom, a region entirely outside the set,
  `max_iter == 0`, and a non-multiple-of-8 width.
- 130 headless self-test checks, runnable with no CMake and no dependencies via
  `scripts/selftest.sh`.
- Statistics checked against hand-computed values rather than golden output.

### Statistical profiler

- Upper-tail trimming (drops the slowest 5%) with raw and trimmed figures shown
  side by side, plus how much of the raw mean was scheduler noise.
- Unbiased sample standard deviation, coefficient of variation, SEM, 95% CI,
  and min/median/p95/p99.
- CLT sample-size gating (n >= 30) surfaced in the UI.
- Reports whether `high_resolution_clock` is steady and its measured
  granularity, rather than assuming.

### Multi-core

- Optional multi-threaded rendering with a `CPU threads` slider, 1..physical
  cores, defaulting to 1.
- **Dynamic row stealing** rather than static bands, because Mandelbrot rows
  differ enormously in cost and equal bands would leave most threads idle. Also
  self-balances across heterogeneous core clusters.
- The calling thread participates, so `threads == 1` takes a fast path with no
  atomics, mutex or wakeups at all.
- Measured 13x on 15 cores (Apple M5 Pro), ~87% scaling efficiency.

### GUI

- Dear ImGui + SDL2 dashboard overlaying a live fractal render: kernel
  drop-down, live latency and dispersion, frame counter, hardware panel,
  progress-barred statistical sweep, and a correctness check.
- A second plain-English panel rating throughput on a 14-tier scale from
  "Absolute Garbage" to "Godly", driven by **pixel-iterations per second**
  rather than pixels per second so the rating tracks the kernel and not the
  view.
- Pause / Step / Reset stats controls, with space as a shortcut.

### Constraints

- Zero heap allocation on the hot path, **proven at runtime** by a thread-scoped
  `operator new`/`delete` tripwire displayed live in the dashboard, not asserted
  in a comment.
- All pixel buffers statically allocated and aligned to the destructive
  interference size (128 bytes on arm64, where the cache line really is 128).

### Build

- CMake with `enable_language(ASM_NASM)` on x86-64; the arm64 kernel is
  assembled by the C compiler as a preprocessed `.S`.
- `-ffast-math` is paired with **`-ffp-contract=off`** and
  **`-fno-associative-math`**. Both are required, not cosmetic: without them the
  compiler fuses mul+add into FMA and reassociates the coordinate arithmetic,
  which makes the C++ kernels disagree with the hand-written assembly. The
  source-level `#pragma clang fp` directives do **not** suppress either.
- Warns loudly at configure time and in the UI if built as Debug, because at
  `-O0` the intrinsics kernel is ~11x slower and misleadingly appears ~10x
  slower than the assembly it is actually within 1% of.
- `scripts/bootstrap_build.sh` builds the whole app with no CMake at all.
