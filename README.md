# MandelbrotBenchmark

A real-time Mandelbrot renderer built to measure, not to look pretty: it runs the
same escape-time kernel through a scalar C++ loop, a compiler-intrinsics SIMD
kernel, and a hand-rolled assembly kernel, and reports the latency difference
with enough statistical care that the numbers mean something.

Every kernel is required to produce **bit-identical** output to the scalar
reference. A faster kernel that disagrees is a bug, not an optimisation, and the
suite is built to catch exactly that.

---

## Architecture support

The project builds a per-architecture SIMD + assembly pair:

| Target | Kernel A | Kernel B | Kernel C |
|---|---|---|---|
| **x86-64** | Scalar C++ | AVX2 intrinsics (`_mm256_*`), 8 px/step | **NASM** `mandelbrot_avx2.asm`, System V AMD64 ABI |
| **arm64** | Scalar C++ | NEON intrinsics, 4 px/step | **AArch64 asm** `mandelbrot_neon.S`, AAPCS64 |

Both assembly kernels are hand-written, take their arguments directly in
registers (`rdi`/`rsi`/`rdx`/`rcx` and `x0`/`x1`/`w2`/`w3`), and use **no stack
frame at all** — every register they touch is caller-saved, so there is no
prologue, no epilogue and no frame pointer.

> **Note on portability.** NASM is x86-only; it cannot assemble AArch64. The
> arm64 sibling therefore goes through clang's integrated assembler as a
> preprocessed `.S` file, which is why it carries `#if defined(__APPLE__)` blocks
> for the Mach-O vs ELF differences (symbol underscore prefixing, `@PAGE`
> vs `:lo12:`, and `L` vs `.L` local labels).

Universal / fat binaries are rejected at configure time: a single translation
unit cannot serve both instruction sets.

---

## Building

### Requirements

- CMake ≥ 3.21
- A C++20 compiler (GCC or Clang; MSVC is explicitly unsupported)
- **NASM** — required on x86-64 only
- git and network access on first configure (dependencies are fetched)

```bash
# macOS
brew install cmake nasm

# Debian / Ubuntu
sudo apt install cmake nasm build-essential

# Fedora
sudo dnf install cmake nasm gcc-c++
```

**If you use CLion, you already have CMake and Ninja** — no Homebrew needed. They
ship inside the app bundle:

```
/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake
/Applications/CLion.app/Contents/bin/ninja/mac/aarch64/ninja
```

On arm64 you do not need NASM at all; the AArch64 kernel is assembled by clang.

### Configure and build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/MandelbrotBenchmark
```

SDL2, Dear ImGui and hwinfo are fetched automatically via `FetchContent`.

> **Note.** `main.cpp` cannot be compiled on its own. It is one translation unit
> of nine, and it needs the SDL2 and Dear ImGui include paths plus the assembly
> kernel object at link time. `c++ src/main.cpp -o main` will fail on
> `#include <SDL.h>`, and would still fail at link even with the headers found.

### In CLion

Three things to get right:

1. **Build the project, not the file.** If the run/build dropdown in the toolbar
   says `main.cpp`, that is a single-file configuration and it will fail exactly
   as above. Pick the **`MandelbrotBenchmark`** configuration instead. If it is
   not offered, use *File → Reload CMake Project* so CLion picks up
   `CMakeLists.txt`.
2. **Use a Release profile.** CLion creates a *Debug* profile by default, and
   `CMakeLists.txt` only defaults to Release when the build type is unset — so a
   Debug profile really does compile at `-O0`. Benchmark numbers from that build
   are meaningless. Add a Release profile under *Settings → Build, Execution,
   Deployment → CMake*.
3. **Do not add `src/main.cpp` to `mb_kernels`.** It belongs only to the
   `MandelbrotBenchmark` executable. `mb_kernels` is deliberately kept free of
   SDL2/ImGui include paths so the measured code does not depend on the
   presentation layer, and `mb_selftest` links `mb_kernels` while defining its
   own `main()` — so listing it in both breaks the build twice over.

### Building without CMake

If you would rather not install CMake, `scripts/bootstrap_build.sh` does the same
job with nothing but a compiler, `curl`, `tar` and `make`. It downloads Dear ImGui
and SDL2 into `./.deps`, builds SDL2 static once, then compiles and links
everything with the same flags CMake uses:

```bash
./scripts/bootstrap_build.sh --jobs 8
./build-manual/MandelbrotBenchmark
```

Re-running it reuses `./.deps` and the compiled Dear ImGui objects, so subsequent
builds take a couple of seconds; the nine project translation units are always
recompiled. To undo it entirely: `rm -rf .deps build-manual`.

This path uses the native sysctl/procfs hardware provider rather than hwinfo,
since hwinfo is CMake-only; the dashboard reports the same fields either way. Use
the CMake build if you specifically want hwinfo linked in.

### Options

| Option | Default | Effect |
|---|---|---|
| `MB_USE_HWINFO` | `ON` | Use hwinfo for hardware interrogation. `OFF` falls back to the built-in native provider (sysctl on macOS, `/proc` + `/sys` on Linux), which reports the same fields. |
| `MB_USE_SYSTEM_SDL2` | `OFF` | Link a system SDL2 instead of fetching one. |
| `MB_BUILD_TESTS` | `ON` | Build the headless self-test target. |
| `MB_NATIVE_ARCH` | `ON` | Compile with `-march=native` / `-mcpu=native`. Turn off for distributable binaries. |

### Self-test

The correctness and statistics checks need neither a GPU nor a display, and can
run over SSH:

```bash
ctest --test-dir build --output-on-failure
```

There is also a standalone path that needs **no CMake, no NASM and no
dependencies at all** — useful for CI or for bisecting a kernel regression:

```bash
./scripts/selftest.sh
```

---

## The five components

### 1. Hardware interrogator — `hardware_info.{hpp,cpp}`

Reports CPU model, physical/logical core counts, max clock, the L1d/L1i/L2/L3
cache sizes, cache line size and total RAM.

Two interchangeable providers fill one `HardwareSnapshot`: hwinfo (default) and a
native provider that is *always* compiled. hwinfo overlays the native values and
leaves any field it cannot supply untouched, so the dashboard degrades gracefully
rather than going blank if hwinfo's API drifts or the library is unavailable.

On heterogeneous ARM CPUs the snapshot additionally breaks out per-cluster core
counts and cache geometry (`hw.perflevelN.*`), because a single "core count"
hides the fact that which cluster the OS scheduled you onto changes your latency.

### 2. Math kernels

All five kernels implement one signature:

```c
void kernel(uint16_t* out, const KernelParams* p, int32_t row_begin, int32_t row_end);
```

`KernelParams` is exactly one cache line, and the assembly kernels index it by
hard-coded byte offset — so `kernel_api.hpp` carries a `static_assert` per field
guarding those offsets. Move a field and the C++ build breaks before the assembly
silently reads the wrong value.

The `[row_begin, row_end)` band is how the same kernels are handed disjoint
horizontal strips across threads, with no signature change — see "Multi-core
rendering". The self-test verifies that a banded pair of calls equals a single
whole-image call, which is what made threading safe to add.

Row widths are padded up to the widest SIMD batch (`vec_width`), so **no kernel
needs a scalar tail loop** — a tail loop would be dead weight on the hot path and
a second code path to keep bit-exact.

### 3. Build configuration — `CMakeLists.txt`

`enable_language(ASM_NASM)` on x86-64, with `src/asm/mandelbrot_avx2.asm` added
directly to the target's source list. Applies `-O3 -march=native -ffast-math
-DNDEBUG`.

**`-ffast-math` needs two corrections, and this is not cosmetic.** Both were
caught by the self-test:

1. **`-ffp-contract=off`.** `-ffast-math` implies `-ffp-contract=fast`, and the
   backend then fuses mul+add into `fmadd`/`fmla`/`vfmadd` *regardless of any
   source-level pragma*. The "scalar" reference was measured emitting 2 `fmadd`
   instructions with `#pragma clang fp contract(off)` in force. Since the
   hand-written assembly contains no FMA, this alone breaks bit-exactness.
2. **`-fno-associative-math`.** `-ffast-math` permits reassociation, which
   rewrites `cx0 + (x + lane)*dx` into `(cx0 + x*dx) + lane*dx` so the invariant
   part can be hoisted out of the loop. Algebraically identical, different
   rounding: **14,925 mismatched pixels** on a boundary zoom, with contraction
   already disabled.

Everything else `-ffast-math` enables is retained and harmless here — the kernels
contain no division and no transcendentals.

### 4. Statistical profiler — `profiler.hpp`

- **Sample size.** A single timing is a sample of size one from a distribution
  whose tail is shaped by the OS scheduler, not by the hardware. The profiler
  collects n ≥ 30 before flagging a measurement as trustworthy (the conventional
  threshold for treating the sampling distribution of the mean as approximately
  normal under the CLT) and reports n and the CLT status in the UI.
- **Trimming.** Execution-time distributions are right-skewed: a preempted run
  can be 10× the median, but nothing runs faster than the hardware allows. The
  profiler drops the **slowest 5%** and reports raw *and* trimmed statistics side
  by side, plus how much of the raw mean was scheduler noise. This is an
  upper-tail trim, deliberately not a symmetric trimmed mean — the fast tail is
  signal.
- **Dispersion.** Unbiased sample standard deviation (n−1), coefficient of
  variation, SEM, a 95% CI half-width, and min/median/p95/p99.
- **Timing** uses `std::chrono::high_resolution_clock` as specified. That clock
  is *permitted* to alias the non-monotonic `system_clock`, so the dashboard
  reports `is_steady` and the measured granularity rather than assuming.

The sweep runs warm-up iterations before sampling, discarding them: they pay for
cold caches, first-touch page faults on the output buffer, and frequency ramp.

### 5. GUI — `main.cpp`

Dear ImGui + SDL2 (`SDL_Renderer` backend, so there is no OpenGL loader to
manage). The dashboard overlays the live fractal and provides the kernel
drop-down, live latency and standard deviation, a frame counter, the hardware
panel, a progress-barred statistical sweep, and a correctness check.

Only **one kernel invocation** is inside the timed region. Colourisation, texture
upload and UI are deliberately outside it: they cost the same for every kernel and
would dilute the comparison.

The benchmark sweep is a **time-budgeted state machine** rather than a blocking
loop, so a 1024-sample sweep of a slow scalar kernel keeps the window responsive
and the progress bar live.

Controls: drag to pan, scroll to zoom, keys `1`–`N` to switch kernel, **space to
pause**, `Esc` to quit.

**Pause, not stop/start.** There is no run to end — the benchmark is a continuous
loop — so "stop" would imply tearing something down that does not exist. Pause
freezes the measurement and the image while leaving the window fully interactive,
and it stops the kernel pegging a core, which on a heavy view at a few FPS is the
difference between a warm laptop and a hot one. Alongside it: **Step** (render one
frame while frozen, deliberately unsampled, so a single cold frame does not
pollute the window) and **Reset stats**.

While paused, changing the view or settings still re-renders a single frame so the
picture matches the controls, but takes no sample. Paused frames are also excluded
from the FPS window — otherwise pausing would report a flattering frame rate for
work never done.

### Multi-core rendering

`src/thread_pool.{hpp,cpp}` splits each frame across threads. This is what the
`[row_begin, row_end)` parameters on every kernel signature were reserved for, and
the self-test's "banded call equals whole-image call" check is what made it safe to
add. The `CPU threads` slider selects 1..physical-cores.

Measured on this machine (Apple M5 Pro, 15 cores, AArch64 assembly kernel,
640×360, max_iter 512):

| threads | ms | MIter/s | speedup | verdict shown |
|---|---|---|---|---|
| 1 | 15.62 | 2068 | 1.00x | Very Good |
| 2 | 8.04 | 4021 | 1.94x | Amazing |
| 4 | 4.15 | 7787 | 3.77x | Legendary |
| 8 | 2.14 | 15087 | 7.29x | Godly |
| 15 | 1.24 | 26121 | **12.63x** | Godly |

12.63x on 15 cores is ~84% scaling efficiency. It is not 15x, and the reasons are
worth knowing: this part is heterogeneous (5 "Super" + 10 "Performance" cores, per
`hw.perflevelN`), so the slower cluster caps the frame; and at 1.24 ms the pool's
wake/complete round trip is a measurable fraction of the frame.

Design decisions that matter:

- **Dynamic row stealing, not static bands.** Mandelbrot rows differ enormously in
  cost — a row through the centre of the set runs to `max_iter` on most pixels
  while a row above it escapes immediately. Splitting into N equal contiguous
  bands would leave most threads idle waiting for whoever got the middle. Workers
  instead pull one row at a time from a shared atomic counter, which self-balances
  regardless of where the expensive rows are *and* absorbs the heterogeneous
  cores: a thread on a slower core simply completes fewer rows. The dashboard
  shows the resulting rows-per-thread spread so the balancing is visible rather
  than asserted.
- **The calling thread participates**, so `threads == 1` takes a fast path with no
  atomics, no wakeups and no mutex at all — a direct kernel call, identical in
  cost to not having a pool. That keeps single-threaded numbers comparable with
  every measurement taken before threading existed.
- **Spin, then park.** Workers spin on a generation counter before falling back to
  a condition variable, because in continuous rendering the next frame usually
  arrives before a futex round trip would complete. Parking still happens when
  idle or paused, so the app does not burn 15 cores sitting still.
- **No per-frame allocation.** Threads spawn once; the job descriptor, row counter
  and per-thread tallies are fixed-size members. The zero-allocation tripwire
  still reads 0 with threading on.
- **Cache-line isolation.** The row counter, completion counter, generation
  counter and each thread's row tally are separately `alignas`-ed. These are
  precisely the "thread-shared variables" the alignment requirement exists for:
  without it, every row steal would invalidate the completion counter's line on
  every core.

**Correctness is asserted, speedup is only reported.** The self-test verifies that
every kernel at 2, 3, 5, 8, all-cores and an over-subscribed count produces
**bit-identical** output to the single-threaded result, that row tallies sum
exactly to the image height (every row claimed exactly once), and that 200
consecutive frames at varying thread counts stay exact — which is what catches a
generation- or completion-counter race that only appears after many cycles.
Speedup itself is printed but not asserted, because a threshold would be flaky on
a shared machine.

### Statistics invalidation

Averaging frames that rendered different amounts of work produces a mean that
describes neither. `RenderFractal()` therefore compares a `WorkloadSignature`
(resolution, max_iter, centre, span) each frame and discards the latency window
when it changes.

This replaced a set of scattered manual `rolling_.Reset()` calls that covered the
sliders and combos but **not mouse pan and zoom** — so panning left a stale window
while the current frame's iteration count updated immediately, and the resulting
throughput figure was silently wrong. It is also why the simple panel briefly
reported "1.2x faster" for a kernel measured at 3.97x on that same view.

### The "How fast is this?" panel

A second, smaller window pinned to the top-right translates the same measurements
into plain language, for anyone who does not want to read a confidence interval.
It shows a single-word verdict on a 14-tier scale from **Absolute Garbage** to
**Godly**, a log-scaled bar, the speedup over the scalar baseline, a smoothness
rating from FPS, and a measurement-quality rating from the coefficient of
variation. Expanding "Show the whole scale" prints every tier and its threshold,
so the verdict is not a black box.

**The grade is driven by pixel-iterations per second, not pixels per second.**
That distinction is the whole design. Measured with the same three kernels across
five very different views:

| metric | scalar range | NEON range | spread |
|---|---|---|---|
| Mpixel/s | 2.9 – 1181 | 9.3 – 3088 | ~400x |
| **MIter/s** | **542 – 590** | **1720 – 2242** | **~1.3x** |

A rating built on pixels/second would call the same kernel "Godly" on a
zoomed-out low-iteration view and "Absolute Garbage" on a deep boundary zoom,
while the hardware did identical work. Iterations/second does not have that
defect, so the tiers are cut against it and the verdict tracks the *kernel*
rather than the *view*.

Thresholds are calibrated against real measurements on the development machine,
not invented:

| scenario | MIter/s | verdict |
|---|---|---|
| Debug (`-O0`) build, any kernel | 150 – 180 | Potato |
| Release scalar C++ | 542 – 590 | Poor |
| Release NEON, deep zoom | 1726 | Good |
| Release NEON, typical view | 2052 – 2242 | Very Good |
| Release NEON, trivial view | 3088 | Great |
| 8-wide AVX2 (projected) | ~4400 | Elite |
| 8 threads (measured) | 15087 | Godly |
| all 15 cores (measured) | 26121 | Godly |

The top tiers are out of reach for a single narrow-SIMD core by design, and are
earned by finishing the optimisation work rather than by owning a fast laptop.
They are now genuinely reachable: the `CPU threads` slider walks the scale from
Very Good at 1 thread to Godly at 8, all measured rather than projected. Note also that a Debug build lands in
*Potato*, which is a second, independent signal that you are not benchmarking
what you think you are.

`SumIterations()` runs outside the timed region, so computing this figure never
inflates a latency sample.

**What the verdict is and is not rating.** It rates the *kernel* on *one core*.
With the `CPU threads` slider at its default of 1, on a 15-core machine it is
exercising roughly a fifteenth of the chip — "Very Good" means the kernel is close
to what one core can do at this vector width, not that the machine is very good.
Turn the slider up and the same panel starts rating the machine instead, and stops
projecting because it is then measuring.

Because that is so easy to misread, the panel carries a **"Your computer"**
section that says it outright, in plain English:

```
Apple M5 Pro
   15 cores, 24.0 GiB memory
   Can do 4 numbers per instruction

What this test is using
   1 core out of 15
   That is 7% of your processor
   14 cores are sitting idle

So how is your computer doing?
   The one core doing the work is performing well. The rating
   above is about that core, not the whole machine.
   All 15 cores could reach roughly 33 billion steps/sec
   Estimate, not measured.

Memory
   This frame writes 4.0 MiB
   Your fastest big cache is 8.0 MiB
   Fits in cache, which is ideal
```

The all-core figure is a **linear projection and is labelled as one everywhere it
appears**. Real scaling is always lower — shared L2 and memory bandwidth, power
and thermal limits, and on heterogeneous ARM parts the fact that not all cores run
at the same speed. Presenting it as a measurement would be dishonest, so the UI
never does.

The memory line is a genuine, actionable observation rather than filler: whether
the output buffer fits in the largest cache measurably affects throughput, and
changing the resolution dropdown lets you see it. On this machine every supported
resolution fits (4.0 MiB at 1920×1080 against an 8 MiB L2), which is itself worth
knowing — it means these numbers are not memory-bound.

Every tier also carries its own one-line description, shown for all fourteen at
once under "Show the whole scale" so the verdict can be placed in context rather
than taken on faith.

The "compared to the simple version" figure prefers the **sweep**, because the
sweep measures every kernel back to back on identical parameters. A live
cross-check against a remembered scalar baseline is only shown when both rates
were captured on the same workload; otherwise the panel says to run a sweep rather
than print a number it cannot stand behind.

---

## Constraint: zero heap allocation on hot paths

Rather than assert this in a comment, the suite **proves it at runtime**.
`alloc_tracker.cpp` replaces every global `operator new`/`delete` overload
(including the sized and aligned C++17 forms) with a counting wrapper. The render
loop measures the allocation delta across the measured region and displays it —
it reads `0`, in green, and turns red if it ever doesn't.

The counters are kept **both** process-wide (atomic) and per-thread
(`thread_local`), and `AllocationScope` reads the per-thread ones. That
distinction is load-bearing: with process-wide counting the scope also caught
allocations made by SDL's Cocoa/display thread while the kernel was running, and
the GUI reported a false `12 allocations in measured region -- regression!` even
though the kernel allocated nothing. A measurement that blames your code for
another thread's work is worse than no measurement, so the scope is thread-scoped
and `tests/verify_kernels.cpp` has a regression test that hammers allocations from
a second thread and asserts the scope still reads 0.

All pixel buffers live in a single statically allocated `RenderTarget` (~16 MiB
in BSS): the iteration buffer, a reference buffer for verification, the packed
pixel buffer, and the palette LUT. Nothing is ever resized; changing resolution
only changes how much of each buffer is used. The profiler's sample window and
its sort scratch are fixed-capacity members, and `std::sort` is in-place.

## Constraint: memory alignment

Buffers and the shared counters are aligned to the destructive-interference size,
which is ≥ 64 bytes on every target and therefore always satisfies the mandated
`alignas(64)`.

Worth calling out: **Apple Silicon reports a 128-byte cache line**
(`hw.cachelinesize == 128`). Aligning to 64 there would still let two
"isolated" objects share one physical line and false-share, so `arch.hpp` uses
128 on arm64 and 64 elsewhere. The dashboard shows the real line size so the
discrepancy is visible rather than assumed away.

---

## Verification status

This was developed on an Apple M5 Pro (arm64), which shapes what could be
executed versus only statically verified. Being precise about the difference:

**The documented CMake build is verified end to end** on arm64 with CMake 4.3.1 +
Ninja: configure resolves `SDL2::SDL2`, Dear ImGui and `lfreist-hwinfo::hwinfo`,
189/189 targets build with no warnings from project code, the binary reports
`Source: hwinfo (lfreist) + native fallback`, and `ctest` passes.

Two notes on the dependency pins, both learned the hard way:

- **hwinfo publishes no release tags.** The repository has only a `c++11` tag and
  a moving `main` branch, so `Dependencies.cmake` pins a commit SHA and sets
  `GIT_SHALLOW FALSE` (a shallow clone cannot check out an arbitrary SHA).
- **hwinfo's option names and API are not what its README suggests.** Cache sizes
  and clocks live on `CPU::Core` (`core.cache.l1_data`, `core.max_frequency_hz`),
  not on `CPU`; RAM is `Memory::size()`/`available()`, not `total_Bytes()`. It
  also returns `-1` sentinels for values a platform does not expose, which wrap
  to ~1.8e19 in an unsigned field — `hardware_info.cpp` range-checks every value
  before trusting it. On macOS hwinfo leaves most cache fields unpopulated, which
  is exactly why the native provider is always compiled in underneath.

**Executed and passing on arm64** — all 60 self-test checks:

- All three arm64 kernels (scalar, NEON intrinsics, AArch64 assembly) produce
  **bit-identical** output across five views, including a deep boundary zoom, a
  region entirely outside the set, `max_iter == 0`, and a non-multiple-of-8
  width.
- Banded (`row_begin`/`row_end`) calls equal whole-image calls for every kernel.
- Profiler statistics checked against hand-computed values (mean, unbiased
  stddev, median, interpolated p95, the 5% upper-tail trim, SEM), plus the
  scheduler-noise case: 39 samples at 100 ns with one 100 µs spike recovers a
  trimmed mean of exactly 100 ns.
- Zero heap allocation across kernel + profiler + colourise, with the counter
  first proven live by a deliberate `new[]`.
- The full GUI application links and runs its render loop headlessly
  (`SDL_VIDEODRIVER=dummy`) with no errors, against real SDL2 2.30.9 and Dear
  ImGui v1.91.5.

**Statically verified for x86-64, not executed** (this machine cannot run
x86-64; Rosetta is not installed and does not support AVX2 for native binaries):

- `mandelbrot_avx2.asm` assembles cleanly under NASM 2.16.03 for **both**
  `macho64` and `elf64`, exporting `_mandelbrot_avx2_nasm` and
  `mandelbrot_avx2_nasm` respectively; the `win64` guard fires as intended.
- Disassembly confirms the intended instruction sequence, including
  `vcmple_oqps` for the non-signalling compare and `vpermq $0x8` decoding as
  `[0,2,0,0]` — the lane-order fix for the `uint16` pack.
- **Zero** `push`/`pop`/`leave` instructions and no `rsp`/`rbp` references,
  confirming the no-stack-frame claim rather than just asserting it.
- Every x86-64 translation unit compiles warning-free under `-Wall -Wextra`,
  and the x86-64 self-test **links** against the NASM object — so the
  `extern "C"` declaration matches the assembly symbol.
- The AVX2 intrinsics kernel was confirmed to emit **no** FMA instructions.

What this does not prove: that the NASM kernel computes correct *values* on real
hardware. Its operation order is identical to the AArch64 sibling that is
verified bit-exact, and `tests/verify_kernels.cpp` will check it the moment it is
run on an x86-64 box — that is the first thing to do there.

On x86-64 macOS the linker emits `no platform load command found in
mandelbrot_avx2.asm.o`. It is harmless (NASM does not write `LC_BUILD_VERSION`;
the linker assumes macOS) and does not occur on Linux.

## Notes and honest limitations

- **Single-precision limits zoom depth.** The kernels are `float` by design. Past
  roughly 10⁻⁵ span the image is quantised by f32, not by `max_iter`. The UI
  reports float ULPs per pixel and refuses to zoom below 0.5 ULP/pixel instead of
  silently rendering blocks.
- **Threading is opt-in and defaults to 1.** The `CPU threads` slider goes from 1
  to your physical core count. It defaults to 1 so the out-of-the-box reading is
  the per-core ISA figure, which is what the kernel comparison is about; turning
  it up measures the machine instead. See "Multi-core rendering" below.
- **The NASM kernel could go faster.** It runs one dependency chain per batch. The
  obvious next step is 2× unrolling to overlap two independent chains and hide
  FP latency; the same applies to the NEON kernels.
- **Interesting codegen result.** Clang independently chose `vtestps` for the
  mask early-out in the AVX2 intrinsics kernel — the same instruction the NASM
  kernel uses by hand, and for the same reason (it keeps the early-out off the
  general-purpose register file). Clang also packs the `uint16` store more
  cleverly than the hand-written version, using a 128-bit `vpackusdw` and
  avoiding the `vpermq` lane fixup.
- **HiDPI is off.** The window is created without `SDL_WINDOW_ALLOW_HIGHDPI` to
  keep one coordinate system for the fractal blit and the mouse mapping. The
  fractal is upscaled from the selected render resolution regardless.

## Layout

```
CMakeLists.txt              build config, NASM enablement, FP flag rationale
cmake/Dependencies.cmake    SDL2 / ImGui / hwinfo acquisition
scripts/selftest.sh         dependency-free correctness + stats check
src/
  arch.hpp                  ISA detection, cache-line constants
  kernel_api.hpp            the ABI contract + offset static_asserts
  kernels_scalar.cpp        Kernel A  (reference)
  kernels_avx2.cpp          Kernel B  (x86-64)
  kernels_neon.cpp          Kernel B' (arm64)
  asm/mandelbrot_avx2.asm   Kernel C  (NASM, x86-64)
  asm/mandelbrot_neon.S     Kernel C' (AArch64)
  cpu_features.{hpp,cpp}    CPUID / XGETBV runtime gating
  hardware_info.{hpp,cpp}   Component 1
  profiler.hpp              Component 4
  timing.{hpp,cpp}          nanosecond clock + granularity probe
  alloc_tracker.{hpp,cpp}   zero-allocation tripwire
  render_target.{hpp,cpp}   static buffers, palette, verification diff
  kernel_registry.{hpp,cpp} available-kernel list
  performance_grade.{hpp,cpp} plain-language rating tiers, calibrated
  thread_pool.{hpp,cpp}     multi-core row-stealing frame renderer
  main.cpp                  Component 5
tests/verify_kernels.cpp    bit-exactness + hand-checked statistics
```

---

## License

Released into the public domain under [The Unlicense](LICENSE) — do whatever you
want with it, no attribution required.

The dependencies are fetched at build time rather than vendored, so this repository
contains none of their code. If you distribute a compiled binary, note that their
licenses do ask for attribution in that case:

| Dependency | License |
|---|---|
| SDL2 | zlib |
| Dear ImGui | MIT |
| hwinfo | MIT |
| NASM | BSD-2-Clause (build tool only, never linked in) |
