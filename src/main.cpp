// main.cpp — Component 5: the GUI, the render loop and the benchmark driver.
//
// Structure of a frame:
//
//   1. drain SDL events, apply pan/zoom
//   2. build KernelParams for the current view
//   3. time exactly one kernel invocation, inside an AllocationScope
//   4. colourise via LUT, upload to the streaming texture
//   5. draw the fractal, then the ImGui dashboard on top
//
// Only step 3 is measured. Colourisation, texture upload and UI are deliberately
// outside the timed region: they are the same cost for every kernel and would
// dilute the thing being compared.
//
// Hot-path allocation: steps 2-4 touch nothing but pre-allocated static buffers.
// The AllocationScope in step 3 proves it at runtime rather than asserting it in
// a comment — the dashboard shows the measured count, which should read 0.

#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "alloc_tracker.hpp"
#include "arch.hpp"
#include "cpu_features.hpp"
#include "hardware_info.hpp"
#include "kernel_api.hpp"
#include "kernel_registry.hpp"
#include "performance_grade.hpp"
#include "profiler.hpp"
#include "render_target.hpp"
#include "thread_pool.hpp"
#include "timing.hpp"

namespace {

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------
struct Resolution {
  const char* label;
  int width;
  int height;
};

// All widths are multiples of 8 so the padded vector width equals the visible
// width and no kernel wastes work on padding columns.
constexpr Resolution kResolutions[] = {
    {"480 x 270", 480, 270},   {"640 x 360", 640, 360},   {"960 x 540", 960, 540},
    {"1280 x 720", 1280, 720}, {"1920 x 1080", 1920, 1080},
};
constexpr int kResolutionCount = static_cast<int>(sizeof(kResolutions) / sizeof(kResolutions[0]));

struct ZoomTarget {
  const char* label;
  double center_x;
  double center_y;
  double span_x;
  int max_iter;
};

constexpr ZoomTarget kZoomTargets[] = {
    {"Full set", -0.6, 0.0, 3.2, 256},
    {"Seahorse Valley", -0.745, 0.1015, 0.05, 512},
    {"Elephant Valley", 0.2925, 0.0195, 0.03, 512},
    {"Triple Spiral", -0.0885, 0.6545, 0.025, 1024},
    {"Mini Mandelbrot", -1.74995, 0.0, 0.0012, 2048},
};
constexpr int kZoomTargetCount =
    static_cast<int>(sizeof(kZoomTargets) / sizeof(kZoomTargets[0]));

// ---------------------------------------------------------------------------
// View -> KernelParams
// ---------------------------------------------------------------------------
struct View {
  double center_x = kZoomTargets[0].center_x;
  double center_y = kZoomTargets[0].center_y;
  double span_x = kZoomTargets[0].span_x;
  int max_iter = kZoomTargets[0].max_iter;
};

// Everything that changes how much work a kernel does. Two jobs:
//
//   1. Invalidating the latency window when it changes. Averaging frames that
//      rendered different amounts of work yields a mean describing neither, and
//      dividing the *current* frame's iteration count by a stale mean produces a
//      throughput figure that is simply wrong. This previously happened on every
//      mouse pan and zoom, because those paths did not reset the window -- the
//      scattered manual resets covered the sliders and combos but not the mouse.
//   2. Deciding whether a stored scalar baseline is still comparable. Comparing
//      rates captured on different views is unreliable: scalar's own throughput
//      was measured anywhere from 542 to 1181 MIter/s depending on the view.
struct WorkloadSignature {
  int resolution_index = -1;
  int max_iter = -1;
  // Thread count belongs here even though it does not change a single pixel: it
  // changes timing completely, so old samples are not comparable, and a scalar
  // baseline taken at 1 thread must not be compared against 15.
  int thread_count = -1;
  double center_x = 0.0;
  double center_y = 0.0;
  double span_x = 0.0;

  bool operator==(const WorkloadSignature&) const = default;
};

// Workload plus what else affects the final image. Palette changes need a
// re-render but must not discard timing statistics, since they do not touch the
// kernel's arithmetic.
struct RenderSignature {
  WorkloadSignature workload{};
  int palette = -1;
  int kernel = -1;

  bool operator==(const RenderSignature&) const = default;
};

mb::KernelParams BuildParams(const View& v, int width, int height) {
  mb::KernelParams p{};

  // Square pixels: one step size for both axes.
  const double step = v.span_x / static_cast<double>(width);

  p.dx = static_cast<float>(step);
  p.dy = static_cast<float>(step);
  p.cx0 = static_cast<float>(v.center_x - v.span_x * 0.5);
  p.cy0 = static_cast<float>(v.center_y - step * static_cast<double>(height) * 0.5);
  p.escape_r2 = 4.0f;
  p.width = width;
  p.vec_width = mb::PaddedWidth(width);
  p.height = height;
  p.max_iter = std::clamp(v.max_iter, 0, mb::kMaxIterationCap);
  p.stride = mb::kRenderStride;
  return p;
}

// How many float ULPs separate two adjacent pixels. Below ~1 the render is
// quantised by single-precision, not by the kernel, and further zooming shows
// blocky artefacts rather than more detail.
double UlpsPerPixel(const View& v, int width) {
  const double step = v.span_x / static_cast<double>(width);
  const double magnitude =
      std::max({1.0, std::fabs(v.center_x), std::fabs(v.center_y)});
  const double ulp = static_cast<double>(std::numeric_limits<float>::epsilon()) * magnitude;
  return ulp > 0.0 ? step / ulp : 0.0;
}

// ---------------------------------------------------------------------------
// Benchmark sweep — a time-budgeted state machine
// ---------------------------------------------------------------------------
// Running a full sweep synchronously would freeze the window for many seconds.
// Instead Step() runs as many samples as fit in a per-frame budget, so the UI
// keeps painting and the progress bar stays live. Sample buffers are fixed-size
// members, so a sweep allocates nothing.
class BenchmarkSweep {
 public:
  static constexpr std::size_t kSampleCapacity = 2048;

  bool running() const { return phase_ == Phase::Warmup || phase_ == Phase::Sampling; }
  bool has_results() const { return any_results_; }
  int warmup_target() const { return warmup_target_; }
  int sample_target() const { return sample_target_; }

  void Configure(int warmup, int samples) {
    warmup_target_ = std::clamp(warmup, 0, 512);
    sample_target_ =
        std::clamp(samples, static_cast<int>(mb::kCltMinSamples), static_cast<int>(kSampleCapacity));
  }

  void Start(const mb::KernelParams& params, int thread_count) {
    params_ = params;
    thread_count_ = thread_count;
    kernel_ = 0;
    completed_in_phase_ = 0;
    any_results_ = false;
    for (int i = 0; i < mb::kMaxKernels; ++i) {
      profilers_[i].Reset();
      has_stats_[i] = false;
    }
    phase_ = Phase::Warmup;
    AdvanceToNextRunnableKernel();
  }

  void Cancel() { phase_ = Phase::Idle; }

  // Fraction of total work completed, for the progress bar.
  float progress() const {
    const auto& reg = mb::Kernels();
    int runnable = 0;
    for (int i = 0; i < reg.count; ++i) {
      if (reg[i].available) {
        ++runnable;
      }
    }
    if (runnable == 0 || phase_ == Phase::Idle) {
      return 0.0f;
    }
    const int per_kernel = warmup_target_ + sample_target_;
    int done = 0;
    for (int i = 0; i < kernel_ && i < reg.count; ++i) {
      if (reg[i].available) {
        done += per_kernel;
      }
    }
    done += completed_in_phase_ + (phase_ == Phase::Sampling ? warmup_target_ : 0);
    return static_cast<float>(done) / static_cast<float>(runnable * per_kernel);
  }

  const char* current_kernel_name() const {
    const auto& reg = mb::Kernels();
    return (kernel_ >= 0 && kernel_ < reg.count) ? reg[kernel_].name : "";
  }

  bool stats_for(int kernel_index, mb::LatencyStats& out) const {
    if (kernel_index < 0 || kernel_index >= mb::kMaxKernels || !has_stats_[kernel_index]) {
      return false;
    }
    out = stats_[kernel_index];
    return true;
  }

  const mb::KernelParams& params() const { return params_; }
  int thread_count() const { return thread_count_; }

  void Step(double budget_ms) {
    if (!running()) {
      return;
    }
    const auto& reg = mb::Kernels();
    const mb::Timestamp deadline_base = mb::Now();

    while (running()) {
      if (mb::ElapsedNanos(deadline_base, mb::Now()) > budget_ms * 1e6) {
        return;  // out of budget for this frame
      }
      if (kernel_ >= reg.count) {
        phase_ = Phase::Idle;
        return;
      }

      const mb::KernelDesc& d = reg[kernel_];
      mb::RenderTarget& t = mb::Target();

      if (phase_ == Phase::Warmup) {
        // Warm-up runs are discarded: they pay for cold caches, the first-touch
        // page faults on the output buffer, and any frequency ramp. With threads
        // enabled they also pay for the first wake of every parked worker.
        mb::Pool().RenderFrame(d.fn, t.iterations, &params_, thread_count_);
        if (++completed_in_phase_ >= warmup_target_) {
          phase_ = Phase::Sampling;
          completed_in_phase_ = 0;
        }
        continue;
      }

      const mb::Timestamp t0 = mb::Now();
      mb::Pool().RenderFrame(d.fn, t.iterations, &params_, thread_count_);
      const double ns = mb::ElapsedNanos(t0, mb::Now());
      profilers_[kernel_].Push(ns);

      if (++completed_in_phase_ >= sample_target_) {
        stats_[kernel_] = profilers_[kernel_].Compute();
        has_stats_[kernel_] = true;
        any_results_ = true;
        ++kernel_;
        completed_in_phase_ = 0;
        phase_ = Phase::Warmup;
        AdvanceToNextRunnableKernel();
      }
    }
  }

 private:
  enum class Phase { Idle, Warmup, Sampling };

  void AdvanceToNextRunnableKernel() {
    const auto& reg = mb::Kernels();
    while (kernel_ < reg.count && !reg[kernel_].available) {
      ++kernel_;
    }
    if (kernel_ >= reg.count) {
      phase_ = Phase::Idle;
    }
  }

  Phase phase_ = Phase::Idle;
  int kernel_ = 0;
  int completed_in_phase_ = 0;
  int warmup_target_ = 16;
  int sample_target_ = 128;
  bool any_results_ = false;
  int thread_count_ = 1;
  mb::KernelParams params_{};

  mb::LatencyProfiler<kSampleCapacity> profilers_[mb::kMaxKernels];
  mb::LatencyStats stats_[mb::kMaxKernels]{};
  bool has_stats_[mb::kMaxKernels]{};
};

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------
struct VerificationRow {
  bool valid = false;
  bool exact = false;
  std::uint64_t mismatches = 0;
  int max_diff = 0;
};

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------
class App {
 public:
  bool Init() {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
      std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
      return false;
    }

    window_ = SDL_CreateWindow("Mandelbrot Benchmark — Scalar vs SIMD vs Assembly",
                               SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1440, 900,
                               SDL_WINDOW_RESIZABLE);
    if (window_ == nullptr) {
      std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
      return false;
    }

    renderer_ = SDL_CreateRenderer(window_, -1,
                                   SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer_ == nullptr) {
      // Fall back to software rather than refusing to start: the point of the
      // app is the CPU kernels, not the presentation path.
      renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
    }
    if (renderer_ == nullptr) {
      std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
      return false;
    }

    // One streaming texture, created at the maximum size and never recreated.
    // Its pitch matches kRenderStride so the row layout is identical to the
    // pixel buffer regardless of the selected resolution.
    texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ABGR8888,
                                 SDL_TEXTUREACCESS_STREAMING, mb::kRenderStride,
                                 mb::kMaxRenderHeight);
    if (texture_ == nullptr) {
      std::fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
      return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini side effects
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.93f;  // legible over the fractal

    if (!ImGui_ImplSDL2_InitForSDLRenderer(window_, renderer_)) {
      std::fprintf(stderr, "ImGui_ImplSDL2_InitForSDLRenderer failed\n");
      return false;
    }
    if (!ImGui_ImplSDLRenderer2_Init(renderer_)) {
      std::fprintf(stderr, "ImGui_ImplSDLRenderer2_Init failed\n");
      return false;
    }

    kernel_index_ = mb::Kernels().default_index;
    sweep_.Configure(16, 128);

    // One worker per physical core. Logical cores are deliberately not used as
    // the cap: this kernel is pure FP throughput with no memory stalls to hide,
    // so SMT siblings contend for the same vector units and add variance without
    // adding much throughput.
    const mb::HardwareSnapshot& hw = mb::QueryHardware();
    int cores = hw.physical_cores > 0 ? hw.physical_cores : 1;
    if (cores > mb::kMaxRenderThreads) {
      cores = mb::kMaxRenderThreads;
    }
    mb::Pool().Start(cores);
    max_thread_count_ = mb::Pool().max_thread_count();
    thread_count_ = 1;

    // Touch the hardware snapshot and clock probe now, while allocating is
    // still allowed, so neither happens mid-render.
    mb::QueryHardware();
    mb::MeasuredClockGranularityNanos();
    mb::EnsurePaletteLut(mb::Target(), palette_, view_.max_iter);

    return true;
  }

  void Shutdown() {
    // Join the workers before tearing down anything they could touch.
    mb::Pool().Stop();
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    if (texture_ != nullptr) {
      SDL_DestroyTexture(texture_);
    }
    if (renderer_ != nullptr) {
      SDL_DestroyRenderer(renderer_);
    }
    if (window_ != nullptr) {
      SDL_DestroyWindow(window_);
    }
    SDL_Quit();
  }

  void Run() {
    while (!quit_) {
      const mb::Timestamp frame_start = mb::Now();
      HandleEvents();
      if (quit_) {
        break;
      }
      const bool sampled = RenderFractal();
      DrawUi();
      Present();
      ++frame_counter_;
      // Paused frames are not representative frame times -- they skip the
      // kernel entirely -- so they must not enter the FPS window, or pausing
      // would report a flattering frame rate for work never done.
      if (sampled) {
        frame_time_.Push(mb::ElapsedNanos(frame_start, mb::Now()));
      }
    }
  }

 private:
  // -------------------------------------------------------------------------
  // Geometry: where the fractal is drawn inside the window, letterboxed to
  // preserve the render aspect ratio.
  // -------------------------------------------------------------------------
  SDL_Rect FractalRect() const {
    int win_w = 0;
    int win_h = 0;
    SDL_GetRendererOutputSize(renderer_, &win_w, &win_h);

    const Resolution& res = kResolutions[resolution_index_];
    const double src_aspect = static_cast<double>(res.width) / static_cast<double>(res.height);
    const double win_aspect = static_cast<double>(win_w) / static_cast<double>(win_h);

    SDL_Rect r;
    if (win_aspect > src_aspect) {
      r.h = win_h;
      r.w = static_cast<int>(std::lround(win_h * src_aspect));
      r.x = (win_w - r.w) / 2;
      r.y = 0;
    } else {
      r.w = win_w;
      r.h = static_cast<int>(std::lround(win_w / src_aspect));
      r.x = 0;
      r.y = (win_h - r.h) / 2;
    }
    return r;
  }

  // Window point -> complex plane.
  void WindowToComplex(int px, int py, double& out_x, double& out_y) const {
    const SDL_Rect r = FractalRect();
    const Resolution& res = kResolutions[resolution_index_];
    const double u = r.w > 0 ? (static_cast<double>(px - r.x) / r.w) : 0.5;
    const double v = r.h > 0 ? (static_cast<double>(py - r.y) / r.h) : 0.5;

    const double step = view_.span_x / static_cast<double>(res.width);
    const double span_y = step * static_cast<double>(res.height);

    out_x = (view_.center_x - view_.span_x * 0.5) + u * view_.span_x;
    out_y = (view_.center_y - span_y * 0.5) + v * span_y;
  }

  void HandleEvents() {
    ImGuiIO& io = ImGui::GetIO();
    SDL_Event event;

    while (SDL_PollEvent(&event) != 0) {
      ImGui_ImplSDL2_ProcessEvent(&event);

      switch (event.type) {
        case SDL_QUIT:
          quit_ = true;
          return;

        case SDL_WINDOWEVENT:
          if (event.window.event == SDL_WINDOWEVENT_CLOSE &&
              event.window.windowID == SDL_GetWindowID(window_)) {
            quit_ = true;
            return;
          }
          break;

        case SDL_KEYDOWN:
          if (io.WantCaptureKeyboard) {
            break;
          }
          if (event.key.keysym.sym == SDLK_ESCAPE) {
            quit_ = true;
            return;
          }
          if (event.key.keysym.sym == SDLK_SPACE) {
            paused_ = !paused_;
            break;
          }
          // Number keys 1..N cycle kernels, matching the dropdown order.
          if (event.key.keysym.sym >= SDLK_1 && event.key.keysym.sym <= SDLK_9) {
            const int wanted = event.key.keysym.sym - SDLK_1;
            if (wanted < mb::Kernels().count && mb::Kernels()[wanted].available) {
              kernel_index_ = wanted;
            }
          }
          break;

        case SDL_MOUSEBUTTONDOWN:
          if (!io.WantCaptureMouse && event.button.button == SDL_BUTTON_LEFT) {
            dragging_ = true;
          }
          break;

        case SDL_MOUSEBUTTONUP:
          if (event.button.button == SDL_BUTTON_LEFT) {
            dragging_ = false;
          }
          break;

        case SDL_MOUSEMOTION:
          if (dragging_ && !io.WantCaptureMouse) {
            const SDL_Rect r = FractalRect();
            if (r.w > 0 && r.h > 0) {
              // Drag moves the image with the cursor, so the view centre moves
              // the opposite way.
              const Resolution& res = kResolutions[resolution_index_];
              const double step = view_.span_x / static_cast<double>(res.width);
              const double span_y = step * static_cast<double>(res.height);
              view_.center_x -= static_cast<double>(event.motion.xrel) / r.w * view_.span_x;
              view_.center_y -= static_cast<double>(event.motion.yrel) / r.h * span_y;
            }
          }
          break;

        case SDL_MOUSEWHEEL: {
          if (io.WantCaptureMouse || event.wheel.y == 0) {
            break;
          }
          int mx = 0;
          int my = 0;
          SDL_GetMouseState(&mx, &my);
          ZoomAt(mx, my, event.wheel.y > 0 ? 0.8 : 1.25);
          break;
        }

        default:
          break;
      }
    }
  }

  // Zoom about a window point, keeping the complex value under the cursor fixed.
  void ZoomAt(int px, int py, double factor) {
    double anchor_x = 0.0;
    double anchor_y = 0.0;
    WindowToComplex(px, py, anchor_x, anchor_y);

    const SDL_Rect r = FractalRect();
    if (r.w <= 0 || r.h <= 0) {
      return;
    }
    const double u = static_cast<double>(px - r.x) / r.w;
    const double v = static_cast<double>(py - r.y) / r.h;

    const Resolution& res = kResolutions[resolution_index_];
    const double new_span_x = view_.span_x * factor;

    // Refuse to zoom past the point where single precision stops resolving
    // adjacent pixels — beyond that the image degrades instead of improving.
    View probe = view_;
    probe.span_x = new_span_x;
    if (factor < 1.0 && UlpsPerPixel(probe, res.width) < 0.5) {
      return;
    }

    const double new_step = new_span_x / static_cast<double>(res.width);
    const double new_span_y = new_step * static_cast<double>(res.height);

    view_.span_x = new_span_x;
    view_.center_x = anchor_x - (u - 0.5) * new_span_x;
    view_.center_y = anchor_y - (v - 0.5) * new_span_y;
  }

  // -------------------------------------------------------------------------
  // The measured region
  // -------------------------------------------------------------------------
  WorkloadSignature CurrentWorkload() const {
    WorkloadSignature s;
    s.resolution_index = resolution_index_;
    s.max_iter = view_.max_iter;
    s.thread_count = thread_count_;
    s.center_x = view_.center_x;
    s.center_y = view_.center_y;
    s.span_x = view_.span_x;
    return s;
  }

  RenderSignature CurrentRender() const {
    RenderSignature s;
    s.workload = CurrentWorkload();
    s.palette = static_cast<int>(palette_);
    s.kernel = kernel_index_;
    return s;
  }

  // Returns true if this frame produced a timing sample.
  bool RenderFractal() {
    const Resolution& res = kResolutions[resolution_index_];
    const mb::KernelRegistry& reg = mb::Kernels();

    if (kernel_index_ < 0 || kernel_index_ >= reg.count || !reg[kernel_index_].available) {
      kernel_index_ = reg.default_index;
    }
    const mb::KernelDesc& kernel = reg[kernel_index_];

    const RenderSignature signature = CurrentRender();

    // Single authoritative invalidation point for the statistics. Covers the
    // sliders, the combos, the preset jumps, kernel switches *and* mouse
    // pan/zoom, which the previous scattered rolling_.Reset() calls missed.
    if (!(signature.workload == last_render_.workload) ||
        signature.kernel != last_render_.kernel) {
      rolling_.Reset();
      live_mega_iters_per_sec_ = 0.0;
      last_iteration_total_ = 0;
    }

    // While paused, re-render only when something visible actually changed, so
    // the view stays interactive without the benchmark loop pegging a core.
    const bool changed = !(signature == last_render_);
    if (paused_ && !changed && !step_once_) {
      return false;
    }
    const bool take_sample = !paused_;
    step_once_ = false;
    last_render_ = signature;

    mb::RenderTarget& t = mb::Target();
    mb::EnsurePaletteLut(t, palette_, view_.max_iter);

    const mb::KernelParams params = BuildParams(view_, res.width, res.height);

    // --- measured: exactly one kernel invocation, nothing else ---------------
    {
      const mb::AllocationScope scope;
      const mb::Timestamp t0 = mb::Now();
      mb::Pool().RenderFrame(kernel.fn, t.iterations, &params, thread_count_);
      const double ns = mb::ElapsedNanos(t0, mb::Now());

      if (take_sample) {
        rolling_.Push(ns);
        hot_path_allocations_ = scope.allocations();
      }
    }
    // -----------------------------------------------------------------------

    // Total iterations actually performed this frame. Deliberately after the
    // timed region: it is the denominator-free half of the throughput figure the
    // simple panel grades on, and it must not be charged to the kernel.
    last_iteration_total_ = mb::SumIterations(t, res.width, res.height);

    mb::Colourize(t, res.width, res.height);

    const SDL_Rect region{0, 0, res.width, res.height};
    SDL_UpdateTexture(texture_, &region, t.pixels,
                      static_cast<int>(mb::kRenderStride * sizeof(std::uint32_t)));

    last_params_ = params;
    return take_sample;
  }

  // Runs every available kernel over the current view and diffs against the
  // scalar reference. Uses the pre-allocated reference buffer, so still no heap
  // traffic; costs one extra pass per kernel, hence a button rather than
  // per-frame.
  void RunVerification() {
    const Resolution& res = kResolutions[resolution_index_];
    const mb::KernelParams params = BuildParams(view_, res.width, res.height);
    const mb::KernelRegistry& reg = mb::Kernels();
    mb::RenderTarget& t = mb::Target();

    mandelbrot_scalar_cpp(t.reference, &params, 0, params.height);

    for (int i = 0; i < mb::kMaxKernels; ++i) {
      verification_[i] = VerificationRow{};
    }

    for (int i = 0; i < reg.count; ++i) {
      if (!reg[i].available) {
        continue;
      }
      reg[i].fn(t.iterations, &params, 0, params.height);
      const mb::VerifyResult r =
          mb::CompareBuffers(t.iterations, t.reference, params.width, params.height,
                             params.stride);

      VerificationRow row;
      row.valid = true;
      row.exact = r.exact();
      row.mismatches = r.mismatched_pixels;
      row.max_diff = r.max_abs_difference;
      verification_[i] = row;
    }
    verification_done_ = true;
    verified_pixels_ = static_cast<std::uint64_t>(params.width) * params.height;

    // The buffer now holds whichever kernel ran last; the next frame overwrites
    // it with the selected kernel's output anyway.
  }

  // -------------------------------------------------------------------------
  // Dashboard
  // -------------------------------------------------------------------------
  void DrawUi() {
    ImGui_ImplSDLRenderer2_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    const mb::LatencyStats live = rolling_.Compute();
    const mb::LatencyStats frame = frame_time_.Compute();
    UpdateThroughputEstimate(live);

    ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(470, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Benchmark Dashboard", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    DrawRunControls();
    DrawKernelSection(live);
    DrawLatencySection(live, frame);
    DrawViewSection();
    DrawSweepSection();
    DrawVerificationSection();
    DrawHardwareSection();

    ImGui::End();

    DrawSimplePanel(live, frame);

    if (sweep_.running()) {
      // Yield most of the frame to the sweep; the UI only needs enough time to
      // stay responsive.
      sweep_.Step(24.0);
    }

    ImGui::Render();
  }

  // Converts the latest latency window plus this frame's iteration count into a
  // workload-independent throughput figure, and remembers the scalar kernel's
  // rate so the simple panel can express a speedup.
  void UpdateThroughputEstimate(const mb::LatencyStats& live) {
    if (live.trimmed_mean_ns > 0.0 && last_iteration_total_ > 0) {
      const double seconds = live.trimmed_mean_ns / 1e9;
      live_mega_iters_per_sec_ =
          static_cast<double>(last_iteration_total_) / seconds / 1e6;
    } else {
      live_mega_iters_per_sec_ = 0.0;
    }

    // Only latch the baseline once the window is statistically meaningful,
    // otherwise a couple of cold-cache frames would set a misleadingly low bar
    // and inflate every speedup shown afterwards. The workload is recorded with
    // it, because a rate captured on one view cannot be compared against a rate
    // captured on another.
    const mb::KernelDesc& active = mb::Kernels()[kernel_index_];
    if (active.fn == &mandelbrot_scalar_cpp && live.clt_satisfied &&
        live_mega_iters_per_sec_ > 0.0) {
      scalar_baseline_mips_ = live_mega_iters_per_sec_;
      scalar_baseline_workload_ = CurrentWorkload();
      scalar_baseline_valid_ = true;
    }
  }

  struct ScalarComparison {
    bool valid = false;
    double ratio = 1.0;
    const char* source = "";
  };

  // How much faster the active kernel is than the scalar reference.
  //
  // The sweep is strongly preferred because it measures every kernel back to
  // back on identical parameters. A live cross-check against a remembered
  // baseline is only trustworthy when both were measured on the same workload,
  // and reporting one when they were not is how this panel came to claim "1.2x
  // faster" for a kernel that is really ~3.97x faster on that view.
  ScalarComparison CompareToScalar(double current_mips) const {
    ScalarComparison c;

    mb::LatencyStats scalar_stats;
    mb::LatencyStats active_stats;
    if (sweep_.stats_for(0, scalar_stats) &&
        sweep_.stats_for(kernel_index_, active_stats) &&
        scalar_stats.trimmed_mean_ns > 0.0 && active_stats.trimmed_mean_ns > 0.0) {
      c.valid = true;
      c.ratio = scalar_stats.trimmed_mean_ns / active_stats.trimmed_mean_ns;
      c.source = "from the last sweep";
      return c;
    }

    if (scalar_baseline_valid_ && scalar_baseline_mips_ > 0.0 && current_mips > 0.0 &&
        scalar_baseline_workload_ == CurrentWorkload()) {
      c.valid = true;
      c.ratio = current_mips / scalar_baseline_mips_;
      c.source = "measured live on this view";
      return c;
    }

    return c;
  }

  // -------------------------------------------------------------------------
  // Simple panel — the same measurements, in plain language
  // -------------------------------------------------------------------------
  // Anchored to the top-right and intentionally jargon-free. The main dashboard
  // answers "what are the numbers"; this answers "is that good or bad", which is
  // the question most people actually have.
  void DrawSimplePanel(const mb::LatencyStats& live, const mb::LatencyStats& frame) {
    // Wider than the numeric dashboard needs: this panel is mostly prose, and
    // per-tier descriptions wrap badly below about this width.
    constexpr float kPanelWidth = 375.0f;
    constexpr float kMargin = 16.0f;

    const ImGuiIO& io = ImGui::GetIO();
    // Pinned every frame rather than ImGuiCond_FirstUseEver so it stays glued to
    // the right edge when the window is resized.
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x - kPanelWidth - kMargin, kMargin), ImGuiCond_Always);
    // Fixed width, height driven by content. SetNextWindowSize would be ignored
    // here: AlwaysAutoResize takes precedence over it, so constraints are the
    // way to pin one axis and let the other grow.
    ImGui::SetNextWindowSizeConstraints(ImVec2(kPanelWidth, 0.0f),
                                        ImVec2(kPanelWidth, FLT_MAX));

    if (!ImGui::Begin("How fast is this?", nullptr,
                      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
                          ImGuiWindowFlags_NoResize)) {
      ImGui::End();
      return;
    }

    const mb::KernelDesc& kernel = mb::Kernels()[kernel_index_];
    const double mips = live_mega_iters_per_sec_;
    const mb::PerformanceGrade grade = mb::GradeIterationRate(mips);

    if (paused_) {
      ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.32f, 1.0f), "PAUSED - numbers frozen");
      ImGui::Spacing();
    }

    // --- the headline verdict ---------------------------------------------
    ImGui::SetWindowFontScale(1.9f);
    ImGui::TextColored(ImVec4(grade.r, grade.g, grade.b, 1.0f), "%s", grade.label);
    ImGui::SetWindowFontScale(1.0f);

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(grade.r, grade.g, grade.b, 1.0f));
    ImGui::ProgressBar(grade.meter, ImVec2(-1.0f, 10.0f), "");
    ImGui::PopStyleColor();

    ImGui::TextWrapped("%s", grade.blurb);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- what is actually being measured ---------------------------------
    ImGui::TextUnformatted("Work done per second");
    if (mips > 0.0) {
      if (mips >= 1000.0) {
        ImGui::Text("   %.2f billion pixel-steps", mips / 1000.0);
      } else {
        ImGui::Text("   %.0f million pixel-steps", mips);
      }
    } else {
      ImGui::TextDisabled("   measuring...");
    }
    ImGui::TextDisabled("This is the score above. It stays honest");
    ImGui::TextDisabled("whether you zoom in or out.");
    ImGui::Spacing();

    // --- plain-language kernel description --------------------------------
    ImGui::TextUnformatted("Method");
    ImGui::Text("   %s", kernel.name);
    if (kernel.lanes > 1) {
      ImGui::Text("   Does %d pixels at the same time", kernel.lanes);
    } else {
      ImGui::TextUnformatted("   Does 1 pixel at a time");
    }
    ImGui::Spacing();

    // --- speed relative to the scalar baseline ---------------------------
    ImGui::TextUnformatted("Compared to the simple version");
    const ScalarComparison cmp = CompareToScalar(mips);
    if (cmp.valid) {
      if (cmp.ratio >= 1.05) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.45f, 1.0f), "   %.1fx faster", cmp.ratio);
      } else if (cmp.ratio <= 0.95) {
        ImGui::TextColored(ImVec4(0.95f, 0.6f, 0.3f, 1.0f), "   %.1fx slower",
                           1.0 / cmp.ratio);
      } else {
        ImGui::TextUnformatted("   about the same");
      }
      ImGui::TextDisabled("   %s", cmp.source);
    } else {
      ImGui::TextDisabled("   Run sweep for a real comparison");
      ImGui::TextDisabled("   (compares all kernels on identical work)");
    }
    ImGui::Spacing();

    // --- responsiveness ---------------------------------------------------
    const double fps = frame.trimmed_mean_ns > 0.0 ? 1e9 / frame.trimmed_mean_ns : 0.0;
    const mb::SimpleRating smooth = mb::RateSmoothness(fps);
    ImGui::TextUnformatted("Feel");
    ImGui::Text("   %.0f frames per second", fps);
    ImGui::TextColored(ImVec4(smooth.r, smooth.g, smooth.b, 1.0f), "   %s", smooth.label);
    ImGui::Spacing();

    // --- how repeatable the measurement is -------------------------------
    const mb::SimpleRating consistency = mb::RateConsistency(live.cv_pct);
    ImGui::TextUnformatted("Measurement quality");
    ImGui::TextColored(ImVec4(consistency.r, consistency.g, consistency.b, 1.0f), "   %s",
                       consistency.label);
    ImGui::TextDisabled("   timings vary by %.1f%%", live.cv_pct);
    if (!live.clt_satisfied) {
      ImGui::TextDisabled("   still collecting samples");
    }
    ImGui::Spacing();

    ImGui::Text("Each frame takes %.1f ms", live.trimmed_mean_ms());
    ImGui::Spacing();

    DrawPlainEnglishMachineReport(mips);

    // --- the full scale, so the verdict is not a black box ---------------
    ImGui::Spacing();
    if (ImGui::TreeNode("Show the whole scale")) {
      ImGui::TextDisabled("Ratings, best to worst.");
      ImGui::TextDisabled("Numbers are millions of pixel-steps/sec.");
      ImGui::Spacing();

      for (int i = mb::kGradeTierCount - 1; i >= 0; --i) {
        const mb::GradeTier tier = static_cast<mb::GradeTier>(i);
        double lower = 0.0;
        double upper = 0.0;
        mb::GradeTierRange(tier, lower, upper);

        const bool active = (tier == grade.tier);
        // Sample the tier's own colour by grading a value inside it.
        const mb::PerformanceGrade sample =
            mb::GradeIterationRate(lower > 0.0 ? lower : 1.0);
        const float dim = active ? 1.0f : 0.60f;

        ImGui::TextColored(ImVec4(sample.r * dim, sample.g * dim, sample.b * dim, 1.0f),
                           "%s%s", active ? "> " : "  ", mb::GradeTierLabel(tier));
        ImGui::SameLine();
        if (upper > 0.0) {
          ImGui::TextDisabled("(%.0f-%.0f)", lower, upper);
        } else {
          ImGui::TextDisabled("(%.0f+)", lower);
        }

        // Short description for every tier, not just the active one.
        ImGui::Indent(18.0f);
        if (active) {
          ImGui::TextWrapped("%s", mb::GradeTierBlurb(tier));
        } else {
          ImGui::TextDisabled("%s", mb::GradeTierBlurb(tier));
        }
        ImGui::Unindent(18.0f);
      }
      ImGui::TreePop();
    }

    ImGui::End();
  }

  // -------------------------------------------------------------------------
  // "Your computer" — plain-English machine context
  // -------------------------------------------------------------------------
  // The grade above is about one kernel on one core. Without this section a
  // reader will naturally take "Very Good" as a verdict on their computer, which
  // it is not. Everything measured is stated as measured; the all-core figure is
  // a projection and is labelled as one, because this benchmark is
  // single-threaded on purpose and never measures it.
  void DrawPlainEnglishMachineReport(double mips) {
    if (!ImGui::CollapsingHeader("Your computer", ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }

    const mb::HardwareSnapshot& hw = mb::QueryHardware();
    const mb::KernelDesc& kernel = mb::Kernels()[kernel_index_];
    const Resolution& res = kResolutions[resolution_index_];

    // --- what it is -------------------------------------------------------
    ImGui::TextWrapped("%s", hw.cpu_model.c_str());
    ImGui::TextDisabled("   %d cores, %s memory", hw.physical_cores,
                        mb::FormatBytes(hw.ram_total_bytes).c_str());
    ImGui::TextDisabled("   Can do %d numbers per instruction",
                        mb::kNativeVectorLanes);
    ImGui::Spacing();

    // --- what this test is using -----------------------------------------
    const mb::MachineUtilisation util =
        mb::DescribeUtilisation(mips, hw.physical_cores, thread_count_);

    ImGui::TextUnformatted("What this test is using");
    ImGui::Text("   %d core%s out of %d", util.cores_used,
                util.cores_used == 1 ? "" : "s", util.cores_total);
    if (util.using_whole_chip) {
      ImGui::TextColored(ImVec4(0.40f, 0.88f, 0.45f, 1.0f),
                         "   That is the whole processor");
    } else {
      ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                         "   That is %.0f%% of your processor", util.percent_of_chip);
      ImGui::TextDisabled("   %d core%s sitting idle", util.cores_total - util.cores_used,
                          (util.cores_total - util.cores_used) == 1 ? " is" : "s are");
      ImGui::TextDisabled("   Drag \"CPU threads\" up to use them");
    }
    ImGui::Spacing();

    // --- honest verdict on the machine -----------------------------------
    ImGui::TextUnformatted("So how is your computer doing?");
    if (mips > 0.0) {
      if (util.using_whole_chip) {
        // Nothing left to extrapolate: this is the machine's real throughput.
        ImGui::TextWrapped(
            "Every core is working, so the number above is your whole processor's "
            "actual speed on this job. Not an estimate.");
      } else if (util.cores_total > 1) {
        ImGui::TextWrapped(
            "The %d core%s doing the work %s performing well. The rating above "
            "covers only those, not the whole machine.",
            util.cores_used, util.cores_used == 1 ? "" : "s",
            util.cores_used == 1 ? "is" : "are");
        if (util.projection_valid) {
          ImGui::Spacing();
          ImGui::Text("   All %d cores could reach roughly", util.cores_total);
          if (util.projected_all_core_mips >= 1000.0) {
            ImGui::Text("   %.0f billion steps/sec",
                        util.projected_all_core_mips / 1000.0);
          } else {
            ImGui::Text("   %.0f million steps/sec", util.projected_all_core_mips);
          }
          ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                             "   Estimate, not measured.");
          ImGui::TextWrapped(
              "   Real scaling is always below %dx. Move the slider to all %d and "
              "the panel will measure it instead of guessing.",
              util.cores_total, util.cores_total);
        }
      } else {
        ImGui::TextWrapped("This machine reports a single core.");
      }
    } else {
      ImGui::TextDisabled("   measuring...");
    }
    ImGui::Spacing();

    // --- a real, measurable observation about memory ----------------------
    // Whether the output buffer fits in cache genuinely affects the numbers, and
    // it is something a reader can act on by changing resolution.
    const std::uint64_t working_set =
        static_cast<std::uint64_t>(res.width) * res.height * sizeof(std::uint16_t);
    const std::uint64_t biggest_cache = hw.l3_bytes > 0 ? hw.l3_bytes : hw.l2_bytes;

    ImGui::TextUnformatted("Memory");
    ImGui::TextDisabled("   This frame writes %s",
                        mb::FormatBytes(working_set).c_str());
    if (biggest_cache > 0) {
      ImGui::TextDisabled("   Your fastest big cache is %s",
                          mb::FormatBytes(biggest_cache).c_str());
      if (working_set <= biggest_cache) {
        ImGui::TextColored(ImVec4(0.40f, 0.88f, 0.45f, 1.0f),
                           "   Fits in cache, which is ideal");
      } else {
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                           "   Too big for cache; going to RAM");
        ImGui::TextDisabled("   Try a smaller resolution to compare");
      }
    }
    ImGui::Spacing();

    // --- why the numbers move --------------------------------------------
    ImGui::TextUnformatted("Why the speed changes");
    ImGui::TextDisabled("   Method: %s", kernel.impl);
    if (kernel.lanes > 1) {
      ImGui::TextWrapped(
          "   Doing %d pixels per instruction instead of 1 is where the speed "
          "comes from.",
          kernel.lanes);
    } else {
      ImGui::TextWrapped(
          "   This one does a single pixel at a time, so it leaves most of each "
          "instruction unused. Try kernel 2 or 3.");
    }
    ImGui::Spacing();
  }

  // Pause rather than stop/start: "stop" implies tearing something down, but
  // there is no run to end -- the benchmark is a continuous loop. Pause freezes
  // the measurement and the image while leaving the window fully interactive,
  // and it stops the kernel pegging a core, which at a few FPS on a heavy view is
  // the difference between a warm laptop and a hot one.
  void DrawRunControls() {
    if (paused_) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.52f, 0.24f, 1.0f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.64f, 0.30f, 1.0f));
      if (ImGui::Button("Resume")) {
        paused_ = false;
      }
      ImGui::PopStyleColor(2);
    } else {
      if (ImGui::Button(" Pause ")) {
        paused_ = true;
      }
    }

    ImGui::SameLine();
    // Only meaningful while frozen: renders exactly one frame so you can inspect
    // a view without restarting continuous measurement. Deliberately unsampled --
    // a single cold frame is not a data point worth polluting the window with.
    ImGui::BeginDisabled(!paused_);
    if (ImGui::Button("Step")) {
      step_once_ = true;
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (ImGui::Button("Reset stats")) {
      rolling_.Reset();
      frame_time_.Reset();
      live_mega_iters_per_sec_ = 0.0;
      last_iteration_total_ = 0;
      scalar_baseline_mips_ = 0.0;
      scalar_baseline_valid_ = false;
    }

    ImGui::SameLine();
    if (paused_) {
      ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.32f, 1.0f), "PAUSED");
    } else {
      ImGui::TextDisabled("space");
    }

    ImGui::Separator();
    ImGui::Spacing();
  }

  void DrawKernelSection(const mb::LatencyStats& live) {
    const mb::KernelRegistry& reg = mb::Kernels();

    ImGui::TextUnformatted("Active kernel");
    ImGui::Separator();

    // The mandated drop-down for switching kernels on the fly.
    if (ImGui::BeginCombo("##kernel", reg[kernel_index_].name)) {
      for (int i = 0; i < reg.count; ++i) {
        const mb::KernelDesc& d = reg[i];
        if (!d.available) {
          ImGui::BeginDisabled();
          ImGui::Selectable(d.name, false);
          ImGui::EndDisabled();
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Unavailable: %s", d.unavailable_reason);
          }
          continue;
        }
        const bool selected = (i == kernel_index_);
        if (ImGui::Selectable(d.name, selected)) {
          kernel_index_ = i;
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("%s", d.detail);
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(keys 1-%d)", reg.count);

    const mb::KernelDesc& active = reg[kernel_index_];
    ImGui::Text("%s / %s   -   %d px per step", active.isa, active.impl, active.lanes);
    ImGui::TextWrapped("%s", active.detail);

    // --- parallelism -------------------------------------------------------
    ImGui::Spacing();
    if (max_thread_count_ > 1) {
      ImGui::SliderInt("CPU threads", &thread_count_, 1, max_thread_count_,
                       thread_count_ == 1 ? "%d (single core)" : "%d");
      ImGui::SameLine();
      if (ImGui::SmallButton("All")) {
        thread_count_ = max_thread_count_;
      }

      // Row stealing means each thread takes whatever row is next, so an uneven
      // split is expected and healthy — rows differ hugely in cost. Showing the
      // spread makes the balancing visible instead of theoretical.
      if (thread_count_ > 1) {
        int min_rows = 0;
        int max_rows = 0;
        mb::Pool().LastRowSpread(min_rows, max_rows);
        ImGui::TextDisabled("Rows per thread last frame: %d-%d (dynamic steal)",
                            min_rows, max_rows);
      }
    } else {
      ImGui::TextDisabled("Single core only: no additional cores reported.");
    }

    // Throughput in pixels/second is the architecture-neutral figure of merit.
    const double pixels = static_cast<double>(last_params_.width) * last_params_.height;
    if (live.trimmed_mean_ns > 0.0) {
      ImGui::Text("Throughput: %.1f Mpixel/s", pixels / live.trimmed_mean_ns * 1000.0);
    }
    ImGui::Spacing();
  }

  void DrawLatencySection(const mb::LatencyStats& live, const mb::LatencyStats& frame) {
    if (!ImGui::CollapsingHeader("Execution latency", ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }

#if defined(MB_KERNEL_BUILD_IS_DEBUG)
    // A Debug build of mb_kernels is missing -O3. Measured on this project: at
    // -O0 the NEON intrinsics kernel runs ~11x slower and looks ~10x slower than
    // the hand-written assembly kernel it is actually within 1% of at -O3 --
    // because the .S/.asm kernel is raw machine code the compiler never touches,
    // while the C++ kernels lose inlining and loop-invariant hoisting. CLion
    // creates a Debug CMake profile by default, so this is easy to hit by
    // accident. Surfaced here rather than only in the CMake configure log,
    // since that log is easy to have scrolled past by the time you're reading
    // this number.
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                       "DEBUG BUILD (%s) -- these numbers do not measure the "
                       "hardware, mostly the missing -O3.",
                       MB_KERNEL_BUILD_TYPE);
    ImGui::TextWrapped(
        "Switch to a Release CMake profile and rebuild before trusting any "
        "latency or speedup figure below.");
    ImGui::Spacing();
#endif

    // Headline numbers: trimmed mean is the throughput estimate, stddev is the
    // dispersion the spec asks for.
    ImGui::Text("Trimmed mean  %10.3f us   (%.3f ms)", live.trimmed_mean_us(),
                live.trimmed_mean_ms());
    ImGui::Text("Std deviation %10.3f us   (CV %.2f%%)", live.trimmed_stddev_ns / 1000.0,
                live.cv_pct);
    ImGui::Text("Raw mean      %10.3f us   +/- %.3f us (95%% CI)", live.mean_us(),
                live.ci95_ns / 1000.0);
    ImGui::Text("Nanoseconds   %10.0f ns", live.trimmed_mean_ns);

    ImGui::Spacing();
    if (ImGui::BeginTable("dist", 2, ImGuiTableFlags_SizingStretchProp)) {
      const auto row = [](const char* k, const char* fmt, double v) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(k);
        ImGui::TableNextColumn();
        ImGui::Text(fmt, v);
      };
      row("min", "%.3f us", live.min_ns / 1000.0);
      row("median", "%.3f us", live.median_ns / 1000.0);
      row("p95", "%.3f us", live.p95_ns / 1000.0);
      row("p99", "%.3f us", live.p99_ns / 1000.0);
      row("max", "%.3f us", live.max_ns / 1000.0);
      ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Text("Window: n = %u / %zu", live.n, rolling_.kCapacity);
    ImGui::SameLine();
    if (live.clt_satisfied) {
      ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "  CLT ok (n >= %u)",
                         mb::kCltMinSamples);
    } else {
      ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.3f, 1.0f), "  collecting (n < %u)",
                         mb::kCltMinSamples);
    }
    ImGui::Text("Upper-tail trim: dropped %u slowest of %u (5%%)", live.n_dropped, live.n);
    if (live.mean_ns > 0.0 && live.trimmed_mean_ns > 0.0) {
      const double noise = (live.mean_ns - live.trimmed_mean_ns) / live.mean_ns * 100.0;
      ImGui::TextDisabled("Scheduler noise removed from the mean: %.2f%%", noise);
    }

    ImGui::Spacing();
    ImGui::Text("Frames: %llu   %.1f FPS",
                static_cast<unsigned long long>(frame_counter_),
                frame.trimmed_mean_ns > 0.0 ? 1e9 / frame.trimmed_mean_ns : 0.0);

    // The zero-allocation tripwire.
    if (hot_path_allocations_ == 0) {
      ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f),
                         "Heap allocations in measured region: 0");
    } else {
      ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                         "Heap allocations in measured region: %llu  <-- regression!",
                         static_cast<unsigned long long>(hot_path_allocations_));
    }

    ImGui::TextDisabled("Clock: %s, granularity %.1f ns",
                        mb::kClockIsSteady ? "steady" : "NOT steady",
                        mb::MeasuredClockGranularityNanos());
    if (!mb::kClockIsSteady) {
      ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.3f, 1.0f),
                         "high_resolution_clock is not monotonic on this platform.");
    }
    ImGui::Spacing();
  }

  void DrawViewSection() {
    if (!ImGui::CollapsingHeader("View & workload")) {
      return;
    }

    // No manual statistics reset here or below: RenderFractal() invalidates the
    // window whenever the workload signature changes, which covers these
    // controls and mouse pan/zoom uniformly.
    ImGui::SliderInt("Max iterations", &view_.max_iter, 16, 4096);

    int res = resolution_index_;
    if (ImGui::BeginCombo("Resolution", kResolutions[res].label)) {
      for (int i = 0; i < kResolutionCount; ++i) {
        if (ImGui::Selectable(kResolutions[i].label, i == res)) {
          resolution_index_ = i;
        }
      }
      ImGui::EndCombo();
    }

    int pal = static_cast<int>(palette_);
    if (ImGui::BeginCombo("Palette", mb::PaletteName(palette_))) {
      for (int i = 0; i < static_cast<int>(mb::Palette::kCount); ++i) {
        const mb::Palette candidate = static_cast<mb::Palette>(i);
        if (ImGui::Selectable(mb::PaletteName(candidate), i == pal)) {
          palette_ = candidate;
        }
      }
      ImGui::EndCombo();
    }

    if (ImGui::BeginCombo("Go to", "select a location")) {
      for (int i = 0; i < kZoomTargetCount; ++i) {
        if (ImGui::Selectable(kZoomTargets[i].label, false)) {
          view_.center_x = kZoomTargets[i].center_x;
          view_.center_y = kZoomTargets[i].center_y;
          view_.span_x = kZoomTargets[i].span_x;
          view_.max_iter = kZoomTargets[i].max_iter;
        }
      }
      ImGui::EndCombo();
    }

    ImGui::Text("center  %+.12f  %+.12f", view_.center_x, view_.center_y);
    ImGui::Text("span    %.3e   (zoom %.1fx)", view_.span_x,
                kZoomTargets[0].span_x / view_.span_x);

    const double ulps = UlpsPerPixel(view_, kResolutions[resolution_index_].width);
    if (ulps < 4.0) {
      ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.3f, 1.0f),
                         "Single-precision limit: %.2f float ULPs per pixel", ulps);
      ImGui::TextWrapped(
          "Adjacent pixels are within a few float ULPs of each other, so detail is "
          "now limited by the f32 kernels rather than by max_iter. Zoom is clamped "
          "at 0.5 ULP/pixel.");
    } else {
      ImGui::TextDisabled("Precision headroom: %.0f float ULPs per pixel", ulps);
    }

    ImGui::TextDisabled("Drag to pan, scroll to zoom, Esc to quit.");
    ImGui::Spacing();
  }

  void DrawSweepSection() {
    if (!ImGui::CollapsingHeader("Statistical sweep", ImGuiTreeNodeFlags_DefaultOpen)) {
      return;
    }

    ImGui::TextWrapped(
        "Benchmarks every available kernel over the current view with a fixed "
        "sample size, discarding warm-up runs and trimming the slow tail.");

    int warmup = sweep_.warmup_target();
    int samples = sweep_.sample_target();
    bool reconfigure = false;
    ImGui::BeginDisabled(sweep_.running());
    reconfigure |= ImGui::SliderInt("Warm-up runs", &warmup, 0, 64);
    reconfigure |= ImGui::SliderInt("Samples/kernel", &samples, static_cast<int>(mb::kCltMinSamples),
                                    1024);
    ImGui::EndDisabled();
    if (reconfigure) {
      sweep_.Configure(warmup, samples);
    }

    if (sweep_.running()) {
      if (ImGui::Button("Cancel sweep")) {
        sweep_.Cancel();
      }
      ImGui::SameLine();
      ImGui::ProgressBar(sweep_.progress(), ImVec2(-1, 0));
      ImGui::Text("Running: %s", sweep_.current_kernel_name());
    } else {
      if (ImGui::Button("Run sweep")) {
        const Resolution& res = kResolutions[resolution_index_];
        sweep_.Start(BuildParams(view_, res.width, res.height), thread_count_);
      }
      ImGui::SameLine();
      ImGui::TextDisabled("%d warm-up + %d samples per kernel", warmup, samples);
    }

    if (!sweep_.has_results()) {
      ImGui::Spacing();
      return;
    }

    // Baseline for the speedup column: the scalar kernel, index 0.
    mb::LatencyStats baseline;
    const bool have_baseline = sweep_.stats_for(0, baseline);

    ImGui::Spacing();
    constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                       ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("sweep", 6, kFlags)) {
      ImGui::TableSetupColumn("kernel");
      ImGui::TableSetupColumn("trimmed");
      ImGui::TableSetupColumn("stddev");
      ImGui::TableSetupColumn("CV%");
      ImGui::TableSetupColumn("Mpx/s");
      ImGui::TableSetupColumn("speedup");
      ImGui::TableHeadersRow();

      const mb::KernelRegistry& reg = mb::Kernels();
      const double pixels = static_cast<double>(sweep_.params().width) * sweep_.params().height;

      for (int i = 0; i < reg.count; ++i) {
        mb::LatencyStats s;
        if (!sweep_.stats_for(i, s)) {
          continue;
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(reg[i].name);
        ImGui::TableNextColumn();
        ImGui::Text("%.1f us", s.trimmed_mean_ns / 1000.0);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f us", s.trimmed_stddev_ns / 1000.0);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f", s.cv_pct);
        ImGui::TableNextColumn();
        ImGui::Text("%.1f", s.trimmed_mean_ns > 0.0 ? pixels / s.trimmed_mean_ns * 1000.0 : 0.0);
        ImGui::TableNextColumn();
        if (have_baseline && s.trimmed_mean_ns > 0.0) {
          ImGui::Text("%.2fx", baseline.trimmed_mean_ns / s.trimmed_mean_ns);
        } else {
          ImGui::TextUnformatted("-");
        }
      }
      ImGui::EndTable();
    }

    ImGui::TextDisabled("%d x %d, max_iter %d, n = %d per kernel, %d thread%s",
                        sweep_.params().width, sweep_.params().height,
                        sweep_.params().max_iter, sweep_.sample_target(),
                        sweep_.thread_count(), sweep_.thread_count() == 1 ? "" : "s");
    ImGui::Spacing();
  }

  void DrawVerificationSection() {
    if (!ImGui::CollapsingHeader("Correctness")) {
      return;
    }
    ImGui::TextWrapped(
        "Every kernel must reproduce the scalar reference bit-for-bit. A faster "
        "kernel that disagrees is a bug, not an optimisation.");

    if (ImGui::Button("Verify kernels")) {
      RunVerification();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("costs one extra pass per kernel");

    if (!verification_done_) {
      ImGui::Spacing();
      return;
    }

    const mb::KernelRegistry& reg = mb::Kernels();
    ImGui::Spacing();
    for (int i = 0; i < reg.count; ++i) {
      const VerificationRow& row = verification_[i];
      if (!row.valid) {
        continue;
      }
      if (row.exact) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "PASS  %s", reg[i].name);
      } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           "FAIL  %s  (%llu px differ, max delta %d)", reg[i].name,
                           static_cast<unsigned long long>(row.mismatches), row.max_diff);
      }
    }
    ImGui::TextDisabled("%llu pixels compared per kernel",
                        static_cast<unsigned long long>(verified_pixels_));
    ImGui::Spacing();
  }

  void DrawHardwareSection() {
    if (!ImGui::CollapsingHeader("Hardware")) {
      return;
    }
    const mb::HardwareSnapshot& hw = mb::QueryHardware();
    const mb::CpuFeatures& cpu = mb::DetectCpuFeatures();

    ImGui::Text("%s", hw.cpu_model.c_str());
    if (!hw.cpu_vendor.empty()) {
      ImGui::TextDisabled("vendor: %s", hw.cpu_vendor.c_str());
    }
    ImGui::Text("Architecture   %s", hw.arch_name.c_str());
    ImGui::Text("Physical cores %d   (logical %d)", hw.physical_cores, hw.logical_cores);
    ImGui::Text("Max clock      %s", mb::FormatClock(hw.max_clock_mhz).c_str());

    ImGui::Spacing();
    ImGui::TextUnformatted("Cache hierarchy");
    ImGui::Text("  L1 data       %s", mb::FormatBytes(hw.l1d_bytes).c_str());
    ImGui::Text("  L1 instr      %s", mb::FormatBytes(hw.l1i_bytes).c_str());
    ImGui::Text("  L2            %s", mb::FormatBytes(hw.l2_bytes).c_str());
    ImGui::Text("  L3            %s", mb::FormatBytes(hw.l3_bytes).c_str());
    ImGui::Text("  line size     %u bytes", hw.cache_line_bytes);
    if (hw.cache_line_bytes > 64) {
      ImGui::TextDisabled("  (buffers aligned to %zu B, not just 64)", mb::kCacheLineBytes);
    }

    if (hw.cluster_count > 0) {
      ImGui::Spacing();
      ImGui::TextUnformatted("Core clusters");
      for (int i = 0; i < hw.cluster_count; ++i) {
        const mb::CoreCluster& c = hw.clusters[static_cast<std::size_t>(i)];
        ImGui::Text("  %-12s %2d cores, L1d %s, L2 %s", c.name.c_str(), c.physical_cores,
                    mb::FormatBytes(c.l1d_bytes).c_str(), mb::FormatBytes(c.l2_bytes).c_str());
      }
      ImGui::TextDisabled("Heterogeneous cores: which cluster runs the loop");
      ImGui::TextDisabled("affects latency and is chosen by the OS scheduler.");
    }

    ImGui::Spacing();
    ImGui::Text("System RAM     %s", mb::FormatBytes(hw.ram_total_bytes).c_str());
    if (hw.ram_available_bytes > 0) {
      ImGui::TextDisabled("  available: %s", mb::FormatBytes(hw.ram_available_bytes).c_str());
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("SIMD support");
#if MB_ARCH_X86
    ImGui::Text("  AVX %d   AVX2 %d   FMA %d", cpu.avx, cpu.avx2, cpu.fma);
    ImGui::Text("  OSXSAVE %d   YMM state preserved %d", cpu.osxsave, cpu.ymm_state);
#elif MB_ARCH_ARM
    ImGui::Text("  NEON / ASIMD  %d  (mandatory in AArch64)", cpu.neon);
#endif
    ImGui::Text("  SIMD kernels usable: %s", cpu.simd_kernels_usable ? "yes" : "no");

    ImGui::Spacing();
    if (!hw.os_name.empty()) {
      ImGui::Text("OS             %s %s", hw.os_name.c_str(), hw.os_version.c_str());
    }
    if (!hw.kernel_version.empty()) {
      ImGui::TextDisabled("kernel: %s", hw.kernel_version.c_str());
    }
    ImGui::TextDisabled("source: %s", hw.provider.c_str());
    ImGui::Spacing();
  }

  void Present() {
    const ImGuiIO& io = ImGui::GetIO();

    SDL_RenderSetScale(renderer_, 1.0f, 1.0f);
    SDL_SetRenderDrawColor(renderer_, 12, 12, 16, 255);
    SDL_RenderClear(renderer_);

    const Resolution& res = kResolutions[resolution_index_];
    const SDL_Rect src{0, 0, res.width, res.height};
    const SDL_Rect dst = FractalRect();
    SDL_RenderCopy(renderer_, texture_, &src, &dst);

    // ImGui's SDL_Renderer backend draws in logical units; apply the framebuffer
    // scale only for the UI pass so the fractal blit above stays 1:1.
    SDL_RenderSetScale(renderer_, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
#if defined(IMGUI_VERSION_NUM) && IMGUI_VERSION_NUM >= 19100
    ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer_);
#else
    ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData());
#endif

    SDL_RenderPresent(renderer_);
  }

  // --- SDL / ImGui
  SDL_Window* window_ = nullptr;
  SDL_Renderer* renderer_ = nullptr;
  SDL_Texture* texture_ = nullptr;
  bool quit_ = false;
  bool dragging_ = false;

  // --- view / workload
  View view_{};
  int resolution_index_ = 1;  // 640 x 360
  int kernel_index_ = 0;
  mb::Palette palette_ = mb::Palette::Ember;
  mb::KernelParams last_params_{};

  // --- statistics
  mb::LatencyProfiler<512> rolling_;
  mb::LatencyProfiler<256> frame_time_;
  std::uint64_t frame_counter_ = 0;
  std::uint64_t hot_path_allocations_ = 0;

  // Throughput for the simple panel's grade.
  std::uint64_t last_iteration_total_ = 0;
  double live_mega_iters_per_sec_ = 0.0;
  double scalar_baseline_mips_ = 0.0;
  WorkloadSignature scalar_baseline_workload_{};
  bool scalar_baseline_valid_ = false;

  // Parallelism. Defaults to 1 so the out-of-the-box reading is the single-core
  // figure every other measurement in this project was taken at.
  int thread_count_ = 1;
  int max_thread_count_ = 1;

  // Run control.
  bool paused_ = false;
  bool step_once_ = false;
  RenderSignature last_render_{};

  BenchmarkSweep sweep_;

  VerificationRow verification_[mb::kMaxKernels]{};
  bool verification_done_ = false;
  std::uint64_t verified_pixels_ = 0;
};

}  // namespace

int main(int, char**) {
  // Report the environment before opening a window: if SDL cannot initialise,
  // this is still useful output.
  const mb::HardwareSnapshot& hw = mb::QueryHardware();
  std::printf("MandelbrotBenchmark %s\n", MB_ARCH_NAME);
  std::printf("  CPU     : %s (%d physical cores)\n", hw.cpu_model.c_str(), hw.physical_cores);
  std::printf("  Caches  : L1d %s, L2 %s, L3 %s, line %u B\n",
              mb::FormatBytes(hw.l1d_bytes).c_str(), mb::FormatBytes(hw.l2_bytes).c_str(),
              mb::FormatBytes(hw.l3_bytes).c_str(), hw.cache_line_bytes);
  std::printf("  RAM     : %s\n", mb::FormatBytes(hw.ram_total_bytes).c_str());
  std::printf("  Source  : %s\n", hw.provider.c_str());

  const mb::KernelRegistry& reg = mb::Kernels();
  std::printf("  Kernels :\n");
  for (int i = 0; i < reg.count; ++i) {
    std::printf("    [%d] %-28s %-12s %d px/step  %s\n", i + 1, reg[i].name, reg[i].impl,
                reg[i].lanes, reg[i].available ? "available" : reg[i].unavailable_reason);
  }
  std::printf("  Buffers : %.1f MiB statically allocated\n",
              static_cast<double>(sizeof(mb::RenderTarget)) / (1024.0 * 1024.0));
  std::fflush(stdout);

  static App app;  // static: keeps the ~17 MiB of state out of the stack
  if (!app.Init()) {
    app.Shutdown();
    return 1;
  }
  app.Run();
  app.Shutdown();
  return 0;
}
