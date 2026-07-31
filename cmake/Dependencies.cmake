# Dependencies.cmake — third-party acquisition.
#
# Everything is fetched at configure time so a clean checkout builds with nothing
# but a compiler, CMake, git and (on x86-64) NASM. Each dependency can instead be
# taken from the system if you prefer; see the MB_USE_SYSTEM_* options.

include(FetchContent)

# Keep FetchContent quiet about timestamps on CMake >= 3.24.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(FETCHCONTENT_QUIET FALSE)

# ---------------------------------------------------------------------------
# SDL2 — windowing, input, and the 2D renderer used to blit the fractal.
# ---------------------------------------------------------------------------
if(MB_USE_SYSTEM_SDL2)
  find_package(SDL2 REQUIRED)
  message(STATUS "SDL2: using system package")
else()
  # Build a static SDL2 and skip everything we do not use.
  set(SDL_SHARED OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC ON CACHE BOOL "" FORCE)
  set(SDL_TEST OFF CACHE BOOL "" FORCE)
  set(SDL_TESTS OFF CACHE BOOL "" FORCE)
  set(SDL2_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
  set(SDL2_DISABLE_UNINSTALL ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
    SDL2
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-2.30.9
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(SDL2)
  message(STATUS "SDL2: fetched release-2.30.9 (static)")
endif()

# SDL2 exports different target names depending on shared/static/system.
if(TARGET SDL2::SDL2)
  set(MB_SDL2_TARGET SDL2::SDL2)
elseif(TARGET SDL2::SDL2-static)
  set(MB_SDL2_TARGET SDL2::SDL2-static)
elseif(TARGET SDL2)
  set(MB_SDL2_TARGET SDL2)
else()
  message(FATAL_ERROR
    "Could not determine the SDL2 target name. Configure with "
    "-DMB_USE_SYSTEM_SDL2=ON and a working SDL2 installation, or report this.")
endif()
message(STATUS "SDL2 target: ${MB_SDL2_TARGET}")

# ---------------------------------------------------------------------------
# Dear ImGui — dashboard overlay.
#
# ImGui ships no CMakeLists.txt, so we compile the core plus the two backends we
# need into our own static library. FetchContent_MakeAvailable only calls
# add_subdirectory() when the fetched tree contains a CMakeLists.txt, so this
# populates without trying to configure it.
# ---------------------------------------------------------------------------
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.5
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(mb_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer2.cpp
)

target_include_directories(mb_imgui SYSTEM PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(mb_imgui PUBLIC ${MB_SDL2_TARGET})
target_compile_features(mb_imgui PUBLIC cxx_std_17)

# Deliberately NOT given mb_flags: ImGui has no reason to be built with
# -ffast-math, and keeping it on default FP semantics means the only fast-math
# code in the binary is the code we actually measure.
set_target_properties(mb_imgui PROPERTIES CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)

message(STATUS "Dear ImGui: fetched v1.91.5 (SDL2 + SDL_Renderer backends)")

# ---------------------------------------------------------------------------
# hwinfo (Leon Freist) — the Hardware Interrogator's primary data source.
# ---------------------------------------------------------------------------
if(MB_USE_HWINFO)
  # hwinfo publishes no release tags at all — the repository has only a `c++11`
  # tag and a moving `main` branch. Pinning a commit SHA is therefore the only
  # way to get a reproducible build; `main` would silently change under us.
  # GIT_SHALLOW must be off because a shallow clone cannot check out an
  # arbitrary SHA.
  set(MB_HWINFO_COMMIT 88c5072c4a137d54e94c7e712ae28ac284f1dd9b)  # main, 2026-07

  # Real option names, verified against that commit's CMakeLists.txt. Note
  # HWINFO_SHARED defaults to ON upstream, which would drop a dylib next to the
  # executable; we want everything static.
  set(HWINFO_SHARED OFF CACHE BOOL "" FORCE)
  set(HWINFO_STATIC ON CACHE BOOL "" FORCE)

  # Only the three components the dashboard actually reads. The rest pull in
  # WMI / OpenCL / IOKit probing we never call, and cost build time.
  set(HWINFO_CPU ON CACHE BOOL "" FORCE)
  set(HWINFO_RAM ON CACHE BOOL "" FORCE)
  set(HWINFO_OS ON CACHE BOOL "" FORCE)
  set(HWINFO_GPU OFF CACHE BOOL "" FORCE)
  set(HWINFO_GPU_OPENCL OFF CACHE BOOL "" FORCE)
  set(HWINFO_DISK OFF CACHE BOOL "" FORCE)
  set(HWINFO_BATTERY OFF CACHE BOOL "" FORCE)
  set(HWINFO_NETWORK OFF CACHE BOOL "" FORCE)
  set(HWINFO_MAINBOARD OFF CACHE BOOL "" FORCE)
  # BUILD_EXAMPLES / BUILD_TESTING upstream default to PROJECT_IS_TOP_LEVEL,
  # which is already false for a fetched dependency. Deliberately not forced —
  # BUILD_TESTING is a shared CTest variable and clobbering it is antisocial.

  FetchContent_Declare(
    hwinfo
    GIT_REPOSITORY https://github.com/lfreist/hwinfo.git
    GIT_TAG ${MB_HWINFO_COMMIT}
    GIT_SHALLOW FALSE
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(hwinfo)

  # The exported target name has changed across hwinfo revisions.
  if(TARGET lfreist-hwinfo::hwinfo)
    set(MB_HWINFO_TARGET lfreist-hwinfo::hwinfo)
  elseif(TARGET hwinfo::hwinfo)
    set(MB_HWINFO_TARGET hwinfo::hwinfo)
  elseif(TARGET hwinfo)
    set(MB_HWINFO_TARGET hwinfo)
  else()
    message(FATAL_ERROR
      "hwinfo was fetched but no usable target was found (tried "
      "lfreist-hwinfo::hwinfo, hwinfo::hwinfo, hwinfo). Reconfigure with "
      "-DMB_USE_HWINFO=OFF to fall back to the native sysctl / procfs provider, "
      "which reports the same fields.")
  endif()
  message(STATUS "hwinfo target: ${MB_HWINFO_TARGET}")
else()
  set(MB_HWINFO_TARGET "")
  message(STATUS "hwinfo: disabled, using the native provider "
                 "(sysctl on macOS, /proc + /sys on Linux)")
endif()
