// hardware_info.hpp — Component 1: the Hardware Interrogator.
//
// One snapshot struct, two interchangeable providers:
//
//   * hwinfo (Leon Freist) — the default, selected by MB_USE_HWINFO.
//   * a native provider    — sysctl on macOS, /proc + /sys on Linux.
//
// Both fill the same struct, so the UI never knows which ran. The snapshot is
// built once at startup and then only read, so std::string here never touches
// the render hot path.
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace mb {

// Apple Silicon (and big.LITTLE ARM generally) exposes heterogeneous core
// clusters with different cache geometry per cluster; a single "core count"
// throws away information that matters when reading benchmark numbers.
struct CoreCluster {
  std::string name;
  int physical_cores = 0;
  std::uint64_t l1d_bytes = 0;
  std::uint64_t l2_bytes = 0;
};

struct HardwareSnapshot {
  // --- CPU identity
  std::string cpu_model = "unknown";
  std::string cpu_vendor;
  std::string arch_name = "unknown";

  int physical_cores = 0;
  int logical_cores = 0;

  // 0 means "not exposed by this platform". Apple Silicon, for instance, does
  // not publish a max clock through sysctl.
  std::uint32_t max_clock_mhz = 0;
  std::uint32_t base_clock_mhz = 0;

  // --- Cache hierarchy (bytes; 0 means unknown)
  std::uint64_t l1d_bytes = 0;
  std::uint64_t l1i_bytes = 0;
  std::uint64_t l2_bytes = 0;
  std::uint64_t l3_bytes = 0;
  std::uint32_t cache_line_bytes = 0;

  // --- Heterogeneous clusters (empty on uniform CPUs)
  std::array<CoreCluster, 4> clusters{};
  int cluster_count = 0;

  // --- Memory
  std::uint64_t ram_total_bytes = 0;
  std::uint64_t ram_available_bytes = 0;

  // --- OS
  std::string os_name;
  std::string os_version;
  std::string kernel_version;

  // --- Provenance, surfaced in the UI so a reader knows where numbers came from.
  std::string provider = "none";
};

// Polls the system once. Safe to call repeatedly; the result is cached.
const HardwareSnapshot& QueryHardware();

// Formatting helpers used by the dashboard.
std::string FormatBytes(std::uint64_t bytes);
std::string FormatClock(std::uint32_t mhz);

}  // namespace mb
