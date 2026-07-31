// hardware_info.cpp — implementation of the Hardware Interrogator.
//
// Layout: the native provider comes first and is always compiled, then the
// hwinfo provider layers on top when MB_USE_HWINFO is set, falling back to the
// native values for any field hwinfo leaves at zero. Keeping the native path
// unconditional means the dashboard degrades gracefully rather than going blank
// if hwinfo is unavailable or its API drifts.

#include "hardware_info.hpp"

#include <cstdio>
#include <cstring>

#include "arch.hpp"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#endif

#if defined(MB_USE_HWINFO)
// Only the three components enabled in cmake/Dependencies.cmake. The umbrella
// <hwinfo/hwinfo.h> also pulls in battery/disk/gpu/network/mainboard, whose
// libraries we deliberately do not build.
#include <hwinfo/cpu.h>
#include <hwinfo/os.h>
#include <hwinfo/ram.h>

#include <algorithm>
#endif

namespace mb {
namespace {

// ---------------------------------------------------------------------------
// Native provider — macOS
// ---------------------------------------------------------------------------
#if defined(__APPLE__)

bool SysctlU64(const char* name, std::uint64_t& out) {
  std::uint64_t value = 0;
  std::size_t len = sizeof(value);
  if (sysctlbyname(name, &value, &len, nullptr, 0) != 0) {
    return false;
  }
  // Some keys report 4 bytes, others 8; sysctl writes only `len` bytes.
  if (len == sizeof(std::uint32_t)) {
    std::uint32_t narrow = 0;
    std::memcpy(&narrow, &value, sizeof(narrow));
    out = narrow;
  } else {
    out = value;
  }
  return true;
}

bool SysctlInt(const char* name, int& out) {
  std::uint64_t v = 0;
  if (!SysctlU64(name, v)) {
    return false;
  }
  out = static_cast<int>(v);
  return true;
}

bool SysctlString(const char* name, std::string& out) {
  std::size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) {
    return false;
  }
  std::string buf(len, '\0');
  if (sysctlbyname(name, buf.data(), &len, nullptr, 0) != 0) {
    return false;
  }
  // Trim the trailing NUL sysctl includes in `len`.
  while (!buf.empty() && buf.back() == '\0') {
    buf.pop_back();
  }
  out = buf;
  return true;
}

void FillNative(HardwareSnapshot& s) {
  s.provider = "native sysctl (macOS)";

  SysctlString("machdep.cpu.brand_string", s.cpu_model);
  if (s.cpu_model == "unknown" || s.cpu_model.empty()) {
    SysctlString("hw.model", s.cpu_model);
  }
  SysctlString("machdep.cpu.vendor", s.cpu_vendor);

  SysctlInt("hw.physicalcpu", s.physical_cores);
  SysctlInt("hw.logicalcpu", s.logical_cores);

  // Apple Silicon leaves the frequency keys absent; Intel Macs populate them.
  std::uint64_t hz = 0;
  if (SysctlU64("hw.cpufrequency_max", hz) && hz > 0) {
    s.max_clock_mhz = static_cast<std::uint32_t>(hz / 1'000'000ull);
  }
  if (SysctlU64("hw.cpufrequency", hz) && hz > 0) {
    s.base_clock_mhz = static_cast<std::uint32_t>(hz / 1'000'000ull);
  }

  SysctlU64("hw.l1dcachesize", s.l1d_bytes);
  SysctlU64("hw.l1icachesize", s.l1i_bytes);
  SysctlU64("hw.l2cachesize", s.l2_bytes);
  SysctlU64("hw.l3cachesize", s.l3_bytes);

  std::uint64_t line = 0;
  if (SysctlU64("hw.cachelinesize", line)) {
    s.cache_line_bytes = static_cast<std::uint32_t>(line);
  }

  SysctlU64("hw.memsize", s.ram_total_bytes);

  // Heterogeneous core clusters (hw.nperflevels >= 2 on Apple Silicon).
  int levels = 0;
  if (SysctlInt("hw.nperflevels", levels) && levels > 0) {
    const int capped = levels > static_cast<int>(s.clusters.size())
                           ? static_cast<int>(s.clusters.size())
                           : levels;
    for (int i = 0; i < capped; ++i) {
      char key[64];
      CoreCluster c;

      std::snprintf(key, sizeof(key), "hw.perflevel%d.name", i);
      if (!SysctlString(key, c.name)) {
        c.name = "cluster " + std::to_string(i);
      }
      std::snprintf(key, sizeof(key), "hw.perflevel%d.physicalcpu", i);
      SysctlInt(key, c.physical_cores);
      std::snprintf(key, sizeof(key), "hw.perflevel%d.l1dcachesize", i);
      SysctlU64(key, c.l1d_bytes);
      std::snprintf(key, sizeof(key), "hw.perflevel%d.l2cachesize", i);
      SysctlU64(key, c.l2_bytes);

      s.clusters[static_cast<std::size_t>(s.cluster_count++)] = c;
    }
  }

  s.os_name = "macOS";
  SysctlString("kern.osproductversion", s.os_version);
  SysctlString("kern.osrelease", s.kernel_version);
}

// ---------------------------------------------------------------------------
// Native provider — Linux
// ---------------------------------------------------------------------------
#elif defined(__linux__)

std::string ReadFirstLine(const char* path) {
  std::ifstream f(path);
  std::string line;
  if (f && std::getline(f, line)) {
    return line;
  }
  return {};
}

// Parses "32K" / "8192K" / "16M" as emitted by /sys cache size files.
std::uint64_t ParseSizeSuffix(const std::string& text) {
  if (text.empty()) {
    return 0;
  }
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str()) {
    return 0;
  }
  switch (*end) {
    case 'K':
    case 'k':
      return value * 1024ull;
    case 'M':
    case 'm':
      return value * 1024ull * 1024ull;
    default:
      return value;
  }
}

void FillNative(HardwareSnapshot& s) {
  s.provider = "native /proc + /sys (Linux)";

  // /proc/cpuinfo: model name, vendor, physical core count via "cpu cores".
  {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    int cpu_cores = 0;
    while (std::getline(f, line)) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) {
        continue;
      }
      std::string key = line.substr(0, colon);
      std::string value = line.substr(colon + 1);
      while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
        key.pop_back();
      }
      while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
        value.erase(value.begin());
      }

      if (key == "model name" && s.cpu_model == "unknown") {
        s.cpu_model = value;
      } else if (key == "vendor_id" && s.cpu_vendor.empty()) {
        s.cpu_vendor = value;
      } else if (key == "cpu cores" && cpu_cores == 0) {
        cpu_cores = std::atoi(value.c_str());
      } else if (key == "cpu MHz" && s.base_clock_mhz == 0) {
        s.base_clock_mhz = static_cast<std::uint32_t>(std::atof(value.c_str()));
      }
    }
    s.physical_cores = cpu_cores;
  }

  s.logical_cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
  if (s.physical_cores == 0) {
    s.physical_cores = s.logical_cores;
  }

  // cpufreq reports kHz.
  {
    const std::string khz =
        ReadFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (!khz.empty()) {
      s.max_clock_mhz = static_cast<std::uint32_t>(std::strtoull(khz.c_str(), nullptr, 10) / 1000);
    }
  }

  // Walk cpu0's cache index* entries and bucket by level + type.
  for (int idx = 0; idx < 10; ++idx) {
    char base[128];
    std::snprintf(base, sizeof(base), "/sys/devices/system/cpu/cpu0/cache/index%d", idx);

    char path[192];
    std::snprintf(path, sizeof(path), "%s/level", base);
    const std::string level_str = ReadFirstLine(path);
    if (level_str.empty()) {
      continue;
    }
    std::snprintf(path, sizeof(path), "%s/type", base);
    const std::string type = ReadFirstLine(path);
    std::snprintf(path, sizeof(path), "%s/size", base);
    const std::uint64_t size = ParseSizeSuffix(ReadFirstLine(path));
    std::snprintf(path, sizeof(path), "%s/coherency_line_size", base);
    const std::string line_str = ReadFirstLine(path);
    if (!line_str.empty() && s.cache_line_bytes == 0) {
      s.cache_line_bytes = static_cast<std::uint32_t>(std::strtoul(line_str.c_str(), nullptr, 10));
    }

    const int level = std::atoi(level_str.c_str());
    if (level == 1 && type == "Data") {
      s.l1d_bytes = size;
    } else if (level == 1 && type == "Instruction") {
      s.l1i_bytes = size;
    } else if (level == 2) {
      s.l2_bytes = size;
    } else if (level == 3) {
      s.l3_bytes = size;
    }
  }

  struct sysinfo info{};
  if (sysinfo(&info) == 0) {
    const std::uint64_t unit = info.mem_unit ? info.mem_unit : 1;
    s.ram_total_bytes = static_cast<std::uint64_t>(info.totalram) * unit;
    s.ram_available_bytes = static_cast<std::uint64_t>(info.freeram) * unit;
  }

  s.os_name = "Linux";
  {
    std::ifstream f("/etc/os-release");
    std::string line;
    while (std::getline(f, line)) {
      if (line.rfind("PRETTY_NAME=", 0) == 0) {
        std::string value = line.substr(12);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.size() - 2);
        }
        s.os_name = value;
        break;
      }
    }
  }
  struct utsname uts{};
  if (uname(&uts) == 0) {
    s.kernel_version = uts.release;
    s.os_version = uts.version;
  }
}

// ---------------------------------------------------------------------------
// Native provider — anything else
// ---------------------------------------------------------------------------
#else

void FillNative(HardwareSnapshot& s) {
  s.provider = "none (unsupported platform)";
}

#endif

// ---------------------------------------------------------------------------
// hwinfo provider — overlays the native values
// ---------------------------------------------------------------------------
#if defined(MB_USE_HWINFO)

// hwinfo returns int64_t sentinels (-1) for values a platform does not expose,
// and those land in unsigned fields as ~1.8e19. Anything outside a plausible
// range is treated as "unknown" so the dashboard never shows a 16-exabyte L2.
bool PlausibleCache(std::uint64_t bytes) {
  return bytes > 0 && bytes <= (1ull << 30);  // <= 1 GiB
}
bool PlausibleFrequencyHz(std::uint64_t hz) {
  return hz > 0 && hz <= 100'000'000'000ull;  // <= 100 GHz
}

void FillHwinfo(HardwareSnapshot& s) {
  bool any = false;

  const std::vector<hwinfo::CPU> cpus = hwinfo::getAllCPUs();
  if (!cpus.empty()) {
    const hwinfo::CPU& cpu = cpus.front();
    any = true;

    if (!cpu.modelName().empty()) {
      s.cpu_model = cpu.modelName();
    }
    if (!cpu.vendor().empty()) {
      s.cpu_vendor = cpu.vendor();
    }
    if (cpu.numPhysicalCores() > 0) {
      s.physical_cores = static_cast<int>(cpu.numPhysicalCores());
    }
    if (cpu.numLogicalCores() > 0) {
      s.logical_cores = static_cast<int>(cpu.numLogicalCores());
    }

    // Cache geometry and clocks are per-core in hwinfo, not per-package. Take
    // the maximum across cores: on a heterogeneous CPU the interesting figure
    // is the biggest cache / fastest cluster, and taking core 0 would silently
    // report whichever cluster happened to be enumerated first.
    std::uint64_t l1d = 0;
    std::uint64_t l1i = 0;
    std::uint64_t l2 = 0;
    std::uint64_t l3 = 0;
    std::uint64_t max_hz = 0;
    std::uint64_t regular_hz = 0;

    for (const hwinfo::CPU::Core& core : cpu.cores()) {
      if (PlausibleCache(core.cache.l1_data)) {
        l1d = std::max(l1d, core.cache.l1_data);
      }
      if (PlausibleCache(core.cache.l1_instruction)) {
        l1i = std::max(l1i, core.cache.l1_instruction);
      }
      if (PlausibleCache(core.cache.l2)) {
        l2 = std::max(l2, core.cache.l2);
      }
      if (PlausibleCache(core.cache.l3)) {
        l3 = std::max(l3, core.cache.l3);
      }
      if (PlausibleFrequencyHz(core.max_frequency_hz)) {
        max_hz = std::max(max_hz, core.max_frequency_hz);
      }
      if (PlausibleFrequencyHz(core.regular_frequency_hz)) {
        regular_hz = std::max(regular_hz, core.regular_frequency_hz);
      }
    }

    // Only overwrite the native provider's values where hwinfo actually has
    // something. On macOS it leaves most of the cache fields unpopulated.
    if (l1d != 0) {
      s.l1d_bytes = l1d;
    }
    if (l1i != 0) {
      s.l1i_bytes = l1i;
    }
    if (l2 != 0) {
      s.l2_bytes = l2;
    }
    if (l3 != 0) {
      s.l3_bytes = l3;
    }
    if (max_hz != 0) {
      s.max_clock_mhz = static_cast<std::uint32_t>(max_hz / 1'000'000ull);
    }
    if (regular_hz != 0) {
      s.base_clock_mhz = static_cast<std::uint32_t>(regular_hz / 1'000'000ull);
    }
  }

  const hwinfo::Memory memory;
  if (memory.size() > 0) {
    s.ram_total_bytes = memory.size();
    any = true;
  }
  if (memory.available() > 0) {
    s.ram_available_bytes = memory.available();
  }

  const hwinfo::OS os;
  if (!os.name().empty()) {
    s.os_name = os.name();
    any = true;
  }
  if (!os.version().empty()) {
    s.os_version = os.version();
  }
  if (!os.kernel().empty()) {
    s.kernel_version = os.kernel();
  }

  if (any) {
    s.provider = "hwinfo (lfreist) + native fallback";
  }
}

#endif  // MB_USE_HWINFO

HardwareSnapshot Build() {
  HardwareSnapshot s;
  s.arch_name = MB_ARCH_NAME;

  FillNative(s);
#if defined(MB_USE_HWINFO)
  FillHwinfo(s);
#endif

  // Never report a cache line smaller than what we align to.
  if (s.cache_line_bytes == 0) {
    s.cache_line_bytes = static_cast<std::uint32_t>(kCacheLineBytes);
  }
  if (s.logical_cores == 0) {
    s.logical_cores = s.physical_cores;
  }
  return s;
}

}  // namespace

const HardwareSnapshot& QueryHardware() {
  static const HardwareSnapshot cached = Build();
  return cached;
}

std::string FormatBytes(std::uint64_t bytes) {
  if (bytes == 0) {
    return "n/a";
  }
  static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  char buf[64];
  if (unit == 0 || value >= 100.0) {
    std::snprintf(buf, sizeof(buf), "%.0f %s", value, kUnits[unit]);
  } else {
    std::snprintf(buf, sizeof(buf), "%.1f %s", value, kUnits[unit]);
  }
  return buf;
}

std::string FormatClock(std::uint32_t mhz) {
  if (mhz == 0) {
    return "not exposed";
  }
  char buf[64];
  if (mhz >= 1000) {
    std::snprintf(buf, sizeof(buf), "%.2f GHz", static_cast<double>(mhz) / 1000.0);
  } else {
    std::snprintf(buf, sizeof(buf), "%u MHz", mhz);
  }
  return buf;
}

}  // namespace mb
