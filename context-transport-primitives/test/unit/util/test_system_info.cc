/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Unit tests for SystemInfo (system_info.cc): CPU info accessors, TLS,
 * memfd bookkeeping paths, hostname resolution, and directory listing.
 */

#include "basic_test.h"

#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/memory/backend/posix_shm_mmap.h>

#include <cstdlib>
#include <string>
#include <vector>

using ctp::SystemInfo;
using ctp::ipc::PosixShmMmap;

TEST_CASE("SystemInfoCpu") {
  int cpus = SystemInfo::GetCpuCount();
  REQUIRE(cpus >= 1);

  // Frequency getters: values are platform-dependent (may be 0 inside
  // containers without cpufreq), the point is exercising the read paths.
  auto *sysinfo = CTP_SYSTEM_INFO;
  sysinfo->RefreshCpuFreqKhz();
  (void)sysinfo->GetCpuFreqKhz(0);
  (void)sysinfo->GetCpuMaxFreqKhz(0);
  (void)sysinfo->GetCpuMinFreqKhz(0);
  (void)sysinfo->GetCpuMinFreqMhz(0);
  (void)sysinfo->GetCpuMaxFreqMhz(0);

  // Setters write to /sys cpufreq files; as an unprivileged user the
  // streams silently fail to open, which still covers the formatting and
  // write paths without changing system state.
  sysinfo->SetCpuFreqMhz(0, 1000);
  sysinfo->SetCpuFreqKhz(0, 1000000);
  sysinfo->SetCpuMinFreqKhz(0, 1000000);
  sysinfo->SetCpuMaxFreqKhz(0, 1000000);

  REQUIRE(SystemInfo::GetPageSize() >= 512);
  REQUIRE(SystemInfo::GetPid() > 0);
  REQUIRE(SystemInfo::GetTid() >= 0);
  (void)SystemInfo::GetUid();
  (void)SystemInfo::GetGid();
}

TEST_CASE("SystemInfoMemoryAndCpuTimes") {
  REQUIRE(SystemInfo::GetRamCapacity() > 0);
  (void)SystemInfo::GetRamAvailable();
  auto times = SystemInfo::GetCpuTimes();
  (void)times;
  SystemInfo::YieldThread();
}

TEST_CASE("SystemInfoTls") {
  ctp::ThreadLocalKey key;
  int value = 42;
  REQUIRE(SystemInfo::CreateTls(key, &value));
  int other = 7;
  REQUIRE(SystemInfo::SetTls(key, &other));
}

TEST_CASE("SystemInfoMemfdDir") {
  // CLIO_MEMFD_DIR override takes precedence (GetCompat also accepts the
  // legacy CLIO_MEMFD_DIR spelling).
  SystemInfo::Setenv("CLIO_MEMFD_DIR", "/tmp/ctp_memfd_test_override", 1);
  std::string dir = SystemInfo::GetMemfdDir();
  REQUIRE(dir == "/tmp/ctp_memfd_test_override");
  std::string path = SystemInfo::GetMemfdPath("unit_test_seg");
  REQUIRE(path.find("unit_test_seg") != std::string::npos);
  SystemInfo::Unsetenv("CLIO_MEMFD_DIR");

  // Default (user-derived) directory.
  std::string default_dir = SystemInfo::GetMemfdDir();
  REQUIRE(!default_dir.empty());
}

TEST_CASE("SystemInfoHostnameResolution") {
  // localhost must resolve to at least one address.
  std::vector<std::string> ips = SystemInfo::ResolveHostname("localhost");
  REQUIRE(!ips.empty());

  // RFC 2606 reserved TLD: resolution must fail and return empty.
  std::vector<std::string> none =
      SystemInfo::ResolveHostname("no-such-host.invalid");
  REQUIRE(none.empty());
}

TEST_CASE("SystemInfoSharedMemory") {
  const std::string seg = "ctp_sysinfo_test_seg";
  constexpr size_t kSize = 64 * 1024;

  // Create, map, write, unmap, reopen, destroy.
  ctp::File fd;
  SystemInfo::DestroySharedMemory(seg);  // clean slate; missing is a no-op
  REQUIRE(SystemInfo::CreateNewSharedMemory(fd, seg, kSize));
  void *mapped = SystemInfo::MapSharedMemory(fd, kSize, 0);
  REQUIRE(mapped != nullptr);
  memset(mapped, 0x5A, kSize);
  SystemInfo::UnmapMemory(mapped, kSize);

  ctp::File fd2;
  REQUIRE(SystemInfo::OpenSharedMemory(fd2, seg));
  SystemInfo::CloseSharedMemory(fd2);
  SystemInfo::CloseSharedMemory(fd);
  SystemInfo::DestroySharedMemory(seg);

  // Reopening a destroyed segment fails.
  ctp::File fd3;
  REQUIRE_FALSE(SystemInfo::OpenSharedMemory(fd3, seg));

  // Private anonymous mapping.
  void *priv = SystemInfo::MapPrivateMemory(kSize);
  REQUIRE(priv != nullptr);
  memset(priv, 1, kSize);
  SystemInfo::UnmapMemory(priv, kSize);
}

TEST_CASE("SystemInfoListDirectory") {
  std::vector<std::string> entries = SystemInfo::ListDirectory("/tmp");
  // /tmp exists; "." and ".." must be filtered out.
  for (const auto &e : entries) {
    REQUIRE(e != ".");
    REQUIRE(e != "..");
  }

  std::vector<std::string> missing =
      SystemInfo::ListDirectory("/nonexistent_ctp_dir");
  REQUIRE(missing.empty());
}

TEST_CASE("SystemInfoProcessAndModule") {
  // The current process must report as alive; a clearly-unused PID must not.
  REQUIRE(SystemInfo::IsProcessAlive(SystemInfo::GetPid()));
  // PID 0x7FFFFFFF is not a realistic live process on this host.
  REQUIRE_FALSE(SystemInfo::IsProcessAlive(0x7FFFFFFF));

  // GetModuleDirectory resolves the directory of the loaded module via
  // dladdr/realpath (POSIX) or GetModuleFileNameA (Windows); it must return a
  // non-empty absolute path for this binary.
  std::string mod_dir = SystemInfo::GetModuleDirectory();
  REQUIRE(!mod_dir.empty());
#ifdef _WIN32
  // Windows absolute paths are drive-letter rooted ("D:\...") or UNC ("\\...").
  bool is_absolute = (mod_dir.size() >= 2 && mod_dir[1] == ':') ||
                     (mod_dir.size() >= 2 && mod_dir[0] == '\\' &&
                      mod_dir[1] == '\\');
  REQUIRE(is_absolute);
#else
  REQUIRE(mod_dir.front() == '/');
#endif
}

TEST_CASE("SystemInfoSharedMemoryError") {
  // GetLastSharedMemoryError() renders whatever the platform's shared-memory
  // calls last reported: strerror(errno) on POSIX, FormatMessage over
  // GetLastError() plus the numeric code on Windows. It must always produce
  // something a human can read -- the point of it is that the previous
  // "shm_open failed: {strerror(errno)}" reported a Win32 commit-limit failure
  // as EAGAIN, because the Win32 calls do not set errno at all.
  std::string msg = SystemInfo::GetLastSharedMemoryError();
  REQUIRE(!msg.empty());

  // After a call that genuinely failed it must still be non-empty, and must
  // not fall through to the "unknown error" placeholder.
  ctp::File missing;
  REQUIRE_FALSE(
      SystemInfo::OpenSharedMemory(missing, "ctp_no_such_segment_xyz"));
  std::string after = SystemInfo::GetLastSharedMemoryError();
  REQUIRE(!after.empty());
  REQUIRE(after != "unknown error");
}

TEST_CASE("SystemInfoSharedMemoryCreateFailure") {
  // A name far past any platform's limit: memfd_create(2) caps the name at
  // 249 bytes, and the macOS/Windows branches open a file whose path this
  // makes far too long. Every platform therefore fails the create, which is
  // the one path that reports through GetLastSharedMemoryError().
  const std::string too_long(4096, 'x');

  ctp::File fd;
  REQUIRE_FALSE(SystemInfo::CreateNewSharedMemory(fd, too_long, 1024 * 1024));

  // The same failure one layer up: shm_init() must report it and return false
  // rather than going on to map a backend it never created.
  PosixShmMmap backend;
  REQUIRE_FALSE(backend.shm_init(ctp::ipc::MemoryBackendId::GetRoot(),
                                 1024 * 1024, too_long));

  // Destroying a segment that was never created is a no-op everywhere, and
  // must stay one now that the Windows branch actually deletes a file.
  SystemInfo::DestroySharedMemory(too_long);
  SystemInfo::DestroySharedMemory("ctp_no_such_segment_xyz");
}
