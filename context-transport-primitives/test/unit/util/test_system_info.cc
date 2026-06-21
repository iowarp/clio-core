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

#include <cstdlib>
#include <string>
#include <vector>

using ctp::SystemInfo;

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
