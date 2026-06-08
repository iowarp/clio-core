/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CTP_SYSINFO_INFO_H_
#define CTP_SYSINFO_INFO_H_

#include "clio_ctp/constants/macros.h"
#if CTP_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
#include <sys/sysinfo.h>
#endif
#include <unistd.h>
#endif

#include <fstream>
#include <iostream>

#include "clio_ctp/thread/thread_model/thread_model.h"
#include "clio_ctp/util/formatter.h"
#include "clio_ctp/util/singleton.h"

#define CTP_SYSTEM_INFO \
  ctp::LockfreeCrossSingleton<ctp::SystemInfo>::GetInstance()
#define CTP_SYSTEM_INFO_T ctp::SystemInfo *

namespace ctp {

/** Dynamically load shared libraries */
struct SharedLibrary {
  void *handle_ = nullptr;
  // Windows: captured from FormatMessageA on the most recent Load() failure.
  // POSIX: unused (we route GetError() through dlerror() which has its own
  // thread-local string slot).
  std::string error_string_;

  SharedLibrary() = default;
  CTP_DLL SharedLibrary(const std::string &name);
  CTP_DLL ~SharedLibrary();

  // Delete copy operations
  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

  // Move operations
  CTP_DLL SharedLibrary(SharedLibrary &&other) noexcept;
  CTP_DLL SharedLibrary &operator=(SharedLibrary &&other) noexcept;

  CTP_DLL void Load(const std::string &name);
  CTP_DLL void *GetSymbol(const std::string &name);
  CTP_DLL std::string GetError() const;

  bool IsNull() { return handle_ == nullptr; }
};

/** File wrapper.
 *  windows_fd_ is a Windows HANDLE, but HANDLE is just `void*` — store it as
 *  void* so this header never needs <windows.h> (which would leak Yield/min/max
 *  macros). system_info.cc casts to/from HANDLE where it calls the Win32 API. */
union File {
  int posix_fd_;
  void* windows_fd_;
};

/** Aggregate CPU tick counts from /proc/stat */
struct CpuTimes {
  uint64_t user;
  uint64_t nice;
  uint64_t system;
  uint64_t idle;
  uint64_t iowait;
  uint64_t irq;
  uint64_t softirq;
  uint64_t steal;

  uint64_t TotalActive() const {
    return user + nice + system + irq + softirq + steal;
  }
  uint64_t Total() const { return TotalActive() + idle + iowait; }
};

/** A unification of certain OS system calls */
class SystemInfo {
 public:
  int pid_;
  int ncpu_;
  int page_size_;
  int uid_;
  int gid_;
  size_t ram_size_;
#if CTP_IS_HOST
  std::vector<size_t> cur_cpu_freq_;
#endif

 public:
  CTP_CROSS_FUN
  SystemInfo() { RefreshInfo(); }

  CTP_CROSS_FUN
  void RefreshInfo() {
#if CTP_IS_HOST
    pid_ = GetPid();
    ncpu_ = GetCpuCount();
    page_size_ = GetPageSize();
    uid_ = GetUid();
    gid_ = GetGid();
    ram_size_ = GetRamCapacity();
    cur_cpu_freq_.resize(ncpu_);
    RefreshCpuFreqKhz();
#endif
  }

  CTP_DLL void RefreshCpuFreqKhz();

  CTP_DLL size_t GetCpuFreqKhz(int cpu);

  CTP_DLL size_t GetCpuMaxFreqKhz(int cpu);

  CTP_DLL size_t GetCpuMinFreqKhz(int cpu);

  CTP_DLL size_t GetCpuMinFreqMhz(int cpu);

  CTP_DLL size_t GetCpuMaxFreqMhz(int cpu);

  CTP_DLL void SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz);

  CTP_DLL void SetCpuFreqKhz(int cpu, size_t cpu_freq_khz);

  CTP_DLL void SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz);

  CTP_DLL void SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz);

  CTP_DLL static int GetCpuCount();

  CTP_DLL static int GetPageSize();

  CTP_DLL static int GetTid();

  CTP_DLL static int GetPid();

  CTP_DLL static int GetUid();

  CTP_DLL static int GetGid();

  CTP_DLL static size_t GetRamCapacity();

  CTP_DLL static size_t GetRamAvailable();

  CTP_DLL static CpuTimes GetCpuTimes();

  static float ComputeCpuUtilization(const CpuTimes &prev,
                                     const CpuTimes &curr) {
    uint64_t total_d = curr.Total() - prev.Total();
    if (total_d == 0) return 0.0f;
    uint64_t active_d = curr.TotalActive() - prev.TotalActive();
    return static_cast<float>(active_d) / static_cast<float>(total_d) * 100.0f;
  }

  CTP_DLL static void YieldThread();

  CTP_DLL static bool CreateTls(ThreadLocalKey &key, void *data);

  CTP_DLL static bool SetTls(const ThreadLocalKey &key, void *data);

  CTP_DLL static void *GetTls(const ThreadLocalKey &key);

  CTP_DLL static bool CreateNewSharedMemory(File &fd, const std::string &name,
                                             size_t size);

  CTP_DLL static bool OpenSharedMemory(File &fd, const std::string &name);

  CTP_DLL static void CloseSharedMemory(File &file);

  CTP_DLL static void DestroySharedMemory(const std::string &name);

  CTP_DLL static void *MapPrivateMemory(size_t size);

  CTP_DLL static void *MapSharedMemory(const File &fd, size_t size, i64 off);

  CTP_DLL static void UnmapMemory(void *ptr, size_t size);

  CTP_DLL static void *AlignedAlloc(size_t alignment, size_t size);

  /** Free memory returned by AlignedAlloc. POSIX accepts plain free()
   *  for aligned_alloc() pointers, but Windows _aligned_malloc() requires
   *  the matching _aligned_free() — using free() corrupts the CRT heap. */
  CTP_DLL static void AlignedFree(void *ptr);

  CTP_DLL static std::string Getenv(
      const char *name, size_t max_size = ctp::Unit<size_t>::Megabytes(1));

  static std::string Getenv(
      const std::string &name,
      size_t max_size = ctp::Unit<size_t>::Megabytes(1)) {
    return Getenv(name.c_str(), max_size);
  }

  CTP_DLL static void Setenv(const char *name, const std::string &value,
                              int overwrite);

  CTP_DLL static void Unsetenv(const char *name);

  /** Get the per-user chimaera tmp directory path (/tmp/chimaera_$USER) */
  CTP_DLL static std::string GetMemfdDir();

  /** Get the full path for a named file in the memfd directory */
  CTP_DLL static std::string GetMemfdPath(const std::string &name);

  /** Ensure the per-user memfd directory exists */
  CTP_DLL static void EnsureMemfdDir();

  CTP_DLL static bool IsProcessAlive(int pid);

  /** Local hostname (best-effort, empty on failure). */
  CTP_DLL static std::string GetHostname();

  /** User's home directory (HOME on POSIX, USERPROFILE on Windows).
   *  Returns empty string if neither is set. */
  CTP_DLL static std::string GetHomeDir();

  /** Set the calling thread's name (best-effort; truncated to OS limits). */
  CTP_DLL static void SetCurrentThreadName(const std::string &name);

  /** Total CPU time (user + kernel) consumed by the calling thread, in
   *  nanoseconds. POSIX uses clock_gettime(CLOCK_THREAD_CPUTIME_ID),
   *  Windows uses GetThreadTimes. Returns 0 on platforms without thread
   *  CPU-time support. */
  CTP_DLL static uint64_t ThreadCpuTimeNs();

  /** Terminate the calling process immediately on platforms that need it,
   *  skipping C++ static destructors, atexit handlers, and CRT cleanup.
   *  Windows: TerminateProcess(exit_code) — used to dodge the libzmq
   *  teardown abort that fires during ZMQ destructor unwind on Windows.
   *  POSIX: no-op (returns) so static destructors, leak sanitizers, and
   *  coverage instrumentation can finish normally.
   *
   *  Call sites should follow with `return exit_code;` so the Linux path
   *  has a normal-return out of main(). This API is deliberately NOT
   *  marked [[noreturn]] — its no-op behaviour on Linux means it can
   *  fall through. */
  CTP_DLL static void TerminateProcessNow(int exit_code);

  /** IPv4/IPv6 addresses bound to local interfaces (loopback included). */
  CTP_DLL static std::vector<std::string> GetLocalInterfaceIps();

  /** Resolve a hostname/IP literal to a list of IP strings (best-effort). */
  CTP_DLL static std::vector<std::string> ResolveHostname(
      const std::string &host);

  /** List non-special entries of a directory ("." and ".." filtered out). */
  CTP_DLL static std::vector<std::string> ListDirectory(
      const std::string &path);

  /** Remove a file (returns true on success). */
  CTP_DLL static bool RemoveFile(const std::string &path);

  CTP_DLL static std::string GetModuleDirectory();

  /** Directory of the shared library containing the given symbol. */
  CTP_DLL static std::string GetModuleDirectoryFor(void *symbol);

  CTP_DLL static std::string GetLibrarySearchPathVar();

  CTP_DLL static char GetPathListSeparator();

  CTP_DLL static std::string GetSharedLibExtension();

  /** Name of a shared library that's universally available on this OS
   *  and exports the standard libm math symbols (sin/cos/tan/...). Used
   *  by the cross-platform SharedLibrary smoke tests so they don't have
   *  to compile-switch on the library name. POSIX: "libm.so.6";
   *  Windows: "ucrtbase.dll" (UCRT exports the C math entry points). */
  CTP_DLL static std::string GetMathLibraryName();
};

}  // namespace ctp

#undef WIN32_LEAN_AND_MEAN

#endif  // CTP_SYSINFO_INFO_H_
