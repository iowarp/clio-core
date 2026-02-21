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

#ifndef HSHM_SYSINFO_INFO_H_
#define HSHM_SYSINFO_INFO_H_

#include "hermes_shm/constants/macros.h"
#if HSHM_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
#include <sys/sysinfo.h>
#endif
#include <unistd.h>
#endif

#include <fstream>
#include <iostream>

#include "hermes_shm/thread/thread_model/thread_model.h"
#include "hermes_shm/util/formatter.h"
#include "hermes_shm/util/singleton.h"

#define HSHM_SYSTEM_INFO \
  hshm::LockfreeCrossSingleton<hshm::SystemInfo>::GetInstance()
#define HSHM_SYSTEM_INFO_T hshm::SystemInfo *

namespace hshm {

/** Dynamically load shared libraries */
struct SharedLibrary {
  void *handle_;

  SharedLibrary() = default;
  HSHM_DLL SharedLibrary(const std::string &name);
  HSHM_DLL ~SharedLibrary();

  // Delete copy operations
  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

  // Move operations
  HSHM_DLL SharedLibrary(SharedLibrary &&other) noexcept;
  HSHM_DLL SharedLibrary &operator=(SharedLibrary &&other) noexcept;

  HSHM_DLL void Load(const std::string &name);
  HSHM_DLL void *GetSymbol(const std::string &name);
  HSHM_DLL std::string GetError() const;

  bool IsNull() { return handle_ == nullptr; }
};

/** File wrapper */
union File {
  int posix_fd_;
  HANDLE windows_fd_;
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
#if HSHM_IS_HOST
  std::vector<size_t> cur_cpu_freq_;
#endif

 public:
  HSHM_CROSS_FUN
  SystemInfo() { RefreshInfo(); }

  HSHM_CROSS_FUN
  void RefreshInfo() {
#if HSHM_IS_HOST
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

  HSHM_DLL void RefreshCpuFreqKhz();

  HSHM_DLL size_t GetCpuFreqKhz(int cpu);

  HSHM_DLL size_t GetCpuMaxFreqKhz(int cpu);

  HSHM_DLL size_t GetCpuMinFreqKhz(int cpu);

  HSHM_DLL size_t GetCpuMinFreqMhz(int cpu);

  HSHM_DLL size_t GetCpuMaxFreqMhz(int cpu);

  HSHM_DLL void SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz);

  HSHM_DLL void SetCpuFreqKhz(int cpu, size_t cpu_freq_khz);

  HSHM_DLL void SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz);

  HSHM_DLL void SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz);

  HSHM_DLL static int GetCpuCount();

  HSHM_DLL static int GetPageSize();

  HSHM_DLL static int GetTid();

  HSHM_DLL static int GetPid();

  HSHM_DLL static int GetUid();

  HSHM_DLL static int GetGid();

  HSHM_DLL static size_t GetRamCapacity();

  HSHM_DLL static void YieldThread();

  HSHM_DLL static bool CreateTls(ThreadLocalKey &key, void *data);

  HSHM_DLL static bool SetTls(const ThreadLocalKey &key, void *data);

  HSHM_DLL static void *GetTls(const ThreadLocalKey &key);

  HSHM_DLL static bool CreateNewSharedMemory(File &fd, const std::string &name,
                                             size_t size);

  HSHM_DLL static bool OpenSharedMemory(File &fd, const std::string &name);

  HSHM_DLL static void CloseSharedMemory(File &file);

  HSHM_DLL static void DestroySharedMemory(const std::string &name);

  HSHM_DLL static void *MapPrivateMemory(size_t size);

  HSHM_DLL static void *MapSharedMemory(const File &fd, size_t size, i64 off);

  /**
   * Map a contiguous memory region with mixed private/shared mapping
   *
   * Creates a contiguous virtual memory region where:
   * - First private_size bytes: MAP_PRIVATE | MAP_ANONYMOUS (process-local)
   * - Remaining shared_size bytes: MAP_SHARED from fd (inter-process)
   *
   * @param fd File descriptor for shared memory (from shm_open)
   * @param private_size Size of private region at the beginning
   * @param shared_size Size of shared region following the private region
   * @param shared_offset Offset into fd for the shared mapping (usually 0)
   * @return Pointer to the beginning of the contiguous region (private region start),
   *         or nullptr on failure
   *
   * Note: The entire region must be unmapped with a single UnmapMemory call
   *       using total_size = private_size + shared_size
   */
  HSHM_DLL static void *MapMixedMemory(const File &fd, size_t private_size,
                                        size_t shared_size, i64 shared_offset = 0);

  HSHM_DLL static void UnmapMemory(void *ptr, size_t size);

  HSHM_DLL static void *AlignedAlloc(size_t alignment, size_t size);

  HSHM_DLL static std::string Getenv(
      const char *name, size_t max_size = hshm::Unit<size_t>::Megabytes(1));

  static std::string Getenv(
      const std::string &name,
      size_t max_size = hshm::Unit<size_t>::Megabytes(1)) {
    return Getenv(name.c_str(), max_size);
  }

  HSHM_DLL static void Setenv(const char *name, const std::string &value,
                              int overwrite);

  HSHM_DLL static void Unsetenv(const char *name);

  HSHM_DLL static bool IsProcessAlive(int pid);

  HSHM_DLL static std::string GetModuleDirectory();

  HSHM_DLL static std::string GetLibrarySearchPathVar();

  HSHM_DLL static char GetPathListSeparator();

  HSHM_DLL static std::string GetSharedLibExtension();
};

}  // namespace hshm

#undef WIN32_LEAN_AND_MEAN

#endif  // HSHM_SYSINFO_INFO_H_
