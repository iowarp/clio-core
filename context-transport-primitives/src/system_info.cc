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

#define CTP_COMPILING_DLL
#define __HSHM_IS_COMPILING__

#include <clio_ctp/util/env_compat.h>
#include "clio_ctp/introspect/system_info.h"

#include <climits>
#ifdef __linux__
#include <linux/limits.h>  // PATH_MAX on some Linux toolchains
#endif
// LCOV_EXCL_START — compile-time fallback, unreachable on standard Linux
#ifndef PATH_MAX
#define PATH_MAX 4096  // POSIX default; not always in <climits> under NVHPC
#endif
// LCOV_EXCL_STOP
#include <cstdlib>
#include <string>

#include "clio_ctp/constants/macros.h"
// MSan: inform sanitizer that mmap-backed memory is initialized by the kernel
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define CTP_MSAN_UNPOISON(ptr, size) __msan_unpoison((ptr), (size))
#else
#define CTP_MSAN_UNPOISON(ptr, size) ((void)0)
#endif
#else
#define CTP_MSAN_UNPOISON(ptr, size) ((void)0)
#endif
#if CTP_ENABLE_PROCFS_SYSINFO
#include <arpa/inet.h>
#include <dirent.h>
#include <dlfcn.h>
#include <ifaddrs.h>
#include <libgen.h>
#include <limits.h>
#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>
// LINUX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#if __linux__
#include <sys/sysinfo.h>
#else
#include <sys/sysctl.h>
#endif
#include <sys/types.h>
#include <unistd.h>
#if __linux__
#include <linux/memfd.h>
#endif
#if __APPLE__
#include <pthread.h>
#endif
// WINDOWS
#elif CTP_ENABLE_WINDOWS_SYSINFO
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <windows.h>
#include <filesystem>
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "ws2_32.lib")
#else
#error \
    "Must define either CTP_ENABLE_PROCFS_SYSINFO or CTP_ENABLE_WINDOWS_SYSINFO"
#endif

namespace ctp {

#if CTP_ENABLE_WINDOWS_SYSINFO
// Translate a POSIX shm_open()-style name ("/ctp_xxx") into a Win32
// kernel-object name. Win32 names can't contain `/`; the "Local\" prefix
// places the object in the user's session namespace, which is correct
// for per-host SHM. Returns the cleaned name suitable for both
// CreateFileMapping and OpenFileMapping.
static std::string WinShmName(const std::string &posix_name) {
  std::string base = posix_name;
  while (!base.empty() && (base.front() == '/' || base.front() == '\\')) {
    base.erase(0, 1);
  }
  for (auto &c : base) {
    if (c == '/' || c == '\\') c = '_';
  }
  return "Local\\" + base;
}
#endif

void SystemInfo::RefreshCpuFreqKhz() {
#if CTP_IS_HOST
  for (int i = 0; i < ncpu_; ++i) {
    cur_cpu_freq_[i] = GetCpuFreqKhz(i);
  }
#endif
}

size_t SystemInfo::GetCpuFreqKhz(int cpu) {
#if CTP_IS_HOST
#if CTP_ENABLE_PROCFS_SYSINFO
  // Read /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_cur_freq
  // Use snprintf to build the path so MSan can track the buffer as initialized
  char cpu_path[256];
  snprintf(cpu_path, sizeof(cpu_path),
           "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_cur_freq", cpu);
  std::ifstream cpu_file(cpu_path);
  size_t freq_khz = 0;
  cpu_file >> freq_khz;
  return freq_khz;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMaxFreqKhz(int cpu) {
#if CTP_IS_HOST
#if CTP_ENABLE_PROCFS_SYSINFO
  // Read /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq
  char cpu_path[256];
  snprintf(cpu_path, sizeof(cpu_path),
           "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
  std::ifstream cpu_file(cpu_path);
  size_t freq_khz = 0;
  cpu_file >> freq_khz;
  return freq_khz;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMinFreqKhz(int cpu) {
#if CTP_IS_HOST
#if CTP_ENABLE_PROCFS_SYSINFO
  // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
  std::string cpu_str = ctp::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_min_freq", cpu);
  std::ifstream cpu_file(cpu_str);
  size_t freq_khz;
  cpu_file >> freq_khz;
  return freq_khz;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMinFreqMhz(int cpu) {
  return GetCpuMinFreqKhz(cpu) / 1000;
}

size_t SystemInfo::GetCpuMaxFreqMhz(int cpu) {
  return GetCpuMaxFreqKhz(cpu) / 1000;
}

void SystemInfo::SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz) {
  SetCpuFreqKhz(cpu, cpu_freq_mhz * 1000);
}

void SystemInfo::SetCpuFreqKhz(int cpu, size_t cpu_freq_khz) {
  SetCpuMinFreqKhz(cpu, cpu_freq_khz);
  SetCpuMaxFreqKhz(cpu, cpu_freq_khz);
}

void SystemInfo::SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz) {
#if CTP_ENABLE_PROCFS_SYSINFO
  std::string cpu_str = ctp::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq", cpu);
  std::ofstream min_freq_file(cpu_str);
  min_freq_file << cpu_freq_khz;
#endif
}

void SystemInfo::SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz) {
#if CTP_ENABLE_PROCFS_SYSINFO
  std::string cpu_str = ctp::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", cpu);
  std::ofstream max_freq_file(cpu_str);
  max_freq_file << cpu_freq_khz;
#endif
}

int SystemInfo::GetCpuCount() {
#if CTP_ENABLE_PROCFS_SYSINFO

#if __linux__
  return get_nprocs_conf();
#else
  int count;
  using size_t = std::size_t;
  size_t count_len = sizeof(count);
#if __APPLE__
  if (sysctlbyname("hw.physicalcpu", &count, &count_len, NULL, 0) == -1) {
#else
  int mib[2];
  if (sysctl(mib, 2, &count, &count_len, NULL, 0) == -1) {
#endif
    perror("sysctl");
    return 1;
  }
  return count;
#endif

#elif CTP_ENABLE_WINDOWS_SYSINFO
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwNumberOfProcessors;

#endif
}

int SystemInfo::GetPageSize() {
#if CTP_ENABLE_PROCFS_SYSINFO
  return getpagesize();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  if (sys_info.dwAllocationGranularity != 0) {
    return sys_info.dwAllocationGranularity;
  }
  return sys_info.dwPageSize;
#endif
}

int SystemInfo::GetTid() {
#if CTP_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
  return (pid_t)syscall(SYS_gettid);
#elif __APPLE__
  uint64_t tid = 0;
  pthread_threadid_np(nullptr, &tid);
  return static_cast<int>(tid);
#else
  return GetPid();
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return GetCurrentThreadId();
#endif
}

int SystemInfo::GetPid() {
#if CTP_ENABLE_PROCFS_SYSINFO
#ifdef SYS_getpid
#ifdef __OpenBSD__
  return (pid_t)getpid();
#else
  return (pid_t)syscall(SYS_getpid);
#endif
#else
#warning "GetPid is not defined"
  return 0;
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return GetCurrentProcessId();
#endif
}

int SystemInfo::GetUid() {
#if CTP_ENABLE_PROCFS_SYSINFO
  return getuid();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
};

int SystemInfo::GetGid() {
#if CTP_ENABLE_PROCFS_SYSINFO
  return getgid();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
};

size_t SystemInfo::GetRamCapacity() {
#if CTP_ENABLE_PROCFS_SYSINFO
#if __APPLE__ || __OpenBSD__
  int mib[2];
  uint64_t mem_total;  // Use uint64_t for memory sizes

  mib[0] = CTL_HW;
#if __APPLE__
  mib[1] = HW_MEMSIZE;  // This is what you're looking for
#else
  mib[1] = HW_PHYSMEM;
#endif
  using size_t = std::size_t;
  size_t len = sizeof(mem_total);
  if (sysctl(mib, 2, &mem_total, &len, NULL, 0) == -1) {
    perror("sysctl");
    return 1;
  } else {
    return mem_total;
  }
#else
  struct sysinfo info;
  sysinfo(&info);
  return info.totalram;
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
  MEMORYSTATUSEX mem_info;
  mem_info.dwLength = sizeof(mem_info);
  GlobalMemoryStatusEx(&mem_info);
  return (size_t)mem_info.ullTotalPhys;
#endif
}

size_t SystemInfo::GetRamAvailable() {
#if CTP_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) return 0;
  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      size_t kb = 0;
      std::sscanf(line.c_str(), "MemAvailable: %zu", &kb);
      return kb * 1024;  // convert kB to bytes
    }
  }
  return 0;
#else
  return 0;
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
  MEMORYSTATUSEX mem_info;
  mem_info.dwLength = sizeof(mem_info);
  GlobalMemoryStatusEx(&mem_info);
  return static_cast<size_t>(mem_info.ullAvailPhys);
#else
  return 0;
#endif
}

CpuTimes SystemInfo::GetCpuTimes() {
  CpuTimes ct = {};
#if CTP_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
  std::ifstream stat("/proc/stat");
  if (stat.is_open()) {
    std::string cpu_label;
    stat >> cpu_label >> ct.user >> ct.nice >> ct.system >> ct.idle
         >> ct.iowait >> ct.irq >> ct.softirq >> ct.steal;
  }
#endif
#endif
  return ct;
}

void SystemInfo::YieldThread() {
#if CTP_ENABLE_PROCFS_SYSINFO
  sched_yield();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  Yield();
#endif
}

bool SystemInfo::CreateTls(ThreadLocalKey &key, void *data) {
#if CTP_ENABLE_PROCFS_SYSINFO
  int ret = pthread_key_create(&key.pthread_key_, nullptr);
  if (ret != 0) {
    return false;
  }
  return SetTls(key, data);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  key.windows_key_ = TlsAlloc();
  if (key.windows_key_ == TLS_OUT_OF_INDEXES) {
    return false;
  }
  return TlsSetValue(key.windows_key_, data);
#endif
}

bool SystemInfo::SetTls(const ThreadLocalKey &key, void *data) {
#if CTP_ENABLE_PROCFS_SYSINFO
  return pthread_setspecific(key.pthread_key_, data) == 0;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return TlsSetValue(key.windows_key_, data);
#endif
}

void *SystemInfo::GetTls(const ThreadLocalKey &key) {
#if CTP_ENABLE_PROCFS_SYSINFO
  return pthread_getspecific(key.pthread_key_);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return TlsGetValue(key.windows_key_);
#endif
}

std::string SystemInfo::GetMemfdDir() {
  // Default is /tmp/chimaera_$USER, but some sites have a tiny /
  // partition where /tmp fills up (ares compute nodes are a known
  // example: a stale prior run -- or any other user's clutter --
  // can leave no space, mkdir under /tmp silently fails, and the
  // subsequent memfd symlink + shm_open returns ENOENT here. Allow
  // env override so deployments can point chimaera's per-user
  // bookkeeping at an NFS-backed location (e.g. $HOME).
  const char *override_dir = ctp::env::GetCompat("MEMFD_DIR");
  if (override_dir && *override_dir) {
    return std::string(override_dir);
  }
  const char *user = getenv("USER");
  if (!user) user = "unknown";
  return std::string("/tmp/chimaera_") + user;
}

std::string SystemInfo::GetMemfdPath(const std::string &name) {
  // Strip leading '/' from name if present
  const char *base = name.c_str();
  if (base[0] == '/') {
    base++;
  }
  return GetMemfdDir() + "/" + base;
}

void SystemInfo::EnsureMemfdDir() {
  std::string dir = GetMemfdDir();
#if CTP_ENABLE_PROCFS_SYSINFO && __linux__
  mkdir(dir.c_str(), 0700);
#endif
}

bool SystemInfo::CreateNewSharedMemory(File &fd, const std::string &name,
                                       size_t size) {
#if CTP_ENABLE_PROCFS_SYSINFO
#if __linux__
  fd.posix_fd_ = memfd_create(name.c_str(), 0);
  if (fd.posix_fd_ < 0) {
    return false;
  }
  int ret = ftruncate(fd.posix_fd_, size);
  if (ret < 0) {
    close(fd.posix_fd_);
    return false;
  }
  EnsureMemfdDir();
  std::string memfd_path = GetMemfdPath(name);
  unlink(memfd_path.c_str());
  std::string proc_path =
      "/proc/" + std::to_string(getpid()) + "/fd/" + std::to_string(fd.posix_fd_);
  if (symlink(proc_path.c_str(), memfd_path.c_str()) < 0) {
    close(fd.posix_fd_);
    return false;
  }
  return true;
#else
  fd.posix_fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd.posix_fd_ < 0) {
    return false;
  }
  int ret = ftruncate(fd.posix_fd_, size);
  if (ret < 0) {
    close(fd.posix_fd_);
    return false;
  }
  return true;
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
  // POSIX shared memory names start with `/` (e.g. "/ctp_shm_42"). Win32
  // kernel object names can't contain `/`, so map to "Local\<base>" — the
  // per-session namespace, which is right for single-host SHM. Without
  // this the mapping was created anonymously (nullptr name) and could
  // not be reopened by name, breaking every OpenSharedMemory.
  std::string win_name = WinShmName(name);
  fd.windows_fd_ =
      CreateFileMapping(INVALID_HANDLE_VALUE,    // use paging file
                        nullptr,                 // default security
                        PAGE_READWRITE,          // read/write access
                        0,           // maximum object size (high-order DWORD)
                        static_cast<DWORD>(size),  // low-order DWORD
                        win_name.c_str());         // mapping object name
  return fd.windows_fd_ != nullptr;
#endif
}

bool SystemInfo::OpenSharedMemory(File &fd, const std::string &name) {
#if CTP_ENABLE_PROCFS_SYSINFO
#if __linux__
  std::string memfd_path = GetMemfdPath(name);
  fd.posix_fd_ = open(memfd_path.c_str(), O_RDWR);
  return fd.posix_fd_ >= 0;
#else
  fd.posix_fd_ = shm_open(name.c_str(), O_RDWR, 0666);
  return fd.posix_fd_ >= 0;
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
  std::string win_name = WinShmName(name);
  fd.windows_fd_ =
      OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, win_name.c_str());
  return fd.windows_fd_ != nullptr;
#endif
}

void SystemInfo::CloseSharedMemory(File &file) {
#if CTP_ENABLE_PROCFS_SYSINFO
  close(file.posix_fd_);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  CloseHandle(file.windows_fd_);
#endif
}

void SystemInfo::DestroySharedMemory(const std::string &name) {
#if CTP_ENABLE_PROCFS_SYSINFO
#if __linux__
  std::string memfd_path = GetMemfdPath(name);
  unlink(memfd_path.c_str());
#else
  shm_unlink(name.c_str());
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
#endif
}

void *SystemInfo::MapPrivateMemory(size_t size) {
#if CTP_ENABLE_PROCFS_SYSINFO
#if __APPLE__ || __OpenBSD__
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#else
  void *ptr = mmap64(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
  if (ptr != MAP_FAILED && ptr != nullptr) {
    CTP_MSAN_UNPOISON(ptr, size);
  }
  return ptr;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  void *ptr = VirtualAlloc(nullptr, size, MEM_COMMIT, PAGE_READWRITE);
  if (ptr) {
    CTP_MSAN_UNPOISON(ptr, size);
  }
  return ptr;
#endif
}

void *SystemInfo::MapSharedMemory(const File &fd, size_t size, i64 off) {
#if CTP_ENABLE_PROCFS_SYSINFO
#if __APPLE__ || __OpenBSD__
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd.posix_fd_, static_cast<off_t>(off));
#else
  void *ptr = mmap64(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd.posix_fd_, off);
#endif
  if (ptr == MAP_FAILED) {
    perror("mmap");
    return nullptr;
  }
  CTP_MSAN_UNPOISON(ptr, size);
  return ptr;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  // Convert i64 to low and high dwords
  DWORD highDword = (DWORD)((off >> 32) & 0xFFFFFFFF);
  DWORD lowDword = (DWORD)(off & 0xFFFFFFFF);
  void *ret = MapViewOfFile(fd.windows_fd_,       // handle to map object
                            FILE_MAP_ALL_ACCESS,  // read/write permission
                            highDword,            // file offset high
                            lowDword,             // file offset low
                            size);                // number of bytes to map
  if (ret == nullptr) {
    DWORD error = GetLastError();
    LPVOID msg_buf;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (LPTSTR)&msg_buf, 0, NULL);
    printf("MapViewOfFile failed with error: %s\n", (char *)msg_buf);
    LocalFree(msg_buf);
  }
  return ret;
#endif
}

void SystemInfo::UnmapMemory(void *ptr, size_t size) {
#if CTP_ENABLE_PROCFS_SYSINFO
  munmap(ptr, size);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  VirtualFree(ptr, size, MEM_RELEASE);
#endif
}

void *SystemInfo::AlignedAlloc(size_t alignment, size_t size) {
#if CTP_ENABLE_PROCFS_SYSINFO
  return aligned_alloc(alignment, size);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return _aligned_malloc(size, alignment);
#endif
}

void SystemInfo::AlignedFree(void *ptr) {
#if CTP_ENABLE_PROCFS_SYSINFO
  free(ptr);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  _aligned_free(ptr);
#endif
}

bool SystemInfo::IsProcessAlive(int pid) {
#if CTP_ENABLE_PROCFS_SYSINFO
  return kill(pid, 0) != -1 || errno != ESRCH;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                         static_cast<DWORD>(pid));
  if (h == NULL) return false;
  CloseHandle(h);
  return true;
#endif
}

std::string SystemInfo::GetModuleDirectory() {
#if CTP_ENABLE_PROCFS_SYSINFO
  Dl_info dl_info;
  void *addr = reinterpret_cast<void *>(&SystemInfo::GetModuleDirectory);
  if (dladdr(addr, &dl_info) == 0) return "";
  char resolved[PATH_MAX];
  if (realpath(dl_info.dli_fname, resolved) == nullptr) return "";
  std::string resolved_str(resolved);
  auto pos = resolved_str.rfind('/');
  return (pos != std::string::npos) ? resolved_str.substr(0, pos) : std::string();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  HMODULE hModule = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&SystemInfo::GetModuleDirectory),
          &hModule)) {
    return "";
  }
  char path[MAX_PATH];
  if (GetModuleFileNameA(hModule, path, MAX_PATH) == 0) return "";
  std::string path_str(path);
  auto pos2 = path_str.rfind('\\');
  if (pos2 == std::string::npos) pos2 = path_str.rfind('/');
  return (pos2 != std::string::npos) ? path_str.substr(0, pos2) : std::string();
#endif
}

uint64_t SystemInfo::ThreadCpuTimeNs() {
#if CTP_ENABLE_PROCFS_SYSINFO
  struct timespec ts;
#ifdef CLOCK_THREAD_CPUTIME_ID
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
  }
#endif
  return 0;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  FILETIME create_t, exit_t, kernel_t, user_t;
  if (!::GetThreadTimes(::GetCurrentThread(), &create_t, &exit_t, &kernel_t,
                         &user_t)) {
    return 0;
  }
  auto to_u64 = [](FILETIME f) {
    ULARGE_INTEGER u;
    u.LowPart = f.dwLowDateTime;
    u.HighPart = f.dwHighDateTime;
    return u.QuadPart;  // 100-ns intervals
  };
  return (to_u64(kernel_t) + to_u64(user_t)) * 100ull;
#endif
}

std::string SystemInfo::GetMathLibraryName() {
#if CTP_ENABLE_PROCFS_SYSINFO
  return "libm.so.6";
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return "ucrtbase.dll";
#endif
}

void SystemInfo::TerminateProcessNow(int exit_code) {
#if CTP_ENABLE_PROCFS_SYSINFO
  // No-op on POSIX: callers fall through and main() returns normally,
  // so static destructors, atexit handlers, leak sanitizers, and coverage
  // instrumentation all run as they should.
  (void)exit_code;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  // TerminateProcess skips DLL detach notifications and the entire
  // destructor chain, which is exactly what we need to dodge libzmq's
  // signaler abort during shutdown on Windows.
  ::TerminateProcess(::GetCurrentProcess(), static_cast<UINT>(exit_code));
  // Fallback in case TerminateProcess somehow returns (it shouldn't).
  ::_exit(exit_code);
#endif
}

void SystemInfo::SetCurrentThreadName(const std::string &name) {
#if CTP_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
  // pthread_setname_np truncates names longer than 15 chars (+ NUL).
  pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
#endif
#elif CTP_ENABLE_WINDOWS_SYSINFO
  // SetThreadDescription needs wide chars.
  std::wstring wname(name.begin(), name.end());
  SetThreadDescription(GetCurrentThread(), wname.c_str());
#endif
}

std::string SystemInfo::GetHomeDir() {
#if CTP_ENABLE_PROCFS_SYSINFO
  const char *home = std::getenv("HOME");
  return home ? std::string(home) : std::string();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  // USERPROFILE is the canonical home on Windows (C:\Users\<name>).
  // Fall back to HOMEDRIVE+HOMEPATH or HOME if a user explicitly set them.
  char buf[MAX_PATH];
  DWORD len = ::GetEnvironmentVariableA("USERPROFILE", buf, sizeof(buf));
  if (len > 0 && len < sizeof(buf)) return std::string(buf, len);
  len = ::GetEnvironmentVariableA("HOME", buf, sizeof(buf));
  if (len > 0 && len < sizeof(buf)) return std::string(buf, len);
  return std::string();
#endif
}

std::string SystemInfo::GetHostname() {
  char buf[256] = {0};
#if CTP_ENABLE_PROCFS_SYSINFO
  if (gethostname(buf, sizeof(buf) - 1) != 0) return "";
#elif CTP_ENABLE_WINDOWS_SYSINFO
  DWORD len = sizeof(buf);
  if (!GetComputerNameExA(ComputerNameDnsHostname, buf, &len)) return "";
#endif
  return std::string(buf);
}

std::string SystemInfo::GetModuleDirectoryFor(void *symbol) {
  if (!symbol) return "";
#if CTP_ENABLE_PROCFS_SYSINFO
  Dl_info dl_info;
  if (dladdr(symbol, &dl_info) == 0) return "";
  char resolved[PATH_MAX];
  if (realpath(dl_info.dli_fname, resolved) == nullptr) return "";
  std::string resolved_str(resolved);
  auto pos = resolved_str.rfind('/');
  return (pos != std::string::npos) ? resolved_str.substr(0, pos)
                                    : std::string();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  HMODULE hModule = nullptr;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(symbol), &hModule)) {
    return "";
  }
  char path[MAX_PATH];
  if (GetModuleFileNameA(hModule, path, MAX_PATH) == 0) return "";
  std::string path_str(path);
  auto pos2 = path_str.rfind('\\');
  if (pos2 == std::string::npos) pos2 = path_str.rfind('/');
  return (pos2 != std::string::npos) ? path_str.substr(0, pos2) : std::string();
#endif
}

std::vector<std::string> SystemInfo::GetLocalInterfaceIps() {
  std::vector<std::string> ips;
#if CTP_ENABLE_PROCFS_SYSINFO
  struct ifaddrs *ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0 || ifaddr == nullptr) return ips;
  for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    char buf[INET6_ADDRSTRLEN] = {0};
    const int fam = ifa->ifa_addr->sa_family;
    if (fam == AF_INET) {
      auto *sa = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
      if (inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf))) {
        ips.emplace_back(buf);
      }
    } else if (fam == AF_INET6) {
      auto *sa = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr);
      if (inet_ntop(AF_INET6, &sa->sin6_addr, buf, sizeof(buf))) {
        std::string s(buf);
        auto pct = s.find('%');
        ips.push_back(pct == std::string::npos ? s : s.substr(0, pct));
      }
    }
  }
  freeifaddrs(ifaddr);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  ULONG buf_len = 16 * 1024;
  std::vector<char> buf(buf_len);
  auto *addrs = reinterpret_cast<IP_ADAPTER_ADDRESSES *>(buf.data());
  const ULONG flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                      GAA_FLAG_SKIP_DNS_SERVER;
  ULONG rc = GetAdaptersAddresses(AF_UNSPEC, flags, nullptr, addrs, &buf_len);
  if (rc == ERROR_BUFFER_OVERFLOW) {
    buf.resize(buf_len);
    addrs = reinterpret_cast<IP_ADAPTER_ADDRESSES *>(buf.data());
    rc = GetAdaptersAddresses(AF_UNSPEC, flags, nullptr, addrs, &buf_len);
  }
  if (rc != NO_ERROR) return ips;
  for (auto *a = addrs; a != nullptr; a = a->Next) {
    for (auto *u = a->FirstUnicastAddress; u != nullptr; u = u->Next) {
      char text[INET6_ADDRSTRLEN] = {0};
      auto *sa = u->Address.lpSockaddr;
      if (sa->sa_family == AF_INET) {
        auto *in4 = reinterpret_cast<sockaddr_in *>(sa);
        if (inet_ntop(AF_INET, &in4->sin_addr, text, sizeof(text))) {
          ips.emplace_back(text);
        }
      } else if (sa->sa_family == AF_INET6) {
        auto *in6 = reinterpret_cast<sockaddr_in6 *>(sa);
        if (inet_ntop(AF_INET6, &in6->sin6_addr, text, sizeof(text))) {
          std::string s(text);
          auto pct = s.find('%');
          ips.push_back(pct == std::string::npos ? s : s.substr(0, pct));
        }
      }
    }
  }
#endif
  return ips;
}

std::vector<std::string> SystemInfo::ResolveHostname(const std::string &host) {
  std::vector<std::string> ips;
  struct addrinfo hints {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) {
    return ips;
  }
  for (struct addrinfo *p = res; p != nullptr; p = p->ai_next) {
    char buf[INET6_ADDRSTRLEN] = {0};
    if (p->ai_family == AF_INET) {
      auto *sa = reinterpret_cast<struct sockaddr_in *>(p->ai_addr);
      if (inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf))) {
        ips.emplace_back(buf);
      }
    } else if (p->ai_family == AF_INET6) {
      auto *sa = reinterpret_cast<struct sockaddr_in6 *>(p->ai_addr);
      if (inet_ntop(AF_INET6, &sa->sin6_addr, buf, sizeof(buf))) {
        std::string s(buf);
        auto pct = s.find('%');
        ips.push_back(pct == std::string::npos ? s : s.substr(0, pct));
      }
    }
  }
  freeaddrinfo(res);
  return ips;
}

std::vector<std::string> SystemInfo::ListDirectory(const std::string &path) {
  std::vector<std::string> entries;
#if CTP_ENABLE_PROCFS_SYSINFO
  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) return entries;
  while (struct dirent *entry = readdir(dir)) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    entries.emplace_back(entry->d_name);
  }
  closedir(dir);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  std::error_code ec;
  if (!std::filesystem::is_directory(path, ec)) return entries;
  for (auto &p : std::filesystem::directory_iterator(path, ec)) {
    if (ec) break;
    entries.push_back(p.path().filename().string());
  }
#endif
  return entries;
}

bool SystemInfo::RemoveFile(const std::string &path) {
#if CTP_ENABLE_PROCFS_SYSINFO
  return ::unlink(path.c_str()) == 0;
#elif CTP_ENABLE_WINDOWS_SYSINFO
  std::error_code ec;
  return std::filesystem::remove(path, ec);
#endif
}

std::string SystemInfo::GetLibrarySearchPathVar() {
#if CTP_ENABLE_PROCFS_SYSINFO
  return "LD_LIBRARY_PATH";
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return "PATH";
#endif
}

char SystemInfo::GetPathListSeparator() {
#if CTP_ENABLE_PROCFS_SYSINFO
  return ':';
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return ';';
#endif
}

std::string SystemInfo::GetSharedLibExtension() {
#if CTP_ENABLE_WINDOWS_SYSINFO
  return ".dll";
#elif __APPLE__
  return ".dylib";
#else
  return ".so";
#endif
}

std::string SystemInfo::Getenv(const char *name, size_t max_size) {
#if CTP_ENABLE_PROCFS_SYSINFO
  char *var = getenv(name);
  if (var == nullptr) {
    return "";
  }
  return std::string(var);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  std::string var;
  var.resize(max_size);
  DWORD len = GetEnvironmentVariable(name, var.data(),
                                     static_cast<DWORD>(var.size()));
  if (len == 0) {
    return "";
  }
  var.resize(len);
  return var;
#endif
  std::cout << "undefined" << std::endl;
  return "";
}

void SystemInfo::Setenv(const char *name, const std::string &value,
                        int overwrite) {
#if CTP_ENABLE_PROCFS_SYSINFO
  setenv(name, value.c_str(), overwrite);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  // _putenv_s updates BOTH the CRT environment block (which std::getenv
  // reads) and the Win32 process environment block (which
  // GetEnvironmentVariable reads). SetEnvironmentVariable only touches
  // the Win32 block, which would leave std::getenv blind to the new
  // value — and chi::env::GetCompat goes through std::getenv. Honor the
  // overwrite flag manually since _putenv_s always overwrites.
  if (!overwrite) {
    char probe[2];
    if (::GetEnvironmentVariableA(name, probe, sizeof(probe)) != 0) {
      return;  // already set, caller asked us not to overwrite
    }
  }
  (void)_putenv_s(name, value.c_str());
#endif
}

void SystemInfo::Unsetenv(const char *name) {
#if CTP_ENABLE_PROCFS_SYSINFO
  unsetenv(name);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  // Setting an env var to an empty string via _putenv_s removes it from
  // both the CRT and Win32 environment blocks.
  (void)_putenv_s(name, "");
#endif
}

SharedLibrary::SharedLibrary(const std::string &name) : handle_(nullptr) {
  Load(name);
}

SharedLibrary::~SharedLibrary() {
  if (handle_) {
#if CTP_ENABLE_PROCFS_SYSINFO
    dlclose(handle_);
#elif CTP_ENABLE_WINDOWS_SYSINFO
    ::FreeLibrary((HMODULE)handle_);
#endif
    handle_ = nullptr;
  }
}

void SharedLibrary::Load(const std::string &name) {
#if CTP_ENABLE_PROCFS_SYSINFO
  handle_ = dlopen(name.c_str(), RTLD_GLOBAL | RTLD_NOW);
#elif CTP_ENABLE_WINDOWS_SYSINFO
  handle_ = LoadLibraryA(name.c_str());
  if (!handle_) {
    DWORD err = ::GetLastError();
    char *buf = nullptr;
    DWORD len = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buf), 0, nullptr);
    if (buf) {
      error_string_.assign(buf, len);
      ::LocalFree(buf);
    } else {
      error_string_ = "LoadLibraryA failed: " + std::to_string(err);
    }
  } else {
    error_string_.clear();
  }
#endif
}

std::string SharedLibrary::GetError() const {
#if CTP_ENABLE_PROCFS_SYSINFO
  const char *err = dlerror();
  return err ? std::string(err) : std::string();
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return error_string_;
#endif
}

void *SharedLibrary::GetSymbol(const std::string &name) {
#if CTP_ENABLE_PROCFS_SYSINFO
  return dlsym(handle_, name.c_str());
#elif CTP_ENABLE_WINDOWS_SYSINFO
  return (void *)::GetProcAddress((HMODULE)handle_, name.c_str());
#endif
}

SharedLibrary::SharedLibrary(SharedLibrary &&other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

SharedLibrary &SharedLibrary::operator=(SharedLibrary &&other) noexcept {
  if (this != &other) {
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

}  // namespace ctp
