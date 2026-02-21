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

#define HSHM_COMPILING_DLL
#define __HSHM_IS_COMPILING__

#include "hermes_shm/introspect/system_info.h"

#include <cstdlib>
#include <filesystem>

#include "hermes_shm/constants/macros.h"
#if HSHM_ENABLE_PROCFS_SYSINFO
#include <dlfcn.h>
#include <signal.h>
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
// WINDOWS
#elif HSHM_ENABLE_WINDOWS_SYSINFO
#include <windows.h>
#else
#error \
    "Must define either HSHM_ENABLE_PROCFS_SYSINFO or HSHM_ENABLE_WINDOWS_SYSINFO"
#endif

namespace hshm {

void SystemInfo::RefreshCpuFreqKhz() {
#if HSHM_IS_HOST
  for (int i = 0; i < ncpu_; ++i) {
    cur_cpu_freq_[i] = GetCpuFreqKhz(i);
  }
#endif
}

size_t SystemInfo::GetCpuFreqKhz(int cpu) {
#if HSHM_IS_HOST
#if HSHM_ENABLE_PROCFS_SYSINFO
  // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_cur_freq", cpu);
  std::ifstream cpu_file(cpu_str);
  size_t freq_khz;
  cpu_file >> freq_khz;
  return freq_khz;
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMaxFreqKhz(int cpu) {
#if HSHM_IS_HOST
#if HSHM_ENABLE_PROCFS_SYSINFO
  // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", cpu);
  std::ifstream cpu_file(cpu_str);
  size_t freq_khz;
  cpu_file >> freq_khz;
  return freq_khz;
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMinFreqKhz(int cpu) {
#if HSHM_IS_HOST
#if HSHM_ENABLE_PROCFS_SYSINFO
  // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_min_freq", cpu);
  std::ifstream cpu_file(cpu_str);
  size_t freq_khz;
  cpu_file >> freq_khz;
  return freq_khz;
#elif HSHM_ENABLE_WINDOWS_SYSINFO
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
#if HSHM_ENABLE_PROCFS_SYSINFO
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq", cpu);
  std::ofstream min_freq_file(cpu_str);
  min_freq_file << cpu_freq_khz;
#endif
}

void SystemInfo::SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", cpu);
  std::ofstream max_freq_file(cpu_str);
  max_freq_file << cpu_freq_khz;
#endif
}

int SystemInfo::GetCpuCount() {
#if HSHM_ENABLE_PROCFS_SYSINFO

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

#elif HSHM_ENABLE_WINDOWS_SYSINFO
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwNumberOfProcessors;

#endif
}

int SystemInfo::GetPageSize() {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return getpagesize();
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  if (sys_info.dwAllocationGranularity != 0) {
    return sys_info.dwAllocationGranularity;
  }
  return sys_info.dwPageSize;
#endif
}

int SystemInfo::GetTid() {
#if HSHM_ENABLE_PROCFS_SYSINFO
#ifdef SYS_gettid
#ifdef __linux__
  return (pid_t)syscall(SYS_gettid);
#else
  return GetPid();
#endif
#else
#warning "GetTid is not defined"
  return GetPid();
#endif
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return GetCurrentThreadId();
#endif
}

int SystemInfo::GetPid() {
#if HSHM_ENABLE_PROCFS_SYSINFO
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
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return GetCurrentProcessId();
#endif
}

int SystemInfo::GetUid() {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return getuid();
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
};

int SystemInfo::GetGid() {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return getgid();
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return 0;
#endif
};

size_t SystemInfo::GetRamCapacity() {
#if HSHM_ENABLE_PROCFS_SYSINFO
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
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  MEMORYSTATUSEX mem_info;
  mem_info.dwLength = sizeof(mem_info);
  GlobalMemoryStatusEx(&mem_info);
  return (size_t)mem_info.ullTotalPhys;
#endif
}

void SystemInfo::YieldThread() {
#if HSHM_ENABLE_PROCFS_SYSINFO
  sched_yield();
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  Yield();
#endif
}

bool SystemInfo::CreateTls(ThreadLocalKey &key, void *data) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  key.pthread_key_ = pthread_key_create(&key.pthread_key_, nullptr);
  return key.pthread_key_ == 0;
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  key.windows_key_ = TlsAlloc();
  if (key.windows_key_ == TLS_OUT_OF_INDEXES) {
    return false;
  }
  return TlsSetValue(key.windows_key_, data);
#endif
}

bool SystemInfo::SetTls(const ThreadLocalKey &key, void *data) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return pthread_setspecific(key.pthread_key_, data) == 0;
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return TlsSetValue(key.windows_key_, data);
#endif
}

void *SystemInfo::GetTls(const ThreadLocalKey &key) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return pthread_getspecific(key.pthread_key_);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return TlsGetValue(key.windows_key_);
#endif
}

#if HSHM_ENABLE_PROCFS_SYSINFO && __linux__
static std::string GetMemfdDir() {
  const char *user = getenv("USER");
  if (!user) user = "unknown";
  return std::string("/tmp/chimaera_") + user;
}

static std::string GetMemfdPath(const std::string &name) {
  // Strip leading '/' from name if present
  const char *base = name.c_str();
  if (base[0] == '/') {
    base++;
  }
  return GetMemfdDir() + "/" + base;
}

static void EnsureMemfdDir() {
  std::string dir = GetMemfdDir();
  mkdir(dir.c_str(), 0700);
}
#endif

bool SystemInfo::CreateNewSharedMemory(File &fd, const std::string &name,
                                       size_t size) {
#if HSHM_ENABLE_PROCFS_SYSINFO
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
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  fd.windows_fd_ =
      CreateFileMapping(INVALID_HANDLE_VALUE,  // use paging file
                        nullptr,               // default security
                        PAGE_READWRITE,        // read/write access
                        0,         // maximum object size (high-order DWORD)
                        size,      // maximum object size (low-order DWORD)
                        nullptr);  // name of mapping object
  return fd.windows_fd_ != nullptr;
#endif
}

bool SystemInfo::OpenSharedMemory(File &fd, const std::string &name) {
#if HSHM_ENABLE_PROCFS_SYSINFO
#if __linux__
  std::string memfd_path = GetMemfdPath(name);
  fd.posix_fd_ = open(memfd_path.c_str(), O_RDWR);
  return fd.posix_fd_ >= 0;
#else
  fd.posix_fd_ = shm_open(name.c_str(), O_RDWR, 0666);
  return fd.posix_fd_ >= 0;
#endif
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  fd.windows_fd_ = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
  return fd.windows_fd_ != nullptr;
#endif
}

void SystemInfo::CloseSharedMemory(File &file) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  close(file.posix_fd_);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  CloseHandle(file.windows_fd_);
#endif
}

void SystemInfo::DestroySharedMemory(const std::string &name) {
#if HSHM_ENABLE_PROCFS_SYSINFO
#if __linux__
  std::string memfd_path = GetMemfdPath(name);
  unlink(memfd_path.c_str());
#else
  shm_unlink(name.c_str());
#endif
#elif HSHM_ENABLE_WINDOWS_SYSINFO
#endif
}

void *SystemInfo::MapPrivateMemory(size_t size) {
#if HSHM_ENABLE_PROCFS_SYSINFO
#if __APPLE__ || __OpenBSD__
  return mmap(nullptr, size, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#else
  return mmap64(nullptr, size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return VirtualAlloc(nullptr, size, MEM_COMMIT, PAGE_READWRITE);
#endif
}

void *SystemInfo::MapSharedMemory(const File &fd, size_t size, i64 off) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  void *ptr = mmap64(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd.posix_fd_, off);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    return nullptr;
  }
  return ptr;
#elif HSHM_ENABLE_WINDOWS_SYSINFO
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

void *SystemInfo::MapMixedMemory(const File &fd, size_t private_size,
                                  size_t shared_size, i64 shared_offset) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  // Calculate total size
  size_t total_size = private_size + shared_size;

  // Step 1: Reserve the entire contiguous address space
  // Map the entire region as private/anonymous to reserve virtual addresses
  void *ptr = mmap64(nullptr, total_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    perror("MapMixedMemory: initial mmap failed");
    return nullptr;
  }

  // Step 2: Remap the shared portion using MAP_FIXED
  // This replaces the shared portion with actual shared memory from the fd
  // The private portion remains as private/anonymous
  char *shared_ptr = static_cast<char*>(ptr) + private_size;
  void *result = mmap64(shared_ptr, shared_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, fd.posix_fd_, shared_offset);
  if (result == MAP_FAILED || result != shared_ptr) {
    int saved_errno = errno;
    fprintf(stderr, "MapMixedMemory: MAP_FIXED mmap failed\n");
    fprintf(stderr, "  Requested addr: %p\n", shared_ptr);
    fprintf(stderr, "  Size: %zu bytes\n", shared_size);
    fprintf(stderr, "  fd: %d\n", fd.posix_fd_);
    fprintf(stderr, "  offset: %ld\n", shared_offset);
    fprintf(stderr, "  errno: %d (%s)\n", saved_errno, strerror(saved_errno));
    if (result != MAP_FAILED && result != shared_ptr) {
      fprintf(stderr, "  Got addr: %p (unexpected!)\n", result);
      munmap(result, shared_size);
    }
    // Clean up: unmap the entire region
    munmap(ptr, total_size);
    return nullptr;
  }

  // Success: we now have [private_size bytes private | shared_size bytes shared]
  return ptr;

#elif HSHM_ENABLE_WINDOWS_SYSINFO
  // Windows doesn't easily support MAP_FIXED-like behavior for mixed mappings
  // Fall back to just returning the shared mapping
  // The private region won't work as expected on Windows
  (void)private_size;  // Unused on Windows
  return MapSharedMemory(fd, shared_size, shared_offset);
#endif
}

void SystemInfo::UnmapMemory(void *ptr, size_t size) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  munmap(ptr, size);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  VirtualFree(ptr, size, MEM_RELEASE);
#endif
}

void *SystemInfo::AlignedAlloc(size_t alignment, size_t size) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return aligned_alloc(alignment, size);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return _aligned_malloc(size, alignment);
#endif
}

bool SystemInfo::IsProcessAlive(int pid) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return kill(pid, 0) != -1 || errno != ESRCH;
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                         static_cast<DWORD>(pid));
  if (h == NULL) return false;
  CloseHandle(h);
  return true;
#endif
}

std::string SystemInfo::GetModuleDirectory() {
#if HSHM_ENABLE_PROCFS_SYSINFO
  Dl_info dl_info;
  void *addr = reinterpret_cast<void *>(&SystemInfo::GetModuleDirectory);
  if (dladdr(addr, &dl_info) == 0) return "";
  char resolved[PATH_MAX];
  if (realpath(dl_info.dli_fname, resolved) == nullptr) return "";
  return std::filesystem::path(resolved).parent_path().string();
#elif HSHM_ENABLE_WINDOWS_SYSINFO
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
  return std::filesystem::path(path).parent_path().string();
#endif
}

std::string SystemInfo::GetLibrarySearchPathVar() {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return "LD_LIBRARY_PATH";
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return "PATH";
#endif
}

char SystemInfo::GetPathListSeparator() {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return ':';
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return ';';
#endif
}

std::string SystemInfo::GetSharedLibExtension() {
#if HSHM_ENABLE_WINDOWS_SYSINFO
  return ".dll";
#elif __APPLE__
  return ".dylib";
#else
  return ".so";
#endif
}

std::string SystemInfo::Getenv(const char *name, size_t max_size) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  char *var = getenv(name);
  if (var == nullptr) {
    return "";
  }
  return std::string(var);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
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
#if HSHM_ENABLE_PROCFS_SYSINFO
  setenv(name, value.c_str(), overwrite);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  SetEnvironmentVariable(name, value.c_str());
#endif
}

void SystemInfo::Unsetenv(const char *name) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  unsetenv(name);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  SetEnvironmentVariable(name, nullptr);
#endif
}

SharedLibrary::SharedLibrary(const std::string &name) : handle_(nullptr) {
  Load(name);
}

SharedLibrary::~SharedLibrary() {
  if (handle_) {
#if HSHM_ENABLE_PROCFS_SYSINFO
    dlclose(handle_);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
    ::FreeLibrary((HMODULE)handle_);
#endif
    handle_ = nullptr;
  }
}

void SharedLibrary::Load(const std::string &name) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  handle_ = dlopen(name.c_str(), RTLD_GLOBAL | RTLD_NOW);
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  handle_ = LoadLibraryA(name.c_str());
#endif
}

std::string SharedLibrary::GetError() const {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return std::string(dlerror());
#elif HSHM_ENABLE_WINDOWS_SYSINFO
  return std::string();
#endif
}

void *SharedLibrary::GetSymbol(const std::string &name) {
#if HSHM_ENABLE_PROCFS_SYSINFO
  return dlsym(handle_, name.c_str());
#elif HSHM_ENABLE_WINDOWS_SYSINFO
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

}  // namespace hshm
