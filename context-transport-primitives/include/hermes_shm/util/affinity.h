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

#ifndef HSHM_SHM_AFFINITY_H
#define HSHM_SHM_AFFINITY_H

#ifndef _WIN32

// Reference:
// https://stackoverflow.com/questions/63372288/getting-list-of-pids-from-proc-in-linux

#include <dirent.h>
#include <sched.h>
#include <sys/sysinfo.h>

#include <algorithm>
#include <string>
#include <vector>

namespace hshm {

class ProcessAffiner {
 public:
  /** Set the CPU affinity of a process */
  static void SetCpuAffinity(int pid, int cpu_id) {
    // Create a CPU set and set CPU affinity to CPU 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    // Set the CPU affinity of the process
    int result = sched_setaffinity(pid, sizeof(cpuset), &cpuset);
    if (result == -1) {
      // HLOG(kError, "Failed to set CPU affinity for process {}", pid);
    }
  }

 private:
  int n_cpu_;
  std::vector<cpu_set_t> cpus_;
  std::vector<int> ignore_pids_;

 public:
  ProcessAffiner() {
    n_cpu_ = get_nprocs_conf();
    cpus_.resize(n_cpu_);
    Clear();
  }

  inline bool isdigit(char digit) { return ('0' <= digit && digit <= '9'); }

  inline int GetNumCPU() { return n_cpu_; }

  inline void SetCpu(int cpu) { CPU_SET(cpu, cpus_.data()); }

  inline void SetCpus(int off, int len) {
    for (int i = 0; i < len; ++i) {
      SetCpu(off + i);
    }
  }

  inline void SetCpus(const std::vector<int> &cpu_ids) {
    for (int cpu_id : cpu_ids) {
      SetCpu(cpu_id);
    }
  }

  inline void ClearCpu(int cpu) { CPU_CLR(cpu, cpus_.data()); }

  void IgnorePids(const std::vector<int> &pids) { ignore_pids_ = pids; }

  inline void ClearCpus(int off, int len) {
    for (int i = 0; i < len; ++i) {
      ClearCpu(off + i);
    }
  }

  inline void Clear() {
    for (cpu_set_t &cpu : cpus_) {
      CPU_ZERO(&cpu);
    }
  }

  int AffineAll(void) {
    DIR *procdir;
    struct dirent *entry;
    size_t count = 0;

    // Open /proc directory.
    procdir = opendir("/proc");
    if (!procdir) {
      perror("opendir failed");
      return 0;
    }

    // Iterate through all files and folders of /proc.
    while ((entry = readdir(procdir))) {
      // Skip anything that is not a PID folder.
      if (!is_pid_folder(entry)) {
        continue;
      }
      // Get the PID of the running process
      int proc_pid = atoi(entry->d_name);
      if (std::find(ignore_pids_.begin(), ignore_pids_.end(), proc_pid) !=
          ignore_pids_.end()) {
        continue;
      }
      // Set the affinity of all running process to this mask
      count += Affine(proc_pid);
    }
    closedir(procdir);
    return count;
  }
  int Affine(std::vector<pid_t> &&pids) { return Affine(pids); }
  int Affine(std::vector<pid_t> &pids) {
    // Set the affinity of all running process to this mask
    size_t count = 0;
    for (pid_t &pid : pids) {
      count += Affine(pid);
    }
    return count;
  }
  int Affine(int pid) { return SetAffinitySafe(pid, n_cpu_, cpus_.data()); }

  void PrintAffinity(int pid) { PrintAffinity("", pid); }
  void PrintAffinity(std::string prefix, int pid) {
    std::vector<cpu_set_t> cpus(n_cpu_);
    sched_getaffinity(pid, n_cpu_, cpus.data());
    PrintAffinity(prefix, pid, cpus.data());
  }

  void PrintAffinity(std::string prefix, int pid, cpu_set_t *cpus) {
    std::string affinity = "";
    for (int i = 0; i < n_cpu_; ++i) {
      if (CPU_ISSET(i, cpus)) {
        affinity += std::to_string(i) + ", ";
      }
    }
    printf("%s: CPU affinity[pid=%d]: %s\n", prefix.c_str(), pid,
           affinity.c_str());
  }

 private:
  int SetAffinitySafe(int pid, int n_cpu, cpu_set_t *cpus) {
    int ret = sched_setaffinity(pid, n_cpu, cpus);
    if (ret == -1) {
      return 0;
    }
    return 1;
  }

  // Helper function to check if a struct dirent from /proc is a PID folder.
  int is_pid_folder(const struct dirent *entry) {
    const char *p;
    for (p = entry->d_name; *p; p++) {
      if (!isdigit(*p)) return false;
    }
    return true;
  }
};

}  // namespace hshm

#endif  // _WIN32

#endif  // HSHM_SHM_AFFINITY_H
