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

#ifndef CLIO_RUN_UTIL_CLIO_RUN_LOCAL_RUNTIME_H_
#define CLIO_RUN_UTIL_CLIO_RUN_LOCAL_RUNTIME_H_

/**
 * Shared client-side helpers for the clio_run stop/status subcommands:
 * discover the LOCAL runtime's port, pid and leftover filesystem artifacts
 * without needing a responsive runtime. All lookups are pid/port-scoped so
 * co-resident runtimes (fallback topology) are never misattributed.
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef __APPLE__
// macOS has no /proc: process introspection goes through libproc
// (proc_pidpath) and the KERN_PROC sysctl (process state / zombie check).
#include <libproc.h>
#include <sys/proc.h>
#include <sys/sysctl.h>
#endif

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/config_parse.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/runtime_pid_record.h"
#include "clio_runtime/types.h"

namespace clio::run::cli {

/**
 * Resolve the local runtime's port without requiring a live runtime:
 * ConfigManager if available, else the CLIO_PORT environment variable,
 * else the built-in default (9413).
 * @return the runtime port to target
 */
inline u32 ResolveRuntimePort() {
  auto *config = CLIO_CONFIG_MANAGER;
  if (config) {
    return config->GetPort();
  }
  if (const char *env = clio::run::env::GetCompat("PORT")) {
    int port = std::atoi(env);
    if (port > 0) return static_cast<u32>(port);
  }
  return 9413;
}

/**
 * Path of the runtime's main shared-memory segment entry in the per-user
 * memfd directory. Mirrors ConfigManager::GetSharedMemorySegmentName's
 * default port-keyed recipe so it also works when no config is loaded.
 * @param port the runtime port the segment name is keyed on
 * @return absolute path of the main segment entry
 */
inline std::string MainSegmentPath(u32 port) {
  const std::string name =
      ctp::ConfigParse::ExpandPath("chi_main_segment_${USER}") + "_" +
      std::to_string(port);
  return ctp::SystemInfo::GetMemfdPath(name);
}

/**
 * Discover the local runtime's pid: on Linux from its main segment symlink,
 * whose target is /proc/<pid>/fd/<n> (the memfd the runtime created), and
 * otherwise from the pid record the runtime writes alongside its segments
 * (see clio_runtime/runtime_pid_record.h). POSIX-only;
 * returns -1 on Windows.
 * @param port the runtime port (segment names are port-keyed)
 * @return the owning pid (may be dead), or -1 if the runtime left no trace
 */
inline int DiscoverRuntimePid(u32 port) {
#ifdef _WIN32
  (void)port;
  return -1;
#else
  std::error_code ec;
  auto target = std::filesystem::read_symlink(MainSegmentPath(port), ec);
  if (!ec) {
    const std::string t = target.string();
    constexpr const char *kProc = "/proc/";
    if (t.rfind(kProc, 0) == 0) {
      int pid = std::atoi(t.c_str() + std::string(kProc).size());
      if (pid > 0) {
        return pid;
      }
    }
  }
  return ReadRuntimePidRecord(port);
#endif
}

/**
 * Check whether a pid is a zombie (exited but not yet reaped by its parent).
 * kill(pid, 0) treats zombies as alive, but a zombie cannot run, service
 * tasks, or hold sockets — for stop purposes it is dead. The stop CLI is a
 * sibling of the runtime (the parent is Jarvis / a test harness / a shell),
 * so it can never reap the zombie itself.
 *
 * This matters most on the kill-escalation path: after SIGKILL the runtime
 * lingers as a zombie until its parent (the harness / Jarvis) reaps it, and
 * a stop that cannot see through that reports "survived SIGKILL".
 * @param pid candidate pid
 * @return true if the process is a zombie
 */
inline bool PidIsZombie(int pid) {
#if defined(__APPLE__)
  // No /proc: ask the kernel for the process's state directly.
  struct kinfo_proc info;
  size_t len = sizeof(info);
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  if (sysctl(mib, 4, &info, &len, nullptr, 0) != 0 || len == 0) {
    return false;
  }
  return info.kp_proc.p_stat == SZOMB;
#elif defined(__linux__)
  std::ifstream stat_file("/proc/" + std::to_string(pid) + "/stat");
  if (!stat_file.is_open()) {
    return false;
  }
  std::string stat_line;
  std::getline(stat_file, stat_line);
  // Field 3 (state) follows the parenthesized comm, which may contain spaces.
  size_t close_paren = stat_line.rfind(')');
  if (close_paren == std::string::npos ||
      close_paren + 2 >= stat_line.size()) {
    return false;
  }
  return stat_line[close_paren + 2] == 'Z';
#else
  (void)pid;
  return false;
#endif
}

/**
 * Check whether a pid is alive AND runnable (not a zombie). This is the
 * liveness test the stop flow uses: a zombie counts as stopped.
 * @param pid candidate pid
 * @return true if the process exists and is not a zombie
 */
inline bool PidIsRunning(int pid) {
  return ctp::SystemInfo::IsProcessAlive(pid) && !PidIsZombie(pid);
}

/**
 * Check that a pid currently belongs to a clio_run process (guards kill
 * escalation against pid recycling). POSIX-only; false on Windows.
 * @param pid candidate runtime pid
 * @return true if the process's executable is a clio_run binary
 */
inline bool PidIsClioRun(int pid) {
#if defined(_WIN32)
  (void)pid;
  return false;
#elif defined(__APPLE__)
  // No /proc/<pid>/exe: libproc reports the executable path, and does so for
  // same-user processes without any special entitlement.
  char exe_path[PROC_PIDPATHINFO_MAXSIZE];
  if (proc_pidpath(pid, exe_path, sizeof(exe_path)) <= 0) {
    return false;
  }
  return std::filesystem::path(exe_path).filename().string().rfind(
             "clio_run", 0) == 0;
#else
  std::error_code ec;
  auto exe = std::filesystem::read_symlink(
      "/proc/" + std::to_string(pid) + "/exe", ec);
  if (ec) {
    return false;
  }
  return exe.filename().string().rfind("clio_run", 0) == 0;
#endif
}

/**
 * List this port's leftover runtime artifacts: port-keyed chi_* segment
 * entries and control-socket files in the per-user memfd dir, memfd-dir
 * entries whose /proc symlink owner is dead, and legacy /dev/shm/chi_*
 * files (older builds used shm_open naming; still seen on HPC deployments).
 * @param port the runtime port to scope the scan to
 * @return absolute paths of stale artifacts found
 */
inline std::vector<std::string> ListStaleArtifacts(u32 port) {
  std::vector<std::string> stale;
  const std::string memfd_dir = ctp::SystemInfo::GetMemfdDir();
  const std::string port_suffix = "_" + std::to_string(port);
  const std::string ipc_name = "clio_" + std::to_string(port) + ".ipc";
  const std::string zmq_ipc_name =
      "clio_zmq_" + std::to_string(port + 3) + ".ipc";

  for (const auto &name : ctp::SystemInfo::ListDirectory(memfd_dir)) {
    const std::string full_path = memfd_dir + "/" + name;
    bool matches = (name == ipc_name) || (name == zmq_ipc_name);
    // Port-keyed segment names: chi_<segment>_<user>_<port>
    if (!matches && name.rfind("chi_", 0) == 0 &&
        name.size() >= port_suffix.size() &&
        name.compare(name.size() - port_suffix.size(), port_suffix.size(),
                     port_suffix) == 0) {
      matches = true;
    }
    if (!matches) {
      // Entries owned by a dead process (on-demand segments etc.)
      std::error_code ec;
      auto target = std::filesystem::read_symlink(full_path, ec);
      if (!ec) {
        const std::string t = target.string();
        constexpr const char *kProc = "/proc/";
        if (t.rfind(kProc, 0) == 0) {
          int owner = std::atoi(t.c_str() + std::string(kProc).size());
          matches = owner > 0 && !ctp::SystemInfo::IsProcessAlive(owner);
        }
      }
    }
    if (matches) {
      stale.push_back(full_path);
    }
  }

#ifndef _WIN32
  // Legacy shm_open naming from older builds.
  std::error_code ec;
  for (auto it = std::filesystem::directory_iterator("/dev/shm", ec);
       !ec && it != std::filesystem::directory_iterator(); ++it) {
    const std::string name = it->path().filename().string();
    if (name.rfind("chi_", 0) == 0) {
      stale.push_back(it->path().string());
    }
  }
#endif
  return stale;
}

}  // namespace clio::run::cli

#endif  // CLIO_RUN_UTIL_CLIO_RUN_LOCAL_RUNTIME_H_
