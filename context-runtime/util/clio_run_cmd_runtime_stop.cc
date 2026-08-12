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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#ifndef _WIN32
#include <csignal>
#include <unistd.h>
#endif

#include "clio_ctp/util/config_parse.h"
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/pool_query.h"
#include "clio_runtime/types.h"
#include "clio_run_commands.h"
#include "clio_run_local_runtime.h"

namespace {

using clio::run::cli::DiscoverRuntimePid;
using clio::run::cli::PidIsClioRun;
using clio::run::cli::PidIsRunning;
using clio::run::cli::ResolveRuntimePort;

/** Parsed clio_run stop options */
struct StopOptions {
  bool force = false;               /**< --force: immediate ungraceful death */
  clio::run::u32 grace_period_ms = 5000; /**< --grace-period: drain budget */
};

/**
 * Parse the stop subcommand flags (--force, --grace-period <ms>).
 * @param argc argument count for the subcommand
 * @param argv argument vector for the subcommand
 * @return parsed options (defaults: graceful, 5000 ms grace)
 */
StopOptions ParseStopArgs(int argc, char* argv[]) {
  StopOptions opts;
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--force") == 0) {
      opts.force = true;
    } else if (std::strcmp(argv[i], "--grace-period") == 0 && i + 1 < argc) {
      opts.grace_period_ms =
          static_cast<clio::run::u32>(std::atoi(argv[++i]));
      if (opts.grace_period_ms == 0) opts.grace_period_ms = 5000;
    }
  }
  return opts;
}

#ifndef _WIN32

/**
 * Remove leftover filesystem artifacts after the local runtime is down.
 * Removes: memfd-dir entries whose symlink target points at the (dead)
 * runtime pid, the port-keyed chi_* segment entries, the port's
 * control-socket files, and any entry owned by a DEAD process (the same
 * dead-only rule ClearUserIpcs applies at the next runtime start — e.g.
 * segments of crashed clients). Entries owned by live processes
 * (co-resident runtimes, active clients) are never touched.
 * @param pid the dead runtime's pid (<= 0 to skip pid-target matching)
 * @param port the dead runtime's port
 * @return number of filesystem entries removed
 */
size_t SweepDeadRuntimeArtifacts(int pid, clio::run::u32 port) {
  size_t removed = 0;
  const std::string memfd_dir = ctp::SystemInfo::GetMemfdDir();
  const std::string proc_prefix =
      pid > 0 ? "/proc/" + std::to_string(pid) + "/" : "";
  const std::string port_suffix = "_" + std::to_string(port);
  const std::string ipc_name = "clio_" + std::to_string(port) + ".ipc";
  const std::string zmq_ipc_name =
      "clio_zmq_" + std::to_string(port + 3) + ".ipc";
  const int own_pid = ctp::SystemInfo::GetPid();

  for (const auto& name : ctp::SystemInfo::ListDirectory(memfd_dir)) {
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
      std::error_code ec;
      auto target = std::filesystem::read_symlink(full_path, ec);
      if (!ec) {
        const std::string t = target.string();
        if (!proc_prefix.empty() && t.rfind(proc_prefix, 0) == 0) {
          matches = true;  // owned by the dead runtime
        } else if (t.rfind("/proc/", 0) == 0) {
          // Owned by some other DEAD process (crashed client etc.)
          int owner = std::atoi(t.c_str() + 6);
          matches = owner > 0 && owner != own_pid &&
                    !ctp::SystemInfo::IsProcessAlive(owner);
        }
      } else if (pid > 0) {
        // No symlink to read: on platforms without memfd (macOS/BSD) the
        // entries are plain files, so fall back to the pid-keyed names the
        // runtime gives its on-demand and per-thread segments — the same
        // fallback IpcManager::UnlinkOwnPidEntries uses on its own entries.
        for (const auto& prefix : {"clio_" + std::to_string(pid) + "_",
                                   "clio-" + std::to_string(pid) + "-"}) {
          if (name.rfind(prefix, 0) == 0) {
            matches = true;
            break;
          }
        }
      }
    }
    if (matches && ctp::SystemInfo::RemoveFile(full_path)) {
      HLOG(kInfo, "stop: swept stale artifact {}", name);
      removed++;
    }
  }
  return removed;
}

/**
 * Last-resort local stop: the runtime did not respond to (or did not die
 * from) the StopRuntimeTask. Discover its pid from the main segment symlink
 * and escalate SIGTERM -> SIGKILL, then sweep its leftover artifacts. Also
 * handles the "runtime already dead but artifacts remain" case (crash /
 * pkill -9) by sweeping without killing.
 * @param port the runtime port to target
 * @return 0 if the runtime is gone and artifacts are swept; 1 on failure
 */
int EscalateLocalKill(clio::run::u32 port) {
  int pid = DiscoverRuntimePid(port);
  if (pid <= 0) {
    HLOG(kWarning,
         "stop: no runtime segment found for port {} - sweeping any "
         "port-scoped leftovers",
         port);
    SweepDeadRuntimeArtifacts(-1, port);
    return 0;
  }

  if (PidIsRunning(pid) && PidIsClioRun(pid)) {
    HLOG(kWarning,
         "stop: runtime (pid {}) unresponsive - escalating to SIGTERM", pid);
    kill(pid, SIGTERM);
    for (int i = 0; i < 100 && PidIsRunning(pid); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (PidIsRunning(pid)) {
      HLOG(kWarning, "stop: SIGTERM ineffective - escalating to SIGKILL");
      kill(pid, SIGKILL);
      for (int i = 0; i < 50 && PidIsRunning(pid); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    if (PidIsRunning(pid)) {
      HLOG(kError, "stop: runtime pid {} survived SIGKILL", pid);
      return 1;
    }
  } else if (PidIsRunning(pid)) {
    HLOG(kWarning,
         "stop: pid {} from stale segment is not clio_run (pid recycled) - "
         "treating runtime as dead",
         pid);
  } else {
    HLOG(kWarning, "stop: runtime (pid {}) is already dead", pid);
  }

  size_t swept = SweepDeadRuntimeArtifacts(pid, port);
  HLOG(kWarning,
       "stop: escalation complete - runtime is down, {} stale artifacts "
       "swept",
       swept);
  return 0;
}

#else  // _WIN32

/** Windows stub: pid discovery via /proc symlinks is POSIX-only. */
int EscalateLocalKill(clio::run::u32 /*port*/) {
  HLOG(kError,
       "stop: runtime unresponsive and kill escalation is not supported on "
       "Windows");
  return 1;
}

#endif  // _WIN32

/**
 * Wait for the runtime process to exit (or become a zombie awaiting its
 * parent's reap, which counts as stopped). Pid-based liveness is used
 * instead of ClientConnect probes: probe responses proved unreliable during
 * teardown (futures can complete without a live server), while process
 * death is unambiguous.
 * @param pid the runtime's pid
 * @param timeout_ms how long to poll before declaring failure
 * @return true if the process stopped within the timeout
 */
bool WaitForRuntimeExit(int pid, clio::run::u64 timeout_ms) {
#ifdef _WIN32
  (void)pid;
  (void)timeout_ms;
  return false;
#else
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!PidIsRunning(pid)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return !PidIsRunning(pid);
#endif
}

/**
 * Deliver the StopRuntimeTask to the local runtime and wait for the process
 * to die (pid-based). Assumes the client is initialized.
 * @param opts parsed stop options (force + grace period)
 * @param pid the runtime's pid (from DiscoverRuntimePid; <= 0 if unknown)
 * @return 0 if the runtime stopped; 1 if it must be escalated
 */
int SendStopTask(const StopOptions& opts, int pid) {
  clio::run::admin::Client admin_client(clio::run::kAdminPoolId);

  auto* ipc_manager = CLIO_IPC;
  if (!ipc_manager || !ipc_manager->IsInitialized()) {
    HLOG(kError, "IPC manager not available - is Clio runtime running?");
    return 1;
  }

  clio::run::u32 shutdown_flags = 0;
  if (opts.force) {
    shutdown_flags |= clio::run::admin::StopRuntimeTask::kForceShutdown;
  }

  HLOG(kDebug, "Sending stop runtime task (grace period: {}ms, force: {})...",
       opts.grace_period_ms, opts.force);
  const auto start_time = std::chrono::steady_clock::now();

  clio::run::Future<clio::run::admin::StopRuntimeTask> stop_task;
  try {
    stop_task = admin_client.AsyncStopRuntime(
        clio::run::PoolQuery::Local(), shutdown_flags, opts.grace_period_ms);
    if (stop_task.IsNull()) {
      HLOG(kError, "Failed to create stop runtime task");
      return 1;
    }
  } catch (const std::exception& e) {
    HLOG(kError, "Error creating stop runtime task: {}", e.what());
    return 1;
  }

  // Harvest the ack for logging only. A missing response is NOT an error: a
  // forced runtime _exit()s ~500 ms after handling the task and may vanish
  // before the reply flushes. The authoritative check is the liveness poll
  // below.
  if (stop_task.Wait(2.0f)) {
    HLOG(kDebug, "Stop task acknowledged (return code: {})",
         stop_task->GetReturnCode());
  } else {
    HLOG(kDebug, "No stop-task ack (runtime may have exited first)");
  }

  // Wait for the runtime process to actually die. The graceful budget must
  // exceed the runtime-side watchdog deadline (grace + 15 s margin), so even
  // a wedged teardown resolves within this window via the watchdog's
  // _exit(2). Force mode only needs the ~500 ms ack-flush delay plus slack.
  const clio::run::u64 wait_ms =
      opts.force ? 10000
                 : static_cast<clio::run::u64>(opts.grace_period_ms) + 25000;
  if (pid > 0) {
    if (!WaitForRuntimeExit(pid, wait_ms)) {
      HLOG(kError, "Runtime (pid {}) did not stop within {} ms", pid, wait_ms);
      return 1;
    }
  } else {
    // Pid discovery failed (no main-segment symlink): fall back to the
    // legacy probe-based wait, then let the caller escalate if needed.
    if (!ipc_manager->WaitForLocalRuntimeStop(
            static_cast<clio::run::u32>(wait_ms / 1000))) {
      HLOG(kError, "Runtime did not stop within {} ms", wait_ms);
      return 1;
    }
  }

  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_time).count();
  HLOG(kDebug, "Runtime stopped in {}ms", duration);
  return 0;
}

}  // namespace

int RuntimeStop(int argc, char* argv[]) {
  HLOG(kDebug, "Stopping Clio runtime...");
  const StopOptions opts = ParseStopArgs(argc, argv);

  // Self-cleanup on EVERY exit path: even a failed ClientInit creates this
  // process's per-thread MPSC segment (clio-<pid>-<tid>) in the memfd dir,
  // and a stop tool must not leave litter of its own. Idempotent with the
  // ClientFinalize below.
  struct OwnEntriesGuard {
    ~OwnEntriesGuard() {
      auto* ipc = CLIO_IPC;
      if (ipc) {
        ipc->UnlinkOwnPidEntries();
      }
    }
  } own_entries_guard;

  // Bound the client-connect wait so stop escalates quickly against a dead
  // or wedged runtime instead of burning the default 30 s. An explicit user
  // CLIO_WAIT_SERVER wins.
  if (!clio::run::env::GetCompat("WAIT_SERVER")) {
#ifdef _WIN32
    _putenv_s("CLIO_WAIT_SERVER", "5");
#else
    setenv("CLIO_WAIT_SERVER", "5", 1);
#endif
  }

  try {
    HLOG(kDebug, "Initializing Clio client...");
    if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
      HLOG(kWarning,
           "No responsive runtime to deliver the stop task to - escalating");
      return EscalateLocalKill(ResolveRuntimePort());
    }

    // RAII guard: call ClientFinalize() on every return path so the background
    // ZMQ receive thread is joined and the DEALER socket is closed before the
    // ZMQ shared-context static destructor runs.
    struct ClientFinalizeGuard {
      ~ClientFinalizeGuard() {
        auto* mgr = CLIO_RUNTIME_MANAGER;
        if (mgr) {
          mgr->ClientFinalize();
        }
      }
    } finalize_guard;

    const clio::run::u32 port = ResolveRuntimePort();
    const int pid = DiscoverRuntimePid(port);
    if (SendStopTask(opts, pid) == 0) {
#ifndef _WIN32
      // Idempotent residue sweep: a graceful exit unlinks everything itself,
      // but a watchdog-forced exit can race its own unlink pass. Cheap no-op
      // in the common case.
      SweepDeadRuntimeArtifacts(pid, port);
#endif
      return 0;
    }

    // The task path failed (undeliverable task or the runtime outlived the
    // wait): guarantee convergence with a direct kill + artifact sweep.
    return EscalateLocalKill(ResolveRuntimePort());

  } catch (const std::exception& e) {
    HLOG(kError, "Error stopping runtime: {} - escalating", e.what());
    return EscalateLocalKill(ResolveRuntimePort());
  }
}
