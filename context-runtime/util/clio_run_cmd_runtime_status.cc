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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/pool_query.h"
#include "clio_runtime/types.h"
#include "clio_run_commands.h"
#include "clio_run_local_runtime.h"

namespace {

void PrintRuntimeStatusUsage() {
  HIPRINT("Usage: clio_run status");
  HIPRINT("  Report the local Clio runtime's state.");
  HIPRINT("  Exit codes:");
  HIPRINT("    0  RUNNING - runtime answered a heartbeat probe");
  HIPRINT("    1  STOPPED (clean) - no runtime, no stale artifacts");
  HIPRINT("    2  STOPPED (stale artifacts) - no responsive runtime, but");
  HIPRINT("       leftover segments/sockets exist (listed on stdout)");
}

/**
 * Probe the local runtime with a heartbeat task. Assumes CLIO_INIT(kClient)
 * succeeded.
 * @return true if the runtime acked the heartbeat within 2 seconds
 */
bool ProbeHeartbeat() {
  try {
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);
    auto task = admin_client.AsyncHeartbeat(clio::run::PoolQuery::Local());
    if (task.IsNull()) {
      return false;
    }
    return task.Wait(2.0f);
  } catch (const std::exception&) {
    return false;
  }
}

/**
 * Report the not-running outcome: scan for stale artifacts and pick the
 * exit code (1 = clean, 2 = stale artifacts present).
 * @param port the runtime port that was probed
 * @return process exit code
 */
int ReportStopped(clio::run::u32 port) {
  const auto stale = clio::run::cli::ListStaleArtifacts(port);
  const int pid = clio::run::cli::DiscoverRuntimePid(port);
  if (pid > 0 && ctp::SystemInfo::IsProcessAlive(pid) &&
      clio::run::cli::PidIsClioRun(pid)) {
    HIPRINT("clio runtime (port {}): UNRESPONSIVE - process {} is alive but "
            "did not answer the heartbeat probe", port, pid);
  }
  if (stale.empty()) {
    HIPRINT("clio runtime (port {}): STOPPED (clean)", port);
    return 1;
  }
  HIPRINT("clio runtime (port {}): STOPPED (stale artifacts)", port);
  for (const auto& path : stale) {
    HIPRINT("  stale: {}", path);
  }
  return 2;
}

}  // namespace

int RuntimeStatus(int argc, char* argv[]) {
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 ||
        std::strcmp(argv[i], "-h") == 0) {
      PrintRuntimeStatusUsage();
      return 0;
    }
  }

  // Self-cleanup on EVERY exit path: even a failed ClientInit creates this
  // process's per-thread MPSC segment (clio-<pid>-<tid>) in the memfd dir,
  // and a status probe must not leave litter of its own. Idempotent with
  // the ClientFinalize below.
  struct OwnEntriesGuard {
    ~OwnEntriesGuard() {
      auto* ipc = CLIO_IPC;
      if (ipc) {
        ipc->UnlinkOwnPidEntries();
      }
    }
  } own_entries_guard;

  // Bound the client-connect wait: status must return promptly whether the
  // runtime is up, down, or wedged. An explicit CLIO_WAIT_SERVER wins.
  if (!clio::run::env::GetCompat("WAIT_SERVER")) {
#ifdef _WIN32
    _putenv_s("CLIO_WAIT_SERVER", "2");
#else
    setenv("CLIO_WAIT_SERVER", "2", 1);
#endif
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    return ReportStopped(clio::run::cli::ResolveRuntimePort());
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

  const clio::run::u32 port = clio::run::cli::ResolveRuntimePort();
  if (ProbeHeartbeat()) {
    const int pid = clio::run::cli::DiscoverRuntimePid(port);
    if (pid > 0) {
      HIPRINT("clio runtime (port {}): RUNNING (pid {})", port, pid);
    } else {
      HIPRINT("clio runtime (port {}): RUNNING", port);
    }
    return 0;
  }
  return ReportStopped(port);
}
