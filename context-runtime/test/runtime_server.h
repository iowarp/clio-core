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

#ifndef CLIO_RUNTIME_TEST_RUNTIME_SERVER_H_
#define CLIO_RUNTIME_TEST_RUNTIME_SERVER_H_

/**
 * RuntimeServer — cross-platform test helper that launches the canonical
 * `clio_run` runtime daemon as a SEPARATE process and manages its lifetime.
 *
 * This replaces the older `fork()` + in-process `CLIO_INIT(kServer)` +
 * `sleep(300)` pattern used by the IPC/transport/external-client tests. That
 * pattern is:
 *   - broken on macOS: the forked child dlopen()s the ChiMod .dylibs and spawns
 *     worker threads *after* fork without exec(); on macOS this deadlocks/fails
 *     in dyld (fork-without-exec is unsupported for a process this complex).
 *     The child creates the main shm segment (so a file-based readiness probe
 *     passes) but never becomes responsive, so the client times out and the
 *     leaked `sleep(300)` child keeps holding the runtime's TCP port.
 *   - impossible on Windows: there is no `fork()`.
 *
 * The process spawn/wait/kill live in ctp::SystemInfo (SpawnProcess /
 * IsChildRunning / TerminateChild), so this header pulls in NO OS headers — no
 * <windows.h> to clash with the ctp lightbeam <winsock2.h> a client TU also
 * needs (that clash is why RuntimeServer tests were POSIX-only, issue #476).
 * Tests that do not otherwise use fork() can therefore build on Windows too.
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/config_parse.h"

// Downstream POSIX tests that include this header (test_clio_run_cli,
// test_cte_fallback, ...) call kill()/waitpid()/open() directly and have long
// relied on it to pull the declarations. Keep providing them on POSIX: only
// <windows.h> was the #476 clash with the ctp lightbeam <winsock2.h> — these
// POSIX headers do not conflict, and the block is skipped on Windows. (The
// process spawn/wait/kill this header itself needs now live in ctp::SystemInfo.)
#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// Defined in clio::run::test; tests reach it as clio::run::test::RuntimeServer.
namespace clio {
namespace run {
namespace test {

/** Portable setenv() (routed through SystemInfo so no OS headers leak in). */
inline void SetEnvVar(const char *key, const std::string &val) {
  ctp::SystemInfo::Setenv(key, val, /*overwrite=*/1);
}

/** Portable unsetenv(). */
inline void UnsetEnvVar(const char *key) { ctp::SystemInfo::Unsetenv(key); }

class RuntimeServer {
 public:
  RuntimeServer() = default;
  ~RuntimeServer() { Stop(); }
  RuntimeServer(const RuntimeServer &) = delete;
  RuntimeServer &operator=(const RuntimeServer &) = delete;

  /**
   * Spawn `clio_run start` as a child process. CLIO_PORT / CLIO_BIND_ADDR are
   * exported first so both the daemon and this (client) process agree on where
   * the runtime lives. Returns true if the process was spawned (use
   * WaitForReady() to confirm it actually came up).
   */
  /**
   * @param detached  Spawn the daemon with NO controlling console/terminal
   *   (Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP; POSIX:
   *   POSIX_SPAWN_SETSID). This is the console-less spawn from issue #721, where
   *   the ZeroMQ transport failed to initialize Winsock ("WSASTARTUP not yet
   *   performed") and the daemon stayed alive but unreachable. A serviceable
   *   daemon after a detached spawn proves the transport initializes regardless
   *   of console.
   */
  bool Start(unsigned port = 10500,
             const std::string &bind_addr = "127.0.0.1",
             bool ephemeral = false,
             bool detached = false) {
    port_ = port;
    SetEnv("CLIO_PORT", std::to_string(port));
    SetEnv("CLIO_BIND_ADDR", bind_addr);
    const std::string exe = RuntimeExe();
    // Point the daemon at the directory holding clio_run for ChiMod (.so/.dylib
    // /.dll) discovery — the modules are built alongside it. Not every test
    // sets CLIO_REPO_PATH in its CTest ENVIRONMENT, so set it unconditionally.
    {
      size_t slash = exe.find_last_of("/\\");
      if (slash != std::string::npos) SetEnv("CLIO_REPO_PATH", exe.substr(0, slash));
    }
    const std::string log = ServerLogPath();

    // Redirect the daemon's stdout/stderr to the log so its worker chatter does
    // not flood the test output (and is inspectable on failure). When `detached`,
    // spawn console-less to reproduce issue #721.
    std::vector<std::string> args;
    args.push_back("start");
    if (ephemeral) args.push_back("--ephemeral");
    proc_ = ctp::SystemInfo::SpawnProcess(exe, args, log, detached);
    if (!proc_.valid) return false;
    started_ = true;
    return true;
  }

  /**
   * Poll until the daemon is genuinely able to serve clients. Two stages:
   *
   *   1. the runtime's main shared-memory segment exists — portable: on POSIX a
   *      file under /tmp/clio_$USER, on Windows a named mapping, both behind
   *      SystemInfo::OpenSharedMemory;
   *   2. its request ROUTER is BOUND, read from the daemon's captured log.
   *
   * Stage 2 exists because stage 1 fires early in daemon init, well before the
   * transport comes up. This used to be papered over with a flat 1 s sleep,
   * which is a guess, not a signal: on a loaded Windows Debug runner the ROUTER
   * can still be unbound when it expires, and the daemon can also die inside it
   * (issue #848) with the old code returning true for a dead process. Either
   * way the caller was handed a daemon it could not reach, and callers that
   * retry on a fresh port (see test_daemon_detached_spawn) never got the chance
   * — they had already been told the daemon was ready. That is the
   * cr_detached_spawn flake: the client's DEALER dials port+3 and times out.
   *
   * Returns false if the timeout elapses or the daemon exits early, which is
   * what lets those callers retry instead of failing outright.
   */
  bool WaitForReady(int timeout_ms = 30000) {
    // Segment names are port-keyed (see ConfigManager::GetSharedMemorySegmentName)
    // so they match the daemon started on port_.
    const std::string seg =
        ctp::ConfigParse::ExpandPath("chi_main_segment_${USER}") + "_" +
        std::to_string(port_);
    const int attempts = timeout_ms / 200;
    int i = 0;
    bool segment_up = false;
    for (; i < attempts && !segment_up; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (!IsRunning()) return false;  // daemon died during startup
      ctp::File fd;
      if (ctp::SystemInfo::OpenSharedMemory(fd, seg)) {
        ctp::SystemInfo::CloseSharedMemory(fd);
        segment_up = true;
      }
    }
    if (!segment_up) return false;

    // Spend whatever budget is left waiting for the daemon to log that its
    // admin pool exists. The daemon logs it at INFO, so a caller that wants
    // this stage must leave CTP_LOG_LEVEL at info or lower; if the line never
    // shows up we fall through to the settle sleep rather than failing a
    // daemon that is fine.
    //
    // The marker is the ADMIN POOL, not the ROUTER bind. Ordering the daemon's
    // own startup log, the ROUTER binds second of twenty-one markers -- before
    // the chimods load, before the workers spawn, and before the admin pool is
    // created. Treating it as "ready" would hand callers a daemon that cannot
    // yet route a task, which is strictly worse than the sleep it replaced.
    // Waiting for the admin pool means the runtime can actually serve work.
    const std::string log_path = ServerLogPath();
    bool marker_seen = false;
    for (; i < attempts && !marker_seen; ++i) {
      if (!IsRunning()) return false;
      if (ServerLogHasAdminPool(log_path)) {
        marker_seen = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Keep the historical settle in BOTH cases -- marker seen or budget spent.
    // The marker is necessary but not provably sufficient: startup work that
    // logs nothing (e.g. compose/WAL replay re-creating restartable pools) can
    // still be in flight. Retaining the settle makes this readiness check
    // strictly stronger than the sleep-only version it replaces, never weaker.
    // Do not call a daemon that died inside the settle ready.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return IsRunning();
  }

  /**
   * @return true if the daemon's captured log shows the admin pool created --
   * i.e. the runtime can route a task, not merely accept a connection.
   */
  static bool ServerLogHasAdminPool(const std::string &log_path) {
    std::ifstream f(log_path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str().find("Admin chimod pool created successfully") !=
           std::string::npos;
  }

  /** Stop the daemon (SIGTERM then SIGKILL on POSIX — so ServerFinalize's leak
   *  report runs; TerminateProcess on Windows) and reap it. Idempotent; called
   *  by the destructor. */
  void Stop() {
    if (!started_) return;
    started_ = false;
    ctp::SystemInfo::TerminateChild(proc_);
  }

  /** True while the daemon process is still alive. */
  bool IsRunning() {
    if (!started_) return false;
    return ctp::SystemInfo::IsChildRunning(proc_);
  }

#ifndef _WIN32
  /** The spawned daemon's pid (-1 if not started or already reaped). */
  pid_t Pid() const { return proc_.valid ? proc_.pid : -1; }

  /**
   * Wait for the daemon to exit on its own (e.g. after an external
   * `clio_run stop`) and capture its exit code. Reaps the process, so the
   * destructor will not try to kill it again.
   * @param timeout_ms how long to poll before giving up
   * @param exit_code out: WEXITSTATUS if the daemon exited normally, or
   *                  128+signal if it was terminated by a signal
   * @return true if the daemon exited within the timeout
   */
  bool WaitExit(int timeout_ms, int *exit_code) {
    if (!proc_.valid || proc_.pid <= 0) return false;
    const int attempts = timeout_ms / 100;
    for (int i = 0; i <= attempts; ++i) {
      int status = 0;
      pid_t ret = waitpid(proc_.pid, &status, WNOHANG);
      if (ret == proc_.pid) {
        if (exit_code != nullptr) {
          if (WIFEXITED(status)) {
            *exit_code = WEXITSTATUS(status);
          } else if (WIFSIGNALED(status)) {
            *exit_code = 128 + WTERMSIG(status);
          } else {
            *exit_code = -1;
          }
        }
        proc_.pid = -1;
        proc_.valid = false;
        started_ = false;
        return true;
      }
      if (ret < 0) {  // already reaped elsewhere
        proc_.pid = -1;
        proc_.valid = false;
        started_ = false;
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
  }

  /**
   * Abandon ownership of the daemon: the destructor will neither kill nor
   * reap it. Use after the test dispatched the process by other means (e.g.
   * an external tool killed and reaped it is impossible — reaping is ours —
   * so pair with a final waitpid by the caller).
   */
  void Disown() {
    started_ = false;
    proc_.pid = -1;
    proc_.valid = false;
  }
#endif

 private:
  /** Absolute path to the clio_run binary. CMake passes CLIO_RUN_EXE via
   *  $<TARGET_FILE:clio_run>; fall back to CLIO_REPO_PATH/clio_run otherwise. */
  static std::string RuntimeExe() {
#ifdef CLIO_RUN_EXE
    return std::string(CLIO_RUN_EXE);
#else
    const char *repo = std::getenv("CLIO_REPO_PATH");
    std::string dir = (repo && *repo) ? std::string(repo) : std::string(".");
#ifdef _WIN32
    return dir + "\\clio_run.exe";
#else
    return dir + "/clio_run";
#endif
#endif
  }

  static std::string ServerLogPath() {
    const char *override_path = std::getenv("CLIO_TEST_SERVER_LOG");
    if (override_path && *override_path) return override_path;
    // Portable temp dir (no <windows.h> GetTempPath needed).
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = ".";
    return (dir / "clio_run_test_server.log").string();
  }

  static void SetEnv(const char *key, const std::string &val) {
    SetEnvVar(key, val);
  }

  bool started_ = false;
  // Port the daemon was started on; segment names are port-keyed so multiple
  // runtimes (the fallback topology) can coexist on one node + ${USER}.
  unsigned port_ = 0;
  // Platform-opaque child handle (see ctp::SpawnedProcess) — no OS types here.
  ctp::SpawnedProcess proc_;
};

}  // namespace test
}  // namespace run
}  // namespace clio

#endif  // CLIO_RUNTIME_TEST_RUNTIME_SERVER_H_
