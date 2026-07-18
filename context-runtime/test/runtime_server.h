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
   * Poll until the runtime's main shared-memory segment exists, signalling that
   * the daemon has initialized far enough to serve clients. Portable: on POSIX
   * the segment is a file under /tmp/clio_$USER, on Windows a named mapping
   * — SystemInfo::OpenSharedMemory abstracts both. Returns false if the timeout
   * elapses or the daemon exits early.
   */
  bool WaitForReady(int timeout_ms = 30000) {
    // Segment names are port-keyed (see ConfigManager::GetSharedMemorySegmentName)
    // so they match the daemon started on port_.
    const std::string seg =
        ctp::ConfigParse::ExpandPath("chi_main_segment_${USER}") + "_" +
        std::to_string(port_);
    const int attempts = timeout_ms / 200;
    for (int i = 0; i < attempts; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (!IsRunning()) return false;  // daemon died during startup
      ctp::File fd;
      if (ctp::SystemInfo::OpenSharedMemory(fd, seg)) {
        ctp::SystemInfo::CloseSharedMemory(fd);
        // Let the daemon finish binding its transports / starting workers.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        return true;
      }
    }
    return false;
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
