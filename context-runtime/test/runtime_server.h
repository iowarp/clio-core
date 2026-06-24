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
 * Spawning a real binary (posix_spawn on POSIX, CreateProcess on Windows) and
 * connecting to it as an ordinary external client is portable across Linux,
 * macOS and Windows — which is why the tests that use this helper no longer
 * need an `if(NOT WIN32)` guard.
 */

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/config_parse.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

// Defined in clio::run::test; tests reach it as clio::run::test::RuntimeServer.
namespace clio {
namespace run {
namespace test {

/** Portable setenv(): tests set CLIO_IPC_MODE / CLIO_WITH_RUNTIME etc., and
 *  Windows has no setenv(). */
inline void SetEnvVar(const char *key, const std::string &val) {
#ifdef _WIN32
  _putenv_s(key, val.c_str());
#else
  setenv(key, val.c_str(), 1);
#endif
}

/** Portable unsetenv(). On Windows _putenv_s(key, "") removes the variable. */
inline void UnsetEnvVar(const char *key) {
#ifdef _WIN32
  _putenv_s(key, "");
#else
  unsetenv(key);
#endif
}

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
  bool Start(unsigned port = 10500,
             const std::string &bind_addr = "127.0.0.1",
             bool ephemeral = false) {
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

#ifdef _WIN32
    std::string cmd = "\"" + exe + "\" start";
    if (ephemeral) cmd += " --ephemeral";
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    // Redirect child stdout/stderr to the log file so the daemon's worker
    // chatter does not flood the test output (and is inspectable on failure).
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;
    HANDLE hlog = CreateFileA(log.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hlog != INVALID_HANDLE_VALUE) {
      si.dwFlags |= STARTF_USESTDHANDLES;
      si.hStdOutput = hlog;
      si.hStdError = hlog;
      si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }
    ZeroMemory(&pi_, sizeof(pi_));
    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back('\0');
    BOOL ok = CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr,
                             TRUE, 0, nullptr, nullptr, &si, &pi_);
    if (hlog != INVALID_HANDLE_VALUE) CloseHandle(hlog);
    if (!ok) return false;
    started_ = true;
    return true;
#else
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 1, log.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&actions, 1, 2);
    const char *argv_plain[] = {exe.c_str(), "start", nullptr};
    const char *argv_eph[] = {exe.c_str(), "start", "--ephemeral", nullptr};
    int rc = posix_spawn(&pid_, exe.c_str(), &actions, nullptr,
                         const_cast<char *const *>(ephemeral ? argv_eph
                                                             : argv_plain),
                         environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) {
      pid_ = -1;
      return false;
    }
    started_ = true;
    return true;
#endif
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

  /** Stop the daemon (SIGTERM then SIGKILL on POSIX; TerminateProcess on
   *  Windows) and reap it. Idempotent; called by the destructor. */
  void Stop() {
    if (!started_) return;
    started_ = false;
#ifdef _WIN32
    TerminateProcess(pi_.hProcess, 0);
    WaitForSingleObject(pi_.hProcess, 5000);
    CloseHandle(pi_.hProcess);
    CloseHandle(pi_.hThread);
#else
    if (pid_ <= 0) return;
    kill(pid_, SIGTERM);
    for (int i = 0; i < 50; ++i) {  // up to 5s for a clean shutdown
      int status;
      if (waitpid(pid_, &status, WNOHANG) != 0) {
        pid_ = -1;
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    kill(pid_, SIGKILL);
    int status;
    waitpid(pid_, &status, 0);
    pid_ = -1;
#endif
  }

  /** True while the daemon process is still alive. */
  bool IsRunning() {
    if (!started_) return false;
#ifdef _WIN32
    DWORD code = 0;
    if (!GetExitCodeProcess(pi_.hProcess, &code)) return false;
    return code == STILL_ACTIVE;
#else
    if (pid_ <= 0) return false;
    int status;
    return waitpid(pid_, &status, WNOHANG) == 0;  // 0 => still running
#endif
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
#ifdef _WIN32
    char tmp[MAX_PATH];
    DWORD n = GetTempPathA(MAX_PATH, tmp);
    std::string dir = (n > 0 && n < MAX_PATH) ? std::string(tmp, n) : ".\\";
    return dir + "clio_run_test_server.log";
#else
    return "/tmp/clio_run_test_server.log";
#endif
  }

  static void SetEnv(const char *key, const std::string &val) {
    SetEnvVar(key, val);
  }

  bool started_ = false;
  // Port the daemon was started on; segment names are port-keyed so multiple
  // runtimes (the fallback topology) can coexist on one node + ${USER}.
  unsigned port_ = 0;
#ifdef _WIN32
  PROCESS_INFORMATION pi_{};
#else
  pid_t pid_ = -1;
#endif
};

}  // namespace test
}  // namespace run
}  // namespace clio

#endif  // CLIO_RUNTIME_TEST_RUNTIME_SERVER_H_
