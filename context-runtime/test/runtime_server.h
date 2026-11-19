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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/config_parse.h"
#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/msan.h"

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
   * @param recover  Recover the node's persistent state (a plain
   *   `clio_run start`). Off by default: every test gets an empty node
   *   (`start --fresh`), because all tests on a host share the conf dir
   *   (/tmp/clio_$USER) and a recovering start would bring back the pools and
   *   logs an earlier test left there. Only a test's own recovery phase (after
   *   stopping a daemon it started) passes true.
   */
  bool Start(unsigned port = 10500,
             const std::string &bind_addr = "127.0.0.1",
             bool ephemeral = false,
             bool detached = false,
             bool recover = false) {
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
    if (!recover) args.push_back("--fresh");
    if (ephemeral) args.push_back("--ephemeral");
    proc_ = ctp::SystemInfo::SpawnProcess(exe, args, log, detached);
    if (!proc_.valid) return false;
    started_ = true;
    reaped_ = false;
    wait_status_ = 0;
    last_pid_ = proc_.pid;
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
   *
   * The default cap is 120 s: a daemon that dies returns false at once, and a
   * ready one returns as soon as it is ready, so the cap only decides how slow
   * a still-starting daemon may be. 30 s was within reach of an ASan daemon
   * loading every ChiMod on a loaded runner.
   */
  bool WaitForReady(int timeout_ms = 120000) {
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
    // The marker is an INFO line. When the daemon cannot print INFO -- its
    // compile-time CTP_LOG_LEVEL (the same build as this test) or the
    // inherited CTP_LOG_LEVEL env is above info -- it can never appear, and
    // polling for it used to burn the whole budget on every Start: in the
    // leak-check preset (CLIO_CTP_LOG_LEVEL=3) a 3-daemon test outran its
    // 300 s ctest TIMEOUT. Go straight to the settle instead.
    const std::string log_path = ServerLogPath();
    bool marker_seen = false;
    if (!AdminPoolMarkerPossible()) i = attempts;
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
   * Whether the daemon can print the INFO line WaitForReady waits for.
   * HLOG compiles out levels below CTP_LOG_LEVEL (the daemon is built in the
   * same tree as this test, so this translation unit sees the same value),
   * and the daemon filters at run time by the CTP_LOG_LEVEL env var it
   * inherits from this process.
   * @return false when INFO is compiled out or filtered out at run time
   */
  static bool AdminPoolMarkerPossible() {
    if (CTP_LOG_LEVEL > kInfo) return false;
    const char *env = std::getenv("CTP_LOG_LEVEL");
    if (env == nullptr || *env == '\0') return true;
    const std::string v(env);
    if (v == "debug" || v == "DEBUG" || v == "info" || v == "INFO") return true;
    if (v == "success" || v == "SUCCESS" || v == "warning" || v == "WARNING" ||
        v == "error" || v == "ERROR" || v == "fatal" || v == "FATAL") {
      return false;
    }
    char *end = nullptr;
    const long n = std::strtol(env, &end, 10);
    if (end == env) return true;  // unparsable: the daemon keeps its default
    return n <= kInfo;
  }

  /**
   * @return true if the daemon's captured log shows the admin pool created --
   * i.e. the runtime can route a task, not merely accept a connection.
   */
  static bool ServerLogHasAdminPool(const std::string &log_path) {
    std::ifstream f(log_path, std::ios::binary);
    CTP_MSAN_UNPOISON_OBJ(f);
    if (!f) return false;
    std::ostringstream ss;
    CTP_MSAN_UNPOISON_OBJ(ss);
    ss << f.rdbuf();
    // Every byte here was moved by uninstrumented libstdc++.so -- the stream
    // objects' own state and the log text they copied -- so under MSan the
    // find() below searches memory it considers uninitialized. This readiness
    // check runs in a poll loop, which is why it dominated the report counts.
    std::string contents = ss.str();
    CTP_MSAN_UNPOISON_STRING(contents);
    return contents.find("Admin chimod pool created successfully") !=
           std::string::npos;
  }

  /**
   * Tail of the daemon's captured stdout/stderr, for failure messages.
   *
   * A daemon that dies of an abort — uncaught exception ("libc++abi:
   * terminating with ..."), a libzmq assertion, malloc corruption — says WHY on
   * stderr and nowhere else. That stderr is this log file, which lives in the
   * runner's temp dir and dies with the runner: CI keeps only the ctest output,
   * so a test that reports "exit code 134" and nothing else is unfixable from a
   * CI log. Folding the tail into the failing assertion is what makes such a
   * death diagnosable after the fact.
   *
   * The log is truncated on every Start(), so this is the current cycle's
   * daemon only. Reads at most the last 64 KiB (the first line of that window
   * may be partial).
   * @param max_lines how many trailing lines to return
   */
  static std::string LogTail(size_t max_lines = 40) {
    const std::string path = ServerLogPath();
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CTP_MSAN_UNPOISON_OBJ(f);  // stream state is owned by libstdc++.so
    if (!f.is_open()) return "(daemon log " + path + " not readable)";
    std::streamoff size = f.tellg();
    CTP_MSAN_UNPOISON_OBJ(size);
    constexpr std::streamoff kMaxBytes = 64 * 1024;
    f.seekg(size > kMaxBytes ? size - kMaxBytes : 0);
    CTP_MSAN_UNPOISON_OBJ(f);
    std::vector<std::string> lines;
    std::string line;
    for (;;) {
      std::getline(f, line);
      CTP_MSAN_UNPOISON_OBJ(f);  // getline updates the state in libstdc++.so
      if (!f) break;
      CTP_MSAN_UNPOISON_STRING(line);
      lines.push_back(line);
    }
    if (lines.empty()) return "(daemon log " + path + " is empty)";
    const size_t start =
        lines.size() > max_lines ? lines.size() - max_lines : 0;
    std::string out = "---- last " + std::to_string(lines.size() - start) +
                      " line(s) of " + path + " ----\n";
    for (size_t i = start; i < lines.size(); ++i) {
      out += "    | " + lines[i] + "\n";
    }
    return out;
  }

  /** Stop the daemon (SIGTERM then SIGKILL on POSIX — so ServerFinalize's leak
   *  report runs; TerminateProcess on Windows) and reap it. Idempotent; called
   *  by the destructor. */
  void Stop() {
    if (!started_) return;
    started_ = false;
    ctp::SystemInfo::TerminateChild(proc_);
  }

  /**
   * True while the daemon process is still alive.
   *
   * On POSIX the reap is done here rather than in SystemInfo::IsChildRunning
   * so the wait status is KEPT: a daemon that died during startup is
   * otherwise reaped with its status thrown away, and "did not become ready"
   * cannot say whether it exited, aborted or was killed (see
   * ExitDescription()).
   */
  bool IsRunning() {
    if (!started_) return false;
#ifndef _WIN32
    if (reaped_) return false;
    if (!proc_.valid || proc_.pid <= 0) return false;
    int status = 0;
    pid_t ret = waitpid(proc_.pid, &status, WNOHANG);
    if (ret == 0) return true;
    if (ret == proc_.pid) {
      reaped_ = true;
      wait_status_ = status;
    }
    return false;
#else
    return ctp::SystemInfo::IsChildRunning(proc_);
#endif
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
      if (!reaped_) {
        int status = 0;
        pid_t ret = waitpid(proc_.pid, &status, WNOHANG);
        if (ret == proc_.pid) {
          reaped_ = true;
          wait_status_ = status;
        } else if (ret < 0) {  // already reaped elsewhere
          proc_.pid = -1;
          proc_.valid = false;
          started_ = false;
          return false;
        }
      }
      if (reaped_) {
        if (exit_code != nullptr) *exit_code = ExitCodeOf(wait_status_);
        proc_.pid = -1;
        proc_.valid = false;
        started_ = false;
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
  }

  /**
   * How the daemon ended, for failure messages: "exited with code N",
   * "killed by signal N (NAME)", or "still running" / "not reaped" when
   * there is no status to report. Only meaningful after IsRunning() or
   * WaitExit() observed the exit.
   */
  std::string ExitDescription() const {
    const std::string who =
        "daemon pid " + std::to_string(last_pid_) + " ";
    if (!reaped_) return who + "has not been reaped (still running?)";
    if (WIFEXITED(wait_status_)) {
      return who + "exited with code " +
             std::to_string(WEXITSTATUS(wait_status_));
    }
    if (WIFSIGNALED(wait_status_)) {
      const int sig = WTERMSIG(wait_status_);
      const char *name = strsignal(sig);
      return who + "was killed by signal " + std::to_string(sig) + " (" +
             (name ? name : "?") + ")" +
             (WCOREDUMP(wait_status_) ? ", core dumped" : "");
    }
    return who + "ended with raw wait status " + std::to_string(wait_status_);
  }

  /**
   * Sanitizer reports the daemon wrote. Under `ctest -T memcheck` (and any
   * run with log_path in {A,UB,M,L}SAN_OPTIONS) the sanitizer writes its
   * report to `<log_path>.<pid>` instead of stderr, so a daemon that dies of
   * an ASan error leaves NOTHING in its captured log -- the report sits in a
   * file the CI log never shows. Quote it here.
   * @param max_bytes how much of each report's tail to include
   */
  std::string SanitizerReports(size_t max_bytes = 16 * 1024) const {
    std::string out;
    for (const std::string &base : SanitizerLogPaths()) {
      const std::string path = base + "." + std::to_string(last_pid_);
      std::string text = ReadTail(path, max_bytes);
      if (text.empty()) continue;
      out += "---- sanitizer report " + path + " ----\n" + text + "\n";
    }
    if (out.empty()) {
      out = "(no sanitizer report file for daemon pid " +
            std::to_string(last_pid_) + ")\n";
    }
    return out;
  }

  /**
   * Everything known about a daemon that failed to come up: whether it is
   * still running or how it ended, its sanitizer report (if any), and the
   * tail of its captured log.
   */
  std::string StartupFailureReport() {
    const bool running = IsRunning();
    std::string out = running ? "daemon pid " + std::to_string(last_pid_) +
                                    " is still running (readiness timed out)\n"
                              : ExitDescription() + "\n";
    out += SanitizerReports();
    out += LogTail();
    return out;
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
    // Per user: on a shared machine another user's leftover log at a fixed
    // name cannot be opened, and every daemon spawn then failed.
    const char *user = std::getenv("USER");
    const std::string who = (user && *user) ? std::string(user) : "user";
    return (dir / ("clio_run_test_server_" + who + ".log")).string();
  }

  static void SetEnv(const char *key, const std::string &val) {
    SetEnvVar(key, val);
  }

  /** WEXITSTATUS for a normal exit, 128+signal for a signal, else -1. */
  static int ExitCodeOf(int status) {
#ifndef _WIN32
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
#endif
    (void)status;
    return -1;
  }

  /**
   * The log_path= values of the sanitizer option variables this process
   * passes on to the daemon (ctest memcheck sets ASAN_OPTIONS=log_path=...).
   */
  static std::vector<std::string> SanitizerLogPaths() {
    std::vector<std::string> paths;
    for (const char *var : {"ASAN_OPTIONS", "UBSAN_OPTIONS", "MSAN_OPTIONS",
                            "LSAN_OPTIONS", "TSAN_OPTIONS"}) {
      const char *v = std::getenv(var);
      if (v == nullptr) continue;
      std::string opts(v);
      CTP_MSAN_UNPOISON_STRING(opts);
      const std::string key = "log_path=";
      size_t pos = opts.find(key);
      if (pos == std::string::npos) continue;
      pos += key.size();
      size_t end = opts.find(':', pos);
      std::string path = opts.substr(
          pos, end == std::string::npos ? std::string::npos : end - pos);
      if (path.size() >= 2 && (path.front() == '\'' || path.front() == '"')) {
        path = path.substr(1, path.size() - 2);
      }
      if (path.empty() || path == "stderr" || path == "stdout") continue;
      if (std::find(paths.begin(), paths.end(), path) == paths.end()) {
        paths.push_back(path);
      }
    }
    return paths;
  }

  /** @return the last max_bytes of a file, or "" if it cannot be read. */
  static std::string ReadTail(const std::string &path, size_t max_bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CTP_MSAN_UNPOISON_OBJ(f);
    if (!f.is_open()) return "";
    std::streamoff size = f.tellg();
    CTP_MSAN_UNPOISON_OBJ(size);
    const auto max = static_cast<std::streamoff>(max_bytes);
    f.seekg(size > max ? size - max : 0);
    CTP_MSAN_UNPOISON_OBJ(f);
    std::ostringstream ss;
    CTP_MSAN_UNPOISON_OBJ(ss);
    ss << f.rdbuf();
    std::string text = ss.str();
    CTP_MSAN_UNPOISON_STRING(text);
    return text;
  }

  bool started_ = false;
  // Set once IsRunning()/WaitExit() reaped the daemon; wait_status_ then
  // holds its waitpid status (POSIX) for ExitDescription().
  bool reaped_ = false;
  int wait_status_ = 0;
  // Pid of the most recently started daemon, kept after the reap so failure
  // reports can name it and find its sanitizer log.
  int last_pid_ = -1;
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
