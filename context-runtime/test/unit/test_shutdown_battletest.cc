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

/**
 * Runtime shutdown battletest (Issue #710).
 *
 * Emulates the Issue #526 failure mode — repeated start -> heavy-IO -> stop
 * cycles where stale state accumulated until the pipeline broke — and asserts
 * after EVERY cycle that `clio_run stop` left ZERO artifacts: no surviving
 * process, no memfd-dir entries (segments/sockets), no /dev/shm/chi_*, and a
 * reusable port. Scenarios cover graceful stop, stop under data-plane load,
 * forced stop, the teardown watchdog, a wedged (SIGSTOPped) runtime, and
 * recovery from an outright SIGKILL crash.
 *
 * The harness process NEVER calls CLIO_INIT: the runtime is driven via
 * RuntimeServer (spawns `clio_run start`), stop/status via the clio_run CLI,
 * and data-plane load via self-exec'd `--load-client` child processes. Each
 * cycle therefore uses only fresh processes, which is exactly how Jarvis and
 * the #526 pipeline drive the runtime.
 *
 * Cycle count: CLIO_SHUTDOWN_BT_ITERS (default 24, mirroring #526).
 */

#include "../simple_test.h"
#include "../runtime_server.h"

#ifndef _WIN32

#include <algorithm>
#include <arpa/inet.h>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "clio_ctp/introspect/system_info.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/bdev/bdev_tasks.h"

// glibc declares `environ` in <unistd.h>; macOS declares it in no header and
// forbids referencing it directly outside main(), so route through the
// sanctioned _NSGetEnviron() accessor there (used by posix_spawn below).
#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#endif

namespace {

constexpr unsigned kBtPort = 10600;
/** Exit code the runtime-side watchdog uses when graceful teardown wedges */
constexpr int kWatchdogExitCode = 2;

/** argv[0] of this binary, for self-exec'ing load clients */
const char *g_self_exe = "clio_run_shutdown_battletest";

/** @return battletest cycle count (CLIO_SHUTDOWN_BT_ITERS, default 24) */
int BtIters() {
  if (const char *env = std::getenv("CLIO_SHUTDOWN_BT_ITERS")) {
    int n = std::atoi(env);
    if (n > 0) return n;
  }
  return 24;
}

/**
 * Run a clio_run subcommand as a child process with the current environment
 * (CLIO_PORT etc. are exported by RuntimeServer::Start).
 * @param args subcommand + flags, e.g. "stop --force"
 * @return the command's exit code (-1 if it could not be run)
 */
int RunClioCmd(const std::string &args) {
  const std::string cmd = std::string(CLIO_RUN_EXE) + " " + args;
  int ret = std::system(cmd.c_str());
  if (ret == -1) return -1;
  if (WIFEXITED(ret)) return WEXITSTATUS(ret);
  return -1;
}

/** Snapshot of stale-artifact classes left behind by a runtime on kBtPort */
struct ArtifactScan {
  std::vector<std::string> port_scoped; /**< chi_*_<port>, port .ipc files */
  std::vector<std::string> pid_owned;   /**< entries owned by a given pid */
  std::vector<std::string> dead_owner;  /**< /proc symlinks to dead pids */
  std::vector<std::string> dev_shm_chi; /**< legacy /dev/shm/chi_* */
};

/**
 * Scan the per-user memfd dir and /dev/shm for runtime leftovers.
 * @param port runtime port whose keyed artifacts to collect
 * @param pid runtime pid whose owned entries to collect (<=0 to skip)
 * @return categorized artifact listing
 */
ArtifactScan ScanArtifacts(unsigned port, int pid) {
  ArtifactScan scan;
  const std::string memfd_dir = ctp::SystemInfo::GetMemfdDir();
  const std::string port_suffix = "_" + std::to_string(port);
  const std::string ipc_name = "clio_" + std::to_string(port) + ".ipc";
  const std::string zmq_ipc_name =
      "clio_zmq_" + std::to_string(port + 3) + ".ipc";
  const std::string pid_proc =
      pid > 0 ? "/proc/" + std::to_string(pid) + "/" : "";
  const std::string pid_name =
      pid > 0 ? "clio_" + std::to_string(pid) + "_" : "";

  for (const auto &name : ctp::SystemInfo::ListDirectory(memfd_dir)) {
    const std::string full_path = memfd_dir + "/" + name;
    if (name == ipc_name || name == zmq_ipc_name ||
        (name.rfind("chi_", 0) == 0 && name.size() >= port_suffix.size() &&
         name.compare(name.size() - port_suffix.size(), port_suffix.size(),
                      port_suffix) == 0)) {
      scan.port_scoped.push_back(full_path);
    }
    std::error_code ec;
    auto target = std::filesystem::read_symlink(full_path, ec);
    if (!ec) {
      const std::string t = target.string();
      if (!pid_proc.empty() && t.rfind(pid_proc, 0) == 0) {
        scan.pid_owned.push_back(full_path);
      }
      if (t.rfind("/proc/", 0) == 0) {
        int owner = std::atoi(t.c_str() + 6);
        if (owner > 0 && !ctp::SystemInfo::IsProcessAlive(owner)) {
          scan.dead_owner.push_back(full_path);
        }
      }
    } else if (!pid_name.empty() && name.rfind(pid_name, 0) == 0) {
      scan.pid_owned.push_back(full_path);
    }
  }

  std::error_code ec;
  for (auto it = std::filesystem::directory_iterator("/dev/shm", ec);
       !ec && it != std::filesystem::directory_iterator(); ++it) {
    if (it->path().filename().string().rfind("chi_", 0) == 0) {
      scan.dev_shm_chi.push_back(it->path().string());
    }
  }
  return scan;
}

/** Format a path list for failure messages. */
std::string JoinPaths(const std::vector<std::string> &paths) {
  std::string out;
  for (const auto &p : paths) {
    out += "\n    " + p;
  }
  return out;
}

/**
 * The leak oracle: after a stop, the dead runtime must have left NOTHING —
 * no port-keyed segment/socket entries, no entries owned by its pid, no
 * legacy /dev/shm/chi_*, no surviving process, and the TCP port must be
 * bindable again. Dead-owner entries from killed LOAD CLIENTS are permitted
 * here (they are reaped by the next ServerInit and asserted separately).
 * @param cycle current cycle (for the failure message)
 * @param port the stopped runtime's port
 * @param pid the stopped runtime's pid
 */
void AssertRuntimeGoneAndClean(int cycle, unsigned port, int pid) {
  ArtifactScan scan = ScanArtifacts(port, pid);
  if (!scan.port_scoped.empty()) {
    FAIL("cycle " + std::to_string(cycle) + ": leaked port-scoped artifacts:" +
         JoinPaths(scan.port_scoped));
  }
  if (!scan.pid_owned.empty()) {
    FAIL("cycle " + std::to_string(cycle) + ": leaked pid-owned artifacts:" +
         JoinPaths(scan.pid_owned));
  }
  if (!scan.dev_shm_chi.empty()) {
    FAIL("cycle " + std::to_string(cycle) + ": leaked /dev/shm entries:" +
         JoinPaths(scan.dev_shm_chi));
  }
  if (pid > 0 && !(kill(pid, 0) == -1 && errno == ESRCH)) {
    FAIL("cycle " + std::to_string(cycle) + ": runtime pid " +
         std::to_string(pid) + " still exists after stop");
  }
  // Port-free probe (SO_REUSEADDR to sidestep client-side TIME_WAIT noise;
  // the authoritative recheck is the next cycle's successful Start()).
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  int bind_rc = bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  close(fd);
  if (bind_rc != 0) {
    FAIL("cycle " + std::to_string(cycle) + ": port " + std::to_string(port) +
         " is not bindable after stop (errno " + std::to_string(errno) + ")");
  }
}

/**
 * Assert the freshly-started runtime's ServerInit (ClearUserIpcs) reaped all
 * dead-owner leftovers, e.g. segments of load clients SIGKILLed last cycle.
 * @param cycle current cycle (for the failure message)
 */
void AssertDeadOwnerEntriesReaped(int cycle) {
  ArtifactScan scan = ScanArtifacts(kBtPort, -1);
  if (!scan.dead_owner.empty()) {
    FAIL("cycle " + std::to_string(cycle) +
         ": dead-owner entries survived the next ServerInit:" +
         JoinPaths(scan.dead_owner));
  }
}

/**
 * Spawn N copies of this binary in `--load-client` mode. Each connects as a
 * CLIO client and hammers bdev writes until killed. Output goes to /dev/null.
 * @param n number of load clients
 * @param port runtime port to attack
 * @return pids of the spawned clients
 */
std::vector<pid_t> SpawnLoadClients(int n, unsigned port) {
  std::vector<pid_t> pids;
  const std::string port_str = std::to_string(port);
  for (int i = 0; i < n; ++i) {
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, 1, 2);
    const char *argv_child[] = {g_self_exe, "--load-client", port_str.c_str(),
                                nullptr};
    pid_t pid = -1;
    int rc = posix_spawn(&pid, g_self_exe, &actions, nullptr,
                         const_cast<char *const *>(argv_child), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc == 0 && pid > 0) {
      pids.push_back(pid);
    }
  }
  REQUIRE(static_cast<int>(pids.size()) == n);
  return pids;
}

/**
 * SIGKILL and reap the load clients (they are expected to die ugly — that is
 * the point: #526's clients never shut down cleanly either).
 * @param pids load-client pids from SpawnLoadClients
 */
void KillLoadClients(std::vector<pid_t> &pids) {
  for (pid_t pid : pids) {
    kill(pid, SIGKILL);
  }
  for (pid_t pid : pids) {
    int status = 0;
    waitpid(pid, &status, 0);
  }
  pids.clear();
}

/**
 * One start -> [load] -> stop -> assert-clean cycle.
 * @param cycle cycle index (failure messages)
 * @param stop_args clio_run stop arguments ("stop", "stop --force", ...)
 * @param n_load_clients data-plane load processes to run during the stop
 * @param load_ms how long to let the load build before stopping
 * @param allow_watchdog whether daemon exit code 2 (watchdog) is acceptable
 */
void RunStopCycle(int cycle, const std::string &stop_args, int n_load_clients,
                  int load_ms, bool allow_watchdog) {
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kBtPort));
  REQUIRE(server.WaitForReady());
  AssertDeadOwnerEntriesReaped(cycle);
  const int pid = static_cast<int>(server.Pid());

  std::vector<pid_t> load;
  if (n_load_clients > 0) {
    load = SpawnLoadClients(n_load_clients, kBtPort);
    std::this_thread::sleep_for(std::chrono::milliseconds(load_ms));
  }

  const int stop_rc = RunClioCmd(stop_args);
  if (stop_rc != 0) {
    KillLoadClients(load);
    FAIL("cycle " + std::to_string(cycle) + ": `clio_run " + stop_args +
         "` returned " + std::to_string(stop_rc));
  }

  int exit_code = -1;
  const bool exited = server.WaitExit(45000, &exit_code);
  KillLoadClients(load);
  if (!exited) {
    FAIL("cycle " + std::to_string(cycle) +
         ": daemon did not exit within 45000 ms after stop");
  }
  const bool code_ok =
      exit_code == 0 || (allow_watchdog && exit_code == kWatchdogExitCode);
  if (!code_ok) {
    FAIL("cycle " + std::to_string(cycle) + ": unexpected daemon exit code " +
         std::to_string(exit_code));
  }
  if (exit_code == kWatchdogExitCode) {
    INFO("cycle " + std::to_string(cycle) + ": watchdog-forced exit (2)");
  }

  AssertRuntimeGoneAndClean(cycle, kBtPort, pid);
}

}  // namespace

// ===========================================================================
// Scenarios
// ===========================================================================

TEST_CASE("ShutdownBattletest - Graceful Churn", "[bt_churn]") {
  const int iters = BtIters();
  for (int cycle = 1; cycle <= iters; ++cycle) {
    INFO("graceful churn cycle " + std::to_string(cycle) + "/" +
         std::to_string(iters));
    RunStopCycle(cycle, "stop", /*n_load_clients=*/0, /*load_ms=*/0,
                 /*allow_watchdog=*/false);
  }
}

TEST_CASE("ShutdownBattletest - Churn Under Load", "[bt_load]") {
  const int iters = BtIters();
  for (int cycle = 1; cycle <= iters; ++cycle) {
    INFO("load churn cycle " + std::to_string(cycle) + "/" +
         std::to_string(iters));
    RunStopCycle(cycle, "stop", /*n_load_clients=*/4, /*load_ms=*/2000,
                 /*allow_watchdog=*/true);
  }
}

TEST_CASE("ShutdownBattletest - Force Under Load", "[bt_force]") {
  const int iters = BtIters();
  for (int cycle = 1; cycle <= iters; ++cycle) {
    INFO("force churn cycle " + std::to_string(cycle) + "/" +
         std::to_string(iters));
    const auto start = std::chrono::steady_clock::now();
    RunStopCycle(cycle, "stop --force", /*n_load_clients=*/4,
                 /*load_ms=*/2000, /*allow_watchdog=*/false);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
    INFO("force cycle " + std::to_string(cycle) + " took " +
         std::to_string(ms) + " ms");
  }
}

TEST_CASE("ShutdownBattletest - Watchdog Bound", "[bt_watchdog]") {
  // Grace period deliberately far too small to drain 4 clients' load: the
  // invariant is BOUNDED convergence + zero artifacts, whether the teardown
  // makes it (exit 0) or the watchdog fires (exit 2).
  const int iters = std::min(BtIters(), 5);
  for (int cycle = 1; cycle <= iters; ++cycle) {
    INFO("watchdog cycle " + std::to_string(cycle) + "/" +
         std::to_string(iters));
    const auto start = std::chrono::steady_clock::now();
    RunStopCycle(cycle, "stop --grace-period 100", /*n_load_clients=*/4,
                 /*load_ms=*/2000, /*allow_watchdog=*/true);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
    // grace(0.1s) + watchdog margin(15s) + stop-side wait slack
    REQUIRE(ms < 60000);
  }
}

TEST_CASE("ShutdownBattletest - Wedged Runtime", "[bt_wedged]") {
  // SIGSTOP freezes every runtime thread: no worker can service the stop
  // task, so the CLI's kill escalation (SIGTERM has no effect on a stopped
  // process -> SIGKILL) must converge and sweep the artifacts client-side.
  const int iters = std::min(BtIters(), 5);
  for (int cycle = 1; cycle <= iters; ++cycle) {
    INFO("wedged cycle " + std::to_string(cycle) + "/" + std::to_string(iters));
    clio::run::test::RuntimeServer server;
    REQUIRE(server.Start(kBtPort));
    REQUIRE(server.WaitForReady());
    const int pid = static_cast<int>(server.Pid());

    REQUIRE(kill(pid, SIGSTOP) == 0);

    const int stop_rc = RunClioCmd("stop");
    if (stop_rc != 0) {
      kill(pid, SIGKILL);  // do not leak a stopped daemon on failure
      FAIL("cycle " + std::to_string(cycle) +
           ": stop on wedged runtime returned " + std::to_string(stop_rc));
    }

    int exit_code = -1;
    REQUIRE(server.WaitExit(10000, &exit_code));
    REQUIRE(exit_code == 128 + SIGKILL);  // escalation had to SIGKILL it

    AssertRuntimeGoneAndClean(cycle, kBtPort, pid);
  }
}

TEST_CASE("ShutdownBattletest - Crash Recovery", "[bt_crash]") {
  // Emulate today's `pkill -9` reality: SIGKILL the daemon mid-load, then
  // prove `status` reports the stale wreckage (exit 2), `stop` doubles as
  // the recovery tool (sweep -> rc 0), `status` then reports clean (exit 1),
  // and a fresh start succeeds.
  const int iters = std::min(BtIters(), 5);
  for (int cycle = 1; cycle <= iters; ++cycle) {
    INFO("crash cycle " + std::to_string(cycle) + "/" + std::to_string(iters));
    clio::run::test::RuntimeServer server;
    REQUIRE(server.Start(kBtPort));
    REQUIRE(server.WaitForReady());
    const int pid = static_cast<int>(server.Pid());

    // status on a live runtime: RUNNING (exit 0)
    REQUIRE(RunClioCmd("status") == 0);

    std::vector<pid_t> load = SpawnLoadClients(2, kBtPort);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    kill(pid, SIGKILL);
    int exit_code = -1;
    REQUIRE(server.WaitExit(10000, &exit_code));
    KillLoadClients(load);

    // The corpse left artifacts: status must expose them.
    REQUIRE(RunClioCmd("status") == 2);
    // stop must recover: discover the dead pid, sweep, and return success.
    REQUIRE(RunClioCmd("stop") == 0);
    // Now the node is clean.
    REQUIRE(RunClioCmd("status") == 1);
    AssertRuntimeGoneAndClean(cycle, kBtPort, pid);
  }
}

TEST_CASE("ShutdownBattletest - Transport Churn", "[bt_transports]") {
  // The stop CLI has distinct paths per IPC mode (SHM maps the task queue;
  // TCP/IPC go via ZMQ). Churn each mode.
  const int iters = std::max(2, std::min(BtIters(), 8));
  for (const char *mode : {"SHM", "TCP", "IPC"}) {
    clio::run::test::SetEnvVar("CLIO_IPC_MODE", mode);
    for (int cycle = 1; cycle <= iters; ++cycle) {
      INFO(std::string("transport ") + mode + " cycle " +
           std::to_string(cycle) + "/" + std::to_string(iters));
      RunStopCycle(cycle, "stop", /*n_load_clients=*/2, /*load_ms=*/1000,
                   /*allow_watchdog=*/true);
    }
  }
  clio::run::test::UnsetEnvVar("CLIO_IPC_MODE");
}

// ===========================================================================
// Load-client mode (self-exec'd data-plane stress)
// ===========================================================================

namespace {

inline clio::run::priv::vector<clio::run::bdev::Block> WrapBlock(
    const clio::run::bdev::Block &block) {
  clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
  blocks.push_back(block);
  return blocks;
}

/**
 * Data-plane load client: connect to the runtime and hammer 1 MiB bdev
 * writes until killed. Large buffers force on-demand shared-memory segment
 * creation (IncreaseClientShm/IncreaseServerShm) — the artifact class that
 * accumulated in Issue #526. All failures exit quietly: the runtime dying
 * underneath us is an EXPECTED part of every scenario.
 * @param port runtime port to attack
 * @return process exit code (always 0; the harness SIGKILLs us anyway)
 */
int RunLoadClient(unsigned port) {
  constexpr clio::run::u64 kRamSize = 64ull * 1024 * 1024;
  constexpr clio::run::u64 kIoSize = 1024 * 1024;
  clio::run::test::SetEnvVar("CLIO_PORT", std::to_string(port));
  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  clio::run::test::SetEnvVar("CLIO_WAIT_SERVER", "10");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    return 0;
  }
  try {
    clio::run::PoolId pool_id(9600, 0);
    clio::run::bdev::Client client(pool_id);
    const std::string pool_name = "bt_load_" + std::to_string(getpid());
    auto create_task = client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), pool_name, pool_id,
        clio::run::bdev::BdevType::kRam, kRamSize);
    create_task.Wait();
    if (create_task->return_code_ != 0) return 0;
    client.pool_id_ = create_task->new_pool_id_;

    auto alloc_task =
        client.AsyncAllocateBlocks(clio::run::PoolQuery::Local(), kIoSize);
    alloc_task.Wait();
    if (alloc_task->return_code_ != 0 || alloc_task->blocks_.size() == 0) {
      return 0;
    }
    clio::run::bdev::Block block = alloc_task->blocks_[0];

    while (true) {
      auto buffer = CLIO_IPC->AllocateBuffer(kIoSize);
      if (buffer.IsNull()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      std::memset(buffer.ptr_, 0xA5, kIoSize);
      auto write_task = client.AsyncWrite(
          clio::run::PoolQuery::Local(), WrapBlock(block),
          buffer.shm_.template Cast<void>().template Cast<void>(), kIoSize);
      write_task.Wait(1.0f);
      CLIO_IPC->FreeBuffer(buffer);
    }
  } catch (...) {
    // Runtime went away mid-flight — expected.
  }
  return 0;
}

}  // namespace

/**
 * Custom main: `--load-client <port>` runs the data-plane stressor; anything
 * else runs the simple_test suite with argv[1] as the test-name filter (the
 * same contract as SIMPLE_TEST_MAIN, which we cannot use because of the
 * load-client dispatch).
 */
int main(int argc, char *argv[]) {
  if (argc > 0 && argv[0] != nullptr && argv[0][0] != '\0') {
    g_self_exe = argv[0];
  }
  if (argc >= 3 && std::strcmp(argv[1], "--load-client") == 0) {
    return RunLoadClient(static_cast<unsigned>(std::atoi(argv[2])));
  }
  std::string filter = argc > 1 ? argv[1] : "";
  return SimpleTest::run_all_tests(filter);
}

#else  // _WIN32

/** The battletest drives POSIX signals/spawn; Windows builds compile a stub. */
int main() { return 0; }

#endif  // _WIN32
