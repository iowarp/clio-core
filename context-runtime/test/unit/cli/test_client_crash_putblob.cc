/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Single-node client-crash PutBlob robustness guard (issue #722 / "client dies
 * mid-operation").
 *
 * A client attaches to an out-of-process runtime daemon, fires a burst of
 * AsyncPutBlob requests, and then terminates IMMEDIATELY (_exit) WITHOUT waiting
 * for the responses and without finalizing.
 *
 * SCOPE: a SAME-HOST client is co-located with the daemon and both submits and
 * completes PutBlobs over shared memory (origin kClientShm) — its requests never
 * traverse the daemon's client-facing ZMQ transport, so the undeliverable-
 * response re-queue in IpcCpu2CpuZmq::SendOut (#722) is NOT reachable here
 * regardless of CLIO_IPC_MODE (verified empirically). This single-node case
 * therefore guards the runtime's RESILIENCE to a client dying mid-PutBlob: the
 * daemon must keep running and must not leak the abandoned task / the client's
 * orphaned SHM buffers. The network drop path (#722 proper) is exercised by the
 * distributed docker suite (context-transfer-engine/test/integration/
 * client_crash), where the client lives in a SEPARATE container and cannot use
 * shared memory.
 *
 * Assertions (on the surviving daemon, after a graceful stop):
 *   (1) No SHM leak — via the EXISTING leak detector: a graceful stop runs
 *       ServerFinalize -> ReportRuntimeLeaks("ServerFinalize"), which logs any
 *       outstanding SHM bytes at ERROR (meaningful under the leak-check build;
 *       the ABSENCE of those lines is a build-independent clean signal).
 *   (2) No unbounded re-queue (best-effort): if the ZMQ path is ever reached, a
 *       dead client must not trigger the #722 runaway (22.9M retries).
 *
 * POSIX-only (fork()/_exit + <sys/wait.h>); the whole cli/ directory is gated
 * `if(NOT WIN32)`. The binary also exposes a standalone `crash-client` mode used
 * by the distributed docker suite's separate-container client.
 */
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <clio_ctp/introspect/system_info.h>
#include <fstream>

#include "clio_ctp/introspect/system_info.h"
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

// Unique port so this test can run concurrently with the other live-runtime cli
// tests (they serialize via the RESOURCE_LOCK, but keep a distinct port anyway).
constexpr unsigned kPort = 10613;
// CTE core pool composed below (clio_cte_core at pool 512.0).
constexpr clio::run::u32 kCorePoolMajor = 512;

// Run the clio_run binary as a subprocess with a timeout. Returns its exit code
// (or negative on spawn/timeout failure). Mirrors test_cte_fallback.cc.
int RunCliTimed(const std::vector<std::string> &args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char *> argv;
  for (auto &a : full) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    int n = open("/dev/null", O_WRONLY);
    if (n >= 0) { dup2(n, 1); dup2(n, 2); close(n); }
    execv(argv[0], argv.data());
    _exit(127);
  }
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -3;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
// A burst of PutBlobs, fired right before the client dies, so that essentially
// all of their responses land after the client is gone (undeliverable). Doubles
// as the #722 stress (many undeliverable responses, not just one). Overridable
// via env for experimentation.
int EnvInt(const char *name, int def) {
  const char *e = std::getenv(name);
  if (!e || !*e) return def;
  int v = std::atoi(e);
  return v > 0 ? v : def;
}
// Let the abandoned burst reach the daemon and run before stopping it. This is
// a floor, not a budget: whatever is still in flight when the stop arrives is
// drained by the graceful stop below, which must leak nothing either way. (The
// #722 undeliverable-response drop window this used to outwait is not reached
// on a single node -- the client's responses never fail to send -- so there is
// no drop marker to poll for here; see NOTE ON SCOPE in the test.)
constexpr int kDropWaitSec = 9;

std::string ReadWholeFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

size_t CountOccurrences(const std::string &hay, const std::string &needle) {
  size_t n = 0, pos = 0;
  while ((pos = hay.find(needle, pos)) != std::string::npos) {
    ++n;
    pos += needle.size();
  }
  return n;
}

int g_nocrash_ok = 0;  // diagnostic: PutBlobs that returned rc==0 in nocrash mode

/** How long a graceful stop (SIGTERM -> ServerFinalize -> exit) may take.
 *  Only a hang reaches it; an ASan daemon finalizes in seconds. */
constexpr int kGracefulStopMs = 60000;

/**
 * Stop the daemon with SIGTERM and wait for it to exit ON ITS OWN, so its
 * ServerFinalize (and the leak report in it) runs to completion.
 * RuntimeServer::Stop() SIGKILLs after a 5 s grace, which on a slow runner
 * kills the daemon mid-finalize and leaves a log with no leak report -- a
 * vacuous pass for a test that asserts the report's absence.
 * @param server the running daemon
 */
void StopGracefully(clio::run::test::RuntimeServer &server) {
  REQUIRE(server.Pid() > 0);
  REQUIRE(kill(server.Pid(), SIGTERM) == 0);
  int exit_code = -1;
  const bool exited = server.WaitExit(kGracefulStopMs, &exit_code);
  if (!exited) {
    FAIL("daemon did not exit within " + std::to_string(kGracefulStopMs) +
         " ms of SIGTERM\n" + clio::run::test::RuntimeServer::LogTail());
  }
  if (exit_code != 0) {
    FAIL("daemon exited with code " + std::to_string(exit_code) +
         " after SIGTERM (finalize did not complete)\n" +
         clio::run::test::RuntimeServer::LogTail());
  }
}

/**
 * Print one progress line at once (unbuffered stderr), so a step that hangs
 * until the ctest TIMEOUT still shows the last step that completed.
 * @param what the step just finished
 */
void Progress(const std::string &what) {
  std::cout.flush();
  std::fprintf(stderr, "  [STEP] %s\n", what.c_str());
  std::fflush(stderr);
}

/** How long a forked crash client may live. It attaches, does one round trip
 *  and fires a burst; a healthy one exits in seconds. */
constexpr unsigned kChildLifeSec = 60;
/** How long the parent waits for a forked crash client (> kChildLifeSec). */
constexpr int kChildWaitMs = 90000;

/**
 * Wait for a forked crash client with a deadline. A child stuck in its
 * connect round trip used to leave an unbounded waitpid() hanging until the
 * ctest TIMEOUT, with no output; this fails fast with the daemon's log tail.
 * @param child pid returned by fork()
 * @param status receives the waitpid() status
 */
void WaitChild(pid_t child, int *status) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kChildWaitMs);
  while (true) {
    const pid_t r = waitpid(child, status, WNOHANG);
    if (r == child) return;
    REQUIRE(r == 0);
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(child, SIGKILL);
      waitpid(child, status, 0);
      FAIL("crash client " + std::to_string(child) + " still running after " +
           std::to_string(kChildWaitMs) + " ms\n" +
           clio::run::test::RuntimeServer::LogTail());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

// Attach as a client, fire a burst of PutBlobs, then _exit WITHOUT waiting for
// the responses. Never returns. Used two ways:
//   - forked from the single-node TEST_CASE (quiet=true), where the client is
//     co-located with the daemon and completes over SHM; and
//   - as a standalone `crash-client` process in the distributed docker suite,
//     where the client lives in a SEPARATE container (no shared /dev/shm) and is
//     therefore forced onto the network path — so the daemon's response becomes
//     undeliverable when this process dies (the #722 path).
[[noreturn]] void CrashClientMain(bool quiet) {
  // In the forked single-node case, silence the child so its worker/log chatter
  // does not pollute the parent's test output (the daemon log is asserted on).
  if (quiet) {
    // Forked from a test case: never outlive it. SIGALRM's default action
    // terminates the child if its connect round trip (mk.Wait) never ends.
    alarm(kChildLifeSec);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDOUT_FILENO);
      close(devnull);
    }
  }
  setenv("CLIO_WITH_RUNTIME", "0", 1);   // client-only, never spawn a runtime
  // Transport mode inherited from the environment (CLIO_IPC_MODE). Use a network
  // transport (tcp/ipc) so the client does NOT attach shared memory: its PutBlob
  // requests travel over the wire and the daemon must send a NETWORK response,
  // which becomes undeliverable when this process dies (the #722 path). A
  // co-located SHM client would instead complete via shared memory with no
  // network response to fail.
  if (!std::getenv("CLIO_IPC_MODE")) setenv("CLIO_IPC_MODE", "tcp", 1);

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    _exit(2);
  }
  auto *ipc = CLIO_IPC;
  if (ipc == nullptr) _exit(3);

  clio::cte::core::Client core;
  core.Init(clio::run::PoolId(kCorePoolMajor, 0));

  // One completed round-trip proves the client is fully connected (its dial-back
  // response listener exists) before we start abandoning requests.
  auto mk = core.AsyncGetOrCreateTag("/client_crash/tag",
                                     clio::cte::core::TagId::GetNull(),
                                     clio::run::PoolQuery::Local());
  mk.Wait();
  clio::cte::core::TagId tag_id = mk->tag_id_;
  if (mk->GetReturnCode() != 0 || tag_id.IsNull()) _exit(4);

  const int burst = EnvInt("CRASH_BURST", 64);
  const clio::run::u64 blob_size =
      static_cast<clio::run::u64>(EnvInt("CRASH_BLOB_KB", 256)) * 1024u;
  const int flush_ms = EnvInt("CRASH_FLUSH_MS", 300);
  // Diagnostic control: when set, Wait() on every future and exit cleanly (no
  // crash) — used to confirm the network submit path is exercised at all.
  const bool nocrash = std::getenv("CRASH_NOCRASH") != nullptr;

  // Fire the burst without waiting. AsyncPutBlob enqueues + sends; we do NOT
  // Wait(), do NOT FreeBuffer, and do NOT finalize — we just die.
  for (int i = 0; i < burst; ++i) {
    ctp::ipc::FullPtr<char> wbuf = ipc->AllocateBuffer(blob_size);
    if (wbuf.IsNull()) break;
    std::memset(wbuf.ptr_, static_cast<int>('a' + (i & 15)), blob_size);
    std::string blob_name = std::to_string(i);
    auto pb = core.AsyncPutBlob(tag_id, blob_name, 0, blob_size,
                                wbuf.shm_.template Cast<void>(), -1.0f,
                                clio::cte::core::Context(), 0u,
                                clio::run::PoolQuery::Local());
    if (nocrash) {  // diagnostic: prove the submit path round-trips
      pb.Wait();
      if (pb->GetReturnCode() == 0) ++g_nocrash_ok;
      ipc->FreeBuffer(wbuf);
    }
  }

  if (nocrash) {
    // "verify" mode: a healthy round-trip after a prior crash proves the daemon
    // survived and is still responsive. Exit non-zero if any PutBlob failed.
    FILE *rf = fopen("/tmp/crash_client_result.txt", "w");
    if (rf) { fprintf(rf, "ok=%d of %d\n", g_nocrash_ok, burst); fclose(rf); }
    clio::run::CLIO_RUNTIME_FINALIZE();
    _exit(g_nocrash_ok == burst ? 0 : 5);
  }

  // Give the client's send path a moment to flush the requests onto the wire so
  // the daemon actually receives and processes them (their responses then land
  // after we die). Then terminate mid-operation.
  if (flush_ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(flush_ms));
  }
  _exit(0);
}

}  // namespace

TEST_CASE("ClientCrash - PutBlob then terminate drops task with no leak [leak]",
          "[cli][cte][crash][leak]") {
  const std::string log_path = "/tmp/clio_client_crash_putblob.log";
  ::unlink(log_path.c_str());
  clio::run::test::SetEnvVar("CLIO_TEST_SERVER_LOG", log_path);
  // Make sure kWarning (drop marker) and kError (leak report) are not filtered.
  clio::run::test::SetEnvVar("CTP_LOG_LEVEL", "info");
  clio::run::test::SetEnvVar("CLIO_PORT", std::to_string(kPort));

  const fs::path work = fs::temp_directory_path() / "clio_client_crash_test";
  fs::remove_all(work);
  fs::create_directories(work);

  // Compose a self-contained CTE core pool (512.0) backed by a ram device; CTE
  // creates its own bdev target locally. Mirrors test_restart_compose.yaml.
  const fs::path compose_yaml = work / "compose.yaml";
  {
    std::ofstream f(compose_yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: cte_client_crash\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << (work / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 256mb\n"
         "    dpe:\n"
         "      dpe_type: random\n";
  }

  // Start the runtime daemon out-of-process (ephemeral: no default compose), then
  // compose the CTE core pool so the client can PutBlob against pool 512.0.
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort, "127.0.0.1", /*ephemeral=*/true));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", "start", compose_yaml.string()}, 60) == 0);

  // Fork the crash-client. Client mode does not dlopen ChiMods, so fork without
  // exec is macOS-safe here (same rationale as test_external_client.cc).
  pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    CrashClientMain(/*quiet=*/true);  // never returns
  }
  int status = 0;
  WaitChild(child, &status);
  INFO("crash-client exited: exited=" << WIFEXITED(status)
       << " code=" << (WIFEXITED(status) ? WEXITSTATUS(status) : -1)
       << " signaled=" << WIFSIGNALED(status));

  // Give the daemon time to (a) execute the abandoned PutBlobs and (b) hit its
  // bounded-drop window for the undeliverable responses.
  std::this_thread::sleep_for(std::chrono::seconds(kDropWaitSec));

  // Graceful stop -> ServerFinalize -> ReportRuntimeLeaks writes to the log.
  // Wait for the daemon's own exit (code 0) so the report is known to have
  // been written before its absence is asserted below.
  StopGracefully(server);

  const std::string log = ReadWholeFile(log_path);
  REQUIRE(!log.empty());

  // NOTE ON SCOPE: a SAME-HOST client is co-located with the daemon and submits
  // /completes PutBlobs over shared memory (origin kClientShm); its requests
  // never traverse the daemon's client-facing ZMQ transport, so the
  // undeliverable-response re-queue in IpcCpu2CpuZmq::SendOut (#722) is NOT
  // reachable here regardless of CLIO_IPC_MODE (verified empirically). What this
  // single-node case guards is the runtime's RESILIENCE to a client dying
  // mid-PutBlob: the daemon must keep running and must not leak the abandoned
  // task or the client's orphaned SHM buffers. The network drop path (#722) is
  // exercised by the distributed docker suite (test/integration/client_crash),
  // where the client lives in a separate container and cannot use SHM.

  // The daemon must still be reapable/alive right up to our graceful stop.
  // (If it had crashed on the client death, WaitForReady/compose would already
  // have failed above, or the log would be truncated mid-run.)

  // ---- Condition (1): no memory leaked in the runtime. ----
  // ReportRuntimeLeaks logs these ERROR lines only when SHM bytes are still
  // outstanding at shutdown (leak-check build). Their ABSENCE is the clean
  // signal and is build-independent (nothing to assert in a non-leak build).
  REQUIRE(CountOccurrences(log, "total SHM bytes leaked") == 0);
  REQUIRE(CountOccurrences(log, "SHM allocator #") == 0);

  // ---- Condition (2), best-effort: no unbounded re-queue. ----
  // If the ZMQ response path ever IS reached (future routing changes), a dead
  // client must not trigger the #722 runaway (22.9M retries). Bounded either
  // way; a huge count here would flag a regression.
  const size_t requeues = CountOccurrences(log, "re-queueing client response");
  REQUIRE(requeues < 5000);

  clio::run::test::UnsetEnvVar("CLIO_TEST_SERVER_LOG");
  fs::remove_all(work);
}

/**
 * Attach as a SHM client, map a data segment and complete one PutBlob from it
 * (so the daemon has resolved, i.e. mapped, that segment), touch `ready`, then
 * idle until killed. Never returns.
 * @param ready file to create once the segment is in use by the daemon
 */
[[noreturn]] void ShmIdleClientMain(const std::string &ready) {
  int devnull = open("/dev/null", O_WRONLY);
  if (devnull >= 0) {
    dup2(devnull, STDOUT_FILENO);
    close(devnull);
  }
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  setenv("CLIO_IPC_MODE", "shm", 1);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) _exit(2);
  auto *ipc = CLIO_IPC;
  if (ipc == nullptr) _exit(3);
  clio::cte::core::Client core;
  core.Init(clio::run::PoolId(kCorePoolMajor, 0));
  auto mk = core.AsyncGetOrCreateTag("/reap/tag",
                                     clio::cte::core::TagId::GetNull(),
                                     clio::run::PoolQuery::Local());
  mk.Wait();
  if (mk->GetReturnCode() != 0) _exit(4);
  const clio::run::u64 size = 64 * 1024;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(size);
  if (buf.IsNull()) _exit(5);
  std::memset(buf.ptr_, 'r', size);
  auto pb = core.AsyncPutBlob(mk->tag_id_, "b", 0, size,
                              buf.shm_.template Cast<void>(), -1.0f,
                              clio::cte::core::Context(), 0u,
                              clio::run::PoolQuery::Local());
  pb.Wait();
  if (pb->GetReturnCode() != 0) _exit(6);
  { std::ofstream(ready) << "ok"; }
  for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}

/**
 * #1192: a SHM client killed with SIGKILL leaves its data segments behind.
 * The daemon must reclaim them once the client is dead, idle and has nothing
 * in flight, while it keeps running -- not only at the next runtime start.
 * (The periodic reaper used to be disabled, so they lived as long as the
 * daemon.)
 */
TEST_CASE("ClientCrash - dead SHM client's segments are reaped while running",
          "[cli][crash][reap][1192]") {
  const std::string log_path = "/tmp/clio_client_reap.log";
  ::unlink(log_path.c_str());
  clio::run::test::SetEnvVar("CLIO_TEST_SERVER_LOG", log_path);
  clio::run::test::SetEnvVar("CTP_LOG_LEVEL", "info");
  clio::run::test::SetEnvVar("CLIO_PORT", std::to_string(kPort));

  const fs::path work = fs::temp_directory_path() / "clio_client_reap_test";
  fs::remove_all(work);
  fs::create_directories(work);
  const fs::path conf = work / "clio.yaml";
  {
    std::ofstream f(conf);
    f << "runtime:\n"
         "  client_reap_grace_s: 2\n"
         "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: cte_client_reap\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << (work / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 64mb\n"
         "    dpe:\n"
         "      dpe_type: random\n";
  }
  clio::run::test::SetEnvVar("CLIO_SERVER_CONF", conf.string());

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort, "127.0.0.1"));
  REQUIRE(server.WaitForReady());

  const std::string ready = (work / "client_ready").string();
  pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) ShmIdleClientMain(ready);  // never returns
  bool up = false;
  for (int i = 0; i < 300 && !up; ++i) {
    up = fs::exists(ready);
    if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  REQUIRE(up);

  // The client's first data segment. Observed through its name entry, not
  // the log: leak-check builds compile kInfo logging out. symlink_status,
  // because the entry links into the (soon dead) client's /proc/<pid>/fd and
  // would read as missing the moment the client dies.
  const fs::path seg = ctp::SystemInfo::GetMemfdPath(
      "clio_" + std::to_string(child) + "_0");
  auto present = [&seg]() {
    std::error_code ec;
    return fs::exists(fs::symlink_status(seg, ec));
  };
  INFO("client segment entry: " << seg.string());
  REQUIRE(present());

  kill(child, SIGKILL);
  int status = 0;
  WaitChild(child, &status);

  // Grace 2 s plus the 1 s reap period: well inside 30 s.
  bool reaped = false;
  for (int i = 0; i < 300 && !reaped; ++i) {
    reaped = !present();
    if (!reaped) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  REQUIRE(reaped);
  REQUIRE(server.IsRunning());

  server.Stop();
  clio::run::test::UnsetEnvVar("CLIO_SERVER_CONF");
  clio::run::test::UnsetEnvVar("CLIO_TEST_SERVER_LOG");
  fs::remove_all(work);
}

TEST_CASE("ClientCrash - a dead client's SHM segments are reclaimed (#1192)",
          "[cli][cte][crash][reap]") {
  // The admin WreapDeadIpcs periodic reclaims the segments of a client that
  // died, after the grace period (runtime.client_reap_grace_s). A short grace
  // keeps the test quick.
  clio::run::test::SetEnvVar("CTP_LOG_LEVEL", "info");
  clio::run::test::SetEnvVar("CLIO_PORT", std::to_string(kPort));
  const fs::path work =
      fs::temp_directory_path() / "clio_client_reap_zombie_test";
  fs::remove_all(work);
  fs::create_directories(work);
  const fs::path conf = work / "clio.yaml";
  { std::ofstream(conf) << "runtime:\n  client_reap_grace_s: 2\n"; }
  clio::run::test::SetEnvVar("CLIO_SERVER_CONF", conf.string());
  const fs::path compose_yaml = work / "compose.yaml";
  {
    std::ofstream f(compose_yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: cte_client_reap\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << (work / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 256mb\n"
         "    dpe:\n"
         "      dpe_type: random\n";
  }
  clio::run::test::RuntimeServer server;
  // Progress lines: this case has hung in CI (leak-check job) with no
  // output before the ctest TIMEOUT; each step reports so the next hang
  // shows where it stopped.
  // Unbuffered stdout for this case: a FAIL message written to a buffered
  // stdout is lost when ctest kills a process that then hangs in teardown.
  std::cout << std::unitbuf;
  REQUIRE(server.Start(kPort, "127.0.0.1", /*ephemeral=*/true));
  Progress("reclaim: daemon started");
  const bool ready = server.WaitForReady(60000);
  Progress(std::string("reclaim: WaitForReady returned ") +
           (ready ? "true" : "false") + ", daemon " +
           (server.IsRunning() ? "running" : "not running"));
  if (!ready) {
    FAIL("daemon not ready within 60 s\n" +
         clio::run::test::RuntimeServer::LogTail());
  }
  REQUIRE(RunCliTimed({"compose", "start", compose_yaml.string()}, 60) == 0);
  Progress("reclaim: compose done");
  pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    CrashClientMain(/*quiet=*/true);  // allocates SHM, PutBlobs, dies
  }
  // Deliberately do NOT wait for the child yet: until we do it is a zombie,
  // which still answers kill(pid, 0) although it has released every fd. The
  // reaper must see through that (/proc/<pid>/stat state 'Z') and reclaim
  // the segments anyway -- a parent that never waits is the common way a
  // dead client lingers. The child is reaped (waitpid) at the end.
  // The client's segments are memfds with a symlink clio_<pid>_<idx> in the
  // per-user memfd dir; reaping unmaps them and removes the symlinks.
  const std::string dir = ctp::SystemInfo::GetMemfdDir();
  const std::string prefix = "clio_" + std::to_string(child) + "_";
  auto segments_left = [&]() {
    size_t n = 0;
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(dir, ec)) {
      if (e.path().filename().string().rfind(prefix, 0) == 0) ++n;
    }
    return n;
  };
  Progress("reclaim: forked crash client " + std::to_string(child));
  size_t left = segments_left();
  INFO("segments of the dead client right after its death: " << left);
  for (int i = 0; i < 400 && left > 0; ++i) {  // grace 2 s + reaper 1 s period
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    left = segments_left();
  }
  Progress("reclaim: segments left " + std::to_string(left));
  REQUIRE(left == 0);
  REQUIRE(server.IsRunning());  // and the daemon survived the reap
  int status = 0;
  WaitChild(child, &status);
  Progress("reclaim: crash client reaped");
  server.Stop();
  for (int i = 0; i < 50 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  clio::run::test::UnsetEnvVar("CLIO_SERVER_CONF");
}

// Custom main: `crash-client` runs the standalone client (distributed suite);
// otherwise run the single-node robustness test(s).
int main(int argc, char *argv[]) {
  if (argc > 1 && std::string(argv[1]) == "crash-client") {
    CrashClientMain(/*quiet=*/false);  // never returns
  }
  int result = SimpleTest::run_all_tests(argc > 1 ? argv[1] : "");
  SIMPLE_TEST_PROCESS_EXIT(result);
  if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize();
  return result;
}
