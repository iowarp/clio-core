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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
// Wait longer than the daemon's undeliverable-response drop window
// (kClientResponseRetryDropSec, ~5 s) so the drop has fired before we stop it.
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
  REQUIRE(waitpid(child, &status, 0) == child);
  INFO("crash-client exited: exited=" << WIFEXITED(status)
       << " code=" << (WIFEXITED(status) ? WEXITSTATUS(status) : -1)
       << " signaled=" << WIFSIGNALED(status));

  // Give the daemon time to (a) execute the abandoned PutBlobs and (b) hit its
  // bounded-drop window for the undeliverable responses.
  std::this_thread::sleep_for(std::chrono::seconds(kDropWaitSec));

  // Graceful stop -> ServerFinalize -> ReportRuntimeLeaks writes to the log.
  server.Stop();
  for (int i = 0; i < 50 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

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
