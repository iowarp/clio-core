/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Fallback-runtime (crash-isolation) end-to-end test.
 *
 * Topology: two runtimes on one node.
 *   - MAIN runtime (port 10650): composes a ram bdev at pool 711.0.
 *   - USER runtime (port 10651): started with CLIO_FALLBACK_PORT=10650, so it
 *     punts tasks for pools it does not own to MAIN. It does NOT own 711.
 *
 * An external SHM client connects to the USER runtime and calls the bdev at
 * 711 (AllocateBlocks + Write). The USER runtime has no container for 711, so
 * it punts the task to MAIN via IpcRun2Fallback. MAIN owns 711, runs the task,
 * and completes the client's (shared) FutureShm in place — the client, polling
 * that FutureShm, sees the result directly without any relay back through USER.
 *
 * The client must be SHM mode so its FutureShm + data live in a shared client
 * segment that MAIN registers (dual RegisterMemory) and can complete in place.
 * Pool 711 is chosen so it is not part of the default bundled compose (which
 * owns 301), guaranteeing the USER runtime genuinely lacks it.
 */
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>

#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {
int RunCliTimed(const std::vector<std::string>& args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& a : full) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    int n = open("/dev/null", O_WRONLY);
    if (n >= 0) { dup2(n, 1); dup2(n, 2); close(n); }
    execv(argv[0], argv.data());
    _exit(127);
  }
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL); waitpid(pid, &status, 0); return -3;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

clio::run::priv::vector<clio::run::bdev::Block> WrapBlock(
    const clio::run::bdev::Block& block) {
  clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
  blocks.push_back(block);
  return blocks;
}

void SetEnvStr(const char* k, const std::string& v) { setenv(k, v.c_str(), 1); }
}  // namespace

TEST_CASE("Fallback runtime - external client task punted to main runtime",
          "[cli][bdev][fallback]") {
  // Each runtime binds port P (main), P+1 (local server) and P+3 (control), so
  // the two runtimes must be spaced by more than 3 to avoid colliding.
  constexpr unsigned kMainPort = 10650;
  constexpr unsigned kUserPort = 10660;
  const fs::path work = fs::temp_directory_path() / "fallback_runtime_test";
  fs::remove_all(work);
  fs::create_directories(work);

  // Compose file for MAIN: a ram bdev at pool 711 (not in the default compose).
  const fs::path main_yaml = work / "main_compose.yaml";
  {
    std::ofstream f(main_yaml);
    f << "compose:\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"ram::fallback_bdev\"\n"
         "    pool_query: local\n"
         "    pool_id: \"711.0\"\n"
         "    bdev_type: ram\n"
         "    capacity: \"64mb\"\n";
  }

  // Compose file for USER: the SAME bdev pool 711, but marked pool_external —
  // the real container lives on MAIN. The user runtime creates a stub so 711
  // resolves locally, and punts its tasks to MAIN.
  const fs::path user_yaml = work / "user_compose.yaml";
  {
    std::ofstream f(user_yaml);
    f << "compose:\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"ram::fallback_bdev_stub\"\n"
         "    pool_query: local\n"
         "    pool_id: \"711.0\"\n"
         "    bdev_type: ram\n"
         "    capacity: \"64mb\"\n"
         "    pool_external: true\n";
  }

  setenv("CLIO_WAIT_SERVER", "20", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

  // --- 1. Start the MAIN runtime (no fallback) and compose bdev 711 on it. ---
  SetEnvStr("CLIO_TEST_SERVER_LOG", "/tmp/fb_main_runtime.log");
  unsetenv("CLIO_FALLBACK_PORT");
  clio::run::test::RuntimeServer main_server;
  REQUIRE(main_server.Start(kMainPort));
  REQUIRE(main_server.WaitForReady());
  // compose start targets CLIO_PORT (= kMainPort, set by Start) over the
  // default control path. Do this before CLIO_IPC_MODE is pinned to SHM.
  REQUIRE(RunCliTimed({"compose", "start", main_yaml.string()}, 60) == 0);

  // --- 2. Start the USER runtime, pointing its fallback at MAIN. ---
  SetEnvStr("CLIO_TEST_SERVER_LOG", "/tmp/fb_user_runtime.log");
  SetEnvStr("CLIO_FALLBACK_PORT", std::to_string(kMainPort));
  clio::run::test::RuntimeServer user_server;
  REQUIRE(user_server.Start(kUserPort));
  REQUIRE(user_server.WaitForReady());
  // Compose the external stub for 711 on the USER runtime (targets CLIO_PORT =
  // kUserPort). Done before CLIO_IPC_MODE is pinned to SHM below.
  SetEnvStr("CLIO_PORT", std::to_string(kUserPort));
  REQUIRE(RunCliTimed({"compose", "start", user_yaml.string()}, 60) == 0);
  // Do not leak the fallback port to the in-process client.
  unsetenv("CLIO_FALLBACK_PORT");

  // --- 3. Connect an SHM client to the USER runtime. ---
  SetEnvStr("CLIO_PORT", std::to_string(kUserPort));
  setenv("CLIO_IPC_MODE", "SHM", 1);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  auto* ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  const auto local = clio::run::PoolQuery::Local();
  constexpr clio::run::u64 kN = 4096;

  // Client bound to bdev pool 711 — which lives only on MAIN.
  clio::run::bdev::Client dev(clio::run::PoolId(711, 0));

  // --- 4. AllocateBlocks: punted USER -> MAIN, completed in place. ---
  clio::run::bdev::Block block;
  {
    auto a = dev.AsyncAllocateBlocks(local, kN);
    a.Wait();
    REQUIRE(a->GetReturnCode() == 0);
    REQUIRE_FALSE(a->blocks_.empty());
    block = a->blocks_[0];
  }

  // --- 5. Write to the allocated block: also punted and completed by MAIN. ---
  {
    ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(kN);
    memset(wb.ptr_, 'q', kN);
    auto w = dev.AsyncWrite(local, WrapBlock(block),
                            wb.shm_.template Cast<void>(), kN);
    w.Wait();
    REQUIRE(w->GetReturnCode() == 0);
    REQUIRE(w->bytes_written_ == kN);
    ipc->FreeBuffer(wb);
  }

  // --- 6. Teardown: stop USER then MAIN. ---
  SetEnvStr("CLIO_PORT", std::to_string(kUserPort));
  RunCliTimed({"stop", "--grace-period", "2000"}, 60);
  SetEnvStr("CLIO_PORT", std::to_string(kMainPort));
  RunCliTimed({"stop", "--grace-period", "2000"}, 60);
  for (int i = 0; i < 200 && (user_server.IsRunning() || main_server.IsRunning());
       ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  user_server.Stop();
  main_server.Stop();
  fs::remove_all(work);
}

SIMPLE_TEST_MAIN()
