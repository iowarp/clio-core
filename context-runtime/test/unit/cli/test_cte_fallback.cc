/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Fallback-runtime CTE test: CTE on the USER runtime backed by a bdev on the
 * MAIN runtime (crash isolation — buggy app CTE isolated from stable storage).
 *
 * Topology (two runtimes, one node):
 *   - MAIN (port 10670): hosts the real bdev at pool (512,1) — the id CTE
 *     derives for its first storage target (512 + device_idx, 1 + node).
 *   - USER (port 10680, CLIO_FALLBACK_PORT=10670): composes CTE core (512,0)
 *     plus an EXTERNAL stub for bdev (512,1) (pool_external: true). CTE's
 *     RegisterTarget reuses that stub; every CTE->bdev call (GetStats during
 *     Create, then AllocateBlocks/Write/Read) is a runtime-internal subtask
 *     that gets serialize-relayed to MAIN and completed locally.
 *
 * The default compose is disabled on both runtimes (CLIO_SERVER_CONF -> a
 * no-compose config) so the only CTE/bdev pools are the ones composed here.
 *
 * A client connects to the USER runtime, writes a blob through CTE, and reads
 * it back — proving the data round-trips to MAIN's bdev and back.
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
#include <clio_cte/core/core_client.h>

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

void SetEnvStr(const char* k, const std::string& v) { setenv(k, v.c_str(), 1); }
}  // namespace

TEST_CASE("Fallback runtime - CTE on user backed by bdev on main",
          "[cli][cte][fallback]") {
  constexpr unsigned kMainPort = 10670;
  constexpr unsigned kUserPort = 10680;
  const fs::path work = fs::temp_directory_path() / "cte_fallback_test";
  fs::remove_all(work);
  fs::create_directories(work);

  // Both runtimes start --ephemeral (no default compose) so the only CTE/bdev
  // pools are the ones composed below; otherwise the default CTE would create
  // bdev (512,1) locally and collide with our external stub.

  // MAIN: the real bdev at (512,1) — the id CTE derives for its first target.
  const fs::path main_yaml = work / "main_compose.yaml";
  {
    std::ofstream f(main_yaml);
    f << "compose:\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"ram::cte_backing\"\n"
         "    pool_query: local\n"
         "    pool_id: \"512.1\"\n"
         "    bdev_type: ram\n"
         "    capacity: \"64mb\"\n";
  }

  // USER: CTE core (512,0) with one ram storage device + an EXTERNAL stub for
  // bdev (512,1). CTE reuses the stub as its target; its bdev I/O punts to MAIN.
  const fs::path user_yaml = work / "user_compose.yaml";
  {
    std::ofstream f(user_yaml);
    f << "compose:\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"ram::cte_backing_stub\"\n"
         "    pool_query: local\n"
         "    pool_id: \"512.1\"\n"
         "    bdev_type: ram\n"
         "    capacity: \"64mb\"\n"
         "    pool_external: true\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cte_user\"\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << (work / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 64mb\n"
         "    dpe:\n"
         "      dpe_type: random\n";
  }

  setenv("CLIO_WAIT_SERVER", "20", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

  // --- 1. MAIN runtime + real bdev (512,1). ---
  SetEnvStr("CLIO_TEST_SERVER_LOG", "/tmp/cte_fb_main.log");
  unsetenv("CLIO_FALLBACK_PORT");
  clio::run::test::RuntimeServer main_server;
  REQUIRE(main_server.Start(kMainPort, "127.0.0.1", /*ephemeral=*/true));
  REQUIRE(main_server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", "start", main_yaml.string()}, 60) == 0);

  // --- 2. USER runtime (fallback -> MAIN) + CTE + external bdev stub. ---
  SetEnvStr("CLIO_TEST_SERVER_LOG", "/tmp/cte_fb_user.log");
  SetEnvStr("CLIO_FALLBACK_PORT", std::to_string(kMainPort));
  clio::run::test::RuntimeServer user_server;
  REQUIRE(user_server.Start(kUserPort, "127.0.0.1", /*ephemeral=*/true));
  REQUIRE(user_server.WaitForReady());
  SetEnvStr("CLIO_PORT", std::to_string(kUserPort));
  REQUIRE(RunCliTimed({"compose", "start", user_yaml.string()}, 60) == 0);
  unsetenv("CLIO_FALLBACK_PORT");

  // --- 3. Client -> USER runtime, drive CTE. ---
  SetEnvStr("CLIO_PORT", std::to_string(kUserPort));
  setenv("CLIO_IPC_MODE", "SHM", 1);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  auto* ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  clio::cte::core::Client core;
  core.Init(clio::run::PoolId(512, 0));

  // Create a tag.
  auto mk = core.AsyncGetOrCreateTag("cte_fb_tag", clio::cte::core::TagId::GetNull(),
                                     clio::run::PoolQuery::Local());
  mk.Wait();
  REQUIRE(mk->GetReturnCode() == 0);
  clio::cte::core::TagId tag = mk->tag_id_;
  REQUIRE(!tag.IsNull());

  // PutBlob: the blob bytes are written through CTE to MAIN's bdev (relayed).
  const char kMsg[] = "cte-over-fallback-bdev-payload";
  constexpr clio::run::u64 kN = sizeof(kMsg);  // includes NUL
  ctp::ipc::FullPtr<char> pbuf = ipc->AllocateBuffer(kN);
  REQUIRE(!pbuf.IsNull());
  memcpy(pbuf.ptr_, kMsg, kN);
  auto pb = core.AsyncPutBlob(tag, "0", 0, kN, pbuf.shm_.template Cast<void>(),
                              -1.0f, clio::cte::core::Context(), 0u,
                              clio::run::PoolQuery::Local());
  pb.Wait();
  REQUIRE(pb->GetReturnCode() == 0);
  ipc->FreeBuffer(pbuf);

  // GetBlob: read it back (relayed read from MAIN's bdev) and verify.
  ctp::ipc::FullPtr<char> gbuf = ipc->AllocateBuffer(kN);
  REQUIRE(!gbuf.IsNull());
  memset(gbuf.ptr_, 0, kN);
  auto gb = core.AsyncGetBlob(tag, "0", 0, kN, 0u,
                              gbuf.shm_.template Cast<void>(),
                              clio::run::PoolQuery::Local());
  gb.Wait();
  REQUIRE(gb->GetReturnCode() == 0);
  REQUIRE(memcmp(gbuf.ptr_, kMsg, kN) == 0);
  ipc->FreeBuffer(gbuf);

  // --- 4. Teardown: USER then MAIN. ---
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
