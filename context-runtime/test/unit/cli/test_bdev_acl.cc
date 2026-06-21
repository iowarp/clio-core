/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Per-RPC access-control test. A compose file marks bdev containers (or
 * specific RPCs) private; an external user client's direct call to a private
 * operation must be rejected with EACCES, while the same call to a public
 * container/RPC succeeds. This is what stops users from writing directly to a
 * block device that a filesystem owns.
 *
 * Three composed bdev pools (IDs chosen to avoid the daemon's default-config
 * pools — the bundled compose already owns bdev 301.0):
 *   701.0  container_visibility: private   -> AllocateBlocks AND Write rejected
 *   702.0  (public, control)               -> AllocateBlocks AND Write succeed
 *   703.0  container_rpc_acl: {Write:private} -> AllocateBlocks ok, Write rejected
 */
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
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

chi::priv::vector<clio::run::bdev::Block> WrapBlock(
    const clio::run::bdev::Block& block) {
  chi::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
  blocks.push_back(block);
  return blocks;
}
}  // namespace

TEST_CASE("Bdev - per-RPC access control rejects external private calls",
          "[cli][bdev][acl]") {
  constexpr unsigned kPort = 10606;
  const fs::path work = fs::temp_directory_path() / "bdev_acl_test";
  fs::remove_all(work);
  fs::create_directories(work);

  const fs::path yaml = work / "compose.yaml";
  {
    std::ofstream f(yaml);
    f << "compose:\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"ram::acl_private\"\n"
         "    pool_query: local\n"
         "    pool_id: \"701.0\"\n"
         "    bdev_type: ram\n"
         "    capacity: \"64mb\"\n"
         "    container_visibility: private\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"ram::acl_public\"\n"
         "    pool_query: local\n"
         "    pool_id: \"702.0\"\n"
         "    bdev_type: ram\n"
         "    capacity: \"64mb\"\n"
         "  - mod_name: clio_bdev\n"
         "    pool_name: \"ram::acl_wronly\"\n"
         "    pool_query: local\n"
         "    pool_id: \"703.0\"\n"
         "    bdev_type: ram\n"
         "    capacity: \"64mb\"\n"
         "    container_rpc_acl:\n"
         "      Write: private\n";
  }

  setenv("CLIO_WAIT_SERVER", "15", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", "start", yaml.string()}, 60) == 0);

  REQUIRE(chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false));
  auto* ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  const auto local = chi::PoolQuery::Local();
  constexpr chi::u64 kN = 4096;

  clio::run::bdev::Client priv(chi::PoolId(701, 0));
  clio::run::bdev::Client pub(chi::PoolId(702, 0));
  clio::run::bdev::Client wronly(chi::PoolId(703, 0));

  auto write_buf = [&](const chi::priv::vector<clio::run::bdev::Block>& blocks,
                       clio::run::bdev::Client& c) {
    ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(kN);
    memset(wb.ptr_, 'z', kN);
    auto w = c.AsyncWrite(local, blocks, wb.shm_.template Cast<void>(), kN);
    w.Wait();
    int rc = static_cast<int>(w->GetReturnCode());
    ipc->FreeBuffer(wb);
    return rc;
  };

  // ---- Control: public pool 702 — allocate + write both succeed. ----
  clio::run::bdev::Block pub_block;
  {
    auto a = pub.AsyncAllocateBlocks(local, kN);
    a.Wait();
    REQUIRE(a->GetReturnCode() == 0);
    REQUIRE_FALSE(a->blocks_.empty());
    pub_block = a->blocks_[0];
    REQUIRE(write_buf(WrapBlock(pub_block), pub) == 0);
  }

  // ---- Private container 701 — every RPC rejected for the external client. ----
  {
    auto a = priv.AsyncAllocateBlocks(local, kN);
    a.Wait();
    REQUIRE(static_cast<int>(a->GetReturnCode()) == EACCES);  // allocate private
    // Write (reusing a valid block) is rejected before the handler runs.
    REQUIRE(write_buf(WrapBlock(pub_block), priv) == EACCES);
  }

  // ---- Per-RPC override 703 — AllocateBlocks public, Write private. ----
  {
    auto a = wronly.AsyncAllocateBlocks(local, kN);
    a.Wait();
    REQUIRE(a->GetReturnCode() == 0);  // AllocateBlocks stays public
    REQUIRE_FALSE(a->blocks_.empty());
    REQUIRE(write_buf(WrapBlock(a->blocks_[0]), wronly) == EACCES);  // Write private
  }

  RunCliTimed({"stop", "--grace-period", "2000"}, 90);
  for (int i = 0; i < 200 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.Stop();
  fs::remove_all(work);
}

SIMPLE_TEST_MAIN()
