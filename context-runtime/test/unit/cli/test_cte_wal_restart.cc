/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CTE write-ahead-log restart test.
 *
 * Drives the full WAL persistence cycle that only daemon restarts reach:
 *   1. start a daemon, compose a CTE pool with restart:true and a
 *      metadata_log_path,
 *   2. connect as a client and create tags / put / delete blobs (writes WAL
 *      entries), flush a metadata snapshot, then write more entries on top,
 *   3. stop the daemon cleanly,
 *   4. start a fresh daemon and re-compose the same pool: the CTE Create
 *      handler takes the restart path — RestoreMetadataFromLog() reads the
 *      snapshot and ReplayTransactionLogs() re-applies the trailing WAL.
 */

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

/** Run the clio_run binary with a hard kill deadline (same as the CLI test). */
int RunCliTimed(const std::vector<std::string>& args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char*> argv;
  argv.reserve(full.size() + 1);
  for (auto& a : full) {
    argv.push_back(a.data());
  }
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) {
    return -1;
  }
  if (pid == 0) {
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, 1);
      dup2(devnull, 2);
      close(devnull);
    }
    execv(argv[0], argv.data());
    _exit(127);
  }

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) {
      if (WIFEXITED(status)) return WEXITSTATUS(status);
      return -2;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -3;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void WaitForExit(clio::run::test::RuntimeServer& server) {
  // Coverage builds flush .gcda on daemon exit, which can take a while.
  for (int i = 0; i < 600 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

}  // namespace

TEST_CASE("CteWalRestart - WAL snapshot, replay on daemon restart",
          "[cli][cte][wal]") {
  constexpr unsigned kPort = 10602;
  const fs::path work_dir = fs::temp_directory_path() / "cte_wal_restart";

  // Fresh WAL/snapshot directory: replay must see only this run's files.
  fs::remove_all(work_dir);
  fs::create_directories(work_dir);

  const fs::path compose_yaml = work_dir / "compose.yaml";
  {
    std::ofstream f(compose_yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cte_wal_test\"\n"
         "    pool_query: local\n"
         "    pool_id: \"700.0\"\n"
         "    restart: true\n"
         "    storage:\n"
         "      - path: " << (work_dir / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 64mb\n"
         "    dpe:\n"
         "      dpe_type: random\n"
         "    performance:\n"
         "      metadata_log_path: " << (work_dir / "meta_log").string()
      << "\n";
  }

  setenv("CLIO_WAIT_SERVER", "15", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);
  // Isolate the restart-log WAL inside work_dir so `compose` registration does
  // not pollute the shared ~/.clio/restart_log.bin. Child clio_run processes
  // (daemons + CLI) inherit this env var, so they all agree on the path, and
  // it is removed with work_dir at the end of the test.
  const fs::path restart_log = work_dir / "restart_log.bin";
  setenv("CLIO_RESTART_LOG", restart_log.string().c_str(), 1);

  // --- Phase 1: daemon up, compose the pool, write data through a client.
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());

  REQUIRE(RunCliTimed({"compose", compose_yaml.string()}, 60) == 0);

  REQUIRE(chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  auto* cte = CLIO_CTE_CLIENT;
  cte->Init(chi::PoolId(700, 0));

  auto* ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  auto put_blob = [&](const clio::cte::core::TagId& tag_id,
                      const std::string& name, char fill) {
    constexpr size_t kSize = 4096;
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kSize);
    REQUIRE(!buf.IsNull());
    memset(buf.ptr_, fill, kSize);
    ctp::ipc::ShmPtr<> shm_ref(buf.shm_);
    auto put = cte->AsyncPutBlob(tag_id, name, 0, kSize, shm_ref, 1.0f);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
    ipc->FreeBuffer(buf);
  };

  // Tag with two blobs; delete one (kCreateTag/kCreateNewBlob/kDelBlob WAL).
  auto tag_a = cte->AsyncGetOrCreateTag("wal_tag_a");
  tag_a.Wait();
  REQUIRE(tag_a->GetReturnCode() == 0);
  put_blob(tag_a->tag_id_, "blob_keep", 'k');
  put_blob(tag_a->tag_id_, "blob_drop", 'd');
  auto del_blob = cte->AsyncDelBlob(tag_a->tag_id_, "blob_drop");
  del_blob.Wait();
  REQUIRE(del_blob->GetReturnCode() == 0);

  // Tag created then deleted entirely (kDelTag WAL entry).
  auto tag_b = cte->AsyncGetOrCreateTag("wal_tag_b");
  tag_b.Wait();
  REQUIRE(tag_b->GetReturnCode() == 0);
  auto del_tag = cte->AsyncDelTag("wal_tag_b");
  del_tag.Wait();

  // Snapshot the metadata, then add entries AFTER the snapshot so the
  // restart exercises BOTH RestoreMetadataFromLog and ReplayTransactionLogs.
  auto flush_meta = cte->AsyncFlushMetadata(chi::PoolQuery::Local(), 0);
  flush_meta.Wait();
  auto flush_data = cte->AsyncFlushData(chi::PoolQuery::Local(), 0, 0);
  flush_data.Wait();

  auto tag_c = cte->AsyncGetOrCreateTag("wal_tag_post_snapshot");
  tag_c.Wait();
  REQUIRE(tag_c->GetReturnCode() == 0);
  put_blob(tag_c->tag_id_, "blob_post_snapshot", 'p');

  // --- Phase 2: clean stop.
  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(server);
  REQUIRE_FALSE(server.IsRunning());
  server.Stop();

  // --- Phase 3: fresh daemon; re-compose triggers restart + WAL replay.
  clio::run::test::RuntimeServer server2;
  REQUIRE(server2.Start(kPort));
  REQUIRE(server2.WaitForReady());

  REQUIRE(RunCliTimed({"compose", compose_yaml.string()}, 60) == 0);

  // --- Phase 4: shut down and clean up.
  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(server2);
  server2.Stop();

  if (std::getenv("CTE_WAL_TEST_KEEP") == nullptr) {
    fs::remove_all(work_dir);
  }
}

SIMPLE_TEST_MAIN()
