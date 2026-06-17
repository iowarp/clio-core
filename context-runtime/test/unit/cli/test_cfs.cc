/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Filesystem chimod integration test. Composes bdev -> cte_core -> filesystem
 * and drives the filesystem client: Open, Write 1 MiB of '5', Getattr (size
 * must be EXACT 1 MiB — the whole point vs the libfuse adapter's
 * physical-size over-report), Read back and verify, then Truncate.
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
#include <clio_cte/filesystem/filesystem_client.h>
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
}  // namespace

TEST_CASE("Cfs - filesystem chimod open/write/getattr/read/truncate",
          "[cli][cfs]") {
  constexpr unsigned kPort = 10604;
  const fs::path work = fs::temp_directory_path() / "cfs_test";
  fs::remove_all(work);
  fs::create_directories(work);

  const fs::path yaml = work / "compose.yaml";
  {
    std::ofstream f(yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cfs_cte\"\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << (work / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 64mb\n"
         "    dpe:\n"
         "      dpe_type: random\n"
         "  - mod_name: clio_cte_filesystem\n"
         "    pool_name: \"cfs\"\n"
         "    pool_query: local\n"
         "    pool_id: \"600.0\"\n"
         "    next_pool_id: \"512.0\"\n";
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

  clio::cte::filesystem::Client cfs;
  cfs.Init(chi::PoolId(600, 0));

  constexpr chi::u64 kSize = 1024 * 1024;  // 1 MiB
  const std::string path = "clio::/cfs_content.bin";

  // Open (create).
  auto open = cfs.AsyncOpen(path, O_CREAT | O_RDWR, 0644);
  open.Wait();
  fprintf(stderr, "[cfs-test] open rc=%u handle=%llu size=%llu created=%u\n",
          open->GetReturnCode(), (unsigned long long)open->handle_,
          (unsigned long long)open->size_, open->created_);
  REQUIRE(open->GetReturnCode() == 0);
  chi::u64 handle = open->handle_;

  // Write 1 MiB of '5'.
  ctp::ipc::FullPtr<char> wbuf = ipc->AllocateBuffer(kSize);
  REQUIRE(!wbuf.IsNull());
  memset(wbuf.ptr_, '5', kSize);
  auto w = cfs.AsyncWrite(handle, 0, kSize, wbuf.shm_.template Cast<void>());
  w.Wait();
  fprintf(stderr, "[cfs-test] write rc=%u bytes_written=%llu new_size=%llu\n",
          w->GetReturnCode(),
          (unsigned long long)w->bytes_written_,
          (unsigned long long)w->new_size_);
  REQUIRE(w->GetReturnCode() == 0);
  REQUIRE(w->bytes_written_ == kSize);
  REQUIRE(w->new_size_ == kSize);
  ipc->FreeBuffer(wbuf);

  // Getattr — size must be EXACTLY 1 MiB (exact logical size).
  auto ga = cfs.AsyncGetattr(path);
  ga.Wait();
  REQUIRE(ga->GetReturnCode() == 0);
  REQUIRE(ga->exists_ == 1);
  REQUIRE(ga->size_ == kSize);

  // Read back and verify every byte is '5'.
  ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kSize);
  REQUIRE(!rbuf.IsNull());
  memset(rbuf.ptr_, 0, kSize);
  auto r = cfs.AsyncRead(handle, 0, kSize, rbuf.shm_.template Cast<void>());
  r.Wait();
  REQUIRE(r->GetReturnCode() == 0);
  REQUIRE(r->bytes_read_ == kSize);
  bool all_five = true;
  for (chi::u64 i = 0; i < kSize; ++i) {
    if (rbuf.ptr_[i] != '5') { all_five = false; break; }
  }
  REQUIRE(all_five);
  ipc->FreeBuffer(rbuf);

  // Truncate to 4 KiB; getattr must reflect it exactly.
  auto tr = cfs.AsyncTruncate(path, 4096);
  tr.Wait();
  REQUIRE(tr->GetReturnCode() == 0);
  auto ga2 = cfs.AsyncGetattr(path);
  ga2.Wait();
  REQUIRE(ga2->GetReturnCode() == 0);
  REQUIRE(ga2->size_ == 4096);

  auto cl = cfs.AsyncClose(handle);
  cl.Wait();

  // ---- GetOrCreateTagAlias (tag-level hard link) ----
  // Talk to the cte_core pool (512.0) directly with the core client.
  clio::cte::core::Client core;
  core.Init(chi::PoolId(512, 0));

  // Create a fresh tag and write a known blob to it.
  const std::string kOrig = "alias_orig_tag";
  auto mk = core.AsyncGetOrCreateTag(kOrig, clio::cte::core::TagId::GetNull(),
                                     chi::PoolQuery::Local());
  mk.Wait();
  REQUIRE(mk->GetReturnCode() == 0);
  clio::cte::core::TagId orig_id = mk->tag_id_;
  REQUIRE(!orig_id.IsNull());

  const char kMsg[] = "hello-alias-payload";
  constexpr chi::u64 kMsgN = sizeof(kMsg);  // includes NUL
  ctp::ipc::FullPtr<char> pbuf = ipc->AllocateBuffer(kMsgN);
  REQUIRE(!pbuf.IsNull());
  memcpy(pbuf.ptr_, kMsg, kMsgN);
  auto pb = core.AsyncPutBlob(orig_id, "0", 0, kMsgN,
                              pbuf.shm_.template Cast<void>(), -1.0f,
                              clio::cte::core::Context(), 0u,
                              chi::PoolQuery::Local());
  pb.Wait();
  REQUIRE(pb->GetReturnCode() == 0);
  ipc->FreeBuffer(pbuf);

  // Alias an EXISTING tag by name -> found_ == 1 and shares the same TagId.
  const std::string kAlias = "alias_link_name";
  auto al = core.AsyncGetOrCreateTagAlias(kOrig, kAlias);
  al.Wait();
  REQUIRE(al->GetReturnCode() == 0);
  REQUIRE(al->found_ == 1);
  REQUIRE(al->tag_id_ == orig_id);

  // Resolving the alias name must yield the SAME TagId (hard link).
  auto rs = core.AsyncGetOrCreateTag(kAlias, clio::cte::core::TagId::GetNull(),
                                     chi::PoolQuery::Local());
  rs.Wait();
  REQUIRE(rs->GetReturnCode() == 0);
  REQUIRE(rs->tag_id_ == orig_id);

  // The alias must expose the original's blob (shared storage).
  ctp::ipc::FullPtr<char> gbuf = ipc->AllocateBuffer(kMsgN);
  REQUIRE(!gbuf.IsNull());
  memset(gbuf.ptr_, 0, kMsgN);
  auto gb = core.AsyncGetBlob(rs->tag_id_, "0", 0, kMsgN, 0u,
                              gbuf.shm_.template Cast<void>(),
                              chi::PoolQuery::Local());
  gb.Wait();
  REQUIRE(gb->GetReturnCode() == 0);
  REQUIRE(memcmp(gbuf.ptr_, kMsg, kMsgN) == 0);
  ipc->FreeBuffer(gbuf);

  // Aliasing a NON-existent tag must report found_ == 0 (error, no binding).
  auto miss = core.AsyncGetOrCreateTagAlias(std::string("no_such_tag_xyz"),
                                            std::string("alias_to_missing"));
  miss.Wait();
  REQUIRE(miss->found_ == 0);

  // ---- Alias unlink: deleting an alias name leaves the tag intact ----
  const std::string kAlias2 = "alias_link_name2";
  auto al2 = core.AsyncGetOrCreateTagAlias(orig_id, kAlias2);
  al2.Wait();
  REQUIRE(al2->found_ == 1);

  auto unlink = core.AsyncDelTag(kAlias2, chi::PoolQuery::Local());
  unlink.Wait();
  REQUIRE(unlink->GetReturnCode() == 0);

  // Original blob still readable after unlinking just one alias.
  ctp::ipc::FullPtr<char> ubuf = ipc->AllocateBuffer(kMsgN);
  REQUIRE(!ubuf.IsNull());
  memset(ubuf.ptr_, 0, kMsgN);
  auto ug = core.AsyncGetBlob(orig_id, "0", 0, kMsgN, 0u,
                              ubuf.shm_.template Cast<void>(),
                              chi::PoolQuery::Local());
  ug.Wait();
  REQUIRE(ug->GetReturnCode() == 0);
  REQUIRE(memcmp(ubuf.ptr_, kMsg, kMsgN) == 0);
  ipc->FreeBuffer(ubuf);

  // The first alias must still resolve to the SAME id.
  auto still = core.AsyncGetOrCreateTag(
      kAlias, clio::cte::core::TagId::GetNull(), chi::PoolQuery::Local());
  still.Wait();
  REQUIRE(still->tag_id_ == orig_id);

  // ---- Cascade delete: deleting the canonical tag removes all aliases ----
  auto del = core.AsyncDelTag(kOrig, chi::PoolQuery::Local());
  del.Wait();
  REQUIRE(del->GetReturnCode() == 0);

  // Blob is gone (whole tag deleted).
  ctp::ipc::FullPtr<char> dbuf = ipc->AllocateBuffer(kMsgN);
  REQUIRE(!dbuf.IsNull());
  auto dg = core.AsyncGetBlob(orig_id, "0", 0, kMsgN, 0u,
                              dbuf.shm_.template Cast<void>(),
                              chi::PoolQuery::Local());
  dg.Wait();
  REQUIRE(dg->GetReturnCode() != 0);
  ipc->FreeBuffer(dbuf);

  // The surviving alias was cascade-removed: re-resolving its name now mints
  // a fresh, DIFFERENT tag id (the old binding to orig_id is gone).
  auto gone = core.AsyncGetOrCreateTag(
      kAlias, clio::cte::core::TagId::GetNull(), chi::PoolQuery::Local());
  gone.Wait();
  REQUIRE(gone->GetReturnCode() == 0);
  REQUIRE(!(gone->tag_id_ == orig_id));

  RunCliTimed({"stop", "--grace-period", "2000"}, 90);
  for (int i = 0; i < 200 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.Stop();
  fs::remove_all(work);
}

SIMPLE_TEST_MAIN()
