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
#include <set>
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

  // ---- Hierarchical tag names + O(1) directory move ----
  using clio::cte::core::TagId;
  auto resolve = [&](const TagId &id) {
    auto g = core.AsyncGetTagName(id);
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(g->found_ == 1);
    return g->tag_name_.str();
  };

  // Creating "/a/b/c/d" builds the whole chain; the returned id is the leaf.
  auto deep = core.AsyncGetOrCreateTag(
      "/a/b/c/d", TagId::GetNull(), chi::PoolQuery::Local());
  deep.Wait();
  REQUIRE(deep->GetReturnCode() == 0);
  TagId d_id = deep->tag_id_;

  // The intermediate "/a/b" already exists from the chain — same id, no dup.
  auto mid = core.AsyncGetOrCreateTag(
      "/a/b", TagId::GetNull(), chi::PoolQuery::Local());
  mid.Wait();
  REQUIRE(mid->GetReturnCode() == 0);
  TagId b_id = mid->tag_id_;
  REQUIRE(!b_id.IsNull());
  REQUIRE(!(b_id == d_id));

  // Names resolve to absolute paths.
  REQUIRE(resolve(d_id) == "/a/b/c/d");
  REQUIRE(resolve(b_id) == "/a/b");

  // Write a blob under the deepest tag so we can prove data survives the move.
  const char kDeepMsg[] = "deep-payload";
  constexpr chi::u64 kDeepN = sizeof(kDeepMsg);
  ctp::ipc::FullPtr<char> hpb = ipc->AllocateBuffer(kDeepN);
  REQUIRE(!hpb.IsNull());
  memcpy(hpb.ptr_, kDeepMsg, kDeepN);
  auto hp = core.AsyncPutBlob(d_id, "0", 0, kDeepN,
                              hpb.shm_.template Cast<void>(), -1.0f,
                              clio::cte::core::Context(), 0u,
                              chi::PoolQuery::Local());
  hp.Wait();
  REQUIRE(hp->GetReturnCode() == 0);
  ipc->FreeBuffer(hpb);

  // Move the mid-level directory "/a/b" -> "/x/y". This touches ONLY /a/b's
  // own binding; "/a/b/c" and "/a/b/c/d" reference ids that do not change, so
  // they must re-resolve under the new parent automatically (the O(1) win).
  auto mv = core.AsyncRenameTag("/a/b", "/x/y", b_id);
  mv.Wait();
  REQUIRE(mv->GetReturnCode() == 0);

  REQUIRE(resolve(b_id) == "/x/y");
  REQUIRE(resolve(d_id) == "/x/y/c/d");  // deep child re-resolved, never moved

  // The blob under the (unmoved) deep id is still intact.
  ctp::ipc::FullPtr<char> hgb = ipc->AllocateBuffer(kDeepN);
  REQUIRE(!hgb.IsNull());
  memset(hgb.ptr_, 0, kDeepN);
  auto hg = core.AsyncGetBlob(d_id, "0", 0, kDeepN, 0u,
                              hgb.shm_.template Cast<void>(),
                              chi::PoolQuery::Local());
  hg.Wait();
  REQUIRE(hg->GetReturnCode() == 0);
  REQUIRE(memcmp(hgb.ptr_, kDeepMsg, kDeepN) == 0);
  ipc->FreeBuffer(hgb);

  // The old path "/a/b" is now free: re-creating it mints a NEW id (the old
  // binding moved away), while "/a" itself still exists as a parent.
  auto recreate = core.AsyncGetOrCreateTag(
      "/a/b", TagId::GetNull(), chi::PoolQuery::Local());
  recreate.Wait();
  REQUIRE(recreate->GetReturnCode() == 0);
  REQUIRE(!(recreate->tag_id_ == b_id));

  // ---- Recursive DelTag: deleting a directory deletes its whole subtree ----
  // Current tree under the moved dir: "/x/y" (b_id) -> "/x/y/c" -> "/x/y/c/d"
  // (d_id, holds a blob). Capture the intermediate id "/x/y/c" too.
  auto cmid = core.AsyncGetOrCreateTag(
      "/x/y/c", TagId::GetNull(), chi::PoolQuery::Local());
  cmid.Wait();
  REQUIRE(cmid->GetReturnCode() == 0);
  TagId c_id = cmid->tag_id_;

  auto rmrf = core.AsyncDelTag(std::string("/x/y"), chi::PoolQuery::Local());
  rmrf.Wait();
  REQUIRE(rmrf->GetReturnCode() == 0);

  // Every tag in the subtree is gone (GetTagName reports not-found).
  for (const TagId &gone_id : {b_id, c_id, d_id}) {
    auto gn = core.AsyncGetTagName(gone_id);
    gn.Wait();
    REQUIRE(gn->GetReturnCode() == 0);
    REQUIRE(gn->found_ == 0);
  }

  // The descendant's blob is gone too.
  ctp::ipc::FullPtr<char> rdb = ipc->AllocateBuffer(kDeepN);
  REQUIRE(!rdb.IsNull());
  auto rdg = core.AsyncGetBlob(d_id, "0", 0, kDeepN, 0u,
                               rdb.shm_.template Cast<void>(),
                               chi::PoolQuery::Local());
  rdg.Wait();
  REQUIRE(rdg->GetReturnCode() != 0);
  ipc->FreeBuffer(rdb);

  // Re-creating the deepest path mints fresh ids (subtree fully removed).
  auto reborn = core.AsyncGetOrCreateTag(
      "/x/y/c/d", TagId::GetNull(), chi::PoolQuery::Local());
  reborn.Wait();
  REQUIRE(reborn->GetReturnCode() == 0);
  REQUIRE(!(reborn->tag_id_ == d_id));

  // ---- Filesystem chimod: directory operations over the hierarchy ----
  auto gattr = [&](const std::string &p) {
    auto g = cfs.AsyncGetattr(p);
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    return std::make_tuple((unsigned)g->exists_, (unsigned)g->is_dir_,
                           (unsigned long long)g->size_);
  };

  // mkdir of an empty directory -> getattr reports a directory (size 0).
  {
    auto mk = cfs.AsyncMkdir("/d1");
    mk.Wait();
    REQUIRE(mk->GetReturnCode() == 0);
    auto [ex, dir, sz] = gattr("/d1");
    REQUIRE(ex == 1);
    REQUIRE(dir == 1);
    REQUIRE(sz == 0);
  }
  // mkdir again -> EEXIST.
  {
    auto mk = cfs.AsyncMkdir("/d1");
    mk.Wait();
    REQUIRE(mk->GetReturnCode() == EEXIST);
  }

  // Create a file inside the directory and write to it.
  const std::string kFile = "/d1/f.txt";
  const char kFileMsg[] = "dir-file-payload!";
  constexpr chi::u64 kFileN = sizeof(kFileMsg);
  {
    auto op = cfs.AsyncOpen(kFile, O_CREAT | O_RDWR, 0644);
    op.Wait();
    REQUIRE(op->GetReturnCode() == 0);
    chi::u64 h = op->handle_;
    ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(kFileN);
    memcpy(wb.ptr_, kFileMsg, kFileN);
    auto w = cfs.AsyncWrite(h, 0, kFileN, wb.shm_.template Cast<void>());
    w.Wait();
    REQUIRE(w->GetReturnCode() == 0);
    REQUIRE(w->bytes_written_ == kFileN);
    ipc->FreeBuffer(wb);
    auto cl = cfs.AsyncClose(h);
    cl.Wait();
  }

  // The file is a regular file; the directory is still a directory.
  {
    auto [fex, fdir, fsz] = gattr(kFile);
    REQUIRE(fex == 1);
    REQUIRE(fdir == 0);
    REQUIRE(fsz == kFileN);
    auto [dex, ddir, dsz] = gattr("/d1");
    REQUIRE(dex == 1);
    REQUIRE(ddir == 1);
  }

  // readdir lists the file and hides the internal directory marker.
  {
    auto rd = cfs.AsyncReaddir("/d1");
    rd.Wait();
    REQUIRE(rd->GetReturnCode() == 0);
    bool saw_file = false;
    for (const auto &e : rd->entries_) {
      std::string full = e.str();
      REQUIRE(full.find(".__clio_dir__") == std::string::npos);  // marker hidden
      if (full == kFile) saw_file = true;
    }
    REQUIRE(saw_file);
  }

  // ---- Hard link via tag alias ----
  const std::string kLink = "/d1/link.txt";
  {
    auto ln = cfs.AsyncLink(kFile, kLink);
    ln.Wait();
    REQUIRE(ln->GetReturnCode() == 0);
  }
  // The link is a regular file of the same size and reads back the same data.
  {
    auto [lex, ldir, lsz] = gattr(kLink);
    REQUIRE(lex == 1);
    REQUIRE(ldir == 0);
    REQUIRE(lsz == kFileN);

    auto op = cfs.AsyncOpen(kLink, O_RDWR, 0644);
    op.Wait();
    REQUIRE(op->GetReturnCode() == 0);
    REQUIRE(op->handle_ != 0);
    ctp::ipc::FullPtr<char> rb = ipc->AllocateBuffer(kFileN);
    memset(rb.ptr_, 0, kFileN);
    auto r = cfs.AsyncRead(op->handle_, 0, kFileN, rb.shm_.template Cast<void>());
    r.Wait();
    REQUIRE(r->GetReturnCode() == 0);
    REQUIRE(r->bytes_read_ == kFileN);
    REQUIRE(memcmp(rb.ptr_, kFileMsg, kFileN) == 0);
    ipc->FreeBuffer(rb);
    auto cl = cfs.AsyncClose(op->handle_);
    cl.Wait();
  }
  // Linking onto an existing name fails.
  {
    auto ln = cfs.AsyncLink(kFile, kLink);
    ln.Wait();
    REQUIRE(ln->GetReturnCode() == EEXIST);
  }

  // rmdir of a non-empty directory fails.
  {
    auto rm = cfs.AsyncRmdir("/d1");
    rm.Wait();
    REQUIRE(rm->GetReturnCode() == ENOTEMPTY);
  }

  // Unlinking the hard link removes only that name; the original file (and its
  // data) survive.
  {
    auto ul = cfs.AsyncUnlink(kLink);
    ul.Wait();
    REQUIRE(ul->GetReturnCode() == 0);
    auto [lex, ldir, lsz] = gattr(kLink);
    REQUIRE(lex == 0);  // link gone
    auto [fex, fdir, fsz] = gattr(kFile);
    REQUIRE(fex == 1);  // original still there
    REQUIRE(fsz == kFileN);
  }

  // Remove the original file, then the now-empty directory.
  {
    auto ul = cfs.AsyncUnlink(kFile);
    ul.Wait();
    REQUIRE(ul->GetReturnCode() == 0);
    auto rm = cfs.AsyncRmdir("/d1");
    rm.Wait();
    REQUIRE(rm->GetReturnCode() == 0);
    auto [ex, dir, sz] = gattr("/d1");
    REQUIRE(ex == 0);  // directory gone
  }

  // ---- Deferred-append pipeline ----
  // Fire many appends concurrently (without awaiting each), so they pile into
  // the pending queue and the periodic AppendSequence drains them across
  // SEVERAL ManyToOne AppendCollect batches. Each append is placed locally +
  // queued; AppendCollect (sorted by UTC/logical) plans the merge against the
  // file tail and AppendExecution writes it into the pages. Because at most one
  // aggregate per tag runs at a time (BatchManager serialization), successive
  // batches see a fully-settled tail, so the bytes land in submission order and
  // span page boundaries cleanly. Verify the exact ordered concatenation.
  {
    const std::string apath = "/append_test.bin";
    auto aop = cfs.AsyncOpen(apath, O_CREAT | O_RDWR, 0644);
    aop.Wait();
    REQUIRE(aop->GetReturnCode() == 0);
    chi::u64 ah = aop->handle_;
    REQUIRE(ah != 0);

    constexpr chi::u64 kChunk = 4096;
    constexpr int kNum = 16;  // concurrent stress (probe for corruption)
    std::vector<ctp::ipc::FullPtr<char>> abufs;
    std::vector<chi::Future<clio::cte::filesystem::AppendTask>> afuts;
    for (int c = 0; c < kNum; ++c) {
      char mark = static_cast<char>('a' + c);
      ctp::ipc::FullPtr<char> ab = ipc->AllocateBuffer(kChunk);
      REQUIRE(!ab.IsNull());
      memset(ab.ptr_, mark, kChunk);
      abufs.push_back(ab);
      afuts.push_back(cfs.AsyncAppend(ah, kChunk, ab.shm_.template Cast<void>()));
    }
    for (auto &f : afuts) { f.Wait(); REQUIRE(f->GetReturnCode() == 0); }
    for (auto &ab : abufs) ipc->FreeBuffer(ab);
    const chi::u64 total = kChunk * kNum;

    // Concurrent appends are ordered by (UTC, logical), not submission order, so
    // we don't pin the order. The pipeline (periodic AppendSequence -> ManyToOne
    // AppendCollect -> AppendPlan -> AppendExecution) must merge them with no
    // overlap or corruption: poll-read until the file is exactly kNum back-to-
    // back kChunk regions, each a single distinct marker, all present once.
    bool matched = false;
    for (int attempt = 0; attempt < 400 && !matched; ++attempt) {
      ctp::ipc::FullPtr<char> rb = ipc->AllocateBuffer(total);
      REQUIRE(!rb.IsNull());
      memset(rb.ptr_, 0, total);
      auto r = cfs.AsyncRead(ah, 0, total, rb.shm_.template Cast<void>());
      r.Wait();
      bool ok = (r->GetReturnCode() == 0 && r->bytes_read_ == total);
      if (ok) {
        std::set<char> seen;
        for (int c = 0; c < kNum && ok; ++c) {
          const char *region = rb.ptr_ + static_cast<size_t>(c) * kChunk;
          char m = region[0];
          for (chi::u64 i = 0; i < kChunk; ++i) {
            if (region[i] != m) { ok = false; break; }
          }
          if (ok && m >= 'a' && m < 'a' + kNum) seen.insert(m);
          else ok = false;
        }
        if (ok && seen.size() == static_cast<size_t>(kNum)) matched = true;
      }
      ipc->FreeBuffer(rb);
      if (!matched) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
      }
    }
    REQUIRE(matched);  // all appends merged intact into their own regions

    auto acl = cfs.AsyncClose(ah);
    acl.Wait();
  }

  RunCliTimed({"stop", "--grace-period", "2000"}, 90);
  for (int i = 0; i < 200 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.Stop();
  fs::remove_all(work);
}

SIMPLE_TEST_MAIN()
