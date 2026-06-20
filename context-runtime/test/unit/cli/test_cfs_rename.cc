/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Focused rename stress + concurrency tests for the filesystem chimod.
 *
 * Rename is the operation the xfstests surfaced as most fragile (issue #597):
 *   * B1 generic/245 — rename-overwrite ignores destination type/emptiness
 *     (clobbers a non-empty dir / wrong-type target instead of EEXIST/ENOTEMPTY).
 *   * B2 generic/035 — renaming OVER an open file strands that file's handle.
 *   * #596 — overlapping renames could orphan a tag (resolvable upward via the
 *     name stored in its info, but not forward via the name->id binding).
 *
 * The cfs integration test (test_cfs.cc) covers the single-threaded happy path.
 * This file adds (a) POSIX-semantics checks for rename-overwrite, and (b)
 * concurrency torture: many renames racing on shared destinations, where the
 * filesystem Rename's non-atomic "DelTag(dst) then RenameTag(src,dst)" is most
 * likely to lose data or leave duplicate/orphan bindings.
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
#include <utility>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/filesystem/filesystem_client.h>

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

TEST_CASE("Cfs - rename POSIX semantics + concurrency", "[cli][cfs][rename]") {
  constexpr unsigned kPort = 10605;
  const fs::path work = fs::temp_directory_path() / "cfs_rename_test";
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

  // ---- helpers -------------------------------------------------------------
  // Create a file and fill it with `n` bytes of marker byte `mark`.
  auto mkfile = [&](const std::string& path, char mark, chi::u64 n) {
    auto op = cfs.AsyncOpen(path, O_CREAT | O_RDWR, 0644);
    op.Wait();
    REQUIRE(op->GetReturnCode() == 0);
    chi::u64 h = op->handle_;
    if (n > 0) {
      ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(n);
      memset(wb.ptr_, mark, n);
      auto w = cfs.AsyncWrite(h, 0, n, wb.shm_.template Cast<void>());
      w.Wait();
      REQUIRE(w->GetReturnCode() == 0);
      ipc->FreeBuffer(wb);
    }
    auto cl = cfs.AsyncClose(h);
    cl.Wait();
  };

  auto exists = [&](const std::string& path) -> bool {
    auto g = cfs.AsyncGetattr(path);
    g.Wait();
    return g->GetReturnCode() == 0 && g->exists_ == 1;
  };
  auto is_dir = [&](const std::string& path) -> bool {
    auto g = cfs.AsyncGetattr(path);
    g.Wait();
    return g->GetReturnCode() == 0 && g->exists_ == 1 && g->is_dir_ == 1;
  };
  auto fsize = [&](const std::string& path) -> chi::u64 {
    auto g = cfs.AsyncGetattr(path);
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    return g->size_;
  };
  // Read the first byte of a file by path (its uniform marker).
  auto first_byte = [&](const std::string& path) -> int {
    auto op = cfs.AsyncOpen(path, O_RDWR, 0644);
    op.Wait();
    if (op->GetReturnCode() != 0) return -1;
    chi::u64 h = op->handle_;
    ctp::ipc::FullPtr<char> rb = ipc->AllocateBuffer(1);
    rb.ptr_[0] = 0;
    auto r = cfs.AsyncRead(h, 0, 1, rb.shm_.template Cast<void>());
    r.Wait();
    int b = (r->GetReturnCode() == 0 && r->bytes_read_ == 1)
                ? (unsigned char)rb.ptr_[0] : -1;
    ipc->FreeBuffer(rb);
    auto cl = cfs.AsyncClose(h);
    cl.Wait();
    return b;
  };
  auto rename_rc = [&](const std::string& src, const std::string& dst) -> int {
    auto mv = cfs.AsyncRename(src, dst);
    mv.Wait();
    return (int)mv->GetReturnCode();
  };
  // Count direct entries of `dir` whose leaf equals `leaf`, and total entries.
  auto readdir_count = [&](const std::string& dir, const std::string& full_path)
      -> std::pair<int, int> {
    auto rd = cfs.AsyncReaddir(dir);
    rd.Wait();
    REQUIRE(rd->GetReturnCode() == 0);
    int total = 0, matches = 0;
    for (const auto& e : rd->entries_) {
      std::string s = e.str();
      if (s.find(".__clio_dir__") != std::string::npos) continue;  // marker
      ++total;
      if (s == full_path) ++matches;
    }
    return {total, matches};
  };

  constexpr chi::u64 kN = 4096;

  // ======================================================================
  // A. Basic rename correctness (sanity): move keeps data, source vanishes.
  // ======================================================================
  {
    mkfile("/A/src.bin", 'a', kN);
    REQUIRE(rename_rc("/A/src.bin", "/A/dst.bin") == 0);
    REQUIRE_FALSE(exists("/A/src.bin"));
    REQUIRE(exists("/A/dst.bin"));
    REQUIRE(fsize("/A/dst.bin") == kN);
    REQUIRE(first_byte("/A/dst.bin") == 'a');
  }

  // ======================================================================
  // B1. rename-overwrite POSIX semantics (issue #597 B1, generic/245).
  // ======================================================================
  {
    // file -> existing file: allowed; destination takes the source's content.
    mkfile("/B/f1.bin", 'x', kN);
    mkfile("/B/f2.bin", 'y', kN);
    REQUIRE(rename_rc("/B/f1.bin", "/B/f2.bin") == 0);
    REQUIRE_FALSE(exists("/B/f1.bin"));
    REQUIRE(exists("/B/f2.bin"));
    REQUIRE(first_byte("/B/f2.bin") == 'x');   // overwritten with source data

    // file -> non-empty directory: must fail, and must NOT destroy the dir.
    mkfile("/B/file.bin", 'm', kN);
    mkfile("/B/dir/inner.bin", 'n', kN);
    int rc = rename_rc("/B/file.bin", "/B/dir");
    REQUIRE(rc != 0);                          // EISDIR / ENOTEMPTY / EEXIST
    REQUIRE(is_dir("/B/dir"));                  // dir survives
    REQUIRE(exists("/B/dir/inner.bin"));        // its child survives
    REQUIRE(exists("/B/file.bin"));             // source untouched

    // directory -> non-empty directory: must fail (ENOTEMPTY), both intact.
    mkfile("/B/d1/c1.bin", 'p', kN);
    mkfile("/B/d2/c2.bin", 'q', kN);
    int rc2 = rename_rc("/B/d1", "/B/d2");
    REQUIRE(rc2 != 0);
    REQUIRE(is_dir("/B/d1"));
    REQUIRE(exists("/B/d1/c1.bin"));
    REQUIRE(is_dir("/B/d2"));
    REQUIRE(exists("/B/d2/c2.bin"));
  }

  // ======================================================================
  // B2. rename a file that is currently OPEN (move, not overwrite).
  // The tag keeps its TagId across rename, so the open handle must stay
  // valid and keep reading the same bytes from the new path.
  // ======================================================================
  {
    mkfile("/C/open.bin", 'o', kN);
    auto op = cfs.AsyncOpen("/C/open.bin", O_RDWR, 0644);
    op.Wait();
    REQUIRE(op->GetReturnCode() == 0);
    chi::u64 h = op->handle_;

    REQUIRE(rename_rc("/C/open.bin", "/C/moved.bin") == 0);
    REQUIRE_FALSE(exists("/C/open.bin"));
    REQUIRE(exists("/C/moved.bin"));

    // The still-open handle must read the original content.
    ctp::ipc::FullPtr<char> rb = ipc->AllocateBuffer(kN);
    memset(rb.ptr_, 0, kN);
    auto r = cfs.AsyncRead(h, 0, kN, rb.shm_.template Cast<void>());
    r.Wait();
    REQUIRE(r->GetReturnCode() == 0);
    REQUIRE(r->bytes_read_ == kN);
    REQUIRE((unsigned char)rb.ptr_[0] == 'o');
    ipc->FreeBuffer(rb);
    auto cl = cfs.AsyncClose(h);
    cl.Wait();
  }

  // ======================================================================
  // D1. Concurrent INDEPENDENT renames: N files each moved to its own
  // destination at once. All must land intact; no source may remain.
  // ======================================================================
  {
    constexpr int kFiles = 16;
    for (int i = 0; i < kFiles; ++i) {
      mkfile("/D1/s" + std::to_string(i) + ".bin",
             (char)('A' + i), kN);
    }
    std::vector<chi::Future<clio::cte::filesystem::RenameTask>> futs;
    futs.reserve(kFiles);
    for (int i = 0; i < kFiles; ++i) {
      futs.push_back(cfs.AsyncRename("/D1/s" + std::to_string(i) + ".bin",
                                     "/D1/d" + std::to_string(i) + ".bin"));
    }
    for (auto& f : futs) { f.Wait(); REQUIRE(f->GetReturnCode() == 0); }
    for (int i = 0; i < kFiles; ++i) {
      REQUIRE_FALSE(exists("/D1/s" + std::to_string(i) + ".bin"));
      REQUIRE(exists("/D1/d" + std::to_string(i) + ".bin"));
      REQUIRE(first_byte("/D1/d" + std::to_string(i) + ".bin") == ('A' + i));
    }
  }

  // ======================================================================
  // D2. Concurrent renames to the SAME destination (N -> 1). Each rename is
  // atomic, so afterward EXACTLY ONE file named dst must exist, holding one
  // source's data, with every source gone and NO orphan/duplicate binding.
  // This is where the non-atomic DelTag(dst)+RenameTag is most dangerous:
  // a losing rename can leave its tag resolvable upward (readdir lists it)
  // but not forward, i.e. a duplicate "dst" or a leaked tag.
  // ======================================================================
  {
    constexpr int kSrc = 12;
    for (int i = 0; i < kSrc; ++i) {
      mkfile("/D2/src" + std::to_string(i) + ".bin",
             (char)('0' + i), kN);
    }
    std::vector<chi::Future<clio::cte::filesystem::RenameTask>> futs;
    futs.reserve(kSrc);
    for (int i = 0; i < kSrc; ++i) {
      futs.push_back(cfs.AsyncRename("/D2/src" + std::to_string(i) + ".bin",
                                     "/D2/dst.bin"));
    }
    for (auto& f : futs) { f.Wait(); }  // return codes may vary; check state

    // Exactly one destination, readable, with some source's marker.
    REQUIRE(exists("/D2/dst.bin"));
    int b = first_byte("/D2/dst.bin");
    REQUIRE(b >= '0');
    REQUIRE(b < '0' + kSrc);
    // Every source name must be gone.
    for (int i = 0; i < kSrc; ++i) {
      REQUIRE_FALSE(exists("/D2/src" + std::to_string(i) + ".bin"));
    }
    // readdir must show dst exactly once and nothing else: no duplicate
    // entries from an orphaned loser tag, no leftover sources.
    auto [total, matches] = readdir_count("/D2", "/D2/dst.bin");
    REQUIRE(matches == 1);
    REQUIRE(total == 1);
  }

  // ======================================================================
  // D3. Concurrent rename racing create on the same name. A renamer moves
  // X -> N while a creator opens N (O_CREAT). Afterward N must resolve to a
  // SINGLE tag (readdir shows it once); X must be gone.
  // ======================================================================
  {
    mkfile("/D3/x.bin", 'z', kN);
    auto mv = cfs.AsyncRename("/D3/x.bin", "/D3/n.bin");
    auto cr = cfs.AsyncOpen("/D3/n.bin", O_CREAT | O_RDWR, 0644);
    mv.Wait();
    cr.Wait();
    if (cr->GetReturnCode() == 0) {
      auto cl = cfs.AsyncClose(cr->handle_);
      cl.Wait();
    }
    REQUIRE_FALSE(exists("/D3/x.bin"));
    REQUIRE(exists("/D3/n.bin"));
    auto [total, matches] = readdir_count("/D3", "/D3/n.bin");
    REQUIRE(matches == 1);   // exactly one binding for n.bin (no duplicate)
    REQUIRE(total == 1);
  }

  // ======================================================================
  // D4. Rename racing readdir: while N files are concurrently renamed from
  // their "a" name to their "b" name, a flurry of readdir calls runs in the
  // same wave. Each rebind is atomic (issue #596), so EVERY readdir must list
  // exactly N files — a file may show under its old or new name, but must
  // never disappear (the "resolvable upward but not forward" hazard) nor be
  // listed twice. Repeated over several rounds to widen the race window.
  // ======================================================================
  {
    constexpr int kFiles = 12;
    constexpr int kRounds = 6;
    for (int i = 0; i < kFiles; ++i) {
      mkfile("/D4/a" + std::to_string(i) + ".bin", (char)('a' + i), kN);
    }
    for (int round = 0; round < kRounds; ++round) {
      const std::string from = (round % 2 == 0) ? "a" : "b";
      const std::string to = (round % 2 == 0) ? "b" : "a";
      std::vector<chi::Future<clio::cte::filesystem::RenameTask>> rmv;
      std::vector<chi::Future<clio::cte::filesystem::ReaddirTask>> rdd;
      rmv.reserve(kFiles);
      for (int i = 0; i < kFiles; ++i) {
        rmv.push_back(cfs.AsyncRename("/D4/" + from + std::to_string(i) + ".bin",
                                      "/D4/" + to + std::to_string(i) + ".bin"));
      }
      for (int k = 0; k < kFiles; ++k) rdd.push_back(cfs.AsyncReaddir("/D4"));
      for (auto &f : rmv) { f.Wait(); REQUIRE(f->GetReturnCode() == 0); }
      // Every concurrent listing must have seen all kFiles (never fewer).
      for (auto &f : rdd) {
        f.Wait();
        REQUIRE(f->GetReturnCode() == 0);
        int n = 0;
        for (const auto &e : f->entries_) {
          if (e.str().find(".__clio_dir__") == std::string::npos) ++n;
        }
        REQUIRE(n == kFiles);
      }
    }
    // Final settle: all kFiles present under their round-parity name, intact.
    const std::string fin = (kRounds % 2 == 0) ? "a" : "b";
    for (int i = 0; i < kFiles; ++i) {
      REQUIRE(exists("/D4/" + fin + std::to_string(i) + ".bin"));
      REQUIRE(first_byte("/D4/" + fin + std::to_string(i) + ".bin") ==
              ('a' + i));
    }
  }

  RunCliTimed({"stop", "--grace-period", "2000"}, 90);
  for (int i = 0; i < 200 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.Stop();
  fs::remove_all(work);
}

SIMPLE_TEST_MAIN()
