/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Distributed CTE filesystem (CFS) correctness test (issue #685).
 *
 * Verifies that the CFS namespace is SHARED across a multi-node runtime: a file
 * CREATED and then MODIFIED through the CFS client on one node must be visible
 * and readable, with the fully-modified bytes, through the CFS client on a
 * DIFFERENT node. With the pre-#685 bug (every tag op issued with
 * PoolQuery::Local()) a path's tag lives only in the issuing node's local
 * container, so the reader on node B gets ENOENT.
 *
 * The writer does not just create-and-write once: it CREATEs the file, then
 * performs a mid-file in-place MODIFY that straddles a 1 MiB page boundary, then
 * GROWs the file past its original end — so create, overwrite, and extend all
 * flow through the (Dynamic-routed) multi-blob CFS write path. The reader must
 * observe the final post-modify+grow content from the other node, byte-for-byte.
 *
 * This is a single binary run in two roles inside a docker cluster (see the
 * docker-compose in this directory), selected by CFS_DIST_ROLE:
 *
 *   writer  (node 1): CLIO_INIT(kClient) -> connects to the LOCAL daemon,
 *                     creates + modifies + grows a shared path, verifies its own
 *                     read-back, then stays alive (so its container/owner
 *                     survives) for CFS_KEEPALIVE_SEC.
 *   reader  (node 2): CLIO_INIT(kClient) -> connects to the LOCAL daemon, waits
 *                     CFS_READ_DELAY_SEC for the writer, then retries opening the
 *                     SAME path and verifies the modified bytes match. ENOENT
 *                     after the retry window == the cross-node namespace is
 *                     broken (#685).
 *
 * Cross-node ordering/liveness is handled WITHOUT any CFS or shared-fs
 * coordination (so it is independent of the very thing under test): the reader
 * simply delays a few seconds (CFS_READ_DELAY_SEC) so the writer has created the
 * file, then retries the open for a bounded window; the writer, after writing,
 * stays alive (CFS_KEEPALIVE_SEC) so its node's daemon — and thus the container
 * that owns the path — survives for the whole of the reader's read window.
 *
 * Exit code 0 == the role's checks passed. The reader's exit code is the
 * authoritative cross-node correctness signal (the run script keys on it).
 */
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/filesystem/filesystem_client.h>

namespace {

// Shared, deterministic file geometry so the reader can prove it read the
// WRITER's create+MODIFY+grow result cross-node (not zeros/garbage/stale bytes).
// The file spans several 1 MiB CFS page-blobs so every stage exercises the
// multi-blob distributed read/write path:
//   * CREATE + initial write of kInitSize bytes (pages 0..2, partial page 2),
//   * an in-place MODIFY overwriting a kModLen region straddling the page-0/
//     page-1 boundary (a mid-file rewrite of existing pages, not a create),
//   * a GROW that extends the file to kFinalSize (fills page 2, adds page 3).
constexpr clio::run::u64 kMiB = 1024u * 1024u;
constexpr clio::run::u64 kInitSize = 2u * kMiB + 512u * 1024u;   // 2.5 MiB
constexpr clio::run::u64 kModOff = 1u * kMiB - 64u * 1024u;      // 960 KiB
constexpr clio::run::u64 kModLen = 128u * 1024u;                 // spans 1 MiB
constexpr clio::run::u64 kFinalSize = 3u * kMiB + 512u * 1024u;  // 3.5 MiB
constexpr const char *kSharedPath = "/dist685/shared.bin";
// The CFS filesystem pool composed in clio_config.yaml (kCfsPoolId = 560.0).
constexpr clio::run::u32 kCfsPoolMajor = 560;

// Read an integer env var (seconds) with a default.
int EnvSecs(const char *name, int def) {
  const char *e = std::getenv(name);
  if (!e || !*e) return def;
  int v = std::atoi(e);
  return v > 0 ? v : def;
}

// Three distinct, node-independent deterministic byte streams. The file's FINAL
// content layers them: the base pattern everywhere, overwritten by the modify
// stream within [kModOff, kModOff+kModLen), and the grow stream past kInitSize.
unsigned char PatternByte(clio::run::u64 i) {
  return static_cast<unsigned char>((i * 131u + 7u) & 0xFFu);
}
unsigned char ModByte(clio::run::u64 i) {
  return static_cast<unsigned char>((i * 197u + 23u) & 0xFFu);
}
unsigned char TailByte(clio::run::u64 i) {
  return static_cast<unsigned char>((i * 251u + 41u) & 0xFFu);
}
// The authoritative post-modify+grow content the reader must see at byte i.
unsigned char ExpectedByte(clio::run::u64 i) {
  if (i >= kInitSize) return TailByte(i);
  if (i >= kModOff && i < kModOff + kModLen) return ModByte(i);
  return PatternByte(i);
}

void Log(const char *role, const std::string &msg) {
  std::fprintf(stderr, "[cfs-dist %s] %s\n", role, msg.c_str());
  std::fflush(stderr);
}

// ---- writer (node A): create + modify + grow, verify local read-back --------
int RunWriter() {
  const char *role = "writer";
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(role, "FAIL: CLIO_INIT(kClient) failed");
    return 2;
  }
  clio::cte::filesystem::Client cfs;
  cfs.Init(clio::run::PoolId(kCfsPoolMajor, 0));
  auto *ipc = CLIO_IPC;

  auto open = cfs.AsyncOpen(kSharedPath, O_CREAT | O_RDWR, 0644);
  open.Wait();
  if (open->GetReturnCode() != 0 || open->handle_ == 0) {
    Log(role, "FAIL: AsyncOpen(create) rc=" +
                  std::to_string(open->GetReturnCode()));
    return 3;
  }
  clio::run::u64 h = open->handle_;

  // One AsyncWrite of [off, off+len), each byte sourced from byte-fn `f`. Each
  // call is a distinct op through the (now Dynamic-routed) CFS write path.
  auto write_region = [&](clio::run::u64 off, clio::run::u64 len,
                          unsigned char (*f)(clio::run::u64),
                          const char *what) -> bool {
    ctp::ipc::FullPtr<char> wbuf = ipc->AllocateBuffer(len);
    for (clio::run::u64 j = 0; j < len; ++j) wbuf.ptr_[j] = f(off + j);
    auto w = cfs.AsyncWrite(h, off, len, wbuf.shm_.template Cast<void>());
    w.Wait();
    bool ok = (w->GetReturnCode() == 0);
    ipc->FreeBuffer(wbuf);
    if (!ok) {
      Log(role, std::string("FAIL: AsyncWrite ") + what + " rc=" +
                    std::to_string(w->GetReturnCode()));
    }
    return ok;
  };

  // (1) CREATE + initial full write.
  if (!write_region(0, kInitSize, PatternByte, "create")) return 4;
  // (2) In-place MODIFY of a mid-file region that straddles a page boundary.
  if (!write_region(kModOff, kModLen, ModByte, "modify")) return 4;
  // (3) GROW the file past its original end (adds new page-blobs).
  if (!write_region(kInitSize, kFinalSize - kInitSize, TailByte, "grow")) {
    return 4;
  }

  // Local read-back sanity on the writer node (same-node routing must already
  // work; this isolates cross-node failures from local ones). Verifies the
  // FINAL post-modify+grow content.
  ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kFinalSize);
  std::memset(rbuf.ptr_, 0, kFinalSize);
  auto r = cfs.AsyncRead(h, 0, kFinalSize, rbuf.shm_.template Cast<void>());
  r.Wait();
  if (r->GetReturnCode() != 0 || r->bytes_read_ != kFinalSize) {
    Log(role, "FAIL: local read-back rc=" + std::to_string(r->GetReturnCode()) +
                  " bytes=" + std::to_string(r->bytes_read_));
    return 5;
  }
  for (clio::run::u64 i = 0; i < kFinalSize; ++i) {
    if (static_cast<unsigned char>(rbuf.ptr_[i]) != ExpectedByte(i)) {
      Log(role, "FAIL: local read-back mismatch at " + std::to_string(i));
      return 6;
    }
  }
  ipc->FreeBuffer(rbuf);
  auto cl = cfs.AsyncClose(h);
  cl.Wait();
  int keepalive = EnvSecs("CFS_KEEPALIVE_SEC", 75);
  Log(role, "created+modified+grew " + std::to_string(kFinalSize) +
                " bytes at " + kSharedPath + "; staying alive " +
                std::to_string(keepalive) + "s for the reader");

  // Keep this node's daemon (and thus the container that owns the path) alive
  // for the whole of the reader's read window, so a cross-node read from the
  // other node still resolves. Independent of the CFS being tested.
  std::this_thread::sleep_for(std::chrono::seconds(keepalive));
  Log(role, "keep-alive elapsed; exiting");
  return 0;  // the writer's own operations succeeded
}

// ---- reader (node B): must SEE the writer's modified file across the cluster -
int RunReader() {
  const char *role = "reader";
  // Give the writer time to create+modify the file before we start looking (the
  // retry loop below tolerates the rest of the skew).
  int delay = EnvSecs("CFS_READ_DELAY_SEC", 12);
  Log(role, "delaying " + std::to_string(delay) + "s for the writer, then reading");
  std::this_thread::sleep_for(std::chrono::seconds(delay));
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(role, "FAIL: CLIO_INIT(kClient) failed");
    return 3;
  }
  clio::cte::filesystem::Client cfs;
  cfs.Init(clio::run::PoolId(kCfsPoolMajor, 0));
  auto *ipc = CLIO_IPC;

  // Retry the cross-node open: the shared namespace should resolve the tag the
  // writer created (on whichever container owns it) from THIS node too, and
  // report the grown size once the writer's modify+grow has landed.
  int rc = 1;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(40);
  clio::run::u64 last_rc = 999, last_bytes = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    auto open = cfs.AsyncOpen(kSharedPath, O_RDONLY, 0644);
    open.Wait();
    // handle_==0 means "not found" (the adapters map that to ENOENT) — the #685
    // symptom on a broken (node-local) namespace. size_ < kFinalSize means the
    // grow half of the modify has not propagated yet; keep retrying.
    if (open->GetReturnCode() == 0 && open->handle_ != 0 &&
        open->size_ >= kFinalSize) {
      clio::run::u64 h = open->handle_;
      ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kFinalSize);
      std::memset(rbuf.ptr_, 0, kFinalSize);
      auto r = cfs.AsyncRead(h, 0, kFinalSize, rbuf.shm_.template Cast<void>());
      r.Wait();
      auto cl = cfs.AsyncClose(h);
      cl.Wait();
      last_rc = r->GetReturnCode();
      last_bytes = r->bytes_read_;
      if (r->GetReturnCode() == 0 && r->bytes_read_ == kFinalSize) {
        bool match = true;
        for (clio::run::u64 i = 0; i < kFinalSize; ++i) {
          if (static_cast<unsigned char>(rbuf.ptr_[i]) != ExpectedByte(i)) {
            match = false;
            Log(role, "FAIL: cross-node byte mismatch at " + std::to_string(i));
            break;
          }
        }
        ipc->FreeBuffer(rbuf);
        if (match) {
          Log(role, "PASS: cross-node read of " + std::to_string(kFinalSize) +
                        " modified bytes from " + kSharedPath +
                        " matched the writer");
          rc = 0;
        }
        break;  // found + read (pass or data-mismatch): done retrying
      }
      ipc->FreeBuffer(rbuf);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  if (rc != 0) {
    Log(role, "FAIL: could not read the writer's modified file cross-node "
              "(ENOENT/short read) — CFS namespace is node-local (#685). "
              "last read rc=" + std::to_string(last_rc) +
              " bytes=" + std::to_string(last_bytes));
  }
  return rc;
}

}  // namespace

int main() {
  const char *role = std::getenv("CFS_DIST_ROLE");
  if (!role) {
    std::fprintf(stderr, "CFS_DIST_ROLE not set (expected 'writer' or 'reader')\n");
    return 64;
  }
  if (std::strcmp(role, "writer") == 0) return RunWriter();
  if (std::strcmp(role, "reader") == 0) return RunReader();
  std::fprintf(stderr, "unknown CFS_DIST_ROLE='%s'\n", role);
  return 64;
}
