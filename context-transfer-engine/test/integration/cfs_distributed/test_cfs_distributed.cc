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
 * Cross-node ordering/liveness is handled WITHOUT any CFS coordination (so it
 * is independent of the very thing under test). When CFS_SYNC_DIR names a
 * directory both containers share (the /workspace bind mount), the roles
 * hand off through plain flag files there:
 *   - the writer writes written.flag ("ok" or "fail <rc>") when it is done,
 *     then keeps its node alive until reader_done.flag appears (cap
 *     CFS_KEEPALIVE_SEC, default 300 s);
 *   - the reader polls written.flag (cap CFS_WRITTEN_WAIT_SEC, default 300 s),
 *     then retries the open for a bounded window, and always writes
 *     reader_done.flag on the way out.
 * Without CFS_SYNC_DIR (manual runs) the old fixed timing applies: the reader
 * delays CFS_READ_DELAY_SEC (12 s) and the writer stays up CFS_KEEPALIVE_SEC
 * (75 s). Fixed timing was flaky on loaded CI runners in both directions.
 *
 * Exit code 0 == the role's checks passed. The reader's exit code is the
 * authoritative cross-node correctness signal (the run script keys on it).
 */
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
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
constexpr const char *kSharedDir = "/dist685";
constexpr const char *kSharedPath = "/dist685/shared.bin";

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

/**
 * Path of a flag file in the shared sync directory.
 * @param name flag file name (e.g. "written.flag")
 * @return the full path, or "" when CFS_SYNC_DIR is unset (legacy timing)
 */
std::string SyncPath(const char *name) {
  const char *dir = std::getenv("CFS_SYNC_DIR");
  if (!dir || !*dir) return "";
  return std::string(dir) + "/" + name;
}

/**
 * Write a flag file atomically (temp file + rename) so a poller never sees a
 * partially written body.
 * @param path destination path
 * @param body file contents
 * @return true on success
 */
bool WriteFlag(const std::string &path, const std::string &body) {
  const std::string tmp = path + ".tmp";
  FILE *f = std::fopen(tmp.c_str(), "w");
  if (!f) return false;
  std::fputs(body.c_str(), f);
  std::fclose(f);
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

/**
 * Poll for a flag file and return its contents.
 * @param path flag file path
 * @param cap_sec give up after this many seconds
 * @param body receives the file contents when found
 * @return true if the file appeared within cap_sec
 */
bool WaitFlag(const std::string &path, int cap_sec, std::string *body) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(cap_sec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (FILE *f = std::fopen(path.c_str(), "r")) {
      char buf[128] = {0};
      size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
      std::fclose(f);
      body->assign(buf, n);
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  return false;
}

// ---- writer (node A): create + modify + grow, verify local read-back --------
/**
 * Create, modify and grow the shared file, then verify a local read-back.
 * @return 0 on success, otherwise the writer's failure code
 */
int WriteSharedFile() {
  const char *role = "writer";
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(role, "FAIL: CLIO_INIT(kClient) failed");
    return 2;
  }
  // The filesystem pool composed in clio_config.yaml, found by name.
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    Log(role, "FAIL: no filesystem pool");
    return 2;
  }
  clio::cte::filesystem::Client &cfs = *CLIO_CFS_CLIENT;
  auto *ipc = CLIO_IPC;

  // The namespace has real directories: creating a file under a parent that
  // does not exist is ENOENT, as POSIX says (#1229). Make the parent first;
  // EEXIST (a rerun against a surviving namespace) is fine.
  auto mkdir = cfs.AsyncMkdir(kSharedDir);
  mkdir.Wait();
  if (mkdir->GetReturnCode() != 0 && mkdir->GetReturnCode() != EEXIST) {
    Log(role, "FAIL: AsyncMkdir rc=" +
                  std::to_string(mkdir->GetReturnCode()));
    return 3;
  }

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
  Log(role, "created+modified+grew " + std::to_string(kFinalSize) +
                " bytes at " + kSharedPath);
  return 0;
}

/**
 * Writer role: write the shared file, tell the reader (written.flag), then
 * keep this node's daemon -- and the container that owns the path -- alive
 * until the reader reports it is done, so its cross-node read resolves.
 * @return 0 if the writer's own operations succeeded and the reader finished
 */
int RunWriter() {
  const char *role = "writer";
  const int rc = WriteSharedFile();
  const std::string written = SyncPath("written.flag");
  if (written.empty()) {
    // Legacy fixed timing (no shared sync dir).
    const int keepalive = EnvSecs("CFS_KEEPALIVE_SEC", 75);
    if (rc == 0) {
      Log(role, "staying alive " + std::to_string(keepalive) +
                    "s for the reader (no CFS_SYNC_DIR)");
      std::this_thread::sleep_for(std::chrono::seconds(keepalive));
    }
    return rc;
  }
  WriteFlag(written, rc == 0 ? "ok" : "fail " + std::to_string(rc));
  if (rc != 0) return rc;

  const int cap = EnvSecs("CFS_KEEPALIVE_SEC", 300);
  Log(role, "waiting up to " + std::to_string(cap) +
                "s for reader_done.flag");
  std::string body;
  if (!WaitFlag(SyncPath("reader_done.flag"), cap, &body)) {
    Log(role, "FAIL: reader did not finish within " + std::to_string(cap) +
                  "s (reader_done.flag never appeared)");
    return 7;
  }
  Log(role, "reader done (" + body + "); exiting");
  return 0;
}

// ---- reader (node B): must SEE the writer's modified file across the cluster -
/**
 * Wait until the writer has finished writing the shared file.
 * With CFS_SYNC_DIR this polls written.flag; otherwise it sleeps the legacy
 * CFS_READ_DELAY_SEC.
 * @param role log tag
 * @return 0 when the reader may proceed, otherwise an exit code
 */
int WaitForWriter(const char *role) {
  const std::string written = SyncPath("written.flag");
  if (written.empty()) {
    const int delay = EnvSecs("CFS_READ_DELAY_SEC", 12);
    Log(role, "delaying " + std::to_string(delay) +
                  "s for the writer (no CFS_SYNC_DIR), then reading");
    std::this_thread::sleep_for(std::chrono::seconds(delay));
    return 0;
  }
  const int cap = EnvSecs("CFS_WRITTEN_WAIT_SEC", 300);
  std::string body;
  if (!WaitFlag(written, cap, &body)) {
    Log(role, "FAIL: writer did not finish within " + std::to_string(cap) +
                  "s (written.flag never appeared)");
    return 8;
  }
  if (body != "ok") {
    Log(role, "FAIL: writer reported '" + body + "'; nothing to read");
    return 9;
  }
  Log(role, "writer finished; reading");
  return 0;
}

/**
 * Reader role body: open the writer's file from this node and verify it.
 * @return 0 if the cross-node read matched, otherwise a failure code
 */
int ReadSharedFile() {
  const char *role = "reader";
  if (int rc = WaitForWriter(role); rc != 0) return rc;
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(role, "FAIL: CLIO_INIT(kClient) failed");
    return 3;
  }
  // The filesystem pool composed in clio_config.yaml, found by name.
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    Log(role, "FAIL: no filesystem pool");
    return 2;
  }
  clio::cte::filesystem::Client &cfs = *CLIO_CFS_CLIENT;
  auto *ipc = CLIO_IPC;

  // Retry the cross-node open: the shared namespace should resolve the tag the
  // writer created (on whichever container owns it) from THIS node too, and
  // report the grown size once the writer's modify+grow has landed.
  int rc = 1;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(EnvSecs("CFS_READ_WINDOW_SEC", 40));
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

/**
 * Reader role: run the check, then always signal reader_done.flag so the
 * writer can release its node instead of waiting out its cap.
 * @return the reader's exit code
 */
int RunReader() {
  const int rc = ReadSharedFile();
  const std::string done = SyncPath("reader_done.flag");
  if (!done.empty()) {
    WriteFlag(done, std::to_string(rc));
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
