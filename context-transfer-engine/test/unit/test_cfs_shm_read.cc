/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * clio-fs zero-IPC read path (issue #817).
 *
 * Proves three things, in order of importance:
 *   1. CORRECTNESS -- bytes returned by the fast path are identical to the
 *      bytes the RPC path returns, including at EOF and across page bounds.
 *   2. IT ACTUALLY RAN -- a fast path that silently never engages looks
 *      exactly like a fast path that works, so the test fails loudly when the
 *      cache is unavailable rather than skipping.
 *   3. LATENCY -- the same read, timed on both paths, same query.
 */

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "../../adapter/cfs/cfs_io.h"
#include "clio_cte/core/core_client.h"
#include "clio_cte/filesystem/filesystem_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"
#include "runtime_server.h"
#include "simple_test.h"

using namespace std::chrono_literals;

namespace {

// A RAM target is what makes page payloads SHM-resident and therefore
// direct-readable; a file target would (correctly) refuse the fast path.
constexpr clio::run::u64 kRamTargetBytes = 256ULL * 1024 * 1024;
/** Own port so this can run alongside nothing else (RESOURCE_LOCK enforces). */
constexpr unsigned kPort = 10617;
const std::string kBackendPath = "/tmp/clio_cfs_shm_read_test.dat";
const std::string kClioPath = "clio::" + kBackendPath;

/** Microseconds, as a double. */
double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/** Run `clio_run <args>` to completion, returning its exit code. */
int RunCli(const std::vector<std::string> &args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char *> argv;
  for (auto &a : full) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
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
    std::this_thread::sleep_for(100ms);
  }
}

// The daemon runs for the whole test case; destroying it SIGTERMs the child.
clio::run::test::RuntimeServer *g_server = nullptr;

/**
 * Bring up a REAL daemon in its own process and compose it from YAML, exactly
 * as a deployment does, then attach as a pure client.
 *
 * This shape is the point. An in-process (co-located) runtime shares the
 * address space and registers its targets by hand, and under it the fast path
 * looked healthy while it was in fact DEAD in every real deployment: compose
 * registers every target as PoolQuery::DirectHash(node), so the runtime's
 * "is this target local" test -- written as IsLocalMode() -- was false for
 * every production target and kShmBlobDirectReadable was never set. Only a
 * composed daemon in another process exposes that.
 */
bool InitRuntime() {
  static bool ok = false;
  static bool tried = false;
  if (tried) {
    return ok;
  }
  tried = true;

  const std::string work = "/tmp/clio_cfs_shm_read_test";
  std::filesystem::remove_all(work);
  std::filesystem::create_directories(work);
  const std::string yaml = work + "/compose.yaml";
  {
    std::ofstream f(yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cfs_shm_cte\"\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << work << "/ram_dev\n"
         "        bdev_type: ram\n"
         "        capacity_limit: " << (kRamTargetBytes / (1024 * 1024))
      << "mb\n"
         "    dpe:\n"
         "      dpe_type: random\n"
         "  - mod_name: clio_cte_filesystem\n"
         "    pool_name: \"clio_cte_filesystem\"\n"
         "    pool_query: local\n"
         "    pool_id: \"560.0\"\n"
         "    next_pool_id: \"512.0\"\n";
  }

  setenv("CLIO_WAIT_SERVER", "15", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

  static clio::run::test::RuntimeServer server;
  g_server = &server;
  if (!server.Start(kPort) || !server.WaitForReady()) {
    return false;
  }
  if (RunCli({"compose", "start", yaml}, 60) != 0) {
    return false;
  }

  // kClient with NO co-located runtime: this process is a plain client of the
  // daemon above, so every shared-memory offset it resolves comes from a
  // segment another process created.
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    return false;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    return false;
  }
  // Storage comes from the compose above; registering another target here
  // would give the DPE a second placement choice and make the outcome depend
  // on which one it picked.
  ok = true;
  return ok;
}

/** Read `count` bytes at `off` through the chimod RPC, bypassing the cache.
 *  This is the exact work CfsIo::DoRead does when the fast path declines, and
 *  is the honest baseline to time the fast path against. */
ssize_t RpcRead(clio::run::u64 handle, clio::run::u64 off, void *buf,
                size_t count) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
  if (shm.IsNull()) {
    return -1;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRead(handle, off, count, ctp::ipc::ShmPtr<>(shm.shm_));
  t.Wait();
  ssize_t ret = -1;
  if (t->GetReturnCode() == 0) {
    std::memcpy(buf, shm.ptr_, static_cast<size_t>(t->bytes_read_));
    ret = static_cast<ssize_t>(t->bytes_read_);
  }
  ipc->FreeBuffer(shm);
  return ret;
}

}  // namespace

TEST_CASE("clio-fs SHM read: correctness and latency", "[cfs][shm][noleak]") {
  REQUIRE(InitRuntime());

  auto *cfs_io = CLIO_CTE_CFS;
  REQUIRE(cfs_io != nullptr);

  // ---- write a file through the adapter ----------------------------------
  const size_t kFileSize = 3 * 1024 * 1024 + 4096;  // spans 4 pages, unaligned
  std::vector<char> src(kFileSize);
  for (size_t i = 0; i < kFileSize; ++i) {
    src[i] = static_cast<char>((i * 31 + 7) % 251);
  }

  cfs_io->RemovePath(kClioPath);  // start from a known state
  int fd = cfs_io->Open(kClioPath, O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(fd >= 0);
  REQUIRE(cfs_io->Write(fd, src.data(), kFileSize) ==
          static_cast<ssize_t>(kFileSize));

  // ---- is the shared-memory substrate present at all? --------------------
  // There is a real difference between "this feature failed" and "the thing
  // this feature is built on does not exist on this platform", and only the
  // first is a bug in #817. The runtime's metadata segment is created
  // non-fatally (a host that cannot back an 8 GB sparse reservation is
  // allowed to refuse), and macOS CI runners do refuse it -- so there is no
  // segment for ANY cache to live in there, #783's included.
  //
  // Report that loudly and stop, rather than failing a build over a platform
  // limitation this change did not introduce and cannot fix from here. Note
  // what is NOT skipped on: a cache that is off while the segment exists
  // still fails below, which is the case that would actually mean regression.
  if (CLIO_CPU_IPC == nullptr ||
      CLIO_CPU_IPC->GetMetadataAllocator() == nullptr) {
    std::printf(
        "\n[#817] SKIP: this runtime has no metadata shared-memory segment, "
        "so the SHM cache cannot exist here (see ServerInitShm). The read "
        "fast path is untested on this platform.\n");
    return;
  }

  // ---- the cache must actually be attached -------------------------------
  // A silently-disabled cache would make every assertion below pass on the
  // RPC path, so this is checked explicitly rather than being inferred.
  auto *cte_client = CLIO_CTE_CLIENT;
  auto *fs_client = CLIO_CFS_CLIENT;
  // Retry the attach for a bounded window. A client can come up before the
  // chimod has registered its cache root, so "not attached yet" is a legal
  // transient -- but "never attaches" is a failure, and this must not be
  // downgraded to a skip: a silently disabled cache would make every
  // assertion below pass on the RPC path and prove nothing.
  for (int i = 0; i < 100; ++i) {
    if ((cte_client->HasShmCache() || cte_client->AttachShmCache()) &&
        (fs_client->HasShmCache() || fs_client->AttachShmCache())) {
      break;
    }
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(cte_client->HasShmCache());
  REQUIRE(fs_client->HasShmCache());

  // Inspecting the mirror directly is not a read(2), so nothing has drained a
  // queued write for us. fsync first, or this races the write path when async
  // writes are enabled.
  REQUIRE(cfs_io->Sync(fd) == 0);

  clio::cte::filesystem::ShmFileRecord rec;
  REQUIRE(fs_client->TryGetFileRecordShm(kBackendPath, &rec));
  REQUIRE(rec.IsFastPathable());
  REQUIRE(rec.size_ == kFileSize);

  // The page payloads must be direct-readable, or the fast path can only ever
  // decline and the latency numbers below would be measuring the RPC path.
  clio::cte::core::ShmBlobRecord page0;
  REQUIRE(cte_client->TryGetBlobRecordShm(rec.tag_id_, "0", &page0));
  std::printf(
      "[#817] page 0: size=%llu covered=%llu blocks=%u flags=0x%x direct=%d "
      "(record %zu B)\n",
      static_cast<unsigned long long>(page0.total_size_),
      static_cast<unsigned long long>(page0.CoveredBytes()), page0.num_blocks_,
      page0.flags_, page0.IsDirectReadable() ? 1 : 0,
      sizeof(clio::cte::core::ShmBlobRecord));
  REQUIRE(page0.IsDirectReadable());

  // ---- correctness: fast path == RPC path, byte for byte ------------------
  struct Case {
    const char *what;
    clio::run::u64 off;
    size_t len;
  };
  const Case cases[] = {
      {"4K at 0", 0, 4096},
      {"4K mid-page", 12345, 4096},
      {"1 byte", 1048575, 1},
      {"crosses a page boundary", 1048576 - 100, 200},
      {"short read at EOF", kFileSize - 10, 4096},
      {"entirely past EOF", kFileSize + 1, 4096},
  };
  for (const Case &c : cases) {
    std::vector<char> fast(c.len, 0), rpc(c.len, 0);
    ssize_t nf = cfs_io->Pread(fd, fast.data(), c.len,
                               static_cast<off_t>(c.off));
    ssize_t nr = RpcRead(cfs_io->HandleOf(fd), c.off, rpc.data(), c.len);
    REQUIRE(nf >= 0);
    REQUIRE(nr == nf);  // the two paths must agree on the length...
    if (nf > 0) {
      REQUIRE(std::memcmp(fast.data(), rpc.data(),
                          static_cast<size_t>(nf)) == 0);  // ...and the bytes
    }
    // Compare against the source buffer -- the authoritative answer.
    clio::run::u64 expect = 0;
    if (c.off < kFileSize) {
      expect = std::min<clio::run::u64>(c.len, kFileSize - c.off);
    }
    REQUIRE(static_cast<clio::run::u64>(nf) == expect);
    if (expect > 0) {
      REQUIRE(std::memcmp(fast.data(), src.data() + c.off,
                          static_cast<size_t>(expect)) == 0);
    }
  }

  // ---- a shrink must not keep being served from the old mirror -----------
  // Truncate frees page blobs back to the bdev, where another blob can take
  // them. If the mirror kept the old size the client would happily read
  // through freed storage, and the placement_gen_ guard could not catch it
  // either -- an un-republished record shows the reader the same generation
  // twice. Both halves (fs record + blob record) are invalidated by the
  // runtime before the pages go away.
  {
    const clio::run::u64 kShrunk = 8192;
    REQUIRE(cfs_io->FtruncateFd(fd, static_cast<off_t>(kShrunk)) == 0);

    clio::cte::filesystem::ShmFileRecord shrunk;
    REQUIRE(fs_client->TryGetFileRecordShm(kBackendPath, &shrunk));
    REQUIRE(shrunk.size_ == kShrunk);

    std::vector<char> after(4096, 0x5A);
    // Inside the surviving prefix: still correct, still served.
    REQUIRE(cfs_io->Pread(fd, after.data(), 4096, 4096) == 4096);
    REQUIRE(std::memcmp(after.data(), src.data() + 4096, 4096) == 0);
    // Past the new EOF: nothing, not stale bytes.
    REQUIRE(cfs_io->Pread(fd, after.data(), 4096,
                          static_cast<off_t>(kShrunk)) == 0);

    // Restore the file for the latency measurement below.
    REQUIRE(cfs_io->Pwrite(fd, src.data(), kFileSize, 0) ==
            static_cast<ssize_t>(kFileSize));
  }

  // ---- latency: same 4 KiB read, both paths ------------------------------
  const size_t kIoSize = 4096;
  const int kIters = 20000;
  std::vector<char> buf(kIoSize);

  // Warm: first touch attaches the RAM bdev segment in this process.
  REQUIRE(cfs_io->Pread(fd, buf.data(), kIoSize, 0) ==
          static_cast<ssize_t>(kIoSize));

  double t0 = NowUs();
  for (int i = 0; i < kIters; ++i) {
    ssize_t n = cfs_io->Pread(fd, buf.data(), kIoSize, 0);
    if (n != static_cast<ssize_t>(kIoSize)) {
      REQUIRE(false);
    }
  }
  double shm_us = (NowUs() - t0) / kIters;

  // Fewer RPC iterations: at ~100 us each, 20000 would take half an hour.
  const int kRpcIters = 300;

  // Time the RPC path through the filesystem client directly, using the same
  // chimod handle the adapter holds, so the only difference between the two
  // measurements is which path serves the bytes.
  clio::run::u64 rpc_handle = cfs_io->HandleOf(fd);
  REQUIRE(rpc_handle != 0);

  REQUIRE(RpcRead(rpc_handle, 0, buf.data(), kIoSize) ==
          static_cast<ssize_t>(kIoSize));
  t0 = NowUs();
  for (int i = 0; i < kRpcIters; ++i) {
    if (RpcRead(rpc_handle, 0, buf.data(), kIoSize) !=
        static_cast<ssize_t>(kIoSize)) {
      REQUIRE(false);
    }
  }
  double rpc_us = (NowUs() - t0) / kRpcIters;

  std::printf(
      "\n[#817] clio-fs 4 KiB pread: SHM %.3f us vs RPC %.3f us (%.1fx)\n",
      shm_us, rpc_us, rpc_us / shm_us);
  std::printf("[#817]   (%.6f ms vs %.6f ms)\n", shm_us / 1000.0,
              rpc_us / 1000.0);

  // Two gates, neither of which is a wall-clock target.
  //
  // An absolute sub-microsecond assertion is the goal on real hardware (0.27
  // to 0.5 us measured on a dev box), but it is the wrong thing to enforce in
  // CI: on a virtualized macOS runner the SAME code measures 1.19 us while a
  // single RPC on that machine costs 52 MILLISECONDS. That is a slow machine,
  // not a broken fast path, and encoding one machine's clock as a correctness
  // condition just makes the suite flaky by runner class.
  //
  // What is machine-independent is that NO ROUND TRIP HAPPENED. The transport
  // floor alone is ~72 us, so anything in single-digit microseconds cannot
  // have gone to the runtime; and the fast path must beat the RPC path it
  // replaces by a wide margin on whatever hardware this is.
  REQUIRE(shm_us < 5.0);
  REQUIRE(rpc_us / shm_us > 20.0);

  cfs_io->Close(fd);
  cfs_io->RemovePath(kClioPath);
}

TEST_CASE("clio-fs async writes: RYOW, fsync, throughput", "[cfs][shm][noleak]") {
  REQUIRE(InitRuntime());
  auto *cfs_io = CLIO_CTE_CFS;
  REQUIRE(cfs_io != nullptr);

  const std::string wpath = "clio::/tmp/clio_cfs_async_write_test.dat";
  const std::string wbare = "/tmp/clio_cfs_async_write_test.dat";
  cfs_io->RemovePath(wpath);

  size_t kChunk = 64 * 1024;
  int kChunks = 128;
  if (const char *e = std::getenv("CFS_WTEST_CHUNK_KB")) {
    kChunk = static_cast<size_t>(std::atoi(e)) * 1024;
    kChunks = static_cast<int>((8ULL * 1024 * 1024) / kChunk);
  }
  std::vector<char> chunk(kChunk);
  for (size_t i = 0; i < kChunk; ++i) {
    chunk[i] = static_cast<char>((i * 17 + 3) % 253);
  }

  // ---- read-your-own-writes, with no fsync in between --------------------
  // The write is queued, so nothing has updated the file's size or its page
  // blobs when the read is issued. If the read did not wait for it, it would
  // either see a stale EOF (short read) or pre-write bytes.
  {
    int fd = cfs_io->Open(wpath, O_CREAT | O_RDWR | O_TRUNC, 0644);
    REQUIRE(fd >= 0);
    REQUIRE(cfs_io->Pwrite(fd, chunk.data(), kChunk, 0) ==
            static_cast<ssize_t>(kChunk));

    std::vector<char> back(kChunk, 0);
    REQUIRE(cfs_io->Pread(fd, back.data(), kChunk, 0) ==
            static_cast<ssize_t>(kChunk));
    REQUIRE(std::memcmp(back.data(), chunk.data(), kChunk) == 0);

    // ...and a read of a DIFFERENT region must still be correct.
    REQUIRE(cfs_io->Pwrite(fd, chunk.data(), kChunk,
                           static_cast<off_t>(kChunk)) ==
            static_cast<ssize_t>(kChunk));
    std::vector<char> back2(1024, 0);
    REQUIRE(cfs_io->Pread(fd, back2.data(), 1024,
                          static_cast<off_t>(kChunk + 512)) == 1024);
    REQUIRE(std::memcmp(back2.data(), chunk.data() + 512, 1024) == 0);

    // stat must report the size the completed write(2) calls established,
    // even though those writes may still be in flight.
    REQUIRE(cfs_io->SizeFd(fd) == static_cast<off_t>(2 * kChunk));
    REQUIRE(cfs_io->Sync(fd) == 0);
    REQUIRE(cfs_io->Close(fd) == 0);
  }

  // NOTE: there is deliberately no assertion that a BURST of overlapping
  // rewrites to one range, with no intervening fsync, ends on the last value.
  // It does not, reliably, and the client does not try to make it: same-blob
  // writes race the #680 write token, and AsyncWrite's future signals before
  // the write durably commits, so a "completed" earlier write can land after a
  // later one. Measured, 512 rewrites of one 4 KiB range, RPC read confirming
  // the loss is durable (not a read-mirror lag): the last write is lost ~30% of
  // runs even when the client serializes on the prior write's future, ~100%
  // without. Since that pattern does not occur in a real workload -- writes are
  // byte-range partial puts, sequential/random offsets are disjoint, and code
  // that depends on same-byte ordering fsyncs -- the client waits for no write
  // at all, and this gap is left to a runtime fix (#680). See DoWrite.

  // ---- throughput: async vs O_SYNC ---------------------------------------
  // Measure the OVERWRITE path in both modes. The first pass over a fresh file
  // allocates blocks and the second does not, so timing one mode on a new file
  // and the other on the warmed one compares allocation against overwrite --
  // not the thing under test. (That confound is why an earlier version of this
  // benchmark reported a 3x difference between two identical code paths.)
  auto write_pass = [&](int flags, bool timed) {
    int fd = cfs_io->Open(wpath, O_CREAT | O_WRONLY | flags, 0644);
    REQUIRE(fd >= 0);
    double t0 = NowUs();
    for (int i = 0; i < kChunks; ++i) {
      REQUIRE(cfs_io->Write(fd, chunk.data(), kChunk) ==
              static_cast<ssize_t>(kChunk));
    }
    REQUIRE(cfs_io->Sync(fd) == 0);  // fsync is part of the cost, not a dodge
    double us = NowUs() - t0;
    REQUIRE(cfs_io->Close(fd) == 0);
    return timed ? us : 0.0;
  };

  write_pass(0, /*timed=*/false);  // warm: allocate every page once
  double async_us = write_pass(0, true);
  double sync_us = write_pass(O_SYNC, true);

  std::printf(
      "\n[#817] clio-fs %d x %zu KiB overwrite+fsync: default %.3f ms vs "
      "O_SYNC %.3f ms (%.2fx)\n",
      kChunks, kChunk / 1024, async_us / 1000.0, sync_us / 1000.0,
      sync_us / async_us);

  // No throughput assertion, in either direction. The queued path is OFF by
  // default because it measured ~2.3x SLOWER than blocking writes (see
  // CfsIo::AsyncWritesEnabled for the numbers and the cause), so asserting a
  // win would be asserting something known to be false today, and asserting a
  // loss would bake in the very thing #784 is expected to fix. The number is
  // printed so a future run can see it move.
  //
  // What IS asserted, above and below, is the CONTRACT: read-your-own-writes
  // without an intervening fsync, stat reflecting completed write(2) calls,
  // fsync/close draining and reporting, and the bytes surviving a reopen --
  // and those run in whichever mode the build defaults to.

  // ---- the bytes survive a close/reopen ----------------------------------
  {
    int fd = cfs_io->Open(wpath, O_RDONLY, 0644);
    REQUIRE(fd >= 0);
    REQUIRE(cfs_io->SizeFd(fd) ==
            static_cast<off_t>(static_cast<size_t>(kChunks) * kChunk));
    std::vector<char> back(kChunk, 0);
    for (int i : {0, kChunks / 2, kChunks - 1}) {
      REQUIRE(cfs_io->Pread(fd, back.data(), kChunk,
                            static_cast<off_t>(static_cast<size_t>(i) *
                                               kChunk)) ==
              static_cast<ssize_t>(kChunk));
      REQUIRE(std::memcmp(back.data(), chunk.data(), kChunk) == 0);
    }
    REQUIRE(cfs_io->Close(fd) == 0);
  }

  cfs_io->RemovePath(wpath);
  (void)wbare;
}

TEST_CASE("clio-fs SHM path under concurrent readers and writers",
          "[cfs][shm][noleak]") {
  REQUIRE(InitRuntime());
  auto *cfs_io = CLIO_CTE_CFS;
  REQUIRE(cfs_io != nullptr);
  if (CLIO_CPU_IPC == nullptr ||
      CLIO_CPU_IPC->GetMetadataAllocator() == nullptr) {
    std::printf("\n[#817] SKIP (concurrency): no metadata segment here\n");
    return;
  }

  // The fast path is reached from arbitrary application threads through the
  // POSIX/STDIO interceptors, and it touches process-wide state that was NOT
  // originally guarded: the CTE client's attached-RAM-bdev cache. Two threads
  // racing the first read of a target rehashed that map concurrently, which is
  // a use-after-free -- a segfault on the hot path, not a wrong answer. This
  // test exists to make that reachable rather than theoretical.
  const std::string cpath = "clio::/tmp/clio_cfs_concurrent_test.dat";
  const size_t kSize = 2 * 1024 * 1024;  // spans 2 pages
  std::vector<char> src(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    src[i] = static_cast<char>((i * 7 + 11) % 251);
  }

  cfs_io->RemovePath(cpath);
  int wfd = cfs_io->Open(cpath, O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(wfd >= 0);
  REQUIRE(cfs_io->Write(wfd, src.data(), kSize) ==
          static_cast<ssize_t>(kSize));
  REQUIRE(cfs_io->Sync(wfd) == 0);

  const int kThreads = 8;
  const int kItersPerThread = 400;
  std::atomic<int> mismatches{0};
  std::atomic<int> errors{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      // Each thread opens its OWN descriptor, so they contend on the fd table,
      // the write windows and the shared client caches simultaneously.
      int fd = cfs_io->Open(cpath, O_RDWR, 0644);
      if (fd < 0) {
        ++errors;
        return;
      }
      std::vector<char> buf(4096);
      for (int i = 0; i < kItersPerThread; ++i) {
        clio::run::u64 off =
            (static_cast<clio::run::u64>(i) * 4096 * (t + 1)) % (kSize - 4096);
        ssize_t n = cfs_io->Pread(fd, buf.data(), 4096,
                                  static_cast<off_t>(off));
        if (n != 4096) {
          ++errors;
          break;
        }
        if (std::memcmp(buf.data(), src.data() + off, 4096) != 0) {
          ++mismatches;
          break;
        }
        // Every eighth iteration, write back the same bytes. Concurrent
        // writers keep the mirror, the write windows and the placement
        // generation moving underneath the readers.
        if ((i & 7) == 0) {
          if (cfs_io->Pwrite(fd, buf.data(), 4096,
                             static_cast<off_t>(off)) != 4096) {
            ++errors;
            break;
          }
        }
      }
      cfs_io->Sync(fd);
      cfs_io->Close(fd);
    });
  }
  for (auto &th : threads) {
    th.join();
  }

  std::printf("[#817] concurrency: %d threads x %d reads, %d mismatches, "
              "%d errors\n",
              kThreads, kItersPerThread, mismatches.load(), errors.load());
  REQUIRE(mismatches.load() == 0);
  REQUIRE(errors.load() == 0);

  cfs_io->Close(wfd);
  cfs_io->RemovePath(cpath);
}

/**
 * write(2) always succeeds; a queued write's failure surfaces at fsync/close.
 *
 * Deliberately runs LAST: it writes until the tier refuses, and the space it
 * consumes is not necessarily all returned, so anything after it in the same
 * process would be running against a full target.
 *
 * The invariant asserted here does not depend on whether the tier actually
 * fills within the cap -- write(2) returning the full count is required
 * either way. What filling adds is the second half of the contract: the
 * failure is not lost, and an intervening size query (which drains the write
 * window) does not swallow it before fsync can report it.
 */
TEST_CASE("clio-fs writes always succeed, errors land on fsync/close",
          "[cfs][shm][noleak]") {
  REQUIRE(InitRuntime());
  auto *cfs_io = CLIO_CTE_CFS;
  REQUIRE(cfs_io != nullptr);

  const std::string path = "clio::/tmp/clio_cfs_write_always_ok.dat";
  cfs_io->RemovePath(path);
  int fd = cfs_io->Open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  REQUIRE(fd >= 0);

  // 4 KiB is the granularity a POSIX application actually writes at, and the
  // one that amplifies worst against a RAM target (measured: ~100 MiB of
  // logical 4 KiB writes fills a 2 GiB target), so it reaches the interesting
  // case fastest.
  const size_t kBuf = 4096;
  const int kMaxWrites = 16384;      // 64 MiB, the full window's worth
  const int kCheckEvery = 256;       // ~1 MiB
  std::vector<char> buf(kBuf, 'w');

  int writes = 0;
  int failed_writes = 0;
  int fsync_failed_after = -1;
  for (int i = 0; i < kMaxWrites; ++i) {
    ssize_t n = cfs_io->Write(fd, buf.data(), kBuf);
    if (n != static_cast<ssize_t>(kBuf)) {
      ++failed_writes;
      break;
    }
    ++writes;
    if ((i + 1) % kCheckEvery == 0) {
      // A size query drains the window. It must NOT consume the latched
      // error -- if it did, the fsync below would report success over a file
      // whose bytes never landed.
      REQUIRE(cfs_io->SizeFd(fd) >= 0);
      if (cfs_io->Sync(fd) != 0) {
        REQUIRE(errno == EIO);
        fsync_failed_after = writes;
        break;
      }
    }
  }

  // THE CONTRACT: no write(2) failed, whatever happened underneath.
  REQUIRE(failed_writes == 0);
  REQUIRE(writes > 0);

  int close_rc = cfs_io->Close(fd);
  if (fsync_failed_after >= 0) {
    std::printf("[#817] %d x 4 KiB writes all returned success; the tier "
                "refused at ~%.1f MiB and fsync reported it (close=%d)\n",
                writes, writes * kBuf / (1024.0 * 1024.0), close_rc);
  } else {
    std::printf("[#817] %d x 4 KiB writes all returned success; tier absorbed "
                "%.1f MiB without refusing (close=%d)\n",
                writes, writes * kBuf / (1024.0 * 1024.0), close_rc);
    REQUIRE(close_rc == 0);
  }

  cfs_io->RemovePath(path);
}

SIMPLE_TEST_MAIN()
