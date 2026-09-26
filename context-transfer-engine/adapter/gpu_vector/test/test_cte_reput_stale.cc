/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * MINIMAL REPRODUCER for the durable stale-replica serve.
 *
 * The defect, as observed through lbann's distributed weight verification:
 * after a blob has been re-put many times, a CROSS-NODE plain GetBlob can be
 * served bytes from an OLD version with rc=0 -- and re-reading later still
 * returns the old bytes. Whether it strikes tracks HOST LOAD, which is why it
 * passed on idle boxes and failed on Delta and under parallel compiles.
 *
 * This strips away the GPU vector, the paging, the workload -- everything but
 * the CTE core client:
 *
 *   node 1 (writer), each round r:
 *     1. plain PutBlob of ONE data blob, every u32 stamped with r, with a
 *        score that DRIFTS round to round -- the flush path's behavior, and
 *        the suspected relocation trigger (see gpu-vector-flush-settle-races:
 *        "a drifting blob score relocates on reput; get reads stale replica")
 *     2. generational PutBlob of a tiny per-round signal blob
 *
 *   node 2 (reader), each round r:
 *     1. generational GetBlob of the round's signal -- this PROVABLY orders
 *        the read after the writer's data put returned rc=0 (the generational
 *        machinery polls until the writer's publish is served; it is the same
 *        primitive every green distributed gate leans on)
 *     2. plain GetBlob of the data blob, rc checked
 *     3. every u32 must be >= r's stamp. A byte from an older round after the
 *        round's signal was observed is THE DEFECT: stale bytes, rc=0.
 *
 * Both sides spin CPU burner threads for the duration: the race only loses
 * reliably on a loaded host.
 *
 * Runs under test/distributed_workloads/'s compose (it accepts the --nodes /
 * --node arguments that harness appends), or standalone with NODE_ID.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Compiled BY the CUDA compiler (add_cuda_executable), whose device pass
// member-checks host bodies; the CTE client does not exist there.
#if !CTP_IS_DEVICE_PASS

namespace core = clio::cte::core;
using clio::run::u32;
using clio::run::u64;

namespace {

int EnvInt(const char *name, int dflt) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != '\0') ? std::atoi(e) : dflt;
}

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/** Generational put: retried because the pool may not exist yet. */
bool GenPut(core::Client &cte, const core::TagId &tag, const std::string &name,
            char *bytes, size_t n, int timeout_s) {
  core::Context ctx;
  ctx.op_flags_ |= core::Context::kGenerational;
  ctx.generation_ = 1;
  const double t0 = Now();
  for (;;) {
    auto f = cte.AsyncPutBlob(tag, name, 0, n, bytes, 1.0f, ctx);
    f.Wait();
    if (f->GetReturnCode() == 0) return true;
    if (Now() - t0 > timeout_s) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

/** Generational get: polls until the writer's put has landed. */
bool GenGet(core::Client &cte, const core::TagId &tag, const std::string &name,
            char *bytes, size_t n, int timeout_s) {
  core::Context ctx;
  ctx.op_flags_ |= core::Context::kGenerational;
  ctx.generation_ = 1;
  const double t0 = Now();
  for (;;) {
    auto f = cte.AsyncGetBlob(tag, name, 0, n, 0u, bytes,
                              clio::run::PoolQuery::Dynamic(), ctx);
    f.Wait();
    if (f->GetReturnCode() == 0) return true;
    if (Now() - t0 > timeout_s) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace

int main(int argc, char **argv) {
  int node = EnvInt("NODE_ID", 1) - 1;       // 0-based below
  int nodes = EnvInt("CLIO_NUM_CONTAINERS", 2);
  int rounds = EnvInt("REPUT_ROUNDS", 200);
  int burners = EnvInt("REPUT_BURNERS", 6);
  int timeout_s = EnvInt("REPUT_TIMEOUT", 240);
  // 256 KB: big enough to take the buffered data path, small enough that 200
  // rounds finish fast.
  const size_t kBytes = 256 * 1024;
  const size_t kElems = kBytes / sizeof(u32);

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? std::atoi(argv[++i]) : 0; };
    if (a == "--nodes") nodes = next();
    else if (a == "--node") node = next();
    else if (a == "--rounds") rounds = next();
    else if (a == "--burners") burners = next();
  }
  if (nodes != 2) {
    std::fprintf(stderr, "[reput] 2 nodes exactly (got %d)\n", nodes);
    return 2;
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "[reput-n%d] runtime init failed\n", node);
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[reput-n%d] cte client init failed\n", node);
    return 1;
  }
  core::Client cte(core::kCtePoolId);
  auto tagf = cte.AsyncGetOrCreateTag(std::string("reput_stale_tag"));
  tagf.Wait();
  if (tagf->GetReturnCode() != 0) {
    std::fprintf(stderr, "[reput-n%d] tag failed\n", node);
    return 1;
  }
  const core::TagId tag = tagf->tag_id_;

  // THE LOAD IS PART OF THE REPRODUCER. Idle hosts win the race that this
  // test exists to lose.
  std::atomic<bool> stop{false};
  std::vector<std::thread> burn;
  for (int b = 0; b < burners; ++b) {
    burn.emplace_back([&stop] {
      volatile u64 x = 0;
      while (!stop.load(std::memory_order_relaxed)) x += 1;
    });
  }

  std::vector<u32> buf(kElems);
  int rc = 0;

  if (node == 0) {
    // ---- WRITER ----
    for (int r = 1; r <= rounds && rc == 0; ++r) {
      for (size_t i = 0; i < kElems; ++i) buf[i] = static_cast<u32>(r);
      // Drifting score, like the flush path: suspected relocation trigger.
      const float score = 0.1f + 0.8f * static_cast<float>(r % 10) / 9.0f;
      auto f = cte.AsyncPutBlob(tag, "reput_data", 0, kBytes,
                                reinterpret_cast<char *>(buf.data()), score);
      f.Wait();
      if (f->GetReturnCode() != 0) {
        std::fprintf(stderr, "[reput-n0] round %d put rc=%d\n", r,
                     f->GetReturnCode());
        rc = 1;
        break;
      }
      // Signal r, generationally: readable only once THIS put has landed --
      // and it is issued after the data put above returned rc=0, so a reader
      // holding signal r is provably reading data put at round >= r.
      u32 stamp = static_cast<u32>(r);
      if (!GenPut(cte, tag, "reput_sig_" + std::to_string(r),
                  reinterpret_cast<char *>(&stamp), sizeof(stamp),
                  timeout_s)) {
        std::fprintf(stderr, "[reput-n0] round %d signal put timeout\n", r);
        rc = 1;
      }
    }
    // Hold the runtime up until the reader is done (it reads our blobs).
    u32 done = 0;
    GenGet(cte, tag, "reput_reader_done", reinterpret_cast<char *>(&done),
           sizeof(done), timeout_s);
  } else {
    // ---- READER ----
    int stale_rounds = 0, worst_lag = 0;
    for (int r = 1; r <= rounds; ++r) {
      u32 stamp = 0;
      if (!GenGet(cte, tag, "reput_sig_" + std::to_string(r),
                  reinterpret_cast<char *>(&stamp), sizeof(stamp),
                  timeout_s)) {
        std::fprintf(stderr, "[reput-n1] round %d signal timeout\n", r);
        rc = 1;
        break;
      }
      std::fill(buf.begin(), buf.end(), 0u);
      auto f = cte.AsyncGetBlob(tag, "reput_data", 0, kBytes, 0u,
                                reinterpret_cast<char *>(buf.data()),
                                clio::run::PoolQuery::Dynamic());
      f.Wait();
      if (f->GetReturnCode() != 0) {
        std::fprintf(stderr, "[reput-n1] round %d get rc=%d\n", r,
                     f->GetReturnCode());
        rc = 1;
        break;
      }
      u32 lo = buf[0], hi = buf[0];
      for (size_t i = 1; i < kElems; ++i) {
        lo = std::min(lo, buf[i]);
        hi = std::max(hi, buf[i]);
      }
      if (lo < static_cast<u32>(r)) {
        // THE DEFECT. Signal r was generationally observed, so the data put
        // of round r returned rc=0 BEFORE this get was issued -- yet the get
        // returned bytes from round `lo` with rc=0.
        ++stale_rounds;
        worst_lag = std::max(worst_lag, r - static_cast<int>(lo));
        std::fprintf(stderr,
                     "[reput-n1] STALE SERVE round %d: got rounds [%u, %u] "
                     "with rc=0 (lag %d)\n",
                     r, lo, hi, r - static_cast<int>(lo));
      }
    }
    u32 done = 1;
    GenPut(cte, tag, "reput_reader_done", reinterpret_cast<char *>(&done),
           sizeof(done), timeout_s);
    if (stale_rounds > 0) {
      std::fprintf(stderr,
                   "[reput-n1] FAIL: %d of %d rounds served stale bytes with "
                   "rc=0 (worst lag %d rounds)\n",
                   stale_rounds, rounds, worst_lag);
      rc = 1;
    } else if (rc == 0) {
      std::fprintf(stderr, "[reput-n1] PASS: %d rounds, no stale serve\n",
                   rounds);
    }
  }

  stop.store(true);
  for (auto &t : burn) t.join();
  std::fprintf(stderr, "[reput-n%d] exit rc=%d\n", node, rc);
  return rc;
}

#else
int main() { return 0; }
#endif  // !CTP_IS_DEVICE_PASS
