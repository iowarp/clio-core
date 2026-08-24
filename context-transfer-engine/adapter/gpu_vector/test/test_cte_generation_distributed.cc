/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Distributed generational put/get -- CTE only, no GPU, no benchmark.
 *
 * Two nodes, ONE blob, two non-overlapping regions. Each iteration a node
 * writes its own region and then reads the OTHER node's region, naming the
 * generation both sides are on. The regions SWAP every iteration, so neither
 * node can be reading a region it just wrote, and a cached or stale answer
 * shows up as the wrong iteration's bytes.
 *
 *   iter 0:  P1 put [0,R)      P2 put [R,2R)
 *            P1 get [R,2R)     P2 get [0,R)
 *   iter 1:  P1 put [R,2R)     P2 put [0,R)
 *            P1 get [0,R)      P2 get [R,2R)
 *
 * There is deliberately NO barrier between the put and the get: the
 * generation is the only thing ordering them. That is the claim under test.
 *
 * WHAT IS ASSERTED. A generational get promises AT LEAST the generation
 * asked for, so a reader may legitimately see a NEWER iteration if the peer
 * has raced ahead. What it must never see is an OLDER one, and what it sees
 * must be one whole snapshot rather than a mix -- so each region carries the
 * iteration that wrote it in its first element, and the check is
 * "seen >= asked, and the rest of the region agrees with seen".
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#if !CTP_IS_DEVICE_PASS

namespace core = clio::cte::core;
using clio::run::u32;
using clio::run::u64;

namespace {

constexpr u64 kRegionElems = 1024;                       // 4 KiB per region
constexpr u64 kRegionBytes = kRegionElems * sizeof(u32);

/** Iteration-salted, so a copy from an earlier iteration fails loudly. */
u32 Pattern(int region, u64 i, int iter) {
  return static_cast<u32>(region * 2654435761ull + i * 40503ull +
                          static_cast<u64>(iter) * 0x9E3779B9ull + 1ull);
}

int EnvInt(const char *name, int dflt) {
  const char *e = std::getenv(name);
  return e != nullptr ? std::atoi(e) : dflt;
}

void Step(int node, const char *what) {
  std::fprintf(stderr, "[gen-node%d] STEP %s\n", node, what);
  std::fflush(stderr);
}

/** Cluster formation only -- never used to order a put against a get. */
bool StartBarrier(core::Client &cte, const core::TagId &tag, int node,
                  int nodes, int timeout_s) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    u32 val = 1;
    auto f = cte.AsyncPutBlob(tag, "genbar_" + std::to_string(node), 0,
                              sizeof(val), reinterpret_cast<char *>(&val),
                              1.0f);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > timeout_s) {
      std::fprintf(stderr, "[gen-node%d] barrier put failed\n", node);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  for (int peer = 1; peer <= nodes;) {
    u32 val = 0;
    auto f = cte.AsyncGetBlob(tag, "genbar_" + std::to_string(peer), 0,
                              sizeof(val), 0u,
                              reinterpret_cast<char *>(&val));
    f.Wait();
    if (f->GetReturnCode() == 0 && val == 1) {
      ++peer;
      continue;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > timeout_s) {
      std::fprintf(stderr, "[gen-node%d] start barrier timeout at %d\n", node,
                   peer);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return true;
}

}  // namespace

int main() {
  const int node = EnvInt("NODE_ID", 1);            // 1-based
  const int nodes = EnvInt("CLIO_NUM_CONTAINERS", 2);
  const int iters = EnvInt("GVGEN_ITERS", 8);
  const int timeout_s = EnvInt("GVGEN_BARRIER_TIMEOUT", 180);

  if (nodes != 2) {
    std::fprintf(stderr, "[gen-node%d] this test is written for 2 nodes\n",
                 node);
    return 1;
  }

  Step(node, "runtime-init");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "[gen-node%d] runtime init failed\n", node);
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[gen-node%d] cte client init failed\n", node);
    return 1;
  }

  core::Client cte(core::kCtePoolId);
  auto tagf = cte.AsyncGetOrCreateTag(std::string("gen_swap_tag"));
  tagf.Wait();
  if (tagf->GetReturnCode() != 0) {
    std::fprintf(stderr, "[gen-node%d] tag failed\n", node);
    return 1;
  }
  const core::TagId tag = tagf->tag_id_;
  const std::string blob = "gen_swap";

  Step(node, "barrier-start");
  if (!StartBarrier(cte, tag, node, nodes, timeout_s)) return 1;

  std::vector<u32> wbuf(kRegionElems), rbuf(kRegionElems);
  int failures = 0, stale = 0, torn = 0;

  for (int it = 0; it < iters; ++it) {
    const u64 gen = static_cast<u64>(it) + 1;   // 0 means "no generation"
    // Regions swap every iteration, so a node never reads what it just wrote.
    const int my_region = ((node - 1) + it) % 2;
    const int peer_region = 1 - my_region;

    // ---- write my region, published AS this generation ----
    for (u64 i = 0; i < kRegionElems; ++i) {
      wbuf[i] = (i == 0) ? static_cast<u32>(it) : Pattern(my_region, i, it);
    }
    core::Context put_ctx;
    put_ctx.op_flags_ |= core::Context::kGenerational;
    put_ctx.generation_ = gen;
    auto pf = cte.AsyncPutBlob(
        tag, blob, static_cast<u64>(my_region) * kRegionBytes, kRegionBytes,
        reinterpret_cast<char *>(wbuf.data()), 0.5f, put_ctx, 0u,
        clio::run::PoolQuery::Dynamic());
    pf.Wait();
    if (pf->GetReturnCode() != 0) {
      std::fprintf(stderr, "[gen-node%d] iter %d put rc=%d\n", node, it,
                   pf->GetReturnCode());
      ++failures;
      continue;
    }

    // ---- read the PEER's region, naming the same generation ----
    // No barrier above: if generations do not order this, the read lands
    // before the peer's put and returns the previous iteration's bytes.
    std::fill(rbuf.begin(), rbuf.end(), 0u);
    core::Context get_ctx;
    get_ctx.op_flags_ |= core::Context::kGenerational;
    get_ctx.generation_ = gen;
    auto gf = cte.AsyncGetBlob(
        tag, blob, static_cast<u64>(peer_region) * kRegionBytes, kRegionBytes,
        0u, reinterpret_cast<char *>(rbuf.data()),
        clio::run::PoolQuery::Dynamic(), get_ctx);
    gf.Wait();
    if (gf->GetReturnCode() != 0) {
      std::fprintf(stderr, "[gen-node%d] iter %d get rc=%d\n", node, it,
                   gf->GetReturnCode());
      ++failures;
      continue;
    }

    const int seen = static_cast<int>(rbuf[0]);
    int bad = 0;
    for (u64 i = 1; i < kRegionElems; ++i) {
      if (rbuf[i] != Pattern(peer_region, i, seen)) ++bad;
    }
    const bool is_stale = seen < it;
    if (is_stale) ++stale;
    if (bad != 0) ++torn;
    if (is_stale || bad != 0) ++failures;
    std::fprintf(stderr,
                 "[gen-node%d] iter %d gen %llu: wrote r%d, read r%d -> "
                 "iter %d %s%s\n",
                 node, it, (unsigned long long)gen, my_region, peer_region,
                 seen, is_stale ? "STALE " : "",
                 bad != 0 ? "TORN" : (is_stale ? "" : "ok"));
  }

  std::fprintf(stderr,
               "[gen-node%d] %s (%d failures: %d stale, %d torn, over %d "
               "iterations)\n",
               node, failures == 0 ? "PASS" : "FAIL", failures, stale, torn,
               iters);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return failures == 0 ? 0 : 1;
}
#else
int main() { return 0; }   // device pass: host-only test
#endif  // !CTP_IS_DEVICE_PASS
