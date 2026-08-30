/**
 * Cross-node primitives shared by the paged benchmarks.
 *
 * The paged workloads distribute through the CTE, not through MPI: a vector's
 * pages ARE blobs, and blobs hash across the cluster, so the paging path is
 * already the transport. What the benches still need on top is a way to
 * combine per-node partials at a gate boundary, and a way to agree on which
 * slice of the global problem each node owns. Both live here so the five
 * workloads cannot drift apart on either.
 *
 * Extracted from the eternia-MD bench, which had the only working version.
 */
#ifndef CLIO_CTE_GPU_VECTOR_BENCH_DIST_H_
#define CLIO_CTE_GPU_VECTOR_BENCH_DIST_H_

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <clio_cte/core/core_client.h>

namespace clio_bench_dist {

using u32 = unsigned int;
using u64 = unsigned long long;

/** Which slice of a global range of `total` items this node owns. */
struct Shard {
  u64 begin = 0;
  u64 end = 0;
  u64 count() const { return end - begin; }
};

/**
 * Contiguous shard of [0, total).
 *
 * The LAST node absorbs the remainder, matching the MPI baselines exactly.
 * That choice is not cosmetic: the paged and MPI editions are gated against
 * the same reference, so an off-by-one in who owns the tail shows up as a
 * numerical difference rather than as a crash, and is correspondingly hard to
 * find.
 */
inline Shard ShardOf(u64 total, u32 node, u32 nodes) {
  if (nodes <= 1) return Shard{0, total};
  const u64 per = total / nodes;
  const u64 begin = static_cast<u64>(node) * per;
  const u64 end = (node == nodes - 1) ? total : begin + per;
  return Shard{begin, end};
}

/**
 * Sum `n` doubles across the nodes, through the CTE.
 *
 * Each node publishes its partial under its own name and then reads every
 * peer's. `round` keeps one gate's partials from being confused with the
 * next's, and `prefix` keeps two different reductions in the same run apart.
 *
 * The put is retried because a node may arrive before the pool exists; the
 * get is retried because a peer may not have published yet. That polling is
 * what makes this a reduction AND a barrier, which is what a gate boundary
 * wants.
 *
 * GENERATIONAL, and that is load-bearing. A plain put/get pair has no
 * readiness gate: the put can report complete while a peer's get still reads
 * unpublished bytes. In the MD bench that was measured as a node printing
 * exactly HALF the true energy -- its own partial plus a peer read of zeros.
 * Each round's blob is written once, so generation 1 is the whole protocol:
 * the get is not served until the writer's put landed.
 */
inline bool ReduceSum(clio::cte::core::Client &cte,
                      const clio::cte::core::TagId &tag, u32 node, u32 nodes,
                      u64 round, double *vals, int n,
                      const char *prefix = "gvred", int timeout_s = 120) {
  if (nodes <= 1) return true;
  const auto t0 = std::chrono::steady_clock::now();
  const auto expired = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() > timeout_s;
  };
  const std::string stem =
      std::string(prefix) + "_" + std::to_string(round) + "_";
  const std::string mine = stem + std::to_string(node);

  clio::cte::core::Context ctx;
  ctx.op_flags_ |= clio::cte::core::Context::kGenerational;
  ctx.generation_ = 1;

  for (;;) {
    auto f = cte.AsyncPutBlob(tag, mine, 0, n * sizeof(double),
                              reinterpret_cast<char *>(vals), 1.0f, ctx);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (expired()) {
      std::fprintf(stderr, "  reduce[%s]: timed out publishing node %u\n",
                   prefix, node);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // ACCUMULATE IN STRICT ASCENDING NODE ORDER, including this node's own
  // partial in its proper place. Seeding `total` with the local value and then
  // adding peers 0,1,2... makes node 2 sum in the order 2,0,1 -- so each node
  // reduces the same set of doubles in a DIFFERENT order, and float addition
  // is not associative. Every node then computes a slightly different total
  // and they drift apart over iterations, which is both a correctness bug and
  // one that hides as "nondeterminism". Ascending order also matches the MPI
  // baselines, which combine partials in rank order on purpose.
  std::vector<double> total(static_cast<size_t>(n), 0.0);
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) {
      for (int i = 0; i < n; ++i) total[i] += vals[i];
      continue;
    }
    const std::string name = stem + std::to_string(peer);
    std::vector<double> got(static_cast<size_t>(n), 0.0);
    for (;;) {
      auto f = cte.AsyncGetBlob(tag, name, 0, n * sizeof(double), 0u,
                                reinterpret_cast<char *>(got.data()),
                                clio::run::PoolQuery::Dynamic(), ctx);
      f.Wait();
      if (f->GetReturnCode() == 0) break;
      if (expired()) {
        std::fprintf(stderr, "  reduce[%s]: timed out waiting for node %u\n",
                     prefix, peer);
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for (int i = 0; i < n; ++i) total[i] += got[i];
  }
  for (int i = 0; i < n; ++i) vals[i] = total[i];
  return true;
}

/**
 * Sum `n` u64s across the nodes, exactly.
 *
 * NOT ReduceSum with a cast. A double holds only 53 bits of mantissa, so a
 * checksum over a large weight set silently loses its low bits above 2^53 --
 * and the gates that consume these are integer-exact BECAUSE integer addition
 * commutes, which is the whole reason those benchmarks can claim bit-equality
 * at any node count. Rounding that to double would throw away the property
 * being tested.
 *
 * Order does not matter here (integer addition is associative), so unlike the
 * double version this does not need to fix an accumulation order.
 */
inline bool ReduceSumU64(clio::cte::core::Client &cte,
                         const clio::cte::core::TagId &tag, u32 node, u32 nodes,
                         u64 round, u64 *vals, int n,
                         const char *prefix = "gvred64", int timeout_s = 120) {
  if (nodes <= 1) return true;
  const auto t0 = std::chrono::steady_clock::now();
  const auto expired = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() > timeout_s;
  };
  const std::string stem =
      std::string(prefix) + "_" + std::to_string(round) + "_";
  const std::string mine = stem + std::to_string(node);

  clio::cte::core::Context ctx;
  ctx.op_flags_ |= clio::cte::core::Context::kGenerational;
  ctx.generation_ = 1;

  for (;;) {
    auto f = cte.AsyncPutBlob(tag, mine, 0, n * sizeof(u64),
                              reinterpret_cast<char *>(vals), 1.0f, ctx);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (expired()) {
      std::fprintf(stderr, "  reduce64[%s]: timed out publishing node %u\n",
                   prefix, node);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  std::vector<u64> total(vals, vals + n);
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
    const std::string name = stem + std::to_string(peer);
    std::vector<u64> got(static_cast<size_t>(n), 0ull);
    for (;;) {
      auto f = cte.AsyncGetBlob(tag, name, 0, n * sizeof(u64), 0u,
                                reinterpret_cast<char *>(got.data()),
                                clio::run::PoolQuery::Dynamic(), ctx);
      f.Wait();
      if (f->GetReturnCode() == 0) break;
      if (expired()) {
        std::fprintf(stderr, "  reduce64[%s]: timed out waiting for node %u\n",
                     prefix, peer);
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for (int i = 0; i < n; ++i) total[i] += got[i];
  }
  for (int i = 0; i < n; ++i) vals[i] = total[i];
  return true;
}

/**
 * Wait for every outstanding put to land, THEN drop the cache.
 *
 * ClearCache on its own is not an invalidate here: it zeroes `flushing` and
 * drops `data` unconditionally, so a page whose put is still outstanding has
 * that put discarded, and a task completing afterwards can land in a frame
 * already reassigned to another page. In grayscott that was invisible for two
 * steps -- nothing is in flight when the clear runs early on -- and then
 * diverged at step 3, as a rel 4.6e-3 checksum error with every counter clean.
 *
 * A node that faulted in a peer's page keeps that frame resident and will
 * happily re-read its own stale copy forever, so SOME invalidation is
 * required; it just has to wait its turn. Returns false if the cache never
 * quiesces, which is a real failure and not something to page over.
 */
template <typename VecT>
inline bool SettleAndInvalidate(VecT &vec, int gpu_id = 0,
                                int timeout_ms = 30000) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const auto tbl = vec.ReadTable(gpu_id);
    bool busy = false;
    for (const auto &p : tbl) {
      if (p.flushing || p.fetching) { busy = true; break; }
    }
    if (!busy) break;
    if (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count() > timeout_ms) {
      std::fprintf(stderr, "  settle: cache still busy after %d ms\n",
                   timeout_ms);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  vec.ClearCache(gpu_id);
  return true;
}

/**
 * Barrier across the nodes. A zero-length ReduceSum would be a no-op, so this
 * reduces one dummy value; the cost is one blob per node per call.
 */
inline bool Barrier(clio::cte::core::Client &cte,
                    const clio::cte::core::TagId &tag, u32 node, u32 nodes,
                    u64 round, const char *prefix = "gvbar",
                    int timeout_s = 120) {
  double one = 1.0;
  return ReduceSum(cte, tag, node, nodes, round, &one, 1, prefix, timeout_s);
}

}  // namespace clio_bench_dist

#endif  // CLIO_CTE_GPU_VECTOR_BENCH_DIST_H_
