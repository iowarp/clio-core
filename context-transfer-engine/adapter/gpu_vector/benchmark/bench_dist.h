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

  std::vector<double> total(vals, vals + n);
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
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
