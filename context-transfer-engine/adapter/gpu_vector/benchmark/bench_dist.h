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

/** One peer blob to read: its name, where it lands, and its size. */
struct PeerGet {
  std::string name;
  char *dst;
  u64 bytes;
  u64 off = 0;  ///< byte offset inside the peer's blob
};

/**
 * Read every peer blob with ALL the gets in flight at once, then retry the
 * ones that failed (a peer that has not created the pool or published yet).
 *
 * The serial form -- issue a get, wait, issue the next -- costs one full
 * round trip plus the runtime's generation wait PER PEER, so a 16-node
 * reduction of 528 doubles took ~55 ms per kmeans iteration against ~9 ms
 * for MPI_Allreduce. Issued together, the waits overlap.
 *  @param cte,tag  the exchange's CTE client and tag
 *  @param ctx      generational context (generation 1)
 *  @param gets     the blobs to read
 *  @param expired  returns true once the caller's timeout has passed
 *  @param what,prefix  names for the timeout message
 *  @return true when every get succeeded before the timeout */
template <typename Expired>
inline bool GetPeers(clio::cte::core::Client &cte,
                     const clio::cte::core::TagId &tag,
                     const clio::cte::core::Context &ctx,
                     const std::vector<PeerGet> &gets, Expired expired,
                     const char *what, const char *prefix) {
  std::vector<size_t> todo(gets.size());
  for (size_t i = 0; i < gets.size(); ++i) todo[i] = i;
  while (!todo.empty()) {
    using FutT = decltype(cte.AsyncGetBlob(tag, gets[0].name, 0, 0, 0u,
                                           gets[0].dst,
                                           clio::run::PoolQuery::Dynamic(),
                                           ctx));
    std::vector<FutT> futs;
    futs.reserve(todo.size());
    for (size_t i : todo) {
      futs.push_back(cte.AsyncGetBlob(tag, gets[i].name, gets[i].off,
                                      gets[i].bytes, 0u,
                                      gets[i].dst,
                                      clio::run::PoolQuery::Dynamic(), ctx));
    }
    std::vector<size_t> again;
    for (size_t j = 0; j < futs.size(); ++j) {
      futs[j].Wait();
      if (futs[j]->GetReturnCode() != 0) again.push_back(todo[j]);
    }
    todo.swap(again);
    if (todo.empty()) break;
    if (expired()) {
      std::fprintf(stderr, "  %s[%s]: timed out waiting for %s\n", what,
                   prefix, gets[todo[0]].name.c_str());
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
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
      std::fprintf(stderr, "  reduce[%s]: timed out publishing node %u (last put rc=%u)\n",
                   prefix, node, f->GetReturnCode());
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
  std::vector<double> got(static_cast<size_t>(n) * nodes, 0.0);
  std::vector<PeerGet> gets;
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
    gets.push_back({stem + std::to_string(peer),
                    reinterpret_cast<char *>(got.data() + peer * n),
                    n * sizeof(double)});
  }
  if (!GetPeers(cte, tag, ctx, gets, expired, "reduce", prefix)) return false;
  std::vector<double> total(static_cast<size_t>(n), 0.0);
  for (u32 peer = 0; peer < nodes; ++peer) {
    const double *src = (peer == node) ? vals : got.data() + peer * n;
    for (int i = 0; i < n; ++i) total[i] += src[i];
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
      std::fprintf(stderr, "  reduce64[%s]: timed out publishing node %u (last put rc=%u)\n",
                   prefix, node, f->GetReturnCode());
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  std::vector<u64> got(static_cast<size_t>(n) * nodes, 0ull);
  std::vector<PeerGet> gets;
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
    gets.push_back({stem + std::to_string(peer),
                    reinterpret_cast<char *>(got.data() + peer * n),
                    n * sizeof(u64)});
  }
  if (!GetPeers(cte, tag, ctx, gets, expired, "reduce64", prefix)) {
    return false;
  }
  std::vector<u64> total(vals, vals + n);
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
    for (int i = 0; i < n; ++i) total[i] += got[peer * n + i];
  }
  for (int i = 0; i < n; ++i) vals[i] = total[i];
  return true;
}

/**
 * Check that every node holds the same u64 after a reduction.
 *
 * A reduction that returns a different total on one node is invisible to a
 * per-node gate: that node compares its own reduced `got` against its own
 * reduced `want`, and if both were shifted the same way (a peer's pair read
 * twice, another's never) they still agree. Measured on the E4 weights
 * DAOS-heavy cell: three nodes at one total, the fourth at another, all four
 * printing checksum=OK. This sums the value over the nodes with a second
 * round and fails unless the sum is exactly `nodes` times this node's value,
 * which is only true when every node holds the same number.
 *  @param cte,tag,node,nodes,round  as ReduceSumU64
 *  @param value   this node's reduced value
 *  @param prefix  blob-name prefix for the agreement round
 *  @return true when all nodes agree (or the round itself failed and was
 *          reported), false when the values differ */
inline bool AgreeU64(clio::cte::core::Client &cte,
                     const clio::cte::core::TagId &tag, u32 node, u32 nodes,
                     u64 round, u64 value, const char *prefix = "gvagree",
                     int timeout_s = 120) {
  if (nodes <= 1) return true;
  u64 sum = value;
  if (!ReduceSumU64(cte, tag, node, nodes, round, &sum, 1, prefix,
                    timeout_s)) {
    return false;
  }
  if (sum != value * static_cast<u64>(nodes)) {
    std::fprintf(stderr,
                 "  agree[%s]: node %u holds %llu but the nodes sum to %llu "
                 "(%u x mine = %llu): the reduction disagreed across nodes\n",
                 prefix, node, (unsigned long long)value,
                 (unsigned long long)sum, nodes,
                 (unsigned long long)(value * static_cast<u64>(nodes)));
    return false;
  }
  return true;
}

/**
 * All-gather a contiguous float slice.
 *
 * Every node owns [lo, hi) of the same array and needs the whole thing. Each
 * publishes its own slice under its own name and reads every peer's straight
 * into that peer's offset -- a slice exchange, NOT a zero-padded ReduceSum.
 * The sum trick would work (zero what you do not own, add), but it moves the
 * whole array per node instead of one slice, and it silently produces garbage
 * if a caller forgets to zero, which is not a failure mode worth inviting.
 *
 * Generational for the same reason the reductions are: a plain put/get pair
 * has no readiness gate, so a peer's get can be served before its writer's put
 * has landed and read whatever was there before.
 */
inline bool AllGatherF32(clio::cte::core::Client &cte,
                         const clio::cte::core::TagId &tag, u32 node, u32 nodes,
                         u64 round, float *data, u64 lo, u64 hi,
                         const u64 *los, const u64 *his,
                         const char *prefix = "gvgat", int timeout_s = 120) {
  if (nodes <= 1) return true;
  const auto t0 = std::chrono::steady_clock::now();
  const auto expired = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() > timeout_s;
  };
  const std::string stem =
      std::string(prefix) + "_" + std::to_string(round) + "_";

  clio::cte::core::Context ctx;
  ctx.op_flags_ |= clio::cte::core::Context::kGenerational;
  ctx.generation_ = 1;

  const u64 mine_bytes = (hi - lo) * sizeof(float);
  for (;;) {
    auto f = cte.AsyncPutBlob(tag, stem + std::to_string(node), 0, mine_bytes,
                              reinterpret_cast<char *>(data + lo), 1.0f, ctx);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (expired()) {
      std::fprintf(stderr, "  gather[%s]: timed out publishing node %u (last put rc=%u)\n",
                   prefix, node, f->GetReturnCode());
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::vector<PeerGet> gets;
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
    gets.push_back({stem + std::to_string(peer),
                    reinterpret_cast<char *>(data + los[peer]),
                    (his[peer] - los[peer]) * sizeof(float)});
  }
  if (!GetPeers(cte, tag, ctx, gets, expired, "gather", prefix)) return false;
  return true;
}

/**
 * All-to-all of float slices ("each node gets only its rows").
 *
 * Every node holds a vector `mine` of `nodes * slice` floats laid out by
 * DESTINATION: elements [q*slice, (q+1)*slice) are node q's. It publishes the
 * whole vector once under its own name, then reads, from every peer, only the
 * slice addressed to it -- `slice` floats at byte offset node*slice*4 of the
 * peer's blob -- into out[p*slice, (p+1)*slice). Its own slice is copied
 * locally. This is MPI_Alltoall over the CTE: the volume each node pulls is
 * nodes*slice floats instead of the all-gather's nodes*nodes*slice.
 *  @param cte,tag,node,nodes,round  as AllGatherF32
 *  @param mine   this node's nodes*slice floats, by destination
 *  @param slice  floats per destination
 *  @param out    nodes*slice floats, by source node (ascending)
 *  @param prefix blob-name prefix
 *  @return true when every slice arrived before the timeout */
inline bool SliceExchangeF32(clio::cte::core::Client &cte,
                             const clio::cte::core::TagId &tag, u32 node,
                             u32 nodes, u64 round, const float *mine,
                             u64 slice, float *out,
                             const char *prefix = "gvslx",
                             int timeout_s = 120) {
  std::memcpy(out + static_cast<u64>(node) * slice,
              mine + static_cast<u64>(node) * slice, slice * sizeof(float));
  if (nodes <= 1) return true;
  const auto t0 = std::chrono::steady_clock::now();
  const auto expired = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() > timeout_s;
  };
  const std::string stem =
      std::string(prefix) + "_" + std::to_string(round) + "_";
  clio::cte::core::Context ctx;
  ctx.op_flags_ |= clio::cte::core::Context::kGenerational;
  ctx.generation_ = 1;
  const u64 all_bytes = static_cast<u64>(nodes) * slice * sizeof(float);
  for (;;) {
    auto f = cte.AsyncPutBlob(tag, stem + std::to_string(node), 0, all_bytes,
                              reinterpret_cast<char *>(const_cast<float *>(mine)),
                              1.0f, ctx);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (expired()) {
      std::fprintf(stderr, "  slx[%s]: timed out publishing node %u (last put rc=%u)\n",
                   prefix, node, f->GetReturnCode());
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  std::vector<PeerGet> gets;
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
    PeerGet g{stem + std::to_string(peer),
              reinterpret_cast<char *>(out + static_cast<u64>(peer) * slice),
              slice * sizeof(float)};
    g.off = static_cast<u64>(node) * slice * sizeof(float);
    gets.push_back(g);
  }
  return GetPeers(cte, tag, ctx, gets, expired, "slx", prefix);
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
