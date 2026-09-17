/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The cluster-batched GNN burst kernel: one coroutine, one copy.
 *
 * WHAT A BURST IS. Cluster-GCN draws a batch's seeds from ONE METIS partition,
 * so the batch needs that partition's nodes plus their 1-hop halo. With the
 * feature rows renumbered partition-contiguous (see gnn_burst_prep.py), that
 * node set maps to a PAGE SET -- typically a few percent of the matrix, known
 * exactly one batch in advance. That page set arriving at once is the burst.
 *
 * WHY THIS SHAPE AND NOT A FULL FORWARD. What is being measured is the TIER
 * behaviour of a bursty arrival: does the burst land in a fast tier, and what
 * does it cost when it does not. The arithmetic is deliberately a cheap,
 * deterministic, order-fixed reduction over exactly the burst's pages --
 *
 *     pool[f] += sum over rows r of the burst's pages of feat[r][f]
 *
 * -- so that the bytes moved and the pages touched are exactly a real
 * Cluster-GCN batch's, while the result stays bit-reproducible and needs no
 * O(N) resident output. Adding a real SpMM on top would change the FLOPs and
 * not one byte of the I/O this exists to measure. (It would also reintroduce
 * the scatter bookkeeping that every other GNN benchmark here avoids for the
 * same reason.)
 *
 * The page list is device memory the host fills per burst; blocks split it
 * round-robin, so every block streams a disjoint slice of the burst.
 */
#ifndef CLIO_GV_BENCH_GNN_BURST_KERNELS_H_
#define CLIO_GV_BENCH_GNN_BURST_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::gnn_burst {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;

/**
 * Stream this block's slice of one burst's page list.
 *
 * @param pages   device array of page ids for THIS burst
 * @param npages  entries in `pages`
 * @param epp     elements per page
 * @param nblocks blocks in the launch (slice stride)
 * @param blk     this block's LOGICAL index
 * @param out     F accumulators, atomically merged
 * @param dim     F
 */
CTP_GPU_FUN inline gy::YCoroMain BurstCoro(gv::DeviceVector<float> vec,
                                           const u64 *pages, u64 npages,
                                           u64 epp, u32 nblocks, u32 blk,
                                           double *out, u32 dim) {
  for (u64 i = blk; i < npages; i += nblocks) {
    const u64 pg = pages[i];
    const u64 off = pg * epp;
    // PUBLISH THE POSITION as an index INTO THIS BURST'S LIST, not the page
    // id: the host organizer thinks in "how far through the burst is this
    // block", and the page id would be a number it has to invert. +1 keeps 0
    // meaning "nothing published" (see YieldPublishCursor).
    gy::YieldPublishCursor(i + 1, pg);
    co_await vec.Fetch(0, off, epp);
    auto h = co_await vec.HoldPage(off, epp);
    // One thread per feature column, fixed order over the page's rows: the
    // same float ops in the same order on every run, so the reduction is
    // reproducible and any difference between arms is I/O, not arithmetic.
    for (u32 f = threadIdx.x; f < dim; f += blockDim.x) {
      double acc = 0.0;
      for (u64 r = 0; r < epp / dim; ++r) {
        acc += static_cast<double>(h[off + r * dim + f]);
      }
      atomicAdd(&out[f], acc);
    }
    __syncthreads();
    vec.UnpinRange(off, epp);
  }
}

}  // namespace clio::gv_bench::gnn_burst

#endif  // CLIO_GV_BENCH_GNN_BURST_KERNELS_H_
