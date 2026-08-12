/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * The 1-hop mean aggregation kernel, as a callable rather than a program.
 *
 * gnn_aggregate (the tool) is a client of a daemon. The GPU training test
 * cannot be: GPU queues are only ever created by ServerInitGpuQueues and there
 * is no client-side attach, so a process that faults on the GPU must host the
 * runtime itself and therefore has its own, separate CTE. Aggregating with the
 * tool and then training on the GPU would mean exporting the aggregated matrix
 * -- 53 GiB at papers100M -- and re-ingesting it, which is precisely the flat
 * copy this pipeline exists to avoid.
 *
 * So the algorithm lives here and both callers use it: the tool against a
 * daemon, the training test against its own in-process runtime.
 *
 * See gnn_aggregate.cc for why this scatters rather than gathers, and for the
 * measured cost model (one sequential pass over the features per destination
 * block).
 */

#ifndef CLIO_CTE_GPU_VECTOR_GNN_AGGREGATE_IMPL_H_
#define CLIO_CTE_GPU_VECTOR_GNN_AGGREGATE_IMPL_H_

#if !CTP_IS_DEVICE_PASS

#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/page.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gnn_agg {

struct Params {
  clio::run::u64 page_bytes = 1u << 20;
  int dim = 0;
  std::int64_t nodes = 0;
  std::int64_t block_rows = 0;  /**< 0 = pick 1M */
  int codec = 10;               /**< ZSTD */
  int preset = 2;               /**< BALANCED */
  bool verbose = true;
};

/**
 * A[i] = mean({X[i]} U {X[j] : j in N(i)}), reading `feat_tag` and writing
 * `out_tag`, both as vector pages ("p<N>", PageBlobName).
 *
 * `indptr` has nodes+1 entries and `indices` indptr[nodes]; the graph must be
 * UNDIRECTED (see gnn_aggregate.cc). Returns false on any CTE error.
 */
inline bool Aggregate(clio::cte::core::Client &comp,
                      const clio::cte::core::TagId &feat_tag,
                      const clio::cte::core::TagId &out_tag,
                      const std::int64_t *indptr, const std::int64_t *indices,
                      const Params &p) {
  const std::int64_t N = p.nodes;
  const int dim = p.dim;
  if (N <= 0 || dim <= 0) return false;

  const std::int64_t rows_per_page =
      (std::int64_t)(p.page_bytes / (dim * sizeof(float)));
  if (rows_per_page <= 0) {
    std::fprintf(stderr, "[agg] page of %llu B cannot hold one %d-d row\n",
                 (unsigned long long)p.page_bytes, dim);
    return false;
  }
  const std::int64_t npages = (N + rows_per_page - 1) / rows_per_page;
  std::int64_t block_rows = p.block_rows > 0 ? p.block_rows
                                             : std::min<std::int64_t>(N, 1 << 20);
  const std::int64_t nblocks = (N + block_rows - 1) / block_rows;

  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_lib_ = p.codec;
  ctx.compress_preset_ = p.preset;

  // SHM, not the heap: a raw host pointer is only resolvable by the runtime
  // when the runtime is in this process, and this code runs both ways.
  ctp::ipc::FullPtr<char> inb = CLIO_IPC->AllocateBuffer((size_t)p.page_bytes);
  ctp::ipc::FullPtr<char> outb = CLIO_IPC->AllocateBuffer((size_t)p.page_bytes);
  if (inb.IsNull() || outb.IsNull()) {
    std::fprintf(stderr, "[agg] could not allocate 2 x %llu B of SHM\n",
                 (unsigned long long)p.page_bytes);
    return false;
  }
  float *pagebuf = reinterpret_cast<float *>(inb.ptr_);
  float *outbuf = reinterpret_cast<float *>(outb.ptr_);
  std::vector<double> acc((size_t)(block_rows * dim));

  if (p.verbose) {
    std::fprintf(stderr,
                 "[agg] N=%lld dim=%d rows/page=%lld pages=%lld block=%lld "
                 "-> %lld pass(es)\n",
                 (long long)N, dim, (long long)rows_per_page, (long long)npages,
                 (long long)block_rows, (long long)nblocks);
  }

  for (std::int64_t b = 0; b < nblocks; ++b) {
    const std::int64_t lo = b * block_rows;
    const std::int64_t hi = std::min(N, lo + block_rows);
    const std::int64_t nrows = hi - lo;
    std::fill(acc.begin(), acc.begin() + (size_t)(nrows * dim), 0.0);

    for (std::int64_t pg = 0; pg < npages; ++pg) {
      char name[32];
      clio::cte::gpu_vector::PageBlobName((clio::run::u64)pg, name);
      auto gf = comp.AsyncGetBlob(feat_tag, name, (clio::run::u64)0,
                                  (clio::run::u64)p.page_bytes, 0,
                                  inb.shm_.template Cast<void>(),
                                  clio::run::PoolQuery::Local());
      gf.Wait();
      if (gf->GetReturnCode() != 0) {
        std::fprintf(stderr, "[agg] read page %lld rc=%d\n", (long long)pg,
                     gf->GetReturnCode());
        return false;
      }
      const std::int64_t u0 = pg * rows_per_page;
      const std::int64_t u1 = std::min(N, u0 + rows_per_page);
      for (std::int64_t u = u0; u < u1; ++u) {
        const float *xu = &pagebuf[(size_t)((u - u0) * dim)];
        if (u >= lo && u < hi) {
          double *a = &acc[(size_t)((u - lo) * dim)];
          for (int d = 0; d < dim; ++d) a[d] += (double)xu[d];
        }
        for (std::int64_t k = indptr[(size_t)u]; k < indptr[(size_t)u + 1]; ++k) {
          const std::int64_t v = indices[(size_t)k];
          if (v < lo || v >= hi) continue;
          double *a = &acc[(size_t)((v - lo) * dim)];
          for (int d = 0; d < dim; ++d) a[d] += (double)xu[d];
        }
      }
    }

    for (std::int64_t v = lo; v < hi; ++v) {
      const double inv =
          1.0 / (double)(indptr[(size_t)v + 1] - indptr[(size_t)v] + 1);
      double *a = &acc[(size_t)((v - lo) * dim)];
      float *o = &outbuf[(size_t)((v % rows_per_page) * dim)];
      for (int d = 0; d < dim; ++d) o[d] = (float)(a[d] * inv);
      const bool page_end = ((v + 1) % rows_per_page == 0) || (v + 1 == N);
      if (page_end) {
        const std::int64_t pg = v / rows_per_page;
        if ((v + 1) == N) {
          const std::int64_t used = (v % rows_per_page) + 1;
          std::memset(&outbuf[(size_t)(used * dim)], 0,
                      (size_t)((rows_per_page - used) * dim) * sizeof(float));
        }
        char name[32];
        clio::cte::gpu_vector::PageBlobName((clio::run::u64)pg, name);
        auto pf = comp.AsyncPutBlob(out_tag, name, (clio::run::u64)0,
                                    (clio::run::u64)p.page_bytes,
                                    outb.shm_.template Cast<void>(), 0.5f, ctx,
                                    0, clio::run::PoolQuery::Local());
        pf.Wait();
        if (pf->GetReturnCode() != 0) {
          std::fprintf(stderr, "[agg] write page %lld rc=%d\n", (long long)pg,
                       pf->GetReturnCode());
          return false;
        }
      }
    }
    if (p.verbose) {
      std::fprintf(stderr, "[agg] block %lld/%lld done\n", (long long)(b + 1),
                   (long long)nblocks);
    }
  }

  CLIO_IPC->FreeBuffer(inb);
  CLIO_IPC->FreeBuffer(outb);
  return true;
}

}  // namespace gnn_agg

#endif  // !CTP_IS_DEVICE_PASS
#endif  // CLIO_CTE_GPU_VECTOR_GNN_AGGREGATE_IMPL_H_
