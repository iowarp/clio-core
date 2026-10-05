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
#include <cmath>
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
 * `out_tag`, both as vector pages (named by page number in decimal).
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
      const std::string name = std::to_string(pg);
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
        const std::string name = std::to_string(pg);
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

/**
 * Recompute `nsample` randomly chosen rows the slow way and compare.
 *
 * At papers100M there is no external reference: the matrix cannot be held in
 * host memory, so nothing can recompute A and diff it. A wrong aggregate --
 * an off-by-one in the page arithmetic, a block boundary that drops
 * contributions, the vector opened on the wrong tag -- would produce numbers
 * that look entirely plausible and are simply wrong. This is the only
 * correctness signal available at that scale, so it is worth the page reads.
 *
 * Cost is about (average degree) page fetches per sampled row, with a one-page
 * cache that the sorted adjacency makes reasonably effective.
 *
 * @return true when every sampled row agrees within `tol`.
 */
inline bool VerifyRows(clio::cte::core::Client &comp,
                       const clio::cte::core::TagId &feat_tag,
                       const clio::cte::core::TagId &out_tag,
                       const std::int64_t *indptr, const std::int64_t *indices,
                       const Params &p, int nsample, double tol,
                       std::uint64_t seed) {
  const std::int64_t N = p.nodes;
  const int dim = p.dim;
  const std::int64_t rows_per_page =
      (std::int64_t)(p.page_bytes / (dim * sizeof(float)));
  if (rows_per_page <= 0 || N <= 0) return false;

  ctp::ipc::FullPtr<char> fb = CLIO_IPC->AllocateBuffer((size_t)p.page_bytes);
  ctp::ipc::FullPtr<char> ab = CLIO_IPC->AllocateBuffer((size_t)p.page_bytes);
  if (fb.IsNull() || ab.IsNull()) return false;
  float *fpage = reinterpret_cast<float *>(fb.ptr_);
  float *apage = reinterpret_cast<float *>(ab.ptr_);
  std::int64_t cached = -1;

  auto fetch = [&](const clio::cte::core::TagId &tag, std::int64_t pg,
                   ctp::ipc::FullPtr<char> &buf) -> bool {
    const std::string name = std::to_string(pg);
    auto gf = comp.AsyncGetBlob(tag, name, (clio::run::u64)0,
                               (clio::run::u64)p.page_bytes, 0,
                               buf.shm_.template Cast<void>(),
                               clio::run::PoolQuery::Local());
    gf.Wait();
    return gf->GetReturnCode() == 0;
  };
  auto feat_row = [&](std::int64_t r, std::vector<double> &out) -> bool {
    const std::int64_t pg = r / rows_per_page;
    if (pg != cached) {
      if (!fetch(feat_tag, pg, fb)) return false;
      cached = pg;
    }
    const float *src = &fpage[(size_t)((r % rows_per_page) * dim)];
    for (int d = 0; d < dim; ++d) out[(size_t)d] = (double)src[d];
    return true;
  };

  std::uint64_t st = seed | 1ull;
  auto next = [&]() {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return st;
  };

  std::vector<double> ref((size_t)dim), tmp((size_t)dim);
  double worst = 0.0;
  int checked = 0;
  for (int i = 0; i < nsample; ++i) {
    const std::int64_t v = (std::int64_t)(next() % (std::uint64_t)N);
    std::fill(ref.begin(), ref.end(), 0.0);
    if (!feat_row(v, tmp)) return false;
    for (int d = 0; d < dim; ++d) ref[(size_t)d] += tmp[(size_t)d];
    for (std::int64_t k = indptr[(size_t)v]; k < indptr[(size_t)v + 1]; ++k) {
      if (!feat_row(indices[(size_t)k], tmp)) return false;
      for (int d = 0; d < dim; ++d) ref[(size_t)d] += tmp[(size_t)d];
    }
    const double inv =
        1.0 / (double)(indptr[(size_t)v + 1] - indptr[(size_t)v] + 1);

    if (!fetch(out_tag, v / rows_per_page, ab)) return false;
    const float *got = &apage[(size_t)((v % rows_per_page) * dim)];
    for (int d = 0; d < dim; ++d) {
      worst = std::max(worst, std::fabs(ref[(size_t)d] * inv - (double)got[d]));
    }
    ++checked;
  }

  CLIO_IPC->FreeBuffer(fb);
  CLIO_IPC->FreeBuffer(ab);
  std::fprintf(stderr, "[agg] verified %d sampled rows: worst_abs=%.3e (tol %.1e) -> %s\n",
               checked, worst, tol, worst <= tol ? "OK" : "MISMATCH");
  return worst <= tol;
}

}  // namespace gnn_agg

#endif  // !CTP_IS_DEVICE_PASS
#endif  // CLIO_CTE_GPU_VECTOR_GNN_AGGREGATE_IMPL_H_
