/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * 1-hop mean neighbour aggregation over a feature matrix that lives in the CTE.
 *
 *   A[i] = mean( {X[i]} U {X[j] : j in N(i)} )      (mean, with self-loop)
 *
 * the same quantity gnn_agg.py precomputes -- but X is never a file here. It is
 * read out of the vector's backing store a page at a time and the result is
 * written back as another tag, so a matrix far larger than memory is aggregated
 * without a flat copy at either end.
 *
 * WHY SCATTER, NOT GATHER. The obvious formulation walks each destination i and
 * reads its neighbours' rows. That is what the reference does, and over an
 * mmapped flat file the page cache makes it fine. Over a compressed, tiered
 * store it is a disaster: papers100M has 1.6e9 edges, so a gather is 1.6e9
 * effectively-random page reads, each potentially a decompress. Instead this
 * inverts the loop:
 *
 *   for each destination BLOCK D that fits in memory:
 *     acc[v] = X[v] for v in D                      (the self term)
 *     stream EVERY source page in order:            (sequential, cache-friendly)
 *       for each source u in the page:
 *         for each v in N(u) with v in D: acc[v] += X[u]
 *     acc[v] /= deg(v)+1, write acc as pages of the output tag
 *
 * Every read of X is sequential; the random access moves to acc, which is in
 * memory by construction. The cost is one full sequential pass over X per
 * block, so pick the block as large as RAM allows: passes = ceil(N/|D|).
 *
 * This relies on the graph being UNDIRECTED (gnn_prep symmetrises it), so
 * "v in N(u)" and "u in N(v)" are the same statement and walking u's adjacency
 * to find destinations is correct.
 *
 * The CSR arrives on stdin in graph.csr layout -- int64 N, int64 E,
 * indptr[N+1], indices[E] -- so it can be piped from a producer instead of
 * staged as a 13 GiB file. It is held in RAM (papers100M: ~13 GiB, which fits
 * where the 53 GiB feature matrix does not).
 *
 * Accumulation is float64 and, within a destination, ordered by source id --
 * a different summation order from the reference's, which follows the adjacency
 * list. Expect agreement to float64 rounding, not bit equality, on arbitrary
 * inputs. (A 5000-node check against a direct gather came back bit-identical,
 * but only because those features are quantised values whose float64 sums are
 * exact; do not read that as an order-independence guarantee.)
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/page.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

size_t ReadFull(std::FILE *f, char *buf, size_t n) {
  size_t got = 0;
  while (got < n) {
    size_t r = std::fread(buf + got, 1, n - got, f);
    if (r == 0) break;
    got += r;
  }
  return got;
}

void Usage() {
  std::fprintf(stderr,
      "usage: gnn_aggregate --feat-tag NAME --out-tag NAME --dim F\n"
      "                     [--page-bytes N] [--block-rows R]\n"
      "                     [--codec ID] [--preset P]\n"
      "  CSR (int64 N,E,indptr[N+1],indices[E]) on stdin\n");
}

}  // namespace

int main(int argc, char **argv) {
  std::string feat_tag, out_tag;
  std::uint64_t page_bytes = 1u << 20;
  std::int64_t block_rows = 0;  // 0 = auto
  int dim = 0, codec = 10, preset = 2;
  clio::run::PoolId storage(600, 0);

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) { Usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--feat-tag") feat_tag = next();
    else if (a == "--out-tag") out_tag = next();
    else if (a == "--dim") dim = std::atoi(next());
    else if (a == "--page-bytes") page_bytes = std::strtoull(next(), nullptr, 10);
    else if (a == "--block-rows") block_rows = std::strtoll(next(), nullptr, 10);
    else if (a == "--codec") codec = std::atoi(next());
    else if (a == "--preset") preset = std::atoi(next());
    else { Usage(); return 2; }
  }
  if (feat_tag.empty() || out_tag.empty() || dim <= 0) { Usage(); return 2; }

  // ---- CSR from stdin ----
  std::int64_t hdr[2];
  if (ReadFull(stdin, reinterpret_cast<char *>(hdr), sizeof(hdr)) != sizeof(hdr)) {
    std::fprintf(stderr, "[agg] could not read CSR header\n");
    return 1;
  }
  const std::int64_t N = hdr[0], E = hdr[1];
  if (N <= 0 || E < 0) {
    std::fprintf(stderr, "[agg] bad CSR header N=%lld E=%lld\n",
                 (long long)N, (long long)E);
    return 1;
  }
  std::fprintf(stderr, "[agg] CSR: N=%lld E=%lld (%.2f GiB indices)\n",
               (long long)N, (long long)E, E * 8.0 / 1073741824.0);

  std::vector<std::int64_t> indptr((size_t)N + 1);
  if (ReadFull(stdin, reinterpret_cast<char *>(indptr.data()),
               indptr.size() * 8) != indptr.size() * 8) {
    std::fprintf(stderr, "[agg] short read on indptr\n");
    return 1;
  }
  std::vector<std::int64_t> indices((size_t)E);
  if (E > 0 && ReadFull(stdin, reinterpret_cast<char *>(indices.data()),
                        indices.size() * 8) != indices.size() * 8) {
    std::fprintf(stderr, "[agg] short read on indices\n");
    return 1;
  }
  if (indptr[N] != E) {
    std::fprintf(stderr, "[agg] indptr[N]=%lld != E=%lld\n",
                 (long long)indptr[N], (long long)E);
    return 1;
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient)) {
    std::fprintf(stderr, "[agg] runtime init failed\n");
    return 1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[agg] cte client init failed\n");
    return 1;
  }

  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  auto ftf = core.AsyncGetOrCreateTag(feat_tag); ftf.Wait();
  auto otf = core.AsyncGetOrCreateTag(out_tag);  otf.Wait();
  if (ftf->GetReturnCode() != 0 || otf->GetReturnCode() != 0) {
    std::fprintf(stderr, "[agg] tag resolution failed\n");
    return 1;
  }
  const auto feat_id = ftf->tag_id_, out_id = otf->tag_id_;

  clio::cte::core::Client comp(storage);
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_lib_ = codec;
  ctx.compress_preset_ = preset;

  const std::int64_t rows_per_page = (std::int64_t)(page_bytes / (dim * sizeof(float)));
  if (rows_per_page <= 0) {
    std::fprintf(stderr, "[agg] page of %llu B cannot hold one %d-d row\n",
                 (unsigned long long)page_bytes, dim);
    return 1;
  }
  const std::int64_t npages = (N + rows_per_page - 1) / rows_per_page;
  if (block_rows <= 0) block_rows = std::min<std::int64_t>(N, 1 << 20);
  const std::int64_t nblocks = (N + block_rows - 1) / block_rows;

  std::fprintf(stderr,
               "[agg] N=%lld dim=%d page=%lluB rows/page=%lld pages=%lld\n"
               "[agg] block=%lld rows -> %lld pass(es) over the features\n",
               (long long)N, dim, (unsigned long long)page_bytes,
               (long long)rows_per_page, (long long)npages,
               (long long)block_rows, (long long)nblocks);

  std::vector<float> pagebuf((size_t)(rows_per_page * dim));
  std::vector<double> acc((size_t)(block_rows * dim));
  std::vector<float> outbuf((size_t)(rows_per_page * dim));
  const double t0 = NowSec();

  auto read_page = [&](std::int64_t pg) -> bool {
    char name[32];
    clio::cte::gpu_vector::PageBlobName((clio::run::u64)pg, name);
    ctp::ipc::ShmPtr<> dp;
    dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(pagebuf.data());
    auto gf = comp.AsyncGetBlob(feat_id, name, (clio::run::u64)0,
                                (clio::run::u64)page_bytes, 0, dp,
                                clio::run::PoolQuery::Local());
    gf.Wait();
    if (gf->GetReturnCode() != 0) {
      std::fprintf(stderr, "[agg] read page %lld rc=%d\n", (long long)pg,
                   gf->GetReturnCode());
      return false;
    }
    return true;
  };

  for (std::int64_t b = 0; b < nblocks; ++b) {
    const std::int64_t lo = b * block_rows;
    const std::int64_t hi = std::min(N, lo + block_rows);
    const std::int64_t nrows = hi - lo;
    std::fill(acc.begin(), acc.begin() + (size_t)(nrows * dim), 0.0);

    // One sequential pass over every feature page. Each source row contributes
    // to whichever destinations in [lo,hi) list it as a neighbour, plus to
    // itself when it falls in this block.
    for (std::int64_t pg = 0; pg < npages; ++pg) {
      if (!read_page(pg)) return 1;
      const std::int64_t u0 = pg * rows_per_page;
      const std::int64_t u1 = std::min(N, u0 + rows_per_page);
      for (std::int64_t u = u0; u < u1; ++u) {
        const float *xu = &pagebuf[(size_t)((u - u0) * dim)];
        if (u >= lo && u < hi) {  // self term
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

    // Normalise by deg+1 and write this block's rows back out as pages. A block
    // is a whole number of pages only if block_rows is; otherwise a page may
    // straddle two blocks, so pages are written when they are complete.
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
        ctp::ipc::ShmPtr<> dp;
        dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
        dp.off_ = reinterpret_cast<clio::run::u64>(outbuf.data());
        auto pf = comp.AsyncPutBlob(out_id, name, (clio::run::u64)0,
                                    (clio::run::u64)page_bytes, dp, 0.5f, ctx, 0,
                                    clio::run::PoolQuery::Local());
        pf.Wait();
        if (pf->GetReturnCode() != 0) {
          std::fprintf(stderr, "[agg] write page %lld rc=%d\n", (long long)pg,
                       pf->GetReturnCode());
          return 1;
        }
      }
    }
    std::fprintf(stderr, "[agg] block %lld/%lld done (%.1fs)\n",
                 (long long)(b + 1), (long long)nblocks, NowSec() - t0);
  }

  std::fprintf(stderr, "[agg] DONE: %lld pages written to '%s' in %.1fs\n",
               (long long)npages, out_tag.c_str(), NowSec() - t0);
  std::fprintf(stdout, "%lld\n", (long long)npages);
  return 0;
}
