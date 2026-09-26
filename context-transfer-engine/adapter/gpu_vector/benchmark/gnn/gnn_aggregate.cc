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
 * MEASURED (4M nodes x 128-d = 1.91 GiB, 50M undirected edges, zstd, one NVMe
 * tier): reads run ~217 MiB/s per pass, so a pass over the features costs
 * roughly (bytes / 217 MiB/s) and the total is that times ceil(N/block_rows).
 * Extrapolating to papers100M (53 GiB, 111M nodes): ~4.2 min per pass, so
 * block_rows=24e6 (accumulator 24.6 GB) gives 5 passes, ~21 min. Sizing the
 * block down to save memory costs a whole pass each time, so prefer the
 * largest accumulator that fits alongside the CSR.
 *
 * Note the memory interaction when piped from gnn_build_csr: the builder peaks
 * around 26 GiB while sorting, and this process holds the CSR (13 GiB at
 * papers100M) plus the accumulator. The handoff overlaps, so budget for both.
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

#include "gnn_aggregate_impl.h"

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

  gnn_agg::Params ap;
  ap.page_bytes = page_bytes;
  ap.dim = dim;
  ap.nodes = N;
  ap.block_rows = block_rows;
  ap.codec = codec;
  ap.preset = preset;
  const double t0 = NowSec();
  if (!gnn_agg::Aggregate(comp, feat_id, out_id, indptr.data(), indices.data(),
                          ap)) {
    return 1;
  }
  const std::int64_t rows_per_page =
      (std::int64_t)(page_bytes / (dim * sizeof(float)));
  const std::int64_t npages = (N + rows_per_page - 1) / rows_per_page;

  std::fprintf(stderr, "[agg] DONE: %lld pages written to '%s' in %.1fs\n",
               (long long)npages, out_tag.c_str(), NowSec() - t0);
  std::fprintf(stdout, "%lld\n", (long long)npages);
  return 0;
}
