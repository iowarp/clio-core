/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Shared dataset loading for the GNN workload tests.
 *
 * Both GNN tests want the same node-feature matrix and CSR graph, and both want
 * to run whether or not anyone has downloaded ogbn-arxiv. gnn_prep.py turns the
 * real dataset into flat binaries; when those are absent we synthesise a graph
 * of the same SHAPE instead of failing, because the properties these tests
 * assert -- that a lossless codec returns exactly the bytes it was given, so a
 * deterministic forward pass is bit-identical -- do not depend on the data
 * being real. Only the compression ratios and timings do, which is why every
 * caller must report which mode it ran in.
 *
 * Host-only: guarded so the nvcc device pass does not parse it.
 */

#ifndef CLIO_CTE_GPU_VECTOR_TEST_GNN_DATASET_H_
#define CLIO_CTE_GPU_VECTOR_TEST_GNN_DATASET_H_

#if !CTP_IS_DEVICE_PASS

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace gnn_test {

inline double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline std::int64_t EnvI64(const char *name, std::int64_t dflt) {
  const char *e = std::getenv(name);
  return e != nullptr ? (std::int64_t)std::atoll(e) : dflt;
}

/** Wire ids the compressor registry uses; only the ones these tests name. */
inline const char *CodecName(int lib) {
  switch (lib) {
    case 0: return "none";
    case 3: return "lz4";
    case 9: return "zlib";
    case 10: return "zstd";
    case 11: return "nvcomp_lz4";
    default: return "other";
  }
}

/** Read a whole flat binary. False (quietly) if it is not there. */
inline bool ReadFileQuiet(const std::string &path, std::vector<char> &out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  std::streamoff n = f.tellg();
  f.seekg(0);
  out.resize((size_t)n);
  f.read(out.data(), n);
  return (bool)f;
}

/** Node features + CSR graph, either the prepared dataset or a synthetic one. */
struct Dataset {
  std::int64_t N = 0;      /**< nodes */
  std::int64_t E = 0;      /**< undirected edges (== indptr[N]) */
  int F = 0;               /**< feature dimension */
  int C = 0;               /**< classes */
  bool real = false;       /**< true when loaded from gnn_prep.py output */
  std::vector<char> feat;  /**< N*F float32, row-major */
  std::vector<char> csr;   /**< int64 N, int64 E, indptr[N+1], indices[E] */
  std::vector<char> labels;/**< N int64 */

  const float *Feat() const {
    return reinterpret_cast<const float *>(feat.data());
  }
  std::int64_t FeatElems() const { return N * (std::int64_t)F; }
  std::int64_t FeatBytes() const {
    return FeatElems() * (std::int64_t)sizeof(float);
  }
  const std::int64_t *Csr() const {
    return reinterpret_cast<const std::int64_t *>(csr.data());
  }
  const std::int64_t *Indptr() const { return Csr() + 2; }
  const std::int64_t *Indices() const { return Csr() + 2 + (N + 1); }
  const char *SourceName() const {
    return real ? "ogbn-arxiv" : "SYNTHETIC (not ogbn-arxiv)";
  }
};

/**
 * Load `<dir>` if gnn_prep.py has populated it, else synthesise.
 *
 * The synthetic graph is a ring plus seeded random edges. The ring matters:
 * it guarantees every node has at least one neighbour, so no row of the mean
 * aggregation is an empty (degree-zero) special case.
 *
 * The synthetic features are quantised to 16 levels per column. Uniform random
 * float32 would be incompressible noise and would make a lossless codec look
 * pointless for reasons that say nothing about real feature matrices.
 */
inline Dataset LoadOrSynthDataset(const std::string &dir, const char *tag) {
  Dataset d;
  {
    std::ifstream mf(dir + "/meta.txt");
    if (mf) {
      mf >> d.N >> d.F >> d.E >> d.C;
      d.real = (d.N > 0 && d.F > 0 && d.C > 0);
    }
  }
  if (d.real) {
    d.real = ReadFileQuiet(dir + "/features.f32", d.feat) &&
             ReadFileQuiet(dir + "/graph.csr", d.csr);
    if (d.real) {
      ReadFileQuiet(dir + "/labels.i64", d.labels);
      return d;
    }
    d.feat.clear();
    d.csr.clear();
  }

  d.N = EnvI64("CLIO_GNN_SYNTH_N", 20000);
  d.F = (int)EnvI64("CLIO_GNN_SYNTH_F", 128);
  d.C = (int)EnvI64("CLIO_GNN_SYNTH_C", 40);
  const std::int64_t deg = EnvI64("CLIO_GNN_SYNTH_DEG", 10);
  std::fprintf(stderr,
               "[%s] %s/meta.txt not found -- SYNTHETIC dataset "
               "(run gnn/gnn_prep.py for the real ogbn-arxiv numbers)\n",
               tag, dir.c_str());

  d.feat.resize((size_t)(d.N * (std::int64_t)d.F * (std::int64_t)sizeof(float)));
  float *fw = reinterpret_cast<float *>(d.feat.data());
  std::mt19937 frng(12345);
  std::uniform_int_distribution<int> lvl(0, 15);
  for (std::int64_t i = 0; i < d.N * (std::int64_t)d.F; ++i) {
    fw[i] = (float)lvl(frng) * 0.0625f - 0.5f;
  }

  std::vector<std::vector<std::int64_t>> adj((size_t)d.N);
  std::mt19937_64 grng(6789);
  for (std::int64_t v = 0; v < d.N; ++v) {
    adj[(size_t)v].push_back((v + 1) % d.N);
    adj[(size_t)((v + 1) % d.N)].push_back(v);
    for (std::int64_t k = 0; k + 2 < deg; ++k) {
      std::int64_t u = (std::int64_t)(grng() % (unsigned long long)d.N);
      if (u == v) continue;
      adj[(size_t)v].push_back(u);
      adj[(size_t)u].push_back(v);
    }
  }
  std::int64_t tot = 0;
  for (auto &row : adj) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    tot += (std::int64_t)row.size();
  }
  d.E = tot;
  d.csr.resize(
      (size_t)((2 + (d.N + 1) + tot) * (std::int64_t)sizeof(std::int64_t)));
  std::int64_t *cw = reinterpret_cast<std::int64_t *>(d.csr.data());
  cw[0] = d.N;
  cw[1] = tot;
  std::int64_t *ip = cw + 2;
  std::int64_t *ix = cw + 2 + (d.N + 1);
  std::int64_t at = 0;
  for (std::int64_t v = 0; v < d.N; ++v) {
    ip[v] = at;
    for (std::int64_t u : adj[(size_t)v]) ix[at++] = u;
  }
  ip[d.N] = at;

  d.labels.resize((size_t)(d.N * (std::int64_t)sizeof(std::int64_t)));
  std::int64_t *lw = reinterpret_cast<std::int64_t *>(d.labels.data());
  for (std::int64_t i = 0; i < d.N; ++i) lw[i] = i % d.C;
  return d;
}

}  // namespace gnn_test

#endif  // !CTP_IS_DEVICE_PASS
#endif  // CLIO_CTE_GPU_VECTOR_TEST_GNN_DATASET_H_
