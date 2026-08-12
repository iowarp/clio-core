/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_dataset_parity.cu
 * @brief Full-dataset differential test: 1 GiB / 4 MiB chunks, lossless.
 *
 * The unit-level harness (neuropress_parity.cu) feeds both sides
 * hand-written feature values. This one starts from BYTES: a deterministic
 * 1 GiB dataset is chunked at 4 MiB and each chunk is pushed through both
 * pipelines end to end, so the comparison covers the parts a
 * feature-level test cannot --
 *
 *   STATISTICS. NeuroPress computes entropy/MAD/second-derivative with its
 *   own CUDA kernels (stats_kernel.cu, entropy_kernel.cu); Clio computes
 *   them with ComputeCompressionFeatures. Both read the SAME device bytes
 *   and the three numbers are diffed per chunk. A divergence here would
 *   feed every downstream prediction a different input, so it has to be
 *   checked before anything else is meaningful.
 *
 *   PREDICTION + SELECTION. Each side then ranks and picks. Predictions are
 *   compared for the action native chose, and the chosen algorithm is
 *   compared directly.
 *
 * Lossless throughout: error_bound = 0, which is also what makes upstream
 * mask every quantize action to -INFINITY, so both sides range over the
 * same algorithm x byte-shuffle space.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/model/predictor.h"
#include "clio_ctp/compress/model/ranking.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

#include "nn_weights.h"
#include "auto_stats_gpu.h"

namespace gpucompress {
bool loadNNFromBinary(const char *filepath);
void cleanupNN();
int runNNInference(double entropy, double mad_norm, double deriv_norm,
                   size_t data_size, double error_bound, cudaStream_t stream,
                   float *out_predicted_ratio, float *out_predicted_comp_time,
                   float *out_predicted_decomp_time, float *out_predicted_psnr,
                   int *out_top_actions);
int runStatsOnlyPipeline(const void *d_input, size_t input_size,
                         cudaStream_t stream, double *out_entropy,
                         double *out_mad, double *out_deriv);
void freeStatsWorkspace();
}  // namespace gpucompress

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

using ctp::compress::model::CandidateConfig;
using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::DataFeatures;
using ctp::compress::model::MakeCompressionFeatures;
using ctp::compress::model::NeuroPressNNPredictor;

namespace {

constexpr size_t kTotalBytes = 1ull << 30;   // 1 GiB
constexpr size_t kChunkBytes = 4ull << 20;   // 4 MiB
constexpr size_t kNumChunks = kTotalBytes / kChunkBytes;  // 256

int g_failures = 0;
long g_checks = 0;
double g_max_rel_entropy = 0, g_max_rel_mad = 0, g_max_rel_deriv = 0;
double g_max_rel_ratio = 0, g_max_rel_ct = 0, g_max_rel_dt = 0;
int g_algo_mismatches = 0;

void Track(double a, double b, double *worst) {
  double denom = std::max(1e-12, std::max(std::fabs(a), std::fabs(b)));
  double rel = std::fabs(a - b) / denom;
  if (rel > *worst) *worst = rel;
}

void Expect(double a, double b, double tol, double *worst,
            const char *what, size_t chunk) {
  ++g_checks;
  Track(a, b, worst);
  double denom = std::max(1e-12, std::max(std::fabs(a), std::fabs(b)));
  if (std::fabs(a - b) / denom > tol) {
    if (g_failures < 20) {  // don't drown the log
      std::printf("  [FAIL] chunk %zu %s: native=%.9g clio=%.9g\n", chunk,
                  what, a, b);
    }
    ++g_failures;
  }
}

/** NeuroPress action index (0-7) -> canonical algorithm name. */
const char *AlgoName(int a) {
  switch (a) {
    case 0: return "nvcomp-lz4";      case 1: return "nvcomp-snappy";
    case 2: return "nvcomp-deflate";  case 3: return "nvcomp-gdeflate";
    case 4: return "nvcomp-zstd";     case 5: return "nvcomp-ans";
    case 6: return "nvcomp-cascaded"; default: return "nvcomp-bitcomp";
  }
}

/** NeuroPress action index (0-7) -> Clio ML base_id. */
int BaseIdForAlgoIdx(int a) {
  switch (a) {
    case 0: return 13; case 1: return 14; case 2: return 17; case 3: return 16;
    case 4: return 15; case 5: return 18; case 6: return 23; default: return 24;
  }
}

/**
 * Deterministic 1 GiB dataset with several statistical regimes, so the run
 * exercises a real spread of entropy/MAD rather than one operating point.
 * Generated identically for both sides (it is the same buffer).
 */
__global__ void FillDataset(float *buf, size_t num_elems,
                            size_t elems_per_chunk) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_elems) return;
  size_t chunk = i / elems_per_chunk;
  size_t local = i % elems_per_chunk;
  switch (chunk % 5) {
    case 0:  // constant -- near-zero entropy
      buf[i] = 42.0f;
      break;
    case 1: {  // smooth sine -- low entropy, low curvature
      float p = 6.28318f * static_cast<float>(local) /
                static_cast<float>(elems_per_chunk);
      buf[i] = sinf(p * 4.0f);
      break;
    }
    case 2:  // stepped ramp -- moderate
      buf[i] = static_cast<float>((local / 512) % 1024) * 0.25f;
      break;
    case 3: {  // smooth + small noise -- mid entropy
      uint32_t x = static_cast<uint32_t>(i) * 747796405u + 2891336453u;
      x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
      float n = static_cast<float>(x & 0xFFFF) / 65535.0f - 0.5f;
      float p = 6.28318f * static_cast<float>(local) /
                static_cast<float>(elems_per_chunk);
      buf[i] = sinf(p * 2.0f) + 0.05f * n;
      break;
    }
    default: {  // hash noise -- max entropy
      uint32_t x = static_cast<uint32_t>(i) * 2654435761u + 0x9E3779B9u;
      x ^= x >> 15; x *= 0x85EBCA6Bu; x ^= x >> 13;
      buf[i] = static_cast<float>(x) / 4294967295.0f;
      break;
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  const std::string weights_dir =
      (argc > 1) ? argv[1] : std::string(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR);
  const std::string nnwt = weights_dir + "/model.nnwt";

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    std::printf("No CUDA device -- skipping dataset parity.\n");
    return 77;
  }

  if (!gpucompress::loadNNFromBinary(nnwt.c_str())) {
    std::printf("FATAL: native failed to load %s\n", nnwt.c_str());
    return 1;
  }
  NeuroPressNNPredictor clio;
  if (!clio.Load(weights_dir) || !clio.IsReady()) {
    std::printf("FATAL: Clio failed to load %s\n", weights_dir.c_str());
    return 1;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStreamCreate(&g_sgd_stream);
  cudaEventCreate(&g_sgd_done);

  // ---- Generate the dataset on-device, once, shared by both sides. ----
  float *d_data = nullptr;
  if (cudaMalloc(&d_data, kTotalBytes) != cudaSuccess) {
    std::printf("FATAL: cudaMalloc %zu bytes failed\n", kTotalBytes);
    return 1;
  }
  const size_t num_elems = kTotalBytes / sizeof(float);
  const size_t elems_per_chunk = kChunkBytes / sizeof(float);
  {
    int threads = 256;
    int blocks = static_cast<int>((num_elems + threads - 1) / threads);
    FillDataset<<<blocks, threads>>>(d_data, num_elems, elems_per_chunk);
    cudaDeviceSynchronize();
  }
  std::printf("Dataset: %zu MiB, %zu chunks of %zu MiB, lossless "
              "(error_bound=0)\n\n",
              kTotalBytes >> 20, kNumChunks, kChunkBytes >> 20);

  // Clio's candidate set: the 8 trained nvcomp algorithms x byte-shuffle,
  // which is what upstream can reach once quantize is masked at eb=0.
  static const int kTrainedBaseIds[] = {13, 14, 15, 16, 17, 18, 23, 24};

  // Per-chunk selection table. Every row is written to CSV so all 256 are
  // inspectable; the console shows a readable subset.
  std::FILE *csv = std::fopen("neuropress_selection_parity.csv", "w");
  if (csv) {
    std::fprintf(csv,
                 "chunk,entropy_native,entropy_clio,mad_native,mad_clio,"
                 "deriv_native,deriv_clio,algo_native,algo_clio,"
                 "shuffle_native,shuffle_clio,ratio_native,ratio_clio,match\n");
  }
  std::printf("%-6s %-9s %-17s %-17s %-7s %s\n", "chunk", "entropy",
              "NATIVE selection", "CLIO selection", "shuf", "match");
  std::printf("%s\n", std::string(78, '-').c_str());

  for (size_t ci = 0; ci < kNumChunks; ++ci) {
    const char *d_chunk =
        reinterpret_cast<const char *>(d_data) + ci * kChunkBytes;

    // ---- (1) statistics, computed independently by each side ----
    double n_ent = 0, n_mad = 0, n_der = 0;
    int rc = gpucompress::runStatsOnlyPipeline(d_chunk, kChunkBytes, stream,
                                               &n_ent, &n_mad, &n_der);
    cudaStreamSynchronize(stream);
    if (rc != 0) {
      std::printf("  [FAIL] chunk %zu: native stats rc=%d\n", ci, rc);
      ++g_failures; ++g_checks;
      continue;
    }

    double c_ent = 0, c_mad = 0, c_der = 0;
    ctp::ComputeCompressionFeatures(d_chunk, kChunkBytes / sizeof(float),
                                    ctp::DataType::FLOAT32, &c_ent, &c_mad,
                                    &c_der);

    // Stats are double-precision reductions on both sides; they should
    // agree far more tightly than the float32 network does.
    Expect(n_ent, c_ent, 1e-9, &g_max_rel_entropy, "entropy", ci);
    Expect(n_mad, c_mad, 1e-9, &g_max_rel_mad, "mad", ci);
    Expect(n_der, c_der, 1e-9, &g_max_rel_deriv, "second_derivative", ci);

    // ---- (2) native inference: predicts and selects ----
    float n_ratio = 0, n_ct = 0, n_dt = 0, n_psnr = 0;
    int top[NN_NUM_CONFIGS] = {0};
    int action = gpucompress::runNNInference(n_ent, n_mad, n_der, kChunkBytes,
                                             /*error_bound=*/0.0, stream,
                                             &n_ratio, &n_ct, &n_dt, &n_psnr,
                                             top);
    cudaStreamSynchronize(stream);
    ++g_checks;
    if (action < 0) {
      std::printf("  [FAIL] chunk %zu: native inference failed\n", ci);
      ++g_failures;
      continue;
    }
    const int n_algo = action % 8;
    const int n_quant = (action / 8) % 2;
    const int n_shuf = (action / 16) % 2;
    ++g_checks;
    if (n_quant != 0) {
      std::printf("  [FAIL] chunk %zu: native chose a QUANTIZE action (%d) "
                  "at error_bound=0\n", ci, action);
      ++g_failures;
    }

    // ---- (3) Clio: rank its own candidates from its OWN stats ----
    DataFeatures data;
    data.chunk_size_bytes = static_cast<double>(kChunkBytes);
    data.shannon_entropy = c_ent;
    data.mad = c_mad;
    data.second_derivative_mean = c_der;
    data.data_type_float = 1.0;

    std::vector<CandidateConfig> cands;
    for (int base : kTrainedBaseIds) {
      for (int shuf = 0; shuf < 2; ++shuf) {
        CandidateConfig c;
        c.base_id = base;
        c.preset_id = 2;
        c.byte_shuffle = (shuf != 0);
        cands.push_back(c);
      }
    }
    ctp::compress::model::RankingWeights w;
    w.use_cost_model = true;
    auto ranked = clio.Rank(data, cands, w);
    ++g_checks;
    if (ranked.empty()) {
      std::printf("  [FAIL] chunk %zu: Clio produced no ranking\n", ci);
      ++g_failures;
      continue;
    }

    // Clio's own winner -> algo index, for the selection comparison.
    int c_algo = -1;
    for (int a = 0; a < 8; ++a) {
      if (BaseIdForAlgoIdx(a) == ranked.front().candidate.base_id) {
        c_algo = a;
        break;
      }
    }
    const int c_shuf = ranked.front().candidate.byte_shuffle ? 1 : 0;
    ++g_checks;
    if (c_algo != n_algo || c_shuf != n_shuf) {
      ++g_algo_mismatches;
      if (g_algo_mismatches <= 10) {
        std::printf("  [DIFF] chunk %zu selection: native algo=%d shuf=%d, "
                    "clio algo=%d shuf=%d\n", ci, n_algo, n_shuf, c_algo,
                    c_shuf);
      }
      ++g_failures;
    }

    // ---- (4) prediction parity for the action NATIVE chose ----
    CandidateConfig chosen;
    chosen.base_id = BaseIdForAlgoIdx(n_algo);
    chosen.preset_id = 2;
    chosen.byte_shuffle = (n_shuf != 0);
    CompressionFeatures f = MakeCompressionFeatures(data, chosen);
    auto p = clio.Predict(f);

    Expect(n_ratio, p.compression_ratio, 1e-3, &g_max_rel_ratio, "ratio", ci);
    Expect(n_ct, p.compression_time_ms, 1e-3, &g_max_rel_ct, "comp_time", ci);
    Expect(n_dt, p.decompression_time_ms, 1e-3, &g_max_rel_dt, "decomp_time",
           ci);

    const bool sel_match = (c_algo == n_algo && c_shuf == n_shuf);
    if (csv) {
      std::fprintf(csv,
                   "%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%s,%s,%d,%d,"
                   "%.9g,%.9g,%s\n",
                   ci, n_ent, c_ent, n_mad, c_mad, n_der, c_der,
                   AlgoName(n_algo),
                   (c_algo >= 0 ? AlgoName(c_algo) : "none"), n_shuf, c_shuf,
                   n_ratio, p.compression_ratio, sel_match ? "YES" : "NO");
    }
    if (ci < 12 || (ci % 32) == 0 || !sel_match) {
      std::printf("%-6zu %9.5f %-17s %-17s %d/%-5d %s\n", ci, n_ent,
                  AlgoName(n_algo), (c_algo >= 0 ? AlgoName(c_algo) : "none"),
                  n_shuf, c_shuf, sel_match ? "yes" : "*** NO ***");
    }
  }

  std::printf("\n----- worst-case relative deviation over %zu chunks -----\n",
              kNumChunks);
  std::printf("  statistics : entropy %.3g   mad %.3g   second_deriv %.3g\n",
              g_max_rel_entropy, g_max_rel_mad, g_max_rel_deriv);
  std::printf("  prediction : ratio   %.3g   comp_time %.3g  decomp_time %.3g\n",
              g_max_rel_ratio, g_max_rel_ct, g_max_rel_dt);
  std::printf("  selection  : %d chunk(s) chose a different algorithm\n",
              g_algo_mismatches);

  if (csv) {
    std::fclose(csv);
    std::printf("\n  full per-chunk table written to "
                "neuropress_selection_parity.csv (%zu rows)\n", kNumChunks);
  }

  cudaFree(d_data);
  gpucompress::freeStatsWorkspace();
  gpucompress::cleanupNN();
  cudaStreamDestroy(stream);

  std::printf("\n===== %ld checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
