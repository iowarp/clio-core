/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */
/**
@file neuropress_selection_parity.cu
@brief Per chunk: what each side PREDICTS, and what it CHOOSES.

The other harnesses answer "do the two compute the same numbers". This one
answers the question a user of the system actually has: given this chunk, does
Clio pick the same codec upstream would, and does it expect the same thing of
it?

That is not implied by the numeric agreement. Selection is an argmin over 32
costs, so two implementations can agree on every prediction to five decimal
places and still diverge on the winner whenever the top two are close -- and
they are close often, because the ratio saturates at the cap of 100 on
compressible data and the cost model then separates candidates only by
predicted time.

Both sides start from the SAME BYTES and each computes its own statistics with
its own kernels, so a divergence in the stats would show here as a divergence
in the choice rather than being papered over by shared inputs.

Prints a table, then asserts agreement on the chosen action.
*/

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/model/predictor.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

#include "api/internal.hpp"
#include "nn/nn_weights.h"
#include "stats/auto_stats_gpu.h"

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

namespace cm = ctp::compress::model;

namespace {

long g_checks = 0;
int g_failures = 0;

const int kBaseIdForAlgo[8] = {13, 14, 17, 16, 15, 18, 23, 24};
const char *kAlgoName[8] = {"lz4", "snappy", "deflate", "gdeflate",
                            "zstd", "ans",   "cascaded", "bitcomp"};

/* Chunk shapes with genuinely different statistics, so the two sides are asked
   a different question each time rather than the same one eight times. */
std::vector<float> MakeChunk(size_t n, int kind) {
  std::vector<float> v(n);
  unsigned seed = 22222u + static_cast<unsigned>(kind) * 7919u;
  for (size_t i = 0; i < n; ++i) {
    seed = seed * 1103515245u + 12345u;
    switch (kind % 8) {
      case 0: v[i] = static_cast<float>(i % 4096) * 0.001f; break;
      case 1: v[i] = static_cast<float>(seed >> 8) * 1e-7f; break;
      case 2: v[i] = ((seed >> 16) % 32 == 0) ? 1.0f : 0.0f; break;
      case 3: v[i] = 0.5f; break;
      case 4: v[i] = std::sin(static_cast<float>(i) * 0.001f); break;
      case 5: v[i] = static_cast<float>((seed >> 20) % 16); break;
      case 6: v[i] = static_cast<float>(i % 64) * 1e-4f; break;
      default: v[i] = ((seed >> 12) % 256) * 0.004f; break;
    }
  }
  return v;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("No CUDA device -- skipping.\n");
    return 77;
  }
  cudaStreamCreate(&g_sgd_stream);
  cudaEventCreate(&g_sgd_done);
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const char *weights_dir = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  const std::string nnwt = std::string(weights_dir) + "/model.nnwt";
  if (!gpucompress::loadNNFromBinary(nnwt.c_str())) return 77;

  cm::NeuroPressNNPredictor predictor;
  if (!predictor.Load(weights_dir) || !predictor.IsReady()) return 77;

  CompContext ctx{};
  ctx.stream = stream;
  if (cudaMalloc(&ctx.d_fused_infer_output, sizeof(NNInferenceOutput)) != cudaSuccess ||
      cudaMalloc(&ctx.d_fused_top_actions, 32 * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&ctx.d_fused_costs, 32 * sizeof(float)) != cudaSuccess) {
    return 77;
  }

  const size_t kElems = 1u << 20;
  const size_t kBytes = kElems * sizeof(float);
  /* Error bounds. 0.0 is the production default and masks the 16 quantize
     actions entirely (rank_val = -INFINITY when quant==1 && eb<=0, on both
     sides) -- so a lossless-only run can never show quantization being chosen
     and says nothing about half the action space. The nonzero bounds unmask
     it. */
  const double kEbs[] = {0.0, 1e-3, 1e-1};
  float *d_chunk = nullptr;
  if (cudaMalloc(&d_chunk, kBytes) != cudaSuccess) return 77;

  AutoStatsGPU *d_stats = nullptr;
  if (cudaMalloc(&d_stats, sizeof(AutoStatsGPU)) != cudaSuccess) return 77;

  /* Upstream's ranking defaults; Clio's RankingWeights already carries the
     same numbers, so this only turns the cost model on. */
  cm::RankingWeights rank;
  rank.use_cost_model = true;

  std::printf("%-3s %-6s %7s  %-20s %-9s %-9s %-9s %-8s  %s\n", "chk", "eb",
              "entropy", "chosen action/codec", "pred_ratio", "pred_ct",
              "pred_dt", "pred_psnr", "side");
  std::printf("%-3s %-6s %7s  %-20s %-9s %-9s %-9s %-8s  %s\n", "---",
              "------", "-------", "--------------------", "---------",
              "---------", "---------", "--------", "----");

  int n_quant = 0, n_shuf = 0;
  for (int c = 0; c < 8; ++c) {
    const std::vector<float> host = MakeChunk(kElems, c);
    if (cudaMemcpy(d_chunk, host.data(), kBytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      continue;
    }

    /* Each side computes its own statistics from these bytes. */
    double n_ent = 0, n_mad = 0, n_der = 0;
    if (gpucompress::runStatsOnlyPipeline(d_chunk, kBytes, stream, &n_ent,
                                          &n_mad, &n_der) != 0) {
      continue;
    }
    cudaStreamSynchronize(stream);

    AutoStatsGPU h{};
    h.entropy = n_ent; h.mad_normalized = n_mad; h.deriv_normalized = n_der;
    h.num_elements = kElems;
    cudaMemcpy(d_stats, &h, sizeof(h), cudaMemcpyHostToDevice);

    for (double kEb : kEbs) {

    /* --- upstream: chosen action comes back as the return value --- */
    int up_action = -1;
    float up_ratio = 0, up_ct = 0, up_dt = 0, up_psnr = 0;
    const int rc = gpucompress::runNNFusedInferenceCtx(
        d_stats, kBytes, kEb, stream, &ctx, &up_action, &up_ratio, &up_ct,
        &up_dt, &up_psnr);
    cudaStreamSynchronize(stream);
    if (rc < 0) continue;
    up_action = rc;   /* the call returns the action it picked */

    /* --- Clio: same 32 candidates, its own stats, GPU ranking --- */
    const void *clio_stats = ctp::ComputeDeviceStatsResident(
        d_chunk, kElems, ctp::DataType::FLOAT32, ctp::DeviceStatsStream());
    if (clio_stats == nullptr) continue;

    std::vector<cm::CompressionFeatures> batch(32);
    for (int a = 0; a < 32; ++a) {
      const int quant = (a / 8) % 2, shuf = (a / 16) % 2;
      cm::CompressionFeatures &f = batch[a];
      f.chunk_size_bytes = static_cast<double>(kBytes);
      f.data_type_float = 1.0;
      f.quantize = quant; f.byte_shuffle = shuf;
      f.error_bound = quant ? kEb : 0.0;
      f.library_config_id = kBaseIdForAlgo[a % 8] * 10 + 2;
      f.config_balanced = 1.0;
    }
    std::vector<int> order;
    const auto preds = predictor.PredictBatchDeviceStats(
        clio_stats, batch, ctp::DeviceStatsStream(), &rank, &order);
    if (preds.empty() || order.empty()) continue;

    const int cl_slot = order[0];
    /* Slot order == action order: the bridge enumerates shuffle-outer,
       quant-inner so slot i IS action i. */
    const int cl_action = cl_slot;

    char up_s[48], cl_s[48];
    std::snprintf(up_s, sizeof(up_s), "%2d %-9s q%d s%d", up_action,
                  kAlgoName[up_action % 8], (up_action / 8) % 2,
                  (up_action / 16) % 2);
    std::snprintf(cl_s, sizeof(cl_s), "%2d %-9s q%d s%d", cl_action,
                  kAlgoName[cl_action % 8], (cl_action / 8) % 2,
                  (cl_action / 16) % 2);

    const bool agree = (up_action == cl_action);
    const cm::CompressionPrediction &cp = preds[cl_slot];
    /* Upstream first, Clio directly beneath it, so a divergence in any column
       is a vertical mismatch rather than something to scan across. */
    std::printf("%-3d %-6g %7.4f  %-20s %-9.4f %-9.4f %-9.4f %-8.3f  %s\n", c,
                kEb, n_ent, up_s, up_ratio, up_ct, up_dt, up_psnr, "upstream");
    std::printf("%-3s %-6s %7s  %-20s %-9.4f %-9.4f %-9.4f %-8.3f  %s%s\n", "",
                "", "", cl_s, cp.compression_ratio, cp.compression_time_ms,
                cp.decompression_time_ms, cp.psnr_db, "clio",
                agree ? "" : "   <-- DISAGREE");

    if ((up_action / 8) % 2) ++n_quant;
    if ((up_action / 16) % 2) ++n_shuf;

    ++g_checks;
    if (!agree) {
      ++g_failures;
    }
    }  // error-bound sweep
  }

  std::printf("\n  chose QUANTIZATION in %d of %ld cases; SHUFFLE in %d\n",
              n_quant, g_checks, n_shuf);
  /* If neither preprocessor is ever chosen, the table is only exercising the
     plain-codec third of the action space and "they agree" is a weaker claim
     than it looks. */
  ++g_checks;
  if (n_quant == 0) {
    ++g_failures;
    std::printf("  FAIL quantization was never selected -- the quantize half "
                "of the action space went untested\n");
  }
  ++g_checks;
  if (n_shuf == 0) {
    ++g_failures;
    std::printf("  FAIL shuffle was never selected\n");
  }

  std::printf("\n===== %ld cases, %d disagreements =====\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
