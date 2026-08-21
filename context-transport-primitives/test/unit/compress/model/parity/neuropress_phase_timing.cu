/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */
/**
@file neuropress_phase_timing.cu
@brief How long the prediction and learning phases take, on each side.

Reports rather than asserts: this is a measurement, and a machine-dependent
one. Failing a build on a timing threshold would make it flaky on the first
busy runner.

TWO numbers per phase, and the distinction matters more than either alone:

  HOST   what the calling thread pays before it can do anything else. With a
         fire-and-forget launch this is just the enqueue cost -- the GPU has
         not finished, and that is the point.
  TOTAL  wall time for N calls with ONE synchronize at the end, divided by N.
         This is the real throughput, and it does include the GPU work.

Quoting only the host number would make any fire-and-forget path look
arbitrarily fast; quoting only the total would hide the latency the caller
actually escaped. Both are printed.
*/

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

#include "api/internal.hpp"
#include "nn/nn_weights.h"
#include "stats/auto_stats_gpu.h"

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

namespace cm = ctp::compress::model;

namespace {
using Clock = std::chrono::high_resolution_clock;
double UsSince(Clock::time_point t0, int n) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() /
         n;
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

  /* Shared inputs. */
  AutoStatsGPU h{};
  h.entropy = 4.5; h.mad_normalized = 0.19; h.deriv_normalized = 0.24;
  h.num_elements = 1 << 20;
  AutoStatsGPU *d_stats = nullptr;
  if (cudaMalloc(&d_stats, sizeof(h)) != cudaSuccess) return 77;
  cudaMemcpy(d_stats, &h, sizeof(h), cudaMemcpyHostToDevice);

  CompContext ctx{};
  ctx.stream = stream;
  if (cudaMalloc(&ctx.d_fused_infer_output, sizeof(NNInferenceOutput)) != cudaSuccess ||
      cudaMalloc(&ctx.d_fused_top_actions, 32 * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&ctx.d_fused_costs, 32 * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&ctx.d_sgd_samples, NN_MAX_SGD_SAMPLES * sizeof(SGDSample)) != cudaSuccess ||
      cudaMalloc(&ctx.d_sgd_grad_buffer, NN_SGD_GRAD_SIZE * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&ctx.d_sgd_output, sizeof(SGDOutput)) != cudaSuccess) {
    return 77;
  }
  cudaMemset(ctx.d_sgd_grad_buffer, 0, NN_SGD_GRAD_SIZE * sizeof(float));

  /* Clio's device-stats handle over a real buffer, so its inference runs the
     same shape as production rather than a host-matrix call. */
  const size_t kElems = 1u << 20;
  float *d_chunk = nullptr;
  if (cudaMalloc(&d_chunk, kElems * sizeof(float)) != cudaSuccess) return 77;
  {
    std::vector<float> host(kElems);
    for (size_t i = 0; i < kElems; ++i) host[i] = static_cast<float>(i % 4096) * 0.001f;
    cudaMemcpy(d_chunk, host.data(), kElems * sizeof(float), cudaMemcpyHostToDevice);
  }

  std::vector<cm::CompressionFeatures> batch(32);
  for (int a = 0; a < 32; ++a) {
    const int quant = (a / 8) % 2, shuf = (a / 16) % 2;
    static const int kBase[8] = {13, 14, 17, 16, 15, 18, 23, 24};
    cm::CompressionFeatures &f = batch[a];
    f.chunk_size_bytes = static_cast<double>(kElems * sizeof(float));
    f.data_type_float = 1.0;
    f.quantize = quant; f.byte_shuffle = shuf;
    f.error_bound = quant ? 1e-3 : 0.0;
    f.library_config_id = kBase[a % 8] * 10 + 2;
    f.config_balanced = 1.0;
  }

  std::vector<cm::TrainingLabels> labels(1);
  labels[0].compression_ratio = 3.0;
  labels[0].compression_time_ms = 12.0;
  labels[0].decompression_time_ms = 0.0;
  labels[0].psnr_db = -1.0;
  std::vector<cm::CompressionFeatures> tfeat(1, batch[4]);

  SGDSample us{};
  us.action = 4; us.actual_ratio = 3.0f; us.actual_comp_time = 12.0f;
  us.actual_decomp_time = 0.0f; us.actual_psnr = 120.0f;

  const int kWarm = 20, kN = 500;

  std::printf("%-34s %12s %12s\n", "phase", "host us", "total us");
  std::printf("%-34s %12s %12s\n", "----------------------------------",
              "------------", "------------");

  /* ---------------- LEARNING ---------------- */
  for (int i = 0; i < kWarm; ++i)
    gpucompress::runNNSGDCtx(d_stats, &us, 1, kElems * 4, 0.0, 0.01f, &ctx);
  cudaDeviceSynchronize();
  auto t0 = Clock::now();
  for (int i = 0; i < kN; ++i)
    gpucompress::runNNSGDCtx(d_stats, &us, 1, kElems * 4, 0.0, 0.01f, &ctx);
  const double up_sgd_host = UsSince(t0, kN);
  cudaDeviceSynchronize();
  const double up_sgd_total = UsSince(t0, kN);
  std::printf("%-34s %12.2f %12.2f\n", "learning  upstream runNNSGDCtx",
              up_sgd_host, up_sgd_total);

  for (int i = 0; i < kWarm; ++i) predictor.Train(tfeat, labels);
  cudaDeviceSynchronize();
  t0 = Clock::now();
  for (int i = 0; i < kN; ++i) predictor.Train(tfeat, labels);
  const double cl_sgd_host = UsSince(t0, kN);
  cudaDeviceSynchronize();
  const double cl_sgd_total = UsSince(t0, kN);
  std::printf("%-34s %12.2f %12.2f\n", "learning  clio Train()", cl_sgd_host,
              cl_sgd_total);

  /* ---------------- PREDICTION ---------------- */
  int chosen = -1;
  for (int i = 0; i < kWarm; ++i)
    gpucompress::runNNFusedInferenceCtx(d_stats, kElems * 4, 1e-3, stream, &ctx,
                                        &chosen);
  cudaDeviceSynchronize();
  t0 = Clock::now();
  for (int i = 0; i < kN; ++i)
    gpucompress::runNNFusedInferenceCtx(d_stats, kElems * 4, 1e-3, stream, &ctx,
                                        &chosen);
  const double up_inf_host = UsSince(t0, kN);
  cudaDeviceSynchronize();
  const double up_inf_total = UsSince(t0, kN);
  std::printf("%-34s %12.2f %12.2f\n", "prediction upstream fused inference",
              up_inf_host, up_inf_total);

  const void *clio_stats = ctp::ComputeDeviceStatsResident(
      d_chunk, kElems, ctp::DataType::FLOAT32, ctp::DeviceStatsStream());
  if (clio_stats != nullptr) {
    for (int i = 0; i < kWarm; ++i)
      predictor.PredictBatchDeviceStats(clio_stats, batch,
                                        ctp::DeviceStatsStream());
    cudaDeviceSynchronize();
    t0 = Clock::now();
    for (int i = 0; i < kN; ++i)
      predictor.PredictBatchDeviceStats(clio_stats, batch,
                                        ctp::DeviceStatsStream());
    const double cl_inf_host = UsSince(t0, kN);
    cudaDeviceSynchronize();
    const double cl_inf_total = UsSince(t0, kN);
    std::printf("%-34s %12.2f %12.2f\n", "prediction clio device-stats",
                cl_inf_host, cl_inf_total);
  }

  std::printf("\n(host = what the caller waits for; total = N calls with one\n"
              " synchronize at the end, i.e. real throughput including GPU work)\n");
  return 0;
}
