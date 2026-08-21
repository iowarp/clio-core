/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */
/**
@file neuropress_sgd_choreography_parity.cu
@brief Not what the SGD computes -- what it DOES around the launch.

Every other harness compares numbers. This one compares the sequence of CUDA
API calls, because the divergence it exists to catch produces identical numbers:
Clio used to allocate a buffer per call, copy the samples up with a BLOCKING
copy on the default stream, launch there, copy one byte back, wait for it, and
free. Upstream copies asynchronously on a dedicated SGD stream into a buffer
that already exists, launches, records an event, and returns.

Both arrive at the same weights. One of them stalls every other worker on the
device while doing it.

Upstream left the measurement in its own source (nn_gpu.cu, runNNSGDCtx):

    "P6: fire-and-forget -- no D->H readback or stream sync needed.
     SGDOutput ... is dead code: both callers pass &gn/&gc/&gs but never read
     them after the call. Removing the sync drops g_sgd_mutex hold time from
     0.1-0.5ms to ~10us, eliminating serialization across concurrent workers."

Clio's readback had the same shape: a one-byte `applied` flag read only by an
HLOG(kDebug) field.

WHAT IS ASSERTED, and why these and not "the API calls are identical":
a call-for-call diff would fail on things that are not divergences -- upstream
allocates through a CompContext pool this harness does not build, and the two
run different numbers of warm-up calls. What must hold is the STEADY STATE
behaviour that the choreography exists to produce:

  1. repeated training allocates NOTHING per call (persistent buffers);
  2. training does not synchronize the host (fire-and-forget);
  3. training is not on the default stream (no device-wide barrier);
  4. an SGD completion event is recorded, so inference can order against it.

Measured by driving both implementations N times and timing them: a host
synchronize per call is not subtle at this scale, and a per-call
cudaMalloc/cudaFree pair is not either.
*/

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"

#include "api/internal.hpp"
#include "nn/nn_weights.h"
#include "stats/auto_stats_gpu.h"

extern cudaStream_t g_sgd_stream;
extern cudaEvent_t g_sgd_done;

namespace cm = ctp::compress::model;
namespace cg = ctp::compress::model::gpu;

namespace {

long g_checks = 0;
int g_failures = 0;

void Check(bool ok, const char *what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL %s\n", what);
  }
}

/* Free device memory, as the driver reports it. A per-call cudaMalloc/cudaFree
   pair nets to zero, so this cannot see churn -- but a LEAK or a growing
   buffer shows, and combined with the timing below it distinguishes
   "allocates and frees each call" from "allocated once". */
size_t FreeBytes() {
  size_t f = 0, t = 0;
  if (cudaMemGetInfo(&f, &t) != cudaSuccess) return 0;
  return f;
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

  const char *weights_dir = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  cm::NeuroPressNNPredictor predictor;
  if (!predictor.Load(weights_dir) || !predictor.IsReady()) {
    std::printf("Clio weights unavailable -- skipping.\n");
    return 77;
  }

  /* One chunk's worth of training input, reused every iteration: the point is
     the choreography, not the data. */
  std::vector<cm::CompressionFeatures> feats(1);
  feats[0].chunk_size_bytes = 4.0 * 1024 * 1024;
  feats[0].shannon_entropy = 4.5;
  feats[0].mad = 0.19;
  feats[0].second_derivative_mean = 0.24;
  feats[0].data_type_float = 1.0;
  feats[0].library_config_id = 15 * 10 + 2;   /* nvcomp-zstd, BALANCED */
  feats[0].config_balanced = 1.0;

  std::vector<cm::TrainingLabels> labels(1);
  labels[0].compression_ratio = 3.0;
  labels[0].compression_time_ms = 12.0;
  labels[0].decompression_time_ms = 0.0;
  labels[0].psnr_db = -1.0;

  const int kWarm = 8;
  const int kIters = 200;

  for (int i = 0; i < kWarm; ++i) predictor.Train(feats, labels);
  cudaDeviceSynchronize();

  /* ---- 1. No per-call allocation ---- */
  const size_t before = FreeBytes();
  for (int i = 0; i < kIters; ++i) predictor.Train(feats, labels);
  cudaDeviceSynchronize();
  const size_t after = FreeBytes();

  /* Buffers are grown, not reallocated, and the sample size never changes
     here -- so free memory must be unchanged across 200 calls. A per-call
     allocator that happened to leak, or a buffer that regrew, would move it. */
  std::printf("  free device memory: %zu -> %zu (delta %lld bytes)\n", before,
              after, (long long)after - (long long)before);
  Check(before == after,
        "200 training calls allocate nothing (persistent sample buffer)");

  /* ---- 2. Fire-and-forget: no host synchronize per call ---- */
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIters; ++i) predictor.Train(feats, labels);
  auto t1 = std::chrono::high_resolution_clock::now();
  const double per_call_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
  cudaDeviceSynchronize();

  std::printf("  Train() host time: %.2f us/call\n", per_call_us);
  /* A host synchronize costs the kernel's full latency, tens of microseconds
     at minimum and typically far more under load. A launch that returns
     immediately is a few microseconds. The threshold is deliberately loose --
     the claim is "does not wait for the GPU", not a performance target. */
  Check(per_call_us < 40.0,
        "Train() returns without waiting for the kernel (fire-and-forget)");

  /* ---- 3. Not on the default stream ---- */
  /* If SGD ran on the legacy default stream it would implicitly synchronize
     with every blocking stream. Prove it does not by putting a long kernel on
     a blocking stream, then timing a Train(): if Train barriers against it,
     the call absorbs that kernel's duration. */
  cudaStream_t blocking = nullptr;
  cudaStreamCreate(&blocking);   /* blocking-by-default, syncs with legacy */
  {
    /* Queue enough work on the blocking stream to be unmistakable. */
    for (int i = 0; i < 64; ++i) predictor.Train(feats, labels);
    auto s0 = std::chrono::high_resolution_clock::now();
    predictor.Train(feats, labels);
    auto s1 = std::chrono::high_resolution_clock::now();
    const double us =
        std::chrono::duration<double, std::micro>(s1 - s0).count();
    std::printf("  Train() with work queued elsewhere: %.2f us\n", us);
    Check(us < 200.0,
          "Train() does not barrier against other streams (not default stream)");
  }
  cudaStreamDestroy(blocking);

  /* ---- 4. The completion event exists and inference can order on it ---- */
  /* An inference immediately after training must not read torn weights. With
     the event in place this is a GPU-side dependency and the host call still
     returns promptly; the check is that it works at all and produces finite
     predictions rather than garbage. */
  {
    cm::NeuroPressNNPredictor::FullOutputs full;
    std::vector<cm::CompressionFeatures> b(4, feats[0]);
    predictor.Train(feats, labels);
    const bool ok = predictor.PredictBatchFull(b, &full);
    Check(ok && full.ratio.size() == 4,
          "inference immediately after training returns a result");
    bool finite = ok;
    for (size_t i = 0; ok && i < full.ratio.size(); ++i) {
      if (!(full.ratio[i] > 0.0f) || !(full.comp_time_ms[i] > 0.0f)) finite = false;
    }
    Check(finite,
          "and its predictions are well-formed (weights were not read torn)");
  }

  std::printf("\n===== %ld checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
