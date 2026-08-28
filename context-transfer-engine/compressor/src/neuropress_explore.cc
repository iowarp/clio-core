/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_explore.cc
 * @brief Sweep steps that need no state beyond the slots themselves.
 */

#include "clio_cte/compressor/neuropress_explore.h"

#include <clio_ctp/compress/preprocess/byte_shuffle.h>
#include <clio_ctp/compress/preprocess/quality_metrics_gpu.h>
#include <clio_ctp/compress/preprocess/quantization.h>

#include <chrono>
#include <cstdlib>

namespace clio::cte::compressor {

/**
 * Measure each explored candidate's decompression time by decompressing its
 * own output back, instead of substituting the NN's prediction.
 *
 * OFF by default, and that default is not timidity. Upstream NeuroPress never
 * decompresses at write time -- verified: gpucompress_compress.cpp passes
 * pred_dt to compute_cost for every alternative, its primary_decomp_time_ms
 * is declared and never assigned, and an nsys trace of its own
 * explore_algo_patterns shows 13,430 kernel launches and zero decompression
 * kernels. Turning this on makes Clio's SELECTION diverge from upstream's on
 * identical data, which is the comparison the benchmark suite rests on.
 *
 * It also costs: decompression here runs 1.2-30x the codec's compression time
 * depending on the algorithm, so the sweep does not merely double.
 */
bool MeasureExploreDecompTime() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_EXPLORE_MEASURE_DT");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return on;
}

bool MeasureExploreQuality() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_MEASURE_QUALITY");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return on;
}

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
bool MeasureStoredChunkQuality(
    const void *orig_device, size_t orig_bytes, const void *compressed,
    size_t compressed_size, size_t post_transform_size, uint32_t shuffle,
    const ctp::compress::preprocess::DeviceQuantizeParams *quant,
    ctp::compress::preprocess::QualityMetrics *out) {
  namespace pp = ctp::compress::preprocess;
  if (!orig_device || !compressed || !out) return false;
  if (orig_bytes == 0 || compressed_size == 0 || post_transform_size == 0) {
    return false;
  }
  if ((orig_bytes % sizeof(float)) != 0) return false;

  // The default stream: NvComp::CachedStream() is private, and this is a
  // diagnostic path where synchronizing with everything is acceptable.
  cudaStream_t stream = nullptr;

  void *d_a = nullptr, *d_b = nullptr, *d_q = nullptr;
  auto cleanup = [&] {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_q) cudaFree(d_q);
  };

  // 1. Codec inverse, into a buffer the size the codec was FED.
  if (cudaMalloc(&d_a, post_transform_size) != cudaSuccess) return false;
  if (!ctp::NvComp::DecompressInto(compressed, post_transform_size, d_a,
                                   stream)) {
    cleanup();
    return false;
  }
  const void *cur = d_a;

  // 2. Byte-shuffle inverse. `shuffle` is the WIDTH, not a flag.
  if (shuffle > 0) {
    if (cudaMalloc(&d_b, post_transform_size) != cudaSuccess) {
      cleanup();
      return false;
    }
    if (!pp::ByteUnshuffleDevice(cur, d_b, post_transform_size, shuffle)) {
      cleanup();
      return false;
    }
    cur = d_b;
  }

  // 3. Quantizer inverse -- the step that returns the data to float32 in the
  //    original domain, and the only one that can carry real error.
  if (quant != nullptr) {
    if (cudaMalloc(&d_q, orig_bytes) != cudaSuccess) { cleanup(); return false; }
    if (!pp::DequantizeDevice(cur, orig_bytes / sizeof(float), *quant, d_q)) {
      cleanup();
      return false;
    }
    cur = d_q;
  }

  pp::QualityMetrics m;
  const bool ok = pp::ComputeQualityDevice(orig_device, cur,
                                           orig_bytes / sizeof(float), stream,
                                           &m);
  if (ok) *out = m;
  cleanup();
  return ok;
}

namespace {

/** The sweep's per-candidate wrapper: pull the slot's transforms and defer to
 *  the shared chain above. */
void MeasureCandidateQuality(ExploreSlot *s) {
  if (!s || !s->ok || s->compressed_size == 0) return;
  if (!s->orig_device || s->orig_bytes == 0) return;
  if (!s->gpu || !s->async.d_out) return;
  ctp::compress::preprocess::QualityMetrics m;
  if (MeasureStoredChunkQuality(s->orig_device, s->orig_bytes, s->async.d_out,
                                s->compressed_size, s->compress_size,
                                s->applied_shuffle,
                                s->applied_quant ? &s->quant_params : nullptr,
                                &m)) {
    s->quality = m;
    s->have_quality = true;
  }
}

}  // namespace
#endif

/** Sweep step 2: finish every launched slot. Kept separate from the launch
 *  step -- every slot must be in flight before any is waited on, or the sweep
 *  serializes into K sequential codec calls. */
void CollectExploreSlots(std::vector<std::unique_ptr<ExploreSlot>> &slots) {
  // All K are already running: the concurrency is the device's, not OS
  // threads. Compressor::Compress() cannot do that -- it uses the thread's
  // cached stream and synchronizes, so two calls serialize by construction.

  // Codecs with no async path compress synchronously here. NeuroPress's
  // action space is all GPU, so this branch is not taken on the real path.
  for (auto& sp : slots) {
    ExploreSlot* s = sp.get();
    if (s->collected || s->gpu != nullptr) continue;
    s->collected = true;
    const auto t0 = std::chrono::high_resolution_clock::now();
    size_t sz = s->capacity;
    s->ok = s->compressor->Compress(s->out_ptr, sz, s->input,
                                    s->compress_size);
    const double wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - t0)
                               .count();
    const double kernel_ms = ctp::LastCodecKernelMs();
    s->time_ms = (kernel_ms >= 0.0) ? kernel_ms : wall_ms;
    s->compressed_size = s->ok ? sz : 0;
  }
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
  for (auto& sp : slots) {
    ExploreSlot* s = sp.get();
    if (s->collected || !s->launched || s->gpu == nullptr) continue;
    s->collected = true;
    s->ok = s->gpu->CompressFinish(&s->async);
    // This slot's OWN events, the same quantity the primary is measured with.
    // No host-clock fallback: a missing reading means no launch happened.
    s->time_ms = (s->async.kernel_ms >= 0.0) ? s->async.kernel_ms : 0.0;
    s->compressed_size = s->ok ? s->async.compressed_size : 0;
    // Decompress the candidate's own output back, on the slot's own stream,
    // timed with the slot's own events -- so dt is the same KIND of quantity
    // as the ct beside it. Only when the codec actually produced something:
    // a failed or non-beneficial candidate has nothing to invert.
    if (s->ok && s->compressed_size > 0 && MeasureExploreDecompTime()) {
      if (s->gpu->DecompressMeasure(&s->async, s->compress_size)) {
        s->decomp_time_ms = s->async.decomp_ms;
      }
    }
    // Quality is measured independently of dt: the two answer different
    // questions and one being off must not silence the other.
    if (s->ok && s->compressed_size > 0 && MeasureExploreQuality()) {
      MeasureCandidateQuality(s);
    }
  }
#endif
}

}  // namespace clio::cte::compressor
