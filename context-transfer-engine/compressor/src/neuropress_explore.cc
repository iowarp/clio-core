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
  }
#endif
}

}  // namespace clio::cte::compressor
