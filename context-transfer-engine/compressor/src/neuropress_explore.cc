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

namespace clio::cte::compressor {

/**
 * Sweep step 2: finish every launched slot and record what it produced.
 *
 * Separated from the launch step because the two must not interleave --
 * every slot has to be in flight before any is waited on, or the sweep
 * serializes into K sequential codec calls and stops measuring what it
 * was written to measure.
 *
 * Takes only the slots: what a codec produced is entirely recorded in
 * them, so this step needs nothing about the chunk or the selection.
 */
void CollectExploreSlots(std::vector<std::unique_ptr<ExploreSlot>> &slots) {
  // Upstream's split, on one thread: each candidate's quantize,
  // shuffle and codec launch were queued above on that candidate's own
  // CUDA stream and none of them waited, so by the time the prep loop
  // ends all K are running together. Nothing here is parallelized with
  // OS threads -- the concurrency is the device's, and it costs one
  // stream plus two events per slot.
  //
  // Compressor::Compress() cannot be used for that: it takes the
  // thread's single cached stream and synchronizes before returning,
  // so two calls on one thread serialize by construction. NvComp's
  // OpenSlot/CompressLaunch/CompressFinish split it at exactly the
  // points upstream splits it.
  //
  // A codec with no async path (any CPU compressor) is compressed
  // synchronously here instead. NeuroPress's action space is entirely
  // GPU codecs, so on the path that matters this branch is not taken.
  for (auto& sp : slots) {
    ExploreSlot* s = sp.get();
    if (s->gpu != nullptr) continue;
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
    if (!s->launched || s->gpu == nullptr) continue;
    s->ok = s->gpu->CompressFinish(&s->async);
    // Kernel time from this slot's OWN events, bracketing its codec
    // launch alone -- the same quantity the primary is measured with,
    // and the same one upstream scores its slots on. There is no
    // host-clock fallback here: the events are created with the slot,
    // so a missing reading means the launch did not happen.
    s->time_ms = (s->async.kernel_ms >= 0.0) ? s->async.kernel_ms : 0.0;
    s->compressed_size = s->ok ? s->async.compressed_size : 0;
  }
#endif
}

}  // namespace clio::cte::compressor
