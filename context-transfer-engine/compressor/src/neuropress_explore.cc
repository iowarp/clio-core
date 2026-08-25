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
    // This slot's OWN events, the same quantity the primary is measured with.
    // No host-clock fallback: a missing reading means no launch happened.
    s->time_ms = (s->async.kernel_ms >= 0.0) ? s->async.kernel_ms : 0.0;
    s->compressed_size = s->ok ? s->async.compressed_size : 0;
  }
#endif
}

}  // namespace clio::cte::compressor
