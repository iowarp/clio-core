/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

// Out-of-line implementation of ThreadModel::BusyWait. Lives in a source file
// (not the header) so the tiered wait policy is defined exactly once and is
// independent of the active thread model: it drives the process-wide
// CTP_THREAD_MODEL singleton, whose API is uniform across every backend
// (every model provides Yield() and SleepForUs()).

#include "clio_ctp/thread/thread_model_manager.h"

// BusyWait drives std::function + the host Timepoint clock and is declared in
// thread_model.h only under CTP_IS_HOST. Guard the definition with the same
// condition so the CUDA/device compilation pass (CTP_IS_HOST == 0), which sees
// no such member, does not try to define one (nvcc: "class ThreadModel has no
// member BusyWait").
#if CTP_IS_HOST
#include <cstddef>
#include <functional>

#include "clio_ctp/util/timer.h"

namespace ctp::thread {

void ThreadModel::BusyWait(const std::function<bool()> &cond) {
  // The 5ms yield window keeps medium waits responsive without pinning a core
  // on a pure busy-spin; phase 2 then backs off to idle.
  constexpr double kYieldUs = 5000.0;
  constexpr size_t kMaxSleepUs = 1024;  // ~1ms cap on the backoff

  // Phase 1: yield-spin. Returns the core each iteration (no pure busy-poll)
  // but stays hot enough to react within a scheduler tick.
  ctp::Timepoint phase_start;
  phase_start.Now();
  while (true) {
    if (cond()) return;
    CTP_THREAD_MODEL->Yield();
    ctp::Timepoint now;
    now.Now();
    if (phase_start.GetUsecFromStart(now) >= kYieldUs) break;
  }

  // Phase 2: exponential-backoff sleep (1, 2, 4, ... us, capped) until done.
  size_t sleep_us = 1;
  while (!cond()) {
    CTP_THREAD_MODEL->SleepForUs(sleep_us);
    if (sleep_us < kMaxSleepUs) sleep_us <<= 1;
  }
}

}  // namespace ctp::thread
#endif  // CTP_IS_HOST
