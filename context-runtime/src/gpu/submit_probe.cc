/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_runtime/gpu/submit_probe.h"

#include <chrono>
#include <cstdio>

namespace clio::run::gpu {

SubmitProbe &SubmitProbe::Get() {
  static SubmitProbe probe;
  return probe;
}

unsigned long long SubmitProbe::NowNs() {
  // On Linux, libstdc++'s steady_clock is clock_gettime(CLOCK_MONOTONIC, ...)
  // under the hood -- same syscall, same resolution. This also gets us
  // QueryPerformanceCounter on Windows, where CLOCK_MONOTONIC doesn't exist.
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void SubmitProbe::Enable(unsigned cap) {
  if (on_) return;
  // Allocated once, up front. The task path must never allocate: a malloc inside
  // the hop we are timing would land in the measurement.
  recs_ = new SubmitProbeHostRec[cap]();
  cap_ = cap;
  count_ = 0;
  on_ = true;
}

SubmitProbeHostRec *SubmitProbe::Open(unsigned long long task_ptr,
                                      unsigned long long t_pop,
                                      unsigned worker_id,
                                      unsigned long long idle_iters,
                                      unsigned sleep_us) {
  if (!on_ || count_ >= cap_) return nullptr;
  // Single GPU lane -> single popping worker, so the bump is uncontended by
  // construction. (If a future scheduler assigns gpu_lanes_ to more than one
  // worker this needs an atomic; today it cannot happen -- see DefaultScheduler.)
  SubmitProbeHostRec *r = &recs_[count_++];
  r->task_ptr = task_ptr;
  r->t_pop = t_pop;
  r->worker_pop = worker_id;
  r->idle_iters_at_pop = idle_iters;
  r->sleep_us_at_pop = sleep_us;
  return r;
}

void SubmitProbe::Dump(const char *path) const {
  if (!on_) return;
  FILE *f = std::fopen(path, "w");
  if (!f) return;
  std::fprintf(f,
               "task_ptr,t_pop,t_pre_route,t_post_route,t_exec_start,"
               "t_sendout_begin,t_sendout_end,idle_iters_at_pop,sleep_us_at_pop,"
               "worker_pop,worker_exec\n");
  for (unsigned i = 0; i < count_; ++i) {
    const SubmitProbeHostRec &r = recs_[i];
    std::fprintf(f, "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%u,%u\n",
                 r.task_ptr, r.t_pop, r.t_pre_route, r.t_post_route,
                 r.t_exec_start, r.t_sendout_begin, r.t_sendout_end,
                 r.idle_iters_at_pop, r.sleep_us_at_pop, r.worker_pop,
                 r.worker_exec);
  }
  std::fclose(f);
}

}  // namespace clio::run::gpu
