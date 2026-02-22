/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Copyright 2024 IOWarp contributors
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_

#include <atomic>
#include <vector>

#include "chimaera/scheduler/scheduler.h"

namespace chi {

/**
 * Default scheduler implementation with I/O-size-based routing.
 * Routes tasks based on io_size_: small I/O and metadata go to the scheduler
 * worker (worker 0), large I/O (>= 4KB) goes to dedicated I/O workers via
 * round-robin, and network tasks go to the last worker.
 */
class DefaultScheduler : public Scheduler {
 public:
  DefaultScheduler()
      : scheduler_worker_(nullptr), net_worker_(nullptr),
        gpu_worker_(nullptr), next_io_idx_{0} {}
  ~DefaultScheduler() override = default;

  void DivideWorkers(WorkOrchestrator *work_orch) override;
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task,
                     Container *container) override;
  void RebalanceWorker(Worker *worker) override;
  void AdjustPolling(RunContext *run_ctx) override;
  Worker *GetGpuWorker() const override { return gpu_worker_; }

 private:
  static constexpr size_t kLargeIOThreshold = 4096;  ///< I/O size threshold

  Worker *scheduler_worker_;              ///< Worker 0: metadata + small I/O
  std::vector<Worker *> io_workers_;      ///< Workers 1..N-2: large I/O
  Worker *net_worker_;                    ///< Worker N-1: network
  Worker *gpu_worker_;                    ///< GPU queue polling worker
  std::atomic<u32> next_io_idx_{0};       ///< Round-robin index for I/O workers
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_DEFAULT_SCHED_H_
