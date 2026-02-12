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
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_LOCAL_SCHED_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_LOCAL_SCHED_H_

#include <atomic>
#include <vector>

#include "chimaera/scheduler/scheduler.h"

namespace chi {

/**
 * Local scheduler implementation.
 * Uses PID+TID hash-based lane mapping and provides no rebalancing.
 * All workers process tasks; scheduler tracks worker groups for routing decisions.
 */
class LocalScheduler : public Scheduler {
 public:
  LocalScheduler() : net_worker_(nullptr), gpu_worker_(nullptr) {}
  ~LocalScheduler() override = default;

  void DivideWorkers(WorkOrchestrator *work_orch) override;
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task) override;
  void RebalanceWorker(Worker *worker) override;
  void AdjustPolling(RunContext *run_ctx) override;
  Worker *GetGpuWorker() const override { return gpu_worker_; }

 private:
  u32 MapByPidTid(u32 num_lanes);

  std::vector<Worker *> scheduler_workers_;
  Worker *net_worker_;
  Worker *gpu_worker_;
  std::atomic<u32> next_sched_idx_{0};
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_LOCAL_SCHED_H_
