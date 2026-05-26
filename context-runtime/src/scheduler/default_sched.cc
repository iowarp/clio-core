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
#include "clio_runtime/scheduler/default_sched.h"

#include "clio_runtime/config_manager.h"
#include "clio_runtime/container.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/work_orchestrator.h"
#include "clio_runtime/worker.h"

namespace clio::run {

void DefaultScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) {
    return;
  }

  u32 total_workers = work_orch->GetTotalWorkerCount();

  scheduler_worker_ = nullptr;
  io_workers_.clear();
  net_send_worker_ = nullptr;
  net_recv_worker_ = nullptr;
  gpu_worker_ = nullptr;

  // Worker 0 is always the scheduler worker
  scheduler_worker_ = work_orch->GetWorker(0);

  // Layout, with worker count growing left-to-right:
  //   N=1: [sched]                                        — net workers alias sched (degenerate)
  //   N=2: [sched, net]                                   — single net thread (recv+send collapsed)
  //   N=3: [sched, net_send, net_recv]                    — split nets, no I/O lane
  //   N>=4: [sched, io..., net_send, net_recv]            — dedicated send + recv
  // The split decouples ZMQ send-side back-pressure from ROUTER recv polling
  // so SWIM heartbeat probe responses can drain even while peer DEALERs are
  // bottlenecked. With N<3 we fall back to a single net worker (degraded, but
  // keeps the runtime usable for trivial test configs).
  if (total_workers >= 3) {
    net_recv_worker_ = work_orch->GetWorker(total_workers - 1);
    net_send_worker_ = work_orch->GetWorker(total_workers - 2);
  } else if (total_workers == 2) {
    net_recv_worker_ = work_orch->GetWorker(1);
    net_send_worker_ = net_recv_worker_;
  } else {
    net_recv_worker_ = scheduler_worker_;
    net_send_worker_ = scheduler_worker_;
  }

  // GPU worker: needs its own worker so kGpuRecv polling doesn't fight with
  // periodic net tasks. Carve it out from before the net pair (N-3) when
  // we have headroom. Below that, alias onto the scheduler worker — the
  // GPU lane drain is cheap (a single Pop per iteration that returns
  // immediately when empty), and Worker::Run() already calls
  // ProcessNewTasksGpu() every loop iteration regardless of role. Leaving
  // gpu_worker_ null at small N hangs any kernel that calls future.Wait()
  // because nothing drains gpu2cpu_queue — see issue #448.
  if (total_workers >= 5) {
    gpu_worker_ = work_orch->GetWorker(total_workers - 3);
  } else {
    gpu_worker_ = scheduler_worker_;
  }

  // I/O workers live between the scheduler and the GPU/net block. With the
  // split there's one fewer worker available for I/O than in the old
  // layout; small I/O and metadata still fall back to the scheduler worker
  // via RuntimeMapTask, so this stays correct for any worker count.
  u32 io_upper_excl = total_workers >= 5 ? total_workers - 3
                    : total_workers >= 3 ? total_workers - 2
                                         : 1;
  for (u32 i = 1; i < io_upper_excl; ++i) {
    Worker *worker = work_orch->GetWorker(i);
    if (worker) {
      io_workers_.push_back(worker);
    }
  }

  // Register both net workers' lanes with the IPC manager so
  // EnqueueNetTask wakes the correct one based on the priority enqueued.
  IpcManager *ipc = CLIO_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(1);
    if (net_send_worker_ && net_recv_worker_) {
      ipc->SetNetLane(net_send_worker_->GetLane(),
                      net_recv_worker_->GetLane());
    }
  }

  int send_id = net_send_worker_ ? (int)net_send_worker_->GetId() : -1;
  int recv_id = net_recv_worker_ ? (int)net_recv_worker_->GetId() : -1;
  HLOG(kInfo,
       "DefaultScheduler: 1 scheduler worker (0), {} I/O workers, "
       "net_send_worker={}, net_recv_worker={}, gpu_worker={}",
       io_workers_.size(), send_id, recv_id,
       gpu_worker_ ? (int)gpu_worker_->GetId() : -1);
}

u32 DefaultScheduler::ClientMapTask(IpcManager *ipc_manager,
                                    const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  if (num_lanes == 0) {
    return 0;
  }

  Task *task_ptr = task.get();

  // Network tasks (Send/Recv from admin pool) → last lane
  if (task_ptr != nullptr && task_ptr->pool_id_ == chi::kAdminPoolId) {
    u32 method_id = task_ptr->method_;
    if (method_id == 14 || method_id == 15 || method_id == 20 || method_id == 21) {
      return num_lanes - 1;
    }
  }

  // Default: scheduler worker (lane 0)
  return 0;
}

u32 DefaultScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task,
                                      Container *container) {
  Task *task_ptr = task.get();

  // ---- Task group affinity: return early if group already pinned ----
  // Use the caller-supplied container directly (no static container lookup).
  Container *grp_container =
      (container != nullptr && task_ptr != nullptr &&
       !task_ptr->task_group_.IsNull())
          ? container
          : nullptr;
  if (grp_container != nullptr) {
    int64_t group_id = task_ptr->task_group_.id_;
    ScopedCoRwReadLock read_lock(grp_container->task_group_lock_);
    auto it = grp_container->task_group_map_.find(group_id);
    if (it != grp_container->task_group_map_.end() && it->second != nullptr) {
      return it->second->GetId();
    }
  }

  // ---- Normal routing: determine selected worker ----
  Worker *selected = nullptr;

  // Periodic Send/Recv routing. The split is by SOCKET OWNERSHIP, not by
  // verb — ZeroMQ sockets are not safe to share across threads, so each
  // socket has exactly one owner thread.
  //   14 = kSend         peer DEALER pool → net_send_worker
  //   15 = kRecv         peer ROUTER (9413) → net_recv_worker
  //   20 = kClientRecv   client-facing ROUTER (9416 / IPC) → net_recv_worker
  //   21 = kClientSend   same client-facing ROUTER → net_recv_worker
  // ClientSend and ClientRecv share the client server socket, so they
  // both run on net_recv_worker. The send worker is therefore dedicated
  // to outbound peer DEALER sends, which is the workload most likely to
  // back-pressure on ZMQ HWM — keeping it off the recv worker means
  // inbound SWIM probe responses can still be polled.
  if (task_ptr != nullptr && task_ptr->IsPeriodic()) {
    if (task_ptr->pool_id_ == chi::kAdminPoolId) {
      u32 method_id = task_ptr->method_;
      Worker *target = nullptr;
      if (method_id == 14) {
        target = net_send_worker_;
      } else if (method_id == 15 || method_id == 20 || method_id == 21) {
        target = net_recv_worker_;
      }
      if (target != nullptr) {
        return target->GetId();
      }
    }
  }

  // Route large I/O to dedicated I/O workers (round-robin).
  // predicted_stat_ is populated by IpcManager::BeginTask via
  // container->GetTaskStats(task) before this scheduler hook runs.
  if (selected == nullptr && task_ptr != nullptr && !io_workers_.empty()) {
    size_t io_size = 0;
    if (task_ptr->GetRunCtx()) {
      io_size = task_ptr->GetRunCtx()->predicted_stat_.io_size_;
    }
    if (io_size >= kLargeIOThreshold) {
      u32 idx = next_io_idx_.fetch_add(1, std::memory_order_relaxed) %
                static_cast<u32>(io_workers_.size());
      selected = io_workers_[idx];
    }
  }

  // Small I/O / metadata → scheduler worker
  if (selected == nullptr && scheduler_worker_ != nullptr) {
    selected = scheduler_worker_;
  }

  // Fallback to current worker
  if (selected == nullptr) {
    selected = worker;
  }

  // ---- Update group map after routing decision ----
  if (grp_container != nullptr && task_ptr != nullptr && selected != nullptr) {
    int64_t group_id = task_ptr->task_group_.id_;
    ScopedCoRwWriteLock write_lock(grp_container->task_group_lock_);
    auto it = grp_container->task_group_map_.find(group_id);
    if (it == grp_container->task_group_map_.end() || it->second == nullptr) {
      grp_container->task_group_map_[group_id] = selected;
    }
  }

  if (selected != nullptr) {
    return selected->GetId();
  }
  return 0;
}

void DefaultScheduler::RebalanceWorker(Worker *worker) { (void)worker; }

void DefaultScheduler::AdjustPolling(RunContext *run_ctx) {
  if (!run_ctx) {
    return;
  }
  // Adaptive polling disabled for now - restore the true period
  // This is critical because co_await on Futures sets yield_time_us_ = 0,
  // so we must restore it here to prevent periodic tasks from busy-looping
  run_ctx->yield_time_us_ = run_ctx->true_period_ns_ / 1000.0;
}

}  // namespace clio::run
