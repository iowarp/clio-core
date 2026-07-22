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

#include <chrono>
#include <cstdlib>

#include "clio_runtime/config_manager.h"
#include "clio_runtime/container.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/work_orchestrator.h"
#include "clio_runtime/worker.h"

namespace clio::run {

// issue #781: steady-clock "now" in microseconds, matching the clock
// Worker::ExecTask stamps into last_exec_start_us_ so RealtimeLoad/IsStalled
// compare against the same timebase.
static inline double NowUs() {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void DefaultScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) {
    return;
  }
  work_orch_ = work_orch;  // #781: retained for elastic SpawnAdditionalWorker

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

// NOTE (issue #781): ClientMapTask has been removed. Client/recv-thread ingress
// no longer picks a worker lane directly — tasks are mapped to workers ONLY in
// the runtime (RuntimeMapTask via RouteTask). The recv threads now deposit onto
// the shared ingress lane and the runtime places the task.

u32 DefaultScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task,
                                      ContainerHold container) {
  Task *task_ptr = task.get();

  // ---- Task group affinity: return early if group already pinned ----
  // Use the caller-supplied container directly (no static container lookup).
  const bool has_group =
      (container != nullptr && task_ptr != nullptr &&
       !task_ptr->task_group_.IsNull());
  if (has_group) {
    int64_t group_id = task_ptr->task_group_.id_;
    ScopedCoRwReadLock read_lock(container->task_group_lock_);
    auto it = container->task_group_map_.find(group_id);
    if (it != container->task_group_map_.end() && it->second != nullptr) {
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
    if (task_ptr->pool_id_ == clio::run::kAdminPoolId) {
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
    size_t io_size = task_ptr->PredictedStat().io_size_;
    if (io_size >= kLargeIOThreshold) {
      u32 idx = next_io_idx_.fetch_add(1, std::memory_order_relaxed) %
                static_cast<u32>(io_workers_.size());
      selected = io_workers_[idx];
    }
  }

  // issue #781: measured-load routing for compute / small-I/O / metadata tasks.
  // Instead of funnelling everything onto the scheduler worker (where one
  // non-yielding/mislabeled task starves every task behind it), pick the
  // general-purpose worker with the SMALLEST measured real-time load. A worker
  // stuck spinning on a heavy task has an enormous RealtimeLoad (load_ + how
  // long its current task has already run), so new quick tasks are steered away
  // from it automatically — no cost prediction, no labels. This is the primary
  // anti-starvation mechanism; LoadBalance() handles any already-queued backlog.
  // A/B toggle (issue #781 evaluation): CLIO_SCHED_FUNNEL=1 restores the legacy
  // "everything to the scheduler worker" routing so the sched_variety benchmark
  // can measure quick-task p99 with vs without measured-load routing. Value-based
  // (must be "1"/"true") — an unset OR empty/0 value means measured-load routing.
  static const bool kFunnel = []() {
    const char *v = std::getenv("CLIO_SCHED_FUNNEL");
    return v != nullptr && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
  }();
  if (selected == nullptr && kFunnel && scheduler_worker_ != nullptr) {
    selected = scheduler_worker_;
  }

  if (selected == nullptr) {
    // Build the compute candidate pool. Prefer the dedicated I/O workers so the
    // scheduler worker (lane 0) stays a pure INGRESS DISPATCHER — a heavy task
    // executing there would stall the dispatch of every incoming task. Only fall
    // back to the scheduler worker when there are no I/O workers.
    Worker *cand[64];
    u32 n = 0;
    for (Worker *w : io_workers_) {
      if (w != nullptr && n < 64) cand[n++] = w;
    }
    if (n == 0 && scheduler_worker_ != nullptr) cand[n++] = scheduler_worker_;
    if (n > 0) {
      double now_us = NowUs();
      // Scan from a rotating offset so that when loads tie (e.g. all idle at 0)
      // selection round-robins instead of always funnelling to the first worker.
      u32 start = next_io_idx_.fetch_add(1, std::memory_order_relaxed) % n;
      double best_load = 0.0;
      for (u32 k = 0; k < n; ++k) {
        Worker *w = cand[(start + k) % n];
        double rl = w->RealtimeLoad(now_us);
        if (selected == nullptr || rl < best_load) {
          selected = w;
          best_load = rl;
        }
      }
      // issue #781: reserve the incoming task's PREDICTED cost on the chosen
      // worker so the NEXT mapping sees it as queued load — the quick-behind-
      // heavy fix. The prediction comes from the learned per-method model via
      // the populated PredictedStat (MOD_NAME::GetTaskStats reports the compute
      // counter); it converges to ~the real cost after a few runs. Released in
      // Worker::ExecTask when the task actually starts.
      if (selected != nullptr && task_ptr != nullptr && container != nullptr) {
        // Read the compute counter FRESH here: RunContext::predicted_stat_ is
        // only populated at EXECUTION time (BeginTask on the destination worker),
        // so it is still empty during routing. GetTaskStats is cheap + const.
        clio::run::TaskStat stat = container->GetTaskStats(task_ptr);
        double predicted = container->InferCpuTime(task_ptr->method_, stat);
        if (predicted > 0.0) {
          // Set PredictedLoad too so ExecTask/EndTask account the SAME value
          // (its own PredictedStat() lookup would still read 0 during the window
          // before BeginTask). sched_reserved_us_ marks it as map-accounted.
          task_ptr->SetPredictedLoad(static_cast<float>(predicted));
          task_ptr->SetSchedReservedUs(static_cast<float>(predicted));
          selected->ReserveLoad(predicted);
        }
      }
    }
  }

  // Fallback to current worker
  if (selected == nullptr) {
    selected = worker;
  }

  // ---- Update group map after routing decision ----
  if (has_group && task_ptr != nullptr && selected != nullptr) {
    int64_t group_id = task_ptr->task_group_.id_;
    ScopedCoRwWriteLock write_lock(container->task_group_lock_);
    auto it = container->task_group_map_.find(group_id);
    if (it == container->task_group_map_.end() || it->second == nullptr) {
      container->task_group_map_[group_id] = selected;
    }
  }

  if (selected != nullptr) {
    return selected->GetId();
  }
  return 0;
}

Worker *DefaultScheduler::PickAltWorker(u32 avoid_id) const {
  // Prefer an I/O worker: it drains bdev leaf tasks (which do not self-send),
  // so it makes forward progress and never mutually back-pressures the caller.
  for (Worker *w : io_workers_) {
    if (w != nullptr && w->GetId() != avoid_id) {
      return w;
    }
  }
  // Fall back to the scheduler worker, then the net workers — any worker whose
  // Run loop drains its own lane breaks the self-block (the caller is no longer
  // the consumer of the lane it is pushing to).
  if (scheduler_worker_ != nullptr && scheduler_worker_->GetId() != avoid_id) {
    return scheduler_worker_;
  }
  if (net_recv_worker_ != nullptr && net_recv_worker_->GetId() != avoid_id) {
    return net_recv_worker_;
  }
  if (net_send_worker_ != nullptr && net_send_worker_->GetId() != avoid_id) {
    return net_send_worker_;
  }
  return nullptr;
}

// issue #781 — SCAFFOLD ONLY (not yet wired). Called by the WorkOrchestrator
// monitor thread every ~500ms. Real implementation will:
//   1. detect a worker stuck > kStallThresholdSec inside one ExecTask
//      (worker->IsStalled()) and, if so, work_orch_->SpawnAdditionalWorker()
//      + StealAll(from stalled lane -> new worker) so the pathological task
//      strands one thread, never the runtime;
//   2. update perf_pdf_ from recently-completed exec times (telemetry);
//   3. retire elastic workers idle > kRetireCooldownSec (hysteresis);
//   4. emit sched.threads.* metrics; WARN when live threads are abnormal.
void DefaultScheduler::LoadBalance() {
  // Runs on the WorkOrchestrator monitor thread every ~500ms (NOT a worker).
  // This pass implements the OBSERVABILITY half of the safety net: detect a
  // worker stuck > kStallThresholdSec inside one ExecTask (a non-yielding /
  // mislabeled task) and warn loudly with live/stalled counts. Because
  // RuntimeMapTask already steers NEW tasks away from a stalled worker (its
  // RealtimeLoad is enormous), the runtime keeps making progress on quick tasks
  // even while a worker is wedged. The backlog-rescue (spawn a replacement +
  // steal the stalled lane) needs a steal-safe MPSC pop and lands in a
  // follow-up — see project_scheduler_antideadlock.md.
  u64 tick = load_balance_ticks_.fetch_add(1, std::memory_order_relaxed) + 1;
  double now_us = NowUs();

  // Telemetry dump every ~5s: the perf-bin PDF is the runtime's OBSERVED live
  // workload distribution (how many quick vs 1-second tasks it actually ran).
  // It adapts continuously as RecordCompletion folds in each finished task.
  if (tick % 10 == 0) {
    HLOG(kWarning,
         "[#781 PDF] observed exec-time bins (cumulative): "
         "<10us={} <50us={} <500us={} <10ms={} <50ms={} <500ms={} <1s={} "
         ">=1s={} | stalls_detected={}",
         perf_pdf_[kLt10us].load(), perf_pdf_[kLt50us].load(),
         perf_pdf_[kLt500us].load(), perf_pdf_[kLt10ms].load(),
         perf_pdf_[kLt50ms].load(), perf_pdf_[kLt500ms].load(),
         perf_pdf_[kLt1s].load(), perf_pdf_[kGe1s].load(),
         stalls_detected_.load());
  }

  auto check = [&](Worker *w) -> bool {
    if (w == nullptr || !w->IsExecuting()) return false;
    if (!w->IsStalled(now_us, kStallThresholdSec)) return false;
    stalls_detected_.fetch_add(1, std::memory_order_relaxed);
    HLOG(kWarning,
         "[#781] worker {} STALLED on one task (load_us={} realtime_load_us={} "
         "threshold_s={}) — routing new work around it; backlog awaits rescue",
         w->GetId(), (double)w->Load(), w->RealtimeLoad(now_us),
         kStallThresholdSec);
    return true;
  };

  u32 stalled = 0;
  if (check(scheduler_worker_)) ++stalled;
  for (Worker *w : io_workers_) if (check(w)) ++stalled;
  if (check(net_send_worker_)) ++stalled;
  if (check(net_recv_worker_)) ++stalled;

  // Observability for the unbounded-pool design (#781): if EVERY general-purpose
  // worker is stalled, no routing target can make progress — this is exactly the
  // "flood of non-yielding tasks" case where the follow-up must SpawnAdditional-
  // Worker(). Make it loud rather than silently wedging.
  u32 compute_workers = (scheduler_worker_ ? 1u : 0u) +
                        static_cast<u32>(io_workers_.size());
  if (compute_workers > 0 && stalled >= compute_workers) {
    HLOG(kWarning,
         "[#781] ALL {} compute workers stalled — runtime cannot place new "
         "tasks; elastic SpawnAdditionalWorker() needed (follow-up)",
         compute_workers);
    // TODO(#781): work_orch_->SpawnAdditionalWorker() + steal a stalled lane.
  }
}

void DefaultScheduler::RecordCompletion(u32 method, double cpu_us,
                                        double wall_us) {
  // Telemetry only (issue #781): bin the measured wall time. This is the
  // runtime OBSERVING what it actually ran — it is not used for routing.
  (void)method;
  (void)cpu_us;
  perf_pdf_[BinFor(wall_us)].fetch_add(1, std::memory_order_relaxed);
}

// issue #781 — SCAFFOLD ONLY (not yet wired). Move up to kStealBatch tasks from
// each ring neighbour (then the most-loaded worker) onto thief's lane using a
// steal-safe lane pop, keeping the pool work-conserving.
bool DefaultScheduler::StealWork(Worker *thief) {
  (void)thief;
  // TODO(#781): neighbour + most-loaded steal.
  return false;
}

void DefaultScheduler::AdjustPolling(const clio::run::shared_ptr<Task> &task) {
  if (task.IsNull()) {
    return;
  }
  // Adaptive polling disabled for now - restore the true period
  // This is critical because co_await on Futures sets yield_time_us_ = 0,
  // so we must restore it here to prevent periodic tasks from busy-looping
  task->SetYieldTimeUs(task->TruePeriodNs() / 1000.0);
}

}  // namespace clio::run
