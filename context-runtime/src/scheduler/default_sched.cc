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
#include <algorithm>

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

  // issue #807: PREFER INLINE EXECUTION on the calling worker for a quick task.
  // When a worker ingests a request off its SHM shard (or routes any cheap
  // task) and is not itself backlogged, run it right here instead of handing it
  // to another worker via a lane + AwakenWorker SIGUSR1. That cross-worker hop
  // is the bulk of the per-request latency vs the original inline design; for a
  // sub-100us task it dominates the actual work. Heavy tasks (high predicted
  // cost) and a backlogged caller fall through to least-loaded routing, so the
  // ingesting worker is never blocked from draining its shard by a big task.
  // The client sharding provides the cross-worker load balancing.
  if (selected == nullptr && worker != nullptr && task_ptr != nullptr &&
      container != nullptr && !kFunnel) {
    constexpr double kInlineCostThresholdUs = 100.0;   // "quick"
    constexpr double kInlineCallerLoadUs = 250.0;      // caller not backlogged
    double now_us = NowUs();
    clio::run::TaskStat stat = container->GetTaskStats(task_ptr);
    double predicted = container->InferCpuTime(task_ptr->method_, stat);
    if (predicted <= kInlineCostThresholdUs &&
        worker->RealtimeLoad(now_us) <= kInlineCallerLoadUs) {
      // Account the predicted cost on the caller so a burst of quick ingests
      // doesn't all decide "I'm idle" simultaneously and pile onto one worker.
      if (predicted > 0.0) {
        task_ptr->SetPredictedLoad(static_cast<float>(predicted));
        task_ptr->SetSchedReservedUs(static_cast<float>(predicted));
        worker->ReserveLoad(predicted);
      }
      return worker->GetId();  // ExecHere -> RouteAndExec runs it inline
    }
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
      // issue #785: bindings are created once and never revisited, and the
      // group lookup runs BEFORE every other routing rule — so a wrong or stale
      // binding silently overrides load, role and health checks for the life of
      // the process. Log each one so the placement is at least observable.
      HLOG(kInfo, "[#785] task group {} bound to worker {} (method={})",
           group_id, selected->GetId(), task_ptr->method_);
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
Worker *DefaultScheduler::PickLeastLoadedLive(double now_us, Worker *avoid_a,
                                              Worker *avoid_b) const {
  // issue #785: least measured-load worker that is alive, has a lane, and is
  // not itself stalled. Used when redistributing a stalled worker's backlog.
  Worker *best = nullptr;
  double best_load = 0.0;
  auto consider = [&](Worker *w) {
    if (w == nullptr || w == avoid_a || w == avoid_b) return;
    if (w->GetLane() == nullptr) return;
    if (w->IsStalled(now_us, kStallThresholdSec)) return;
    double l = w->RealtimeLoad(now_us);
    if (best == nullptr || l < best_load) {
      best = w;
      best_load = l;
    }
  };
  consider(scheduler_worker_);
  for (Worker *w : io_workers_) consider(w);
  for (Worker *w : elastic_workers_) consider(w);
  return best;
}

bool DefaultScheduler::IsWedgedShape(u64 outstanding, u32 live,
                                     u32 live_stalled, double *window_sec) {
  // No work outstanding is not a deadlock, it is an idle runtime.
  if (outstanding == 0) {
    return false;
  }
  // (a) Nothing executing at all: unambiguous — no thread can make progress.
  if (live == 0) {
    if (window_sec) *window_sec = kNoProgressAlarmSec;
    return true;
  }
  // (b) Everything executing is stalled: the lock-cycle shape. CoMutex::Lock()
  // blocks the worker rather than parking the task, so a deadlocked worker is
  // still "executing". Ambiguous against a slow syscall, hence the much longer
  // window.
  if (live_stalled == live) {
    if (window_sec) *window_sec = kAllBlockedAlarmSec;
    return true;
  }
  // Something is running and not stalled: progress is possible.
  return false;
}

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
    HLOG(kWarning, "[#785] lane rescues performed: {}",
         rescues_performed_.load());
  }

  // migratable == may we move this worker's lane/event queue elsewhere?
  //   NO for net_send/net_recv: they own ZMQ sockets, and "ZeroMQ sockets are
  //   not safe to share across threads, so each socket has exactly one owner
  //   thread" (see the routing comment above). Moving their lane would run the
  //   periodic poller — and thus the socket — on a different thread.
  //   NO for the GPU worker: it is bound to its GPU lanes.
  // Non-migratable workers are still detected and warned about; they are simply
  // not rescued.
  auto check = [&](Worker *w, bool migratable) -> bool {
    if (w == nullptr || !w->IsExecuting()) return false;
    if (!w->IsStalled(now_us, kStallThresholdSec)) return false;
    stalls_detected_.fetch_add(1, std::memory_order_relaxed);
    HLOG(kWarning,
         "[#781] worker {} STALLED on one task (load_us={} realtime_load_us={} "
         "threshold_s={})",
         w->GetId(), (double)w->Load(), w->RealtimeLoad(now_us),
         kStallThresholdSec);

    // issue #785: LANE RESCUE. The stalled worker is inside ExecTask and is
    // provably not popping its lane, so its queued backlog is stranded behind a
    // task that may never return. Move the lane to a fresh worker rather than
    // draining a single-consumer MPSC ring from this thread (the "steal-safe
    // pop" problem #781 left open) — a transfer needs no new lane, which
    // matters because lanes are allocated 1:1 with num_threads and there is no
    // spare.
    //
    // Rescue ONCE per stalled worker: a worker with no lane has nothing left to
    // strand, and re-firing every 500ms would spawn a thread per tick for the
    // whole life of the bad task.
    if (!migratable) {
      HLOG(kWarning,
           "[#785] worker {} is NOT migratable (owns a ZMQ socket or GPU "
           "lanes); cannot rescue it — see D7",
           w->GetId());
      return true;
    }
    if (work_orch_ == nullptr || w->GetLane() == nullptr) {
      return true;
    }
    // STEP 1 — drain and redistribute the backlog. This needs NO replacement
    // worker, which is the whole point: when the pool is capped and every
    // replacement is itself wedged, spawning is impossible but re-placing the
    // queued tasks on healthy workers is still both possible and exactly what
    // the small tasks stuck behind heavy ones need. Splitting this from the
    // handover is what makes head-of-line blocking recoverable under
    // saturation rather than only when a spare thread happens to be available.
    //
    // Release first: the donor re-reads assigned_lane_ every Run() iteration,
    // so after this it never pops again and the monitor is the lane's only
    // consumer. That is what makes draining it safe.
    TaskLane *lane = w->ReleaseLane();
    size_t redistributed = 0;
    if (lane != nullptr) {
      Future<Task> queued;
      while (lane->Pop(queued)) {
        Worker *target = PickLeastLoadedLive(now_us, w, nullptr);
        // A target whose lane is (nearly) full is unusable HERE even though it
        // would be fine for a worker thread: TaskLane::Push uses
        // WAIT_FOR_SPACE, so it busy-spins until space appears. On a worker
        // that is back-pressure; on THIS thread it would freeze the monitor
        // loop — no more stall detection, no more rescues, no watchdog. The
        // machinery meant to prevent deadlock would itself deadlock. Leave the
        // remaining tasks in the donor's lane instead; they are handed back to
        // it below and retried on the next tick.
        if (target != nullptr && target->GetLane() != nullptr) {
          TaskLane *tl = target->GetLane();
          size_t depth = tl->GetDepth();
          if (depth > 0 && tl->Size() + 2 >= depth) {
            target = nullptr;  // treat as "nowhere healthy to put it"
          }
        }
        if (target == nullptr || target->GetLane() == nullptr) {
          // Safe to push back: we just popped from this very lane, so it has
          // room and cannot spin.
          if (!lane->Push(queued)) {
            HLOG(kError, "[#785] lost a task re-queuing during rescue");
          }
          break;  // nowhere healthy to put it; leave the rest in place
        }
        // Move the #781 cost reservation with the task, or the donor's
        // queued_load_ never drains and the target under-reports its load.
        Task *tp = queued.get();
        if (tp != nullptr) {
          float resv = tp->SchedReservedUs();
          if (resv != 0.0f) {
            w->ReleaseReservation(resv);
            target->ReserveLoad(resv);
          }
        }
        if (!target->GetLane()->Push(queued)) {
          HLOG(kError, "[#785] failed to re-place task during rescue");
          break;
        }
        CLIO_IPC->AwakenWorker(target->GetLane());
        ++redistributed;
      }
    }

    // Give the lane straight back. Transferring ownership to the replacement
    // was a mistake: after a few rescues the original workers had permanently
    // lost their lanes to elastic ones, so the set of "healthy workers that own
    // a lane" could be EMPTY even with most of the pool idle — and then there
    // was nowhere to redistribute to, which is precisely when it matters.
    //
    // Keeping worker<->lane fixed avoids that entirely. The donor is still
    // wedged so it will not drain the lane, but every LoadBalance tick
    // redistributes whatever has accumulated, bounding the delay by the monitor
    // period rather than by the bad task's runtime. It also removes a whole
    // class of ownership bugs: no orphaned lanes, and RouteTask's
    // resolve-lane-by-worker-id stays consistent.
    w->AdoptLane(lane);

    // A replacement is still wanted for the state that needs a LIVE THREAD to
    // make progress: tasks parked on a co_await are reachable only through the
    // event queue, and the blocked/periodic/retry queues need someone to scan
    // them. Neither can be fixed by re-placing work on another lane.
    Worker *rescuer = work_orch_->FindIdleElasticWorker();
    if (rescuer == nullptr) {
      rescuer = work_orch_->SpawnAdditionalWorker();
    }
    if (rescuer == nullptr) {
      HLOG(kWarning,
           "[#785] worker {} stalled, no replacement available; redistributed "
           "{} queued task(s). Parked tasks stay put until one frees up.",
           w->GetId(), redistributed);
      return true;
    }

    // Transfer the event queue by handing the OBJECT to the rescuer to drain —
    // a parked task's RunContext still points at this exact queue, so both the
    // events already in it and every event a still-running subtask pushes later
    // follow automatically, with no per-task re-pointing.
    // Hand over the queue OBJECT and give the donor a FRESH one, so there is
    // exactly ONE consumer per queue.
    //
    // An earlier version passed w->GetEventQueue() while leaving the donor
    // holding the same pointer. Both then drained it: the rescuer via its
    // adopted list, and the donor via ProcessEventQueue the moment it
    // un-wedged. That is two consumers on a SINGLE-consumer MPSC ring — memory
    // corruption, and timing-sensitive enough to be invisible on this Linux box
    // while segfaulting cr_all_safe_bdev_tests on all three Windows configs.
    // Parked tasks still point at the old object, which the rescuer now solely
    // owns, so wakeups still land correctly.
    rescuer->AdoptEventQueue(w->ReplaceEventQueue());
    // ...and anything this worker had itself inherited from an earlier rescue.
    // Rescues cascade, so without this a replacement that wedges strands every
    // queue it was holding for someone else.
    w->TransferAdoptedEventQueuesTo(rescuer);
    size_t parked_moved = w->MigrateParkedTo(rescuer);

    if (std::find(elastic_workers_.begin(), elastic_workers_.end(), rescuer) ==
        elastic_workers_.end()) {
      elastic_workers_.push_back(rescuer);
    }
    rescues_performed_.fetch_add(1, std::memory_order_relaxed);
    HLOG(kWarning,
         "[#785] RESCUE: stalled worker {} -> worker {} ({} parked task(s) "
         "moved, {} queued task(s) redistributed, rescues={})",
         w->GetId(), rescuer->GetId(), parked_moved, redistributed,
         rescues_performed_.load(std::memory_order_relaxed));
    return true;
  };

  // ---- Progress watchdog (issue #785) ----------------------------------
  // Everything else in this function handles a worker wedged by a task that
  // will eventually return. It is powerless against the classes where the WORK
  // ITSELF cannot proceed — a true dependency cycle (A awaits B, B awaits A), a
  // lock cycle, a lost wakeup — because there is no healthy worker to move the
  // work to. This will not fix those, but it refuses to let them be SILENT,
  // which is the difference between "the cluster hung" and a diagnosable
  // defect.
  //
  // The signature it keys on is deliberately narrow: tasks are outstanding,
  // NOT ONE WORKER IS EXECUTING, and nothing has completed for the alarm
  // window. Everything parked with nothing running is a deadlock. Contrast a
  // merely slow runtime, where workers ARE executing — that is what the stall
  // detector below is for, and conflating the two would make this cry wolf on
  // every long task.
  //
  // Two things it must NOT depend on, both learned the hard way:
  //   - Container::GetWorkRemaining(): most ChiMods do not implement it
  //     (MOD_NAME returns a hardcoded 0), so it reads "no work" during a real
  //     stall and the alarm never arms.
  //   - Worker::TasksProcessed(): periodic pollers complete and re-arm
  //     continuously, so an all-tasks counter never stops advancing. Polling is
  //     not progress; use RealTasksProcessed().
  // work_orch_ is null when LoadBalance is driven directly by a unit test
  // rather than by the WorkOrchestrator's monitor thread. The stall check below
  // already guards this; the watchdog must too, or the no-op path segfaults
  // (caught by clio_run_autogen_coverage_tests' "LoadBalance noop" section).
  if (work_orch_ != nullptr) {
    u64 processed = 0;
    u64 outstanding = 0;
    u32 live = 0;       // workers executing something
    u32 live_stalled = 0;  // ...of which are stuck past the stall threshold
    u64 ob_queued = 0, ob_blocked = 0, ob_retry = 0, ob_periodic = 0;
    for (size_t i = 0; i < work_orch_->GetWorkerCount(); ++i) {
      Worker *w = work_orch_->GetWorker(static_cast<u32>(i));
      if (w == nullptr) continue;
      processed += w->RealTasksProcessed();
      if (w->IsExecuting()) {
        ++live;
        if (w->IsStalled(now_us, kStallThresholdSec)) ++live_stalled;
      }
      WorkerStats st = w->GetWorkerStats();
      ob_queued += st.num_queued_tasks_;
      ob_blocked += st.num_blocked_tasks_;
      ob_retry += st.num_retry_tasks_;
      ob_periodic += st.num_periodic_tasks_;
      outstanding += st.num_queued_tasks_ + st.num_blocked_tasks_ +
                     st.num_retry_tasks_;
    }
    // Two shapes mean "no progress is possible", and BOTH must be covered:
    //
    //  (a) nothing is executing at all — everything is parked waiting on
    //      something that is not coming (dependency cycle, lost wakeup).
    //
    //  (b) everything that IS executing is stalled — the lock-cycle case.
    //      CoMutex::Lock() calls ctp::Mutex::Lock(), a BLOCKING mutex rather
    //      than a coroutine-aware park, so two tasks deadlocked on an A/B-B/A
    //      lock order sit *inside* ExecTask with IsExecuting() true. Keying
    //      only on (a) would have missed the commonest real deadlock entirely,
    //      while this alarm claimed to cover it.
    // The two shapes are NOT equally diagnostic, so they get different
    // windows:
    //   (a) nothing executing at all is unambiguous — no thread can make
    //       progress, period. Alarm after the short window.
    //   (b) everything executing is blocked is AMBIGUOUS: a worker deadlocked
    //       on a lock is indistinguishable from one inside a slow syscall that
    //       will return. Alarming on that quickly would fire on every long
    //       blocking task, and an alarm operators learn to ignore is worse than
    //       none. So it needs a much longer window, after which deadlock is far
    //       likelier than slowness.
    double window_sec = kNoProgressAlarmSec;
    const bool wedged_shape =
        IsWedgedShape(outstanding, live, live_stalled, &window_sec);
    if (!wedged_shape || processed != last_progress_count_) {
      last_progress_count_ = processed;
      last_progress_us_ = now_us;
      progress_alarm_raised_ = false;
    } else if (!progress_alarm_raised_ &&
               (now_us - last_progress_us_) > window_sec * 1e6) {
      progress_alarm_raised_ = true;  // once per stall, not once per tick
      HLOG(kError,
           "[#785] DEADLOCK SUSPECTED: {} task(s) outstanding, {} worker(s) "
           "executing ({} of them stalled), and no non-periodic task completed "
           "in {}s. Migration cannot resolve this shape — either everything is "
           "parked waiting on something that is not coming (dependency cycle, "
           "lost wakeup), or every running worker is blocked (lock cycle). "
           "Worker state:",
           outstanding, live, live_stalled, window_sec);
      for (size_t i = 0; i < work_orch_->GetWorkerCount(); ++i) {
        Worker *w = work_orch_->GetWorker(static_cast<u32>(i));
        if (w == nullptr) continue;
        WorkerStats st = w->GetWorkerStats();
        HLOG(kError,
             "[#785]   worker {}: done={} queued={} blocked={} periodic={} "
             "retry={}",
             w->GetId(), st.num_tasks_processed_, st.num_queued_tasks_,
             st.num_blocked_tasks_, st.num_periodic_tasks_,
             st.num_retry_tasks_);
      }
    }

    // [HANGWATCH] Pure no-forward-progress watchdog (diagnostic). The wedged-
    // shape alarm above only fires when live==0 or EVERY live worker is stalled-
    // in-one-task; it misses the busy-spin / lost-completion class (workers look
    // "live" and are not counted as stalled) that is exactly the embedded-FUSE
    // 2-CPU hang (generic/020). This fires purely on "no real task completed for
    // a while while work is outstanding" and dumps race-safe per-worker state
    // (stats snapshot + atomic reads only; no GetCurrentTask() shared_ptr race)
    // so a CI hang is diagnosable from the log. Runs on the monitor thread,
    // which the spinning workers cannot fully starve.
    {
      static u64 wd_last_processed = ~0ull;
      static double wd_last_us = 0.0;
      static bool wd_alarmed = false;
      // Heartbeat every ~2s while work is outstanding: a CI hang log then shows
      // whether `processed` is advancing (watchdog resets, never alarms) or truly
      // frozen (should alarm). This is the signal that tells us WHY the 12s dump
      // did or did not fire.
      static u64 hb_tick = 0;
      if (outstanding > 0 && (++hb_tick % 4 == 0)) {
        HLOG(kError,
             "[HANGWATCH-HB] processed={} outstanding={} (queued={} blocked={} "
             "retry={} periodic={}) live={} live_stalled={}",
             processed, outstanding, ob_queued, ob_blocked, ob_retry,
             ob_periodic, live, live_stalled);
      }
      if (wd_last_processed == ~0ull) {
        wd_last_processed = processed;
        wd_last_us = now_us;
      }
      if (processed != wd_last_processed || outstanding == 0) {
        wd_last_processed = processed;
        wd_last_us = now_us;
        wd_alarmed = false;
      } else if (!wd_alarmed && outstanding > 0 &&
                 (now_us - wd_last_us) > 12.0 * 1e6) {
        wd_alarmed = true;
        HLOG(kError,
             "[HANGWATCH] no task completed in 12s: outstanding={} processed={} "
             "live={} live_stalled={}. Per-worker:",
             outstanding, processed, live, live_stalled);
        for (size_t i = 0; i < work_orch_->GetWorkerCount(); ++i) {
          Worker *w = work_orch_->GetWorker(static_cast<u32>(i));
          if (w == nullptr) continue;
          TaskLane *ln = w->GetLane();
          WorkerStats st = w->GetWorkerStats();
          HLOG(kError,
               "[HANGWATCH]  w{} exec={} stalled={} lane_size={} queued={} "
               "blocked={} periodic={} retry={} done={}",
               w->GetId(), w->IsExecuting(),
               w->IsStalled(now_us, kStallThresholdSec),
               ln ? static_cast<long>(ln->Size()) : -1L,
               st.num_queued_tasks_, st.num_blocked_tasks_,
               st.num_periodic_tasks_, st.num_retry_tasks_,
               st.num_tasks_processed_);
        }
      }
    }
  }

  u32 stalled = 0;
  if (check(scheduler_worker_, /*migratable=*/true)) ++stalled;
  for (Worker *w : io_workers_) if (check(w, /*migratable=*/true)) ++stalled;
  // Elastic workers spawned by earlier rescues. Iterated by index because
  // check() may append to elastic_workers_ (a cascading rescue), which would
  // invalidate an iterator mid-loop.
  for (size_t i = 0; i < elastic_workers_.size(); ++i) {
    if (check(elastic_workers_[i], /*migratable=*/true)) ++stalled;
  }
  if (check(net_send_worker_, /*migratable=*/false)) ++stalled;
  if (check(net_recv_worker_, /*migratable=*/false)) ++stalled;

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
