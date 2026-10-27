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
#include <cmath>
#include <cstdio>
#include <string>
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
/**
 * Whether the submitting worker can run a task inline: the task runs nested,
 * before the worker returns to its queue, so what it delays is the work
 * QUEUED behind the worker -- not the task currently executing, which is the
 * submitter itself and is waiting for this one. Counting the submitter's own
 * hold made every worker look full to its own subtasks, so nothing ran
 * inline and every step of a chain hopped threads.
 * @param w the submitting worker
 * @param hold_us the new task's predicted hold (us)
 * @param now_us current time (us)
 * @param max_us the most work a worker may hold (us)
 * @return true if the queued work plus this task stays within max_us
 */
static bool CallerHasRoom(const clio::run::Worker *w, double hold_us,
                          double now_us, double max_us) {
  double queued = w->RealtimeLoad(now_us) - static_cast<double>(w->Load()) -
                  w->CurrentTaskAgeMs(now_us) * 1000.0;
  if (!(queued > 0.0)) queued = 0.0;
  // No exception for an empty queue: the submitter also drains its client
  // shard, which this count does not see, so a long task run inline here
  // stalls every request behind it. Longer tasks go to another worker.
  return queued + hold_us <= max_us;
}

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

  // The pool: worker 0 plus the workers between it and the GPU/net block.
  // GrowPool appends to it when every member is holding a full share.
  u32 pool_upper_excl = total_workers >= 5 ? total_workers - 3
                      : total_workers >= 3 ? total_workers - 2
                                           : 1;
  {
    std::lock_guard<std::mutex> g(pool_mu_);
    pool_.clear();
    if (scheduler_worker_ != nullptr) pool_.push_back(scheduler_worker_);
    for (u32 i = 1; i < pool_upper_excl; ++i) {
      Worker *worker = work_orch->GetWorker(i);
      if (worker) pool_.push_back(worker);
    }
    HLOG(kInfo,
         "DefaultScheduler: pool of {} workers, each holding <= {}us of "
         "work; grows when none has room",
         pool_.size(), kMaxWorkerHoldUs);
  }

  // The net workers' lanes are not assigned yet; WorkOrchestrator publishes
  // them to the IPC manager after lane mapping (issue #768).
  IpcManager *ipc = CLIO_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(1);
  }

  int send_id = net_send_worker_ ? (int)net_send_worker_->GetId() : -1;
  int recv_id = net_recv_worker_ ? (int)net_recv_worker_->GetId() : -1;
  HLOG(kInfo,
       "DefaultScheduler: scheduler worker 0, pool of {}, "
       "net_send_worker={}, net_recv_worker={}, gpu_worker={}",
       PoolSnapshot().size(), send_id, recv_id,
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

  // A/B toggle (issue #781 evaluation): CLIO_SCHED_FUNNEL=1 restores the legacy
  // "everything to the scheduler worker" routing so the sched_variety benchmark
  // can measure quick-task p99 with vs without measured-load routing. Value-based
  // (must be "1"/"true") - an unset OR empty/0 value means measured-load routing.
  static const bool kFunnel = []() {
    const char *v = std::getenv("CLIO_SCHED_FUNNEL");
    return v != nullptr && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
  }();
  if (selected == nullptr && kFunnel && scheduler_worker_ != nullptr) {
    selected = scheduler_worker_;
  }

  // Hold-time routing: book the task as the time it will occupy a worker,
  // and keep every worker at or under kMaxWorkerHoldUs.
  if (selected == nullptr && task_ptr != nullptr) {
    const double now_us = NowUs();
    const double hold_us =
        container != nullptr ? PredictHold(task_ptr, *container) : 0.0;
    if (worker != nullptr && InPool(worker) &&
        CallerHasRoom(worker, hold_us, now_us, kMaxWorkerHoldUs)) {
      // The submitter has room: run it here, with no lane hop or wakeup.
      selected = worker;
      map_here_.fetch_add(1, std::memory_order_relaxed);
    } else {
      selected = PickPoolWorker(now_us);
      if (selected != nullptr && !HasRoom(selected, hold_us, now_us)) {
        // Even the least-loaded worker would exceed its share. Grow the pool
        // (the monitor thread spawns at once); this task takes the
        // least-loaded worker meanwhile.
        map_full_.fetch_add(1, std::memory_order_relaxed);
        if (work_orch_ != nullptr) work_orch_->RequestWorker();
      } else {
        map_other_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (selected != nullptr && hold_us > 0.0 && !task_ptr->IsStarted()) {
      // Book the hold on the chosen worker until the task starts (ExecTask
      // moves it from the reservation into load_). A re-routed task that has
      // already started is counted by BookHold when it resumes; ExecTask
      // only releases reservations at first execution.
      Worker::ReserveOn(task_ptr, selected, hold_us);
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

bool DefaultScheduler::HasRoom(const Worker *w, double hold_us,
                               double now_us) {
  const double load = w->RealtimeLoad(now_us);
  return load <= 0.0 || load + hold_us <= kMaxWorkerHoldUs;
}

double DefaultScheduler::PredictHold(Task *task, Container &container) {
  const TaskStat stat = container.GetTaskStats(task);
  const float cpu = container.InferCpuTime(task->method_, stat);
  const float wall = container.InferWallClockTime(task->method_, stat);
  task->SetPredictedLoad(cpu);
  task->SetPredictedWallUs(wall);
  // The wall model alone: it is trained on the time the task occupied its
  // worker. Thread CPU time is too coarse to book (~15.6 ms ticks on Windows).
  const double hold = static_cast<double>(wall);
  // Inference already returns finite, non-negative values; cap what one task
  // can book so a mispredicted giant cannot pin a worker as full forever.
  return std::isfinite(hold) && hold > 0.0 ? std::min(hold, kMaxBookedHoldUs)
                                           : 0.0;
}

Worker *DefaultScheduler::PickPoolWorker(double now_us) const {
  std::lock_guard<std::mutex> g(pool_mu_);
  const size_t n = pool_.size();
  if (n == 0) return nullptr;
  const size_t start = next_io_idx_.fetch_add(1, std::memory_order_relaxed) % n;
  Worker *best = nullptr;
  double best_load = 0.0;
  for (size_t k = 0; k < n; ++k) {
    Worker *w = pool_[(start + k) % n];
    if (w == nullptr) continue;
    const double load = w->RealtimeLoad(now_us);
    if (best == nullptr || load < best_load) {
      best = w;
      best_load = load;
    }
  }
  return best;
}

bool DefaultScheduler::InPool(const Worker *w) const {
  std::lock_guard<std::mutex> g(pool_mu_);
  return std::find(pool_.begin(), pool_.end(), w) != pool_.end();
}

std::vector<Worker *> DefaultScheduler::PoolSnapshot() const {
  std::lock_guard<std::mutex> g(pool_mu_);
  return pool_;
}

void DefaultScheduler::GrowPool() {
  if (work_orch_ == nullptr || spawn_exhausted_) return;
  const double now_us = NowUs();
  std::vector<Worker *> pool = PoolSnapshot();
  // The request may be stale: a worker that has drained back under its share
  // since then serves new traffic without another thread.
  for (Worker *w : pool) {
    if (w != nullptr && w->RealtimeLoad(now_us) < kMaxWorkerHoldUs) return;
  }
  // Reuse an idle spare (one not in the pool, and not a #785 rescuer), else
  // spawn. SpawnAdditionalWorker enforces the thread budget.
  std::vector<Worker *> taken = pool;
  taken.insert(taken.end(), elastic_workers_.begin(), elastic_workers_.end());
  Worker *w = work_orch_->FindIdleElasticWorker(taken);
  if (w == nullptr) w = work_orch_->SpawnAdditionalWorker();
  if (w == nullptr) {
    // The thread budget is spent (SpawnAdditionalWorker said why, once).
    // Stop asking: every later request would fail the same way.
    spawn_exhausted_ = true;
    HLOG(kWarning, "[sched] pool at its thread budget ({} workers); "
         "workers will now hold more than {}us each", pool.size(),
         kMaxWorkerHoldUs);
    return;
  }
  size_t n;
  {
    std::lock_guard<std::mutex> g(pool_mu_);
    pool_.push_back(w);
    n = pool_.size();
  }
  pool_grows_.fetch_add(1, std::memory_order_relaxed);
  HLOG(kInfo, "[sched] every worker holds > {}us: added worker {} (pool {})",
       kMaxWorkerHoldUs, w->GetId(), n);
}

Worker *DefaultScheduler::PickAltWorker(u32 avoid_id) const {
  // Any pool worker, then the net workers: any worker whose Run loop drains
  // its own lane breaks the self-block (the caller is no longer the consumer
  // of the lane it is pushing to).
  {
    std::lock_guard<std::mutex> g(pool_mu_);
    for (Worker *w : pool_) {
      if (w != nullptr && w->GetId() != avoid_id) {
        return w;
      }
    }
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
  for (Worker *w : PoolSnapshot()) consider(w);
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
    HLOG(kDebug,
         "[#781 PDF] observed exec-time bins (cumulative): "
         "<10us={} <50us={} <500us={} <10ms={} <50ms={} <500ms={} <1s={} "
         ">=1s={} | stalls_detected={}",
         perf_pdf_[kLt10us].load(), perf_pdf_[kLt50us].load(),
         perf_pdf_[kLt500us].load(), perf_pdf_[kLt10ms].load(),
         perf_pdf_[kLt50ms].load(), perf_pdf_[kLt500ms].load(),
         perf_pdf_[kLt1s].load(), perf_pdf_[kGe1s].load(),
         stalls_detected_.load());
    // WHICH tasks own the slow bins: pool.method, count, summed and worst
    // wall time in ms, the eight largest by summed time.
    HLOG(kDebug, "[#781 METHODS] wall time by pool.method (top 8):{}",
         MethodTimeReport(8));
    HLOG(kDebug, "[#785] lane rescues performed: {}",
         rescues_performed_.load());
    HLOG(kDebug,
         "[sched] pool: {} workers ({} added by GrowPool); placed here={} "
         "other={} full={}",
         PoolSnapshot().size(), pool_grows_.load(), map_here_.load(),
         map_other_.load(), map_full_.load());
    // Per-worker: executing hold (load_), queued reservations, elapsed.
    std::string per;
    for (Worker *w : PoolSnapshot()) {
      if (w == nullptr) continue;
      const double age_us = w->CurrentTaskAgeMs(now_us) * 1000.0;
      const double queued =
          w->RealtimeLoad(now_us) - static_cast<double>(w->Load()) - age_us;
      per += " w" + std::to_string(w->GetId()) + "=" +
             std::to_string(static_cast<long long>(w->Load())) + "/" +
             std::to_string(static_cast<long long>(queued)) + "/" +
             std::to_string(static_cast<long long>(age_us)) + "/lane=" +
             std::to_string(w->GetLane() != nullptr
                                ? static_cast<long long>(w->GetLane()->Size())
                                : -1LL) +
             (w->IsExecuting() ? "/exec" : "");
    }
    HLOG(kDebug, "[sched] load/queued/elapsed us:{}", per);
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
    // Name any pool creates in flight (#1180): a stalled admin
    // GetOrCreatePool otherwise says nothing about which pool it is creating.
    PoolManager *pool_manager = CLIO_POOL_MANAGER;
    const std::string creates =
        pool_manager != nullptr ? pool_manager->DescribeCreatesInProgress()
                                : std::string();
    HLOG(kWarning,
         "[#781] worker {} STALLED on one task (pool {} method {} running "
         "{} ms; load_us={} realtime_load_us={} threshold_s={}){}{}",
         w->GetId(), w->CurrentPoolMajor(), w->CurrentMethod(),
         w->CurrentTaskAgeMs(now_us), (double)w->Load(),
         w->RealtimeLoad(now_us), kStallThresholdSec,
         creates.empty() ? "" : "; pool creates in progress: ", creates);

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
            Worker::ReserveOn(tp, target, resv);
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
    Worker *rescuer = work_orch_->FindIdleElasticWorker(PoolSnapshot());
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
    // Arm the window on the first tick. last_progress_us_ starts at 0.0, so a
    // process whose FIRST LoadBalance tick already sees a wedged shape with
    // nothing completed (processed == last_progress_count_ == 0, the normal
    // state a second into bring-up) fell straight into the else branch and
    // compared now_us against the epoch -- always greater than any window. The
    // alarm then fired immediately and reported "no non-periodic task completed
    // in 10s" about one second into the process's life. Same sentinel idiom as
    // the HANGWATCH watchdog below.
    if (last_progress_us_ == 0.0) {
      last_progress_us_ = now_us;
    }
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
        HLOG(kDebug,
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
  const std::vector<Worker *> pool = PoolSnapshot();
  for (Worker *w : pool) if (check(w, /*migratable=*/true)) ++stalled;
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
  const u32 compute_workers = static_cast<u32>(pool.size());
  if (compute_workers > 0 && stalled >= compute_workers) {
    HLOG(kWarning,
         "[#781] ALL {} compute workers stalled — runtime cannot place new "
         "tasks; elastic SpawnAdditionalWorker() needed (follow-up)",
         compute_workers);
    // TODO(#781): work_orch_->SpawnAdditionalWorker() + steal a stalled lane.
  }
}

void DefaultScheduler::RecordCompletion(u32 pool_major, u32 method,
                                        double cpu_us, double wall_us) {
  // Telemetry only (issue #781): bin the measured wall time. This is the
  // runtime OBSERVING what it actually ran — it is not used for routing.
  (void)cpu_us;
  perf_pdf_[BinFor(wall_us)].fetch_add(1, std::memory_order_relaxed);
  // Fold the wall time into the per-(pool, method) table: hash the key,
  // claim or match a slot within a short probe run, and give up silently if
  // the run is full (telemetry, never a reason to stall a worker).
  const u64 key =
      ((static_cast<u64>(pool_major) << 16) | (method & 0xffffu)) + 1;
  const size_t start =
      static_cast<size_t>((key * 0x9e3779b97f4a7c15ULL) >> 54);  // 10 bits
  const u64 ns = wall_us > 0.0 ? static_cast<u64>(wall_us * 1000.0) : 0;
  for (size_t probe = 0; probe < 8; ++probe) {
    MethodTime &mt = method_times_[(start + probe) & (kMethodTimeSlots - 1)];
    u64 have = mt.key.load(std::memory_order_acquire);
    if (have == 0) {
      u64 expected = 0;
      if (mt.key.compare_exchange_strong(expected, key,
                                         std::memory_order_acq_rel)) {
        have = key;
      } else {
        have = expected;
      }
    }
    if (have != key) continue;
    mt.count.fetch_add(1, std::memory_order_relaxed);
    mt.wall_ns.fetch_add(ns, std::memory_order_relaxed);
    u64 cur = mt.max_ns.load(std::memory_order_relaxed);
    while (ns > cur && !mt.max_ns.compare_exchange_weak(
                           cur, ns, std::memory_order_relaxed)) {
    }
    return;
  }
}

std::string DefaultScheduler::MethodTimeReport(size_t top_n) const {
  struct Row {
    u64 key, count, wall_ns, max_ns;
  };
  std::vector<Row> rows;
  for (const MethodTime &mt : method_times_) {
    const u64 key = mt.key.load(std::memory_order_acquire);
    if (key == 0) continue;
    rows.push_back({key - 1, mt.count.load(std::memory_order_relaxed),
                    mt.wall_ns.load(std::memory_order_relaxed),
                    mt.max_ns.load(std::memory_order_relaxed)});
  }
  std::sort(rows.begin(), rows.end(),
            [](const Row &a, const Row &b) { return a.wall_ns > b.wall_ns; });
  if (rows.size() > top_n) rows.resize(top_n);
  std::string out;
  char buf[160];
  for (const Row &r : rows) {
    const double sum_ms = static_cast<double>(r.wall_ns) / 1e6;
    const double avg_ms =
        r.count ? sum_ms / static_cast<double>(r.count) : 0.0;
    std::snprintf(buf, sizeof(buf),
                  " %llu.%llu n=%llu sum_ms=%.1f avg_ms=%.3f max_ms=%.1f |",
                  static_cast<unsigned long long>(r.key >> 16),
                  static_cast<unsigned long long>(r.key & 0xffffu),
                  static_cast<unsigned long long>(r.count), sum_ms, avg_ms,
                  static_cast<double>(r.max_ns) / 1e6);
    out += buf;
  }
  return out;
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
