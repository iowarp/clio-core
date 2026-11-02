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
#ifndef CLIO_RUNTIME_INCLUDE_SCHEDULER_DEFAULT_SCHED_H_
#define CLIO_RUNTIME_INCLUDE_SCHEDULER_DEFAULT_SCHED_H_

#include <atomic>
#include <vector>

#include "clio_runtime/scheduler/scheduler.h"

#include <array>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace clio::run {

class Container;

/**
 * Default scheduler: one pool of general workers, each kept to at most
 * kMaxWorkerHoldUs of work.
 *
 * Every task is booked as the time it is predicted to HOLD a worker: the
 * method's learned wall-time model, which sees the task's I/O size and is
 * trained on the time the task really occupied its worker. A task runs on the worker that submitted it when that
 * worker has room for it, else on the least-loaded pool worker; when no
 * worker has room the pool grows by one (on the monitor thread, at once).
 * There are no I/O, compute or cost-class partitions: the hold estimate is
 * the only input.
 *
 * Outside the pool: two network workers -- one owns ROUTER recvs
 * (kRecv / kClientRecv), the other DEALER sends (kSend / kClientSend), since a
 * ZeroMQ socket has exactly one owner thread -- and the GPU polling worker.
 */
class DefaultScheduler : public Scheduler {
 public:
  DefaultScheduler()
      : scheduler_worker_(nullptr), net_send_worker_(nullptr),
        net_recv_worker_(nullptr), gpu_worker_(nullptr), next_io_idx_{0} {}
  ~DefaultScheduler() override = default;

  void DivideWorkers(WorkOrchestrator *work_orch) override;
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task,
                     ContainerHold container) override;
  void LoadBalance() override;

  /**
   * issue #785: the progress watchdog's decision, as a pure function so it can
   * be unit-tested without constructing a real deadlock.
   *
   * A live deadlock test would wedge the runtime permanently — the tasks never
   * complete, teardown never finishes, and CI hangs rather than fails. Testing
   * the predicate directly gets the coverage without that risk.
   *
   * @param outstanding queued + blocked + retry tasks across all workers.
   * @param live workers currently inside ExecTask.
   * @param live_stalled how many of those are past the stall threshold.
   * @param[out] window_sec how long to wait before believing it (the
   *   "everything blocked" shape is ambiguous and needs far longer).
   * @return true if this shape means no progress is possible.
   */
  static bool IsWedgedShape(u64 outstanding, u32 live, u32 live_stalled,
                            double *window_sec);         // issue #781 — safety net, every 500ms
  bool StealWork(Worker *thief) override;  // issue #781 — work-conserving steal
  void RecordCompletion(u32 pool_major, u32 method, double cpu_us,
                        double wall_us) override;

  /**
   * Per-(pool, method) wall-time totals behind the PDF dump. The histogram
   * says HOW MANY tasks were slow; this says WHICH. Open addressing over a
   * fixed array so a completion costs a few atomics and no lock.
   */
  struct MethodTime {
    std::atomic<u64> key{0};      ///< (pool_major << 16 | method) + 1; 0 = free
    std::atomic<u64> count{0};    ///< completions
    std::atomic<u64> wall_ns{0};  ///< summed wall time
    std::atomic<u64> max_ns{0};   ///< slowest single completion
  };
  static constexpr size_t kMethodTimeSlots = 1024;

  /**
   * Render the per-method table as one log line: the top_n entries by summed
   * wall time as " pool.method n=.. sum_ms=.. avg_ms=.. max_ms=.. |".
   * @param top_n How many entries to include
   * @return The rendered entries (empty when nothing has completed)
   */
  std::string MethodTimeReport(size_t top_n) const;
  void AdjustPolling(const clio::run::shared_ptr<Task> &task) override;
  Worker *GetGpuWorker() const override { return gpu_worker_; }
  // Legacy alias — admin_runtime.cc:Create registers transport FDs with the
  // net worker's EventManager. The recv worker is the one that polls those
  // FDs, so it's the natural EventManager owner.
  Worker *GetNetWorker() const override { return net_recv_worker_; }
  Worker *GetNetSendWorker() const override { return net_send_worker_; }
  Worker *GetNetRecvWorker() const override { return net_recv_worker_; }
  Worker *PickAltWorker(u32 avoid_id) const override;

  void GrowPool() override;

  /**
   * Most work (us) a worker should hold: about one OS context switch. Past
   * that, a task waiting behind it would be served sooner by another thread,
   * so the pool grows instead.
   */
  static constexpr double kMaxWorkerHoldUs = 100.0;

  /** Most one task may book (us): bounds the damage of a bad prediction. */
  static constexpr double kMaxBookedHoldUs = 1.0e6;

  /**
   * Whether  w can take a task predicted to hold a worker  hold_us. An
   * idle worker can always take one, however long.
   * @param w the worker
   * @param hold_us the task's predicted hold time (us)
   * @param now_us current time (NowUs)
   * @return true if the worker stays within kMaxWorkerHoldUs
   */
  static bool HasRoom(const Worker *w, double hold_us, double now_us);

 private:
  /// #781: worker stall cutoff. A task holding its worker this long has its
  /// worker's backlog redistributed and a replacement started. 0.3 s was
  /// tried and flagged routine 1-2 s device tasks on every worker; liveness
  /// no longer depends on worker queues (control channel), so 1 s.
  static constexpr double kStallThresholdSec = 1.0;
  static constexpr double kRetireCooldownSec = 5.0;  ///< #781: idle elastic retire
  static constexpr u32 kStealBatch = 4;              ///< #781: tasks stolen per victim

  /// #781 telemetry-only histogram of completed-task exec times. NOT used for
  /// routing (routing uses measured realtime load, not predictions).
  enum PerfBin {
    kLt10us = 0, kLt50us, kLt500us, kLt10ms, kLt50ms, kLt500ms, kLt1s, kGe1s,
    kNumPerfBins
  };
  static PerfBin BinFor(double exec_us) {
    if (exec_us < 10) return kLt10us;
    if (exec_us < 50) return kLt50us;
    if (exec_us < 500) return kLt500us;
    if (exec_us < 10000) return kLt10ms;
    if (exec_us < 50000) return kLt50ms;
    if (exec_us < 500000) return kLt500ms;
    if (exec_us < 1000000) return kLt1s;
    return kGe1s;
  }
  std::array<std::atomic<u64>, kNumPerfBins> perf_pdf_{};  ///< #781 telemetry
  std::array<MethodTime, kMethodTimeSlots> method_times_{};  ///< see MethodTime
  std::atomic<u64> load_balance_ticks_{0};  ///< #781 monitor ticks observed
  std::atomic<u64> stalls_detected_{0};     ///< #781 cumulative stall events
  std::atomic<u64> rescues_performed_{0};   ///< #785 lane transfers off stalled workers
  /** #785: replacement workers spawned by rescues. Only ever touched by
   *  LoadBalance on the monitor thread, so it needs no lock. Kept out of
   *  pool_ (which the mapper scans) so a rescuer is not immediately handed
   *  fresh work, but still checked for stalls so cascades are rescued. */
  std::vector<Worker *> elastic_workers_;

  /** #785 progress watchdog. Monitor-thread only, so plain members. */
  static constexpr double kNoProgressAlarmSec = 10.0;   ///< nothing executing
  static constexpr double kAllBlockedAlarmSec = 60.0;   ///< all blocked (ambiguous)
  u64 last_progress_count_ = 0;
  double last_progress_us_ = 0.0;
  bool progress_alarm_raised_ = false;

  /** #785: least-loaded live worker with a lane, excluding two given workers.
   *  Used to re-place a stalled worker's queued backlog. Monitor thread only. */
  Worker *PickLeastLoadedLive(double now_us, Worker *avoid_a,
                              Worker *avoid_b) const;

  /**
   * Predict how long a task will hold a worker, and record the predictions on
   * it (PredictedLoad = CPU, PredictedWallUs = wall) so the worker books and
   * trains on the same values.
   * @param task the task
   * @param container its execution container
   * @return the hold estimate (us), from the wall model
   */
  static double PredictHold(Task *task, Container &container);

  /**
   * Least-loaded pool worker, scanning from a rotating start so ties spread.
   * @param now_us current time (NowUs)
   * @return the worker, or nullptr if the pool is empty
   */
  Worker *PickPoolWorker(double now_us) const;

  /** Whether  w is in pool_. */
  bool InPool(const Worker *w) const;

  /** Copy of pool_, for monitor-thread scans that may call out. */
  std::vector<Worker *> PoolSnapshot() const;

  /// The routable workers: worker 0, the startup workers between it and the
  /// GPU/network block, and every worker GrowPool added. Read by mappers on
  /// any thread, appended to by GrowPool on the monitor thread.
  std::vector<Worker *> pool_;
  mutable std::mutex pool_mu_;  ///< guards pool_
  std::atomic<u64> pool_grows_{0};  ///< workers GrowPool added (telemetry)
  // Placement telemetry: on the submitting worker, on another pool worker
  // with room, or on the least-loaded worker with no room anywhere.
  std::atomic<u64> map_here_{0};
  std::atomic<u64> map_other_{0};
  std::atomic<u64> map_full_{0};
  /// Set when SpawnAdditionalWorker refused (thread budget spent). Monitor
  /// thread only, like GrowPool.
  bool spawn_exhausted_ = false;

  Worker *scheduler_worker_;              ///< Worker 0 (in pool_)
  Worker *net_send_worker_;               ///< Worker N-2: kSend / kClientSend
  Worker *net_recv_worker_;               ///< Worker N-1: kRecv / kClientRecv
  Worker *gpu_worker_;                    ///< GPU queue polling worker
  /// Rotating scan start for PickPoolWorker. mutable so the const picker
  /// can advance it (ties would otherwise always pick pool_[0]).
  mutable std::atomic<u32> next_io_idx_{0};
  WorkOrchestrator *work_orch_ = nullptr; ///< #781 set in DivideWorkers; elastic spawn
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_SCHEDULER_DEFAULT_SCHED_H_
