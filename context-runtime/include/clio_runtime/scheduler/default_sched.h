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
#include <atomic>
#include <vector>

namespace clio::run {

/**
 * Default scheduler implementation with I/O-size-based routing.
 * Routes tasks based on io_size_: small I/O and metadata go to the scheduler
 * worker (worker 0), large I/O (>= 4KB) goes to dedicated I/O workers via
 * round-robin, and network tasks split across two dedicated workers — one
 * owns ROUTER recvs (kRecv / kClientRecv), the other owns DEALER sends
 * (kSend / kClientSend) — so a backlog on either direction can't starve
 * SWIM heartbeat probes on the other.
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
  void RecordCompletion(u32 method, double cpu_us, double wall_us) override;
  void AdjustPolling(const clio::run::shared_ptr<Task> &task) override;
  Worker *GetGpuWorker() const override { return gpu_worker_; }
  // Legacy alias — admin_runtime.cc:Create registers transport FDs with the
  // net worker's EventManager. The recv worker is the one that polls those
  // FDs, so it's the natural EventManager owner.
  Worker *GetNetWorker() const override { return net_recv_worker_; }
  Worker *GetNetSendWorker() const override { return net_send_worker_; }
  Worker *GetNetRecvWorker() const override { return net_recv_worker_; }
  Worker *PickAltWorker(u32 avoid_id) const override;

 private:
  static constexpr size_t kLargeIOThreshold = 4096;  ///< I/O size threshold
  static constexpr double kStallThresholdSec = 1.0;  ///< #781: worker stall cutoff
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
  std::atomic<u64> load_balance_ticks_{0};  ///< #781 monitor ticks observed
  std::atomic<u64> stalls_detected_{0};     ///< #781 cumulative stall events
  std::atomic<u64> rescues_performed_{0};   ///< #785 lane transfers off stalled workers
  /** #785: replacement workers spawned by rescues. Only ever touched by
   *  LoadBalance on the monitor thread, so it needs no lock. Kept separate from
   *  io_workers_ (which the mapper scans) so a rescuer is not immediately handed
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

  Worker *scheduler_worker_;              ///< Worker 0: metadata + small I/O
  std::vector<Worker *> io_workers_;      ///< Workers 1..N-3: large I/O
  Worker *net_send_worker_;               ///< Worker N-2: kSend / kClientSend
  Worker *net_recv_worker_;               ///< Worker N-1: kRecv / kClientRecv
  Worker *gpu_worker_;                    ///< GPU queue polling worker
  std::atomic<u32> next_io_idx_{0};       ///< Round-robin index for I/O workers
  WorkOrchestrator *work_orch_ = nullptr; ///< #781 set in DivideWorkers; elastic spawn
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_SCHEDULER_DEFAULT_SCHED_H_
