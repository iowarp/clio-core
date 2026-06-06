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

#ifndef CLIO_RUNTIME_IPC_RUN2RUN_H_
#define CLIO_RUNTIME_IPC_RUN2RUN_H_

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <clio_runtime/task.h>
#include <clio_runtime/task_archives.h>
#include <clio_ctp/data_structures/priv/unordered_map_ll.h>

namespace ctp::lbm { class Transport; }

namespace clio::run {

/** Return code set on tasks that fail due to network timeout */
static constexpr int kRun2RunNetworkTimeoutRC = -1000;

/** How long (seconds) to keep a task in the retry queue before failing it */
static constexpr float kRun2RunRetryTimeoutSec = 30.0f;

/** Entry in a retry queue for tasks that could not be sent */
struct RetryEntry {
  ctp::ipc::FullPtr<chi::Task> task;
  chi::u64 target_node_id;
  std::chrono::steady_clock::time_point enqueued_at;
};

/**
 * Encapsulates all run-to-run IPC state and logic: the send/recv maps,
 * retry queues, per-peer network statistics, and the SendIn / SendOut /
 * RecvIn / RecvOut path.
 *
 * Admin::Runtime owns one instance of this class and delegates every
 * cross-node task-transfer operation through it.
 */
class IpcManagerRun2Run {
 public:
  IpcManagerRun2Run();

  /**
   * Send task inputs to remote nodes.
   * Looks up the target node(s) from the task's pool queries, creates a copy
   * per replica, serializes, and sends via Lightbeam.  Dead/unreachable nodes
   * are queued in send_in_retry_ for later retry.
   */
  void SendIn(ctp::ipc::FullPtr<chi::Task> origin_task);

  /**
   * Send task outputs back to the originating node.
   * Reads the return-node from pool_query_, serializes via Lightbeam, then
   * calls DelTask on success.  Failures are queued in send_out_retry_.
   */
  void SendOut(ctp::ipc::FullPtr<chi::Task> origin_task);

  /**
   * Receive task inputs from a remote node (inbound kSerializeIn messages).
   * Deserializes the task, registers it in recv_map_, then dispatches it
   * onto a worker lane.
   * @return 0 on success, non-zero on error.
   */
  int RecvIn(chi::LoadTaskArchive &archive, ctp::lbm::Transport *lbm_transport);

  /**
   * Receive task outputs from a remote node (inbound kSerializeOut messages).
   * Two-pass: first deserializes outputs into replica tasks (exposes bulk
   * buffers), then aggregates each replica into the origin task and completes
   * the origin when all replicas are done.
   * @return 0 on success, non-zero on error.
   */
  int RecvOut(chi::LoadTaskArchive &archive, ctp::lbm::Transport *lbm_transport);

  /**
   * Process the send_in_retry_ and send_out_retry_ queues.
   * Retries sends to nodes that have come back alive, re-routes to recovered
   * nodes via the address map, and times out entries that have exceeded the
   * per-task (or default) timeout.
   */
  void ProcessRetryQueues();

  /**
   * Scan send_map_ for tasks waiting on nodes that have been marked dead and
   * have exceeded their timeout.  Completes those tasks with a network-timeout
   * return code.
   */
  void ScanSendMapTimeouts();

  /**
   * Discard all retry-queue entries targeting node_id before it is marked
   * alive again (restarted node).  Prevents stale tasks from the previous
   * incarnation being re-sent to a fresh runtime.
   */
  void FlushStaleStateForNode(chi::u64 node_id);

  /**
   * Emit per-peer network statistics if the dump interval has elapsed.
   * Called once per Send periodic tick.
   */
  void DumpNetStats(chi::u64 self_node_id, size_t num_peers);

  /**
   * Spawn the dedicated peer-recv and client-recv threads.
   * Must be called once the IpcManager main transport is (or will be) available.
   */
  void StartRecvThreads();

  /**
   * Signal recv threads to stop and join them.
   * Must be called before the main transport is freed.  Idempotent.
   */
  void StopRecvThreads();

  /** Number of tasks waiting for remote responses (send_map_ size). */
  size_t GetSendMapSize() const {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    return send_map_.size();
  }

  /** Number of remotely-received tasks not yet responded to (recv_map_ size). */
  size_t GetRecvMapSize() const {
    std::lock_guard<std::mutex> lk(recv_map_mutex_);
    return recv_map_.size();
  }

 private:
  // ---------------------------------------------------------------------------
  // SendIn sub-functions
  // ---------------------------------------------------------------------------

  /** Resolve the target node_id for one pool query. Returns 0 to skip. */
  chi::u64 SendInResolveTargetNode(chi::IpcManager *ipc_manager,
                                    chi::PoolManager *pool_manager,
                                    ctp::ipc::FullPtr<chi::Task> origin_task,
                                    const chi::PoolQuery &query);

  /**
   * Serialize task_copy and transmit it to target_node_id.
   * On failure marks the node dead (if appropriate) and queues in
   * send_in_retry_.  Updates per-peer NetStats on success.
   */
  void SendInTransmitReplica(chi::Container *container,
                              chi::IpcManager *ipc_manager,
                              ctp::ipc::FullPtr<chi::Task> task_copy,
                              chi::u64 target_node_id,
                              ctp::ipc::FullPtr<chi::Task> origin_task);

  // ---------------------------------------------------------------------------
  // SendOut sub-functions
  // ---------------------------------------------------------------------------

  /**
   * Serialize origin_task (out-direction) and send to target_node_id.
   * Updates per-peer NetStats on success, queues in send_out_retry_ on
   * failure.  Returns the Lightbeam rc.
   */
  int SendOutTransmit(chi::Container *container,
                      chi::IpcManager *ipc_manager,
                      ctp::ipc::FullPtr<chi::Task> origin_task,
                      chi::u64 target_node_id,
                      const chi::Host *target_host);

  // ---------------------------------------------------------------------------
  // RecvIn sub-functions
  // ---------------------------------------------------------------------------

  /**
   * Deserialize, register, and dispatch one inbound task.
   * Returns false if the task could not be loaded or dispatched.
   */
  bool RecvInHandleOne(chi::IpcManager *ipc_manager,
                       chi::PoolManager *pool_manager,
                       const chi::TaskInfo &task_info,
                       chi::LoadTaskArchive &archive,
                       ctp::lbm::Transport *lbm_transport);

  // ---------------------------------------------------------------------------
  // RecvOut sub-functions
  // ---------------------------------------------------------------------------

  /**
   * First pass: load output data from archive into each replica task.
   * Returns non-zero on hard error.
   */
  int RecvOutDeserialize(chi::PoolManager *pool_manager,
                         const std::vector<chi::TaskInfo> &task_infos,
                         chi::LoadTaskArchive &archive);

  /**
   * Second pass: aggregate each replica into its origin task; complete the
   * origin when all replicas have been received.
   * Returns non-zero on hard error.
   */
  int RecvOutAggregate(const std::vector<chi::TaskInfo> &task_infos);

  /**
   * Finalize an origin task once all its replicas have been aggregated:
   * delete replica tasks, remove from send_map_, and call EndTask.
   */
  void RecvOutCompleteOriginTask(size_t net_key,
                                  ctp::ipc::FullPtr<chi::Task> origin_task,
                                  chi::RunContext *origin_rctx);

  // ---------------------------------------------------------------------------
  // Retry helpers
  // ---------------------------------------------------------------------------

  /** Attempt to (re-)send a retry entry's task to node_id via Lightbeam. */
  bool RetrySendToNode(RetryEntry &entry, chi::u64 node_id);

  /**
   * Re-resolve the target node for a retry entry whose original target is dead.
   * Consults the current address_map_ via pool_manager.
   * @return New node ID, or 0 if resolution fails.
   */
  chi::u64 RerouteRetryEntry(RetryEntry &entry);

  // -------------------------------------------------------------------------
  // Dedicated recv threads
  // -------------------------------------------------------------------------
  std::atomic<bool> recv_shutdown_{false};
  std::thread peer_recv_thread_;
  std::thread client_recv_thread_;

  // -------------------------------------------------------------------------
  // Maps for in-flight tasks
  //
  // send_map_: origin task -> send_map_key; written by SendIn on
  //            net_send_worker, read/erased by RecvOut on net_recv_worker.
  //            Guard with send_map_mutex_.
  // recv_map_: received task -> recv_key; written by RecvIn on
  //            net_recv_worker, read/erased by SendOut on net_send_worker.
  //            Guard with recv_map_mutex_.
  // -------------------------------------------------------------------------
  static constexpr size_t kNumMapBuckets = 1024;
  mutable std::mutex send_map_mutex_;
  mutable std::mutex recv_map_mutex_;
  ctp::priv::unordered_map_ll<size_t, ctp::ipc::FullPtr<chi::Task>> send_map_;
  ctp::priv::unordered_map_ll<size_t, ctp::ipc::FullPtr<chi::Task>> recv_map_;

  // Retry queues for tasks that could not be sent due to dead / unreachable
  // nodes.  Guarded by retry_queues_mutex_.
  mutable std::mutex retry_queues_mutex_;
  std::deque<RetryEntry> send_in_retry_;
  std::deque<RetryEntry> send_out_retry_;

  // -------------------------------------------------------------------------
  // Per-peer network statistics
  // -------------------------------------------------------------------------
  struct NetPeerStats {
    uint64_t sin_count = 0;   // SendIn message count
    uint64_t sin_bytes = 0;
    uint64_t sin_lbm_ns = 0;  // time inside lbm_transport->Send for SendIn
    uint64_t sin_ser_ns = 0;  // time inside container->SaveTask for SendIn
    uint64_t sout_count = 0;  // SendOut message count
    uint64_t sout_bytes = 0;
    uint64_t sout_lbm_ns = 0;
    uint64_t sout_ser_ns = 0;
    uint64_t rin_count = 0;   // RecvIn message count
    uint64_t rin_bytes = 0;
    uint64_t rin_des_ns = 0;  // time inside deserialize/dispatch for RecvIn
    uint64_t rout_count = 0;  // RecvOut message count
    uint64_t rout_bytes = 0;
    uint64_t rout_des_ns = 0;
  };

  static constexpr size_t kMaxNetPeers = 64;
  static constexpr double kNetStatsDumpIntervalSec = 1.0;
  NetPeerStats peer_stats_[kMaxNetPeers] = {};
  std::chrono::steady_clock::time_point net_stats_last_dump_ =
      std::chrono::steady_clock::now();

  NetPeerStats &NetStats(chi::u64 peer) {
    return peer_stats_[peer % kMaxNetPeers];
  }
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_IPC_RUN2RUN_H_
