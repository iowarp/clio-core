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

#ifndef ADMIN_RUNTIME_H_
#define ADMIN_RUNTIME_H_

#include "admin_client.h"
#include "admin_tasks.h"
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/container.h>
#include <clio_runtime/pool_manager.h>
#include <clio_runtime/ipc/ipc_run2run.h>
#include <clio_ctp/data_structures/ipc/ring_buffer.h>
#include <clio_ctp/memory/allocator/malloc_allocator.h>
#include <clio_ctp/introspect/system_info.h>

#include <mutex>
#include <random>
#include <unordered_set>

namespace clio::run::admin {

// Admin local queue indices
enum AdminQueueIndex {
  kMetadataQueue = 0,          // Queue for metadata operations
  kClientSendTaskInQueue = 1,  // Queue for client task input processing
  kServerRecvTaskInQueue = 2,  // Queue for server task input reception
  kServerSendTaskOutQueue = 3, // Queue for server task output sending
  kClientRecvTaskOutQueue = 4  // Queue for client task output reception
};

// Forward declarations
// Note: CreateTask and GetOrCreatePoolTask are using aliases defined in
// admin_tasks.h We cannot forward declare using aliases, so we rely on the
// include

/**
 * Runtime implementation for Admin container
 *
 * Critical ChiMod responsible for managing ChiPools and runtime lifecycle.
 * Must always be found by the runtime or a fatal error occurs.
 */
class Runtime : public clio::run::Container {
public:
  // CreateParams type used by CLIO_TASK_CC macro for lib_name access
  using CreateParams = clio::run::admin::CreateParams;

private:
  // Container-specific state
  clio::run::u32 create_count_ = 0;
  clio::run::u32 pools_created_ = 0;
  clio::run::u32 pools_destroyed_ = 0;

  // Runtime state
  bool is_shutdown_requested_ = false;

  // Client for making calls to this ChiMod
  Client client_;

  // System monitor ring buffer and CPU state
  static inline constexpr size_t kSystemStatsRingSize = 1024;
  std::unique_ptr<ctp::ipc::circular_mpsc_ring_buffer<SystemStats, ctp::ipc::MallocAllocator>>
      system_stats_ring_;
  ctp::CpuTimes prev_cpu_times_;

public:
  /**
   * Constructor
   */
  Runtime() = default;

  /**
   * Destructor.
   */
  virtual ~Runtime();

  /**
   * Initialize container with pool information
   */
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;

  /**
   * Schedule a task by resolving Dynamic pool queries.
   */
  clio::run::PoolQuery ScheduleTask(const clio::run::shared_ptr<clio::run::Task> &task) override;

  /**
   * Execute a method on a task
   */
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task - Initialize the Admin container (IS_ADMIN=true)
   * Returns TaskResume for consistency with other methods called from Run
   */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);

  /**
   * Handle GetOrCreatePool task - Pool get-or-create operation (IS_ADMIN=false)
   * This is a coroutine that can co_await nested Create methods
   */
  clio::run::TaskResume GetOrCreatePool(
      clio::run::shared_ptr<
          clio::run::admin::GetOrCreatePoolTask<clio::run::admin::CreateParams>>
          &task);

  /**
   * Handle Destroy task - Alias for DestroyPool (DestroyTask = DestroyPoolTask)
   * This is a coroutine for consistency with GetOrCreatePool
   */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  /**
   * Handle DestroyPool task - Destroy an existing ChiPool
   * This is a coroutine that can co_await pool destruction
   */
  clio::run::TaskResume DestroyPool(clio::run::shared_ptr<DestroyPoolTask> &task);

  /**
   * Handle StopRuntime task - Stop the entire runtime
   * Returns TaskResume for consistency with other methods called from Run
   */
  clio::run::TaskResume StopRuntime(clio::run::shared_ptr<StopRuntimeTask> &task);

  /**
   * Handle Flush task - Flush administrative operations
   */
  clio::run::TaskResume Flush(clio::run::shared_ptr<FlushTask> &task);

  //===========================================================================
  // Distributed Task Scheduling Methods
  //===========================================================================

  /**
   * Handle Send - Send task inputs or outputs over network
   * Returns TaskResume for consistency with other methods called from Run
   */
  clio::run::TaskResume Send(clio::run::shared_ptr<SendTask> &task);

  /**
   * Handle Recv - Receive task inputs or outputs from network
   * Returns TaskResume for consistency with other methods called from Run
   */
  clio::run::TaskResume Recv(clio::run::shared_ptr<RecvTask> &task);

  /**
   * Handle ClientConnect - Respond to client connection request
   * Sets response to 0 to indicate runtime is healthy
   */
  clio::run::TaskResume ClientConnect(clio::run::shared_ptr<ClientConnectTask> &task);

  /**
   * Handle ClientRecv - Receive tasks from ZMQ clients (TCP/IPC)
   * Polls ZMQ ROUTER sockets for incoming task submissions
   */
  clio::run::TaskResume ClientRecv(clio::run::shared_ptr<ClientRecvTask> &task);

  /**
   * Handle ClientSend - Send completed task outputs to ZMQ clients
   * Polls net_queue_ kClientSendTcp/kClientSendIpc priorities
   */
  clio::run::TaskResume ClientSend(clio::run::shared_ptr<ClientSendTask> &task);

  /**
   * Handle WreapDeadIpcs - Periodic task to reap shared memory from dead processes
   * Calls IpcManager::WreapDeadIpcs() to clean up orphaned shared memory segments
   * Returns TaskResume for consistency with other methods called from Run
   */
  clio::run::TaskResume WreapDeadIpcs(clio::run::shared_ptr<WreapDeadIpcsTask> &task);

  /**
   * Handle Monitor - Unified monitor query for admin chimod
   * Supported queries:
   *   "worker_stats" - collect worker statistics (msgpack-encoded)
   *   "pool_stats://<pool_id>:<routing>:<selector>" - delegate to a pool
   *   "system_stats[:<min_event_id>]" - system resource utilization
   *   "bdev_stats" - block device statistics
   */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);

  /** Monitor sub-handler: collect per-worker statistics. */
  void MonitorWorkerStats(clio::run::shared_ptr<MonitorTask> &task);

  /** Monitor sub-handler: return per-container model statistics. */
  void MonitorContainerStats(clio::run::shared_ptr<MonitorTask> &task);

  /** Monitor sub-handler: delegate query to a specific pool. */
  clio::run::TaskResume MonitorPoolStats(clio::run::shared_ptr<MonitorTask> &task);

  /** Monitor sub-handler: return system_stats ring buffer entries. */
  void MonitorSystemStats(clio::run::shared_ptr<MonitorTask> &task);

  /** Monitor sub-handler: collect bdev pool statistics. */
  clio::run::TaskResume MonitorBdevStats(clio::run::shared_ptr<MonitorTask> &task);

  /** Monitor sub-handler: return host info (hostname, IP, node_id). */
  void MonitorGetHostInfo(clio::run::shared_ptr<MonitorTask> &task);

  /**
   * Handle AnnounceShutdown - Mark a departing node as dead immediately
   * and trigger recovery if this node is the new leader.
   */
  clio::run::TaskResume AnnounceShutdown(clio::run::shared_ptr<AnnounceShutdownTask> &task);

  /**
   * Handle RegisterMemory - Register client shared memory with runtime
   * Called by SHM-mode clients after IncreaseMemory() to tell the runtime
   * to attach to the new shared memory segment
   */
  clio::run::TaskResume RegisterMemory(clio::run::shared_ptr<RegisterMemoryTask> &task);

  /**
   * Handle RestartContainers - Re-create pools from the restart registry.
   * Reads the RestartLog write-ahead log (~/.clio/restart_log.bin), the same
   * persistent registry replayed at startup, and re-composes each registered
   * compose file.
   */
  clio::run::TaskResume RestartContainers(clio::run::shared_ptr<RestartContainersTask> &task);

  /**
   * Handle ListContainers - Enumerate active pools/containers in this daemon.
   * Fills the task's pool_names_ / pool_ids_ output vectors.
   */
  clio::run::TaskResume ListContainers(clio::run::shared_ptr<ListContainersTask> &task);

  /**
   * Handle AddNode - Register a new node with this runtime
   * Updates IpcManager's hostfile and calls Expand on all containers
   */
  clio::run::TaskResume AddNode(clio::run::shared_ptr<AddNodeTask> &task);

  /**
   * Handle SubmitBatch - Submit a batch of tasks in a single RPC
   * Deserializes tasks from the batch and executes them in parallel
   * up to 32 tasks at a time, then co_awaits their completion
   * @param task The SubmitBatchTask containing serialized tasks
   * @param rctx Runtime context for the current worker
   */
  clio::run::TaskResume SubmitBatch(clio::run::shared_ptr<SubmitBatchTask> &task);

  /**
   * Handle ChangeAddressTable - Update ContainerId->NodeId mapping
   * Writes WAL entry and updates pool manager's address table
   */
  clio::run::TaskResume ChangeAddressTable(clio::run::shared_ptr<ChangeAddressTableTask> &task);

  /**
   * Handle MigrateContainers - Orchestrate container migration
   * Processes each MigrateInfo entry and broadcasts address table changes
   */
  clio::run::TaskResume MigrateContainers(clio::run::shared_ptr<MigrateContainersTask> &task);

  /**
   * Handle Heartbeat - Liveness probe, just returns success
   */
  clio::run::TaskResume Heartbeat(clio::run::shared_ptr<HeartbeatTask> &task);

  /**
   * Handle QueryTaskProgress - report whether a specific replica task is still
   * alive on this node (issue #628), answered from the run2run recv_map_.
   */
  clio::run::TaskResume QueryTaskProgress(
      clio::run::shared_ptr<QueryTaskProgressTask> &task);

  /**
   * Handle HeartbeatProbe - Periodic SWIM failure detector
   * Sends direct probes, escalates to indirect probes, manages suspicion
   */
  clio::run::TaskResume HeartbeatProbe(clio::run::shared_ptr<HeartbeatProbeTask> &task);

  /**
   * Handle ProbeRequest - Indirect probe on behalf of another node
   * Probes target node and returns result to requester
   */
  clio::run::TaskResume ProbeRequest(clio::run::shared_ptr<ProbeRequestTask> &task);

  /**
   * Handle RecoverContainers - Recreate containers from dead nodes
   * All nodes update address_map_, only dest node creates the container
   */
  clio::run::TaskResume RecoverContainers(clio::run::shared_ptr<RecoverContainersTask> &task);

  /**
   * Handle SystemMonitor - Periodic system resource utilization sampling
   * Samples DRAM, CPU, and (optionally) GPU/HBM utilization into ring buffer
   */
  clio::run::TaskResume SystemMonitor(clio::run::shared_ptr<SystemMonitorTask> &task);

  /**
   * Handle RegisterGpuContainer - Register a GPU container with the GPU orchestrator
   * The GPU orchestrator's gpu::PoolManager will be updated with the new container
   */
  clio::run::TaskResume RegisterGpuContainer(clio::run::shared_ptr<RegisterGpuContainerTask> &task);

  /**
   * Get live task statistics for this task instance.
   * For network periodics, compute_ = total queue depth across priorities;
   * io_size_ is a static stand-in (admin periodics don't have a payload).
   */
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override;

  /**
   * Get remaining work count for this admin container
   * Admin container typically has no pending work, returns 0
   */
  clio::run::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Task Serialization Methods
  //===========================================================================

  /**
   * Serialize task parameters (IN or OUT based on archive mode)
   */
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Deserialize task parameters into an existing task (IN or OUT based on archive mode)
   */
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   */
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   */
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   */
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
                                        bool deep) override;

  /**
   * Create a new task of the specified method type
   */
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override;

private:
  /**
   * Initiate runtime shutdown sequence
   */
  void InitiateShutdown(clio::run::u32 grace_period_ms);

  // SWIM failure detection state
  struct PendingProbe {
    clio::run::Future<HeartbeatTask> future;
    clio::run::u64 target_node_id;
    std::chrono::steady_clock::time_point sent_at;
  };
  struct PendingIndirectProbe {
    clio::run::Future<ProbeRequestTask> future;
    clio::run::u64 target_node_id;   // suspected node
    clio::run::u64 helper_node_id;   // node doing the probe
    std::chrono::steady_clock::time_point sent_at;
  };

  size_t probe_round_robin_idx_ = 0;
  std::vector<PendingProbe> pending_direct_probes_;
  std::vector<PendingIndirectProbe> pending_indirect_probes_;

  // #628: in-flight QueryTaskProgress probes for the cross-node task-progress
  // validity check. Fired (fire-and-poll) by ScanTaskProgress and reaped on a
  // later tick via Future::IsComplete() -- never awaited, so the net-processing
  // tick that drives their transmission is never blocked on itself.
  struct PendingProgressQuery {
    clio::run::Future<QueryTaskProgressTask> future;
    size_t net_key;
    clio::run::u32 replica_id;
  };
  std::vector<PendingProgressQuery> pending_progress_queries_;

  /**
   * Periodic cross-node task-progress validity check (issue #628). Reaps
   * completed probes (Gone -> complete the origin) and fires new probes for
   * replicas the origin has waited on beyond task_progress_interval_ms.
   */
  void ScanTaskProgress();
  std::mt19937 probe_rng_{std::random_device{}()};

  // SWIM probe / suspicion timeouts. The prior 5 s direct + 3 s
  // indirect window was tight enough that any brief Send-side
  // congestion (e.g. 24 ranks * 256 cross-node PutBlobs ramping up
  // on the dedicated net worker) drove the round-trip past the
  // threshold and the peer was wrongly declared kDead, after which
  // every subsequent SendIn skipped zmq_send entirely. 30 s direct
  // probe matches what the rest of the codebase already assumes,
  // and gives the workload's burst phase room to drain.
  // SWIM probe / suspicion timeouts moved to ConfigManager so they can
  // be tuned from the deployment yaml (see clio::run::ConfigManager::
  // GetSwimDirectProbeTimeoutSec, GetSwimIndirectProbeTimeoutSec,
  // GetSwimSuspicionTimeoutSec, GetSwimEnabled). Defaults there match
  // the prior values (30s / 15s / 60s).
  static constexpr size_t kIndirectProbeHelpers = 3;

  // Recovery state
  std::vector<clio::run::RecoveryAssignment> ComputeRecoveryPlan(clio::run::u64 dead_node_id);
  clio::run::TaskResume TriggerRecovery(clio::run::u64 dead_node_id);
  std::unordered_set<clio::run::u64> recovery_initiated_;
};

} // namespace clio::run::admin

#endif // ADMIN_RUNTIME_H_