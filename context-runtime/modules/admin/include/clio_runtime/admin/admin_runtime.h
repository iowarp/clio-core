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
class Runtime : public chi::Container {
public:
  // CreateParams type used by CLIO_TASK_CC macro for lib_name access
  using CreateParams = clio::run::admin::CreateParams;

private:
  // Container-specific state
  chi::u32 create_count_ = 0;
  chi::u32 pools_created_ = 0;
  chi::u32 pools_destroyed_ = 0;

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
  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Schedule a task by resolving Dynamic pool queries.
   */
  chi::PoolQuery ScheduleTask(const ctp::ipc::FullPtr<chi::Task> &task) override;

  /**
   * Execute a method on a task
   */
  chi::TaskResume Run(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext &rctx) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task - Initialize the Admin container (IS_ADMIN=true)
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume Create(ctp::ipc::FullPtr<CreateTask> task, chi::RunContext &rctx);

  /**
   * Handle GetOrCreatePool task - Pool get-or-create operation (IS_ADMIN=false)
   * This is a coroutine that can co_await nested Create methods
   */
  chi::TaskResume GetOrCreatePool(
      ctp::ipc::FullPtr<
          clio::run::admin::GetOrCreatePoolTask<clio::run::admin::CreateParams>>
          task,
      chi::RunContext &rctx);

  /**
   * Handle Destroy task - Alias for DestroyPool (DestroyTask = DestroyPoolTask)
   * This is a coroutine for consistency with GetOrCreatePool
   */
  chi::TaskResume Destroy(ctp::ipc::FullPtr<DestroyTask> task, chi::RunContext &rctx);

  /**
   * Handle DestroyPool task - Destroy an existing ChiPool
   * This is a coroutine that can co_await pool destruction
   */
  chi::TaskResume DestroyPool(ctp::ipc::FullPtr<DestroyPoolTask> task, chi::RunContext &rctx);

  /**
   * Handle StopRuntime task - Stop the entire runtime
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume StopRuntime(ctp::ipc::FullPtr<StopRuntimeTask> task, chi::RunContext &rctx);

  /**
   * Handle Flush task - Flush administrative operations
   */
  chi::TaskResume Flush(ctp::ipc::FullPtr<FlushTask> task, chi::RunContext &rctx);

  //===========================================================================
  // Distributed Task Scheduling Methods
  //===========================================================================

  /**
   * Handle Send - Send task inputs or outputs over network
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume Send(ctp::ipc::FullPtr<SendTask> task, chi::RunContext &rctx);

  /**
   * Handle Recv - Receive task inputs or outputs from network
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume Recv(ctp::ipc::FullPtr<RecvTask> task, chi::RunContext &rctx);

  /**
   * Handle ClientConnect - Respond to client connection request
   * Sets response to 0 to indicate runtime is healthy
   */
  chi::TaskResume ClientConnect(ctp::ipc::FullPtr<ClientConnectTask> task, chi::RunContext &rctx);

  /**
   * Handle ClientRecv - Receive tasks from ZMQ clients (TCP/IPC)
   * Polls ZMQ ROUTER sockets for incoming task submissions
   */
  chi::TaskResume ClientRecv(ctp::ipc::FullPtr<ClientRecvTask> task, chi::RunContext &rctx);

  /**
   * Handle ClientSend - Send completed task outputs to ZMQ clients
   * Polls net_queue_ kClientSendTcp/kClientSendIpc priorities
   */
  chi::TaskResume ClientSend(ctp::ipc::FullPtr<ClientSendTask> task, chi::RunContext &rctx);

  /**
   * Handle WreapDeadIpcs - Periodic task to reap shared memory from dead processes
   * Calls IpcManager::WreapDeadIpcs() to clean up orphaned shared memory segments
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume WreapDeadIpcs(ctp::ipc::FullPtr<WreapDeadIpcsTask> task, chi::RunContext &rctx);

  /**
   * Handle Monitor - Unified monitor query for admin chimod
   * Supported queries:
   *   "worker_stats" - collect worker statistics (msgpack-encoded)
   *   "pool_stats://<pool_id>:<routing>:<selector>" - delegate to a pool
   *   "system_stats[:<min_event_id>]" - system resource utilization
   *   "bdev_stats" - block device statistics
   */
  chi::TaskResume Monitor(ctp::ipc::FullPtr<MonitorTask> task, chi::RunContext &rctx);

  /** Monitor sub-handler: collect per-worker statistics. */
  void MonitorWorkerStats(ctp::ipc::FullPtr<MonitorTask> task);

  /** Monitor sub-handler: return per-container model statistics. */
  void MonitorContainerStats(ctp::ipc::FullPtr<MonitorTask> task);

  /** Monitor sub-handler: delegate query to a specific pool. */
  chi::TaskResume MonitorPoolStats(ctp::ipc::FullPtr<MonitorTask> task);

  /** Monitor sub-handler: return system_stats ring buffer entries. */
  void MonitorSystemStats(ctp::ipc::FullPtr<MonitorTask> task);

  /** Monitor sub-handler: collect bdev pool statistics. */
  chi::TaskResume MonitorBdevStats(ctp::ipc::FullPtr<MonitorTask> task);

  /** Monitor sub-handler: return host info (hostname, IP, node_id). */
  void MonitorGetHostInfo(ctp::ipc::FullPtr<MonitorTask> task);

  /**
   * Handle AnnounceShutdown - Mark a departing node as dead immediately
   * and trigger recovery if this node is the new leader.
   */
  chi::TaskResume AnnounceShutdown(ctp::ipc::FullPtr<AnnounceShutdownTask> task,
                                    chi::RunContext &rctx);

  /**
   * Handle RegisterMemory - Register client shared memory with runtime
   * Called by SHM-mode clients after IncreaseMemory() to tell the runtime
   * to attach to the new shared memory segment
   */
  chi::TaskResume RegisterMemory(ctp::ipc::FullPtr<RegisterMemoryTask> task, chi::RunContext &rctx);

  /**
   * Handle RestartContainers - Re-create pools from saved restart configs
   * Reads conf_dir/restart/ directory and re-creates pools from saved YAML
   */
  chi::TaskResume RestartContainers(ctp::ipc::FullPtr<RestartContainersTask> task, chi::RunContext &rctx);

  /**
   * Handle ListContainers - Enumerate active pools/containers in this daemon.
   * Fills the task's pool_names_ / pool_ids_ output vectors.
   */
  chi::TaskResume ListContainers(ctp::ipc::FullPtr<ListContainersTask> task, chi::RunContext &rctx);

  /**
   * Handle AddNode - Register a new node with this runtime
   * Updates IpcManager's hostfile and calls Expand on all containers
   */
  chi::TaskResume AddNode(ctp::ipc::FullPtr<AddNodeTask> task, chi::RunContext &rctx);

  /**
   * Handle SubmitBatch - Submit a batch of tasks in a single RPC
   * Deserializes tasks from the batch and executes them in parallel
   * up to 32 tasks at a time, then co_awaits their completion
   * @param task The SubmitBatchTask containing serialized tasks
   * @param rctx Runtime context for the current worker
   */
  chi::TaskResume SubmitBatch(ctp::ipc::FullPtr<SubmitBatchTask> task, chi::RunContext &rctx);

  /**
   * Handle ChangeAddressTable - Update ContainerId->NodeId mapping
   * Writes WAL entry and updates pool manager's address table
   */
  chi::TaskResume ChangeAddressTable(ctp::ipc::FullPtr<ChangeAddressTableTask> task, chi::RunContext &rctx);

  /**
   * Handle MigrateContainers - Orchestrate container migration
   * Processes each MigrateInfo entry and broadcasts address table changes
   */
  chi::TaskResume MigrateContainers(ctp::ipc::FullPtr<MigrateContainersTask> task, chi::RunContext &rctx);

  /**
   * Handle Heartbeat - Liveness probe, just returns success
   */
  chi::TaskResume Heartbeat(ctp::ipc::FullPtr<HeartbeatTask> task, chi::RunContext &rctx);

  /**
   * Handle HeartbeatProbe - Periodic SWIM failure detector
   * Sends direct probes, escalates to indirect probes, manages suspicion
   */
  chi::TaskResume HeartbeatProbe(ctp::ipc::FullPtr<HeartbeatProbeTask> task, chi::RunContext &rctx);

  /**
   * Handle ProbeRequest - Indirect probe on behalf of another node
   * Probes target node and returns result to requester
   */
  chi::TaskResume ProbeRequest(ctp::ipc::FullPtr<ProbeRequestTask> task, chi::RunContext &rctx);

  /**
   * Handle RecoverContainers - Recreate containers from dead nodes
   * All nodes update address_map_, only dest node creates the container
   */
  chi::TaskResume RecoverContainers(ctp::ipc::FullPtr<RecoverContainersTask> task, chi::RunContext &rctx);

  /**
   * Handle SystemMonitor - Periodic system resource utilization sampling
   * Samples DRAM, CPU, and (optionally) GPU/HBM utilization into ring buffer
   */
  chi::TaskResume SystemMonitor(ctp::ipc::FullPtr<SystemMonitorTask> task, chi::RunContext &rctx);

  /**
   * Handle RegisterGpuContainer - Register a GPU container with the GPU orchestrator
   * The GPU orchestrator's gpu::PoolManager will be updated with the new container
   */
  chi::TaskResume RegisterGpuContainer(ctp::ipc::FullPtr<RegisterGpuContainerTask> task, chi::RunContext &rctx);

  /**
   * Get live task statistics for this task instance.
   * For network periodics, compute_ = total queue depth across priorities;
   * io_size_ is a static stand-in (admin periodics don't have a payload).
   */
  chi::TaskStat GetTaskStats(const chi::Task *task) const override;

  /**
   * Get remaining work count for this admin container
   * Admin container typically has no pending work, returns 0
   */
  chi::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Task Serialization Methods
  //===========================================================================

  /**
   * Serialize task parameters (IN or OUT based on archive mode)
   */
  void SaveTask(chi::u32 method, chi::SaveTaskArchive &archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Deserialize task parameters into an existing task (IN or OUT based on archive mode)
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive &archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   */
  ctp::ipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive &archive) override;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   */
  void LocalLoadTask(chi::u32 method, chi::DefaultLoadArchive &archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   */
  ctp::ipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::DefaultLoadArchive &archive) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveTask(chi::u32 method, chi::DefaultSaveArchive &archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  ctp::ipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task_ptr,
                                        bool deep) override;

  /**
   * Create a new task of the specified method type
   */
  ctp::ipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;
  void Aggregate(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task,
                 const ctp::ipc::FullPtr<chi::Task>& replica_task) override;
  void DelTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr) override;

private:
  /**
   * Initiate runtime shutdown sequence
   */
  void InitiateShutdown(chi::u32 grace_period_ms);

  // SWIM failure detection state
  struct PendingProbe {
    chi::Future<HeartbeatTask> future;
    chi::u64 target_node_id;
    std::chrono::steady_clock::time_point sent_at;
  };
  struct PendingIndirectProbe {
    chi::Future<ProbeRequestTask> future;
    chi::u64 target_node_id;   // suspected node
    chi::u64 helper_node_id;   // node doing the probe
    std::chrono::steady_clock::time_point sent_at;
  };

  size_t probe_round_robin_idx_ = 0;
  std::vector<PendingProbe> pending_direct_probes_;
  std::vector<PendingIndirectProbe> pending_indirect_probes_;
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
  // be tuned from the deployment yaml (see chi::ConfigManager::
  // GetSwimDirectProbeTimeoutSec, GetSwimIndirectProbeTimeoutSec,
  // GetSwimSuspicionTimeoutSec, GetSwimEnabled). Defaults there match
  // the prior values (30s / 15s / 60s).
  static constexpr size_t kIndirectProbeHelpers = 3;

  // Recovery state
  std::vector<chi::RecoveryAssignment> ComputeRecoveryPlan(chi::u64 dead_node_id);
  chi::TaskResume TriggerRecovery(chi::u64 dead_node_id);
  std::unordered_set<chi::u64> recovery_initiated_;
};

} // namespace clio::run::admin

#endif // ADMIN_RUNTIME_H_