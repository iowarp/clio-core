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
#include <chimaera/chimaera.h>
#include <chimaera/container.h>
#include <chimaera/pool_manager.h>
#include <hermes_shm/data_structures/priv/unordered_map_ll.h>

#include <deque>
#include <random>
#include <unordered_set>

namespace chimaera::admin {

/** Return code set on tasks that fail due to network timeout */
static constexpr int kNetworkTimeoutRC = -1000;

/** How long (seconds) to keep a task in retry queue before failing it */
static constexpr float kRetryTimeoutSec = 30.0f;

/** Entry in a retry queue for tasks that couldn't be sent */
struct RetryEntry {
  hipc::FullPtr<chi::Task> task;
  chi::u64 target_node_id;
  std::chrono::steady_clock::time_point enqueued_at;
};

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
  // CreateParams type used by CHI_TASK_CC macro for lib_name access
  using CreateParams = chimaera::admin::CreateParams;

private:
  // Container-specific state
  chi::u32 create_count_ = 0;
  chi::u32 pools_created_ = 0;
  chi::u32 pools_destroyed_ = 0;

  // Runtime state
  bool is_shutdown_requested_ = false;

  // Client for making calls to this ChiMod
  Client client_;

  // Network task tracking maps (keyed by net_key for efficient lookup)
  // Using lock-free unordered_map_ll with 1024 buckets for high concurrency
  // Thread safety: All Send/Recv tasks are routed to a single dedicated net worker
  static constexpr size_t kNumMapBuckets = 1024;
  hshm::priv::unordered_map_ll<size_t, hipc::FullPtr<chi::Task>> send_map_{kNumMapBuckets};  // Tasks sent to remote nodes
  hshm::priv::unordered_map_ll<size_t, hipc::FullPtr<chi::Task>> recv_map_{kNumMapBuckets};  // Tasks received from remote nodes

public:
  /**
   * Constructor
   */
  Runtime() = default;

  /**
   * Destructor
   */
  virtual ~Runtime() = default;

  /**
   * Initialize container with pool information
   */
  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Execute a method on a task
   */
  chi::TaskResume Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext &rctx) override;

  /**
   * Delete/cleanup a task
   */
  void DelTask(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task - Initialize the Admin container (IS_ADMIN=true)
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume Create(hipc::FullPtr<CreateTask> task, chi::RunContext &rctx);

  /**
   * Handle GetOrCreatePool task - Pool get-or-create operation (IS_ADMIN=false)
   * This is a coroutine that can co_await nested Create methods
   */
  chi::TaskResume GetOrCreatePool(
      hipc::FullPtr<
          chimaera::admin::GetOrCreatePoolTask<chimaera::admin::CreateParams>>
          task,
      chi::RunContext &rctx);

  /**
   * Handle Destroy task - Alias for DestroyPool (DestroyTask = DestroyPoolTask)
   * This is a coroutine for consistency with GetOrCreatePool
   */
  chi::TaskResume Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &rctx);

  /**
   * Handle DestroyPool task - Destroy an existing ChiPool
   * This is a coroutine that can co_await pool destruction
   */
  chi::TaskResume DestroyPool(hipc::FullPtr<DestroyPoolTask> task, chi::RunContext &rctx);

  /**
   * Handle StopRuntime task - Stop the entire runtime
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume StopRuntime(hipc::FullPtr<StopRuntimeTask> task, chi::RunContext &rctx);

  /**
   * Handle Flush task - Flush administrative operations
   */
  chi::TaskResume Flush(hipc::FullPtr<FlushTask> task, chi::RunContext &rctx);

  //===========================================================================
  // Distributed Task Scheduling Methods
  //===========================================================================

  /**
   * Handle Send - Send task inputs or outputs over network
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume Send(hipc::FullPtr<SendTask> task, chi::RunContext &rctx);

  /**
   * Helper: Send task inputs to remote node
   */
  void SendIn(hipc::FullPtr<chi::Task> origin_task, chi::RunContext &rctx);

  /**
   * Helper: Send task outputs back to origin node
   */
  void SendOut(hipc::FullPtr<chi::Task> origin_task);

  /**
   * Handle Recv - Receive task inputs or outputs from network
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume Recv(hipc::FullPtr<RecvTask> task, chi::RunContext &rctx);

  /**
   * Handle ClientConnect - Respond to client connection request
   * Sets response to 0 to indicate runtime is healthy
   */
  chi::TaskResume ClientConnect(hipc::FullPtr<ClientConnectTask> task, chi::RunContext &rctx);

  /**
   * Handle ClientRecv - Receive tasks from ZMQ clients (TCP/IPC)
   * Polls ZMQ ROUTER sockets for incoming task submissions
   */
  chi::TaskResume ClientRecv(hipc::FullPtr<ClientRecvTask> task, chi::RunContext &rctx);

  /**
   * Handle ClientSend - Send completed task outputs to ZMQ clients
   * Polls net_queue_ kClientSendTcp/kClientSendIpc priorities
   */
  chi::TaskResume ClientSend(hipc::FullPtr<ClientSendTask> task, chi::RunContext &rctx);

  /**
   * Handle WreapDeadIpcs - Periodic task to reap shared memory from dead processes
   * Calls IpcManager::WreapDeadIpcs() to clean up orphaned shared memory segments
   * Returns TaskResume for consistency with other methods called from Run
   */
  chi::TaskResume WreapDeadIpcs(hipc::FullPtr<WreapDeadIpcsTask> task, chi::RunContext &rctx);

  /**
   * Handle Monitor - Collect and return worker statistics
   * Iterates through all workers and collects their current statistics
   * Returns serialized statistics in JSON format
   */
  chi::TaskResume Monitor(hipc::FullPtr<MonitorTask> task, chi::RunContext &rctx);

  /**
   * Handle RegisterMemory - Register client shared memory with runtime
   * Called by SHM-mode clients after IncreaseMemory() to tell the runtime
   * to attach to the new shared memory segment
   */
  chi::TaskResume RegisterMemory(hipc::FullPtr<RegisterMemoryTask> task, chi::RunContext &rctx);

  /**
   * Handle RestartContainers - Re-create pools from saved restart configs
   * Reads conf_dir/restart/ directory and re-creates pools from saved YAML
   */
  chi::TaskResume RestartContainers(hipc::FullPtr<RestartContainersTask> task, chi::RunContext &rctx);

  /**
   * Handle AddNode - Register a new node with this runtime
   * Updates IpcManager's hostfile and calls Expand on all containers
   */
  chi::TaskResume AddNode(hipc::FullPtr<AddNodeTask> task, chi::RunContext &rctx);

  /**
   * Handle SubmitBatch - Submit a batch of tasks in a single RPC
   * Deserializes tasks from the batch and executes them in parallel
   * up to 32 tasks at a time, then co_awaits their completion
   * @param task The SubmitBatchTask containing serialized tasks
   * @param rctx Runtime context for the current worker
   */
  chi::TaskResume SubmitBatch(hipc::FullPtr<SubmitBatchTask> task, chi::RunContext &rctx);

  /**
   * Handle ChangeAddressTable - Update ContainerId->NodeId mapping
   * Writes WAL entry and updates pool manager's address table
   */
  chi::TaskResume ChangeAddressTable(hipc::FullPtr<ChangeAddressTableTask> task, chi::RunContext &rctx);

  /**
   * Handle MigrateContainers - Orchestrate container migration
   * Processes each MigrateInfo entry and broadcasts address table changes
   */
  chi::TaskResume MigrateContainers(hipc::FullPtr<MigrateContainersTask> task, chi::RunContext &rctx);

  /**
   * Handle Heartbeat - Liveness probe, just returns success
   */
  chi::TaskResume Heartbeat(hipc::FullPtr<HeartbeatTask> task, chi::RunContext &rctx);

  /**
   * Handle HeartbeatProbe - Periodic SWIM failure detector
   * Sends direct probes, escalates to indirect probes, manages suspicion
   */
  chi::TaskResume HeartbeatProbe(hipc::FullPtr<HeartbeatProbeTask> task, chi::RunContext &rctx);

  /**
   * Handle ProbeRequest - Indirect probe on behalf of another node
   * Probes target node and returns result to requester
   */
  chi::TaskResume ProbeRequest(hipc::FullPtr<ProbeRequestTask> task, chi::RunContext &rctx);

  /**
   * Handle RecoverContainers - Recreate containers from dead nodes
   * All nodes update address_map_, only dest node creates the container
   */
  chi::TaskResume RecoverContainers(hipc::FullPtr<RecoverContainersTask> task, chi::RunContext &rctx);

  /**
   * Helper: Receive task inputs from remote node
   */
  void RecvIn(hipc::FullPtr<RecvTask> task, chi::LoadTaskArchive& archive, hshm::lbm::Transport* lbm_transport);

  /**
   * Helper: Receive task outputs from remote node
   */
  void RecvOut(hipc::FullPtr<RecvTask> task, chi::LoadTaskArchive& archive, hshm::lbm::Transport* lbm_transport);

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
                hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Deserialize task parameters into an existing task (IN or OUT based on archive mode)
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive &archive,
                hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   */
  hipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive &archive) override;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   */
  void LocalLoadTask(chi::u32 method, chi::LocalLoadTaskArchive &archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   */
  hipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::LocalLoadTaskArchive &archive) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveTask(chi::u32 method, chi::LocalSaveTaskArchive &archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  hipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, hipc::FullPtr<chi::Task> orig_task_ptr,
                                        bool deep) override;

  /**
   * Create a new task of the specified method type
   */
  hipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;

  /**
   * Aggregate a replica task into the origin task (for merging replica results)
   */
  void Aggregate(chi::u32 method,
                 hipc::FullPtr<chi::Task> origin_task_ptr,
                 hipc::FullPtr<chi::Task> replica_task_ptr) override;

  /**
   * Attempt to send a retried task to the given node
   * @param entry The retry entry containing the task
   * @param node_id The node to send to
   * @return true if send succeeded
   */
  bool RetrySendToNode(RetryEntry &entry, chi::u64 node_id);

  /**
   * Re-resolve target node for a retried task whose original target is dead.
   * Uses the task's pool_query_ to look up the current container-to-node
   * mapping from the address_map_, which may have been updated by recovery.
   * @param entry The retry entry to re-resolve
   * @return New node ID, or 0 if re-resolution failed
   */
  chi::u64 RerouteRetryEntry(RetryEntry &entry);

  /**
   * Process retry queues: retry sends to revived nodes, re-route via
   * recovery address map updates, timeout stale entries
   */
  void ProcessRetryQueues();

  /**
   * Scan send_map_ for tasks waiting on dead nodes and time them out
   */
  void ScanSendMapTimeouts();

private:
  /**
   * Initiate runtime shutdown sequence
   */
  void InitiateShutdown(chi::u32 grace_period_ms);

  // Retry queues for tasks that failed to send due to dead nodes
  std::deque<RetryEntry> send_in_retry_;
  std::deque<RetryEntry> send_out_retry_;

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

  static constexpr float kDirectProbeTimeoutSec = 5.0f;
  static constexpr float kIndirectProbeTimeoutSec = 3.0f;
  static constexpr size_t kIndirectProbeHelpers = 3;
  static constexpr float kSuspicionTimeoutSec = 10.0f;

  // Recovery state
  std::vector<chi::RecoveryAssignment> ComputeRecoveryPlan(chi::u64 dead_node_id);
  void TriggerRecovery(chi::u64 dead_node_id);
  std::unordered_set<chi::u64> recovery_initiated_;
};

} // namespace chimaera::admin

#endif // ADMIN_RUNTIME_H_