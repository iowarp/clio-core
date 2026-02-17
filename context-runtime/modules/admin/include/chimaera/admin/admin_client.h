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

#ifndef ADMIN_CLIENT_H_
#define ADMIN_CLIENT_H_

#include <chimaera/chimaera.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>

#include "admin_tasks.h"

/**
 * Client API for Admin ChiMod
 *
 * Critical ChiMod for managing ChiPools and runtime lifecycle.
 * Provides methods for external programs to create/destroy pools and stop
 * runtime.
 */

namespace chimaera::admin {

class Client : public chi::ContainerClient {
 public:
  /**
   * Default constructor
   */
  Client() {
    HLOG(kWarning,
         "AdminClient: Default constructor called - pool_id_ will be "
         "PoolId(0,0)");
  }

  /**
   * Constructor with pool ID
   */
  explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create the Admin container (asynchronous)
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the admin pool (user-provided)
   * @param custom_pool_id Explicit pool ID for the pool being created
   */
  chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                      const std::string& pool_name,
                                      const chi::PoolId& custom_pool_id) {
    auto* ipc_manager = CHI_IPC;

    // Allocate CreateTask for admin container creation
    // Note: Admin uses BaseCreateTask pattern, not GetOrCreatePoolTask
    // The custom_pool_id is the ID for the pool being created (not the task
    // pool) Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, pool_query, "", pool_name,
        custom_pool_id, this);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Destroy an existing ChiPool (asynchronous)
   */
  chi::Future<DestroyPoolTask> AsyncDestroyPool(
      const chi::PoolQuery& pool_query, chi::PoolId target_pool_id,
      chi::u32 destruction_flags = 0) {
    auto* ipc_manager = CHI_IPC;

    // Allocate DestroyPoolTask
    auto task = ipc_manager->NewTask<DestroyPoolTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_pool_id,
        destruction_flags);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Create a periodic SendTask for polling the network queue
   * This task polls net_queue_ and processes send operations
   * @param pool_query Pool query for routing
   * @param transfer_flags Transfer flags
   * @param period_us Period in microseconds (default 25us)
   * @return Future for the periodic SendTask
   */
  chi::Future<SendTask> AsyncSendPoll(const chi::PoolQuery& pool_query,
                                      chi::u32 transfer_flags = 0,
                                      double period_us = 25) {
    auto* ipc_manager = CHI_IPC;

    // Allocate SendTask for polling
    auto task = ipc_manager->NewTask<SendTask>(chi::CreateTaskId(), pool_id_,
                                               pool_query, transfer_flags);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Receive tasks from network (asynchronous)
   * Can be used for both SerializeIn (receiving inputs) and SerializeOut
   * (receiving outputs)
   */
  chi::Future<RecvTask> AsyncRecv(const chi::PoolQuery& pool_query,
                                  chi::u32 transfer_flags = 0,
                                  double period_us = 25) {
    auto* ipc_manager = CHI_IPC;

    // Allocate RecvTask
    auto task = ipc_manager->NewTask<RecvTask>(chi::CreateTaskId(), pool_id_,
                                               pool_query, transfer_flags);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Flush administrative operations (asynchronous)
   */
  chi::Future<FlushTask> AsyncFlush(const chi::PoolQuery& pool_query) {
    auto* ipc_manager = CHI_IPC;

    // Allocate FlushTask
    auto task = ipc_manager->NewTask<FlushTask>(chi::CreateTaskId(), pool_id_,
                                                pool_query);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Stop the entire Chimaera runtime (asynchronous)
   */
  chi::Future<StopRuntimeTask> AsyncStopRuntime(
      const chi::PoolQuery& pool_query, chi::u32 shutdown_flags = 0,
      chi::u32 grace_period_ms = 5000) {
    auto* ipc_manager = CHI_IPC;

    // Allocate StopRuntimeTask
    auto task = ipc_manager->NewTask<StopRuntimeTask>(
        chi::CreateTaskId(), pool_id_, pool_query, shutdown_flags,
        grace_period_ms);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Compose - Create a pool from a PoolConfig (asynchronous)
   * @param pool_config Configuration for the pool to create
   * @return Future for the compose task
   */
  chi::Future<ComposeTask<chi::PoolConfig>> AsyncCompose(
      const chi::PoolConfig& pool_config) {
    auto* ipc_manager = CHI_IPC;

    // Create ComposeTask with PoolConfig passed directly to constructor
    auto task_ptr =
        ipc_manager->NewTask<chimaera::admin::ComposeTask<chi::PoolConfig>>(
            chi::CreateTaskId(), chi::kAdminPoolId, pool_config.pool_query_,
            pool_config);

    // Submit to runtime and return Future
    return ipc_manager->Send(task_ptr);
  }

  /**
   * ClientConnect - Check if runtime is alive (asynchronous)
   * Polls for ZMQ connect requests and responds
   * @param pool_query Pool routing information
   * @param period_us Period in microseconds (default 5000us = 5ms, 0 =
   * one-shot)
   * @return Future for the connect task
   */
  chi::Future<ClientConnectTask> AsyncClientConnect(
      const chi::PoolQuery& pool_query, double period_us = 5000) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ClientConnectTask>(chi::CreateTaskId(),
                                                        pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * ClientRecv - Receive tasks from ZMQ clients (asynchronous, periodic)
   * Polls ZMQ ROUTER sockets for incoming client task submissions
   * @param pool_query Pool routing information
   * @param period_us Period in microseconds (default 100us)
   * @return Future for the client recv task
   */
  chi::Future<ClientRecvTask> AsyncClientRecv(const chi::PoolQuery& pool_query,
                                              double period_us = 100) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ClientRecvTask>(chi::CreateTaskId(),
                                                     pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * ClientSend - Send completed task outputs to ZMQ clients (asynchronous, periodic)
   * Polls net_queue_ kClientSendTcp/kClientSendIpc priorities
   * @param pool_query Pool routing information
   * @param period_us Period in microseconds (default 100us)
   * @return Future for the client send task
   */
  chi::Future<ClientSendTask> AsyncClientSend(const chi::PoolQuery& pool_query,
                                              double period_us = 100) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ClientSendTask>(chi::CreateTaskId(),
                                                     pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * WreapDeadIpcs - Periodic task to reap shared memory from dead processes
   * Calls IpcManager::WreapDeadIpcs() to clean up orphaned shared memory
   * segments
   * @param pool_query Pool routing information
   * @param period_us Period in microseconds (default 1000000us = 1s)
   * @return Future for the WreapDeadIpcs task
   */
  chi::Future<WreapDeadIpcsTask> AsyncWreapDeadIpcs(
      const chi::PoolQuery& pool_query, double period_us = 1000000) {
    auto* ipc_manager = CHI_IPC;

    // Allocate WreapDeadIpcsTask
    auto task = ipc_manager->NewTask<WreapDeadIpcsTask>(chi::CreateTaskId(),
                                                        pool_id_, pool_query);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Monitor worker statistics (asynchronous)
   * Collects current statistics from all workers including:
   * - Number of queued, blocked, and periodic tasks
   * - Worker idle status and suspend periods
   *
   * @param pool_query Query for routing this task
   * @param period_us Period in microseconds for periodic monitoring (0 =
   * one-shot)
   * @return Future for MonitorTask that will contain worker statistics
   */
  chi::Future<MonitorTask> AsyncMonitor(const chi::PoolQuery& pool_query,
                                        double period_us = 0) {
    auto* ipc_manager = CHI_IPC;

    // Allocate MonitorTask
    auto task = ipc_manager->NewTask<MonitorTask>(chi::CreateTaskId(), pool_id_,
                                                  pool_query);
    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Submit a batch of tasks in a single RPC (asynchronous)
   * Allows efficient submission of multiple tasks with minimal network overhead
   *
   * @param pool_query Query for routing this task
   * @param batch TaskBatch containing the tasks to submit
   * @return Future for SubmitBatchTask with completion results
   */
  chi::Future<SubmitBatchTask> AsyncSubmitBatch(
      const chi::PoolQuery& pool_query, const TaskBatch& batch) {
    auto* ipc_manager = CHI_IPC;

    // Allocate SubmitBatchTask with batch data
    auto task = ipc_manager->NewTask<SubmitBatchTask>(
        chi::CreateTaskId(), pool_id_, pool_query, batch);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }
  /**
   * RegisterMemory - Tell runtime to attach to a client shared memory segment
   * @param pool_query Pool routing information
   * @param alloc_id Allocator ID (major=pid, minor=index) to register
   * @return Future for RegisterMemoryTask
   */
  chi::Future<RegisterMemoryTask> AsyncRegisterMemory(
      const chi::PoolQuery& pool_query, const hipc::AllocatorId& alloc_id) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<RegisterMemoryTask>(
        chi::CreateTaskId(), pool_id_, pool_query, alloc_id);

    return ipc_manager->SendZmq(task, chi::IpcMode::kTcp);
  }
  /**
   * RestartContainers - Re-create pools from saved restart configs
   * @param pool_query Pool routing information
   * @return Future for the RestartContainers task
   */
  chi::Future<RestartContainersTask> AsyncRestartContainers(
      const chi::PoolQuery& pool_query) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<RestartContainersTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * AddNode - Register a new node with all nodes in the cluster
   * @param pool_query Pool routing (use Broadcast to reach all nodes)
   * @param new_node_ip IP address of the new node
   * @param new_node_port Port of the new node's runtime
   * @return Future for the AddNode task
   */
  chi::Future<AddNodeTask> AsyncAddNode(const chi::PoolQuery& pool_query,
                                        const std::string& new_node_ip,
                                        chi::u32 new_node_port) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<AddNodeTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        new_node_ip, new_node_port);
    return ipc_manager->Send(task);
  }

  /**
   * ChangeAddressTable - Update ContainerId->NodeId mapping on nodes
   * @param pool_query Pool routing (use Broadcast to reach all nodes)
   * @param target_pool_id Pool whose address table to update
   * @param container_id Container being remapped
   * @param new_node_id New node ID for the container
   * @return Future for the ChangeAddressTable task
   */
  chi::Future<ChangeAddressTableTask> AsyncChangeAddressTable(
      const chi::PoolQuery& pool_query,
      const chi::PoolId& target_pool_id,
      chi::ContainerId container_id,
      chi::u32 new_node_id) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<ChangeAddressTableTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        target_pool_id, container_id, new_node_id);
    return ipc_manager->Send(task);
  }

  /**
   * Heartbeat - Liveness probe to a specific node
   * @param pool_query Pool routing (use Physical(node_id) to target a node)
   * @return Future for the HeartbeatTask
   */
  chi::Future<HeartbeatTask> AsyncHeartbeat(const chi::PoolQuery& pool_query) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<HeartbeatTask>(
        chi::CreateTaskId(), pool_id_, pool_query);
    return ipc_manager->Send(task);
  }

  /**
   * HeartbeatProbe - Periodic SWIM failure detector
   * @param pool_query Pool routing (use Local())
   * @param period_us Period in microseconds (default 2000000us = 2s)
   * @return Future for the HeartbeatProbeTask
   */
  chi::Future<HeartbeatProbeTask> AsyncHeartbeatProbe(
      const chi::PoolQuery& pool_query, double period_us = 2000000) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<HeartbeatProbeTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * ProbeRequest - Ask a helper node to probe a target on our behalf
   * @param pool_query Pool routing (use Physical(helper_node_id))
   * @param target_node_id Node to probe
   * @return Future for the ProbeRequestTask
   */
  chi::Future<ProbeRequestTask> AsyncProbeRequest(
      const chi::PoolQuery& pool_query, chi::u64 target_node_id) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ProbeRequestTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_node_id);

    return ipc_manager->Send(task);
  }

  /**
   * MigrateContainers - Orchestrate container migration
   * @param pool_query Pool routing
   * @param migrations Vector of MigrateInfo describing migrations to perform
   * @return Future for the MigrateContainers task
   */
  chi::Future<MigrateContainersTask> AsyncMigrateContainers(
      const chi::PoolQuery& pool_query,
      const std::vector<chi::MigrateInfo>& migrations) {
    auto* ipc_manager = CHI_IPC;
    // Serialize migrations using cereal binary archive
    std::ostringstream os;
    {
      cereal::BinaryOutputArchive ar(os);
      ar(migrations);
    }
    auto task = ipc_manager->NewTask<MigrateContainersTask>(
        chi::CreateTaskId(), pool_id_, pool_query, os.str());
    return ipc_manager->Send(task);
  }
  /**
   * RecoverContainers - Broadcast recovery plan to all surviving nodes
   * @param pool_query Pool routing (typically Broadcast)
   * @param assignments Recovery assignments computed by leader
   * @param dead_node_id ID of the dead node being recovered
   * @return Future for the RecoverContainers task
   */
  chi::Future<RecoverContainersTask> AsyncRecoverContainers(
      const chi::PoolQuery& pool_query,
      const std::vector<chi::RecoveryAssignment>& assignments,
      chi::u64 dead_node_id) {
    auto* ipc_manager = CHI_IPC;
    std::ostringstream os;
    {
      cereal::BinaryOutputArchive ar(os);
      ar(assignments);
    }
    auto task = ipc_manager->NewTask<RecoverContainersTask>(
        chi::CreateTaskId(), pool_id_, pool_query, os.str(), dead_node_id);
    return ipc_manager->Send(task);
  }
};

}  // namespace chimaera::admin

#endif  // ADMIN_CLIENT_H_