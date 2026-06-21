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

#include <clio_runtime/clio_runtime.h>

#include "clio_ctp/data_structures/serialization/global_serialize.h"

#include "admin_tasks.h"

/**
 * Client API for Admin ChiMod
 *
 * Critical ChiMod for managing ChiPools and runtime lifecycle.
 * Provides methods for external programs to create/destroy pools and stop
 * runtime.
 */

namespace clio::run::admin {

class Client : public clio::run::ContainerClient {
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
  explicit Client(const clio::run::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create the Admin container (asynchronous)
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the admin pool (user-provided)
   * @param custom_pool_id Explicit pool ID for the pool being created
   */
  clio::run::Future<CreateTask> AsyncCreate(const clio::run::PoolQuery& pool_query,
                                      const std::string& pool_name,
                                      const clio::run::PoolId& custom_pool_id) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate CreateTask for admin container creation
    // Note: Admin uses BaseCreateTask pattern, not GetOrCreatePoolTask
    // The custom_pool_id is the ID for the pool being created (not the task
    // pool) Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query, "", pool_name,
        custom_pool_id, this);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Destroy an existing ChiPool (asynchronous)
   */
  clio::run::Future<DestroyPoolTask> AsyncDestroyPool(
      const clio::run::PoolQuery& pool_query, clio::run::PoolId target_pool_id,
      clio::run::u32 destruction_flags = 0) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate DestroyPoolTask
    auto task = ipc_manager->NewTask<DestroyPoolTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, target_pool_id,
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
  clio::run::Future<SendTask> AsyncSendPoll(const clio::run::PoolQuery& pool_query,
                                      clio::run::u32 transfer_flags = 0,
                                      double period_us = 25) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate SendTask for polling
    auto task = ipc_manager->NewTask<SendTask>(clio::run::CreateTaskId(), pool_id_,
                                               pool_query, transfer_flags);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
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
  clio::run::Future<RecvTask> AsyncRecv(const clio::run::PoolQuery& pool_query,
                                  clio::run::u32 transfer_flags = 0,
                                  double period_us = 25) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate RecvTask
    auto task = ipc_manager->NewTask<RecvTask>(clio::run::CreateTaskId(), pool_id_,
                                               pool_query, transfer_flags);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Flush administrative operations (asynchronous)
   */
  clio::run::Future<FlushTask> AsyncFlush(const clio::run::PoolQuery& pool_query) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate FlushTask
    auto task = ipc_manager->NewTask<FlushTask>(clio::run::CreateTaskId(), pool_id_,
                                                pool_query);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Stop the entire CLIO Runtime runtime (asynchronous)
   */
  clio::run::Future<StopRuntimeTask> AsyncStopRuntime(
      const clio::run::PoolQuery& pool_query, clio::run::u32 shutdown_flags = 0,
      clio::run::u32 grace_period_ms = 5000) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate StopRuntimeTask
    auto task = ipc_manager->NewTask<StopRuntimeTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, shutdown_flags,
        grace_period_ms);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Compose - Create a pool from a PoolConfig (asynchronous)
   * @param pool_config Configuration for the pool to create
   * @return Future for the compose task
   */
  clio::run::Future<ComposeTask<clio::run::PoolConfig>> AsyncCompose(
      const clio::run::PoolConfig& pool_config) {
    auto* ipc_manager = CLIO_IPC;

    // Create ComposeTask with PoolConfig passed directly to constructor
    auto task_ptr =
        ipc_manager->NewTask<clio::run::admin::ComposeTask<clio::run::PoolConfig>>(
            clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_config.pool_query_,
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
  clio::run::Future<ClientConnectTask> AsyncClientConnect(
      const clio::run::PoolQuery& pool_query, double period_us = 5000) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<ClientConnectTask>(clio::run::CreateTaskId(),
                                                        pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
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
  clio::run::Future<ClientRecvTask> AsyncClientRecv(const clio::run::PoolQuery& pool_query,
                                              double period_us = 100) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<ClientRecvTask>(clio::run::CreateTaskId(),
                                                     pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
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
  clio::run::Future<ClientSendTask> AsyncClientSend(const clio::run::PoolQuery& pool_query,
                                              double period_us = 100) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<ClientSendTask>(clio::run::CreateTaskId(),
                                                     pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
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
  clio::run::Future<WreapDeadIpcsTask> AsyncWreapDeadIpcs(
      const clio::run::PoolQuery& pool_query, double period_us = 1000000) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate WreapDeadIpcsTask
    auto task = ipc_manager->NewTask<WreapDeadIpcsTask>(clio::run::CreateTaskId(),
                                                        pool_id_, pool_query);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Unified Monitor - Query any chimod for runtime state (asynchronous)
   * All chimods implement kMonitor:9 with this task type.
   *
   * @param pool_query Query for routing this task
   * @param query Free-form query string
   * @return Future for MonitorTask with results
   */
  clio::run::Future<MonitorTask> AsyncMonitor(const clio::run::PoolQuery& pool_query,
                                        const std::string& query) {
    auto* ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<MonitorTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, query);
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
  clio::run::Future<SubmitBatchTask> AsyncSubmitBatch(
      const clio::run::PoolQuery& pool_query, const TaskBatch& batch) {
    auto* ipc_manager = CLIO_IPC;

    // Allocate SubmitBatchTask with batch data
    auto task = ipc_manager->NewTask<SubmitBatchTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, batch);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }
  /**
   * RegisterMemory - Tell runtime to attach to a client shared memory segment
   * @param pool_query Pool routing information
   * @param alloc_id Allocator ID (major=pid, minor=index) to register
   * @return Future for RegisterMemoryTask
   */
  clio::run::Future<RegisterMemoryTask> AsyncRegisterMemory(
      const clio::run::PoolQuery& pool_query, const ctp::ipc::AllocatorId& alloc_id) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<RegisterMemoryTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, alloc_id);

    return clio::run::IpcCpu2CpuZmq::ClientSend(ipc_manager, task, clio::run::IpcMode::kTcp);
  }
  /**
   * RestartContainers - Re-create pools from saved restart configs
   * @param pool_query Pool routing information
   * @return Future for the RestartContainers task
   */
  clio::run::Future<RestartContainersTask> AsyncRestartContainers(
      const clio::run::PoolQuery& pool_query) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<RestartContainersTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * ListContainers - Enumerate active pools/containers in the local daemon.
   * @param pool_query Pool routing information (use Dynamic/local).
   * @return Future for the ListContainers task; on completion the task's
   *         pool_names_ / pool_ids_ vectors hold the active pools.
   */
  clio::run::Future<ListContainersTask> AsyncListContainers(
      const clio::run::PoolQuery& pool_query) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<ListContainersTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * AddNode - Register a new node with all nodes in the cluster
   * @param pool_query Pool routing (use Broadcast to reach all nodes)
   * @param new_node_ip IP address of the new node
   * @param new_node_port Port of the new node's runtime
   * @return Future for the AddNode task
   */
  clio::run::Future<AddNodeTask> AsyncAddNode(const clio::run::PoolQuery& pool_query,
                                        const std::string& new_node_ip,
                                        clio::run::u32 new_node_port) {
    auto* ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<AddNodeTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query,
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
  clio::run::Future<ChangeAddressTableTask> AsyncChangeAddressTable(
      const clio::run::PoolQuery& pool_query,
      const clio::run::PoolId& target_pool_id,
      clio::run::ContainerId container_id,
      clio::run::u32 new_node_id) {
    auto* ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<ChangeAddressTableTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query,
        target_pool_id, container_id, new_node_id);
    return ipc_manager->Send(task);
  }

  /**
   * Heartbeat - Liveness probe to a specific node
   * @param pool_query Pool routing (use Physical(node_id) to target a node)
   * @return Future for the HeartbeatTask
   */
  clio::run::Future<HeartbeatTask> AsyncHeartbeat(const clio::run::PoolQuery& pool_query) {
    auto* ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<HeartbeatTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);
    return ipc_manager->Send(task);
  }

  /**
   * HeartbeatProbe - Periodic SWIM failure detector
   * @param pool_query Pool routing (use Local())
   * @param period_us Period in microseconds (default 2000000us = 2s)
   * @return Future for the HeartbeatProbeTask
   */
  clio::run::Future<HeartbeatProbeTask> AsyncHeartbeatProbe(
      const clio::run::PoolQuery& pool_query, double period_us = 2000000) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<HeartbeatProbeTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
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
  clio::run::Future<ProbeRequestTask> AsyncProbeRequest(
      const clio::run::PoolQuery& pool_query, clio::run::u64 target_node_id) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<ProbeRequestTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, target_node_id);

    return ipc_manager->Send(task);
  }

  /**
   * MigrateContainers - Orchestrate container migration
   * @param pool_query Pool routing
   * @param migrations Vector of MigrateInfo describing migrations to perform
   * @return Future for the MigrateContainers task
   */
  clio::run::Future<MigrateContainersTask> AsyncMigrateContainers(
      const clio::run::PoolQuery& pool_query,
      const std::vector<clio::run::MigrateInfo>& migrations) {
    auto* ipc_manager = CLIO_IPC;
    // Serialize migrations using GlobalSerialize
    std::vector<char> buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(buf);
      ar(migrations);
      ar.Finalize();
    }
    auto task = ipc_manager->NewTask<MigrateContainersTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, std::string(buf.begin(), buf.end()));
    return ipc_manager->Send(task);
  }
  /**
   * RecoverContainers - Broadcast recovery plan to all surviving nodes
   * @param pool_query Pool routing (typically Broadcast)
   * @param assignments Recovery assignments computed by leader
   * @param dead_node_id ID of the dead node being recovered
   * @return Future for the RecoverContainers task
   */
  clio::run::Future<RecoverContainersTask> AsyncRecoverContainers(
      const clio::run::PoolQuery& pool_query,
      const std::vector<clio::run::RecoveryAssignment>& assignments,
      clio::run::u64 dead_node_id) {
    auto* ipc_manager = CLIO_IPC;
    std::vector<char> buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(buf);
      ar(assignments);
      ar.Finalize();
    }
    auto task = ipc_manager->NewTask<RecoverContainersTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, std::string(buf.begin(), buf.end()), dead_node_id);
    return ipc_manager->Send(task);
  }
  /**
   * SystemMonitor - Periodic system resource utilization sampling
   * @param pool_query Pool routing (use Local())
   * @param period_us Period in microseconds (default 1000000us = 1s)
   * @return Future for the SystemMonitorTask
   */
  clio::run::Future<SystemMonitorTask> AsyncSystemMonitor(
      const clio::run::PoolQuery &pool_query, double period_us = 1000000) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<SystemMonitorTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query);
    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }
    return ipc_manager->Send(task);
  }

  /**
   * AnnounceShutdown - Broadcast that a node is shutting down
   * Receiving nodes mark the departing node as dead immediately.
   * @param pool_query Pool routing (use Broadcast())
   * @param shutting_down_node_id Node ID that is shutting down
   * @return Future for the AnnounceShutdownTask
   */
  clio::run::Future<AnnounceShutdownTask> AsyncAnnounceShutdown(
      const clio::run::PoolQuery &pool_query, clio::run::u64 shutting_down_node_id) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<AnnounceShutdownTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, shutting_down_node_id);
    return ipc_manager->Send(task);
  }
};

}  // namespace clio::run::admin

#endif  // ADMIN_CLIENT_H_