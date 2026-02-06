#ifndef ADMIN_CLIENT_H_
#define ADMIN_CLIENT_H_

#include <chimaera/chimaera.h>

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
    HLOG(kWarning, "AdminClient: Default constructor called - pool_id_ will be PoolId(0,0)");
  }

  /**
   * Constructor with pool ID
   */
  explicit Client(const chi::PoolId& pool_id) {
    HLOG(kInfo, "AdminClient: Constructor called with pool_id={}", pool_id);
    Init(pool_id);
    HLOG(kInfo, "AdminClient: After Init, pool_id_={}", pool_id_);
  }

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
    // The custom_pool_id is the ID for the pool being created (not the task pool)
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(chi::CreateTaskId(),
                                                 chi::kAdminPoolId, pool_query, "", pool_name, custom_pool_id, this);

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Destroy an existing ChiPool (asynchronous)
   */
  chi::Future<DestroyPoolTask> AsyncDestroyPool(const chi::PoolQuery& pool_query,
      chi::PoolId target_pool_id, chi::u32 destruction_flags = 0) {
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
    auto task = ipc_manager->NewTask<SendTask>(
        chi::CreateTaskId(), pool_id_, pool_query, transfer_flags);

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
   * Can be used for both SerializeIn (receiving inputs) and SerializeOut (receiving outputs)
   */
  chi::Future<RecvTask> AsyncRecv(const chi::PoolQuery& pool_query,
      chi::u32 transfer_flags = 0,
      double period_us = 25) {
    auto* ipc_manager = CHI_IPC;

    // Allocate RecvTask
    auto task = ipc_manager->NewTask<RecvTask>(
        chi::CreateTaskId(), pool_id_, pool_query, transfer_flags);

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
  chi::Future<StopRuntimeTask> AsyncStopRuntime(const chi::PoolQuery& pool_query,
      chi::u32 shutdown_flags = 0, chi::u32 grace_period_ms = 5000) {
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
    auto task_ptr = ipc_manager->NewTask<chimaera::admin::ComposeTask<chi::PoolConfig>>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,
        pool_config.pool_query_,
        pool_config
    );

    // Submit to runtime and return Future
    return ipc_manager->Send(task_ptr);
  }

  /**
   * Heartbeat - Check if runtime is alive (asynchronous)
   * Polls for ZMQ heartbeat requests and responds
   * @param pool_query Pool routing information
   * @param period_us Period in microseconds (default 5000us = 5ms, 0 = one-shot)
   * @return Future for the heartbeat task
   */
  chi::Future<HeartbeatTask> AsyncHeartbeat(const chi::PoolQuery& pool_query,
      double period_us = 5000) {
    auto* ipc_manager = CHI_IPC;

    // Allocate HeartbeatTask
    auto task = ipc_manager->NewTask<HeartbeatTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * WreapDeadIpcs - Periodic task to reap shared memory from dead processes
   * Calls IpcManager::WreapDeadIpcs() to clean up orphaned shared memory segments
   * @param pool_query Pool routing information
   * @param period_us Period in microseconds (default 1000000us = 1s)
   * @return Future for the WreapDeadIpcs task
   */
  chi::Future<WreapDeadIpcsTask> AsyncWreapDeadIpcs(const chi::PoolQuery& pool_query,
      double period_us = 1000000) {
    auto* ipc_manager = CHI_IPC;

    // Allocate WreapDeadIpcsTask
    auto task = ipc_manager->NewTask<WreapDeadIpcsTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

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
   * @param period_us Period in microseconds for periodic monitoring (0 = one-shot)
   * @return Future for MonitorTask that will contain worker statistics
   */
  chi::Future<MonitorTask> AsyncMonitor(const chi::PoolQuery& pool_query,
      double period_us = 0) {
    auto* ipc_manager = CHI_IPC;

    HLOG(kInfo, "AsyncMonitor: Creating MonitorTask with pool_id_={}", pool_id_);
    // Allocate MonitorTask
    auto task = ipc_manager->NewTask<MonitorTask>(
        chi::CreateTaskId(), pool_id_, pool_query);
    HLOG(kInfo, "AsyncMonitor: Task pool_id={}, method={}", task->pool_id_, task->method_);

    HLOG(kInfo, "AsyncMonitor: Task created, IsNull={}", task.IsNull());

    // Set task as periodic if period is specified
    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    HLOG(kInfo, "AsyncMonitor: Sending task");
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
  chi::Future<SubmitBatchTask> AsyncSubmitBatch(const chi::PoolQuery& pool_query,
                                                 const TaskBatch& batch) {
    auto* ipc_manager = CHI_IPC;

    HLOG(kInfo, "AsyncSubmitBatch: Creating SubmitBatchTask with {} tasks",
         batch.GetTaskCount());

    // Allocate SubmitBatchTask with batch data
    auto task = ipc_manager->NewTask<SubmitBatchTask>(
        chi::CreateTaskId(), pool_id_, pool_query, batch);

    HLOG(kInfo, "AsyncSubmitBatch: Task created, sending to runtime");

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

};

}  // namespace chimaera::admin

#endif  // ADMIN_CLIENT_H_