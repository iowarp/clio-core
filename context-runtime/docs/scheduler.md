# IOWarp Scheduler Development Guide

## Overview

The Chimaera runtime uses a pluggable scheduler architecture to control how tasks are mapped to workers and how workers are organized. This document explains how to build custom schedulers for the IOWarp runtime.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Scheduler Interface](#scheduler-interface)
3. [Worker Lifecycle](#worker-lifecycle)
4. [Implementing a Custom Scheduler](#implementing-a-custom-scheduler)
5. [DefaultScheduler Example](#defaultscheduler-example)
6. [Best Practices](#best-practices)
7. [Integration Points](#integration-points)

## Architecture Overview

### Component Responsibilities

The IOWarp runtime separates concerns across three main components:

- **ConfigManager**: Manages configuration (number of threads, queue depth, etc.)
- **WorkOrchestrator**: Creates workers, spawns threads, assigns lanes to workers (1:1 mapping for all workers)
- **Scheduler**: Decides worker partitioning, task-to-worker mapping, and load balancing
- **IpcManager**: Manages shared memory, queues, and provides task routing infrastructure

### Data Flow

```
┌─────────────────┐
│  ConfigManager  │──→ num_threads, queue_depth
└─────────────────┘
         │
         ↓
┌─────────────────┐
│ WorkOrchestrator│──→ Creates num_threads + 1 workers
└─────────────────┘
         │
         ↓
┌─────────────────┐
│   Scheduler     │──→ Tracks worker groups for routing decisions
└─────────────────┘    Updates IpcManager with scheduler queue count
         │
         ↓
┌─────────────────┐
│ WorkOrchestrator│──→ Maps ALL workers to lanes (1:1 mapping)
└─────────────────┘    Spawns OS threads for each worker
         │
         ↓
┌─────────────────┐
│   IpcManager    │──→ num_sched_queues used for client task mapping
└─────────────────┘
```

## Scheduler Interface

All schedulers must inherit from the `Scheduler` base class and implement the following methods:

### Required Methods

```cpp
class Scheduler {
public:
  virtual ~Scheduler() = default;

  // Partition workers into groups after WorkOrchestrator creates them
  virtual void DivideWorkers(WorkOrchestrator *work_orch) = 0;

  // Map tasks from clients to worker lanes
  virtual u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) = 0;

  // Map tasks from runtime workers to other workers
  virtual u32 RuntimeMapTask(Worker *worker, const Future<Task> &task) = 0;

  // Rebalance load across workers (called periodically by workers)
  virtual void RebalanceWorker(Worker *worker) = 0;

  // Adjust polling intervals for periodic tasks
  virtual void AdjustPolling(RunContext *run_ctx) = 0;
};
```

### Method Details

#### `DivideWorkers(WorkOrchestrator *work_orch)`

**Purpose**: Partition workers into functional groups after they've been created.

**Called**: Once during initialization, after WorkOrchestrator creates all workers but before threads are spawned.

**Responsibilities**:
- Access workers via `work_orch->GetWorker(worker_id)`
- Organize workers into scheduler-specific groups (e.g., task workers, network worker)
- **Update IpcManager** with the total worker count via `IpcManager::SetNumSchedQueues()`

**Important**: All workers are assigned lanes by `WorkOrchestrator::SpawnWorkerThreads()` using 1:1 mapping. The scheduler does NOT control lane assignment — it only tracks worker groups for routing decisions.

**Example**:
```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  u32 total_workers = work_orch->GetTotalWorkerCount();

  // Track workers: first N-1 are task workers, last is network
  for (u32 i = 0; i < total_workers - 1; ++i) {
    Worker *worker = work_orch->GetWorker(i);
    if (worker) {
      task_workers_.push_back(worker);
    }
  }

  // Last worker is network worker
  net_worker_ = work_orch->GetWorker(total_workers - 1);

  // IMPORTANT: Update IpcManager with worker count
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(total_workers);
  }
}
```

#### `ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task)`

**Purpose**: Determine which worker lane a task from a client should be assigned to.

**Called**: When clients submit tasks to the runtime.

**Responsibilities**:
- Return a lane ID in range `[0, num_sched_queues)`
- Use `ipc_manager->GetNumSchedQueues()` to get valid lane count
- Route special tasks (e.g., network Send/Recv) to the appropriate worker
- Common strategies: PID+TID hash, round-robin, locality-aware

**Example**:
```cpp
u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  if (num_lanes == 0) return 0;

  // Route network tasks (Send/Recv from admin pool) to last worker
  Task *task_ptr = task.get();
  if (task_ptr != nullptr && task_ptr->pool_id_ == chi::kAdminPoolId) {
    u32 method_id = task_ptr->method_;
    if (method_id == 14 || method_id == 15) {  // kSend or kRecv
      return num_lanes - 1;
    }
  }

  // PID+TID hash-based mapping for other tasks
  auto *sys_info = HSHM_SYSTEM_INFO;
  pid_t pid = sys_info->pid_;
  auto tid = HSHM_THREAD_MODEL->GetTid();

  size_t hash = std::hash<pid_t>{}(pid) ^ (std::hash<void*>{}(&tid) << 1);
  return static_cast<u32>(hash % num_lanes);
}
```

#### `RuntimeMapTask(Worker *worker, const Future<Task> &task)`

**Purpose**: Determine which worker should execute a task when routing from within the runtime.

**Called**: By `Worker::RouteLocal()` to decide whether a task should execute on the current worker or be forwarded to another.

**Responsibilities**:
- Return a worker ID for task execution
- Route periodic network tasks (Send/Recv) to the dedicated network worker
- For all other tasks, return the current worker's ID (no migration)

**Example**:
```cpp
u32 MyScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  // Route periodic network tasks to the network worker
  Task *task_ptr = task.get();
  if (task_ptr != nullptr && task_ptr->IsPeriodic()) {
    if (task_ptr->pool_id_ == chi::kAdminPoolId) {
      u32 method_id = task_ptr->method_;
      if (method_id == 14 || method_id == 15) {  // kSend or kRecv
        if (net_worker_ != nullptr) {
          return net_worker_->GetId();
        }
      }
    }
  }

  // All other tasks execute on the current worker
  return worker ? worker->GetId() : 0;
}
```

#### `RebalanceWorker(Worker *worker)`

**Purpose**: Balance load across workers by stealing or delegating tasks.

**Called**: Periodically by workers after processing tasks.

**Responsibilities**:
- Implement work stealing algorithms
- Migrate tasks between workers
- Optional — can be a no-op for simple schedulers

**Example**:
```cpp
void MyScheduler::RebalanceWorker(Worker *worker) {
  // Simple schedulers can leave this empty
  (void)worker;
}
```

#### `AdjustPolling(RunContext *run_ctx)`

**Purpose**: Adjust polling intervals for periodic tasks based on work done.

**Called**: After each execution of a periodic task.

**Responsibilities**:
- Modify `run_ctx->yield_time_us_` based on `run_ctx->did_work_`
- Implement adaptive polling (exponential backoff when idle)
- Reduce CPU usage for idle periodic tasks

**Example**:
```cpp
void MyScheduler::AdjustPolling(RunContext *run_ctx) {
  if (!run_ctx) return;

  const double kMaxPollingIntervalUs = 100000.0; // 100ms

  if (run_ctx->did_work_) {
    // Reset to original period when work is done
    run_ctx->yield_time_us_ = run_ctx->true_period_ns_ / 1000.0;
  } else {
    // Exponential backoff when idle
    double current = run_ctx->yield_time_us_;
    if (current <= 0.0) {
      current = run_ctx->true_period_ns_ / 1000.0;
    }
    run_ctx->yield_time_us_ = std::min(current * 2.0, kMaxPollingIntervalUs);
  }
}
```

## Worker Lifecycle

Understanding the worker lifecycle is crucial for scheduler implementation:

```
1. ConfigManager loads configuration (num_threads, queue_depth)
   ↓
2. WorkOrchestrator::Init()
   - Creates num_threads + 1 workers
   - Calls Scheduler::DivideWorkers()
   ↓
3. Scheduler::DivideWorkers()
   - Tracks workers into functional groups (task workers, network worker)
   - Updates IpcManager::SetNumSchedQueues()
   ↓
4. WorkOrchestrator::StartWorkers()
   - Calls SpawnWorkerThreads()
   - Maps ALL workers to lanes (1:1 mapping: worker i → lane i)
   - Spawns actual OS threads
   ↓
5. Workers run task processing loops
   - Process tasks from assigned lanes
   - Call Scheduler::RuntimeMapTask() for task routing in RouteLocal()
   - Call Scheduler::RebalanceWorker() periodically
```

## Implementing a Custom Scheduler

### Step 1: Create Header File

Create `context-runtime/include/chimaera/scheduler/my_scheduler.h`:

```cpp
#ifndef CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_MY_SCHEDULER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_MY_SCHEDULER_H_

#include <vector>
#include "chimaera/scheduler/scheduler.h"

namespace chi {

class MyScheduler : public Scheduler {
public:
  MyScheduler() : net_worker_(nullptr) {}
  ~MyScheduler() override = default;

  // Implement required interface methods
  void DivideWorkers(WorkOrchestrator *work_orch) override;
  u32 ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) override;
  u32 RuntimeMapTask(Worker *worker, const Future<Task> &task) override;
  void RebalanceWorker(Worker *worker) override;
  void AdjustPolling(RunContext *run_ctx) override;

private:
  // Your scheduler-specific state
  std::vector<Worker*> scheduler_workers_;
  Worker *net_worker_;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_SCHEDULER_MY_SCHEDULER_H_
```

### Step 2: Implement Methods

Create `context-runtime/src/scheduler/my_scheduler.cc`:

```cpp
#include "chimaera/scheduler/my_scheduler.h"
#include "chimaera/config_manager.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) return;

  u32 total_workers = work_orch->GetTotalWorkerCount();

  scheduler_workers_.clear();
  net_worker_ = nullptr;

  // Network worker is always the last worker
  net_worker_ = work_orch->GetWorker(total_workers - 1);

  // Scheduler workers are all workers except the last one
  u32 num_sched_workers = (total_workers == 1) ? 1 : (total_workers - 1);
  for (u32 i = 0; i < num_sched_workers; ++i) {
    Worker *worker = work_orch->GetWorker(i);
    if (worker) {
      scheduler_workers_.push_back(worker);
    }
  }

  // CRITICAL: Update IpcManager with the number of workers
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(total_workers);
  }
}

u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  if (num_lanes == 0) return 0;

  // Implement your mapping strategy here
  return 0; // Simple: always map to lane 0
}

u32 MyScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  return worker ? worker->GetId() : 0;
}

void MyScheduler::RebalanceWorker(Worker *worker) {
  (void)worker;
}

void MyScheduler::AdjustPolling(RunContext *run_ctx) {
  if (!run_ctx) return;
  // Implement adaptive polling or leave with default behavior
}

}  // namespace chi
```

### Step 3: Register Scheduler

Update `context-runtime/src/ipc_manager.cc` to create your scheduler:

```cpp
bool IpcManager::ServerInit() {
  // ... existing initialization code ...

  // Create scheduler based on configuration
  ConfigManager *config = CHI_CONFIG_MANAGER;
  std::string sched_name = config->GetLocalSched();

  if (sched_name == "my_scheduler") {
    scheduler_ = new MyScheduler();
  } else if (sched_name == "default") {
    scheduler_ = new DefaultScheduler();
  } else {
    HLOG(kError, "Unknown scheduler: {}", sched_name);
    return false;
  }

  return true;
}
```

### Step 4: Configure

Update your configuration file to use the new scheduler:

```yaml
runtime:
  local_sched: "my_scheduler"
  num_threads: 4
  queue_depth: 1024
```

## DefaultScheduler Example

The `DefaultScheduler` provides a reference implementation with these characteristics:

### Worker Partitioning
- Tracks all workers except the last as scheduler workers
- Last worker is designated as the network worker
- All workers get lanes assigned by WorkOrchestrator (1:1 mapping)
- `SetNumSchedQueues(total_workers)` includes all workers for client task mapping

### Task Mapping Strategy
- **Client Tasks**: PID+TID hash-based mapping for regular tasks
  - Ensures different processes/threads use different lanes
  - Network tasks (Send/Recv from admin pool, methods 14/15) are routed to the last worker (network worker)
- **Runtime Tasks**: Tasks execute on the current worker, except periodic Send/Recv tasks which are routed to the network worker

### Load Balancing
- No active rebalancing (simple design)
- Tasks processed by worker that picks them up

### Polling Adjustment
- Currently disabled (early return) to avoid hanging issues
- When enabled, implements exponential backoff for idle periodic tasks

### Code Reference

See implementation in:
- Header: `context-runtime/include/chimaera/scheduler/default_sched.h`
- Implementation: `context-runtime/src/scheduler/default_sched.cc`

## Best Practices

### 1. Always Update IpcManager in DivideWorkers

```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  // ... partition workers ...

  // CRITICAL: Update IpcManager with worker count
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(total_workers);
  }
}
```

**Why**: Clients use `GetNumSchedQueues()` to map tasks to lanes. If this doesn't match the actual number of workers/lanes, tasks will be mapped to non-existent or wrong workers.

### 2. Route Network Tasks to the Network Worker

Both `ClientMapTask` and `RuntimeMapTask` should route Send/Recv tasks (methods 14/15 from admin pool) to the dedicated network worker (last worker). This prevents network I/O from blocking task processing workers.

### 3. Validate Lane IDs

```cpp
u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  if (num_lanes == 0) return 0;

  u32 lane = ComputeLane(...);
  return lane % num_lanes;  // Ensure lane is in valid range
}
```

### 4. Handle Null Pointers

```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) return;
  // ... proceed ...
}

u32 MyScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  return worker ? worker->GetId() : 0;
}
```

### 5. Consider Thread Safety

If your scheduler maintains shared state accessed by multiple workers:
- Use atomic operations for counters
- Use mutexes for complex data structures
- Prefer lock-free designs when possible

### 6. Test with Different Configurations

Test your scheduler with various `num_threads` values:
- Single thread (num_threads = 1): single worker serves dual role
- Small (num_threads = 2-4)
- Large (num_threads = 16+)

## Integration Points

### Singletons and Macros

Access runtime components via global macros:

```cpp
// Configuration
ConfigManager *config = CHI_CONFIG_MANAGER;
u32 num_threads = config->GetNumThreads();

// IPC Manager
IpcManager *ipc = CHI_IPC;
u32 num_lanes = ipc->GetNumSchedQueues();

// System Info
auto *sys_info = HSHM_SYSTEM_INFO;
pid_t pid = sys_info->pid_;

// Thread Model
auto tid = HSHM_THREAD_MODEL->GetTid();
```

### Worker Access

Access workers through WorkOrchestrator:

```cpp
u32 total_workers = work_orch->GetTotalWorkerCount();
Worker *worker = work_orch->GetWorker(worker_id);

// Get worker properties
u32 id = worker->GetId();
TaskLane *lane = worker->GetLane();
```

### Logging

Use Hermes logging macros:

```cpp
HLOG(kInfo, "Scheduler initialized with {} workers", num_workers);
HLOG(kDebug, "Mapping task to lane {}", lane_id);
HLOG(kWarning, "Worker {} has empty queue", worker_id);
HLOG(kError, "Invalid configuration: {}", error_msg);
```

### Configuration Access

Read configuration values:

```cpp
ConfigManager *config = CHI_CONFIG_MANAGER;
u32 num_threads = config->GetNumThreads();
u32 queue_depth = config->GetQueueDepth();
std::string sched_name = config->GetLocalSched();
```

## Advanced Topics

### Work Stealing

Implement work stealing in `RebalanceWorker`:

```cpp
void MyScheduler::RebalanceWorker(Worker *worker) {
  TaskLane *my_lane = worker->GetLane();
  if (my_lane->Empty()) {
    for (Worker *victim : scheduler_workers_) {
      if (victim == worker) continue;

      TaskLane *victim_lane = victim->GetLane();
      if (!victim_lane->Empty()) {
        Future<Task> stolen_task;
        if (victim_lane->Pop(stolen_task)) {
          my_lane->Push(stolen_task);
          break;
        }
      }
    }
  }
}
```

### Locality-Aware Mapping

Map tasks based on data locality:

```cpp
u32 MyScheduler::ClientMapTask(IpcManager *ipc_manager, const Future<Task> &task) {
  // Extract data location from task
  PoolId pool_id = task->pool_id_;

  // Map to worker closest to data
  return ComputeLocalityMap(pool_id, ipc_manager->GetNumSchedQueues());
}
```

### Priority-Based Scheduling

Use task priorities for scheduling:

```cpp
void MyScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  u32 total = work_orch->GetTotalWorkerCount();
  u32 high_prio_count = total / 2;

  for (u32 i = 0; i < high_prio_count; ++i) {
    high_priority_workers_.push_back(work_orch->GetWorker(i));
  }

  for (u32 i = high_prio_count; i < total - 1; ++i) {
    low_priority_workers_.push_back(work_orch->GetWorker(i));
  }

  // Network worker
  net_worker_ = work_orch->GetWorker(total - 1);
}
```

## Troubleshooting

### Tasks Not Being Processed

**Symptom**: Tasks submitted but never execute

**Check**:
1. Did you call `IpcManager::SetNumSchedQueues()` in `DivideWorkers`?
2. Are all workers getting lanes via WorkOrchestrator's 1:1 mapping?
3. Does `ClientMapTask` return lane IDs in valid range?

### Client Mapping Errors

**Symptom**: Assertion failures or crashes in `ClientMapTask`

**Check**:
1. Is returned lane ID in range `[0, num_sched_queues)`?
2. Did you check for `num_lanes == 0`?
3. Are you using modulo to wrap lane IDs?

### Worker Crashes

**Symptom**: Workers crash during initialization

**Check**:
1. Are you checking for null pointers?
2. Does `DivideWorkers` handle `total_workers < expected`?
3. Is the single-worker case handled (when `total_workers == 1`)?

## References

- **Scheduler Interface**: `context-runtime/include/chimaera/scheduler/scheduler.h`
- **DefaultScheduler**: `context-runtime/src/scheduler/default_sched.cc`
- **WorkOrchestrator**: `context-runtime/src/work_orchestrator.cc`
- **IpcManager**: `context-runtime/src/ipc_manager.cc`
- **Configuration**: `context-runtime/docs/deployment.md`
