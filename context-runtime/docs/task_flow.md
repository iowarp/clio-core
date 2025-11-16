# Chimaera Task Execution Flow and Lane Scheduling System

This document provides comprehensive technical documentation of the task flow and lane scheduling system in the Chimaera distributed task execution framework, detailing the complete lifecycle from server initialization through task completion.

## Executive Summary

The Chimaera framework implements a sophisticated multi-tier task execution system using shared memory IPC, lane-based scheduling, and fiber-based context switching. The system architecture comprises:

- **Three-segment shared memory model** for inter-process communication
- **Multi-lane task queues** with per-lane worker assignment
- **Work orchestration** with configurable worker types
- **Container-based execution** with ChiMod plugin architecture
- **Boost::context fibers** for cooperative task yielding

## System Architecture Overview

### Core Execution Flow

Tasks progress through seven distinct stages in their lifecycle:

1. **Server Initialization** - Runtime creates shared memory segments and worker pools
2. **Client Connection** - Clients attach to shared memory and discover queues
3. **Task Submission** - Tasks allocated in shared memory and enqueued to lanes
4. **Lane Scheduling** - Lanes assigned to workers via round-robin scheduling
5. **Worker Processing** - Workers poll lanes and extract tasks for execution
6. **Container Routing** - Tasks routed to appropriate container implementations
7. **Task Completion** - Tasks execute, potentially yield, and eventually complete

## Architecture Components

### Primary System Components

#### IpcManager (`include/chimaera/ipc_manager.h`)
- **Role**: Central hub for inter-process communication and memory management
- **Responsibilities**:
  - Creates and manages three shared memory segments
  - Allocates tasks and buffers in appropriate segments
  - Maintains external task queue and worker queues
  - Provides client/server initialization paths
- **Key Data Structures**:
  - `IpcSharedHeader`: Contains external queue and worker queue references
  - Three allocators: main, client_data, runtime_data

#### TaskQueue (`include/chimaera/task_queue.h`)
- **Role**: Multi-lane, lock-free queue system for task distribution
- **Implementation**: Wrapper around `hipc::multi_mpsc_queue`
- **Features**:
  - Configurable number of lanes (default: from config)
  - Per-lane headers with worker assignment and task count
  - Static methods for task emplacement and popping
  - Automatic worker notification on empty→non-empty transition

#### WorkOrchestrator (`include/chimaera/work_orchestrator.h`)
- **Role**: Worker thread lifecycle and lane scheduling manager
- **Responsibilities**:
  - Creates and manages worker threads of different types
  - Implements round-robin lane-to-worker assignment
  - Detects stack growth direction for boost::context
  - Routes lane notifications to appropriate workers
- **Worker Types**:
  - Low latency workers (CPU-bound tasks)
  - High latency workers (I/O-bound tasks)
  - Reinforcement workers (backup processing)
  - Process reaper workers (cleanup tasks)

#### Worker (`include/chimaera/worker.h`)
- **Role**: Task execution engine with fiber-based context switching
- **Features**:
  - Polls assigned lanes from active queue
  - Processes batches of tasks (up to 64 per lane)
  - Manages execution stacks (64KB default)
  - Handles task yielding and resumption
  - Maintains blocked queue for yielded tasks

#### Container (`include/chimaera/container.h`)
- **Role**: Base class for ChiMod-specific execution environments
- **Provides**:
  - Local queue creation and management
  - Lane access methods (by ID or hash)
  - Default implementations for common operations
  - Integration with WorkOrchestrator for lane scheduling

### Memory Architecture

```
┌──────────────────────────────────────────────────────┐
│         IpcSharedHeader (in Main Segment)           │
│  ┌────────────────────────────────────────────────┐ │
│  │ external_queue: delay_ar<TaskQueue>            │ │
│  │ worker_queues: delay_ar<vector<mpsc_queue>>    │ │
│  │ num_workers: u32                               │ │
│  └────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│              Main Segment (Segment 0)                │
│  - Task allocation (all task types)                  │
│  - TaskQueue instances (external and container)      │
│  - Worker active queues                              │
│  - Shared data structures                            │
├──────────────────────────────────────────────────────┤
│           Client Data Segment (Segment 1)            │
│  - Client-allocated buffers                          │
│  - Client-specific temporary data                    │
│  - Input data for tasks                              │
├──────────────────────────────────────────────────────┤
│          Runtime Data Segment (Segment 2)            │
│  - Runtime-allocated buffers                         │
│  - Worker execution stacks                           │
│  - Container internal data                           │
└──────────────────────────────────────────────────────┘
```

## Detailed Flow Analysis

### Phase 1: Server Initialization

#### 1.1 Runtime Startup (`util/chimaera_start_runtime.cc`)

**Location**: `util/chimaera_start_runtime.cc:120-158`

**Process**:
1. Main function calls `chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer, false)`
2. Chimaera manager initializes in server mode (`src/chimaera_manager.cc:54-93`)
3. Initialization sequence:
   - ConfigManager loads configuration
   - IpcManager creates shared memory segments
   - PoolManager initializes container registry
   - ModuleManager loads available ChiMods
   - WorkOrchestrator creates worker threads
   - Workers start polling for tasks

#### 1.2 IPC Initialization (`src/ipc_manager.cc:37-57`)

**Server Initialization Path**:
The server initialization follows a three-step process: creating shared memory segments for IPC, initializing the external task queue with configurable lanes, and optionally setting up ZeroMQ networking for remote access.

**Shared Memory Creation** (`src/ipc_manager.cc:173-235`):
1. Creates three POSIX shared memory segments:
   - Main segment: Default size from config (segment 0)
   - Client data segment: For client buffers (segment 1)
   - Runtime data segment: For runtime buffers (segment 2)
2. Creates allocators for each segment with IDs (1,0), (2,0), (3,0)
3. Main allocator includes custom header space for `IpcSharedHeader`

**External Queue Creation** (`src/ipc_manager.cc:281-319`):
The server initializes the external task queue by retrieving the shared header from the main allocator and configuring the queue with the specified number of lanes from the configuration, setting up concurrent task processing capabilities.

#### 1.3 Worker Initialization (`src/work_orchestrator.cc:129-192`)

**Stack Detection** (`src/work_orchestrator.cc:52-121`):
- Detects stack growth direction using boost::context
- Allocates test stack and creates fiber context
- Determines if architecture uses downward (x86_64) or upward stacks

**Worker Creation**:
The initialization process detects the stack growth direction for boost::context, creates thread-local storage for worker contexts, sets up worker queues in shared memory, and creates different types of workers according to configuration settings.

### Phase 2: Client Connection

#### 2.1 Client Initialization (`src/chimaera_manager.cc:27-51`)

**Process**:
Client initialization involves setting up the configuration manager, attaching to existing shared memory segments created by the server, and initializing the pool manager for container access.

#### 2.2 Client IPC Attachment (`src/ipc_manager.cc:237-279`)

**Memory Attachment**:
The client attaches to the three existing POSIX shared memory segments by name and retrieves references to the allocators for each segment to enable shared memory operations.

**Queue Discovery** (`src/ipc_manager.cc:321-340`):
The client discovers existing queues by accessing the shared header from the main allocator and obtaining a reference to the external task queue for task submission.

### Phase 3: Task Submission

#### 3.1 Task Allocation (`include/chimaera/ipc_manager.h:57-65`)

The task allocation template function creates a context allocator from the main allocator and uses it to construct a new task object of the specified type with forwarded arguments in shared memory.

**Example Client Usage** (`chimods/admin/include/admin/admin_client.h:45-59`):
Clients allocate tasks in shared memory by specifying the task type, node ID, target pool, and pool query, then submit the task to the runtime for processing.

#### 3.2 Task Enqueueing (`include/chimaera/ipc_manager.h:106-124`)

The task enqueueing function converts the task pointer to a typed pointer, selects a lane using thread ID hashing for load distribution, and emplaces the task into the chosen lane for worker processing.

**Lane Selection Algorithm**:
- Thread ID provides natural distribution across lanes
- Hash function ensures consistent lane assignment per thread
- Number of lanes is configurable via `ConfigManager`
- Default configuration typically uses 4-16 lanes

### Phase 4: Lane Scheduling and Worker Notification

#### 4.1 Task Emplacement and Notification (`src/task_queue.cc:27-44`)

The task emplacement function checks if the lane is empty before insertion, pushes the task to the lane, and notifies workers only when the lane transitions from empty to non-empty to optimize performance.

**Notification Optimization**:
- Notifications only sent when lane becomes non-empty
- Reduces worker wake-ups and context switches
- Lane remains in worker queue until fully drained

#### 4.2 Lane-to-Worker Assignment

**Initial Assignment** (`src/work_orchestrator.cc:404-435`):
The round-robin scheduler assigns each lane to an available worker by iterating through all lanes, selecting the next worker in rotation, and storing the worker assignment in the lane header.

**Runtime Notification** (`src/work_orchestrator.cc:437-460`):
When a lane becomes ready, the system reads the assigned worker ID from the lane header, retrieves the worker's queue from shared memory, and enqueues the lane to the worker's active queue for processing.

### Phase 5: Worker Task Processing

#### 5.1 Worker Main Loop (`src/worker.cc:68-143`)

The worker main loop sets the thread-local context, continuously polls the active queue for lanes, processes up to 64 tasks per lane in batches, re-enqueues lanes with remaining tasks, checks blocked tasks for completion, and sleeps or yields when no work is available.

**Batch Processing Strategy**:
- Process up to 64 tasks per lane before switching
- Improves cache locality and reduces queue contention
- Ensures fairness by re-enqueueing non-empty lanes

#### 5.2 Task Processing Pipeline (`src/worker.cc:89-114`)

The task processing pipeline involves four steps: domain resolution to validate execution location, container lookup from the pool manager, local scheduling through the container monitor, and finally task execution with the appropriate container.

### Phase 6: Container Routing and Execution

#### 6.1 Domain Resolution (`src/worker.cc:178-197`)

The domain resolution function validates the pool ID and determines execution location, currently defaulting to local execution with support for remote and global execution planned for future releases.

#### 6.2 Container Lookup (`src/worker.cc:199-208`)

Container lookup retrieves the appropriate container instance from the pool manager using the task's pool ID.

#### 6.3 Monitor-Based Scheduling (`src/worker.cc:210-227`)

The monitor-based scheduling creates a temporary run context and calls the container's monitor in local schedule mode to assign the task to an appropriate container-local lane.

**Monitor Modes**:
- `kLocalSchedule`: Assign task to container-local lane
- `kRemoteSchedule`: Route to remote node (future)
- `kTaskComplete`: Post-execution monitoring

#### 6.4 Task Execution with Fibers (`src/worker.cc:293-325`)

Task execution begins by allocating a 64KB execution stack and creating a run context with worker information, task details, and container reference, then initiating fiber-based execution.

#### 6.5 Fiber Execution (`src/worker.cc:374-425`)

Fiber execution sets the thread-local context, either resumes an existing fiber or creates a new one with boost::context, jumps to the fiber execution function, and handles task completion or blocking by cleaning up resources or maintaining context for later resumption.

### Phase 7: Task Completion and Cleanup

#### 7.1 Container Execution (`src/worker.cc:468-511`)

The fiber execution function retrieves context from thread-local storage, executes the task in the appropriate container, handles periodic task rescheduling or marks one-shot tasks as complete, and returns control to the worker context.

#### 7.2 Blocked Task Management (`src/worker.cc:327-372`)

Blocked task management checks the blocked queue for tasks whose subtasks have completed, resumes ready tasks, and calculates optimal sleep time based on estimated completion times of remaining blocked tasks.

#### 7.3 Periodic Task Rescheduling (`src/worker.cc:441-466`)

Periodic task rescheduling checks if the lane is still assigned to the current worker, either adding the task to the blocked queue with a period delay or re-enqueueing it if the lane has been reassigned to another worker.

## Queue Architecture and Data Flow

### Queue Hierarchy

```
                    Client Process
                         │
                         ▼
              ┌──────────────────┐
              │  External Queue  │
              │  (IpcManager)    │
              │  N lanes × 1 pri │
              └──────────────────┘
                    │ │ │ │
                    ▼ ▼ ▼ ▼
         ┌──────────────────────────┐
         │    Worker Active Queues   │
         │ mpsc_queue<FullPtr<Lane>>│
         │    (One per worker)       │
         └──────────────────────────┘
                    │ │ │ │
                    ▼ ▼ ▼ ▼
         ┌──────────────────────────┐
         │  Container Local Queues   │
         │  (Per-container lanes)    │
         │  M lanes × 1 priority     │
         └──────────────────────────┘
```

### Lane Header Structure

Each lane maintains a header (`TaskQueueHeader`) containing:
The TaskQueueHeader structure maintains lane metadata including the container pool ID, assigned worker ID, current task count, and a flag indicating whether the lane is currently in a worker queue.

### Worker Queue Types

```
┌─────────────────────────────────────────────┐
│              Worker Queues                  │
├─────────────────────────────────────────────┤
│ Active Queue:                               │
│   - mpsc_queue<FullPtr<TaskLane>>          │
│   - Lanes with pending tasks               │
│                                             │
│ Blocked Queue:                              │
│   - priority_queue<RunContext*>            │
│   - Tasks waiting on I/O or timer          │
│   - Ordered by estimated completion time   │
└─────────────────────────────────────────────┘
```

## Performance Optimizations

### Lane-Based Concurrency
- **Cache Affinity**: Tasks from same thread likely hit same lane
- **Reduced Contention**: Multiple lanes reduce lock contention
- **Batch Processing**: Process up to 64 tasks per lane dequeue
- **Worker Affinity**: Lanes stick to assigned workers

### Memory Layout Optimizations
- **Shared Memory Zero-Copy**: No serialization between processes
- **Stack Reuse**: 64KB stacks allocated/freed per task
- **Allocator Caching**: Allocator pointers cached in IpcManager
- **NUMA Awareness**: Future enhancement for multi-socket systems

### Scheduling Optimizations
- **Empty-to-NonEmpty Notification**: Reduces spurious wake-ups
- **Blocked Queue Priority**: Tasks ordered by completion time
- **Periodic Task Optimization**: Reuse context for periodic tasks
- **Lane Re-enqueueing**: Fair scheduling without starvation

## Implementation Patterns

### Singleton Access Pattern
All system managers are accessed through HSHM global pointer variables that provide centralized access to configuration, IPC and memory management, container pool registry, ChiMod loading, and worker management functionality.

### Thread-Local Storage Pattern
Worker context information is stored in thread-local variables that provide access to the current worker, run context, task, container, and lane from any point in the execution flow.

### Task State Management
Task lifecycle is managed through an atomic completion flag, execution context pointer, periodic task identification methods, and period retrieval functions for timing control.

## Configuration and Tuning

### Configurable Parameters
- **Number of Lanes**: External queue lanes (default: from config)
- **Worker Counts**: Per-type worker thread counts
- **Memory Segment Sizes**: Main, client, runtime segments
- **Queue Depths**: Per-lane queue capacity (default: 1024)
- **Batch Size**: Tasks processed per lane (default: 64)
- **Stack Size**: Execution stack per task (default: 64KB)

### Performance Tuning Guidelines
1. **Lane Count**: 2-4x number of client threads
2. **Worker Count**: Match CPU core count for CPU-bound work
3. **Queue Depth**: Balance memory vs burst capacity
4. **Batch Size**: Trade-off latency vs throughput
5. **Stack Size**: Minimize for memory efficiency

## Error Handling and Recovery

### Failure Scenarios

#### Memory Allocation Failures
Allocation failures are handled gracefully by checking if the returned task pointer is null and returning appropriate error codes to enable system degradation rather than crashes.

#### Worker Crashes
- WorkOrchestrator monitors worker health
- Failed workers can be restarted
- Lanes reassigned to healthy workers
- Tasks in crashed worker's queue recoverable

#### Container Exceptions
Workers catch all exceptions thrown by container execution, log error details, and mark tasks as failed to prevent system-wide crashes while maintaining error visibility.

### Debugging Support

#### Trace Points
- Task creation and deletion
- Queue enqueue/dequeue operations
- Worker state transitions
- Container method invocations
- Fiber context switches

#### Performance Metrics
- Queue depths and throughput
- Worker utilization percentages
- Task execution latencies
- Memory allocation statistics
- Context switch frequencies

## Summary

The Chimaera task flow and lane scheduling system provides a highly optimized, scalable framework for distributed task execution. Key architectural decisions include:

1. **Three-segment shared memory model** enables zero-copy IPC between clients and runtime
2. **Multi-lane queue architecture** reduces contention and improves cache locality
3. **Per-lane worker assignment** ensures predictable task routing and load balancing
4. **Fiber-based execution** allows efficient context switching for I/O-bound tasks
5. **Batch processing** optimizes throughput while maintaining fairness
6. **Container abstraction** provides clean plugin architecture for ChiMods

The system achieves high performance through careful attention to memory layout, lock-free data structures, and intelligent scheduling algorithms. The modular design allows for future enhancements including remote execution, GPU offload, and advanced scheduling policies.