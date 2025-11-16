# Admin ChiMod Documentation

## Overview

The Admin ChiMod is a critical component of the Chimaera runtime system that manages ChiPools and runtime lifecycle operations. It provides essential functionality for pool creation/destruction, runtime shutdown, and distributed task communication between nodes.

**Key Responsibilities:**
- Pool management (creation, destruction)
- Runtime lifecycle control (initialization, shutdown)
- Distributed task routing and communication
- Administrative operations (flush, monitoring)

## CMake Integration

### External Projects

To use the Admin ChiMod in external projects:

```cmake
find_package(chimaera_admin REQUIRED)      # Admin ChiMod package
find_package(chimaera REQUIRED)            # Core Chimaera (automatically includes ChimaeraCommon.cmake)

target_link_libraries(your_application
  chimaera::admin_client        # Admin client library
  ${CMAKE_THREAD_LIBS_INIT}     # Threading support
)
# Core Chimaera library dependencies are automatically included by ChiMod libraries
```

### Required Headers

```cpp
#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>
```

## API Reference

### Client Class: `chimaera::admin::Client`

The Admin client provides the primary interface for interacting with the admin container.

#### Constructor

```cpp
// Default constructor
Client()

// Constructor with pool ID
explicit Client(const chi::PoolId& pool_id)
```

#### Container Management

##### `Create()` - Synchronous
Creates and initializes the admin container.

```cpp
void Create(const hipc::MemContext& mctx, const chi::PoolQuery& pool_query, 
           const std::string& pool_name)
```

**Parameters:**
- `mctx`: Memory context for task allocation
- `pool_query`: Pool domain query (typically `chi::PoolQuery::Local()`)
- `pool_name`: Pool name (MUST be "admin" for admin containers)

**Usage:**
```cpp
chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
const chi::PoolId pool_id = chi::kAdminPoolId;  // Use predefined admin pool ID
chimaera::admin::Client admin_client(pool_id);

auto pool_query = chi::PoolQuery::Local();
admin_client.Create(HSHM_MCTX, pool_query, "admin");  // Pool name MUST be "admin"
```

##### `AsyncCreate()` - Asynchronous
Creates and initializes the admin container asynchronously.

```cpp
hipc::FullPtr<CreateTask> AsyncCreate(const hipc::MemContext& mctx,
                                     const chi::PoolQuery& pool_query,
                                     const std::string& pool_name)
```

**Returns:** Task pointer for asynchronous completion checking

#### Pool Management Operations

##### `DestroyPool()` - Synchronous
Destroys an existing ChiPool.

```cpp
void DestroyPool(const hipc::MemContext& mctx,
                const chi::PoolQuery& pool_query, 
                chi::PoolId target_pool_id,
                chi::u32 destruction_flags = 0)
```

**Parameters:**
- `mctx`: Memory context for task allocation
- `pool_query`: Pool domain query
- `target_pool_id`: ID of the pool to destroy
- `destruction_flags`: Optional flags controlling destruction behavior (default: 0)

**Throws:** `std::runtime_error` if destruction fails

##### `AsyncDestroyPool()` - Asynchronous
Destroys an existing ChiPool asynchronously.

```cpp
hipc::FullPtr<DestroyPoolTask> AsyncDestroyPool(
    const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
    chi::PoolId target_pool_id, chi::u32 destruction_flags = 0)
```

#### Distributed Task Operations

##### `ClientSendTaskIn()` - Synchronous
Sends a task to remote nodes for distributed processing.

```cpp
template <typename TaskType>
void ClientSendTaskIn(const hipc::MemContext& mctx,
                     const std::vector<chi::PoolQuery>& pool_queries,
                     const hipc::FullPtr<TaskType>& task_to_send)
```

**Parameters:**
- `mctx`: Memory context for task allocation
- `pool_queries`: Vector of pool queries identifying target nodes
- `task_to_send`: Pointer to the task to send

**Throws:** `std::runtime_error` if task send fails

##### `AsyncClientSendTaskIn()` - Asynchronous
```cpp
template <typename TaskType>
hipc::FullPtr<ClientSendTaskInTask> AsyncClientSendTaskIn(
    const hipc::MemContext& mctx,
    const std::vector<chi::PoolQuery>& pool_queries,
    const hipc::FullPtr<TaskType>& task_to_send)
```

##### `ServerRecvTaskIn()` - Synchronous
Polls and receives tasks from remote nodes.

```cpp
void ServerRecvTaskIn(const hipc::MemContext& mctx,
                     const chi::PoolQuery& pool_query)
```

This is typically called periodically to check for incoming tasks from remote nodes.

##### `AsyncServerRecvTaskIn()` - Asynchronous
```cpp
hipc::FullPtr<ServerRecvTaskInTask> AsyncServerRecvTaskIn(
    const hipc::MemContext& mctx, const chi::PoolQuery& pool_query)
```

##### `ServerSendTaskOut()` - Synchronous
Sends completed task results to remote nodes.

```cpp
template <typename TaskType>
void ServerSendTaskOut(const hipc::MemContext& mctx,
                      const chi::PoolQuery& pool_query,
                      chi::u32 target_node_id,
                      const hipc::FullPtr<TaskType>& completed_task)
```

**Parameters:**
- `mctx`: Memory context for task allocation
- `pool_query`: Pool domain query
- `target_node_id`: ID of the target node to send results to
- `completed_task`: Pointer to the completed task

##### `AsyncServerSendTaskOut()` - Asynchronous
```cpp
template <typename TaskType>
hipc::FullPtr<ServerSendTaskOutTask> AsyncServerSendTaskOut(
    const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
    chi::u32 target_node_id, const hipc::FullPtr<TaskType>& completed_task)
```

##### `ClientRecvTaskOut()` - Synchronous
Polls and receives task results from remote nodes.

```cpp
void ClientRecvTaskOut(const hipc::MemContext& mctx,
                      const chi::PoolQuery& pool_query)
```

##### `AsyncClientRecvTaskOut()` - Asynchronous
```cpp
hipc::FullPtr<ClientRecvTaskOutTask> AsyncClientRecvTaskOut(
    const hipc::MemContext& mctx, const chi::PoolQuery& pool_query)
```

#### Administrative Operations

##### `Flush()` - Synchronous
Flushes all administrative operations.

```cpp
void Flush(const hipc::MemContext& mctx, const chi::PoolQuery& pool_query)
```

**Throws:** `std::runtime_error` if flush operation fails

##### `AsyncFlush()` - Asynchronous
```cpp
hipc::FullPtr<FlushTask> AsyncFlush(const hipc::MemContext& mctx,
                                   const chi::PoolQuery& pool_query)
```

#### Runtime Control

##### `AsyncStopRuntime()` - Asynchronous Only
Stops the entire Chimaera runtime system.

```cpp
hipc::FullPtr<StopRuntimeTask> AsyncStopRuntime(
    const hipc::MemContext& mctx, const chi::PoolQuery& pool_query,
    chi::u32 shutdown_flags = 0, chi::u32 grace_period_ms = 5000)
```

**Parameters:**
- `mctx`: Memory context for task allocation
- `pool_query`: Pool domain query
- `shutdown_flags`: Optional flags controlling shutdown behavior (default: 0)
- `grace_period_ms`: Grace period in milliseconds for clean shutdown (default: 5000ms)

**Note:** This operation is only available asynchronously as the runtime shutdown process requires careful coordination.

## Task Types

### CreateTask
Container creation task for the admin module. This is an alias for `chimaera::admin::BaseCreateTask<CreateParams, Method::kCreate, true>`.

**Key Fields:**
- Inherits from `BaseCreateTask` with admin-specific `CreateParams`
- `chimod_name_`: Name of the ChiMod being created
- `pool_name_`: Name of the pool (must be "admin" for admin containers)
- `chimod_params_`: Serialized parameters
- `pool_id_`: Pool identifier (input/output)
- `result_code_`: Operation result (0 = success)
- `error_message_`: Error description if creation failed

### DestroyPoolTask
Pool destruction task.

**Key Fields:**
- `target_pool_id_`: ID of the pool to destroy
- `destruction_flags_`: Flags controlling destruction behavior
- `result_code_`: Operation result (0 = success)
- `error_message_`: Error description if destruction failed

### StopRuntimeTask
Runtime shutdown task.

**Key Fields:**
- `shutdown_flags_`: Flags controlling shutdown behavior
- `grace_period_ms_`: Grace period for clean shutdown
- `result_code_`: Operation result (0 = success)
- `error_message_`: Error description if shutdown failed

### FlushTask
Administrative flush task.

**Key Fields:**
- `result_code_`: Operation result (0 = success)
- `total_work_done_`: Total work remaining across all containers

### Distributed Task Communication Tasks

#### ClientSendTaskInTask
Sends task input to remote nodes.

**Key Fields:**
- `pool_queries_`: Target node pool queries
- `task_to_send_`: Task to transmit
- `transfer_flags_`: Transfer behavior flags
- `result_code_`: Transfer result
- `error_message_`: Error description if transfer failed

#### ServerRecvTaskInTask
Receives task input from remote nodes (periodic polling).

#### ServerSendTaskOutTask
Sends completed task results to remote nodes.

**Key Fields:**
- `completed_task_`: Completed task to send
- `transfer_flags_`: Transfer behavior flags
- `result_code_`: Transfer result
- `error_message_`: Error description if transfer failed

#### ClientRecvTaskOutTask
Receives task results from remote nodes (periodic polling).

## Configuration

### CreateParams Structure
The admin module uses minimal configuration parameters:

```cpp
struct CreateParams {
  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "chimaera_admin";
  
  // Default constructor
  CreateParams() = default;
  
  // Constructor with allocator
  explicit CreateParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc);
};
```

**Important:** The `chimod_lib_name` does NOT include the `_runtime` suffix as it is automatically appended by the module manager.

## Usage Examples

### Basic Admin Container Setup
```cpp
#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>

int main() {
  // Initialize Chimaera (client mode with embedded runtime)
  chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

  // Create admin client with proper admin pool ID
  const chi::PoolId pool_id = chi::kAdminPoolId;
  chimaera::admin::Client admin_client(pool_id);

  // Create admin container (pool name MUST be "admin")
  auto pool_query = chi::PoolQuery::Local();
  admin_client.Create(HSHM_MCTX, pool_query, "admin");
  
  // Perform admin operations...
  admin_client.Flush(HSHM_MCTX, pool_query);
  
  return 0;
}
```

### Pool Management
```cpp
// Destroy a specific pool
chi::PoolId target_pool = chi::PoolId(8000, 0);
try {
  admin_client.DestroyPool(HSHM_MCTX, pool_query, target_pool);
  std::cout << "Pool destroyed successfully" << std::endl;
} catch (const std::runtime_error& e) {
  std::cerr << "Pool destruction failed: " << e.what() << std::endl;
}
```

### Distributed Task Operations
```cpp
// Send a task to remote nodes
std::vector<chi::PoolQuery> target_nodes = {
  chi::PoolQuery::Remote(1),  // Node 1
  chi::PoolQuery::Remote(2)   // Node 2
};

// Assume we have a task to send
hipc::FullPtr<SomeTaskType> task = /* create task */;

try {
  admin_client.ClientSendTaskIn(HSHM_MCTX, target_nodes, task);
  std::cout << "Task sent to remote nodes successfully" << std::endl;
} catch (const std::runtime_error& e) {
  std::cerr << "Task send failed: " << e.what() << std::endl;
}
```

### Runtime Shutdown
```cpp
// Gracefully stop the runtime with 10 second grace period
auto stop_task = admin_client.AsyncStopRuntime(
  HSHM_MCTX, pool_query, 0, 10000);  // 10 seconds

// Don't wait for completion as runtime will shut down
std::cout << "Runtime shutdown initiated" << std::endl;
```

## Dependencies

- **HermesShm**: Shared memory framework and IPC
- **Chimaera core runtime**: Base runtime objects and task framework
- **cereal**: Serialization library for network communication
- **Boost.Fiber** and **Boost.Context**: Coroutine support

## Installation

1. Build Chimaera with the admin module:
   ```bash
   cmake --preset debug
   cmake --build build
   ```

2. Install to system or custom prefix:
   ```bash
   cmake --install build --prefix /usr/local
   ```

3. For external projects, set CMAKE_PREFIX_PATH:
   ```bash
   export CMAKE_PREFIX_PATH="/usr/local:/path/to/hermes-shm:/path/to/other/deps"
   ```

## Error Handling

Most synchronous methods throw `std::runtime_error` on failure. The error message contains details about the failure cause.

For asynchronous operations, check the `result_code_` field of the returned task:
- `0`: Success
- Non-zero: Error occurred (check `error_message_` field)

**Example:**
```cpp
auto task = admin_client.AsyncDestroyPool(HSHM_MCTX, pool_query, target_pool);
task->Wait();

if (task->result_code_ != 0) {
  std::string error = task->error_message_.str();
  std::cerr << "Operation failed: " << error << std::endl;
}

// Clean up
auto* ipc_manager = CHI_IPC;
ipc_manager->DelTask(task);
```

## Important Notes

1. **Pool ID for CreateTask**: All ChiMod CreateTask operations must use `chi::kAdminPoolId`, not the client's `pool_id_`.

2. **Admin Pool Name**: The admin pool name MUST always be "admin". Multiple admin pools are NOT supported.

3. **Admin Dependency**: The admin module is required by all other ChiMods and must be linked in all Chimaera applications.

3. **Asynchronous Operations**: Always clean up task pointers after completion using `ipc_manager->DelTask(task)`.

4. **Pool Queries**: Use `chi::PoolQuery::Local()` for local operations and `chi::PoolQuery::Remote(node_id)` for distributed operations.

5. **Thread Safety**: All operations are designed to be called from the main thread. Multi-threaded access requires external synchronization.