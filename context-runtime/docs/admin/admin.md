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

##### `AsyncCreate()`
Creates and initializes the admin container asynchronously.

```cpp
chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                    const std::string& pool_name,
                                    const chi::PoolId& custom_pool_id)
```

**Parameters:**
- `pool_query`: Pool domain query (typically `chi::PoolQuery::Local()`)
- `pool_name`: Pool name (MUST be "admin" for admin containers)
- `custom_pool_id`: Explicit pool ID for the container

**Returns:** Future for asynchronous completion checking

**Usage:**
```cpp
chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
const chi::PoolId pool_id = chi::kAdminPoolId;  // Use predefined admin pool ID
chimaera::admin::Client admin_client(pool_id);

auto pool_query = chi::PoolQuery::Local();
auto task = admin_client.AsyncCreate(pool_query, "admin", pool_id);
task.Wait();

if (task->GetReturnCode() != 0) {
  std::cerr << "Admin creation failed" << std::endl;
  return;
}
```

#### Pool Management Operations

##### `AsyncDestroyPool()`
Destroys an existing ChiPool asynchronously.

```cpp
chi::Future<DestroyPoolTask> AsyncDestroyPool(
    const chi::PoolQuery& pool_query,
    chi::PoolId target_pool_id, chi::u32 destruction_flags = 0)
```

**Parameters:**
- `pool_query`: Pool domain query
- `target_pool_id`: ID of the pool to destroy
- `destruction_flags`: Optional flags controlling destruction behavior (default: 0)

#### Network Communication Operations

##### `AsyncSendPoll()` - Asynchronous
Creates a periodic task to poll the network queue and send outgoing messages.

```cpp
chi::Future<SendTask> AsyncSendPoll(const chi::PoolQuery& pool_query,
                                    chi::u32 transfer_flags = 0,
                                    double period_us = 25)
```

**Parameters:**
- `pool_query`: Pool domain query
- `transfer_flags`: Transfer behavior flags (default: 0)
- `period_us`: Period in microseconds for polling (default: 25us, 0 = one-shot)

##### `AsyncRecv()` - Asynchronous
Creates a periodic task to receive incoming messages from the network.

```cpp
chi::Future<RecvTask> AsyncRecv(const chi::PoolQuery& pool_query,
                                chi::u32 transfer_flags = 0,
                                double period_us = 25)
```

**Parameters:**
- `pool_query`: Pool domain query
- `transfer_flags`: Transfer behavior flags (default: 0)
- `period_us`: Period in microseconds for polling (default: 25us, 0 = one-shot)

#### Administrative Operations

##### `AsyncFlush()`
Flushes all administrative operations asynchronously.

```cpp
chi::Future<FlushTask> AsyncFlush(const chi::PoolQuery& pool_query)
```

**Parameters:**
- `pool_query`: Pool domain query

**Returns:** Future for asynchronous completion checking

#### Runtime Control

##### `AsyncStopRuntime()` - Asynchronous Only
Stops the entire Chimaera runtime system.

```cpp
chi::Future<StopRuntimeTask> AsyncStopRuntime(
    const chi::PoolQuery& pool_query,
    chi::u32 shutdown_flags = 0, chi::u32 grace_period_ms = 5000)
```

**Parameters:**
- `pool_query`: Pool domain query
- `shutdown_flags`: Optional flags controlling shutdown behavior (default: 0)
- `grace_period_ms`: Grace period in milliseconds for clean shutdown (default: 5000ms)

**Note:** This operation is only available asynchronously as the runtime shutdown process requires careful coordination.

#### Compose Operation

##### `AsyncCompose()` - Asynchronous
Creates a pool from a PoolConfig (for declarative pool creation).

```cpp
chi::Future<ComposeTask<chi::PoolConfig>> AsyncCompose(
    const chi::PoolConfig& pool_config)
```

**Parameters:**
- `pool_config`: Configuration for the pool to create

#### Heartbeat Operation

##### `AsyncHeartbeat()` - Asynchronous
Polls for ZMQ heartbeat requests and responds.

```cpp
chi::Future<HeartbeatTask> AsyncHeartbeat(const chi::PoolQuery& pool_query,
                                          double period_us = 5000)
```

**Parameters:**
- `pool_query`: Pool domain query
- `period_us`: Period in microseconds (default: 5000us = 5ms, 0 = one-shot)

## Task Types

### CreateTask
Container creation task for the admin module. This is an alias for `chimaera::admin::BaseCreateTask<CreateParams, Method::kCreate, true>`.

**Key Fields:**
- Inherits from `BaseCreateTask` with admin-specific `CreateParams`
- `chimod_name_`: Name of the ChiMod being created
- `pool_name_`: Name of the pool (must be "admin" for admin containers)
- `chimod_params_`: Serialized parameters
- `pool_id_`: Pool identifier (input/output)
- `return_code_`: Operation result (0 = success)
- `error_message_`: Error description if creation failed

### DestroyPoolTask
Pool destruction task.

**Key Fields:**
- `target_pool_id_`: ID of the pool to destroy
- `destruction_flags_`: Flags controlling destruction behavior
- `return_code_`: Operation result (0 = success)
- `error_message_`: Error description if destruction failed

### StopRuntimeTask
Runtime shutdown task.

**Key Fields:**
- `shutdown_flags_`: Flags controlling shutdown behavior
- `grace_period_ms_`: Grace period for clean shutdown
- `return_code_`: Operation result (0 = success)
- `error_message_`: Error description if shutdown failed

### FlushTask
Administrative flush task.

**Key Fields:**
- `return_code_`: Operation result (0 = success)
- `total_work_done_`: Total work remaining across all containers

### SendTask / RecvTask
Network communication tasks for sending and receiving messages.

**Key Fields:**
- `transfer_flags_`: Transfer behavior flags
- `return_code_`: Transfer result

## Configuration

### CreateParams Structure
The admin module uses minimal configuration parameters:

```cpp
struct CreateParams {
  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "chimaera_admin";

  // Default constructor
  CreateParams() = default;
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

  // Create admin container asynchronously (pool name MUST be "admin")
  auto pool_query = chi::PoolQuery::Local();
  auto create_task = admin_client.AsyncCreate(pool_query, "admin", pool_id);
  create_task.Wait();

  if (create_task->GetReturnCode() != 0) {
    std::cerr << "Admin creation failed" << std::endl;
    return 1;
  }

  // Perform admin operations...
  auto flush_task = admin_client.AsyncFlush(pool_query);
  flush_task.Wait();

  return 0;
}
```

### Pool Management
```cpp
// Destroy a specific pool
chi::PoolId target_pool = chi::PoolId(8000, 0);
auto destroy_task = admin_client.AsyncDestroyPool(pool_query, target_pool);
destroy_task.Wait();

if (destroy_task->return_code_ != 0) {
  std::cerr << "Pool destruction failed" << std::endl;
} else {
  std::cout << "Pool destroyed successfully" << std::endl;
}
```

### Runtime Shutdown
```cpp
// Gracefully stop the runtime with 10 second grace period
auto stop_task = admin_client.AsyncStopRuntime(
  pool_query, 0, 10000);  // 10 seconds

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

All operations are asynchronous and return `chi::Future<TaskType>`. Check the `return_code_` field of the returned task after calling `Wait()`:
- `0`: Success
- Non-zero: Error occurred (check `error_message_` field)

**Example:**
```cpp
auto task = admin_client.AsyncDestroyPool(pool_query, target_pool);
task.Wait();

if (task->return_code_ != 0) {
  std::string error = task->error_message_.str();
  std::cerr << "Operation failed: " << error << std::endl;
}
```

## Important Notes

1. **Pool ID for CreateTask**: All ChiMod CreateTask operations must use `chi::kAdminPoolId`, not the client's `pool_id_`.

2. **Admin Pool Name**: The admin pool name MUST always be "admin". Multiple admin pools are NOT supported.

3. **Admin Dependency**: The admin module is required by all other ChiMods and must be linked in all Chimaera applications.

4. **Future API**: Asynchronous operations return `chi::Future<TaskType>`. Call `.Wait()` on the future and access task data with `->`.

5. **Pool Queries**: Use `chi::PoolQuery::Local()` for local operations and `chi::PoolQuery::Remote(node_id)` for distributed operations.

6. **Thread Safety**: All operations are designed to be called from the main thread. Multi-threaded access requires external synchronization.
