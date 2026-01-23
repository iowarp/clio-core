# MOD_NAME ChiMod Documentation

## Overview

The MOD_NAME ChiMod serves as a template and example module for developing custom ChiMods within the Chimaera framework. It demonstrates various ChiMod patterns and provides testing functionality for concurrency primitives such as CoMutex and CoRwLock. This module is primarily used for development, testing, and as a reference implementation for new ChiMod development.

**Key Features:**
- Template for custom ChiMod development
- Custom operation support with configurable parameters
- CoMutex (Coroutine Mutex) testing and validation
- CoRwLock (Coroutine Reader-Writer Lock) testing
- Recursive task.Wait() testing functionality
- Configurable worker count and operation parameters

## CMake Integration

### External Projects

To use the MOD_NAME ChiMod in external projects:

```cmake
find_package(chimaera-MOD_NAME REQUIRED)
find_package(chimaera-admin REQUIRED)  # Always required
find_package(chimaera-core REQUIRED)

target_link_libraries(your_application
  chimaera::MOD_NAME_client     # MOD_NAME client library
  chimaera::admin_client        # Admin client (required)
  chimaera::cxx                 # Main chimaera library
  hshm::cxx                     # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}     # Threading support
)
```

### Required Headers

```cpp
#include <chimaera/chimaera.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/admin/admin_client.h>  // Required for CreateTask
```

## API Reference

### Client Class: `chimaera::MOD_NAME::Client`

The MOD_NAME client provides the primary interface for module operations and testing.

#### Constructor

```cpp
// Default constructor
Client()

// Constructor with pool ID
explicit Client(const chi::PoolId& pool_id)
```

#### Container Management

##### `AsyncCreate()`
Creates and initializes the MOD_NAME container asynchronously.

```cpp
chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                    const std::string& pool_name,
                                    const chi::PoolId& custom_pool_id)
```

**Parameters:**
- `pool_query`: Pool domain query (typically `chi::PoolQuery::Local()`)
- `pool_name`: Name for the pool
- `custom_pool_id`: Explicit pool ID for the container

**Returns:** Future for asynchronous completion checking

**Usage:**
```cpp
chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
const chi::PoolId pool_id = chi::PoolId(9000, 0);
chimaera::MOD_NAME::Client mod_client(pool_id);

auto pool_query = chi::PoolQuery::Local();
auto create_task = mod_client.AsyncCreate(pool_query, "my_mod_name", pool_id);
create_task.Wait();

if (create_task->GetReturnCode() != 0) {
  std::cerr << "MOD_NAME creation failed" << std::endl;
  return;
}
```

#### Custom Operations

##### `AsyncCustom()`
Executes a custom operation with configurable parameters asynchronously.

```cpp
chi::Future<CustomTask> AsyncCustom(const chi::PoolQuery& pool_query,
                                    const std::string& input_data,
                                    chi::u32 operation_id)
```

**Parameters:**
- `pool_query`: Pool domain query
- `input_data`: Input data string for the operation
- `operation_id`: Identifier for the type of operation to perform

**Returns:** Future for asynchronous completion checking. Access output data via `task->data_` after calling `Wait()`.

**Usage:**
```cpp
std::string input = "test data for processing";
auto custom_task = mod_client.AsyncCustom(pool_query, input, 1);
custom_task.Wait();

if (custom_task->result_code_ == 0) {
  std::cout << "Custom operation succeeded. Output: " << custom_task->data_.str() << std::endl;
} else {
  std::cout << "Custom operation failed with code: " << custom_task->result_code_ << std::endl;
}
```

#### Concurrency Testing Operations

##### `AsyncCoMutexTest()`
Tests CoMutex (Coroutine Mutex) functionality asynchronously.

```cpp
chi::Future<CoMutexTestTask> AsyncCoMutexTest(
    const chi::PoolQuery& pool_query,
    chi::u32 test_id, chi::u32 hold_duration_ms)
```

**Parameters:**
- `pool_query`: Pool domain query
- `test_id`: Identifier for the test instance
- `hold_duration_ms`: Duration to hold the mutex lock in milliseconds

**Returns:** Future for asynchronous completion checking

**Usage:**
```cpp
// Test CoMutex with 1 second hold duration
auto mutex_task = mod_client.AsyncCoMutexTest(pool_query, 1, 1000);
mutex_task.Wait();
std::cout << "CoMutex test result: " << mutex_task->result_ << std::endl;
```

##### `AsyncCoRwLockTest()`
Tests CoRwLock (Coroutine Reader-Writer Lock) functionality asynchronously.

```cpp
chi::Future<CoRwLockTestTask> AsyncCoRwLockTest(
    const chi::PoolQuery& pool_query,
    chi::u32 test_id, bool is_writer, chi::u32 hold_duration_ms)
```

**Parameters:**
- `pool_query`: Pool domain query
- `test_id`: Identifier for the test instance
- `is_writer`: True for write lock test, false for read lock test
- `hold_duration_ms`: Duration to hold the lock in milliseconds

**Returns:** Future for asynchronous completion checking

**Usage:**
```cpp
// Test read lock
auto read_task = mod_client.AsyncCoRwLockTest(pool_query, 1, false, 500);
read_task.Wait();

// Test write lock
auto write_task = mod_client.AsyncCoRwLockTest(pool_query, 2, true, 500);
write_task.Wait();

std::cout << "Read lock test result: " << read_task->result_ << std::endl;
std::cout << "Write lock test result: " << write_task->result_ << std::endl;
```

##### `AsyncWaitTest()`
Tests recursive task.Wait() functionality with specified depth.

```cpp
chi::Future<WaitTestTask> AsyncWaitTest(const chi::PoolQuery& pool_query,
                                        chi::u32 depth,
                                        chi::u32 test_id)
```

**Parameters:**
- `pool_query`: Pool routing information
- `depth`: Number of recursive calls to make
- `test_id`: Test identifier for tracking

**Returns:** Future for asynchronous completion checking


## Task Types

### CreateTask
Container creation task for the MOD_NAME module. This is an alias for `chimaera::admin::GetOrCreatePoolTask<CreateParams>`.

**Key Fields:**
- Inherits from `BaseCreateTask` with MOD_NAME-specific `CreateParams`
- Processed by admin module for pool creation
- Contains serialized MOD_NAME configuration parameters

### CustomTask
Custom operation task for demonstrating module-specific functionality.

**Key Fields:**
- `data_`: Input/output data string (INOUT)
- `operation_id_`: Operation type identifier (IN)
- `result_code_`: Operation result (OUT, 0 = success)

### CoMutexTestTask
Task for testing CoMutex functionality.

**Key Fields:**
- `test_id_`: Test instance identifier (IN)
- `hold_duration_ms_`: Duration to hold mutex lock in milliseconds (IN)
- `result_`: Test result code (OUT)

### CoRwLockTestTask
Task for testing CoRwLock functionality.

**Key Fields:**
- `test_id_`: Test instance identifier (IN)
- `is_writer_`: True for write lock, false for read lock (IN)
- `hold_duration_ms_`: Duration to hold lock in milliseconds (IN)
- `result_`: Test result code (OUT)

### WaitTestTask
Task for testing recursive task.Wait() functionality.

**Key Fields:**
- `depth_`: Number of recursive calls to make (IN)
- `test_id_`: Test identifier for tracking (IN)
- `result_`: Test result code (OUT)

### DestroyTask
Standard destruction task (alias for `chimaera::admin::DestroyTask`).

## Configuration

### CreateParams Structure
Configuration parameters for MOD_NAME container creation:

```cpp
struct CreateParams {
  std::string config_data_;     // Configuration data string
  chi::u32 worker_count_;       // Number of worker threads (default: 1)
  
  // Required: chimod library name for module manager
  static constexpr const char* chimod_lib_name = "chimaera_MOD_NAME";
}
```

**Parameter Guidelines:**
- **config_data_**: Custom configuration string for module behavior
- **worker_count_**: Number of worker threads for parallel processing (default: 1)

**Important:** The `chimod_lib_name` does NOT include the `_runtime` suffix as it is automatically appended by the module manager.

## Usage Examples

### Complete Module Setup and Testing
```cpp
#include <chimaera/chimaera.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/admin/admin_client.h>

int main() {
  // Initialize Chimaera client
  chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

  // Create admin client first (always required)
  const chi::PoolId admin_pool_id = chi::kAdminPoolId;
  chimaera::admin::Client admin_client(admin_pool_id);
  auto admin_task = admin_client.AsyncCreate(chi::PoolQuery::Local(), "admin", admin_pool_id);
  admin_task.Wait();

  // Create MOD_NAME client
  const chi::PoolId mod_pool_id = chi::PoolId(9000, 0);
  chimaera::MOD_NAME::Client mod_client(mod_pool_id);

  // Initialize MOD_NAME container
  auto create_task = mod_client.AsyncCreate(chi::PoolQuery::Local(), "my_mod_name", mod_pool_id);
  create_task.Wait();

  if (create_task->GetReturnCode() != 0) {
    std::cerr << "MOD_NAME creation failed" << std::endl;
    return 1;
  }

  // Test custom operations
  std::string input_data = "Hello, Chimaera!";
  auto custom_task = mod_client.AsyncCustom(chi::PoolQuery::Local(), input_data, 1);
  custom_task.Wait();

  if (custom_task->result_code_ == 0) {
    std::cout << "Custom operation successful!" << std::endl;
    std::cout << "Input: " << input_data << std::endl;
    std::cout << "Output: " << custom_task->data_.str() << std::endl;
  }

  return 0;
}
```

### Concurrency Testing
```cpp
// Test CoMutex functionality
std::cout << "Testing CoMutex..." << std::endl;
for (int i = 0; i < 5; ++i) {
  auto mutex_task = mod_client.AsyncCoMutexTest(chi::PoolQuery::Local(), i, 100);  // 100ms hold
  mutex_task.Wait();
  std::cout << "CoMutex test " << i << " result: " << mutex_task->result_ << std::endl;
}

// Test CoRwLock functionality
std::cout << "Testing CoRwLock..." << std::endl;

// Test multiple readers (should allow concurrency)
for (int i = 0; i < 3; ++i) {
  auto read_task = mod_client.AsyncCoRwLockTest(chi::PoolQuery::Local(), i, false, 200);  // Read lock, 200ms
  read_task.Wait();
  std::cout << "Read lock test " << i << " result: " << read_task->result_ << std::endl;
}

// Test exclusive writer (should serialize with other operations)
auto write_task = mod_client.AsyncCoRwLockTest(chi::PoolQuery::Local(), 100, true, 300);  // Write lock, 300ms
write_task.Wait();
std::cout << "Write lock test result: " << write_task->result_ << std::endl;
```

### Asynchronous Operations
```cpp
// Example of using asynchronous operations for parallel execution
std::vector<chi::Future<CustomTask>> tasks;

// Submit multiple async operations
for (int i = 0; i < 5; ++i) {
  std::string input = "Async operation " + std::to_string(i);
  auto task = mod_client.AsyncCustom(chi::PoolQuery::Local(),
                                    input, i);
  tasks.push_back(std::move(task));
}

// Wait for all tasks to complete and collect results
for (size_t i = 0; i < tasks.size(); ++i) {
  tasks[i].Wait();

  std::cout << "Task " << i << " completed:" << std::endl;
  std::cout << "  Result code: " << tasks[i]->result_code_ << std::endl;
  std::cout << "  Output data: " << tasks[i]->data_.str() << std::endl;
}
```

### Template for Custom ChiMod Development
```cpp
// Use MOD_NAME as a template for developing your own ChiMod

// 1. Copy the MOD_NAME directory structure
// 2. Rename files and classes from MOD_NAME to YourModuleName
// 3. Update CreateParams structure with your configuration
// 4. Replace CustomTask with your domain-specific tasks
// 5. Implement your module logic in the runtime

// Example custom task (replace CustomTask):
struct YourCustomTask : public chi::Task {
  IN your_input_type_ input_param_;
  OUT your_output_type_ output_param_;
  OUT chi::u32 result_code_;
  
  // Constructor and serialization methods...
};

// Update client methods to match your functionality:
class YourClient : public chi::ContainerClient {
public:
  your_return_type YourCustomMethod(params...) {
    // Your custom implementation
  }
};
```

## Dependencies

- **HermesShm**: Shared memory framework and IPC
- **Chimaera core runtime**: Base runtime objects and task framework
- **Admin ChiMod**: Required for pool creation and management
- **cereal**: Serialization library for network communication
- **Boost.Fiber** and **Boost.Context**: Coroutine support for CoMutex/CoRwLock

## Installation

1. Build Chimaera with the MOD_NAME module:
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

All operations are asynchronous and return `chi::Future<TaskType>`. Check task result codes after calling `Wait()`:

```cpp
auto task = mod_client.AsyncCustom(pool_query, input, 1);
task.Wait();

if (task->result_code_ != 0) {
  std::cerr << "Custom operation failed with code: " << task->result_code_ << std::endl;
}
```

## Development Guidelines

### Using MOD_NAME as a Template

1. **File Structure**: Copy the entire `modules/MOD_NAME/` directory structure
2. **Renaming**: Replace all instances of `MOD_NAME` with your module name
3. **Configuration**: Update `CreateParams` with your module-specific parameters
4. **Tasks**: Replace or extend the example tasks with your domain-specific operations
5. **Client API**: Implement methods that make sense for your use case
6. **Runtime Logic**: Implement the actual processing logic in the runtime files

### Best Practices

1. **Naming Convention**: Use descriptive names for your module and operations
2. **Parameter Validation**: Always validate input parameters in tasks
3. **Error Handling**: Provide meaningful error codes and messages
4. **Documentation**: Document your API thoroughly following this template
5. **Testing**: Use the testing patterns demonstrated in MOD_NAME
6. **Resource Management**: Always clean up resources and memory properly

### Task Design Patterns

1. **Standard Tasks**: Request-response pattern with input/output parameters
2. **Long-Running**: Tasks that may take significant time to complete
3. **Batch Operations**: Tasks that process multiple items efficiently

## Important Notes

1. **Template Purpose**: MOD_NAME is primarily a template and testing module, not a production service.

2. **Admin Dependency**: The MOD_NAME module requires the admin module to be initialized first.

3. **Concurrency Testing**: The CoMutex and CoRwLock tests are useful for validating runtime behavior.

4. **Thread Safety**: Operations are designed for single-threaded access per client instance.

5. **Development Template**: Use this module as a starting point for custom ChiMod development.

6. **Async-Only API**: All client operations are asynchronous and return `chi::Future<TaskType>`. Call `Wait()` to block for completion.