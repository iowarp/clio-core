# Chimaera Module Development Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Coding Style](#coding-style)
4. [Module Structure](#module-structure)
    - [Task Definition](#task-definition-mod_name_tasksh)
    - [Client Implementation](#client-implementation-mod_name_clienthcc)
    - [Runtime Container](#runtime-container-mod_name_runtimehcc)
    - [Execution Modes and Dynamic Scheduling](#execution-modes-and-dynamic-scheduling)
5. [Configuration and Code Generation](#configuration-and-code-generation)
6. [Task Development](#task-development)
7. [Synchronization Primitives](#synchronization-primitives)
8. [Pool Query and Task Routing](#pool-query-and-task-routing)
9. [Client-Server Communication](#client-server-communication)
10. [Memory Management](#memory-management)
    - [CHI_CLIENT Buffer Allocation](#chi_client-buffer-allocation)
    - [Shared-Memory Compatible Data Structures](#shared-memory-compatible-data-structures)
11. [Build System Integration](#build-system-integration)
12. [External ChiMod Development](#external-chimod-development)
13. [Example Module](#example-module)

## Overview

Chimaera modules (ChiMods) are dynamically loadable components that extend the runtime with new functionality. Each module consists of:
- **Client library**: Minimal code for task submission from user processes
- **Runtime library**: Server-side execution logic
- **Task definitions**: Shared structures for client-server communication
- **Configuration**: YAML metadata describing the module

**Header Organization**: All ChiMod headers are organized under the namespace directory structure (`include/[namespace]/[module_name]/`) to provide clear namespace separation and prevent header conflicts.

## Architecture

### Core Principles
1. **Client-Server Separation**: Clients only submit tasks; runtime handles all logic
2. **Shared Memory Communication**: Tasks are allocated in shared memory segments
3. **Task-Based Processing**: All operations are expressed as tasks with methods
4. **Zero-Copy Design**: Data stays in shared memory; only pointers are passed

### Key Components
```
ChiMod/
├── include/
│   └── [namespace]/
│       └── MOD_NAME/
│           ├── MOD_NAME_client.h     # Client API
│           ├── MOD_NAME_runtime.h    # Runtime container
│           ├── MOD_NAME_tasks.h      # Task definitions
│           └── autogen/
│               └── MOD_NAME_methods.h    # Method constants
├── src/
│   ├── MOD_NAME_client.cc        # Client implementation
│   ├── MOD_NAME_runtime.cc       # Runtime implementation
│   └── autogen/
│       └── MOD_NAME_lib_exec.cc  # Auto-generated virtual method implementations
├── chimaera_mod.yaml              # Module configuration
└── CMakeLists.txt                 # Build configuration
```

**Include Directory Structure:**
- All ChiMod headers are organized under the namespace directory
- Structure: `include/[namespace]/[module_name]/`
- Example: Admin headers are in `include/[namespace]/admin/` (where `[namespace]` is the namespace from `chimaera_repo.yaml`)
- Headers follow naming pattern: `[module_name]_[type].h`
- Auto-generated headers are in the `autogen/` subdirectory
- **Note**: The namespace comes from `chimaera_repo.yaml` and the chimod directory name doesn't need to match the namespace

## Coding Style

### General Guidelines
1. **Namespace**: All module code under `chimaera::MOD_NAME`
2. **Naming Conventions**:
   - Classes: `PascalCase` (e.g., `CustomTask`)
   - Methods: `PascalCase` for public, `camelCase` for private
   - Variables: `snake_case_` with trailing underscore for members
   - Constants: `kConstantName`
   - Enums: `kEnumValue`

3. **Header Guards**: Use `#ifndef MOD_NAME_COMPONENT_H_`
4. **Includes**: System headers first, then library headers, then local headers
5. **Comments**: Use Doxygen-style comments for public APIs

### Code Formatting
```cpp
namespace chimaera::MOD_NAME {

/**
 * Brief description
 * 
 * Detailed description if needed
 * @param param_name Parameter description
 * @return Return value description
 */
class ExampleClass {
 public:
  // Public methods
  void PublicMethod();
  
 private:
  // Private members with trailing underscore
  u32 member_variable_;
};

}  // namespace chimaera::MOD_NAME
```

## Module Structure

### Task Definition (MOD_NAME_tasks.h)

Task definition patterns:

1. **CreateParams Structure**: Define configuration parameters for container creation
   - CreateParams use cereal serialization and do NOT require allocator-based constructors
   - Only need default constructor and parameter-based constructors
   - Allocator is NOT passed to CreateParams - it's handled internally by BaseCreateTask
2. **CreateTask Template**: Use GetOrCreatePoolTask template for container creation (non-admin modules)
3. **Custom Tasks**: Define custom tasks with SHM/Emplace constructors and HSHM data members

```cpp
#ifndef MOD_NAME_TASKS_H_
#define MOD_NAME_TASKS_H_

#include <chimaera/chimaera.h>
#include <[namespace]/MOD_NAME/autogen/MOD_NAME_methods.h>
// Include admin tasks for GetOrCreatePoolTask
#include <[namespace]/admin/admin_tasks.h>

namespace chimaera::MOD_NAME {

/**
 * CreateParams for MOD_NAME chimod
 * Contains configuration parameters for MOD_NAME container creation
 */
struct CreateParams {
  // MOD_NAME-specific parameters
  std::string config_data_;
  chi::u32 worker_count_;

  // Required: chimod library name for module manager
  static constexpr const char* chimod_lib_name = "chimaera_MOD_NAME";

  // Default constructor
  CreateParams() : worker_count_(1) {}

  // Constructor with parameters
  CreateParams(const std::string& config_data = "",
               chi::u32 worker_count = 1)
      : config_data_(config_data), worker_count_(worker_count) {}

  // Serialization support for cereal
  template<class Archive>
  void serialize(Archive& ar) {
    ar(config_data_, worker_count_);
  }

  /**
   * Load configuration from PoolConfig (for compose mode)
   * Required for compose feature support
   * @param pool_config Pool configuration from compose section
   */
  void LoadConfig(const chi::PoolConfig& pool_config) {
    // Parse YAML config string
    YAML::Node config = YAML::Load(pool_config.config_);

    // Load module-specific parameters from YAML
    if (config["config_data"]) {
      config_data_ = config["config_data"].as<std::string>();
    }
    if (config["worker_count"]) {
      worker_count_ = config["worker_count"].as<chi::u32>();
    }
  }
};

/**
 * CreateTask - Initialize the MOD_NAME container
 * Type alias for GetOrCreatePoolTask with CreateParams (uses kGetOrCreatePool method)
 * Non-admin modules should use GetOrCreatePoolTask instead of BaseCreateTask
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * Custom operation task
 */
struct CustomTask : public chi::Task {
  // Task-specific data using HSHM macros
  INOUT chi::string data_;      // Input/output string
  IN chi::u32 operation_id_;     // Input parameter
  OUT chi::u32 result_code_;     // Output result

  // SHM constructor
  explicit CustomTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), 
        data_(alloc), 
        operation_id_(0), 
        result_code_(0) {}

  // Emplace constructor
  explicit CustomTask(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
      const chi::TaskId &task_id,
      const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query,
      const std::string &data,
      chi::u32 operation_id)
      : chi::Task(alloc, task_id, pool_id, pool_query, 10),
        data_(alloc, data),
        operation_id_(operation_id),
        result_code_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kCustom;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }
};

}  // namespace chimaera::MOD_NAME

#endif  // MOD_NAME_TASKS_H_
```

### Client Implementation (MOD_NAME_client.h/cc)

The client provides an **async-only API** for task submission. All operations return `chi::Future<TaskType>` objects:

```cpp
#ifndef MOD_NAME_CLIENT_H_
#define MOD_NAME_CLIENT_H_

#include <chimaera/chimaera.h>
#include <[namespace]/MOD_NAME/MOD_NAME_tasks.h>

namespace chimaera::MOD_NAME {

class Client : public chi::ContainerClient {
 public:
  Client() = default;
  explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Async Create operation - returns Future for task completion
   * Caller must call task.Wait() and check GetReturnCode()
   */
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery& pool_query,
      const std::string& pool_name,
      const chi::PoolId& custom_pool_id,
      const CreateParams& params = CreateParams()) {
    auto* ipc_manager = CHI_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name,  // ChiMod name from CreateParams
        pool_name,              // User-provided pool name
        custom_pool_id,         // Target pool ID to create
        this,                   // Client pointer for PostWait callback
        params);                // CreateParams with configuration

    // Submit to runtime and return Future
    return ipc_manager->Send(task);
  }

  /**
   * Async Custom operation - example of a typical async method
   */
  chi::Future<CustomTask> AsyncCustom(
      const chi::PoolQuery& pool_query,
      const std::string& input_data,
      chi::u32 operation_id) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CustomTask>(
        chi::CreateTaskId(),
        pool_id_,           // Use client's pool_id_ for non-Create operations
        pool_query,
        input_data,
        operation_id);
    return ipc_manager->Send(task);
  }
};

}  // namespace chimaera::MOD_NAME

#endif  // MOD_NAME_CLIENT_H_
```

**Usage Pattern:**
```cpp
// Create client and initialize
chimaera::MOD_NAME::Client client;
const chi::PoolId pool_id(7000, 0);

// Async create
auto create_task = client.AsyncCreate(chi::PoolQuery::Dynamic(), "my_pool", pool_id);
create_task.Wait();

if (create_task->GetReturnCode() != 0) {
    std::cerr << "Create failed" << std::endl;
    return;
}

// Perform operations
auto op_task = client.AsyncCustom(chi::PoolQuery::Local(), "data", 1);
op_task.Wait();

// Access results
std::cout << "Result: " << op_task->result_code_ << std::endl;
```

### ChiMod CreateTask Pool Assignment Requirements

**CRITICAL**: All ChiMod clients implementing Create functions MUST use the explicit `chi::kAdminPoolId` variable when constructing CreateTask operations. You CANNOT use `pool_id_` for CreateTask operations.

#### Why This is Required

CreateTask operations are actually GetOrCreatePoolTask operations that must be processed by the admin ChiMod to create or find the target pool. The `pool_id_` variable is not initialized until after the Create operation completes successfully.

#### Correct Usage
```cpp
// CORRECT: Always use chi::kAdminPoolId for CreateTask
auto task = ipc_manager->NewTask<CreateTask>(
    chi::CreateTaskId(),
    chi::kAdminPoolId,          // REQUIRED: Use admin pool for CreateTask
    pool_query,
    CreateParams::chimod_lib_name,
    pool_name,
    pool_id_,                   // Target pool ID to create
    params);
```

#### Incorrect Usage
```cpp
// WRONG: Never use pool_id_ for CreateTask operations
auto task = ipc_manager->NewTask<CreateTask>(
    chi::CreateTaskId(),
    pool_id_,                   // WRONG: pool_id_ is not initialized yet
    pool_query,
    CreateParams::chimod_lib_name,
    pool_name,
    pool_id_,
    params);
```

#### Key Points

1. **Admin Pool Processing**: CreateTask is a GetOrCreatePoolTask that must be handled by the admin pool
2. **Uninitialized Variable**: `pool_id_` is not set until after Create completes
3. **Universal Requirement**: This applies to ALL ChiMod clients, including admin, bdev, and custom modules
4. **Create Responsibility**: Create operations are responsible for allocating new pool IDs using the admin pool

### ChiMod Name Requirements

**CRITICAL**: All ChiMod clients MUST use `CreateParams::chimod_lib_name` instead of hardcoding module names.

#### Correct Usage
```cpp
// CORRECT: Use CreateParams::chimod_lib_name
auto task = ipc_manager->NewTask<CreateTask>(
    chi::CreateTaskId(),
    chi::kAdminPoolId,
    pool_query,
    CreateParams::chimod_lib_name,  // Dynamic reference to CreateParams
    pool_name,
    pool_id,
    params);
```

#### Incorrect Usage
```cpp
// WRONG: Never hardcode module names
auto task = ipc_manager->NewTask<CreateTask>(
    chi::CreateTaskId(),
    chi::kAdminPoolId,
    pool_query,
    "chimaera_MOD_NAME",  // Hardcoded name breaks flexibility
    pool_name,
    pool_id,
    params);
```

#### Why This is Required

1. **Namespace Flexibility**: Allows ChiMods to work with different namespace configurations
2. **Single Source of Truth**: The module name is defined once in CreateParams
3. **External ChiMods**: Essential for external ChiMods using custom namespaces
4. **Maintainability**: Changes to module names only require updating CreateParams

#### Implementation Pattern

All non-admin ChiMods using `GetOrCreatePoolTask` must follow this pattern:

```cpp
// In AsyncCreate method
auto task = ipc_manager->NewTask<CreateTask>(
    chi::CreateTaskId(),
    chi::kAdminPoolId,                    // Always use admin pool
    pool_query,
    CreateParams::chimod_lib_name,        // REQUIRED: Use static member
    pool_name,                            // Pool identifier
    pool_id,                              // Target pool ID
    /* ...CreateParams arguments... */);
```

**Note**: The admin ChiMod uses `BaseCreateTask` directly and doesn't require the chimod name parameter.

### Runtime Container (MOD_NAME_runtime.h/cc)

The runtime container executes tasks server-side:

```cpp
#ifndef MOD_NAME_RUNTIME_H_
#define MOD_NAME_RUNTIME_H_

#include <chimaera/chimaera.h>
#include <[namespace]/MOD_NAME/MOD_NAME_tasks.h>

namespace chimaera::MOD_NAME {

class Container : public chi::Container {
 public:
  Container() = default;
  ~Container() override = default;

  /**
   * Initialize container with pool information (REQUIRED)
   * This is called by the framework before Create is called
   */
  void Init(const chi::PoolId& pool_id, const std::string& pool_name) override {
    // Call base class initialization
    chi::Container::Init(pool_id, pool_name);

    // Initialize the client for this ChiMod
    client_ = Client(pool_id);
  }

  /**
   * Create the container (Method::kCreate)
   * This method creates queues and sets up container resources
   * NOTE: Container is already initialized via Init() before Create is called
   */
  void Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx) {
    // Container is already initialized via Init() before Create is called
    // Do NOT call Init() here

    // Additional container-specific initialization logic here
    std::cout << "Container created and initialized for pool: " << pool_name_
              << " (ID: " << pool_id_ << ")" << std::endl;
  }

  /**
   * Custom operation (Method::kCustom)
   */
  void Custom(hipc::FullPtr<CustomTask> task, chi::RunContext& ctx) {
    // Process the operation
    std::string result = processData(task->data_.str(),
                                    task->operation_id_);
    task->data_ = chi::priv::string(main_allocator_, result);
    task->result_code_ = 0;
    // Task completion is handled by the framework
  }

 private:
  std::string processData(const std::string& input, u32 op_id) {
    // Business logic here
    return input + "_processed";
  }
};

}  // namespace chimaera::MOD_NAME

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(chimaera::MOD_NAME::Container)

#endif  // MOD_NAME_RUNTIME_H_
```

### Execution Modes and Dynamic Scheduling

The Chimaera runtime supports two execution modes through the `ExecMode` enum in `RunContext`, enabling sophisticated task routing patterns:

#### ExecMode Overview

```cpp
/**
 * Execution mode for task processing
 */
enum class ExecMode : u32 {
  kExec = 0,              /**< Normal task execution (default) */
  kDynamicSchedule = 1    /**< Dynamic scheduling - route after execution */
};
```

The execution mode is accessible through the `RunContext` parameter passed to all task methods:

```cpp
void YourMethod(hipc::FullPtr<YourTask> task, chi::RunContext& rctx) {
  // Check execution mode
  if (rctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    // Dynamic scheduling logic - modify task routing
    task->pool_query_ = chi::PoolQuery::Broadcast();
    return;  // Return early - task will be re-routed
  }

  // Normal execution logic (kExec mode)
  // ... perform actual work ...
}
```

#### kExec Mode (Default)

**Purpose**: Normal task execution mode where tasks are processed completely and then marked as finished.

**Behavior**:
- Tasks execute their full logic
- Results are written to task output parameters
- Worker calls `EndTask()` after execution completes
- Task is marked as completed and cleaned up

**When to use**: The default mode for all standard task processing.

#### kDynamicSchedule Mode

**Purpose**: Two-phase execution where the first execution determines routing, then the task is re-routed and executed again.

**Behavior**:
1. Worker sets `exec_mode = kDynamicSchedule` before first execution
2. Task method examines state and modifies `task->pool_query_` for routing
3. Task returns early without performing full execution
4. Worker calls `RerouteDynamicTask()` instead of `EndTask()`
5. Task is re-routed using the updated `pool_query_`
6. Task executes again in normal `kExec` mode with the new routing

**When to use**:
- Tasks that need runtime-dependent routing decisions
- Cache optimization patterns (check local, then broadcast if not found)
- Conditional distributed execution based on state

#### Example: GetOrCreatePool with Dynamic Scheduling

The admin ChiMod's `GetOrCreatePool` method demonstrates the canonical dynamic scheduling pattern:

```cpp
void Runtime::GetOrCreatePool(
    hipc::FullPtr<chimaera::admin::GetOrCreatePoolTask<chimaera::admin::CreateParams>> task,
    chi::RunContext &rctx) {

  auto *pool_manager = CHI_POOL_MANAGER;
  std::string pool_name = task->pool_name_.str();

  // PHASE 1: Dynamic scheduling - determine routing
  if (rctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    // Check if pool exists locally first
    chi::PoolId existing_pool_id = pool_manager->FindPoolByName(pool_name);

    if (!existing_pool_id.IsNull()) {
      // Pool exists locally - route to local execution only
      HLOG(kDebug, "Admin: Pool '{}' found locally (ID: {}), using Local query",
            pool_name, existing_pool_id);
      task->pool_query_ = chi::PoolQuery::Local();
    } else {
      // Pool doesn't exist - broadcast creation to all nodes
      HLOG(kDebug, "Admin: Pool '{}' not found locally, broadcasting creation",
            pool_name);
      task->pool_query_ = chi::PoolQuery::Broadcast();
    }
    return;  // Return early - worker will re-route task
  }

  // PHASE 2: Normal execution - actually create/get the pool
  HLOG(kDebug, "Admin: Executing GetOrCreatePool task - ChiMod: {}, Pool: {}",
        task->chimod_name_.str(), pool_name);

  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    if (!pool_manager->CreatePool(task.Cast<chi::Task>(), &rctx)) {
      task->return_code_ = 2;
      task->error_message_ = "Failed to create or get pool via PoolManager";
      return;
    }

    task->return_code_ = 0;
    pools_created_++;

    HLOG(kDebug, "Admin: Pool operation completed successfully - ID: {}, Name: {}",
          task->new_pool_id_, pool_name);

  } catch (const std::exception &e) {
    task->return_code_ = 99;
    task->error_message_ = chi::priv::string(
        CHI_IPC->GetMainAlloc(),
        std::string("Exception during pool creation: ") + e.what());
    HLOG(kError, "Admin: Pool creation failed with exception: {}", e.what());
  }
}
```

#### Using Dynamic() PoolQuery

The `PoolQuery::Dynamic()` factory method triggers dynamic scheduling:

```cpp
// Client code - request dynamic routing
auto pool_query = chi::PoolQuery::Dynamic();
client.Create(pool_query, "my_pool_name", pool_id);
```

**What happens internally:**
1. Worker recognizes `Dynamic()` pool query
2. Sets `rctx.exec_mode = ExecMode::kDynamicSchedule`
3. Routes task to local node first
4. Task method checks cache and updates `pool_query_`
5. Worker re-routes with updated query
6. Task executes again in normal mode with correct routing

#### Benefits of Dynamic Scheduling

**Performance Optimization:**
- Avoids redundant operations (e.g., pool creation when pool already exists)
- Reduces network overhead by checking local state first
- Enables intelligent routing based on runtime conditions

**Cache Optimization Pattern:**
```cpp
// Check local cache first
if (rctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
  if (LocalCacheHas(resource_id)) {
    task->pool_query_ = chi::PoolQuery::Local();  // Found locally
  } else {
    task->pool_query_ = chi::PoolQuery::Broadcast();  // Need to fetch
  }
  return;
}

// Normal execution with optimized routing
auto resource = GetResource(resource_id);
```

**State-Dependent Routing:**
```cpp
// Route based on runtime conditions
if (rctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
  if (ShouldExecuteDistributed(task)) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
  } else {
    task->pool_query_ = chi::PoolQuery::Local();
  }
  return;
}
```

#### Implementation Guidelines

**DO:**
- ✅ Check `rctx.exec_mode` at the start of your method
- ✅ Return early after modifying `pool_query_` in dynamic mode
- ✅ Keep dynamic scheduling logic lightweight (fast checks only)
- ✅ Use dynamic scheduling for cache optimization patterns

**DON'T:**
- ❌ Perform expensive operations in dynamic scheduling mode
- ❌ Modify task output parameters in dynamic scheduling mode
- ❌ Call `Wait()` or spawn subtasks in dynamic scheduling mode
- ❌ Use dynamic scheduling for simple operations that don't need routing optimization

#### Worker Implementation Details

The worker automatically handles dynamic scheduling:

```cpp
// Worker::ExecTask() logic
if (run_ctx->exec_mode == ExecMode::kDynamicSchedule) {
  // After task returns, call RerouteDynamicTask instead of EndTask
  RerouteDynamicTask(task_ptr, run_ctx);
  return;
}

// Normal mode - end task after execution
EndTask(task_ptr, run_ctx);
```

The `RerouteDynamicTask()` method:
1. Resets task flags (`TASK_STARTED`, `TASK_ROUTED`)
2. Re-routes task using updated `pool_query_`
3. Sets `exec_mode = kExec` for next execution
4. Schedules task for execution again

## Configuration and Code Generation

### Overview
Chimaera uses a two-level configuration system with automated code generation:

1. **chimaera_repo.yaml**: Repository-wide configuration (namespace, version, etc.)
2. **chimaera_mod.yaml**: Module-specific configuration (method IDs, metadata)
3. **chi_refresh_repo**: Utility script that generates autogen files from YAML configurations

### chimaera_repo.yaml
Located at `chimods/chimaera_repo.yaml`, this file defines repository-wide settings:

```yaml
# Repository Configuration
namespace: chimaera        # MUST match namespace in all chimaera_mod.yaml files
version: 1.0.0
description: "Chimaera Runtime ChiMod Repository"

# Module discovery - directories to scan for ChiMods
modules:
  - MOD_NAME
  - admin  
  - bdev
```

**Key Requirements:**
- The `namespace` field MUST be identical in both chimaera_repo.yaml and all chimaera_mod.yaml files
- Used by build system for CMake package generation and installation paths
- Determines export target names: `${namespace}::${module}_runtime`, `${namespace}::${module}_client`

### chimaera_mod.yaml
Each ChiMod must have its own configuration file specifying methods and metadata:

```yaml
# MOD_NAME ChiMod Configuration
module_name: MOD_NAME
namespace: chimaera        # MUST match chimaera_repo.yaml namespace
version: 1.0.0

# Inherited Methods (fixed IDs)
kCreate: 0        # Container creation (required)
kDestroy: 1       # Container destruction (required)
kNodeFailure: -1  # Not implemented (-1 means disabled)
kRecover: -1      # Not implemented 
kMigrate: -1      # Not implemented
kUpgrade: -1      # Not implemented

# Custom Methods (start from 10, use sequential IDs)
kCustom: 10       # Custom operation method
kCoMutexTest: 20  # CoMutex synchronization testing method
kCoRwLockTest: 21 # CoRwLock reader-writer synchronization testing method
```

**Method ID Assignment Rules:**
- **0-9**: Reserved for system methods (kCreate=0, kDestroy=1, etc.)
- **10+**: Custom methods (assign sequential IDs starting from 10)
- **Disabled methods**: Use -1 to disable inherited methods not implemented
- **Consistency**: Once assigned, never change method IDs (breaks compatibility)

### chi_refresh_repo Utility

The `chi_refresh_repo` utility automatically generates autogen files from YAML configurations.

#### Usage
```bash
# From project root, regenerate all autogen files
./build/bin/chi_refresh_repo chimods

# The utility will:
# 1. Read chimaera_repo.yaml for global settings
# 2. Scan each module's chimaera_mod.yaml 
# 3. Generate MOD_NAME_methods.h with method constants
# 4. Generate MOD_NAME_lib_exec.cc with virtual method dispatch
```

#### Generated Files
For each ChiMod, the utility generates:

1. **`include/[namespace]/MOD_NAME/autogen/MOD_NAME_methods.h`**:
   ```cpp
   namespace chimaera::MOD_NAME {
   namespace Method {
   GLOBAL_CONST chi::u32 kCreate = 0;
   GLOBAL_CONST chi::u32 kDestroy = 1;
   GLOBAL_CONST chi::u32 kCustom = 10;
   GLOBAL_CONST chi::u32 kCoMutexTest = 20;
   }  // namespace Method
   }  // namespace chimaera::MOD_NAME
   ```

2. **`src/autogen/MOD_NAME_lib_exec.cc`**:
   - Virtual method dispatch (Runtime::Run, etc.)
   - Task serialization support (SaveIn/Out, LoadIn/Out)
   - Memory management (Del, NewCopy, Aggregate)

#### When to Run chi_refresh_repo
**ALWAYS** run chi_refresh_repo when:
- Adding new methods to chimaera_mod.yaml
- Changing method IDs or names
- Adding new ChiMods to the repository
- Modifying namespace or version information

#### Important Notes
- **Never manually edit autogen files** - they are overwritten by chi_refresh_repo
- **Run chi_refresh_repo before building** after YAML changes
- **Commit autogen files to git** so other developers don't need to regenerate
- **Method IDs are permanent** - changing them breaks binary compatibility

### Workflow Summary
1. Define methods in `chimaera_mod.yaml` with sequential IDs
2. Implement corresponding methods in `MOD_NAME_runtime.h/cc`
3. Run `./build/bin/chi_refresh_repo chimods` to generate autogen files
4. Build project with `make` - autogen files provide the dispatch logic
5. Autogen files handle virtual method routing, serialization, and memory management

This automated approach ensures consistency across all ChiMods and reduces boilerplate code maintenance.

## Task Development

### Task Requirements
1. **Inherit from chi::Task**: All tasks must inherit the base Task class
2. **Two Constructors**: SHM and emplace constructors are mandatory
3. **Serializable Types**: Use HSHM types (chi::string, chi::vector, etc.) for member variables
4. **Method Assignment**: Set the method_ field to identify the operation
5. **FullPtr Usage**: All task method signatures use `hipc::FullPtr<TaskType>` instead of raw pointers
6. **Copy Method**: Optional - implement for tasks that need to be replicated across nodes
7. **Aggregate Method**: Optional - implement for tasks that need to combine results from replicas

### Optional Task Methods: Copy and Aggregate

For tasks that will be distributed across multiple nodes or need to combine results from multiple executions, you can optionally implement `Copy()` and `Aggregate()` methods.

#### Copy Method

The `Copy()` method is used to create a deep copy of a task, typically when distributing work across multiple nodes. This is useful for:
- Remote task execution via networking
- Task replication for fault tolerance
- Creating independent task replicas with separate data

**Signature:**
```cpp
void Copy(const hipc::FullPtr<YourTask> &other);
```

**Implementation Pattern:**
```cpp
struct WriteTask : public chi::Task {
  IN Block block_;
  IN hipc::ShmPtr<> data_;
  IN size_t length_;
  OUT chi::u64 bytes_written_;

  /**
   * Copy from another WriteTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<WriteTask> &other) {
    // Copy task-specific fields only
    // Base Task fields are copied automatically by NewCopy
    block_ = other->block_;
    data_ = other->data_;
    length_ = other->length_;
    bytes_written_ = other->bytes_written_;
  }
};
```

**Key Points:**
- **DO NOT** call `chi::Task::Copy()` - base fields are copied automatically by autogenerated NewCopy
- Copy only task-specific fields from the source task
- The destination task (`this`) is already constructed - don't call constructors
- For pointer fields, decide if you need deep or shallow copy based on ownership
- NewCopy in autogen code handles calling base Task::Copy before your Copy method

#### Aggregate Method

The `Aggregate()` method combines results from multiple task replicas into a single result. This is commonly used for:
- Combining results from distributed task execution
- Merging partial results from parallel operations
- Accumulating metrics from multiple nodes

**Signature:**
```cpp
void Aggregate(const hipc::FullPtr<YourTask> &other);
```

**Implementation Patterns:**

**Pattern 1: Last-Writer-Wins (Simple Override)**
```cpp
struct WriteTask : public chi::Task {
  IN Block block_;
  IN hipc::ShmPtr<> data_;
  OUT chi::u64 bytes_written_;

  /**
   * Aggregate results from another WriteTask
   * For write operations, we typically just copy the result from the completed replica
   */
  void Aggregate(const hipc::FullPtr<WriteTask> &other) {
    // Simply copy the result - last writer wins
    Copy(other);
  }
};
```

**Pattern 2: Accumulation (Sum/Max/Min)**
```cpp
struct GetStatsTask : public chi::Task {
  OUT chi::u64 total_bytes_;
  OUT chi::u64 operation_count_;
  OUT chi::u64 max_latency_us_;

  /**
   * Aggregate statistics from multiple replicas
   * Accumulate totals and find maximum values
   */
  void Aggregate(const hipc::FullPtr<GetStatsTask> &other) {
    // Sum cumulative metrics
    total_bytes_ += other->total_bytes_;
    operation_count_ += other->operation_count_;

    // Take maximum for latency
    max_latency_us_ = std::max(max_latency_us_, other->max_latency_us_);
  }
};
```

**Pattern 3: List/Vector Merging**
```cpp
struct AllocateBlocksTask : public chi::Task {
  OUT chi::ipc::vector<Block> blocks_;

  /**
   * Aggregate block allocations from multiple replicas
   * Combine all allocated blocks into a single list
   */
  void Aggregate(const hipc::FullPtr<AllocateBlocksTask> &other) {
    // Append blocks from other task to this task's list
    blocks_.insert(blocks_.end(),
                   other->blocks_.begin(),
                   other->blocks_.end());
  }
};
```

**Pattern 4: Custom Logic**
```cpp
struct QueryTask : public chi::Task {
  OUT chi::ipc::vector<Result> results_;
  OUT chi::u32 error_count_;

  /**
   * Aggregate query results with custom deduplication
   */
  void Aggregate(const hipc::FullPtr<QueryTask> &other) {
    // Merge results with deduplication
    for (const auto &result : other->results_) {
      if (!ContainsResult(results_, result)) {
        results_.push_back(result);
      }
    }

    // Accumulate error counts
    error_count_ += other->error_count_;
  }

private:
  bool ContainsResult(const chi::ipc::vector<Result> &vec, const Result &r) {
    // Custom deduplication logic
    return std::find(vec.begin(), vec.end(), r) != vec.end();
  }
};
```

#### When to Implement Copy and Aggregate

**Implement Copy when:**
- Your task will be sent to remote nodes for execution
- Task data needs to be replicated for fault tolerance
- You need independent copies with separate data ownership

**Implement Aggregate when:**
- Your task returns results that can be combined (sums, lists, statistics)
- You're using distributed execution patterns (e.g., map-reduce)
- Multiple replicas produce partial results that need merging

**Skip Copy and Aggregate when:**
- Tasks are only executed locally on a single node
- Results don't need to be combined across executions
- Tasks have no output parameters (side-effects only)
- Default shallow copy behavior is sufficient

#### Copy/Aggregate Usage in Networking

When tasks are sent across nodes using Send/Recv:

1. **Send Phase**: The `Copy()` method creates a replica of the origin task
   ```cpp
   hipc::FullPtr<Task> replica;
   container->NewCopy(task->method_, origin_task, replica, /* replica_flag */);
   // Internally calls task->Copy(origin_task)
   ```

2. **Recv Phase**: The `Aggregate()` method combines replica results back into the origin
   ```cpp
   container->Aggregate(task->method_, origin_task, replica);
   // Internally calls origin_task->Aggregate(replica)
   ```

3. **Autogeneration**: The code generator creates dispatcher methods that call your Copy/Aggregate implementations:
   ```cpp
   // In autogen/MOD_NAME_lib_exec.cc
   void NewCopy(Runtime* runtime, chi::u32 method,
                hipc::FullPtr<chi::Task> orig_task,
                hipc::FullPtr<chi::Task>& new_task,
                bool deep_copy) {
     switch (method) {
       case Method::kWrite: {
         auto orig = orig_task.Cast<WriteTask>();
         new_task = CHI_IPC->NewTask<WriteTask>(...);
         new_task.Cast<WriteTask>()->Copy(orig);
         break;
       }
     }
   }

   void Aggregate(Runtime* runtime, chi::u32 method,
                  hipc::FullPtr<chi::Task> task,
                  const hipc::FullPtr<chi::Task> &replica) {
     switch (method) {
       case Method::kWrite: {
         task.Cast<WriteTask>()->Aggregate(replica.Cast<WriteTask>());
         break;
       }
     }
   }
   ```

#### Complete Example: ReadTask with Copy and Aggregate

```cpp
struct ReadTask : public chi::Task {
  IN Block block_;
  OUT hipc::ShmPtr<> data_;
  INOUT size_t length_;
  OUT chi::u64 bytes_read_;

  /** SHM constructor */
  explicit ReadTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), length_(0), bytes_read_(0) {}

  /** Emplace constructor */
  explicit ReadTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                    const chi::TaskId &task_node,
                    const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query,
                    const Block &block,
                    hipc::ShmPtr<> data,
                    size_t length)
      : chi::Task(alloc, task_node, pool_id, pool_query, 10),
        block_(block), data_(data), length_(length), bytes_read_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRead;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Copy from another ReadTask
   * Used when creating replicas for remote execution
   */
  void Copy(const hipc::FullPtr<ReadTask> &other) {
    // Copy task-specific fields only
    // Base Task fields are copied automatically by NewCopy
    block_ = other->block_;
    data_ = other->data_;
    length_ = other->length_;
    bytes_read_ = other->bytes_read_;
  }

  /**
   * Aggregate results from replica
   * For read operations, simply copy the data from the completed replica
   */
  void Aggregate(const hipc::FullPtr<ReadTask> &other) {
    // For reads, we just take the result from the replica
    Copy(other);
  }
};
```

### Task Naming Conventions

**CRITICAL**: All task names MUST follow consistent naming patterns to ensure proper code generation and maintenance.

#### Required Naming Pattern

The naming convention enforces consistency across function names, task types, and method constants:

```
Function Name → Task Name → Method Constant
FunctionName()  → FunctionNameTask  → kFunctionName
```

#### Examples

**Correct Naming:**
```cpp
// Client Function: AsyncGetStats()
// Task: GetStatsTask
// Method: kGetStats

// In bdev_client.h (async-only API)
chi::Future<GetStatsTask> AsyncGetStats(const chi::PoolQuery& pool_query);

// In bdev_tasks.h
struct GetStatsTask : public chi::Task {
  OUT PerfMetrics metrics_;
  OUT chi::u64 remaining_size_;
  // ... constructors and methods
};

// In chimaera_mod.yaml
kGetStats: 14    # Get performance statistics

// In bdev_runtime.h
void GetStats(hipc::FullPtr<GetStatsTask> task, chi::RunContext& ctx);
```

**Incorrect Naming Examples:**
```cpp
// WRONG: Function and task names don't match
chi::Future<StatTask> AsyncGetStats(...);  // Task name doesn't match function
struct StatTask { ... };                    // Task name doesn't match function

// WRONG: Method constant doesn't match function  
GLOBAL_CONST chi::u32 kStat = 14;    // Method doesn't match function name

// WRONG: Runtime method doesn't match function
void Stat(hipc::FullPtr<StatTask> task, ...);  // Runtime method doesn't match
```

#### Naming Rules

1. **Client Function Names**: Use `Async` prefix with descriptive verbs (e.g., `AsyncGetStats`, `AsyncAllocateBlocks`, `AsyncWriteData`)
2. **Task Names**: Remove `Async` prefix and append "Task" (e.g., `GetStatsTask`, `AllocateBlocksTask`)
3. **Method Constants**: Prefix with "k" and match the base function name (e.g., `kGetStats`, `kAllocateBlocks`)
4. **Runtime Methods**: Match the base function name without `Async` prefix (e.g., `GetStats()`)

#### Backward Compatibility

When renaming tasks, provide backward compatibility aliases:

```cpp
// In bdev_tasks.h - provide alias for old name
using StatTask = GetStatsTask;  // Backward compatibility

// In autogen/bdev_methods.h - provide constant alias
GLOBAL_CONST chi::u32 kGetStats = 14;
GLOBAL_CONST chi::u32 kStat = kGetStats;  // Backward compatibility

// In bdev_runtime.h - provide wrapper methods
void GetStats(hipc::FullPtr<GetStatsTask> task, chi::RunContext& ctx);  // Primary
void Stat(hipc::FullPtr<StatTask> task, chi::RunContext& ctx) {         // Wrapper
  GetStats(task, ctx);
}
```

#### Benefits of Consistent Naming

1. **Code Generation**: Automated tools can reliably generate method dispatch code
2. **Maintenance**: Clear correlation between client functions and runtime implementations
3. **Documentation**: Self-documenting code with predictable naming patterns
4. **Debugging**: Easy to trace from client calls to runtime execution
5. **Testing**: Consistent patterns make it easier to write comprehensive tests

### Method System and Auto-Generated Files

#### Method Definitions (autogen/MOD_NAME_methods.h)
Method IDs are now defined as namespace constants instead of enum class values. This eliminates the need for static casting:

```cpp
#ifndef MOD_NAME_AUTOGEN_METHODS_H_
#define MOD_NAME_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>

namespace chimaera::MOD_NAME {

namespace Method {
  // Inherited methods
  GLOBAL_CONST chi::u32 kCreate = 0;
  GLOBAL_CONST chi::u32 kDestroy = 1;
  GLOBAL_CONST chi::u32 kNodeFailure = 2;
  GLOBAL_CONST chi::u32 kRecover = 3;
  GLOBAL_CONST chi::u32 kMigrate = 4;
  GLOBAL_CONST chi::u32 kUpgrade = 5;
  
  // Module-specific methods
  GLOBAL_CONST chi::u32 kCustom = 10;
}

} // namespace chimaera::MOD_NAME

#endif // MOD_NAME_AUTOGEN_METHODS_H_
```

**Key Changes:**
- **Namespace instead of enum class**: Use `Method::kMethodName` directly
- **GLOBAL_CONST values**: No more static casting required
- **Include chimaera.h**: Required for GLOBAL_CONST macro
- **Direct assignment**: `method_ = Method::kCreate;` (no casting)

#### BaseCreateTask Template System

For modules that need container creation functionality, use the BaseCreateTask template instead of implementing custom CreateTask. However, there are different approaches depending on whether your module is the admin module or a regular ChiMod:

##### GetOrCreatePoolTask vs BaseCreateTask Usage

**For Non-Admin Modules (Recommended Pattern):**

All non-admin ChiMods should use `GetOrCreatePoolTask` which is a specialized version of BaseCreateTask designed for external pool creation:

```cpp
#include <[namespace]/admin/admin_tasks.h>  // Include admin templates

namespace chimaera::MOD_NAME {

/**
 * CreateParams for MOD_NAME container creation
 */
struct CreateParams {
  // Module-specific configuration
  std::string config_data_;
  chi::u32 worker_count_;
  
  // Required: chimod library name
  static constexpr const char* chimod_lib_name = "chimaera_MOD_NAME";

  // Constructors
  CreateParams() : worker_count_(1) {}

  CreateParams(const std::string& config_data = "",
               chi::u32 worker_count = 1)
      : config_data_(config_data), worker_count_(worker_count) {}

  // Cereal serialization
  template<class Archive>
  void serialize(Archive& ar) {
    ar(config_data_, worker_count_);
  }
};

/**
 * CreateTask - Non-admin modules should use GetOrCreatePoolTask
 * This uses Method::kGetOrCreatePool and is designed for external pool creation
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

}  // namespace chimaera::MOD_NAME
```

**For Admin Module Only:**

The admin module itself uses BaseCreateTask directly with Method::kCreate:

```cpp
namespace chimaera::admin {

/**
 * CreateTask - Admin uses BaseCreateTask with Method::kCreate and IS_ADMIN=true
 */
using CreateTask = BaseCreateTask<CreateParams, Method::kCreate, true>;

}  // namespace chimaera::admin
```

#### BaseCreateTask Template Parameters

The BaseCreateTask template has three parameters with smart defaults:

```cpp
template <typename CreateParamsT, 
          chi::u32 MethodId = Method::kGetOrCreatePool, 
          bool IS_ADMIN = false>
struct BaseCreateTask : public chi::Task
```

**Template Parameters:**
1. **CreateParamsT**: Your module's parameter structure (required)
2. **MethodId**: Method ID for the task (default: `kGetOrCreatePool`)
3. **IS_ADMIN**: Whether this is an admin operation (default: `false`)

**GetOrCreatePoolTask Template:**

The `GetOrCreatePoolTask` template is a convenient alias that uses the optimal defaults for non-admin modules:

```cpp
template<typename CreateParamsT>
using GetOrCreatePoolTask = BaseCreateTask<CreateParamsT, Method::kGetOrCreatePool, false>;
```

**When to Use Each Pattern:**
- **GetOrCreatePoolTask**: For all non-admin ChiMods (recommended)
- **BaseCreateTask with Method::kCreate**: Only for admin module internal operations
- **BaseCreateTask with Method::kGetOrCreatePool**: Same as GetOrCreatePoolTask (not typically used directly)

#### BaseCreateTask Structure

BaseCreateTask provides a unified structure for container creation and pool operations:

```cpp
template <typename CreateParamsT, chi::u32 MethodId, bool IS_ADMIN>
struct BaseCreateTask : public chi::Task {
  // Pool operation parameters
  INOUT chi::string chimod_name_;     // ChiMod name for loading
  IN chi::string pool_name_;          // Target pool name
  INOUT chi::string chimod_params_;   // Serialized CreateParamsT
  INOUT chi::PoolId pool_id_;          // Input: requested ID, Output: actual ID
  
  // Results
  OUT chi::u32 result_code_;           // 0 = success, non-zero = error
  OUT chi::string error_message_;     // Error description if failed
  
  // Runtime flag set by template parameter
  volatile bool is_admin_;             // Set to IS_ADMIN template value
  
  // Serialization methods
  template<typename... Args>
  void SetParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc, Args &&...args);
  
  CreateParamsT GetParams(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc) const;
};
```

**Key Features:**
- **Single pool_id**: Serves as both input (requested) and output (result)
- **Serialized parameters**: `chimod_params_` stores serialized CreateParamsT
- **Error checking**: Use `result_code_ != 0` to check for failures
- **Template-driven behavior**: IS_ADMIN template parameter sets volatile variable
- **No static casting**: Direct method assignment using namespace constants

#### Usage Examples

**Non-Admin ChiMod Container Creation (Recommended):**
```cpp
// Use GetOrCreatePoolTask for all non-admin modules
using CreateTask = chimaera::admin::GetOrCreatePoolTask<MyCreateParams>;
```

**Admin Module Container Creation:**
```cpp
// Admin module uses BaseCreateTask with Method::kCreate and IS_ADMIN=true
using CreateTask = chimaera::admin::BaseCreateTask<AdminCreateParams, Method::kCreate, true>;
```

**Alternative (Not Recommended):**
```cpp
// Direct BaseCreateTask usage - GetOrCreatePoolTask is cleaner
using CreateTask = chimaera::admin::BaseCreateTask<MyCreateParams, Method::kGetOrCreatePool, false>;
```

#### Migration from Custom CreateTask

If you have existing custom CreateTask implementations, migrate to BaseCreateTask:

**Before (Custom Implementation):**
```cpp
struct CreateTask : public chi::Task {
  // Custom constructor implementations
  explicit CreateTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                      const chi::TaskId &task_id,
                      const chi::PoolId &pool_id,
                      const chi::PoolQuery &pool_query)
      : chi::Task(alloc, task_id, pool_id, pool_query, 0) {
    method_ = Method::kCreate;  // Static casting required
    // ... initialization code ...
  }
};
```

**After (GetOrCreatePoolTask - Recommended for Non-Admin Modules):**
```cpp
// Create params structure
struct CreateParams {
  static constexpr const char* chimod_lib_name = "chimaera_mymodule";
  // ... other params ...
  template<class Archive> void serialize(Archive& ar) { /* ... */ }
};

// Simple type alias using GetOrCreatePoolTask - no custom implementation needed
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;
```

**Benefits of Migration:**
- **No static casting**: Direct use of `Method::kCreate`
- **Standardized structure**: Consistent across all modules
- **Built-in serialization**: SetParams/GetParams methods included
- **Error handling**: Standardized result_code and error_message
- **Less boilerplate**: No need to implement constructors manually

### Data Annotations
- `IN`: Input-only parameters (read by runtime)
- `OUT`: Output-only parameters (written by runtime)
- `INOUT`: Bidirectional parameters

### Task Lifecycle
1. Client allocates task in shared memory using `ipc_manager->NewTask()`
2. Client enqueues task pointer to IPC queue
3. Worker dequeues and executes task
4. Framework calls `ipc_manager->DelTask()` to deallocate task from shared memory
5. Task memory is properly reclaimed from the appropriate memory segment

**Note**: Individual `DelTaskType` methods are no longer required. The framework's autogenerated Del dispatcher automatically calls `ipc_manager->DelTask()` for proper shared memory deallocation.

## C++20 Coroutines for ChiMod Development

Chimaera uses C++20 coroutines to enable cooperative task execution within runtime methods. This section covers the coroutine primitives and patterns used in ChiMod development.

### Overview

Coroutines allow runtime methods to suspend execution (`co_await`) and resume later without blocking the worker thread. This is essential for:
- **Nested Pool Creation**: Create methods that need to create sub-pools
- **Subtask Execution**: Runtime methods that spawn and wait for child tasks
- **I/O Operations**: Async I/O that needs to yield while waiting

### TaskResume Return Type

All runtime methods that use coroutines must return `chi::TaskResume`:

```cpp
class Runtime : public chi::Container {
public:
  // Coroutine method - can use co_await and co_return
  chi::TaskResume Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx);

  // Non-coroutine method - regular void return
  void Custom(hipc::FullPtr<CustomTask> task, chi::RunContext& rctx);
};
```

**Key Points:**
- Methods returning `TaskResume` are coroutines and can use `co_await`/`co_return`
- Methods returning `void` are regular functions and cannot use coroutine keywords
- The `TaskResume` type integrates with Chimaera's task scheduling system

### co_await - Suspending Execution

Use `co_await` to suspend the current coroutine until an operation completes:

```cpp
chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx) {
  // Create a subtask (e.g., create a BDev pool for storage)
  auto bdev_task = bdev_client_.AsyncCreate(
      chi::PoolQuery::Local(),
      "storage_device",
      chi::PoolId(7001, 0),
      chimaera::bdev::BdevType::kFile);

  // Suspend until subtask completes - worker can process other tasks
  co_await bdev_task;

  // Execution resumes here after bdev_task completes
  if (bdev_task->GetReturnCode() != 0) {
    task->SetReturnCode(1);
    task->error_message_ = "Failed to create storage device";
    co_return;
  }

  // Continue with remaining initialization
  task->SetReturnCode(0);
  co_return;
}
```

**What Happens During co_await:**
1. Coroutine state is saved
2. Worker thread is released to process other tasks
3. When awaited operation completes, coroutine is scheduled for resumption
4. Execution continues from the point after `co_await`

### co_return - Completing Coroutines

Use `co_return` to complete a coroutine. For `TaskResume` coroutines, `co_return` takes no value:

```cpp
chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx) {
  // Early return on error
  if (!ValidateParams(task)) {
    task->SetReturnCode(1);
    co_return;  // Complete coroutine immediately
  }

  // Normal completion
  task->SetReturnCode(0);
  co_return;  // Complete coroutine
}
```

**Important Notes:**
- Always use `co_return` (not `return`) in coroutine methods
- `co_return` takes no arguments for `TaskResume` coroutines
- Ensure all code paths end with `co_return`

### Common Coroutine Patterns

#### Pattern 1: Nested Pool Creation

The most common use case is creating dependent pools during container initialization:

```cpp
chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext& rctx) {
  // Initialize container state
  pool_id_ = task->pool_id_;

  // Create dependent BDev pool for storage
  auto bdev_task = bdev_client_.AsyncCreate(
      chi::PoolQuery::Local(),
      task->storage_path_.str(),
      chi::PoolId(pool_id_.IsLocal() + 1, 0),
      chimaera::bdev::BdevType::kFile,
      task->storage_size_);

  co_await bdev_task;

  if (bdev_task->GetReturnCode() != 0) {
    task->SetReturnCode(2);
    task->error_message_ = "Storage initialization failed";
    co_return;
  }

  storage_pool_id_ = bdev_task->new_pool_id_;
  task->SetReturnCode(0);
  co_return;
}
```

#### Pattern 2: Sequential Subtask Execution

Execute multiple subtasks in sequence:

```cpp
chi::TaskResume Runtime::Initialize(hipc::FullPtr<InitTask> task, chi::RunContext& rctx) {
  // Step 1: Initialize storage
  auto storage_task = storage_client_.AsyncInit(chi::PoolQuery::Local());
  co_await storage_task;

  if (storage_task->GetReturnCode() != 0) {
    task->SetReturnCode(1);
    co_return;
  }

  // Step 2: Initialize network (depends on storage)
  auto network_task = network_client_.AsyncInit(chi::PoolQuery::Local());
  co_await network_task;

  if (network_task->GetReturnCode() != 0) {
    task->SetReturnCode(2);
    co_return;
  }

  // Step 3: Complete initialization
  task->SetReturnCode(0);
  co_return;
}
```

#### Pattern 3: Parallel Subtask Execution

For independent subtasks, launch them all first, then await:

```cpp
chi::TaskResume Runtime::ParallelInit(hipc::FullPtr<InitTask> task, chi::RunContext& rctx) {
  // Launch multiple independent tasks
  auto task1 = client1_.AsyncInit(chi::PoolQuery::Local());
  auto task2 = client2_.AsyncInit(chi::PoolQuery::Local());
  auto task3 = client3_.AsyncInit(chi::PoolQuery::Local());

  // Await all tasks (order doesn't matter for independent tasks)
  co_await task1;
  co_await task2;
  co_await task3;

  // Check all results
  if (task1->GetReturnCode() != 0 ||
      task2->GetReturnCode() != 0 ||
      task3->GetReturnCode() != 0) {
    task->SetReturnCode(1);
    co_return;
  }

  task->SetReturnCode(0);
  co_return;
}
```

### When to Use Coroutines

**Use coroutines (TaskResume return type) when:**
- ✅ Creating nested/dependent pools in Create methods
- ✅ Spawning and waiting for subtasks
- ✅ Performing async I/O operations that need to yield
- ✅ Any operation that might block and should yield to other tasks

**Use regular void methods when:**
- ✅ Simple synchronous operations
- ✅ Operations that complete quickly without waiting
- ✅ Methods that don't spawn subtasks or wait for external events

### PoolManager Coroutine Integration

The PoolManager methods that create/destroy pools are coroutines:

```cpp
// In PoolManager (internal usage)
TaskResume CreatePool(hipc::FullPtr<chi::Task> task, chi::RunContext* rctx);
TaskResume DestroyPool(hipc::FullPtr<chi::Task> task, chi::RunContext* rctx);
```

**Why These Are Coroutines:**
- `CreatePool` co_awaits the container's Create method
- Create methods may need to co_await nested pool creations
- The coroutine chain allows proper suspension and resumption

### Admin Runtime Coroutine Methods

The admin runtime has coroutine methods for pool management:

```cpp
// Admin runtime coroutine methods
chi::TaskResume GetOrCreatePool(hipc::FullPtr<GetOrCreatePoolTask<T>> task, chi::RunContext& rctx);
chi::TaskResume DestroyPool(hipc::FullPtr<DestroyPoolTask> task, chi::RunContext& rctx);
```

These methods co_await PoolManager operations internally.

### Best Practices

**DO:**
- ✅ Use `co_return` (not `return`) in coroutine methods
- ✅ Check return codes after `co_await`
- ✅ Use `TaskResume` return type for methods that need `co_await`
- ✅ Handle errors before continuing after `co_await`

**DON'T:**
- ❌ Mix `return` and `co_return` in the same method
- ❌ Use `co_await` in non-coroutine (void) methods
- ❌ Forget to `co_return` at the end of coroutine methods
- ❌ Ignore return codes from awaited tasks

### Migration from Non-Coroutine Methods

To convert a regular method to a coroutine:

1. **Change return type**: `void` → `chi::TaskResume`
2. **Change all `return;`**: → `co_return;`
3. **Add `co_await`**: For any async operations that need waiting
4. **Update autogen**: Ensure dispatch code handles the new return type

### Framework Del Implementation
The autogenerated Del dispatcher handles task cleanup:

```cpp
inline void Del(Runtime* runtime, chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) {
  auto* ipc_manager = CHI_IPC;
  Method method_enum = static_cast<Method>(method);

  switch (method_enum) {
    case Method::kCreate: {
      ipc_manager->DelTask(task_ptr.Cast<CreateTask>());
      break;
    }
    case Method::kCustom: {
      ipc_manager->DelTask(task_ptr.Cast<CustomTask>());
      break;
    }
    default:
      ipc_manager->DelTask(task_ptr);
      break;
  }
}
```

This ensures proper shared memory deallocation without requiring module-specific cleanup code.

## Synchronization Primitives

Chimaera provides specialized cooperative synchronization primitives designed for the runtime's task-based architecture. **These should be used instead of standard synchronization primitives** like `std::mutex`, `std::shared_mutex`, or `pthread_mutex` when synchronizing access to module data structures.

### Why Use Chimaera Synchronization Primitives?

**Critical: Always use CoMutex and CoRwLock for module synchronization:**

1. **Cooperative Design**: Compatible with Chimaera's fiber-based task execution
2. **TaskId Grouping**: Tasks sharing the same TaskId can proceed together (bypassing locks)
3. **Deadlock Prevention**: Designed to prevent deadlocks in the runtime environment
4. **Runtime Integration**: Automatically integrate with CHI_CUR_WORKER and task context
5. **Performance**: Optimized for the runtime's execution model

**Do NOT use these standard synchronization primitives in module code:**
- ❌ `std::mutex` - Can cause fiber blocking issues
- ❌ `std::shared_mutex` - Not compatible with task execution model
- ❌ `pthread_mutex_t` - Can deadlock with runtime scheduling
- ❌ `std::condition_variable` - Incompatible with cooperative scheduling

### CoMutex: Cooperative Mutual Exclusion

CoMutex provides mutual exclusion with TaskId grouping support. Tasks sharing the same TaskId can bypass the lock and execute concurrently.

#### Basic Usage

```cpp
#include <chimaera/comutex.h>

class Runtime : public chi::Container {
private:
  // Static member for shared synchronization across all container instances
  static chi::CoMutex shared_mutex_;
  
  // Instance member for per-container synchronization
  chi::CoMutex instance_mutex_;

public:
  void SomeTask(hipc::FullPtr<SomeTaskType> task, chi::RunContext& rctx) {
    // Manual lock/unlock
    shared_mutex_.Lock();
    // ... critical section ...
    shared_mutex_.Unlock();
    
    // OR use RAII scoped lock (recommended)
    chi::ScopedCoMutex lock(instance_mutex_);
    // ... critical section ...
    // Automatically unlocks when leaving scope
  }
};

// Static member definition (required)
chi::CoMutex Runtime::shared_mutex_;
```

#### Key Features

1. **Automatic Task Context**: Uses CHI_CUR_WORKER internally - no task parameters needed
2. **TaskId Grouping**: Tasks with the same TaskId bypass the mutex
3. **RAII Support**: ScopedCoMutex for automatic lock management
4. **Try-Lock Support**: Non-blocking lock attempts

#### API Reference

```cpp
namespace chi {
  class CoMutex {
  public:
    // Blocking operations
    void Lock();                    // Block until lock acquired
    void Unlock();                  // Release the lock
    bool TryLock();                 // Non-blocking lock attempt
    
    // No task parameters needed - uses CHI_CUR_WORKER automatically
  };
  
  // RAII wrapper (recommended)
  class ScopedCoMutex {
  public:
    explicit ScopedCoMutex(CoMutex& mutex);  // Locks in constructor
    ~ScopedCoMutex();                        // Unlocks in destructor
  };
}
```

### CoRwLock: Cooperative Reader-Writer Lock

CoRwLock provides reader-writer semantics with TaskId grouping. Multiple readers can proceed concurrently, but writers have exclusive access.

#### Basic Usage

```cpp
#include <chimaera/corwlock.h>

class Runtime : public chi::Container {
private:
  static chi::CoRwLock data_lock_;  // Protect shared data structures

public:
  void ReadTask(hipc::FullPtr<ReadTaskType> task, chi::RunContext& rctx) {
    // Manual reader lock
    data_lock_.ReadLock();
    // ... read operations ...
    data_lock_.ReadUnlock();
    
    // OR use RAII scoped reader lock (recommended)
    chi::ScopedCoRwReadLock lock(data_lock_);
    // ... read operations ...
    // Automatically unlocks when leaving scope
  }
  
  void WriteTask(hipc::FullPtr<WriteTaskType> task, chi::RunContext& rctx) {
    // RAII scoped writer lock (recommended)
    chi::ScopedCoRwWriteLock lock(data_lock_);
    // ... write operations ...
    // Automatically unlocks when leaving scope
  }
};

// Static member definition
chi::CoRwLock Runtime::data_lock_;
```

#### Key Features

1. **Multiple Readers**: Concurrent read access when no writers are active
2. **Exclusive Writers**: Writers get exclusive access, blocking all other operations
3. **TaskId Grouping**: Tasks with same TaskId can bypass reader locks
4. **Automatic Context**: Uses CHI_CUR_WORKER for task identification
5. **RAII Support**: Scoped locks for both readers and writers

#### API Reference

```cpp
namespace chi {
  class CoRwLock {
  public:
    // Reader operations
    void ReadLock();                // Acquire reader lock
    void ReadUnlock();              // Release reader lock
    bool TryReadLock();             // Non-blocking reader lock attempt
    
    // Writer operations  
    void WriteLock();               // Acquire exclusive writer lock
    void WriteUnlock();             // Release writer lock
    bool TryWriteLock();            // Non-blocking writer lock attempt
  };
  
  // RAII wrappers (recommended)
  class ScopedCoRwReadLock {
  public:
    explicit ScopedCoRwReadLock(CoRwLock& lock);  // Acquire read lock
    ~ScopedCoRwReadLock();                        // Release read lock
  };
  
  class ScopedCoRwWriteLock {
  public:
    explicit ScopedCoRwWriteLock(CoRwLock& lock); // Acquire write lock
    ~ScopedCoRwWriteLock();                       // Release write lock
  };
}
```

### TaskId Grouping Behavior

Both CoMutex and CoRwLock support TaskId grouping, which allows related tasks to bypass synchronization:

```cpp
// Tasks created with the same TaskId can proceed together
auto task_id = chi::CreateTaskId();

// These tasks share the same TaskId - they can bypass CoMutex/CoRwLock
auto task1 = ipc_manager->NewTask<Task1>(task_id, pool_id, pool_query, ...);
auto task2 = ipc_manager->NewTask<Task2>(task_id, pool_id, pool_query, ...);

// This task has a different TaskId - must respect locks normally
auto task3 = ipc_manager->NewTask<Task3>(chi::CreateTaskId(), pool_id, pool_query, ...);
```

**Key Points:**
- Tasks with the same TaskId are considered "grouped" and can bypass locks
- Use TaskId grouping for logically related operations that don't need mutual exclusion
- Different TaskIds must respect normal lock semantics

### Best Practices

1. **Use RAII Wrappers**: Always prefer `ScopedCoMutex` and `ScopedCoRw*Lock` over manual lock/unlock
2. **Static vs Instance**: Use static members for cross-container synchronization, instance members for per-container data
3. **Member Definition**: Don't forget to define static members in your .cc file
4. **Choose Appropriate Lock**: Use CoRwLock for read-heavy workloads, CoMutex for simple mutual exclusion
5. **Minimal Critical Sections**: Keep locked sections as small as possible
6. **TaskId Design**: Group related tasks that can safely bypass locks

### Example: Module with Synchronized Data Structure

```cpp
// In MOD_NAME_runtime.h
class Runtime : public chi::Container {
private:
  // Synchronized data structure
  chi::hash_map<chi::u32, ModuleData> data_map_;
  
  // Synchronization primitives
  static chi::CoRwLock data_lock_;        // For data_map_ access
  static chi::CoMutex operation_mutex_;   // For exclusive operations

public:
  void ReadData(hipc::FullPtr<ReadDataTask> task, chi::RunContext& rctx);
  void WriteData(hipc::FullPtr<WriteDataTask> task, chi::RunContext& rctx);
  void ExclusiveOperation(hipc::FullPtr<ExclusiveTask> task, chi::RunContext& rctx);
};

// In MOD_NAME_runtime.cc  
chi::CoRwLock Runtime::data_lock_;
chi::CoMutex Runtime::operation_mutex_;

void Runtime::ReadData(hipc::FullPtr<ReadDataTask> task, chi::RunContext& rctx) {
  chi::ScopedCoRwReadLock lock(data_lock_);  // Multiple readers allowed
  
  // Safe to read data_map_ concurrently
  auto it = data_map_.find(task->key_);
  if (it != data_map_.end()) {
    task->result_data_ = it->second;
    task->result_ = 0;  // Success
  } else {
    task->result_ = 1;  // Not found
  }
}

void Runtime::WriteData(hipc::FullPtr<WriteDataTask> task, chi::RunContext& rctx) {
  chi::ScopedCoRwWriteLock lock(data_lock_);  // Exclusive writer access
  
  // Safe to modify data_map_ exclusively
  data_map_[task->key_] = task->new_data_;
  task->result_ = 0;  // Success
}

void Runtime::ExclusiveOperation(hipc::FullPtr<ExclusiveTask> task, chi::RunContext& rctx) {
  chi::ScopedCoMutex lock(operation_mutex_);  // Exclusive operation
  
  // Perform operation that requires complete exclusivity
  // ... complex operation ...
  task->result_ = 0;  // Success
}
```

This synchronization model ensures thread-safe access to module data structures while maintaining compatibility with Chimaera's cooperative task execution system.

## Pool Query and Task Routing

### Overview of PoolQuery

PoolQuery is a fundamental component of Chimaera's task routing system that determines where and how tasks are executed across the distributed runtime. It provides flexible routing strategies for load balancing, locality optimization, and distributed execution patterns.

### PoolQuery Types

The `chi::PoolQuery` class provides six different routing modes through static factory methods:

#### 1. Local Mode
```cpp
chi::PoolQuery::Local()
```
- **Purpose**: Routes tasks to the local node only
- **Use Case**: Operations that must execute on the calling node
- **Example**: MPI-based container creation, node-specific diagnostics
```cpp
// Client usage in MPI environment
const chi::PoolId custom_pool_id(7000, 0);
client.Create(chi::PoolQuery::Local(), "my_pool", custom_pool_id);
```

#### 2. Direct ID Mode
```cpp
chi::PoolQuery::DirectId(ContainerId container_id)
```
- **Purpose**: Routes to a specific container by its unique ID
- **Use Case**: Targeted operations on known containers
- **Example**: Container-specific configuration changes
```cpp
// Route to container with ID 42
auto query = chi::PoolQuery::DirectId(ContainerId(42));
client.UpdateConfig(query, new_config);
```

#### 3. Direct Hash Mode
```cpp
chi::PoolQuery::DirectHash(u32 hash)
```
- **Purpose**: Routes using consistent hash-based load balancing
- **Use Case**: Distributing operations across containers deterministically
- **Example**: Key-value store operations where keys map to specific containers
```cpp
// Hash-based routing for a key
u32 hash = std::hash<std::string>{}(key);
auto query = chi::PoolQuery::DirectHash(hash);
client.Put(query, key, value);
```

#### 4. Range Mode
```cpp
chi::PoolQuery::Range(u32 offset, u32 count)
```
- **Purpose**: Routes to a range of containers
- **Use Case**: Batch operations across multiple containers
- **Example**: Parallel scan operations, bulk updates
```cpp
// Process containers 10-19 (10 containers starting at offset 10)
auto query = chi::PoolQuery::Range(10, 10);
client.BulkUpdate(query, update_data);
```

#### 5. Broadcast Mode
```cpp
chi::PoolQuery::Broadcast()
```
- **Purpose**: Routes to all containers in the pool
- **Use Case**: Global operations affecting all containers
- **Example**: Configuration updates, global cache invalidation
```cpp
// Broadcast configuration change to all containers
auto query = chi::PoolQuery::Broadcast();
client.InvalidateCache(query);
```

#### 6. Physical Mode
```cpp
chi::PoolQuery::Physical(u32 node_id)
```
- **Purpose**: Routes to a specific physical node by ID
- **Use Case**: Node-specific operations in distributed deployments
- **Example**: Remote node administration, cross-node data migration
```cpp
// Execute on physical node 3
auto query = chi::PoolQuery::Physical(3);
client.NodeDiagnostics(query);
```

#### 7. Dynamic Mode (Recommended for Create Operations)
```cpp
chi::PoolQuery::Dynamic()
```
- **Purpose**: Intelligent routing with automatic caching optimization
- **Use Case**: Create operations that benefit from local cache checking
- **Behavior**: Uses dynamic scheduling (ExecMode::kDynamicSchedule) for cache optimization
  1. Check if pool exists locally using PoolManager
  2. If pool exists: change pool_query to Local (execute locally using existing pool)
  3. If pool doesn't exist: change pool_query to Broadcast (create pool on all nodes)
- **Benefits**:
  - Avoids redundant pool creation attempts
  - Eliminates unnecessary network overhead for existing pools
  - Automatic fallback to broadcast creation when needed
- **Example**: Container creation with automatic caching
```cpp
// Recommended: Use Dynamic() for Create operations
const chi::PoolId custom_pool_id(7000, 0);
client.Create(chi::PoolQuery::Dynamic(), "my_pool", custom_pool_id);

// Dynamic scheduling will:
// - Check local cache for "my_pool"
// - If found: switch to Local mode (fast path)
// - If not found: switch to Broadcast mode (creation path)
```

### PoolQuery Usage Guidelines

#### Best Practices

1. **Never use null queries**: Always specify an explicit PoolQuery type
2. **Default to Dynamic for Create**: Use `PoolQuery::Dynamic()` for container creation to enable automatic caching optimization
3. **Alternative: Use Broadcast or Local explicitly**:
   - Use `Broadcast()` when you want to force distributed creation regardless of cache
   - Use `Local()` in MPI jobs when you want node-local containers only
4. **Consider locality**: Prefer local execution to minimize network overhead for regular operations
5. **Use appropriate granularity**: Match routing mode to operation scope

#### Common Patterns

**Container Creation Pattern (Recommended)**:
```cpp
// Recommended: Use Dynamic for automatic cache optimization
// This checks local cache first and falls back to broadcast creation if needed
const chi::PoolId custom_pool_id(7000, 0);
client.Create(chi::PoolQuery::Dynamic(), "my_pool_name", custom_pool_id);
```

**Container Creation Pattern (Explicit Broadcast)**:
```cpp
// Alternative: Use Broadcast to force distributed creation regardless of cache
// This ensures the container is created across all nodes in distributed environments
const chi::PoolId custom_pool_id(7000, 0);
client.Create(chi::PoolQuery::Broadcast(), "my_pool_name", custom_pool_id);
```

**Container Creation Pattern (MPI Environments)**:
```cpp
// In MPI jobs, Local may be more efficient for node-local containers
// Use Local when you want node-local containers only
const chi::PoolId custom_pool_id(7000, 0);
client.Create(chi::PoolQuery::Local(), "my_pool_name", custom_pool_id);
```

**Load-Balanced Operations**:
```cpp
// Use hash-based routing for even distribution
for (const auto& item : items) {
  u32 hash = ComputeHash(item.id);
  auto query = chi::PoolQuery::DirectHash(hash);
  client.Process(query, item);
}
```

**Batch Processing**:
```cpp
// Process containers in chunks
const u32 total_containers = pool_info->num_containers_;
const u32 batch_size = 10;
for (u32 offset = 0; offset < total_containers; offset += batch_size) {
  u32 count = std::min(batch_size, total_containers - offset);
  auto query = chi::PoolQuery::Range(offset, count);
  client.BatchProcess(query, batch_data);
}
```

### Runtime Routing Implementation

The runtime uses PoolQuery to determine task routing through several stages:

1. **Query Validation**: Ensures the query parameters are valid
2. **Container Resolution**: Maps query to specific container(s)
3. **Task Distribution**: Routes task to appropriate worker queues
4. **Load Balancing**: Applies distribution strategies for multi-container queries

### PoolQuery in Task Definitions

Tasks must include PoolQuery in their constructors:

```cpp
class CustomTask : public chi::Task {
 public:
  CustomTask(hipc::Allocator *alloc,
             const chi::TaskId &task_id,
             const chi::PoolId &pool_id,
             const chi::PoolQuery &pool_query,  // Required parameter
             /* custom parameters */)
      : chi::Task(alloc, task_id, pool_id, pool_query, method_id) {
    // Task initialization
  }
};
```

### Advanced PoolQuery Features

#### Query Introspection
```cpp
PoolQuery query = PoolQuery::Range(0, 10);

// Check routing mode
if (query.IsRangeMode()) {
  u32 offset = query.GetRangeOffset();
  u32 count = query.GetRangeCount();
  // Process range parameters
}

// Get routing mode enum
RoutingMode mode = query.GetRoutingMode();
switch (mode) {
  case RoutingMode::Local:
    // Handle local routing
    break;
  case RoutingMode::Broadcast:
    // Handle broadcast
    break;
  // ... other cases
}
```

#### Combining with Task Priorities
```cpp
// High-priority broadcast
auto query = chi::PoolQuery::Broadcast();
auto task = ipc_manager->NewTask<UpdateTask>(
    chi::CreateTaskId(),
    pool_id,
    query,
    update_data
);
return ipc_manager->Send(task);
```

### Troubleshooting PoolQuery Issues

**Common Errors**:

1. **Null Query Error**: "NEVER use a null pool query"
   - Solution: Always use a factory method like `PoolQuery::Local()`

2. **Invalid Container ID**: Container not found for DirectId query
   - Solution: Verify container exists before using DirectId

3. **Range Out of Bounds**: Range exceeds available containers
   - Solution: Check pool size before creating Range queries

4. **Node ID Invalid**: Physical node ID doesn't exist
   - Solution: Validate node IDs against cluster configuration

## Client-Server Communication

### Client Implementation Patterns

Chimaera uses an **async-only client API pattern**. All client operations are asynchronous, returning `chi::Future<TaskType>` objects. This design:
- Enables parallel task submission for better performance
- Provides consistent API across all operations
- Allows fine-grained control over task completion timing
- Simplifies the codebase by eliminating duplicate sync/async methods

#### Async Create Pattern

**IMPORTANT**: All ChiMod clients MUST update their `pool_id_` field with the actual pool ID returned from completed CreateTask operations. This is essential because:

1. CreateTask operations may return a different pool ID than initially specified
2. Pool creation may reuse existing pools with different IDs
3. Subsequent client operations depend on the correct pool ID

**Required Pattern for All Client AsyncCreate Methods:**

```cpp
chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                    const std::string& pool_name,
                                    const chi::PoolId& custom_pool_id,
                                    /* other module-specific parameters */) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name,
        pool_name,
        custom_pool_id,
        /* module-specific parameters */
        this  // Client pointer for PostWait callback
    );
    return ipc_manager->Send(task);
}
```

**Required Parameters for All AsyncCreate Methods:**

1. **pool_query**: Task routing strategy (use `Dynamic()` recommended, `Broadcast()` for non-MPI, `Local()` for MPI)
2. **pool_name**: User-provided name for the pool (must be unique, used as file path for file-based modules)
3. **custom_pool_id**: Explicit pool ID for the container being created (must not be null)
4. **Module-specific parameters**: Additional parameters specific to the ChiMod (e.g., BDev type, size)

**Why Pool ID Update Is Required:**

- **Pool Reuse**: CreateTask is actually a GetOrCreatePoolTask that may return an existing pool
- **ID Assignment**: The admin ChiMod may assign a different pool ID than requested
- **Client Consistency**: All subsequent operations must use the correct pool ID
- **Distributed Operation**: Pool IDs must be consistent across all client instances

**Usage Pattern (Caller Side):**

```cpp
// Create a client and async create the pool
chimaera::bdev::Client bdev_client;
const chi::PoolId pool_id(7000, 0);

auto task = bdev_client.AsyncCreate(
    chi::PoolQuery::Dynamic(),
    "/path/to/storage.dat",
    pool_id,
    chimaera::bdev::BdevType::kFile,
    1024 * 1024 * 1024);  // 1GB

// Wait for completion
task.Wait();

// Check result
if (task->GetReturnCode() != 0) {
    std::cerr << "Create failed: " << task->error_message_.str() << std::endl;
    return;
}

// The client's pool_id_ is updated via PostWait callback
// Now can use the client for operations
auto read_task = bdev_client.AsyncRead(chi::PoolQuery::Local(), block, buffer, size);
read_task.Wait();
```

**Examples of Correct AsyncCreate Implementation:**

```cpp
// Admin client AsyncCreate method
chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                    const std::string& pool_name,
                                    const chi::PoolId& custom_pool_id) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,
        pool_query,
        CreateParams::chimod_lib_name,
        pool_name,
        custom_pool_id,
        this);
    return ipc_manager->Send(task);
}

// BDev client AsyncCreate method (with module-specific parameters)
chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                    const std::string& pool_name,
                                    const chi::PoolId& custom_pool_id,
                                    BdevType bdev_type,
                                    chi::u64 total_size = 0,
                                    chi::u32 io_depth = 128,
                                    chi::u32 alignment = 4096) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,
        pool_query,
        CreateParams::chimod_lib_name,
        pool_name,
        custom_pool_id,
        bdev_type, total_size, io_depth, alignment,
        this);
    return ipc_manager->Send(task);
}
```

**Common Mistakes to Avoid:**

- ❌ **Using null PoolId for custom_pool_id**: Create operations REQUIRE explicit, non-null pool IDs
- ❌ **Forgetting PostWait callback**: Ensure client pointer is passed to NewTask for pool_id_ update
- ❌ **Using original pool_id_**: The task may return a different pool ID than initially specified
- ❌ **Accessing results before Wait()**: Always call `task.Wait()` before reading task fields
- ❌ **Implementing synchronous wrappers**: Use async-only pattern, let callers handle waiting
- ❌ **Using Local instead of Dynamic/Broadcast**: Use `Dynamic()` (recommended) or `Broadcast()` for distributed container creation

**Critical Validation:**

The runtime validates that `custom_pool_id` is not null during Create operations. If a null PoolId is provided, the Create operation will fail with an error:

```cpp
// WRONG - This will fail with error
chi::PoolId null_id;  // Null pool ID
client.Create(chi::PoolQuery::Broadcast(), "my_pool", null_id);
// Error: "Cannot create pool with null PoolId. Explicit pool IDs are required."

// CORRECT - Always provide explicit pool IDs
const chi::PoolId custom_pool_id(7000, 0);
client.Create(chi::PoolQuery::Broadcast(), "my_pool", custom_pool_id);
```

This pattern is mandatory for all ChiMod clients and ensures correct pool ID management throughout the client lifecycle.

### Memory Segments
Three shared memory segments are used:
1. **Main Segment**: Tasks and control structures
2. **Client Data Segment**: User data buffers
3. **Runtime Data Segment**: Runtime-only data

### IPC Queue
Tasks are submitted via the IPC manager:
```cpp
// Client side - create and submit task, returns Future
auto task = ipc_manager->NewTask<CustomTask>(...);
auto future = ipc_manager->Send(task);

// Wait for completion
future.Wait();

// Access results
auto result = future->output_field_;
```

## Memory Management

### Allocator Usage
```cpp
// Get context allocator for current segment
hipc::CtxAllocator<CHI_MAIN_ALLOC_T> ctx_alloc(allocator);

// Allocate serializable string
chi::string my_string(ctx_alloc, "initial value");

// Allocate vector
chi::vector<u32> my_vector(ctx_alloc);
my_vector.resize(100);
```

### Best Practices
1. Always use HSHM types for shared data
2. Pass CtxAllocator to constructors
3. Use FullPtr for cross-process references
4. Let framework handle task cleanup via `ipc_manager->DelTask()`

### Task Allocation and Deallocation Pattern
```cpp
// Client side - allocation (NewTask uses main allocator automatically)
auto task = ipc_manager->NewTask<CustomTask>(
    chi::CreateTaskId(),
    pool_id_,
    pool_query,
    input_data,
    operation_id);

// Client side - cleanup (after task completion)
ipc_manager->DelTask(task);

// Runtime side - automatic cleanup (no code needed)
// Framework Del dispatcher calls ipc_manager->DelTask() automatically
```

### CHI_IPC Buffer Allocation

The `CHI_IPC` singleton provides centralized buffer allocation for shared memory operations in client code. Use this for allocating temporary buffers that need to be shared between client and runtime processes.

**Important**: `AllocateBuffer` returns `hipc::FullPtr<char>`, not `hipc::ShmPtr<>`. It is NOT a template function.

#### Basic Usage
```cpp
#include <chimaera/chimaera.h>

// Get the IPC manager singleton
auto* ipc_manager = CHI_IPC;

// Allocate a buffer in shared memory (returns FullPtr<char>)
size_t buffer_size = 1024;
hipc::FullPtr<char> buffer_ptr = ipc_manager->AllocateBuffer(buffer_size);

// Use the buffer (example: copy data into it)
char* buffer_data = buffer_ptr.ptr_;
memcpy(buffer_data, source_data, data_size);

// Alternative: Use directly
strncpy(buffer_ptr.ptr_, "example data", buffer_size);

// The buffer will be automatically freed when buffer_ptr goes out of scope
// or when explicitly deallocated by the framework
```

#### Use Cases for CHI_IPC Buffers
- **Temporary data transfer**: When passing large data to tasks
- **Intermediate storage**: For computations that need shared memory
- **I/O operations**: Reading/writing data that needs to be accessible by runtime

#### Best Practices
```cpp
// ✅ Good: Use CHI_IPC for temporary shared buffers
auto* ipc_manager = CHI_IPC;
hipc::FullPtr<char> temp_buffer = ipc_manager->AllocateBuffer(data_size);

// ✅ Good: Use chi::ipc types for persistent task data
chi::ipc::string task_string(ctx_alloc, "persistent data");

// ❌ Avoid: Don't use CHI_IPC for small, simple task parameters
// Use chi::ipc types directly in task definitions instead
```

### Shared-Memory Compatible Data Structures

For task definitions and any data that needs to be shared between client and runtime processes, always use shared-memory compatible types instead of standard C++ containers.

#### chi::ipc::string
Use `chi::ipc::string` or `chi::priv::string` instead of `std::string` in task definitions:

```cpp
#include <[namespace]/types.h>

// Task definition using shared-memory string
struct CustomTask : public chi::Task {
  INOUT chi::priv::string input_data_;     // Shared-memory compatible string
  INOUT chi::priv::string output_data_;    // Results stored in shared memory
  
  CustomTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T>& alloc,
             const std::string& input) 
    : input_data_(alloc, input),      // Initialize from std::string
      output_data_(alloc) {}          // Empty initialization
      
  // Conversion to std::string when needed
  std::string getResult() const {
    return std::string(output_data_.data(), output_data_.size());
  }
};
```

#### chi::ipc::vector
Use `chi::ipc::vector` instead of `std::vector` for arrays in task definitions:

```cpp
// Task definition using shared-memory vector
struct ProcessArrayTask : public chi::Task {
  INOUT chi::ipc::vector<chi::u32> data_array_;
  INOUT chi::ipc::vector<chi::f32> result_array_;
  
  ProcessArrayTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T>& alloc,
                   const std::vector<chi::u32>& input_data)
    : data_array_(alloc),
      result_array_(alloc) {
    // Copy from std::vector to shared-memory vector
    data_array_.resize(input_data.size());
    std::copy(input_data.begin(), input_data.end(), data_array_.begin());
  }
};
```

#### When to Use Each Type

**Use shared-memory types (chi::ipc::string, chi::priv::string, chi::ipc::vector, etc.) for:**
- Task input/output parameters
- Data that persists across task execution
- Any data structure that needs serialization
- Data shared between client and runtime

**Use std::string/vector for:**
- Local variables in client code
- Temporary computations
- Converting to/from shared-memory types

#### Type Conversion Examples
```cpp
// Converting between std::string and shared-memory string types
std::string std_str = "example data";
chi::priv::string shm_str(ctx_alloc, std_str);          // std -> shared memory
std::string result = std::string(shm_str);         // shared memory -> std

// Converting between std::vector and shared-memory vector types
std::vector<int> std_vec = {1, 2, 3, 4, 5};
chi::ipc::vector<int> shm_vec(ctx_alloc);
shm_vec.assign(std_vec.begin(), std_vec.end());    // std -> shared memory

std::vector<int> result_vec(shm_vec.begin(), shm_vec.end());  // shared memory -> std
```

#### Serialization Support
Both `chi::ipc::string` and `chi::ipc::vector` automatically support serialization for task communication:

```cpp
// Task definition - no manual serialization needed
struct SerializableTask : public chi::Task {
  INOUT chi::priv::string message_;
  INOUT chi::ipc::vector<chi::u64> timestamps_;

  // Cereal automatically handles chi::ipc types
  template<class Archive>
  void serialize(Archive& ar) {
    ar(message_, timestamps_);  // Works automatically
  }
};
```

### Bulk Transfer Support with ar.bulk

For tasks that involve large data transfers (such as I/O operations), Chimaera provides `ar.bulk()` for efficient bulk data serialization. This feature integrates with the Lightbeam networking layer to enable zero-copy data transfer and RDMA optimization.

#### Overview

The `ar.bulk()` method marks data pointers for bulk transfer during task serialization. This is essential for:
- **Large I/O Operations**: Read/write tasks with multi-megabyte payloads
- **Zero-Copy Transfer**: Avoiding unnecessary data copies during serialization
- **RDMA Optimization**: Preparing data for remote direct memory access
- **Distributed Execution**: Sending tasks with large data buffers across nodes

#### Bulk Transfer Flags

Two flags control bulk transfer behavior:

```cpp
// Defined in hermes_shm/lightbeam/lightbeam.h
#define BULK_EXPOSE  // Metadata only - no data transfer (receiver allocates)
#define BULK_XFER    // Marks bulk for actual data transmission
```

**Flag Usage:**
- **BULK_EXPOSE**: Sender exposes metadata (size, pointer info) but doesn't transfer data
  - Receiver sees the bulk size and allocates local buffer
  - Useful when receiver will write data (e.g., Read operations)
- **BULK_XFER**: Marks bulk for actual data transmission
  - Data is transferred over network
  - Used when sender has data to send (e.g., Write operations)

#### Basic Usage Pattern

##### Write Operation (Sender Has Data)

For write operations, the sender has data to transfer:

```cpp
struct WriteTask : public chi::Task {
  IN Block block_;              // Block to write to
  IN hipc::ShmPtr<> data_;       // Data buffer pointer
  IN size_t length_;            // Data length
  OUT chi::u64 bytes_written_;  // Result

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    ar(block_, length_);
    // Use BULK_XFER to transfer data from sender to receiver
    ar.bulk(data_, length_, BULK_XFER);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    ar(bytes_written_);
    // No bulk transfer needed for output (just metadata)
  }
};
```

**Workflow:**
1. **Client Side (SerializeIn)**:
   - Serializes `block_` and `length_` metadata
   - Marks `data_` buffer with `BULK_XFER` flag
   - Lightbeam transmits the data buffer to receiver
2. **Runtime Side (Execute)**:
   - Receives metadata and data buffer
   - Executes write operation using transferred data
   - Sets `bytes_written_` result
3. **Client Side (SerializeOut)**:
   - Receives `bytes_written_` result
   - No bulk transfer needed for small output values

##### Read Operation (Receiver Needs Data)

For read operations, the receiver allocates buffer for incoming data:

```cpp
struct ReadTask : public chi::Task {
  IN Block block_;              // Block to read from
  OUT hipc::ShmPtr<> data_;      // Data buffer pointer (allocated by receiver)
  INOUT size_t length_;         // Requested/actual length
  OUT chi::u64 bytes_read_;     // Result

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    ar(block_, length_);
    // Use BULK_EXPOSE - metadata only, receiver will allocate buffer
    ar.bulk(data_, length_, BULK_EXPOSE);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    ar(length_, bytes_read_);
    // Use BULK_XFER to transfer read data back to client
    ar.bulk(data_, length_, BULK_XFER);
  }
};
```

**Workflow:**
1. **Client Side (SerializeIn)**:
   - Serializes `block_` and `length_` metadata
   - Marks `data_` with `BULK_EXPOSE` (no data sent yet)
   - Receiver sees buffer size needed
2. **Runtime Side (Execute)**:
   - Receives metadata including buffer size
   - Allocates local buffer for `data_`
   - Executes read operation filling the buffer
   - Sets `length_` and `bytes_read_` results
3. **Client Side (SerializeOut)**:
   - Marks `data_` with `BULK_XFER` flag
   - Lightbeam transfers read data back to client
   - Client receives `length_`, `bytes_read_`, and data buffer

#### API Reference

```cpp
template <typename Archive>
void ar.bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags);
```

**Parameters:**
- `ptr`: Pointer to data buffer (`hipc::ShmPtr<>`, `hipc::FullPtr`, or raw pointer)
- `size`: Size of data in bytes
- `flags`: Transfer flags (`BULK_EXPOSE` or `BULK_XFER`)

**Behavior:**
- Records bulk transfer metadata in the archive
- For `BULK_XFER`: Prepares data for network transmission
- For `BULK_EXPOSE`: Records metadata only (size and pointer info)
- Integrates with Lightbeam networking for actual data transfer

#### Advanced Pattern: Bidirectional Transfer

Some operations require data transfer in both directions:

```cpp
struct ProcessTask : public chi::Task {
  INOUT hipc::ShmPtr<> data_;    // Data buffer (modified in-place)
  INOUT size_t length_;         // Buffer length

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    ar(length_);
    // Send input data to runtime
    ar.bulk(data_, length_, BULK_XFER);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    ar(length_);
    // Send modified data back to client
    ar.bulk(data_, length_, BULK_XFER);
  }
};
```

#### Integration with Lightbeam

The `ar.bulk()` calls integrate seamlessly with the Lightbeam networking layer:

1. **Archive Records Bulks**:
   - TaskOutputArchive stores bulk metadata in `bulk_transfers_` vector
   - Each bulk includes pointer, size, and flags

2. **Lightbeam Transmission**:
   - Bulks marked `BULK_XFER` are transmitted via `Send()` and `RecvBulks()`
   - Bulks marked `BULK_EXPOSE` provide metadata only
   - Receiver inspects all bulks to determine buffer sizes

3. **Zero-Copy Optimization**:
   - Data stays in original buffers during serialization
   - Only pointers and metadata are serialized
   - Actual data transfer handled separately by Lightbeam

#### Complete Example: BDev Read Task

```cpp
struct ReadTask : public chi::Task {
  IN Block block_;              // Block descriptor
  OUT hipc::ShmPtr<> data_;      // Data buffer
  INOUT size_t length_;         // Buffer length
  OUT chi::u64 bytes_read_;     // Bytes actually read

  /** SHM constructor */
  explicit ReadTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : chi::Task(alloc), length_(0), bytes_read_(0) {}

  /** Emplace constructor */
  explicit ReadTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                    const chi::TaskId &task_node,
                    const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query,
                    const Block &block,
                    hipc::ShmPtr<> data,
                    size_t length)
      : chi::Task(alloc, task_node, pool_id, pool_query, 10),
        block_(block), data_(data), length_(length), bytes_read_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRead;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    ar(block_, length_);
    // BULK_EXPOSE: Tell receiver the buffer size, but don't send data yet
    // Receiver will allocate local buffer
    ar.bulk(data_, length_, BULK_EXPOSE);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    ar(length_, bytes_read_);
    // BULK_XFER: Transfer the read data back to client
    ar.bulk(data_, length_, BULK_XFER);
  }

  /** Copy from another ReadTask */
  void Copy(const hipc::FullPtr<ReadTask> &other) {
    // Copy task-specific fields only
    // Base Task fields are copied automatically by NewCopy
    block_ = other->block_;
    data_ = other->data_;
    length_ = other->length_;
    bytes_read_ = other->bytes_read_;
  }

  /** Aggregate results from replica */
  void Aggregate(const hipc::FullPtr<ReadTask> &other) {
    // For reads, just copy the result from the replica
    Copy(other);
  }
};
```

#### Best Practices

**DO:**
- ✅ Use `BULK_XFER` when sender has data to transmit (Write operations)
- ✅ Use `BULK_EXPOSE` when receiver needs to allocate buffer (Read operations)
- ✅ Always specify both `SerializeIn()` and `SerializeOut()` for consistency
- ✅ Use `ar.bulk()` for data buffers larger than a few KB
- ✅ Ensure data buffer lifetime extends until serialization completes

**DON'T:**
- ❌ Don't use `ar.bulk()` for small data (< 4KB) - serialize directly instead
- ❌ Don't forget to specify bulk size - it determines receiver buffer allocation
- ❌ Don't mix `ar()` and `ar.bulk()` for the same data - choose one approach
- ❌ Don't use `BULK_EXPOSE` for write operations (sender has data to send)
- ❌ Don't use `BULK_XFER` in SerializeIn for read operations (no data to send yet)

#### Performance Considerations

1. **Buffer Alignment**: Ensure buffers are properly aligned (typically 4KB for I/O operations)
2. **Size Thresholds**: Use bulk transfer for data > 4KB; use regular serialization for smaller data
3. **Zero-Copy**: Lightbeam can use zero-copy techniques when data is in shared memory
4. **RDMA Ready**: The bulk transfer API is designed for future RDMA transport integration

#### Troubleshooting

**Common Issues:**

1. **Missing Data Transfer**:
   - Ensure `BULK_XFER` flag is used when data should be transmitted
   - Check that SerializeOut uses `BULK_XFER` for read operations

2. **Buffer Size Mismatch**:
   - Verify `length_` parameter matches actual buffer size
   - Ensure receiver allocates buffer matching the exposed size

3. **Serialization Order**:
   - Serialize metadata (block, length) before `ar.bulk()` call
   - This ensures receiver knows buffer size before allocating

### chi::unordered_map_ll - Lock-Free Unordered Map

The `chi::unordered_map_ll` is a hash map implementation using a vector of lists design that provides efficient concurrent access when combined with external locking. This container is specifically designed for runtime module data structures that require external synchronization control.

#### Overview

**Key Characteristics:**
- **Vector of Lists Design**: Uses a vector of buckets, each containing a list of key-value pairs
- **External Locking Required**: No internal mutexes - users must provide synchronization
- **Bucket Partitioning**: Hash space is partitioned across multiple buckets for better cache locality
- **Standard API**: Compatible with `std::unordered_map` interface
- **NOT Shared-Memory Compatible**: For runtime-only data structures, not task parameters

#### Basic Usage

```cpp
#include <chimaera/unordered_map_ll.h>

class Runtime : public chi::Container {
private:
  // Runtime data structure with external locking
  chi::unordered_map_ll<chi::u32, ModuleData> data_map_;

  // External synchronization using CoRwLock
  static chi::CoRwLock data_lock_;

public:
  Runtime() : data_map_(32) {}  // 32 buckets for hash partitioning

  void ReadData(hipc::FullPtr<ReadTask> task, chi::RunContext& ctx) {
    chi::ScopedCoRwReadLock lock(data_lock_);

    // Safe to access data_map_ with external lock held
    auto* value = data_map_.find(task->key_);
    if (value) {
      task->result_ = *value;
    }
  }

  void WriteData(hipc::FullPtr<WriteTask> task, chi::RunContext& ctx) {
    chi::ScopedCoRwWriteLock lock(data_lock_);

    // Safe to modify data_map_ with exclusive lock
    data_map_.insert_or_assign(task->key_, task->data_);
  }
};

// Static member definition
chi::CoRwLock Runtime::data_lock_;
```

#### Constructor

```cpp
// Create map with specified bucket count (determines max useful concurrency)
chi::unordered_map_ll<Key, T> map(max_concurrency);

// Example: 32 buckets provides good distribution for most workloads
chi::unordered_map_ll<int, std::string> map(32);
```

**Parameters:**
- `max_concurrency`: Number of buckets (default: 16)
  - Higher values = better distribution, more memory overhead
  - Typical values: 16-64 for most use cases
  - Should be power of 2 for optimal hash distribution

#### API Reference

The container provides a `std::unordered_map`-compatible interface:

```cpp
// Insertion operations
auto [inserted, value_ptr] = map.insert(key, value);          // Insert if not exists
auto [inserted, value_ptr] = map.insert_or_assign(key, value); // Insert or update
T& ref = map[key];                                            // Insert default if missing

// Lookup operations
T* ptr = map.find(key);                    // Returns nullptr if not found
const T* ptr = map.find(key) const;        // Const version
T& ref = map.at(key);                      // Throws if not found
bool exists = map.contains(key);           // Check existence
size_t count = map.count(key);             // Returns 0 or 1

// Removal operations
size_t erased = map.erase(key);            // Returns number of elements erased
void map.clear();                          // Remove all elements

// Size operations
size_t size = map.size();                  // Total element count
bool empty = map.empty();                  // Check if empty
size_t buckets = map.bucket_count();       // Number of buckets

// Iteration
map.for_each([](const Key& key, T& value) {
  // Process each element
  // Note: External lock must be held during iteration
});
```

#### Return Value Semantics

Insert operations return `std::pair<bool, T*>`:
- `first`: `true` if insertion occurred, `false` if key already exists
- `second`: Pointer to the value (existing or newly inserted)

```cpp
auto [inserted, value_ptr] = map.insert(42, "hello");
if (inserted) {
  // New element was inserted
  std::cout << "Inserted: " << *value_ptr << std::endl;
} else {
  // Key already existed
  std::cout << "Existing: " << *value_ptr << std::endl;
}
```

#### External Locking Patterns

**Pattern 1: CoRwLock for Read-Heavy Workloads**
```cpp
class Runtime : public chi::Container {
private:
  chi::unordered_map_ll<chi::u64, CachedData> cache_;
  static chi::CoRwLock cache_lock_;

public:
  void LookupCache(hipc::FullPtr<LookupTask> task, chi::RunContext& ctx) {
    chi::ScopedCoRwReadLock lock(cache_lock_);  // Multiple readers allowed

    auto* data = cache_.find(task->cache_key_);
    if (data) {
      task->result_ = *data;
      task->found_ = true;
    } else {
      task->found_ = false;
    }
  }

  void UpdateCache(hipc::FullPtr<UpdateTask> task, chi::RunContext& ctx) {
    chi::ScopedCoRwWriteLock lock(cache_lock_);  // Exclusive writer

    cache_.insert_or_assign(task->cache_key_, task->new_data_);
  }
};

chi::CoRwLock Runtime::cache_lock_;
```

**Pattern 2: CoMutex for Write-Heavy Workloads**
```cpp
class Runtime : public chi::Container {
private:
  chi::unordered_map_ll<std::string, RequestCounter> counters_;
  static chi::CoMutex counters_mutex_;

public:
  void IncrementCounter(hipc::FullPtr<IncrementTask> task, chi::RunContext& ctx) {
    chi::ScopedCoMutex lock(counters_mutex_);

    auto [inserted, counter_ptr] = counters_.insert(task->counter_name_, RequestCounter{});
    counter_ptr->count++;
    task->new_count_ = counter_ptr->count;
  }
};

chi::CoMutex Runtime::counters_mutex_;
```

**Pattern 3: Instance-Level Locking**
```cpp
class Runtime : public chi::Container {
private:
  // Per-container instance data
  chi::unordered_map_ll<chi::u32, TaskState> active_tasks_;
  chi::CoMutex instance_lock_;  // Instance member, not static

public:
  void RegisterTask(hipc::FullPtr<RegisterTask> task, chi::RunContext& ctx) {
    chi::ScopedCoMutex lock(instance_lock_);  // Lock this container instance only

    active_tasks_.insert(task->task_id_, TaskState{task->start_time_});
  }
};
```

#### When to Use chi::unordered_map_ll

**✅ Use chi::unordered_map_ll for:**
- Runtime container data structures (caches, registries, counters)
- Module-internal state management
- Lookup tables for fast key-value access
- Data structures protected by CoMutex/CoRwLock
- Non-shared memory data (runtime process only)

**❌ Do NOT use chi::unordered_map_ll for:**
- Task input/output parameters (use `chi::ipc::` types instead)
- Shared-memory data structures (not compatible with HSHM allocators)
- Client-side code (use `std::unordered_map` instead)
- Data that needs to be serialized (use `std::unordered_map` with cereal)

#### Performance Considerations

**Bucket Count Selection:**
```cpp
// Small datasets (< 100 elements): 16 buckets
chi::unordered_map_ll<Key, Value> small_map(16);

// Medium datasets (100-10000 elements): 32-64 buckets
chi::unordered_map_ll<Key, Value> medium_map(32);

// Large datasets (> 10000 elements): 64-128 buckets
chi::unordered_map_ll<Key, Value> large_map(64);

// Very large datasets or high concurrency: 128+ buckets
chi::unordered_map_ll<Key, Value> huge_map(128);
```

**Iteration Performance:**
```cpp
// Iteration requires external lock for entire duration
void ProcessAllEntries(hipc::FullPtr<Task> task, chi::RunContext& ctx) {
  chi::ScopedCoRwReadLock lock(data_lock_);  // Hold lock during entire iteration

  size_t count = 0;
  data_map_.for_each([&count](const Key& key, Value& value) {
    // Process entry
    count++;
  });

  task->processed_count_ = count;
  // Lock released when scope exits
}
```

#### Complete Example: Request Tracking Module

```cpp
// In MOD_NAME_runtime.h
#include <chimaera/unordered_map_ll.h>
#include <chimaera/corwlock.h>

class Runtime : public chi::Container {
private:
  // Request tracking data structure
  struct RequestInfo {
    chi::u64 start_time_us_;
    chi::u64 bytes_processed_;
    chi::u32 status_code_;
  };

  // Map of active requests (external locking required)
  chi::unordered_map_ll<chi::u64, RequestInfo> active_requests_;

  // Completed request statistics
  chi::unordered_map_ll<chi::u32, chi::u64> status_counts_;

  // Synchronization primitives
  static chi::CoRwLock requests_lock_;
  static chi::CoMutex stats_mutex_;

public:
  Runtime()
    : active_requests_(64),   // 64 buckets for active requests
      status_counts_(16) {}   // 16 buckets for status codes

  void StartRequest(hipc::FullPtr<StartRequestTask> task, chi::RunContext& ctx) {
    chi::ScopedCoRwWriteLock lock(requests_lock_);

    RequestInfo info{
      .start_time_us_ = task->timestamp_,
      .bytes_processed_ = 0,
      .status_code_ = 0
    };

    active_requests_.insert(task->request_id_, info);
  }

  void CompleteRequest(hipc::FullPtr<CompleteRequestTask> task, chi::RunContext& ctx) {
    {
      // Update active requests
      chi::ScopedCoRwWriteLock lock(requests_lock_);

      auto* info = active_requests_.find(task->request_id_);
      if (info) {
        task->duration_us_ = task->end_time_ - info->start_time_us_;
        task->bytes_processed_ = info->bytes_processed_;

        // Update statistics
        {
          chi::ScopedCoMutex stats_lock(stats_mutex_);
          auto [inserted, count_ptr] = status_counts_.insert_or_assign(
            info->status_code_, 0);
          (*count_ptr)++;
        }

        active_requests_.erase(task->request_id_);
      }
    }
  }

  void GetStatistics(hipc::FullPtr<GetStatsTask> task, chi::RunContext& ctx) {
    // Read statistics with read lock
    chi::ScopedCoRwReadLock lock(requests_lock_);

    task->active_count_ = active_requests_.size();

    // Get status code distribution
    chi::ScopedCoMutex stats_lock(stats_mutex_);
    status_counts_.for_each([&task](const chi::u32& status, const chi::u64& count) {
      task->status_distribution_.push_back({status, count});
    });
  }
};

// Static member definitions
chi::CoRwLock Runtime::requests_lock_;
chi::CoMutex Runtime::stats_mutex_;
```

#### Key Differences from std::unordered_map

| Feature | std::unordered_map | chi::unordered_map_ll |
|---------|-------------------|----------------------|
| Thread Safety | None (external locking required) | None (external locking required) |
| Internal Structure | Implementation-defined | Vector of lists (explicit) |
| Bucket Count | Dynamic rehashing | Fixed at construction |
| Iterator Stability | Unstable across insertions | Stable (list-based) |
| Shared Memory | Not compatible | Not compatible |
| Return Values | Iterators | Pointers to values |
| Use Case | General purpose | Runtime data structures |

#### Summary

`chi::unordered_map_ll` provides a specialized hash map implementation optimized for Chimaera runtime modules:

1. **External Locking**: Must be protected by CoMutex or CoRwLock
2. **Fixed Buckets**: Bucket count set at construction (no rehashing)
3. **Pointer Interface**: Operations return pointers instead of iterators
4. **Runtime Only**: Not for shared-memory or task parameters
5. **Efficient Lookup**: O(1) average case for find/insert/erase operations

For runtime container data structures requiring fast key-value access with external synchronization, `chi::unordered_map_ll` provides an efficient and predictable solution.

## Build System Integration

### CMakeLists.txt Template
ChiMod CMakeLists.txt files should use the standardized ChimaeraCommon.cmake functions for consistency and proper configuration:

```cmake
cmake_minimum_required(VERSION 3.10)

# Create both client and runtime libraries for your module
# This creates targets: ${NAMESPACE}_${CHIMOD_NAME}_runtime and ${NAMESPACE}_${CHIMOD_NAME}_client
# CMake aliases: ${NAMESPACE}::${CHIMOD_NAME}_runtime and ${NAMESPACE}::${CHIMOD_NAME}_client
add_chimod_client(
  CHIMOD_NAME YOUR_MODULE_NAME
  SOURCES src/YOUR_MODULE_NAME_client.cc
)
add_chimod_runtime(
  CHIMOD_NAME YOUR_MODULE_NAME
  SOURCES src/YOUR_MODULE_NAME_runtime.cc src/autogen/YOUR_MODULE_NAME_lib_exec.cc
)

# Installation is automatic - no separate install_chimod() call required
```

### CMakeLists.txt Guidelines

**DO:**
- Use `add_chimod_client()` and `add_chimod_runtime()` utility functions (installation is automatic)
- Set `CHIMOD_NAME` to your module's name
- List source files explicitly in `SOURCES` parameters
- Include autogen source files in runtime `SOURCES`
- Keep the CMakeLists.txt minimal and consistent

**DON'T:**
- Use manual `add_library()` calls - use the utilities instead
- Call `install_chimod()` separately - it's handled automatically
- Include relative paths like `../include/*` - use proper include directories
- Set custom compile definitions - the utilities handle this
- Manually configure target properties - the utilities provide standard settings

### ChiMod Build Functions Reference

#### `add_chimod_client()` Function
Creates a ChiMod client library target with automatic dependency management.

```cmake
add_chimod_client(
  SOURCES source_file1.cc source_file2.cc ...
  [COMPILE_DEFINITIONS definition1 definition2 ...]
  [LINK_LIBRARIES library1 library2 ...]
  [LINK_DIRECTORIES directory1 directory2 ...]
  [INCLUDE_LIBRARIES target1 target2 ...]
  [INCLUDE_DIRECTORIES directory1 directory2 ...]
)
```

**Parameters:**
- **`SOURCES`** (required): List of source files for the client library
- **`COMPILE_DEFINITIONS`** (optional): Additional preprocessor definitions beyond automatic ones
- **`LINK_LIBRARIES`** (optional): Additional libraries to link beyond automatic dependencies
- **`LINK_DIRECTORIES`** (optional): Additional library search directories
- **`INCLUDE_LIBRARIES`** (optional): Target libraries whose include directories should be inherited
- **`INCLUDE_DIRECTORIES`** (optional): Additional include directories beyond automatic ones

**Automatic Behavior:**
- Creates target: `${NAMESPACE}_${CHIMOD_NAME}_client`
- Creates alias: `${NAMESPACE}::${CHIMOD_NAME}_client`
- Automatically links core Chimaera library (`chimaera::cxx` or `hermes_shm::cxx`)
- For non-admin ChiMods: automatically links `chimaera_admin_client`
- Automatically includes module headers from `include/` directory
- Installs library and headers with proper CMake export configuration

**Example:**
```cmake
add_chimod_client(
  SOURCES src/my_module_client.cc
  COMPILE_DEFINITIONS MY_MODULE_DEBUG=1
  LINK_LIBRARIES additional_lib
  INCLUDE_DIRECTORIES ${EXTERNAL_INCLUDE_DIR}
)
```

#### `add_chimod_runtime()` Function
Creates a ChiMod runtime library target with automatic dependency management.

```cmake
add_chimod_runtime(
  SOURCES source_file1.cc source_file2.cc ...
  [COMPILE_DEFINITIONS definition1 definition2 ...]
  [LINK_LIBRARIES library1 library2 ...]
  [LINK_DIRECTORIES directory1 directory2 ...]
  [INCLUDE_LIBRARIES target1 target2 ...]
  [INCLUDE_DIRECTORIES directory1 directory2 ...]
)
```

**Parameters:**
- **`SOURCES`** (required): List of source files for the runtime library (include autogen files)
- **`COMPILE_DEFINITIONS`** (optional): Additional preprocessor definitions beyond automatic ones
- **`LINK_LIBRARIES`** (optional): Additional libraries to link beyond automatic dependencies
- **`LINK_DIRECTORIES`** (optional): Additional library search directories
- **`INCLUDE_LIBRARIES`** (optional): Target libraries whose include directories should be inherited
- **`INCLUDE_DIRECTORIES`** (optional): Additional include directories beyond automatic ones

**Automatic Behavior:**
- Creates target: `${NAMESPACE}_${CHIMOD_NAME}_runtime`
- Creates alias: `${NAMESPACE}::${CHIMOD_NAME}_runtime`
- Automatically defines `CHIMAERA_RUNTIME=1` for runtime code
- Automatically links core Chimaera library (`chimaera::cxx` or `hermes_shm::cxx`)
- Automatically links `rt` library for POSIX real-time support
- For non-admin ChiMods: automatically links both `chimaera_admin_runtime` and `chimaera_admin_client`
- Automatically includes module headers from `include/` directory
- Links to client library if it exists
- Installs library and headers with proper CMake export configuration

**Example:**
```cmake
add_chimod_runtime(
  SOURCES 
    src/my_module_runtime.cc 
    src/autogen/my_module_lib_exec.cc
  COMPILE_DEFINITIONS MY_MODULE_RUNTIME_DEBUG=1
  LINK_LIBRARIES libaio
  INCLUDE_DIRECTORIES ${LIBAIO_INCLUDE_DIR}
)
```

#### Configuration Requirements
Before using these functions, ensure your ChiMod directory contains:

1. **`chimaera_mod.yaml`**: Module configuration file defining the module name
   ```yaml
   module_name: my_module
   ```

2. **Include structure**: Headers organized as `include/[namespace]/[module_name]/`
3. **Source files**: Client and runtime implementations with autogen files for runtime

#### Typical Usage Pattern
Most ChiMods use both functions together:

```cmake
# Create client library
add_chimod_client(
  SOURCES src/my_module_client.cc
)

# Create runtime library
add_chimod_runtime(
  SOURCES 
    src/my_module_runtime.cc 
    src/autogen/my_module_lib_exec.cc
)
```

**Function Dependencies:**
Both functions automatically handle common dependencies:

- **Core Library**: Automatically links appropriate Chimaera core library
- **Runtime Libraries**: `add_chimod_runtime()` automatically links `rt` library for async I/O operations
- **Admin ChiMod Integration**: For non-admin chimods, both functions automatically link admin libraries and include admin headers
- **Client-Runtime Linking**: Runtime automatically links to client library when both exist

This eliminates the need for manual dependency configuration in individual ChiMod CMakeLists.txt files.

### Target Naming and Linking

#### Target Format
The system uses underscore-based target names for consistency with CMake conventions:

**Target Names:**
- **Runtime**: `${NAMESPACE}_${CHIMOD_NAME}_runtime` (e.g., `chimaera_admin_runtime`)
- **Client**: `${NAMESPACE}_${CHIMOD_NAME}_client` (e.g., `chimaera_admin_client`)

**CMake Aliases (Recommended):**
- **Runtime**: `${NAMESPACE}::${CHIMOD_NAME}_runtime` (e.g., `chimaera::admin_runtime`)
- **Client**: `${NAMESPACE}::${CHIMOD_NAME}_client` (e.g., `chimaera::admin_client`)

**Package Names:**
- Format: `${NAMESPACE}_${CHIMOD_NAME}` (e.g., `chimaera_admin`)
- Used with: `find_package(chimaera_admin REQUIRED)`
- Core package: `chimaera` (automatically included by `find_package(chimaera)`)

**External Application Linking (Based on test/unit/external/CMakeLists.txt):**
```cmake
# External applications typically only need ChiMod client libraries
# Core library dependencies are automatically included
find_package(chimaera REQUIRED)
find_package(chimaera_admin REQUIRED)

target_link_libraries(my_external_app
  chimaera::admin_client            # Admin client (includes all dependencies)
  ${CMAKE_THREAD_LIBS_INIT}         # Threading support
)
# Note: chimaera::cxx is automatically included by ChiMod client libraries
```

**Internal Development Linking:**
```cmake
# For internal development within the Chimaera project
target_link_libraries(internal_app
  chimaera::admin_client            # ChiMod client
  chimaera::bdev_client             # BDev client 
  # Core dependencies are automatically linked by ChiMod libraries
)
```

### Automatic Dependencies

The ChiMod build functions automatically handle common dependencies:

**For Runtime Code:**
- **rt library**: Automatically linked for POSIX real-time library support (async I/O operations)
- **Admin ChiMod**: Automatically linked for all non-admin ChiMods (both runtime and client)
- **Admin includes**: Automatically added to include directories for non-admin ChiMods

**For All ChiMods:**
- Creates both client and runtime shared libraries
- Sets proper include directories (include/, ${CMAKE_SOURCE_DIR}/include)
- Automatically links core Chimaera dependencies
- Sets required compile definitions (CHI_CHIMOD_NAME, CHI_NAMESPACE)
- Configures proper build flags and settings

**Simplified Development:**
ChiMod developers no longer need to manually specify:
- `rt` library dependencies
- Admin ChiMod dependencies (`chimaera_admin_runtime`, `chimaera_admin_client`)
- Admin include directories
- Core Chimaera library dependencies
- Common linking patterns

**Important Note for External Applications:**
External applications linking against ChiMod libraries receive all necessary dependencies automatically. The ChiMod client libraries include the core Chimaera library as a transitive dependency.

**Automatic Installation:**
The ChiMod build functions automatically handle installation:
- Installs libraries to the correct destination
- Sets up proper runtime paths
- Configures installation properties
- Includes automatic dependencies in exported CMake packages
- No separate `install_chimod()` call required

### Targets Created by ChiMod Functions

When you call `add_chimod_client()` and `add_chimod_runtime()` with `CHIMOD_NAME YOUR_MODULE_NAME`, they create the following CMake targets using the underscore-based naming format:

#### Target Naming System
- **Actual Target Names**: `${NAMESPACE}_${CHIMOD_NAME}_runtime` and `${NAMESPACE}_${CHIMOD_NAME}_client`
- **CMake Aliases**: `${NAMESPACE}::${CHIMOD_NAME}_runtime` and `${NAMESPACE}::${CHIMOD_NAME}_client` (**recommended**)
- **Package Names**: `${NAMESPACE}_${CHIMOD_NAME}` (for `find_package()`)

#### Runtime Target: `${NAMESPACE}_${CHIMOD_NAME}_runtime`
- **Target Name**: `chimaera_YOUR_MODULE_NAME_runtime` (e.g., `chimaera_admin_runtime`, `chimaera_MOD_NAME_runtime`)
- **CMake Alias**: `chimaera::YOUR_MODULE_NAME_runtime` (e.g., `chimaera::admin_runtime`) - **recommended for linking**
- **Type**: Shared library (`.so` file)
- **Purpose**: Contains server-side execution logic, runs in the Chimaera runtime process
- **Compile Definitions**:
  - `CHI_CHIMOD_NAME="${CHIMOD_NAME}"` - Module name for runtime identification
  - `CHI_NAMESPACE="${NAMESPACE}"` - Project namespace
- **Include Directories**:
  - `include/` - Local module headers
  - `${CMAKE_SOURCE_DIR}/include` - Chimaera framework headers
- **Dependencies**: Links against `chimaera` library, rt library (automatic), admin dependencies (automatic)

#### Client Target: `${NAMESPACE}_${CHIMOD_NAME}_client`
- **Target Name**: `chimaera_YOUR_MODULE_NAME_client` (e.g., `chimaera_admin_client`, `chimaera_MOD_NAME_client`)
- **CMake Alias**: `chimaera::YOUR_MODULE_NAME_client` (e.g., `chimaera::admin_client`) - **recommended for linking**
- **Type**: Shared library (`.so` file)  
- **Purpose**: Contains client-side API, runs in user processes
- **Compile Definitions**:
  - `CHI_CHIMOD_NAME="${CHIMOD_NAME}"` - Module name for client identification
  - `CHI_NAMESPACE="${NAMESPACE}"` - Project namespace
- **Include Directories**:
  - `include/` - Local module headers
  - `${CMAKE_SOURCE_DIR}/include` - Chimaera framework headers
- **Dependencies**: Links against `chimaera` library, admin dependencies (automatic)

#### Namespace Configuration
The namespace is automatically read from `chimaera_repo.yaml` files. The system searches up the directory tree from the CMakeLists.txt location to find the first `chimaera_repo.yaml` file:

**Main project `chimaera_repo.yaml`:**
```yaml
namespace: chimaera  # Main project namespace
```

**Module repository `chimods/chimaera_repo.yaml`:**
```yaml
namespace: chimods   # Modules get this namespace
```

This means modules in the `chimods/` directory will use the "chimods" namespace, creating targets like `chimods_admin_runtime`, while other components use the main project namespace.

#### Example Output Files
For a module named "admin" with namespace "chimods" (from `chimods/chimaera_repo.yaml`), the build produces:
```
build/bin/libchimods_admin_runtime.so    # Runtime library  
build/bin/libchimods_admin_client.so     # Client library
```

#### Using the Targets
You can reference these targets in your CMakeLists.txt using the full target name:
```cmake
# Add custom properties to the runtime target
set_target_properties(chimaera_${CHIMOD_NAME}_runtime PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
)

# Add additional dependencies if needed
target_link_libraries(chimaera_${CHIMOD_NAME}_runtime PRIVATE some_external_lib)

# Or use the global property to get the actual target name
get_property(RUNTIME_TARGET GLOBAL PROPERTY ${CHIMOD_NAME}_RUNTIME_TARGET)
target_link_libraries(${RUNTIME_TARGET} PRIVATE some_external_lib)
```

### Module Configuration (chimaera_mod.yaml)
```yaml
name: MOD_NAME
version: 1.0.0
description: "Module description"
author: "Author Name"
methods:
  - kCreate
  - kCustom
dependencies: []
```

### Auto-Generated Method Files
Each module requires an auto-generated methods file at `include/[namespace]/MOD_NAME/autogen/MOD_NAME_methods.h`. This file must:

1. **Include chimaera.h**: Required for GLOBAL_CONST macro
2. **Use namespace constants**: Define methods as `GLOBAL_CONST chi::u32` values
3. **Follow naming convention**: Method names should start with `k` (e.g., `kCreate`, `kCustom`)

**Required Template:**
```cpp
#ifndef MOD_NAME_AUTOGEN_METHODS_H_
#define MOD_NAME_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>

namespace chimaera::MOD_NAME {

namespace Method {
  // Standard inherited methods (always include these)
  GLOBAL_CONST chi::u32 kCreate = 0;
  GLOBAL_CONST chi::u32 kDestroy = 1;
  GLOBAL_CONST chi::u32 kNodeFailure = 2;
  GLOBAL_CONST chi::u32 kRecover = 3;
  GLOBAL_CONST chi::u32 kMigrate = 4;
  GLOBAL_CONST chi::u32 kUpgrade = 5;
  
  // Module-specific methods (customize these)
  GLOBAL_CONST chi::u32 kCustom = 10;
  // Add more module-specific methods starting from 10+
}

} // namespace chimaera::MOD_NAME

#endif // MOD_NAME_AUTOGEN_METHODS_H_
```

**Important Notes:**
- **GLOBAL_CONST is required**: Do not use `const` or `constexpr` - use `GLOBAL_CONST`
- **Include chimaera.h**: This header defines the GLOBAL_CONST macro
- **Standard methods 0-5**: Always include the inherited methods (kCreate through kUpgrade)
- **Custom methods 10+**: Start custom methods from ID 10 to avoid conflicts
- **No static casting needed**: Use method values directly (e.g., `method_ = Method::kCreate;`)

### Runtime Entry Points
Use the `CHI_TASK_CC` macro to define module entry points:

```cpp
// At the end of your runtime source file (_runtime.cc)
CHI_TASK_CC(your_namespace::YourContainerClass)
```

This macro automatically generates all required extern "C" functions and gets the module name from `YourContainerClass::CreateParams::chimod_lib_name`:
- `alloc_chimod()` - Creates container instance
- `new_chimod()` - Creates and initializes container  
- `get_chimod_name()` - Returns module name
- `destroy_chimod()` - Destroys container instance

**Requirements for CHI_TASK_CC to work:**
1. Your runtime class must define a public typedef: `using CreateParams = your_namespace::CreateParams;`
2. Your CreateParams struct must have: `static constexpr const char* chimod_lib_name = "your_module_name";`

**IMPORTANT:** The `chimod_lib_name` should **NOT** include the `_runtime` suffix. The module manager automatically appends `_runtime` when loading the library. For example, use `"chimaera_mymodule"` not `"chimaera_mymodule_runtime"`.

Example:
```cpp
namespace chimaera::your_module {

struct CreateParams {
  static constexpr const char* chimod_lib_name = "chimaera_your_module";
  // ... other parameters
};

class Runtime : public chi::Container {
public:
  using CreateParams = chimaera::your_module::CreateParams;  // Required for CHI_TASK_CC
  // ... rest of class
};

}  // namespace chimaera::your_module
```
- `is_chimaera_chimod_` - Module identification flag

## Auto-Generated Code Pattern

### Overview

ChiMods use auto-generated source files to implement the Container virtual APIs (Run, Monitor, Del, SaveIn, LoadIn, SaveOut, LoadOut, NewCopy). This approach provides consistent dispatch logic and reduces boilerplate code.

### New Pattern: Auto-Generated Source Files

Instead of using inline functions in headers, ChiMods now use auto-generated `.cc` source files that implement the virtual methods directly. This pattern:

- **Eliminates inline dispatchers**: Virtual methods are implemented directly in auto-generated source
- **Reduces header dependencies**: No need to include autogen headers in runtime files
- **Improves compilation**: Source files compile once, not in every including file
- **Maintains consistency**: All ChiMods use the same dispatch pattern

### File Structure

```
src/
└── autogen/
    └── MOD_NAME_lib_exec.cc    # Auto-generated virtual method implementations
```

The auto-generated source file contains:
- Container virtual method implementations (Run, Del, etc.)
- Switch-case dispatch based on method IDs
- Proper task type casting and method invocation
- IPC manager integration for task lifecycle management

### Auto-Generated Source Template

```cpp
/**
 * Auto-generated execution implementation for MOD_NAME ChiMod
 * Implements Container virtual APIs directly using switch-case dispatch
 * 
 * This file is autogenerated - do not edit manually.
 * Changes should be made to the autogen tool or the YAML configuration.
 */

#include <[namespace]/MOD_NAME/MOD_NAME_runtime.h>
#include <[namespace]/MOD_NAME/autogen/MOD_NAME_methods.h>
#include <chimaera/chimaera.h>

namespace chimaera::MOD_NAME {

//==============================================================================
// Runtime Virtual API Implementations
//==============================================================================

void Runtime::Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr, chi::RunContext& rctx) {
  switch (method) {
    case Method::kCreate: {
      Create(task_ptr.Cast<CreateTask>(), rctx);
      break;
    }
    case Method::kDestroy: {
      Destroy(task_ptr.Cast<DestroyTask>(), rctx);
      break;
    }
    case Method::kCustom: {
      Custom(task_ptr.Cast<CustomTask>(), rctx);
      break;
    }
    default: {
      // Unknown method - do nothing
      break;
    }
  }
}

void Runtime::Del(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) {
  // Use IPC manager to deallocate task from shared memory
  auto* ipc_manager = CHI_IPC;
  
  switch (method) {
    case Method::kCreate: {
      ipc_manager->DelTask(task_ptr.Cast<CreateTask>());
      break;
    }
    case Method::kDestroy: {
      ipc_manager->DelTask(task_ptr.Cast<DestroyTask>());
      break;
    }
    case Method::kCustom: {
      ipc_manager->DelTask(task_ptr.Cast<CustomTask>());
      break;
    }
    default: {
      // For unknown methods, still try to delete from main segment
      ipc_manager->DelTask(task_ptr);
      break;
    }
  }
}

// SaveIn, LoadIn, SaveOut, LoadOut, and NewCopy follow similar patterns...

} // namespace chimaera::MOD_NAME
```

### Runtime Implementation Changes

With the new autogen pattern, runtime source files (`MOD_NAME_runtime.cc`) no longer include autogen headers or implement dispatcher methods:

#### Before (Old Pattern):
```cpp
// No autogen header includes needed with new pattern

void Runtime::Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr, chi::RunContext& rctx) {
  // Dispatch to the appropriate method handler
  chimaera::MOD_NAME::Run(this, method, task_ptr, rctx);
}

// Similar dispatcher implementations for Del, SaveIn, LoadIn, SaveOut, LoadOut, NewCopy...
```

#### After (New Pattern):
```cpp
// No autogen header includes needed
// No dispatcher method implementations needed

// Virtual method implementations are now in src/autogen/MOD_NAME_lib_exec.cc
// Runtime source focuses only on business logic methods like Create(), Custom(), etc.
```

### CMake Integration

The auto-generated source file must be included in the `RUNTIME_SOURCES`:

```cmake
add_chimod_client(
  CHIMOD_NAME MOD_NAME
  SOURCES src/MOD_NAME_client.cc
)
add_chimod_runtime(
  CHIMOD_NAME MOD_NAME
  SOURCES src/MOD_NAME_runtime.cc src/autogen/MOD_NAME_lib_exec.cc
)
```

### Benefits of the New Pattern

1. **Cleaner Runtime Code**: Runtime implementations focus on business logic, not dispatching
2. **Better Compilation**: Source files compile once instead of being inlined in every header include
3. **Consistent Pattern**: All ChiMods use identical dispatch logic
4. **Header Simplification**: No need to include complex autogen headers
5. **Better IDE Support**: Proper source files work better with IDEs than inline templates

### Migration Guide

To migrate from the old inline header pattern to the new source pattern:

1. **Create autogen source directory**: `mkdir -p src/autogen/`
2. **Generate new autogen source**: Create `src/autogen/MOD_NAME_lib_exec.cc` with virtual method implementations
3. **Remove autogen header includes**: Delete `#include <[namespace]/MOD_NAME/autogen/MOD_NAME_lib_exec.h>` from runtime source (replaced by .cc files)
4. **Remove dispatcher methods**: Delete all virtual method implementations from runtime source (Run, Del, etc.)
5. **Update CMakeLists.txt**: Add autogen source to `RUNTIME_SOURCES`
6. **Keep business logic**: Retain the actual task processing methods (Create, Custom, etc.)

### Important Notes

- **Auto-generated files**: These files should be generated by tools, not hand-written
- **Do not edit**: Manual changes to autogen files will be lost when regenerated
- **Template consistency**: All ChiMods should follow the same autogen template pattern
- **Build integration**: Autogen source files must be included in CMake build

## External ChiMod Development

When developing ChiMods in external repositories (outside the main Chimaera project), you need to link against the installed Chimaera libraries and use the CMake package discovery system.

### Prerequisites

Before developing external ChiMods, ensure Chimaera is properly installed:

```bash
# Configure and build Chimaera
cmake --preset debug
cmake --build build

# Install Chimaera libraries and CMake configs  
cmake --install build --prefix /usr/local
```

This installs:
- Core Chimaera library (`libcxx.so`)
- ChiMod libraries (`libchimaera_admin_runtime.so`, `libchimaera_admin_client.so`, etc.)
- CMake package configuration files for external discovery
- Header files for development

### External ChiMod Repository Structure

Your external ChiMod repository should follow this structure:

```
my_external_chimod/
├── chimaera_repo.yaml          # Repository namespace configuration
├── CMakeLists.txt              # Root CMake configuration
├── modules/                    # ChiMod modules directory (name is flexible)
│   └── my_module/
│       ├── chimaera_mod.yaml   # Module configuration
│       ├── CMakeLists.txt      # Module build configuration  
│       ├── include/
│       │   └── [namespace]/
│       │       └── my_module/
│       │           ├── my_module_client.h
│       │           ├── my_module_runtime.h
│       │           ├── my_module_tasks.h
│       │           └── autogen/
│       │               └── my_module_methods.h
│       └── src/
│           ├── my_module_client.cc
│           ├── my_module_runtime.cc
│           └── autogen/
│               └── my_module_lib_exec.cc
```

**Note**: The directory name for modules (shown here as `modules/`) is flexible. You can use `chimods/`, `components/`, `plugins/`, or any other name that fits your project structure. The directory name doesn't need to match the namespace.

### Repository Configuration (chimaera_repo.yaml)

Create a `chimaera_repo.yaml` file in your repository root to define the namespace:

```yaml
# Repository-level configuration
namespace: myproject      # Your custom namespace (replaces "chimaera")
```

This namespace will be used for:
- CMake target names: `myproject_my_module_runtime`, `myproject_my_module_client`
- Library file names: `libmyproject_my_module_runtime.so`, `libmyproject_my_module_client.so`
- C++ namespaces: `myproject::my_module`

### Root CMakeLists.txt

Your repository's root `CMakeLists.txt` must find and link to the installed Chimaera packages:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_external_chimod)

set(CMAKE_CXX_STANDARD_20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required Chimaera packages
# These packages are installed by 'cmake --install build --prefix /usr/local'
find_package(chimaera REQUIRED)              # Core Chimaera (automatically includes ChimaeraCommon.cmake)
find_package(chimaera_admin REQUIRED)        # Admin ChiMod (often required)

# Set CMAKE_PREFIX_PATH if Chimaera is installed in a custom location
# set(CMAKE_PREFIX_PATH "/path/to/[namespace]/install" ${CMAKE_PREFIX_PATH})

# ChimaeraCommon.cmake utilities are automatically included by find_package(chimaera)
# This provides add_chimod_client(), add_chimod_runtime(), and other build functions

# Add subdirectories containing your ChiMods
add_subdirectory(modules/my_module)  # Use your actual directory name
```

### ChiMod CMakeLists.txt

Each ChiMod's `CMakeLists.txt` uses the standard Chimaera build utilities:

```cmake
cmake_minimum_required(VERSION 3.20)

# Create both client and runtime libraries using standard Chimaera utilities
# These functions are provided by ChimaeraCommon.cmake (automatically included via find_package(chimaera))
# Creates targets: my_namespace_my_module_client, my_namespace_my_module_runtime
# Creates aliases: my_namespace::my_module_client, my_namespace::my_module_runtime
add_chimod_client(
  CHIMOD_NAME my_module
  SOURCES src/my_module_client.cc
)
add_chimod_runtime(
  CHIMOD_NAME my_module
  SOURCES
    src/my_module_runtime.cc 
    src/autogen/my_module_lib_exec.cc
)

# Installation is automatic - no separate install_chimod() call required
# Package name: my_namespace_my_module (for find_package)

# Optional: Add additional dependencies if your ChiMod needs external libraries
# get_property(RUNTIME_TARGET GLOBAL PROPERTY my_module_RUNTIME_TARGET)
# get_property(CLIENT_TARGET GLOBAL PROPERTY my_module_CLIENT_TARGET)
# target_link_libraries(${RUNTIME_TARGET} PRIVATE some_external_lib)
# target_link_libraries(${CLIENT_TARGET} PRIVATE some_external_lib)
```

### External Applications Using Your ChiMod
Once installed, external applications can find and link to your ChiMod. Based on our external unit test patterns (see test/unit/external-chimod/CMakeLists.txt):

```cmake
# External application CMakeLists.txt
find_package(my_namespace_my_module REQUIRED)  # Your ChiMod package
find_package(chimaera REQUIRED)                # Core Chimaera (automatically includes utilities)
find_package(chimaera_admin REQUIRED)          # Admin ChiMod (often required)

# Simple linking pattern - ChiMod libraries include all dependencies
target_link_libraries(my_external_app
  my_namespace::my_module_client    # Your ChiMod client
  chimaera::admin_client            # Admin client (if needed)
  ${CMAKE_THREAD_LIBS_INIT}         # Threading support
)
# Core Chimaera library is automatically included by ChiMod dependencies
```

### External ChiMod Implementation

Your external ChiMod implementation follows the same patterns as internal ChiMods:

#### CreateParams Configuration

In your `my_module_tasks.h`, the `CreateParams` must reference your custom namespace:

```cpp
struct CreateParams {
  // Your module-specific parameters
  std::string config_data_;
  chi::u32 worker_count_;
  
  // CRITICAL: Library name must match your namespace
  static constexpr const char* chimod_lib_name = "myproject_my_module";
  
  // Constructors and serialization...
};
```

#### C++ Namespace

Use your custom namespace throughout your implementation:

```cpp
// In all header and source files
namespace myproject::my_module {

// Your ChiMod implementation...
class Runtime : public chi::Container {
  // Implementation...
};

class Client : public chi::ContainerClient {  
  // Implementation...
};

} // namespace myproject::my_module
```

### Building External ChiMods

```bash
# Configure your external ChiMod project
mkdir build && cd build
cmake ..

# Build your ChiMods
make

# Optional: Install your ChiMods
make install
```

The build system will automatically:
- Link all necessary core Chimaera dependencies
- Link against `chimaera::admin_client` and `chimaera::admin_runtime` (for non-admin modules)
- Generate libraries with your custom namespace: `libmyproject_my_module_runtime.so`
- Configure proper include paths and dependencies

### Usage in Applications

Applications using your external ChiMod would reference it as:

```cpp
#include <chimaera/chimaera.h>
#include <[namespace]/my_module/my_module_client.h>
#include <[namespace]/admin/admin_client.h>

int main() {
  // Initialize Chimaera (client mode with embedded runtime)
  chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

  // Create your ChiMod client
  const chi::PoolId pool_id = chi::PoolId(7000, 0);
  myproject::my_module::Client client(pool_id);

  // Use your ChiMod
  auto pool_query = chi::PoolQuery::Local();
  client.Create(pool_query);
}
```

### CHIMAERA_INIT Initialization Modes

Chimaera provides a unified initialization function `CHIMAERA_INIT()` that supports different operational modes:

**Client Mode with Embedded Runtime (Most Common):**
```cpp
// Initialize both client and runtime in single process
// Recommended for: Applications, unit tests, and benchmarks
chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
```

**Client-Only Mode (Advanced):**
```cpp
// Initialize client only - connects to external runtime
// Recommended for: Production deployments with separate runtime process
chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false);
```

**Runtime/Server Mode (Advanced):**
```cpp
// Initialize runtime/server only - no client
// Recommended for: Standalone runtime processes
chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer, false);
```

**Usage Example (Unit Tests/Benchmarks):**
```cpp
#include <chimaera/chimaera.h>
#include <[namespace]/my_module/my_module_client.h>

TEST(MyModuleTest, BasicOperation) {
  // Initialize both client and runtime in single process
  chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

  // Create your ChiMod client
  const chi::PoolId pool_id = chi::PoolId(7000, 0);
  myproject::my_module::Client client(pool_id);

  // Test your ChiMod functionality
  auto pool_query = chi::PoolQuery::Local();
  client.Create(pool_query, "test_pool");

  // Assertions and test logic...
}
```

**When to Use Each Mode:**
- **Client with Embedded Runtime** (`kClient, true`): Unit tests, benchmarks, and standalone applications
- **Client Only** (`kClient, false`): Production applications connecting to existing external runtime
- **Server/Runtime Only** (`kServer, false`): Dedicated runtime processes

### Dependencies and Installation Paths

External ChiMod development requires these components to be installed:

1. **Core Package**: `chimaera` (includes main library and ChimaeraCommon.cmake utilities)
2. **Admin ChiMod**: `chimaera::admin_client` and `chimaera::admin_runtime` (required for most modules)
3. **CMake Configs**: Package discovery files (automatically installed with packages)
4. **Headers**: All Chimaera framework headers (installed with packages)

All build utilities (`add_chimod_client()`, `add_chimod_runtime()`) are automatically available via `find_package(chimaera)`.

If Chimaera is installed in a custom location, set `CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH="/path/to/[namespace]/install:$CMAKE_PREFIX_PATH"
```

### Common External Development Issues

**ChimaeraCommon.cmake Not Found:**
- Ensure Chimaera was installed with `cmake --install build --prefix <path>`
- Verify `CMAKE_PREFIX_PATH` includes the Chimaera installation directory
- Check that `find_package(chimaera REQUIRED)` succeeded (ChimaeraCommon.cmake is included automatically)

**Library Name Mismatch:**
- Ensure `CreateParams::chimod_lib_name` exactly matches your namespace and module name
- For namespace "myproject" and module "my_module": `chimod_lib_name = "myproject_my_module"`
- The system automatically appends "_runtime" to find the runtime library
- Target names use format: `myproject_my_module_runtime` and `myproject_my_module_client`

**Missing Dependencies:**
- The ChiMod build functions automatically link admin and rt library dependencies
- Ensure all external dependencies (Boost, MPI, etc.) are available in your build environment  
- Use the same dependency versions that Chimaera was built with
- For runtime code, rt library is automatically included for async I/O support

### External ChiMod Checklist

- [ ] **Repository Configuration**: `chimaera_repo.yaml` with custom namespace
- [ ] **CMake Setup**: Root CMakeLists.txt finds `chimaera` package  
- [ ] **ChiMod Configuration**: `chimaera_mod.yaml` with method definitions
- [ ] **Library Name**: `CreateParams::chimod_lib_name` matches namespace pattern
- [ ] **C++ Namespace**: All code uses custom namespace consistently  
- [ ] **Build Integration**: ChiMod CMakeLists.txt uses `add_chimod_client()` and `add_chimod_runtime()` (installation is automatic)
- [ ] **Dependencies**: All required external libraries available at build time
- [ ] **Automatic Linking**: Rely on ChiMod build functions for rt and admin dependencies

## Example Module

See the `chimods/MOD_NAME` directory for a complete working example that demonstrates:
- Task definition with proper constructors
- Client API with async-only methods
- Runtime container with execution logic
- Build system integration
- YAML configuration

### Creating a New Module
1. Copy the MOD_NAME template directory
2. Rename all MOD_NAME occurrences to your module name
3. Update the chimaera_mod.yaml configuration
4. Define your tasks in the _tasks.h file
5. Implement client API in _client.h/cc
6. Implement runtime logic in _runtime.h/cc
7. Add `CHI_TASK_CC(YourContainerClass)` at the end of runtime source
8. Add to the build system
9. Test with client and runtime

## Recent Changes and Best Practices

### Container Initialization Pattern
Starting with the latest version, container initialization has been simplified:

1. **No Separate Init Method**: The `Init` method has been merged with `Create`
2. **Create Does Everything**: The `Create` method now handles both container creation and initialization
3. **Access to Task Data**: Since `Create` receives the CreateTask, you have access to pool_id and pool_query from the task

### Framework-Managed Task Cleanup
Task cleanup is handled by the framework using the IPC manager:

1. **No Custom Del Methods Required**: Individual `DelTaskType` methods are no longer needed
2. **IPC Manager Handles Cleanup**: The framework automatically calls `ipc_manager->DelTask()` to deallocate tasks from shared memory
3. **Memory Segment Deallocation**: Tasks are properly removed from their respective memory segments (typically `kMainSegment`)

### Simplified ChiMod Entry Points
ChiMod entry points are now hidden behind the `CHI_TASK_CC` macro:

1. **Single Macro Call**: Replace complex extern "C" blocks with one macro
2. **Automatic Container Integration**: Works seamlessly with `chi::Container` base class
3. **Cleaner Module Code**: Eliminates boilerplate entry point code

```cpp
// Old approach (complex extern "C" block)
extern "C" {
  chi::ChiContainer* alloc_chimod() { /* ... */ }
  chi::ChiContainer* new_chimod(/*...*/) { /* ... */ }
  const char* get_chimod_name() { /* ... */ }
  void destroy_chimod(/*...*/) { /* ... */ }
  bool is_chimaera_chimod_ = true;
}

// New approach (simple macro)
CHI_TASK_CC(chimaera::MOD_NAME::Runtime)
```

```cpp
void Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx) {
  // Container is already initialized via Init() before Create is called
  // Do NOT call Init() here

  // Container-specific initialization logic
  // All tasks will be routed through the external queue lanes
  // which are automatically mapped to workers at runtime startup

  // Container is now ready for operation
}
```

### FullPtr Parameter Pattern
All runtime methods now use `hipc::FullPtr<TaskType>` instead of raw pointers:

```cpp
// Old pattern (deprecated)
void Custom(CustomTask* task, chi::RunContext& ctx) { ... }

// New pattern (current)
void Custom(hipc::FullPtr<CustomTask> task, chi::RunContext& ctx) { ... }
```

**Benefits of FullPtr:**
- **Shared Memory Safety**: Provides safe access across process boundaries
- **Automatic Dereferencing**: Use `task->field` just like raw pointers
- **Memory Management**: Framework handles allocation/deallocation
- **Null Checking**: Use `task.IsNull()` to check validity

### Migration Guide
When updating existing modules:

1. **Remove Init Override**: Delete custom `Init` method implementations
2. **Update Create Method**: Move initialization logic from `Init` to `Create`
3. **Change Method Signatures**: Replace `TaskType*` with `hipc::FullPtr<TaskType>`
4. **Update Monitor Methods**: Ensure all monitoring methods use FullPtr
5. **Implement kLocalSchedule**: Every Monitor method MUST implement `kLocalSchedule` mode
6. **Remove Del Methods**: Delete all `DelTaskType` methods - framework calls `ipc_manager->DelTask()` automatically
7. **Update Autogen Files**: Ensure Del dispatcher calls `ipc_manager->DelTask()` instead of custom Del methods
8. **Replace Entry Points**: Replace extern "C" blocks with `CHI_TASK_CC(ClassName)` macro
9. **Remove Completion Calls**: Framework handles task completion automatically

## Custom Namespace Configuration

### Overview
While the default namespace is `chimaera`, you can customize the namespace for your ChiMod modules. This is useful for:
- **Project Branding**: Use your own project or company namespace
- **Avoiding Conflicts**: Prevent naming conflicts with other ChiMod collections
- **Module Organization**: Group related modules under a custom namespace

### Configuring Custom Namespace

The namespace is controlled by the `chimaera_repo.yaml` file in your project root:

```yaml
namespace: your_custom_namespace
```

For example:
```yaml
namespace: mycompany
```

### Required Changes for Custom Namespace

When using a custom namespace, you must update several components:

#### 1. **CreateParams chimod_lib_name**
The most critical change is updating the `chimod_lib_name` in your CreateParams:

```cpp
// Default chimaera namespace
struct CreateParams {
  static constexpr const char* chimod_lib_name = "chimaera_your_module";
};

// Custom namespace example
struct CreateParams {
  static constexpr const char* chimod_lib_name = "mycompany_your_module";
};
```

#### 2. **Module Namespace Declaration**
Update your module's C++ namespace:

```cpp
// Default
namespace chimaera::your_module {
  // module code
}

// Custom
namespace mycompany::your_module {
  // module code
}
```

#### 3. **CMake Library Names**
The CMake system automatically uses your custom namespace. Libraries will be named:
- Default: `libchimaera_module_runtime.so`, `libchimaera_module_client.so`
- Custom: `libmycompany_module_runtime.so`, `libmycompany_module_client.so`

#### 4. **Runtime Integration**
If your runtime code references the admin module or other system modules, update the references:

```cpp
// Default admin module reference
auto* admin_chimod = module_manager->GetChiMod("chimaera_admin");

// Custom namespace admin module
auto* admin_chimod = module_manager->GetChiMod("mycompany_admin");
```

### Checklist for Custom Namespace

- [ ] **Update chimaera_repo.yaml** with your custom namespace
- [ ] **Update CreateParams::chimod_lib_name** to use custom namespace prefix
- [ ] **Update C++ namespace declarations** in all module files
- [ ] **Update runtime references** to admin module and other system modules
- [ ] **Update any hardcoded module names** in configuration or startup code
- [ ] **Rebuild all modules** after namespace changes
- [ ] **Update library search paths** if needed for deployment

### Example: Complete Custom Namespace Module

```yaml
# chimaera_repo.yaml
namespace: mycompany
```

```cpp
// mymodule_tasks.h
namespace mycompany::mymodule {

struct CreateParams {
  static constexpr const char* chimod_lib_name = "mycompany_mymodule";
  // ... other parameters
};

using CreateTask = chimaera::admin::BaseCreateTask<CreateParams, Method::kCreate>;

}  // namespace mycompany::mymodule
```

```cpp
// mymodule_runtime.h
namespace mycompany::mymodule {

class Runtime : public chi::Container {
public:
  using CreateParams = mycompany::mymodule::CreateParams;  // Required for CHI_TASK_CC
  // ... rest of class
};

}  // namespace mycompany::mymodule
```

```cpp
// mymodule_runtime.cc
CHI_TASK_CC(mycompany::mymodule::Runtime)
```

### Important Notes

- **Library Name Consistency**: The `chimod_lib_name` must exactly match what the CMake system generates
- **Admin Module**: If you customize the namespace, you may also want to rebuild the admin module with your custom namespace
- **Backward Compatibility**: Changing namespace breaks compatibility with existing deployments using default namespace
- **Documentation**: Update any module-specific documentation to reflect the new namespace

## Advanced Topics

### Task Scheduling
Tasks can be scheduled with different priorities:
- `kLowLatency`: For time-critical operations
- `kHighLatency`: For batch processing

### Automatic Routing Architecture
The framework handles all task routing automatically:

1. **Client-Side Enqueuing**:
   - Tasks are enqueued via `IpcManager::Enqueue()` from client code
   - Lane selection uses PID+TID hash for automatic distribution across lanes
   - Formula: `lane_id = hash(PID, TID) % num_lanes`

2. **Worker-Lane Mapping** (1:1 Direct Mapping):
   - Number of lanes automatically equals number of sched workers (default: 8)
   - Each worker assigned exactly one lane: worker i → lane i
   - No round-robin needed - perfect 1:1 correspondence
   - Lane headers track assigned worker ID

3. **No Configuration Required**:
   - Lane count automatically matches sched worker count from config
   - No separate `task_queue_lanes` configuration needed
   - Change worker count → lane count adjusts automatically

**Example**: With 8 sched workers (default):
- 8 lanes created automatically in external queue
- Worker 0 → Lane 0, Worker 1 → Lane 1, ..., Worker 7 → Lane 7
- Client tasks distributed via hash to lanes 0-7
- Each worker processes tasks from its dedicated lane

### Error Handling
```cpp
void Custom(hipc::FullPtr<CustomTask> task, chi::RunContext& ctx) {
  try {
    // Operation logic
    task->result_code_ = 0;
  } catch (const std::exception& e) {
    task->result_code_ = 1;
    task->data_ = chi::string(main_allocator_, e.what());
  }
  // Framework handles task completion automatically
}
```

## Debugging Tips

1. **Check Shared Memory**: Use `ipcs -m` to view segments
2. **Verify Task State**: Check task completion status
3. **Monitor Queue Depth**: Use GetProcessQueue() to inspect queues
4. **Enable Debug Logging**: Set CHI_DEBUG environment variable
5. **Use GDB**: Attach to runtime process for debugging

### Common Issues and Solutions

**Tasks Not Being Executed:**
- **Cause**: Tasks not being routed to worker queues
- **Solution**: Verify task pool_query is set correctly and pool exists
- **Debug**: Add logging in task execution methods to verify they're being called

**Queue Overflow or Deadlocks:**
- **Cause**: Tasks being enqueued but not dequeued from lanes
- **Solution**: Verify lane creation in Create() method and proper task routing
- **Debug**: Check lane sizes with `lane->Size()` and `lane->IsEmpty()`

**Memory Leaks in Shared Memory:**
- **Cause**: Tasks not being properly cleaned up
- **Solution**: Ensure framework Del dispatcher is working correctly
- **Debug**: Monitor shared memory usage with `ipcs -m`

## Performance Considerations

1. **Minimize Allocations**: Reuse buffers when possible
2. **Batch Operations**: Submit multiple tasks together
3. **Use Appropriate Segments**: Put large data in client_data_segment
4. **Avoid Blocking**: Use async operations when possible
5. **Profile First**: Measure before optimizing

## Unit Testing

Unit testing for ChiMods is covered in the separate [Module Test Guide](module_test_guide.md). This guide provides comprehensive information on:

- Test environment setup and configuration
- Environment variables and module discovery
- Test framework integration patterns
- Complete test examples with fixtures
- CMake integration and build setup
- Best practices for ChiMod testing

The test guide demonstrates how to test both runtime and client components in the same process, enabling comprehensive integration testing without complex multi-process coordination.

## Quick Reference Checklist

When creating a new Chimaera module, ensure you have:

### Task Definition Checklist (`_tasks.h`)
- [ ] Tasks inherit from `chi::Task` or use GetOrCreatePoolTask template (recommended for non-admin modules)
- [ ] **Use GetOrCreatePoolTask**: For non-admin modules instead of BaseCreateTask directly
- [ ] **Use BaseCreateTask with IS_ADMIN=true**: Only for admin module
- [ ] SHM constructor with CtxAllocator parameter (if custom task)
- [ ] Emplace constructor with all required parameters (if custom task)
- [ ] Uses HSHM serializable types (chi::string, chi::vector, etc.)
- [ ] Method constant assigned in constructor (e.g., `method_ = Method::kCreate;`)
- [ ] **No static casting**: Use Method namespace constants directly
- [ ] Include auto-generated methods file for Method constants

### Runtime Container Checklist (`_runtime.h/cc`)
- [ ] Inherits from `chi::Container`
- [ ] **Init() method overridden** - calls base class Init() then initializes client for this ChiMod
- [ ] Create() method does NOT call `chi::Container::Init()` (container is already initialized before Create is called)
- [ ] All task methods use `hipc::FullPtr<TaskType>` parameters
- [ ] **NO custom Del methods needed** - framework calls `ipc_manager->DelTask()` automatically
- [ ] Uses `CHI_TASK_CC(ClassName)` macro for entry points
- [ ] **Routing is automatic** - tasks are routed through external queue lanes mapped to workers (1:1 worker-to-lane mapping)

### Client API Checklist (`_client.h/cc`)
- [ ] Inherits from `chi::ContainerClient`
- [ ] Uses `CHI_IPC->NewTask<TaskType>()` for allocation
- [ ] Uses `CHI_IPC->Send()` for task submission (returns Future)
- [ ] **Async-only API**: All methods return `chi::Future<TaskType>`
- [ ] **CRITICAL**: AsyncCreate passes `this` pointer for PostWait callback to update pool_id_

### Build System Checklist
- [ ] CMakeLists.txt creates both client and runtime libraries
- [ ] chimaera_mod.yaml defines module metadata
- [ ] **Auto-generated methods file**: `autogen/MOD_NAME_methods.h` with Method namespace
- [ ] **Include chimaera.h**: In methods file for GLOBAL_CONST macro
- [ ] **GLOBAL_CONST constants**: Use namespace constants, not enum class
- [ ] Proper install targets configured
- [ ] Links against chimaera library

### Common Pitfalls to Avoid
- [ ] ❌ **CRITICAL: Not updating pool_id_ in Create methods** (leads to incorrect pool ID for subsequent operations)
- [ ] ❌ Using raw pointers instead of FullPtr in runtime methods
- [ ] ❌ **Calling `chi::Container::Init()` in Create method** (container is already initialized by framework before Create is called)
- [ ] ❌ **Not overriding `Init()` method** (required to initialize the client member)
- [ ] ❌ Using non-HSHM types in task data members
- [ ] ❌ Implementing custom Del methods (framework calls `ipc_manager->DelTask()` automatically)
- [ ] ❌ Writing complex extern "C" blocks (use `CHI_TASK_CC` macro instead)
- [ ] ❌ **Using static_cast with Method values** (use Method::kName directly)
- [ ] ❌ Attempting to manually manage task routing (framework handles automatically)
- [ ] ❌ **Missing chimaera.h include** in methods file (GLOBAL_CONST won't work)
- [ ] ❌ **Using enum class for methods** (use namespace with GLOBAL_CONST instead)
- [ ] ❌ **Using BaseCreateTask directly for non-admin modules** (use GetOrCreatePoolTask instead)
- [ ] ❌ **Forgetting GetOrCreatePoolTask template** for container creation (reduces boilerplate)

## Pool Name Requirements
**CRITICAL**: All ChiMod Create functions MUST require a user-provided `pool_name` parameter. Never auto-generate pool names using `pool_id_` during Create operations.

### Why Pool Names Are Required
1. **pool_id_ Not Available**: `pool_id_` is not set until after Create completes
2. **User Intent**: Users should explicitly name their pools for better organization
3. **Uniqueness**: Users can ensure uniqueness better than auto-generation
4. **Debugging**: Named pools are easier to identify during debugging

### Pool Naming Guidelines
- **Descriptive Names**: Use names that identify purpose or content
- **File-based Devices**: For BDev file devices, `pool_name` serves as the file path
- **RAM-based Devices**: For BDev RAM devices, `pool_name` should be unique identifier
- **Unique Identifiers**: Consider timestamp + PID combinations when needed

### Correct Pool Naming Usage
```cpp
// BDev file-based device - pool_name is the file path
std::string file_path = "/path/to/device.dat";
const chi::PoolId bdev_pool_id(7000, 0);
bdev_client.Create(pool_query, file_path, bdev_pool_id,
                   chimaera::bdev::BdevType::kFile);

// BDev RAM-based device - pool_name is unique identifier
std::string pool_name = "my_ram_device_" + std::to_string(timestamp);
const chi::PoolId ram_pool_id(7001, 0);
bdev_client.Create(pool_query, pool_name, ram_pool_id,
                   chimaera::bdev::BdevType::kRam, ram_size);

// Other ChiMods - pool_name is descriptive identifier
std::string pool_name = "my_container_" + user_identifier;
const chi::PoolId mod_pool_id(7002, 0);
mod_client.Create(pool_query, pool_name, mod_pool_id);
```

### Incorrect Pool Naming Usage
```cpp
// WRONG: Using pool_id_ before it's set (will be 0 or garbage)
std::string bad_name = "pool_" + std::to_string(pool_id_.ToU64());

// WRONG: Using empty strings
client.Create(pool_query, "", pool_id);

// WRONG: Auto-generating inside Create function
// Create functions should not auto-generate names
void Create(pool_query) {
    std::string auto_name = "pool_" + generate_id();  // Wrong approach
}
```

### Client Interface Pattern (Async-Only)
All ChiMod clients should follow the async-only interface pattern:
```cpp
class Client : public chi::ContainerClient {
 public:
  // Async-only Create with required pool_name - returns Future
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery& pool_query,
      const std::string& pool_name,
      const chi::PoolId& custom_pool_id /* user-provided name */) {
    auto* ipc_manager = CHI_IPC;
    // Use pool_name directly, never generate internally
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool
        pool_query,
        CreateParams::chimod_lib_name,  // Never hardcode
        pool_name,          // User-provided name
        custom_pool_id,     // Target pool ID
        this                // Client pointer for PostWait callback
    );
    return ipc_manager->Send(task);
  }

  // Example of async-only operation pattern
  chi::Future<CustomTask> AsyncCustom(
      const chi::PoolQuery& pool_query,
      const std::string& input_data,
      chi::u32 operation_id) {
    auto* ipc_manager = CHI_IPC;
    auto task = ipc_manager->NewTask<CustomTask>(
        chi::CreateTaskId(),
        pool_id_,           // Use client's pool_id_ for non-Create operations
        pool_query,
        input_data,
        operation_id);
    return ipc_manager->Send(task);
  }
};
```

**Usage Pattern (Caller Side):**
```cpp
// Initialize and create
chimaera::my_module::Client client;
const chi::PoolId pool_id(7000, 0);

auto create_task = client.AsyncCreate(chi::PoolQuery::Dynamic(), "my_pool", pool_id);
create_task.Wait();

if (create_task->GetReturnCode() != 0) {
    // Handle error
    return;
}

// Perform operations
auto op_task = client.AsyncCustom(chi::PoolQuery::Local(), "data", 1);
op_task.Wait();

// Access results
auto result = op_task->output_data_.str();
```

### BDev-Specific Requirements
- **Single Interface**: Use only one `AsyncCreate()` method (no multiple overloads)
- **File Devices**: `pool_name` parameter serves as the file path
- **RAM Devices**: `pool_name` parameter serves as unique identifier
- **Method Signature**: `AsyncCreate(pool_query, pool_name, custom_pool_id, bdev_type, total_size=0, io_depth=32, alignment=4096)`

## Compose Configuration Feature

The compose feature allows automatic pool creation from YAML configuration files. This enables declarative infrastructure setup where all required pools can be defined in configuration and created during runtime initialization or via utility script.

### CreateParams LoadConfig Requirement

**CRITICAL**: All ChiMod CreateParams structures MUST implement a `LoadConfig()` method to support compose feature.

```cpp
/**
 * Load configuration from PoolConfig (for compose mode)
 * Required for compose feature support
 * @param pool_config Pool configuration from compose section
 */
void LoadConfig(const chi::PoolConfig& pool_config) {
  // Parse YAML config string
  YAML::Node config = YAML::Load(pool_config.config_);

  // Load module-specific parameters from YAML
  if (config["parameter_name"]) {
    parameter_name_ = config["parameter_name"].as<Type>();
  }

  // Parse size strings (e.g., "2GB", "512MB")
  if (config["capacity"]) {
    std::string capacity_str = config["capacity"].as<std::string>();
    total_size_ = hshm::ConfigParse::ParseSize(capacity_str);
  }
}
```

### Compose Configuration Format

```yaml
compose:
- mod_name: chimaera_bdev        # ChiMod library name
  pool_name: ram://test          # Pool name (or file path for BDev)
  pool_query: dynamic            # Either "dynamic" or "local"
  pool_id: 200.0                 # Pool ID in "major.minor" format
  capacity: 2GB                  # Module-specific parameters
  bdev_type: ram                 # Additional parameters as needed
  io_depth: 32
  alignment: 4096

- mod_name: chimaera_another_mod
  pool_name: my_pool
  pool_query: local
  pool_id: 201.0
  custom_param: value
```

### Usage Modes

**1. Automatic During Runtime Init:**
Pools are automatically created when runtime initializes if compose section is present in configuration:
```bash
export CHI_SERVER_CONF=/path/to/config_with_compose.yaml
chimaera_start_runtime
```

**2. Manual via chimaera_compose Utility:**
Create pools using compose configuration against running runtime:
```bash
chimaera_compose /path/to/compose_config.yaml
```

### Implementation Checklist

When adding compose support to a ChiMod:

- [ ] Add `LoadConfig(const chi::PoolConfig& pool_config)` method to CreateParams
- [ ] Parse all module-specific parameters from YAML config
- [ ] Handle optional parameters with defaults
- [ ] Use `hshm::ConfigParse::ParseSize()` for size strings
- [ ] Include `<yaml-cpp/yaml.h>` and `<chimaera/config_manager.h>` in tasks header
- [ ] Test with compose configuration before release

### Example Admin ChiMod LoadConfig

```cpp
void LoadConfig(const chi::PoolConfig& pool_config) {
  // Admin doesn't have additional configuration fields
  // YAML config parsing would go here for modules with config fields
  (void)pool_config;  // Suppress unused parameter warning
}
```

### Example BDev ChiMod LoadConfig

```cpp
void LoadConfig(const chi::PoolConfig& pool_config) {
  YAML::Node config = YAML::Load(pool_config.config_);

  // Load BDev type
  if (config["bdev_type"]) {
    std::string type_str = config["bdev_type"].as<std::string>();
    if (type_str == "file") {
      bdev_type_ = BdevType::kFile;
    } else if (type_str == "ram") {
      bdev_type_ = BdevType::kRam;
    }
  }

  // Load capacity (parse size strings)
  if (config["capacity"]) {
    std::string capacity_str = config["capacity"].as<std::string>();
    total_size_ = hshm::ConfigParse::ParseSize(capacity_str);
  }

  // Load optional parameters
  if (config["io_depth"]) {
    io_depth_ = config["io_depth"].as<chi::u32>();
  }
  if (config["alignment"]) {
    alignment_ = config["alignment"].as<chi::u32>();
  }
}
```