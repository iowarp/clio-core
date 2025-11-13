# IOWarp Core - Unified Development Guide

This repository contains the unified IOWarp Core framework, integrating multiple components:
- **context-transport-primitives** (formerly cte-hermes-shm): Core transport and shared memory primitives
- **runtime**: IOWarp runtime system
- **context-transfer-engine**: Context transfer engine
- **context-assimilation-engine**: Context assimilation engine
- **context-exploration-engine**: Context exploration engine

## Code Style

Use the Google C++ style guide for C++.

You should store the pointer returned by the singleton GetInstance method. Avoid dereferencing GetInstance method directly using either -> or *. E.g., do not do ``hshm::Singleton<T>::GetInstance()->var_``. You should do ``auto *x = hshm::Singleton<T>::GetInstance(); x->var_;``.

Whenever you build a new function, always create a docstring for it. It should document what the parameters mean and the point of the function. It should be something easily parsed by doxygen.

NEVER use a null pool query. If you don't know, always use local.

Local QueueId should be named. NEVER use raw integers. This is the same for priorities. Please name them semantically.

All timing prints MUST include units of measurement in milliseconds (ms). Always print timing values in the order of milliseconds.

## Workflow

Use the incremental logic builder agent when making code changes.

Use the compiler subagent for making changes to cmakes and identifying places that need to be fixed in the code.

Always verify that code continue to compiles after making changes. Avoid commenting out code to fix compilation issues.

Whenever building unit tests, make sure to use the unit testing agent.

Whenever performing filesystem queries or executing programs, use the filesystem ops script agent.

NEVER DO MOCK CODE OR STUB CODE UNLESS SPECIFICALLY STATED OTHERWISE. ALWAYS IMPLEMENT REAL, WORKING CODE.

## Build Configuration

### Component Build Options
The unified IOWarp Core build system provides options to enable/disable components:
- `WRP_CORE_ENABLE_RUNTIME`: Enable runtime component (default: ON)
- `WRP_CORE_ENABLE_CTE`: Enable context-transfer-engine component (default: ON)
- `WRP_CORE_ENABLE_CAE`: Enable context-assimilation-engine component (default: ON)
- `WRP_CORE_ENABLE_CEE`: Enable context-exploration-engine component (default: ON)

Example:
```bash
cmake --preset=debug -DWRP_CORE_ENABLE_CTE=ON -DWRP_CORE_ENABLE_CAE=OFF
```

### Compilation Standards
- Always use the debug CMakePreset when compiling code in this repo
- Never hardcode paths in CMakeLists.txt files
- Use find_package() for all dependencies
- Follow ChiMod build patterns from MODULE_DEVELOPMENT_GUIDE.md
- All compilation warnings have been resolved as of the current state

### ChiMod Build Patterns

This project follows the Chimaera MODULE_DEVELOPMENT_GUIDE.md patterns for proper ChiMod development:

**Required Packages for ChiMod Development:**
```cmake
# Core Chimaera framework (includes ChimaeraCommon.cmake functions)
find_package(chimaera REQUIRED)              # Core library (chimaera::cxx)
find_package(chimaera_admin REQUIRED)        # Admin ChiMod (required for most ChiMods)
```

**ChiMod Creation Pattern:**
```cmake
# Use modern ChiMod build functions instead of manual add_library
add_chimod_runtime(
  CHIMOD_NAME core
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/autogen/core_lib_exec.cc
)

add_chimod_client(
  CHIMOD_NAME core
  SOURCES
    src/core_client.cc
    src/content_transfer_engine.cc
)
```

**Target Naming:**
- **Actual Targets**: `${NAMESPACE}_${CHIMOD_NAME}_runtime`, `${NAMESPACE}_${CHIMOD_NAME}_client`
- **CMake Aliases**: `${NAMESPACE}::${CHIMOD_NAME}_runtime`, `${NAMESPACE}::${CHIMOD_NAME}_client` (recommended)
- **Package Names**: `${NAMESPACE}_${CHIMOD_NAME}` (for external find_package)

## Worker Method Return Types

The following Worker methods return `void`, not `bool`:
- `ExecTask()` - Execute task with context switching capability
- `EndTask()` - End task execution and perform cleanup
- `RerouteDynamicTask()` - End dynamic scheduling task and re-route with updated pool query

These methods handle task execution flow internally and do not return success/failure status.

## Type Aliases

Use the `WorkQueue` typedef for worker queue types:
```cpp
using WorkQueue = chi::ipc::mpsc_queue<hipc::TypedPointer<TaskLane>>;
```

This simplifies code readability and maintenance for worker queue operations.

**TaskLane Typedef:**
The `TaskLane` typedef is defined globally in the `chi` namespace:
```cpp
using TaskLane = chi::ipc::multi_mpsc_queue<hipc::TypedPointer<Task>, TaskQueueHeader>::queue_t;
```

Use `TaskLane*` for all lane pointers in RunContext and other interfaces. Avoid `void*` and explicit type casts.

## ChiMod Client Requirements

### PoolQuery Recommendations for Create Operations

**RECOMMENDED**: Use `PoolQuery::Dynamic()` for all Create operations to leverage automatic caching optimization.

**Dynamic Pool Query Behavior:**
- Routes to the Monitor method with `MonitorModeId::kGlobalSchedule`
- Monitor performs a two-step process:
  1. Check if pool exists locally using PoolManager
  2. If pool exists: change pool_query to Local (task executes locally using existing pool)
  3. If pool doesn't exist: change pool_query to Broadcast (task creates pool on all nodes)
- This optimization avoids unnecessary network overhead when pools already exist locally

**Correct Usage:**
```cpp
// Recommended: Use Dynamic() for automatic caching
admin_client.Create(mctx, chi::PoolQuery::Dynamic(), "admin");
bdev_client.Create(mctx, chi::PoolQuery::Dynamic(), file_path, chimaera::bdev::BdevType::kFile);
```

### CreateTask Pool Assignment
CreateTask operations in all ChiMod clients MUST use `chi::kAdminPoolId` instead of the client's `pool_id_`. This is because CreateTask is actually a GetOrCreatePoolTask that must be processed by the admin ChiMod to create or find the target pool.

**Correct Usage:**
```cpp
auto task = ipc_manager->NewTask<CreateTask>(
    chi::CreateTaskNode(),
    chi::kAdminPoolId,  // Always use admin pool for CreateTask
    pool_query,
    // ... other parameters
);
```

### Container ID Assignment

When creating pools, **the container ID MUST be set to the node ID** of the physical node where the container is created. This ensures proper routing of DirectHash queries to physical nodes.

**Implementation:**
- `ModuleManager::CreateContainer()` takes a `container_id` parameter
- `PoolManager::CreatePool()` passes `ipc_manager->GetNodeId()` as the container ID
- The container's `container_id_` field is set to the physical node ID

This mapping allows DirectHash to correctly route tasks:
1. Hash value determines container ID: `container_id = hash % num_containers`
2. Container ID maps to physical node ID via the address table
3. Task completer reflects the physical node ID where it executed

### ChiMod Name Parameter
ChiMod clients MUST use `CreateParams::chimod_lib_name` instead of hardcoded module names in CreateTask operations.

### Pool Name Requirements
All ChiMod Create functions MUST require a user-provided `pool_name` parameter. Never auto-generate pool names using `pool_id_` during Create operations, as `pool_id_` is not set until after Create completes.

**Admin Pool Name Requirement:**
The admin pool name MUST always be "admin". Multiple admin pools are NOT supported.

## ChiMod Linking Requirements

### Target Naming and Aliases
ChiMod libraries use consistent underscore-based naming:

**Target Names:**
- Runtime: `${NAMESPACE}_${CHIMOD_NAME}_runtime` (e.g., `chimaera_admin_runtime`)
- Client: `${NAMESPACE}_${CHIMOD_NAME}_client` (e.g., `chimaera_admin_client`)

**CMake Aliases:**
- Runtime: `${NAMESPACE}::${CHIMOD_NAME}_runtime` (e.g., `chimaera::admin_runtime`)
- Client: `${NAMESPACE}::${CHIMOD_NAME}_client` (e.g., `chimaera::admin_client`)

**Package Names:**
- Format: `${NAMESPACE}_${CHIMOD_NAME}` (e.g., `chimaera_admin`)
- Used with `find_package(chimaera_admin REQUIRED)`
- Core package: `chimaera` (provides `chimaera::cxx`)

### Automatic Dependency Linking
ChiMod libraries automatically handle common dependencies:

**Automatic Dependencies for Runtime Code:**
- `rt` library: Automatically linked to all ChiMod runtime targets for POSIX real-time library support (async I/O)
- Admin ChiMod: Automatically linked to all non-admin ChiMod runtime and client targets
- Admin includes: Automatically added to include directories for non-admin ChiMods

**For External Applications:**
When linking against installed ChiMod libraries, use the underscore-based target pattern:
```cmake
target_link_libraries(your_target
  chimaera::mod_name_runtime    # Specific ChiMod runtime library
  chimaera::mod_name_client     # Specific ChiMod client library
)
# rt, admin, and chimaera::cxx dependencies are automatically included by ChiMod libraries
# All target names use underscores consistently
```

## ChiMod Runtime Code Standards

### Autogenerated Code Duplication
Runtime code (`*_runtime.cc` files) should **NEVER** duplicate autogenerated code methods. The following methods are automatically generated and must not be manually implemented in runtime source files:

**Prohibited duplicate implementations:**
- `SaveIn()` - Serialization from input parameters
- `LoadIn()` - Deserialization to input parameters
- `SaveOut()` - Serialization from output parameters
- `LoadOut()` - Deserialization to output parameters
- `NewCopy()` - Task copy constructor methods (container dispatcher, not task method)
- `Aggregate()` - Task aggregation dispatcher (container dispatcher, not task method)

## Locking and Synchronization

### CoMutex and CoRwLock

The chimaera runtime provides two simplified coroutine-aware synchronization primitives for runtime code:

**CoMutex (Coroutine Mutex)**
- **Header**: `chimaera/comutex.h`
- **Purpose**: Simplified mutex that uses Yield for blocking
- Uses a single `std::atomic<bool>` for lock state
- Tasks that cannot acquire the lock call `Yield()` to be placed in the blocked queue
- Tasks are retried automatically by the blocked queue mechanism
- No complex data structures (no vectors, maps, or lists)

**CoRwLock (Coroutine Reader-Writer Lock)**
- **Header**: `chimaera/corwlock.h`
- **Purpose**: Simplified reader-writer lock that uses Yield for blocking
- Uses `std::atomic<int>` for reader count and `std::atomic<bool>` for writer state
- Supports multiple concurrent readers or a single writer
- Tasks that cannot acquire the lock call `Yield()` to be placed in the blocked queue

### Task Wait Functionality

**Critical Fix for Infinite Loops**
The task `Wait()` function has been fixed to prevent infinite loops in the blocked task system. When a task calls `Wait()`, it automatically adds itself to the current task's `waiting_for_tasks` list in `RunContext`.

**Usage:**
```cpp
task->Wait();  // Automatically tracked in parent task's waiting list
task->Wait(false);  // Same as above - explicitly tracked
task->Wait(true);   // Called from yield - not tracked to avoid double tracking
```

## Code Quality Standards

### Compilation Standards
- All code must compile without warnings or errors
- Use appropriate variable types to avoid sign comparison warnings (e.g., `size_t` for container sizes)
- Mark unused variables with `(void)variable_name;` to suppress warnings when the variable is intentionally unused
- Follow strict type safety to prevent implicit conversions that generate warnings

### Thread Safety Standards

**Atomic Task Fields:**
Critical task fields that may be accessed from multiple threads should use atomic types for thread safety:

```cpp
std::atomic<u32> return_code_; /**< Task return code (0=success, non-zero=error) */
```

**Usage:**
- **Reading**: Use `task->GetReturnCode()` or `task->return_code_.load()`
- **Writing**: Use `task->SetReturnCode(value)` or `task->return_code_.store(value)`
- **Initialization**: Use `task->return_code_.store(0)` in constructors

## Unit Testing Standards

### Create Method Success Validation
**CRITICAL**: Always check if Create methods completed successfully in unit tests. Many test failures occur because Create operations fail but tests continue executing against invalid or uninitialized objects.

**Success Criteria**: Create methods succeed when the return code is 0.

**Required Pattern for All Unit Tests:**
```cpp
// After any Create operation in unit tests
ASSERT_EQ(client.GetReturnCode(), 0) << "Create operation failed with return code: " << client.GetReturnCode();

// Or for individual task-based creates
auto create_task = client.AsyncCreate(mctx, pool_query, pool_name, /* other params */);
create_task->Wait();
ASSERT_EQ(create_task->GetReturnCode(), 0) << "Create task failed with return code: " << create_task->GetReturnCode();
```

This requirement applies to ALL ChiMod Create operations in unit tests including admin, bdev, and any custom ChiMods.

## Device Configuration

### Directory Management
When configuring devices:
- Use `mkdir` to create each parent directory specified in the devices configuration during the configure phase
- Remove each directory during the clean phase to ensure proper cleanup
- This ensures device paths exist before use and are properly cleaned up after

## Docker Deployment

### Overview

The IOWarp Core runtime can be deployed using Docker containers for easy distributed deployment. The Docker setup includes:
- Dockerfile for building the runtime container
- docker-compose.yml for orchestrating multi-node clusters
- Entrypoint script for configuration generation
- Hostfile for cluster node management

### Quick Start

```bash
# Build and start 3-node cluster
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f iowarp-node1

# Stop cluster
docker-compose down
```

### Configuration Methods

**Method 1: Environment Variables (Recommended)**

Configure via environment variables in docker-compose.yml:
```yaml
environment:
  - CHI_SCHED_WORKERS=8
  - CHI_MAIN_SEGMENT_SIZE=1G
  - CHI_CLIENT_DATA_SEGMENT_SIZE=512M
  - CHI_RUNTIME_DATA_SEGMENT_SIZE=512M
  - CHI_ZMQ_PORT=5555
  - CHI_LOG_LEVEL=info
  - CHI_SHM_SIZE=2147483648
```

### Critical Requirements

**Shared Memory Size:**
**CRITICAL**: Set `shm_size` >= sum of all segment sizes:
```yaml
shm_size: 2gb  # Must be >= main + client_data + runtime_data segments
```

**Hostfile:**
Create hostfile with cluster node IPs (one per line):
```
172.20.0.10
172.20.0.11
172.20.0.12
```

Mount in docker-compose.yml:
```yaml
volumes:
  - ./hostfile:/etc/iowarp/hostfile:ro
environment:
  - CHI_HOSTFILE=/etc/iowarp/hostfile
```

## Documentation

### Contributing Guide
New contributors should start with the comprehensive guide at: `docs/contributing.md`

This guide covers:
- Development environment setup with DevContainers (Windows, macOS, Linux)
- VSCode configuration and recommended extensions
- Code style and formatting with clang-format
- Build system and testing procedures
- Git workflow and project structure

### CTE Core API Documentation
Complete API documentation and usage guide is available at: `context-transfer-engine/docs/cte/cte.md`

This documentation covers:
- Installation and linking instructions
- Complete API reference with examples
- Configuration guide
- Python bindings usage
- Advanced topics and troubleshooting

### External Integration Test
A standalone external integration test is available at: `context-transfer-engine/test/unit/external/`

This test demonstrates MODULE_DEVELOPMENT_GUIDE.md compliant patterns:
- Modern find_package() usage for ChiMod discovery
- Proper target linking with namespace::module_type aliases
- Automatic dependency resolution through ChiMod targets
- External application CMake configuration

## Cleanup Commands

### Remove Temporary CMake Files
To clean all temporary CMake files produced during build:
```bash
# Remove CMake cache and configuration files
find . -name "CMakeCache.txt" -delete
find . -name "cmake_install.cmake" -delete
find . -name "CTestTestfile.cmake" -delete

# Remove generated makefiles
find . -name "Makefile" -delete
find . -name "*.make" -delete

# Remove CMake build directories and files
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "_deps" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove CTest and CPack files
find . -name "DartConfiguration.tcl" -delete
find . -name "CPackConfig.cmake" -delete
find . -name "CPackSourceConfig.cmake" -delete

# Remove build directories
rm -rf build/
rm -rf out/
rm -rf cmake-build-*/

# Remove CMake temporary files
find . -name "*.cmake.in" -not -path "./CMakePresets.json" -delete 2>/dev/null || true
find . -name "CMakeDoxyfile.in" -delete 2>/dev/null || true
find . -name "CMakeDoxygenDefaults.cmake" -delete 2>/dev/null || true

# Remove any .ninja_* files if using Ninja generator
find . -name ".ninja_*" -delete 2>/dev/null || true
find . -name "build.ninja" -delete 2>/dev/null || true
find . -name "rules.ninja" -delete 2>/dev/null || true

# Remove Testing directory created by CTest
find . -name "Testing" -type d -exec rm -rf {} + 2>/dev/null || true

echo "CMake cleanup completed!"
```

## ChiMod Development

When creating or modifying ChiMods (Chimaera modules), refer to the comprehensive module development guide:

**📖 See [context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md](context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md) for complete ChiMod development documentation**

This guide covers:
- Module structure and architecture
- Task development patterns
- Client and runtime implementation
- Build system integration
- Configuration and code generation
- Synchronization primitives
- Execution modes and dynamic scheduling
- External ChiMod development
- Best practices and common pitfalls
