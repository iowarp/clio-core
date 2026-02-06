# IOWarp Core - Unified Development Guide

This repository contains the unified IOWarp Core framework, integrating multiple components:
- **context-transport-primitives** (formerly cte-hermes-shm): Core transport and shared memory primitives
- **runtime**: IOWarp runtime system
- **context-transfer-engine**: Context transfer engine
- **context-assimilation-engine**: Context assimilation engine
- **context-exploration-engine**: Context exploration engine

## Chimods

When building chimods, make sure to edit chimaera_mod.yaml and chimaera_repo.yaml.

If you add new methods to a chimod, please edit chimaera_mod.yaml and use the chi_refresh_repo binary to autogenerate the relevant autogen files.


## ⚠️ CRITICAL BUILD RULE ⚠️

**NEVER HARDCODE ABSOLUTE PATHS INSIDE THE CMAKES**

**NEVER BUILD OUTSIDE OF THE BUILD DIRECTORY. DO NOT PLACE BUILD FILES IN SOURCE DIRECTORIES. NEVER EVER EVER.**

### What This Means:

**NEVER do any of the following:**
- ❌ `rm -rf /workspace/build/*` followed by `cmake ..` (creates in-source build)
- ❌ Building third-party libraries in `/workspace` or any subdirectory
- ❌ Running cmake commands that create `CMakeFiles/`, `CMakeCache.txt`, or `Makefile` in source directories
- ❌ Creating `/tmp/build_*` directories for third-party dependencies and leaving them

**ALWAYS do this:**
- ✅ Build third-party libraries in a SEPARATE directory completely outside `/workspace` (e.g., `~/builds/libpressio`)
- ✅ Keep `/workspace/build` intact - use `cmake ..` from within it to reconfigure
- ✅ Clean build artifacts: `cd /workspace/build && make clean` (NOT `rm -rf /workspace/build/*`)
- ✅ Remove temporary build directories after installation: `rm -rf ~/builds/*`

**If you accidentally create build artifacts in source directories:**
1. Immediately stop
2. Remove ALL CMakeFiles, CMakeCache.txt, cmake_install.cmake, Makefile from source tree
3. Rebuild properly from `/workspace/build`

## Code Style

Keeep code simple. Do not allow functions to be more than 100 lines of code. Make helper functions logically.

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

### Dependency Management Strategy

**IMPORTANT**: IOWarp Core uses **conda for all major dependencies** to avoid library conflicts.

**Why Conda?**
- Prevents version conflicts between system and conda libraries (e.g., system HDF5 with conda libcurl)
- Ensures reproducible builds across different systems
- All dependencies are installed in the devcontainer: boost, hdf5, yaml-cpp, zeromq, cereal, catch2, poco, nlohmann_json, etc.
- No need to install system development packages via apt

**In the Devcontainer:**
- The conda base environment is auto-activated with all dependencies
- CMake will automatically find conda packages via `CMAKE_PREFIX_PATH`
- Do NOT install development libraries via apt (libboost-dev, libhdf5-dev, etc.) - use conda instead

**Outside the Devcontainer:**
- Option 1 (Recommended): Use conda to install dependencies: `conda install boost hdf5 yaml-cpp zeromq cereal catch2 poco nlohmann_json`
- Option 2: Run `install.sh` to build dependencies from source
- Option 3: Use system packages at your own risk (may cause library conflicts)

**ADIOS2 Exception:**
- ADIOS2 is built from source (not from conda) to ensure C++20 compatibility
- Uses ADIOS2 v2.11.0 which supports both x86_64 and ARM64 architectures
- Built with MPI, HDF5, and ZeroMQ support (using conda's HDF5 and ZeroMQ libraries)
- SST is disabled due to ARM64 Linux compatibility issues in the DILL library

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

### RPATH Configuration
The build system supports **absolute RPATHs** for source-only builds (enabled by default):
- **Enable/Disable**: Controlled by `WRP_CORE_ENABLE_RPATH` option (default: ON)
- **Linux**: Uses `${CMAKE_INSTALL_PREFIX}/lib` for runtime library search paths
- **macOS**: Uses `${CMAKE_INSTALL_PREFIX}/lib` for runtime library search paths
- Libraries are linked with absolute paths to the installation directory and all dependency locations
- This configuration is designed for building from source only, not for relocatable binary distributions
- **Disable RPATH**: Set `-DWRP_CORE_ENABLE_RPATH=OFF` if you prefer using `LD_LIBRARY_PATH`

### HSHM Usage

Always use HSHM_MCTX macro unless we are writing GPU code, which necessitates a specific mctx to be created.

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

## PoolManager Coroutine Methods

The following PoolManager methods are coroutines that return `TaskResume`:
- `CreatePool()` - Creates a pool and co_awaits the container's Create method
- `DestroyPool()` - Destroys a pool (coroutine for consistency)

**Why Coroutines:**
These methods are coroutines to properly handle nested pool creation. When a ChiMod's Create method (e.g., CTE Create) needs to create sub-pools (e.g., bdev for storage), it uses `co_await`. The coroutine chain allows proper suspension and resumption:
1. Admin's `GetOrCreatePool` co_awaits `PoolManager::CreatePool`
2. `PoolManager::CreatePool` co_awaits `container->Run()` (the Create method)
3. The Create method can co_await nested pool creations (e.g., bdev Create)
4. When nested operations complete, the chain resumes automatically

**Admin Runtime Methods:**
The following admin runtime methods are also coroutines:
- `GetOrCreatePool()` - co_awaits PoolManager::CreatePool
- `DestroyPool()` - co_awaits PoolManager::DestroyPool
- `Destroy()` - co_awaits DestroyPool (alias)

## Task Wait and Future Pattern

### Task::Wait() Signature and Usage

`Task::Wait()` takes an `is_complete` reference parameter from the Future object:

```cpp
void Task::Wait(std::atomic<u32>& is_complete, double block_time_us = 0.0);
```

**Key Points:**
- Task::Wait() checks the `is_complete` flag from Future, not internal task state
- The method yields execution in a loop until `is_complete` is set to 1
- In runtime mode, uses cooperative multitasking with blocked queue mechanism
- In client mode, busy-waits on the `is_complete` flag with yielding

**Correct Usage Pattern:**
```cpp
// From IpcManager::Recv()
template <typename TaskT> void Recv(Future<TaskT>& future) {
  auto& future_shm = future.GetFutureShm();
  TaskT* task_ptr = future.get();
  task_ptr->Wait(future_shm->is_complete_);  // Pass future's is_complete flag
  // ... deserialization logic ...
}
```

**User-Facing API:**
Users should call `Future::Wait()` instead of `Task::Wait()` directly:
```cpp
auto task = client.AsyncCreate(/* ... */);
task.Wait();  // Calls IpcManager::Recv() which calls Task::Wait()
```

### Task::Yield() for Cooperative Yielding

`Task::Yield()` is used for cooperative yielding during blocking operations (I/O, locks, etc.) without waiting for a specific completion condition:

```cpp
void Task::Yield(double block_time_us = 0.0);
```

**Usage Examples:**
- CoMutex lock contention: `task->Yield();`
- Async I/O polling: `task->Yield();`
- Admin flush waiting: `task->Yield(25);`

**Implementation:**
- Adds task to blocked queue with specified blocking duration
- Yields execution back to worker
- Does NOT track subtask completion (unlike Wait)

## Type Aliases

Use the `WorkQueue` typedef for worker queue types:
```cpp
using WorkQueue = chi::ipc::mpsc_ring_buffer<hipc::TypedPointer<TaskLane>>;
```

This simplifies code readability and maintenance for worker queue operations.

**TaskLane Typedef:**
The `TaskLane` typedef is defined globally in the `chi` namespace:
```cpp
using TaskLane = chi::ipc::multi_mpsc_ring_buffer<hipc::TypedPointer<Task>, TaskQueueHeader>::queue_t;
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

Use the unified `find_package(iowarp-core)` which automatically includes all components and ChiMods:

```cmake
# Single find_package call includes everything
find_package(iowarp-core REQUIRED)
# This automatically provides:
#   Core Components:
#     - All hshm::* modular targets (hshm::cxx, hshm::configure, hshm::serialize, etc.)
#     - chimaera::cxx (core runtime library)
#     - ChiMod build utilities (add_chimod_client, add_chimod_runtime, etc.)
#
#   Core ChiMods (Always Available):
#     - chimaera::admin_client, chimaera::admin_runtime
#     - chimaera::bdev_client, chimaera::bdev_runtime
#
#   Optional ChiMods (if enabled at build time):
#     - wrp_cte::core_client, wrp_cte::core_runtime (if WRP_CORE_ENABLE_CTE=ON)
#     - wrp_cae::core_client, wrp_cae::core_runtime (if WRP_CORE_ENABLE_CAE=ON)

# Then link to the ChiMod libraries you need
target_link_libraries(your_target
  chimaera::admin_client     # Admin ChiMod (always available)
  chimaera::bdev_client      # Block device ChiMod (always available)
  wrp_cte::core_client       # CTE ChiMod (if enabled)
  wrp_cae::core_client       # CAE ChiMod (if enabled)
)
# Dependencies are automatically included by ChiMod libraries
# No need to manually link hshm::cxx or chimaera::cxx
```

**Alternative (Manual):**
If you need finer control, you can still find packages individually:
```cmake
find_package(HermesShm REQUIRED)        # Provides hshm::* targets
find_package(chimaera REQUIRED)         # Provides chimaera::cxx
find_package(chimaera_admin REQUIRED)   # Provides admin ChiMod
find_package(chimaera_bdev REQUIRED)    # Provides bdev ChiMod
find_package(wrp_cte_core REQUIRED)     # Provides CTE ChiMod (if enabled)
find_package(wrp_cae_core REQUIRED)     # Provides CAE ChiMod (if enabled)
```

### HSHM Modular Dependency Targets

HSHM (HermesShm/context-transport-primitives) provides modular INTERFACE library targets for optional dependencies. Each target includes only the specific dependency it represents, along with the associated compile definitions.

**Available Modular Targets:**

- **`hshm::cxx`** - Core HSHM library
  - Provides: Basic shared memory and data structures
  - Links to: `configure`, `thread_all`
  - Always required by all HSHM users

- **`hshm::configure`** - Configuration parsing (yaml-cpp)
  - Provides: YAML configuration file parsing
  - Use instead of linking to yaml-cpp directly
  - Compile definitions: None (yaml-cpp is always enabled)

- **`hshm::serialize`** - Serialization (cereal)
  - Provides: Object serialization/deserialization
  - Use instead of linking to cereal directly
  - Compile definitions: `HSHM_ENABLE_CEREAL`

- **`hshm::interceptor`** - ELF interception
  - Provides: Dynamic library interception support
  - Required for: Adapter real API functionality
  - Compile definitions: `HSHM_ENABLE_ELF`

- **`hshm::lightbeam`** - Network transport (ZeroMQ, libfabric, Thallium)
  - Provides: High-performance network communication
  - Used by: Chimaera runtime for distributed operations
  - Compile definitions: `HSHM_ENABLE_ZMQ`, `HSHM_ENABLE_LIBFABRIC`, `HSHM_ENABLE_THALLIUM`

- **`hshm::thread_all`** - Threading support
  - Provides: pthread, OpenMP support
  - Includes: Thread model definitions
  - Compile definitions: `HSHM_ENABLE_OPENMP`, `HSHM_ENABLE_PTHREADS`, `HSHM_ENABLE_WINDOWS_THREADS`, `HSHM_DEFAULT_THREAD_MODEL`, `HSHM_DEFAULT_THREAD_MODEL_GPU`

- **`hshm::mpi`** - MPI support
  - Provides: Message Passing Interface
  - Use only where MPI is actually needed
  - Compile definitions: `HSHM_ENABLE_MPI`

- **`hshm::compress`** - Compression libraries
  - Provides: Data compression support
  - Compile definitions: `HSHM_ENABLE_COMPRESS`

- **`hshm::encrypt`** - Encryption libraries
  - Provides: Data encryption support
  - Compile definitions: `HSHM_ENABLE_ENCRYPT`

- **`hshm::cuda_cxx`** - CUDA GPU support
  - Provides: CUDA-enabled HSHM library for GPU code
  - Use for: CUDA kernel code and GPU device functions
  - Compile definitions: `HSHM_ENABLE_CUDA=1`, `HSHM_ENABLE_ROCM=0`
  - Note: Only available when `HSHM_ENABLE_CUDA=ON` at build time

- **`hshm::rocm_cxx`** - ROCm GPU support
  - Provides: ROCm-enabled HSHM library for GPU code
  - Use for: HIP kernel code and GPU device functions
  - Compile definitions: `HSHM_ENABLE_ROCM=1`, `HSHM_ENABLE_CUDA=0`
  - Note: Only available when `HSHM_ENABLE_ROCM=ON` at build time

**Linking Guidelines:**

1. **Never link to yaml-cpp directly** - Use `hshm::configure` instead (except within hshm::configure itself)
2. **Never link to cereal directly** - Use `hshm::serialize` instead
3. **Be selective** - Only link to the modular targets you actually need
4. **ChiMod clients** - Should only link to `hshm::cxx` (automatically included)
5. **ChiMod runtimes** - May link to additional modular targets as needed
6. **Tests** - Link only to the specific modular targets they test
7. **GPU code** - Use `hshm::cuda_cxx` or `hshm::rocm_cxx` for GPU kernel code; use `hshm::cxx` for host code

**Example Usage:**
```cmake
# External application needing configuration and serialization
target_link_libraries(my_app
  wrp_cte::core_client      # Provides hshm::cxx automatically
  hshm::configure           # For YAML config parsing
  hshm::serialize           # For object serialization
)

# Adapter needing ELF interception
target_link_libraries(my_adapter
  hshm::cxx
  hshm::interceptor         # For real API functionality
)

# Test needing MPI
target_link_libraries(my_test
  hshm::cxx
  hshm::mpi                 # Only link MPI where needed
)

# GPU application using CUDA
target_link_libraries(my_cuda_kernel
  hshm::cuda_cxx            # For GPU kernel code
)
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

### Test Framework Requirements

**CRITICAL**: Unit tests that initialize the Chimaera runtime MUST use the `simple_test.h` framework. **DO NOT use Catch2** with Chimaera runtime initialization.

**Catch2 Incompatibility:**
- Catch2's test framework causes segmentation faults when used with Chimaera runtime initialization
- This issue was confirmed by copying working test code from `test_bdev_chimod.cc` (which uses simple_test.h) to a Catch2-based test - the identical code segfaulted with Catch2 but worked with simple_test.h
- Root cause: Catch2's test runner infrastructure conflicts with Chimaera's runtime initialization

**Required Test Framework:**
- Use `#include "../../../context-runtime/test/simple_test.h"` instead of Catch2
- Available macros: `TEST_CASE`, `SECTION`, `REQUIRE`, `REQUIRE_FALSE`, `REQUIRE_NOTHROW`, `INFO`, `FAIL`
- Use `SIMPLE_TEST_MAIN()` at the end of your test file
- Note: simple_test.h does NOT provide `CHECK` macro - use `REQUIRE` instead

**Example simple_test.h Test:**
```cpp
#include "../../../context-runtime/test/simple_test.h"

TEST_CASE("My Test", "[mytag]") {
  // Test code here
  REQUIRE(some_condition);
}

SIMPLE_TEST_MAIN()
```

### Chimaera Initialization in Unit Tests

**CRITICAL**: All unit tests MUST use the unified `CHIMAERA_INIT()` function. Do NOT use deprecated initialization functions or direct calls to `CHIMAERA_RUNTIME_INIT()` or `CHIMAERA_CLIENT_INIT()`.

**Required Pattern for All Unit Tests:**
```cpp
// At the beginning of your test or test fixture setup
bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
REQUIRE(success);

// Optional: Wait for initialization to complete
std::this_thread::sleep_for(std::chrono::milliseconds(500));

// Verify core managers are available
REQUIRE(CHI_IPC != nullptr);
REQUIRE(CHI_IPC->IsInitialized());
```

**Initialization Parameters:**
- **Mode**: Always use `chi::ChimaeraMode::kClient` for unit tests
- **default_with_runtime**: Always use `true` for unit tests (starts runtime automatically)
- **Environment Variable**: `CHIMAERA_WITH_RUNTIME` is handled automatically by `CHIMAERA_INIT()`
  - If set to `1`: Runtime will be started
  - If set to `0`: Only client initialization (useful for external runtime scenarios)
  - If not set: Uses the `default_with_runtime` parameter value

**DEPRECATED - Do NOT Use:**
- `initializeBoth()` - Remove from all test fixtures
- `initializeRuntime()` - Remove from all test fixtures
- `initializeClient()` - Remove from all test fixtures
- `chi::CHIMAERA_RUNTIME_INIT()` - Do not call directly in tests
- `chi::CHIMAERA_CLIENT_INIT()` - Do not call directly in tests

**Example Test Fixture:**
```cpp
class MyTestFixture {
public:
  MyTestFixture() {
    // Initialize Chimaera with client mode and runtime
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);

    // Give runtime time to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Verify initialization
    REQUIRE(CHI_IPC != nullptr);
    REQUIRE(CHI_POOL_MANAGER != nullptr);
  }

  ~MyTestFixture() {
    // Cleanup handled automatically
  }
};
```

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

## Python Wheel Distribution

### Building Bundled Wheels

IOWarp Core can be packaged as a self-contained Python wheel that includes all dependencies installed by `install.sh`.

**Quick Build:**
```bash
# Build a bundled wheel with all dependencies
export IOWARP_BUNDLE_BINARIES=ON
python -m build --wheel

# Or use the convenience script
./build_wheel.sh
```

**What Gets Bundled:**
- All IOWarp libraries (libchimaera_cxx.so, libhermes_shm_host.so, ChiMod libraries)
- Dependencies from install.sh (Boost, HDF5, ZeroMQ, yaml-cpp, etc.)
- Command-line tools (wrp_cte, wrp_cae_omni, chimaera_start_runtime, etc.)
- Headers and CMake configuration files
- Conda dependencies (if building in a Conda environment)

**RPATH Configuration:**
- All bundled libraries use relative RPATH (`$ORIGIN`)
- The wheel is fully relocatable and works anywhere it's installed
- No `LD_LIBRARY_PATH` configuration needed

**Complete Documentation:** See `BUILD_WHEEL.md` for:
- Detailed build instructions
- Platform-specific wheels (manylinux, macOS)
- CI/CD integration examples
- Troubleshooting guide
- PyPI distribution

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
