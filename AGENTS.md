# IOWarp Core - Unified Development Guide

This repository contains the unified IOWarp Core framework, integrating multiple components:
- **context-transport-primitives** (formerly context-transport-primitives): Core transport and shared memory primitives
- **runtime**: IOWarp runtime system
- **context-transfer-engine**: Context transfer engine
- **context-assimilation-engine**: Context assimilation engine
- **context-exploration-engine**: Context exploration engine

## Documentation Updates
Whenever you modify the configurations for context-runtime, context-transfer-engine, context-assimilation-engine, or bdev, we should update our documentation accordingly.
First, we should update context-runtime/config/clio_default.yaml to have the default parameters -- even just as comments. We should document the parameter options here as well.

After this, we need to update the following doc:
docs/docs/deployment/configuration.md

## Testing Updates

Never ever re-run tasks without installing your chanages first.
You need sudo if you are in the container.
We use rpaths for libraries. This stuff does not get overriden by LD_LIBRARY_PATH.

## Chimods

When building chimods, make sure to edit clio_mod.yaml and clio_repo.yaml.

If you add new methods to a chimod, please edit clio_mod.yaml and use the clio_run repo refresh binary to autogenerate the relevant autogen files.


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

## GPU Producer-Only Model

The GPU side of CLIO Runtime is a **pure task producer** — kernels do not allocate
tasks, FutureShm, or data buffers. All allocations happen on the host before
kernel launch into client-owned device-memory backends that are registered
with the runtime via `admin::RegisterMemoryTask`. Inside a kernel the only
operation `clio::run::gpu::IpcManager` exposes is `Send` — pack a pre-allocated
task and push it onto the per-device gpu2cpu_queue.

### Lifecycle (host)
1. Runtime init: `gpu::IpcManager::ServerInitGpuQueues` enumerates GPUs and
   allocates one pinned-host gpu2cpu_queue per device. The CPU GPU worker
   polls every queue.
2. Client backend allocation:
   ```cpp
   char *base = nullptr;
   auto alloc_id = ipc->AllocateAndRegisterGpuBackend(
       gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, bytes, &base);
   ```
   Available kinds: `kPinnedHost` (pinned host, fastest), `kManagedUvm`
   (CUDA managed / SYCL shared), `kDeviceMem` (device-only; worker copies
   POD bytes via cudaMemcpy on each pop). First-cut implementation only
   wires `kPinnedHost` end-to-end; the others register correctly but the
   worker pop path logs a warning for `kDeviceMem`.
3. Pre-construct task + FutureShm pairs in the registered backend with
   placement new. Tasks are POD with identical layout on CPU and GPU.

### Lifecycle (kernel)
```cpp
__global__ void MyKernel(IpcManagerGpuInfo info,
                         ctp::ipc::FullPtr<MyTaskT> task) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x == 0) {
    // Mutate POD task fields. No NewTask, no AllocateBuffer.
    task->some_input_ = ...;
    auto fut = CLIO_IPC->Send(task);
    fut.Wait();
    // Read result fields back from the same POD task.
  }
}
```
SYCL kernels get a kernel-scope IpcManager pointer; CUDA/ROCm kernels use
the per-block `__shared__` IpcManager via `GetBlockIpcManager()`. The
single `CLIO_GPU_INIT(gpu_info, ipc_ptr)` macro covers both backends.

### Worker pop path
The CPU GPU worker (`Worker::ProcessNewTaskGpu`) pops a `gpu::Future<Task>`
off `gpu2cpu_queue`, resolves both the task ShmPtr and the FutureShm
ShmPtr via `gpu::IpcManager::FindClientBackend`, dispatches the chimod
method on the local CPU runtime, and signals `FUTURE_COMPLETE` on the
device-side gpu::FutureShm so the kernel poll-loop unblocks.

### Forbidden on the GPU
- `CLIO_IPC->NewTask(...)`, `CLIO_IPC->NewObj(...)`, `CLIO_IPC->AllocateBuffer(...)`
  — these are host-only.
- `PoolQuery::ToLocalGpu(...)`, `PoolQuery::LocalGpuBcast()` — both removed.
  Use `PoolQuery::ToLocalCpu()` exclusively for kernel→runtime submission.
- `IpcCpu2Gpu`, `IpcGpu2Gpu` — both deleted with the GPU runtime concept.

## Code Style

Keeep code simple. Do not allow functions to be more than 100 lines of code. Make helper functions logically.

Use the Google C++ style guide for C++.

You should store the pointer returned by the singleton GetInstance method. Avoid dereferencing GetInstance method directly using either -> or *. E.g., do not do ``ctp::Singleton<T>::GetInstance()->var_``. You should do ``auto *x = ctp::Singleton<T>::GetInstance(); x->var_;``.

Whenever you build a new function, always create a docstring for it. It should document what the parameters mean and the point of the function. It should be something easily parsed by doxygen.

### GPU Compiler Macro Rule

**NEVER** use raw GPU compiler-detection macros (`__CUDACC__`, `__HIPCC__`, `__HIP__`, `__CUDA_ARCH__`, `__HIP_DEVICE_COMPILE__`) anywhere except `context-transport-primitives/include/clio_ctp/constants/macros.h`. We do not want code paths to compile just because a GPU compiler is being used — we need the explicit CMake build flags (`CTP_ENABLE_CUDA`, `CTP_ENABLE_ROCM`) to be set as well.

**Use these macros instead:**
- `CTP_IS_GPU` — true when compiling device code (replaces `__CUDA_ARCH__` / `__HIP_DEVICE_COMPILE__`)
- `CTP_IS_HOST` — true when compiling host code
- `CTP_IS_GPU_COMPILER` — true when compiled by nvcc or hipcc AND the build flag is set (replaces `__CUDACC__` / `__HIPCC__`)
- `CTP_IS_CUDA_COMPILER` — CUDA-specific compiler check (replaces `CTP_ENABLE_CUDA && __CUDACC__`)
- `CTP_IS_ROCM_COMPILER` — ROCm-specific compiler check (replaces `CTP_ENABLE_ROCM && __HIPCC__`)

## File Headers

All C/C++ source files (.h, .hpp, .cc, .cpp) MUST include the IOWarp BSD 3-Clause license header at the top of the file.

**Header Template:**
```c
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
```

**When creating new files:**
- Include the full BSD 3-Clause header at the top
- Use the template from `.header_template` or copy from existing files
- Place the header before any `#include` statements

**Updating existing files:**
Use the automated script:
```bash
./update_headers.sh --dry-run    # Preview changes
./update_headers.sh              # Apply changes
```

**Files to exclude:**
- Autogenerated files (matching `*autogen*`, `*_generated.*`, `*.pb.*`)
- Third-party code in `third_party/`, `external/`, `_deps/`
- Build artifacts

See `HEADER_UPDATE_GUIDE.md` for complete documentation.

NEVER use a null pool query. If you don't know, always use local.

Local QueueId should be named. NEVER use raw integers. This is the same for priorities. Please name them semantically.

All timing prints MUST include units of measurement in milliseconds (ms). Always print timing values in the order of milliseconds.

## Shared Memory Cleanup

### Automatic IPC Cleanup on RuntimeInit

When CLIO Runtime RuntimeInit is called (via `IpcManager::ServerInit()`), it automatically cleans up leftover shared memory segments from previous runs or crashed processes by calling `IpcManager::ClearUserIpcs()`.

**ClearUserIpcs() Behavior:**
- Scans the per-user clio_run directory (`SystemInfo::GetMemfdDir()`, i.e. `/tmp/clio_$USER/`)
- Removes all memfd symlinks and IPC socket files from that directory
- Silently ignores permission errors (EACCES, EPERM) to support multi-user systems
- Other users' active CLIO Runtime processes are not affected
- Logs successfully removed segments at kDebug level
- Returns the count of segments removed

This ensures a clean state for each runtime initialization without requiring manual cleanup scripts.

### IPC Path Convention

**CRITICAL**: Never hardcode `/tmp/clio_*` paths in IpcManager or elsewhere. Always use the `SystemInfo` helpers:
- `ctp::SystemInfo::GetMemfdDir()` — returns `/tmp/clio_$USER`
- `ctp::SystemInfo::GetMemfdPath(name)` — returns `/tmp/clio_$USER/<name>` (strips leading `/`)
- `ctp::SystemInfo::EnsureMemfdDir()` — creates the directory if it doesn't exist

For IPC Unix domain socket paths, use: `ctp::SystemInfo::GetMemfdPath("clio_" + std::to_string(port) + ".ipc")`

## Workflow

Use the incremental logic builder agent when making code changes.

Use the compiler subagent for making changes to cmakes and identifying places that need to be fixed in the code.

Always verify that code continue to compiles after making changes. Avoid commenting out code to fix compilation issues.

Whenever building unit tests, make sure to use the unit testing agent.

Whenever performing filesystem queries or executing programs, use the filesystem ops script agent.

NEVER DO MOCK CODE OR STUB CODE UNLESS SPECIFICALLY STATED OTHERWISE. ALWAYS IMPLEMENT REAL, WORKING CODE.

## Build Configuration

### Dependency Management Strategy

**IMPORTANT**: The Docker devcontainer uses **apt + source builds** for all dependencies. Conda is reserved for CI-only installer paths (`installers/conda/`).

**In the Devcontainer:**
- All dependencies are installed via apt or built from source in `deps-cpu.Dockerfile`
- A Python virtual environment is auto-activated at `/home/iowarp/venv`
- CMake finds packages via standard `/usr/local` and `/usr` paths
- Do NOT use conda in the devcontainer — it is not configured there

**Outside the Devcontainer:**
- Option 1 (Recommended): Run `install.sh` to build dependencies from source
- Option 2: Use system packages via apt
- Option 3: Use conda via `installers/conda/` recipes

**ADIOS2:**
- ADIOS2 is built from source to ensure C++20 compatibility
- Uses ADIOS2 v2.11.0 which supports both x86_64 and ARM64 architectures
- Built with MPI, HDF5, and ZeroMQ support
- SST is disabled due to ARM64 Linux compatibility issues in the DILL library

### Component Build Options
The unified IOWarp Core build system provides options to enable/disable components:
- `CLIO_CORE_ENABLE_RUNTIME`: Enable runtime component (default: ON)
- `CLIO_CORE_ENABLE_CTE`: Enable context-transfer-engine component (default: ON)
- `CLIO_CORE_ENABLE_CAE`: Enable context-assimilation-engine component (default: ON)
- `CLIO_CORE_ENABLE_CEE`: Enable context-exploration-engine component (default: ON)

Example:
```bash
cmake --preset=debug -DWRP_CORE_ENABLE_CTE=ON -DWRP_CORE_ENABLE_CAE=OFF
```

### Compilation Standards
- Always use the debug CMakePreset when compiling code in this repo
- Never hardcode paths in CMakeLists.txt files
- Use find_package() for all dependencies
- Follow Module build patterns from MODULE_DEVELOPMENT_GUIDE.md
- All compilation warnings have been resolved as of the current state

### RPATH Configuration
The build system uses **relative RPATHs** (`$ORIGIN`) for portable, relocatable binaries (enabled by default):
- **Enable/Disable**: Controlled by `CLIO_CORE_ENABLE_RPATH` option (default: ON)
- **Linux**: Uses `$ORIGIN` and `$ORIGIN/../lib` so libraries and binaries find siblings at runtime
- **macOS**: Uses `@loader_path` and `@loader_path/../lib` (equivalent to `$ORIGIN`)
- Works for all deployment targets: system installs, pip wheels, conda packages, and relocatable builds
- **Disable RPATH**: Set `-DWRP_CORE_ENABLE_RPATH=OFF` if you prefer using `LD_LIBRARY_PATH`

### CTP Usage

Always use CTP_MCTX macro unless we are writing GPU code, which necessitates a specific mctx to be created.

### Module Build Patterns

This project follows the CLIO Runtime MODULE_DEVELOPMENT_GUIDE.md patterns for proper Module development:

**Required Packages for Module Development:**
```cmake
# Core Clio framework (includes ClioCoreCommon.cmake functions)
find_package(clio_run REQUIRED)              # Core library (clio::run::cxx)
find_package(clio_admin REQUIRED)        # Admin Module (required for most Modules)
```

**Module Creation Pattern:**
```cmake
# Use modern Module build functions instead of manual add_library
# Module name and namespace come from clio_mod.yaml in the source dir.
add_clio_module_runtime(
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/autogen/core_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/core_client.cc
    src/content_transfer_engine.cc
)
```

**Target Naming:**
- **Actual Targets**: `${PACKAGE_NAME}_${MODULE_NAME}_runtime`, `${PACKAGE_NAME}_${MODULE_NAME}_client` (or `${LIB_NAME}_runtime`/`_client` if `LIB_NAME` is passed to override — used e.g. by the bdev module which installs as `clio_bdev_*`).  `PACKAGE_NAME` is the filesystem-safe form of `NAMESPACE` (e.g. `clio::run` -> `clio_run`).
- **CMake Aliases**: `${NAMESPACE}::${MODULE_NAME}_runtime`, `${NAMESPACE}::${MODULE_NAME}_client` (recommended — e.g. `clio::run::admin_client`, `clio::cte::core_client`)
- **Legacy Aliases**: For clio_run-renamed modules (admin / bdev / MOD_NAME, now under `clio::run::`), the install layout still exposes `clio::run::<module>_<x>` aliases so external consumers (e.g. coeus-adapter) keep working.  The pre-`::` waypoint spellings (`clio_run::`, `clio_cte::`, `clio_cae::`) also resolve as forwarders.
- **Package Names**: `${PACKAGE_NAME}_${MODULE_NAME}` for clio::cte / clio::cae; pinned to `clio_${MODULE_NAME}` for clio::run-namespace modules to keep `find_package(clio_run_admin/_bdev/_MOD_NAME)` backward compat.

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
These methods are coroutines to properly handle nested pool creation. When a Module's Create method (e.g., CTE Create) needs to create sub-pools (e.g., bdev for storage), it uses `co_await`. The coroutine chain allows proper suspension and resumption:
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
using WorkQueue = clio::run::ipc::mpsc_ring_buffer<ctp::ipc::TypedPointer<TaskLane>>;
```

This simplifies code readability and maintenance for worker queue operations.

**TaskLane Typedef:**
The `TaskLane` typedef is defined globally in the `chi` namespace:
```cpp
using TaskLane = clio::run::ipc::multi_mpsc_ring_buffer<ctp::ipc::TypedPointer<Task>, TaskQueueHeader>::queue_t;
```

Use `TaskLane*` for all lane pointers in RunContext and other interfaces. Avoid `void*` and explicit type casts.

## Module Client Requirements

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
admin_client.Create(mctx, clio::run::PoolQuery::Dynamic(), "admin");
bdev_client.Create(mctx, clio::run::PoolQuery::Dynamic(), file_path, clio_run::bdev::BdevType::kFile);
```

### CreateTask Pool Assignment
CreateTask operations in all Module clients MUST use `clio::run::kAdminPoolId` instead of the client's `pool_id_`. This is because CreateTask is actually a GetOrCreatePoolTask that must be processed by the admin Module to create or find the target pool.

**Correct Usage:**
```cpp
auto task = ipc_manager->NewTask<CreateTask>(
    clio::run::CreateTaskNode(),
    clio::run::kAdminPoolId,  // Always use admin pool for CreateTask
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

### Module Name Parameter
Module clients MUST use `CreateParams::chimod_lib_name` instead of hardcoded module names in CreateTask operations.

### Pool Name Requirements
All Module Create functions MUST require a user-provided `pool_name` parameter. Never auto-generate pool names using `pool_id_` during Create operations, as `pool_id_` is not set until after Create completes.

**Admin Pool Name Requirement:**
The admin pool name MUST always be "admin". Multiple admin pools are NOT supported.

## Module Linking Requirements

### Target Naming and Aliases
Module libraries use consistent underscore-based naming:

**Target Names:**
- Runtime: `${NAMESPACE}_${MODULE_NAME}_runtime` (e.g., `clio_admin_runtime`)
- Client: `${NAMESPACE}_${MODULE_NAME}_client` (e.g., `clio_admin_client`)

**CMake Aliases:**
- Runtime: `${NAMESPACE}::${MODULE_NAME}_runtime` (e.g., `clio::run::admin_runtime`)
- Client: `${NAMESPACE}::${MODULE_NAME}_client` (e.g., `clio::run::admin_client`)

**Package Names:**
- Format: `${NAMESPACE}_${MODULE_NAME}` (e.g., `clio_run_admin`)
- Used with `find_package(clio_admin REQUIRED)`
- Core package: `clio_run` (provides `clio::run::cxx`)

### Automatic Dependency Linking
Module libraries automatically handle common dependencies:

**Automatic Dependencies for Runtime Code:**
- `rt` library: Automatically linked to all Module runtime targets for POSIX real-time library support (async I/O)
- Admin Module: Automatically linked to all non-admin Module runtime and client targets
- Admin includes: Automatically added to include directories for non-admin Modules

**For External Applications:**

Use the unified `find_package(clio-core)` which automatically includes all components and Modules:

```cmake
# Single find_package call includes everything
find_package(clio-core REQUIRED)
# This automatically provides:
#   Core Components:
#     - All ctp::* modular targets (ctp::cxx, ctp::configure, ctp::serialize, etc.)
#     - clio::run::cxx (core runtime library)
#     - Module build utilities (add_clio_module_client, add_clio_module_runtime, etc.)
#
#   Core Modules (Always Available):
#     - clio::run::admin_client, clio::run::admin_runtime
#     - clio::run::bdev_client, clio::run::bdev_runtime
#
#   Optional Modules (if enabled at build time):
#     - clio_cte::core_client, clio_cte::core_runtime (if CLIO_CORE_ENABLE_CTE=ON)
#     - clio_cae::core_client, clio_cae::core_runtime (if CLIO_CORE_ENABLE_CAE=ON)

# Then link to the Module libraries you need
target_link_libraries(your_target
  clio::run::admin_client     # Admin Module (always available)
  clio::run::bdev_client      # Block device Module (always available)
  clio_cte::core_client       # CTE Module (if enabled)
  clio_cae::core_client       # CAE Module (if enabled)
)
# Dependencies are automatically included by Module libraries
# No need to manually link ctp::cxx or clio::run::cxx
```

**Alternative (Manual):**
If you need finer control, you can still find packages individually:
```cmake
find_package(ClioCtp REQUIRED)        # Provides ctp::* targets
find_package(clio_run REQUIRED)         # Provides clio::run::cxx
find_package(clio_admin REQUIRED)   # Provides admin Module
find_package(clio_run_bdev REQUIRED)    # Provides bdev Module (library now: clio_bdev_*)
find_package(clio_cte_core REQUIRED)     # Provides CTE Module (if enabled)
find_package(clio_cae_core REQUIRED)     # Provides CAE Module (if enabled)
```

### CTP Modular Dependency Targets

CTP (ClioCtp/context-transport-primitives) provides modular INTERFACE library targets for optional dependencies. Each target includes only the specific dependency it represents, along with the associated compile definitions.

**Available Modular Targets:**

- **`ctp::cxx`** - Core CTP library
  - Provides: Basic shared memory and data structures
  - Links to: `configure`, `thread_all`
  - Always required by all CTP users

- **`ctp::configure`** - Configuration parsing (yaml-cpp)
  - Provides: YAML configuration file parsing
  - Use instead of linking to yaml-cpp directly
  - Compile definitions: None (yaml-cpp is always enabled)

- **`ctp::serialize`** - Serialization (cereal)
  - Provides: Object serialization/deserialization
  - Use instead of linking to cereal directly
  - Compile definitions: `CTP_ENABLE_CEREAL`

- **`ctp::interceptor`** - ELF interception
  - Provides: Dynamic library interception support
  - Required for: Adapter real API functionality
  - Compile definitions: `CTP_ENABLE_ELF`

- **`ctp::lightbeam`** - Network transport (ZeroMQ, libfabric, Thallium)
  - Provides: High-performance network communication
  - Used by: Clio runtime for distributed operations
  - Compile definitions: `CTP_ENABLE_ZMQ`, `CTP_ENABLE_LIBFABRIC`, `CTP_ENABLE_THALLIUM`

- **`ctp::thread_all`** - Threading support
  - Provides: pthread, OpenMP support
  - Includes: Thread model definitions
  - Compile definitions: `CTP_ENABLE_OPENMP`, `CTP_ENABLE_PTHREADS`, `CTP_ENABLE_WINDOWS_THREADS`, `CTP_DEFAULT_THREAD_MODEL`, `CTP_DEFAULT_THREAD_MODEL_GPU`

- **`ctp::mpi`** - MPI support
  - Provides: Message Passing Interface
  - Use only where MPI is actually needed
  - Compile definitions: `CTP_ENABLE_MPI`

- **`ctp::compress`** - Compression libraries
  - Provides: Data compression support
  - Compile definitions: `CTP_ENABLE_COMPRESS`

- **`ctp::encrypt`** - Encryption libraries
  - Provides: Data encryption support
  - Compile definitions: `CTP_ENABLE_ENCRYPT`

- **`ctp::cuda_cxx`** - CUDA GPU support
  - Provides: CUDA-enabled CTP library for GPU code
  - Use for: CUDA kernel code and GPU device functions
  - Compile definitions: `CTP_ENABLE_CUDA=1`, `CTP_ENABLE_ROCM=0`
  - Note: Only available when `CTP_ENABLE_CUDA=ON` at build time

- **`ctp::rocm_cxx`** - ROCm GPU support
  - Provides: ROCm-enabled CTP library for GPU code
  - Use for: HIP kernel code and GPU device functions
  - Compile definitions: `CTP_ENABLE_ROCM=1`, `CTP_ENABLE_CUDA=0`
  - Note: Only available when `CTP_ENABLE_ROCM=ON` at build time

- **`ctp::nixl`** - NIXL (NVIDIA Inference Xfer Library) transport
  - Provides: NIXL-backed data movement (DRAM→FILE via POSIX, DRAM→DRAM via memcpy)
  - Use for: High-performance CPU→storage transfers and future GPU→storage (GDS)
  - Compile definitions: `CTP_ENABLE_NIXL=1`
  - Build option: `CLIO_CORE_ENABLE_NIXL=ON`
  - Note: Requires NIXL installed at `/usr/local` (built with POSIX backend)

- **`ctp::nvshmem`** - NVSHMEM GPU-to-GPU communication
  - Provides: NVSHMEM compile definitions for GPU peer-to-peer communication
  - Compile definitions: `CTP_ENABLE_NVSHMEM=1`
  - Build option: `CLIO_CORE_ENABLE_NVSHMEM=ON`
  - Note: Requires NVSHMEM from NVIDIA developer portal

**Linking Guidelines:**

1. **Never link to yaml-cpp directly** - Use `ctp::configure` instead (except within ctp::configure itself)
2. **Never link to cereal directly** - Use `ctp::serialize` instead
3. **Be selective** - Only link to the modular targets you actually need
4. **Module clients** - Should only link to `ctp::cxx` (automatically included)
5. **Module runtimes** - May link to additional modular targets as needed
6. **Tests** - Link only to the specific modular targets they test
7. **GPU code** - Use `ctp::cuda_cxx` or `ctp::rocm_cxx` for GPU kernel code; use `ctp::cxx` for host code

**Example Usage:**
```cmake
# External application needing configuration and serialization
target_link_libraries(my_app
  clio_cte::core_client      # Provides ctp::cxx automatically
  ctp::configure           # For YAML config parsing
  ctp::serialize           # For object serialization
)

# Adapter needing ELF interception
target_link_libraries(my_adapter
  ctp::cxx
  ctp::interceptor         # For real API functionality
)

# Test needing MPI
target_link_libraries(my_test
  ctp::cxx
  ctp::mpi                 # Only link MPI where needed
)

# GPU application using CUDA
target_link_libraries(my_cuda_kernel
  ctp::cuda_cxx            # For GPU kernel code
)
```

## Module Runtime Code Standards

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

The clio_run runtime provides two simplified coroutine-aware synchronization primitives for runtime code:

**CoMutex (Coroutine Mutex)**
- **Header**: `clio_runtime/comutex.h`
- **Purpose**: Simplified mutex that uses Yield for blocking
- Uses a single `std::atomic<bool>` for lock state
- Tasks that cannot acquire the lock call `Yield()` to be placed in the blocked queue
- Tasks are retried automatically by the blocked queue mechanism
- No complex data structures (no vectors, maps, or lists)

**CoRwLock (Coroutine Reader-Writer Lock)**
- **Header**: `clio_runtime/corwlock.h`
- **Purpose**: Simplified reader-writer lock that uses Yield for blocking
- Uses `std::atomic<int>` for reader count and `std::atomic<bool>` for writer state
- Supports multiple concurrent readers or a single writer
- Tasks that cannot acquire the lock call `Yield()` to be placed in the blocked queue

**CRITICAL**: Do NOT add any print, log, or HLOG statements in lock acquire or release paths (Lock, Unlock, ReadLock, ReadUnlock, WriteLock, WriteUnlock). Logging in lock paths causes severe performance degradation and can introduce deadlocks when the logging system itself acquires locks.

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

This requirement applies to ALL Module Create operations in unit tests including admin, bdev, and any custom Modules.

### Test Framework Requirements

**CRITICAL**: Unit tests that initialize the Clio runtime MUST use the `simple_test.h` framework. **DO NOT use Catch2** with Clio runtime initialization.

**Catch2 Incompatibility:**
- Catch2's test framework causes segmentation faults when used with Clio runtime initialization
- This issue was confirmed by copying working test code from `test_bdev_chimod.cc` (which uses simple_test.h) to a Catch2-based test - the identical code segfaulted with Catch2 but worked with simple_test.h
- Root cause: Catch2's test runner infrastructure conflicts with CLIO Runtime's runtime initialization

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

### CLIO Runtime Initialization in Unit Tests

**CRITICAL**: All unit tests MUST use the unified `CLIO_RUNTIME_INIT()` macro (the umbrella entry point). Do NOT call any private/legacy init functions directly.

**Required Pattern for All Unit Tests:**
```cpp
// At the beginning of your test or test fixture setup
bool success = CLIO_RUNTIME_INIT(clio::run::RuntimeMode::kClient, true);
REQUIRE(success);

// Optional: Wait for initialization to complete
std::this_thread::sleep_for(std::chrono::milliseconds(500));

// Verify core managers are available
REQUIRE(CLIO_IPC != nullptr);
REQUIRE(CLIO_IPC->IsInitialized());
```

**Initialization Parameters:**
- **Mode**: Always use `clio::run::RuntimeMode::kClient` for unit tests
- **default_with_runtime**: Always use `true` for unit tests (starts runtime automatically)
- **Environment Variable**: `CLIO_X` is handled automatically by `CLIO_RUNTIME_INIT()`
  - If set to `1`: Runtime will be started
  - If set to `0`: Only client initialization (useful for external runtime scenarios)
  - If not set: Uses the `default_with_runtime` parameter value

**DEPRECATED - Do NOT Use:**
- `initializeBoth()` - Remove from all test fixtures
- `initializeRuntime()` - Remove from all test fixtures
- `initializeClient()` - Remove from all test fixtures

**Example Test Fixture:**
```cpp
class MyTestFixture {
public:
  MyTestFixture() {
    // Initialize CLIO Runtime with client mode and runtime
    bool success = CLIO_RUNTIME_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);

    // Give runtime time to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Verify initialization
    REQUIRE(CLIO_IPC != nullptr);
    REQUIRE(CLIO_POOL_MANAGER != nullptr);
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
  - CLIO_SCHED_WORKERS=8
  - CLIO_MAIN_SEGMENT_SIZE=1G
  - CLIO_CLIENT_DATA_SEGMENT_SIZE=512M
  - CLIO_RUNTIME_DATA_SEGMENT_SIZE=512M
  - CLIO_PORT=9413              # Override RPC port (default: 9413)
  - CLIO_SERVER_ADDR=127.0.0.1 # Override server address for clients
  - CLIO_X=TCP          # SHM, TCP (default), or IPC
  - CLIO_LOG_LEVEL=info
  - CLIO_SHM_SIZE=2147483648
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
  - CLIO_HOSTFILE=/etc/iowarp/hostfile
```

## IPC Transport Modes

CLIO Runtime clients communicate with the runtime server using one of three IPC transport modes, controlled by the `CLIO_X` environment variable. This variable is read during `IpcManager::ClientInit()`.

**Values:**

| Value | Mode | Description |
|-------|------|-------------|
| `SHM` / `shm` | Shared Memory | Client attaches to the server's shared memory queues and pushes tasks directly. Lowest latency, requires same-machine access to the server's shared memory segment. |
| `TCP` / `tcp` | TCP (ZeroMQ) | Client sends serialized tasks over TCP via lightbeam PUSH/PULL sockets. Works across machines. **This is the default when `CLIO_X` is unset.** |
| `IPC` / `ipc` | Unix Domain Socket (ZeroMQ) | Client sends serialized tasks over a Unix domain socket via lightbeam PUSH/PULL. Same-machine only, avoids TCP overhead. |

**Bulk data handling:**
- In SHM mode, `bulk()` serialization writes the `ShmPtr` (allocator ID + offset) since both client and server can resolve shared memory pointers.
- In TCP/IPC mode, buffers are allocated with null `alloc_id_` (private memory). `bulk()` detects null `alloc_id_` and inlines the actual data bytes into the serialization stream.

**Example:**
```bash
# Use shared memory transport (same machine, lowest latency)
export CLIO_X=SHM

# Use TCP transport (default, works across machines)
export CLIO_X=TCP

# Use Unix domain socket transport (same machine, no TCP overhead)
export CLIO_X=IPC
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
- All IOWarp libraries (libclio_run_cxx.so, libclio_ctp_host.so, Module libraries)
- Dependencies from install.sh (Boost, HDF5, ZeroMQ, yaml-cpp, etc.)
- Command-line tools (clio_cte, clio_cae, clio_run, etc.)
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
- Modern find_package() usage for Module discovery
- Proper target linking with namespace::module_type aliases
- Automatic dependency resolution through Module targets
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

## Module Development

When creating or modifying Modules (CLIO Runtime modules), refer to the comprehensive module development guide:

**📖 See [context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md](context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md) for complete Module development documentation**

This guide covers:
- Module structure and architecture
- Task development patterns
- Client and runtime implementation
- Build system integration
- Configuration and code generation
- Synchronization primitives
- Execution modes and dynamic scheduling
- External Module development
- Best practices and common pitfalls
