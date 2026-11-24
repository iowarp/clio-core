# Module (chimod) anatomy

A module is a pair of libraries built from one source directory: a **client**
library that applications link to build and send tasks, and a **runtime**
library that the runtime process loads to execute them. Both are compiled from
the same **tasks header**, which is the contract between them. The canonical
small example is `context-runtime/modules/MOD_NAME/` (the template/test module);
large real examples are `context-runtime/modules/bdev/`,
`context-runtime/modules/admin/` and `context-transfer-engine/core/`.

Paths below are relative to the repository root. Line numbers were checked on
branch `cte-shm-cache`.

## 1. Directory layout

```
<repo>/                                  # a "module repository"
  clio_repo.yaml                         # namespace + list of module dirs
  <mod>/
    clio_mod.yaml                        # module name, namespace, method ids
    CMakeLists.txt                       # add_clio_module_client / _runtime
    include/<ns_dir>/<mod>/
      <mod>_tasks.h                      # CreateParams + task structs (shared)
      <mod>_client.h                     # Client class (Async* methods)
      <mod>_runtime.h                    # Runtime class (Container subclass)
      autogen/<mod>_methods.h            # GENERATED: Method::k* constants
    src/
      <mod>_client.cc                    # client out-of-line bits (often tiny)
      <mod>_runtime.cc                   # handlers + CLIO_TASK_CC(Runtime)
      autogen/<mod>_lib_exec.cc          # GENERATED: Init/Run/Save/Load/... switches
    test/                                # simple_test.h unit tests
```

`<ns_dir>` is the include directory for the namespace: `clio_runtime/` for
`clio::run`, `clio_cte/` for `clio::cte` (the generator finds the existing
directory; `context-runtime/util/clio_run_cmd_refresh_repo.cc` `IncludePrefix`).

## 2. `clio_repo.yaml` and `clio_mod.yaml`

`clio_repo.yaml` (e.g. `context-transfer-engine/clio_repo.yaml`,
`context-runtime/modules/clio_repo.yaml`):

```yaml
namespace: clio::cte          # default C++ namespace of the repo's modules
version: 1.0.0
modules:                      # directories the generator visits
  - core
  - compressor
```

`clio_mod.yaml` (e.g. `context-runtime/modules/MOD_NAME/clio_mod.yaml`):

```yaml
module_name: MOD_NAME
namespace: clio::run          # overrides the repo namespace
version: 1.0.0
# Inherited (lifecycle) methods: ids 0-9, -1 = not implemented
kCreate: 0
kDestroy: 1
kRestart: 2                   # optional: generates Runtime::Restart (sets is_restart_)
kNodeFailure: -1
kRecover: -1
kMigrate: -1
kUpgrade: -1
kMonitor: 9
# Custom methods: ids >= 10 (interposers: own verbs >= 100)
kCustom: 10
```

Generator rules (`context-runtime/util/clio_run_cmd_refresh_repo.cc`):

- Every integer key `kFoo` becomes method `Foo` with that id; `-1` entries are
  skipped (`:179-225`); ids < 10 count as inherited.
- `kRestart` is not a dispatched method: when >= 0 it generates
  `Runtime::Restart(...) { is_restart_ = true; Init(...); }` (`:387-395`).
- The task type for method `Foo` is `FooTask`; `Create` -> `CreateTask`,
  `Destroy` -> `DestroyTask`, methods starting `GetOrCreate` ->
  `<mod>::GetOrCreateXTask<<mod>::CreateParams>` (`GetTaskTypeName`,
  `:323-337`). **Name your task structs `<Method>Task` and your handlers
  `<Method>`** or the generated code will not compile.
- `kMaxMethodId` = highest id + 1; `GetMethodNames()` maps ids to names (used
  by the cost model and logs).

### Regenerating

After adding or renumbering a method, run the `clio-run` binary's refresh
command on the module repository directory (the one holding
`clio_repo.yaml`):

```bash
<build>/bin/clio-run refresh <path/to/repo>      # canonical form
<build>/bin/clio-run repo refresh <path/to/repo> # legacy form, same thing
```

(`context-runtime/util/clio_run.cc:86-150`). It rewrites, for **every**
module listed in `clio_repo.yaml`, `include/.../autogen/<mod>_methods.h` and
`src/autogen/<mod>_lib_exec.cc`.

- Never hand-edit `autogen/` files.
- **Known drift -- check before running refresh on the CTE repo.**
  `context-transfer-engine/core/clio_mod.yaml` states that methods 36-42,
  `kMultiPutBlob` (48) and `kRegisterReplicaContainer` (49) exist in
  `autogen/core_methods.h` but not in the yaml, so a refresh would DELETE
  them; and `core_shm_cache_writethru/clio_mod.yaml` lacks `kTagAttrs` (103),
  which is in its generated header. Before refreshing a repo, diff each
  module's generated methods header against its yaml and add the missing
  entries to the yaml first; review `git diff` of every `autogen/` file after.
- Also note that not every module directory is listed in a `clio_repo.yaml`
  (e.g. `cache/`, `cache_base/`, `indexer/` are absent from
  `context-transfer-engine/clio_repo.yaml`); refresh only touches listed ones.

## 3. The tasks header (`<mod>_tasks.h`)

### CreateParams and CreateTask

```cpp
struct CreateParams {
  u32 worker_count_;                                   // module config
  static constexpr const char *chimod_lib_name = "clio_MOD_NAME";  // module identity
  CreateParams(u32 worker_count = 1) : worker_count_(worker_count) {}
  template <class Archive> void serialize(Archive &ar) { ar(worker_count_); }
  // Optional: void LoadConfig(const PoolConfig &cfg) -- read compose YAML keys
};
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;
```

(`context-runtime/modules/MOD_NAME/include/clio_runtime/MOD_NAME/MOD_NAME_tasks.h:56-81`.)
`chimod_lib_name` must equal the runtime library's basename without
`_runtime` (see `create-and-ids.md` section 3). The client must also define
the static member out of line (`constexpr const char* CreateParams::chimod_lib_name;`,
`MOD_NAME/src/MOD_NAME_client.cc`).

### A task struct

```cpp
struct FooTask : public clio::run::Task {
  IN    u64 offset_;
  IN    ctp::ipc::ShmPtr<> data_;     // payload buffer (AllocateBuffer)
  IN    u64 size_;
  INOUT clio::run::priv::string name_;
  OUT   u64 result_;

  /** SHM default constructor: required (NewTask/AllocLoadTask use it) */
  FooTask() : Task(), offset_(0), size_(0), name_(CLIO_PRIV_ALLOC), result_(0) {}

  /** Emplace constructor: first four args are always these */
  explicit FooTask(const TaskId &task_node, const PoolId &pool_id,
                   const PoolQuery &pool_query, /* inputs... */)
      : Task(task_node, pool_id, pool_query, Method::kFoo), ... {
    task_id_ = task_node; pool_id_ = pool_id; method_ = Method::kFoo;
    task_flags_.Clear(); pool_query_ = pool_query;
  }

  CTP_CROSS_FUN ~FooTask() {}

  template <typename Ar> CTP_CROSS_FUN void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);                  // ALWAYS first
    ar(offset_, size_, name_);
    ar.bulk(data_, size_, BULK_XFER);       // input payload travels
  }
  template <typename Ar> CTP_CROSS_FUN void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);                 // ALWAYS first
    ar(name_, result_);
  }
  void Copy(const ctp::ipc::FullPtr<FooTask> &other) {          // whole-task copy
    Task::Copy(other.template Cast<Task>());
    offset_ = other->offset_; /* ... every field ... */
  }
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &o) {  // replica -> origin
    Task::AggregateOut(o);                  // ALWAYS first
    result_ += o.template Cast<FooTask>()->result_;   // OUT fields only
  }
  // Optional (collectives): void AggregateIn(const FullPtr<Task> &member)
};
```

Facts and rules:

- `IN`, `OUT`, `INOUT`, `TEMP` are **empty annotation macros**
  (`context-runtime/include/clio_runtime/types.h:753-756`). They document
  direction; what actually travels is exactly what `SerializeIn` /
  `SerializeOut` write. A field missing from them is silently not
  transmitted across processes or nodes.
- Base `Task` fields (`context-runtime/include/clio_runtime/task.h:344-358`):
  `pool_id_`, `task_id_`, `pool_query_`, `method_`, `task_flags_`,
  `period_ns_`, `task_group_` (IN), `return_code_` and `completer_` (OUT,
  atomics), plus per-process `TEMP` state (`fut_` completion record,
  `run_ctx_`). `Task::SerializeIn` / `SerializeOut` handle them
  (`task.h:676-693`); derived tasks call them first.
- `return_code_` is atomic: read with `GetReturnCode()`, write with
  `SetReturnCode(v)` (`task.h:699-706`). 0 = success. Fields touched by
  several threads must be atomic (use `std::atomic`/`ctp::ipc::atomic`).
- **Bulk payloads** (`ar.bulk(ptr, size, flags)`, flags in
  `context-transport-primitives/lightbeam/include/clio_ctp/lightbeam/lightbeam.h:61-63`):
  - `BULK_XFER`: the bytes are transmitted.
  - `BULK_EXPOSE`: only the buffer's metadata is sent so the receiver can
    allocate/register a destination.
  - Pattern for an input payload: `BULK_XFER` in `SerializeIn`. Pattern for an
    output buffer (a read): `BULK_EXPOSE` in `SerializeIn`, `BULK_XFER` in
    `SerializeOut` (`context-transfer-engine/core/include/clio_cte/core/core_tasks.h:3020-3048`
    for `GetBlobTask`; `:2730-2745` for `PutBlobTask`).
  - Make the bulk conditional on the same serialized state on both sides
    (e.g. `if (!context_.emulate_)` after `context_` was serialized).
- **Strings and vectors in tasks** use `clio::run::priv::string` /
  `priv::vector` constructed with `CLIO_PRIV_ALLOC`; `std::string` is fine
  inside `CreateParams` (serialized into `chimod_params_`).
- **`Copy` vs `AggregateOut`** (contract documented at `task.h:750-800`,
  enforced by the `cr_aggregate_out_contract` lint test):
  - `Copy` is a whole-task assignment (used by `NewCopyTask` for replicas).
  - `AggregateOut` merges a **replica's OUT/INOUT fields** into the origin
    after a Broadcast/Range. Never implement it by calling `Copy()` (it would
    overwrite the origin's `task_id_`, `pool_query_`, `completer_` with the
    replica's mid-flight: a real crash, issue #856). Never reassign
    client-owned shared-memory members (IN strings, payload `ShmPtr`s).
    Choose a reduction per field (sum partitioned counters, max an
    owner-reported value, first non-empty string, concatenate vectors) and
    make a single-replica gather land on that replica's value.
  - `AggregateIn` folds a collective member's **inputs** into the aggregate
    task (ManyToOne/AllToOne); default no-op (`task.h:815`).
- **Task structs are ABI.** They cross the client/runtime boundary in shared
  memory and over the network. Adding, removing or reordering a field means
  rebuilding everything (see `build-and-test.md` section 3).
- **Fully-POD tasks for GPU** submission must have identical layout on host
  and device (no SSO strings, fixed-size arrays such as `fixed_string<32>`,
  `CTP_CROSS_FUN` members). See `gpu.md`.
- Named constants: never pass raw integers for queue ids, priorities, flags
  or method ids. Define `constexpr` names (`kCtePutReplace`,
  `NetQueuePriority::kSendInIO`, `Method::kFoo`, ...).

## 4. The client (`<mod>_client.h`)

```cpp
class Client : public clio::run::ContainerClient {
 public:
  Client() = default;
  explicit Client(const PoolId &pool_id) { Init(pool_id); }

  Future<CreateTask> AsyncCreate(const PoolQuery &q, const std::string &pool_name,
                                 const PoolId &custom_pool_id /*, params */);
  Future<FooTask> AsyncFoo(const PoolQuery &q, /* inputs */) {
    auto *ipc = CLIO_CPU_IPC;                          // store the singleton pointer
    auto task = ipc->NewTask<FooTask>(clio::run::CreateTaskId(), pool_id_, q,
                                      /* inputs */);
    return ipc->Send(task);
  }
};
```

- `ContainerClient` (`context-runtime/include/clio_runtime/container.h:824-860`)
  holds `pool_id_`, `return_code_` (of the last Create) and
  `SetPersistent(bool)` (pools created by this client are recorded in the pool
  log and re-created after restart, the API equivalent of compose
  `restart: true`).
- Every operation is an `Async*` method returning `Future<TaskT>`; callers
  `Wait()` and read results via `future->field_`. A synchronous convenience
  wrapper may call `Async*` + `Wait()` but must not be used from inside a
  runtime handler.
- Create is special (`create-and-ids.md` section 4): send to `kAdminPoolId`,
  default `PoolQuery::Dynamic()`, `CreateParams::chimod_lib_name`, a
  user-provided pool name, pass `this` for `PostWait`.
- Other methods use the client's `pool_id_`. Pick the default `PoolQuery` per
  method deliberately: `Dynamic()` when the module's `ScheduleTask` knows best
  (e.g. hash a key to its owner), `Local()` for node-local work, a concrete
  `DirectHash`/`Broadcast` when the semantics demand it. **Never a null
  query.** Unsure: `Local()`.
- Payloads: allocate with `ipc->AllocateBuffer(size)` and pass the
  `ShmPtr`; or, for a caller-owned pointer in a co-located runtime, wrap it
  without copying (`ShmPtr::FromRaw`, see the CTE client in
  `cte-data-path.md`). Do not add a second staging copy.
- The runtime also uses clients: a container calls other pools (or itself)
  through a `Client` member, e.g. `client_ = Client(pool_id)` in the generated
  `Init` (`MOD_NAME/src/autogen/MOD_NAME_lib_exec.cc:21-31`).

## 5. The runtime (`<mod>_runtime.h` / `.cc`)

```cpp
class Runtime : public clio::run::Container {
 public:
  using CreateParams = clio::run::MOD_NAME::CreateParams;   // used by CLIO_TASK_CC
  TaskStat GetTaskStats(const Task *task) const override;   // cost model: REQUIRED
  PoolQuery ScheduleTask(const shared_ptr<Task> &task) override;  // optional (Dynamic)

  void Init(const PoolId&, const std::string&, u32 container_id = 0) override;  // generated
  TaskResume Run(u32 method, shared_ptr<Task> task) override;                   // generated
  TaskResume Create(shared_ptr<CreateTask> &task);
  TaskResume Destroy(shared_ptr<DestroyTask> &task);
  TaskResume Monitor(shared_ptr<MonitorTask> &task);
  TaskResume Foo(shared_ptr<FooTask> &task);
  u64 GetWorkRemaining() const override;
  // Save/Load/LocalSave/LocalLoad/AllocLoad/LocalAllocLoad/NewCopyTask/NewTask/
  // AggregateOut/AggregateIn overrides: declared here, defined in autogen.
 private:
  Client client_;
};
```

(`context-runtime/modules/MOD_NAME/include/clio_runtime/MOD_NAME/MOD_NAME_runtime.h`.)

Handlers are coroutines:

```cpp
TaskResume Runtime::Foo(shared_ptr<FooTask> &task) {
  CLIO_TASK_BODY_BEGIN
  auto sub = other_client_.AsyncBar(PoolQuery::Local(), ...);
  CLIO_CO_AWAIT(sub);                    // suspend; worker runs other tasks
  task->result_ = sub->value_;
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}
```

The macros (`context-runtime/include/clio_runtime/task.h:2122-2137`) select
C++20 stackless coroutines (default) or Boost.Context fibers
(`CLIO_CORE_ENABLE_BOOST_COROUTINES`); always use them, never raw
`co_await`/`co_return`, so a module compiles under both. See
`locking-and-coroutines.md`.

End the runtime `.cc` with `CLIO_TASK_CC(clio::run::MOD_NAME::Runtime)`
(`MOD_NAME_runtime.cc:362`): it exports `alloc_chimod`, `new_chimod`,
`get_chimod_name` (returns `CreateParams::chimod_lib_name`) and
`destroy_chimod` (`container.h:970-990`). Without it the module manager does
not recognize the library.

### Container virtual API (`container.h`)

| member | line | who writes it | purpose |
|---|---|---|---|
| `pool_id_`, `pool_name_`, `container_id_` (== node id) | 106-109 | base | identity |
| `Init(pool_id, pool_name, container_id)` | 152 | **generated** | base init, `client_ = Client(pool_id)`, `DefineModel`, method names |
| `Restart(...)` | 531 | generated if `kRestart` | sets `is_restart_`, calls `Init` |
| `Run(method, task)` | 502 | **generated** | switch: cast, `CLIO_CO_AWAIT(Handler(task))` |
| `GetTaskStats(task)` | 245 | **you** | cost features for every method (section 6) |
| `ScheduleTask(task)` | 488 | you (optional) | resolve Dynamic queries; runs on the static container |
| `InlineOp(method, in, out)` | 270 | you (rare) | synchronous same-thread execution of a pure in-memory method; default false |
| `GetWorkRemaining()` / `UpdateWork` | 510 / 518 | you | outstanding-work accounting (0 if none) |
| `FillExistingCreate(task)` | 651 | you (optional) | write Create outputs for an already-existing pool |
| `GetNextPool()` | 641 | interposers | the pool this one forwards to |
| `SaveTask` / `LoadTask` / `AllocLoadTask` / `LocalSaveTask` / `LocalLoadTask` / `LocalAllocLoadTask` / `NewCopyTask` / `NewTask` / `AggregateOut` / `AggregateIn` | 661-765 | **generated** | method-id switches over your task types |
| `Recover`, `ScheduleRecover`, `Expand`, `Migrate`, `RegisterViz`, `BuildBatch`, `SmashBatch`, `PostGpuContainerCreate` | 543-808 | optional | recovery, viz, batching hooks |

Never re-implement the generated methods (`SaveIn/LoadIn/SaveOut/LoadOut`,
`NewCopy`, `Aggregate`, `Run`, `Init`, the Save/Load family) in your runtime
sources; change the yaml and regenerate. There is no `DelTask` (tasks are
reference counted) and no `Monitor` virtual (Monitor is method 9).

### Lifecycle methods

- **Create**: see `create-and-ids.md` section 4.5.
- **Destroy**: release resources, set `return_code_`. Note that
  `PoolManager::DestroyPool` does not appear to call it
  (`create-and-ids.md` section 8) -- make the destructor safe on its own.
- **Monitor** (method 9): fill `task->results_[container_id_]` with a
  msgpack-encoded answer to `task->query_`; return rc 0 for unknown queries.

## 6. The cost model: `GetTaskStats` for every method

The scheduler places tasks by predicted hold time and picks the network lane
by payload size; both come from `GetTaskStats`
(`scheduling-and-performance.md`). The default returns zeros
(`container.h:245-248`), which makes every method look like the same tiny
constant: a multi-MB task piles onto one worker and its traffic rides the
latency network lane.

Return, for **every** method, using `context-runtime/include/clio_runtime/task_cost.h`:

```cpp
TaskStat GetTaskStats(const Task *task) const override {
  namespace cost = clio::run::cost;
  if (task == nullptr) return TaskStat();
  switch (task->method_) {
    case Method::kWrite: {                       // data task: scale with bytes
      auto *t = static_cast<const WriteTask *>(task);
      return cost::Data(t->size_, cost::kCopyBytesPerCpuUs,
                        cost::kStorageBytesPerWallUs);   // io_size_ = bytes
    }
    case Method::kBatch:                          // control verb over N items
      return cost::PerItem(static_cast<const BatchTask *>(task)->count_, 5, 2, 10.f, 5.f);
    case Method::kCreate:  return cost::ContainerCreate();
    case Method::kDestroy: return cost::ContainerDestroy();
    case Method::kMonitor: return cost::Fixed(10, 15.0f);
    default:               return cost::Metadata();     // deliberate constant, never zeros
  }
}
```

- `TaskStat{io_size_, compute_, wall_time_}` (`task.h:301-305`).
- Rates (`task_cost.h:75-90`): `kCopyBytesPerCpuUs = 10000` (~10 GB/s copy),
  `kStorageBytesPerWallUs = 500` (~500 MB/s; bdev persisted profiles depend
  on this scale -- do not change it), `kMemoryBytesPerWallUs = 5000`,
  `kNetworkBytesPerWallUs = 1000` + `kNetworkRttUs = 50`,
  `kCompressBytesPerCpuUs = 300`, `kDecompressBytesPerCpuUs = 900`. Use a
  better known rate for your device if you have one.
- Presets: `Trivial` (1/2 us), `Metadata` (10/15), `IdlePeriodic` (5/10),
  `Scan` (200/300), `ContainerCreate` (100/2000), `ContainerDestroy`
  (50/500), `DeviceSync` (10/100) (`task_cost.h:147-164`).
- `wall_time_` should seed the time the task **occupies its worker**
  (suspensions excluded), not its end-to-end latency.
- Interposers fall back to the core's stats for core methods
  (`CoreInterposer::GetTaskStats`) but must add cases for their own verbs.

## 7. CMake

```cmake
add_clio_module_client(
  SOURCES src/foo_client.cc
  # LIB_NAME clio_foo            # optional basename override
  # INCLUDE_DIRECTORIES / LINK_LIBRARIES / COMPILE_DEFINITIONS ...
)
add_clio_module_runtime(
  SOURCES src/foo_runtime.cc src/autogen/foo_lib_exec.cc
  # LINK_LIBRARIES ctp::compress ...   # only the modular targets you use
)
if(CLIO_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
```

(`cmake/ClioCoreCommon.cmake:1146-1480`.) Both functions read `clio_mod.yaml`
from the current source dir for `module_name` and `namespace`, and:

- **Target names**: `${PACKAGE_NAME}_${MODULE_NAME}_client` /
  `_runtime`, where `PACKAGE_NAME` is the namespace with `::` replaced by
  `_` (`clio::cte` -> `clio_cte`), or `${LIB_NAME}_client` / `_runtime` if
  `LIB_NAME` is given (admin: `clio_admin`, bdev: `clio_bdev`, MOD_NAME:
  `clio_MOD_NAME`, safe-bdev: `clio_safe_bdev`). The output file name follows
  the target name.
- **Aliases** (use these when linking): `${NAMESPACE}::${MODULE_NAME}_client`
  / `_runtime`, e.g. `clio::run::admin_client`, `clio::cte::core_runtime`.
  Legacy spellings (`clio_run::`, `clio_cte::`, `clio_cae::`, and
  `clio::run::<module>_<x>` for the renamed runtime modules) still resolve.
- **Packages** (install `lib/cmake/<pkg>`): `${PACKAGE_NAME}_${MODULE_NAME}`
  for `clio::cte` / `clio::cae` modules (`find_package(clio_cte_core)`);
  pinned to `clio_<module>` for the `clio::run` modules
  (`find_package(clio_admin)`, `clio_run_bdev` compat).
- **Automatic linking**: the runtime links its own client, `Threads`, `rt`
  (Linux, POSIX AIO), and -- for modules other than admin/bdev -- the admin
  and bdev runtimes; the client links the admin (and bdev) clients. Both get
  `include/` as a public include dir; runtimes get `CLIO_RUNTIME=1`.
- C++20 is required (`cxx_std_20`).

### Linking: CTP modular targets

Link only what you use; never link yaml-cpp or cereal directly.

| target | provides | compile definitions |
|---|---|---|
| `ctp::cxx` | core CTP (shared memory, data structures); links configure + thread_all; always needed (module clients get it automatically) | |
| `ctp::configure` | YAML parsing (yaml-cpp) -- use instead of yaml-cpp | |
| `ctp::serialize` | cereal serialization -- use instead of cereal | `CTP_ENABLE_CEREAL` |
| `ctp::interceptor` | ELF interception (adapters) | `CTP_ENABLE_ELF` |
| `ctp::lightbeam` | ZeroMQ / libfabric / Thallium transports | `CTP_ENABLE_ZMQ`, `CTP_ENABLE_LIBFABRIC`, `CTP_ENABLE_THALLIUM` |
| `ctp::thread_all` | pthread / OpenMP / thread models | `CTP_ENABLE_OPENMP`, `CTP_ENABLE_PTHREADS`, `CTP_ENABLE_WINDOWS_THREADS`, `CTP_DEFAULT_THREAD_MODEL(_GPU)` |
| `ctp::mpi` | MPI (only where needed) | `CTP_ENABLE_MPI` |
| `ctp::compress` / `ctp::encrypt` | compression / encryption | `CTP_ENABLE_COMPRESS` / `CTP_ENABLE_ENCRYPT` |
| `ctp::aio` | async I/O backends | |
| `ctp::nixl` | NIXL transport (`CLIO_CORE_ENABLE_NIXL=ON`, NIXL in `/usr/local` with the POSIX backend) | `CTP_ENABLE_NIXL=1` |
| `ctp::nvshmem` | NVSHMEM (`CLIO_CORE_ENABLE_NVSHMEM=ON`) | `CTP_ENABLE_NVSHMEM=1` |
| `ctp::cuda_cxx` / `ctp::rocm_cxx` | GPU-enabled CTP for kernel code (only when CUDA/ROCm is enabled) | `CTP_ENABLE_CUDA=1` / `CTP_ENABLE_ROCM=1` |

(Aliases in `context-transport-primitives/CMakeLists.txt:188-515`; exported
under `NAMESPACE ctp::` at `:601`.) Module clients should need only
`ctp::cxx`; runtimes add what they use; tests link only what they test; GPU
kernel code uses `ctp::cuda_cxx`/`ctp::rocm_cxx`, host code `ctp::cxx`.

### External projects

```cmake
find_package(clio-core REQUIRED)   # all ctp::* targets, clio::run::cxx, module
                                   # build functions, admin + bdev modules, and
                                   # the CTE/CAE modules if they were built
target_link_libraries(app clio::run::admin_client clio::run::bdev_client
                          clio::cte::core_client)
```

Or individually: `find_package(ClioCtp)`, `find_package(clio-run)`,
`find_package(clio_admin)`, `find_package(clio_run_bdev)`,
`find_package(clio_cte_core)`, `find_package(clio_cae_core)`. An external
module repository uses the same `add_clio_module_*` functions after
`find_package(clio_run REQUIRED)` and `find_package(clio_admin REQUIRED)`;
examples: `context-runtime/test/unit/external-chimod/` and
`context-transfer-engine/test/unit/external/`.

## 8. How client and runtime relate (summary)

1. Both include the same `<mod>_tasks.h` and the generated
   `<mod>_methods.h`: same task layout, same method ids.
2. The client constructs `FooTask` with `NewTask` and `Send`s it; the task's
   `pool_id_`, `method_` and `pool_query_` say where and what.
3. The runtime routes it (`ScheduleTask` for Dynamic), picks a worker
   (`GetTaskStats` -> scheduler), and the generated `Run` casts it back to
   `FooTask` and awaits your `Foo` handler on the container of the right
   node.
4. Across processes or nodes, `SerializeIn` carries inputs and
   `SerializeOut` carries outputs; `AggregateOut` merges multi-container
   replies.
5. The client's `Future` completes, and the caller reads OUT fields and
   `return_code_`.

Any disagreement between the two sides (a stale build, a field not
serialized, a renumbered method) shows up as plausible garbage, not as an
error.
