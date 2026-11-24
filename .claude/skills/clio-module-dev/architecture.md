# Architecture: processes, managers, and how a task travels

Paths are relative to `context-runtime/` unless they name another component
(`CTP` = `context-transport-primitives/`). Line numbers were checked on branch
`cte-shm-cache`; search for the symbol if one has drifted.

## 1. Components of the repository

| directory | what it is |
|---|---|
| `context-transport-primitives/` (CTP, namespace `ctp::`) | shared-memory allocators and data structures, serialization, threads and locks (`ctp::Mutex`, `ctp::RwLock`), lightbeam network transports (ZeroMQ, sockets, SHM MPSC, libfabric, Thallium), system introspection (`ctp::SystemInfo`), GPU macros |
| `context-runtime/` (namespace `clio::run`) | the runtime: IPC manager, workers, scheduler, pool and module managers, the `Container` base class, the `admin`, `bdev`, `safe-bdev` and `MOD_NAME` (template/test) modules, the `clio_run` CLI |
| `context-transfer-engine/` (CTE, namespace `clio::cte`) | tiered blob storage: the `core` module plus interposer modules (shm cache, write-back/write-through caches, indexer, compressor, replication, ...), the filesystem/stream modules and adapters (POSIX, FUSE, HDF5 VOL/VFD, ADIOS2, ...) |
| `context-assimilation-engine/` (CAE) | data ingest modules |
| `context-exploration-engine/` (CEE) | query/exploration |

## 2. Processes

- **Runtime (server) process**: `clio-run start` (kServer mode). Owns the
  shared-memory segments, the workers, every pool's containers, and the
  network endpoints. One runtime per node; nodes are listed in the hostfile.
- **Client processes**: applications and adapters linking module client
  libraries. They build tasks and send them to the local runtime.
- **Co-located runtime**: a process initialized with
  `CLIO_INIT(RuntimeMode::kClient, /*default_with_runtime=*/true)` tries to
  become the runtime itself and otherwise attaches as a client; a ZMQ bind
  decides which process wins (`include/clio_runtime/clio_runtime.h:85-100`,
  `src/clio_runtime.cc:81-105`). Unit tests use this.

### Initialization

`clio::run::CLIO_INIT(mode, default_with_runtime=false, is_restart=false)`
(`include/clio_runtime/clio_runtime.h:113-123`) -> `ClioInitImpl`
(`src/clio_runtime.cc:47-168`):

- `RuntimeMode { kClient, kServer, kRuntime = kServer }` (`clio_runtime.h:58-62`).
- `CLIO_WITH_RUNTIME` ("1"/"true"/"TRUE" = yes) overrides
  `default_with_runtime` (`clio_runtime.cc:58-65`).
- kServer initializes the runtime then the client; kClient initializes the
  client, plus the runtime if requested.
- `atexit(CLIO_RUNTIME_FINALIZE)` is registered; finalize runs
  `ServerFinalize` then `ClientFinalize` (`:163-183`). Call
  `CLIO_RUNTIME_FINALIZE()` before exit in tests (ZMQ context teardown hangs
  otherwise).

## 3. Managers (singletons)

Access each through its macro and **store the pointer before use**:
`auto *ipc = CLIO_IPC; ipc->Foo();` -- never `CLIO_IPC->a()->b`, never
`ctp::Singleton<T>::GetInstance()->var_`.

| macro | class | role | defined |
|---|---|---|---|
| `CLIO_IPC` / `CLIO_CPU_IPC` | `IpcManager` | shared memory, client transports, task allocation (`NewTask`), `Send`/`Recv`, routing (`RouteTask`), network queues | `include/clio_runtime/ipc_manager.h:2251-2252` |
| `CLIO_POOL_MANAGER` | `PoolManager` | pools, containers, address tables, `CreatePool`/`DestroyPool` | `pool_manager.h:606` |
| `CLIO_MODULE_MANAGER` | `ModuleManager` | loads module libraries, constructs containers | `module_manager.h:236` |
| `CLIO_WORK_ORCHESTRATOR` | `WorkOrchestrator` | creates worker threads, lanes, the monitor thread | `work_orchestrator.h:312` |
| `CLIO_CONFIG_MANAGER` | `ConfigManager` | server YAML + env overrides | `config_manager.h:622` |
| `CLIO_RUNTIME_MANAGER` | `RuntimeManager` | lifecycle, compose at start, stop hooks | `manager.h:228` |
| `CLIO_ADMIN` | admin client | | `admin.h:57` |
| `CLIO_CUR_WORKER` | `Worker*` of the calling thread (null off-worker) | | `worker.h:132` |

In GPU device passes `CLIO_IPC` maps to the GPU `IpcManager` (see `gpu.md`).

Runtime internals in one sentence each:

- **Worker** (`include/clio_runtime/worker.h`, `src/worker.cc`): a thread
  that drains its lane, executes tasks as coroutines, resumes suspended ones
  from its event, blocked and periodic queues, and sleeps (epoll) when idle.
- **Scheduler** (`src/scheduler/default_sched.cc`): decides which worker runs a
  task (hold-time based) and assigns worker roles (network send/recv, GPU).
  See `scheduling-and-performance.md`.
- **Lanes**: one task lane per worker (lane i -> worker i). `TaskLane` is
  `ctp::ipc::multi_ext_spsc_queue<Future<Task>, CLIO_QUEUE_ALLOC_T>::ring_buffer_type`
  in namespace `clio::run` (`include/clio_runtime/task.h:874-876`); use
  `TaskLane*` for lane pointers (no `void*` casts). `WorkQueue` is a
  `mpsc_ring_buffer` of `ShmPtr<TaskLane>` (`worker.h:64`; a second,
  equivalent typedef is at `ipc_manager.h:135`).
- **PoolManager** keeps, per pool: metadata, the address map (container ->
  node), the per-node container and a **static container** used for routing.
- **Container**: your module's runtime object, one per pool per node. See
  `module-anatomy.md`.

## 4. A task's life: client call to handler and back

### 4.1 Allocation

`ipc->NewTask<T>(args...)` allocates the task in **private memory** with
`ctp::make_shared<TaskT>(CTP_MALLOC, ...)` and returns a reference-counted
`clio::run::shared_ptr<T>` (`ipc_manager.h:406-439`). There is no `DelTask`:
the task is freed when the last owner drops it. Payload buffers that must be
visible to the runtime come from `ipc->AllocateBuffer(size)` (shared memory in
SHM mode; private, null `alloc_id_` in TCP/IPC mode).

### 4.2 Send

`ipc->Send(task)` returns a `Future<T>` (`ipc_manager.h:551-570`) and picks
the path:

| caller | path | what happens |
|---|---|---|
| client, SHM mode | `IpcCpu2Cpu::SendIn` (`include/clio_runtime/ipc/ipc_cpu2cpu_impl.h:27-131`) | the task is **serialized** (`SaveTaskArchive(kSerializeIn)`) into a sharded MPSC ring in shared memory (`clio-<runtime_pid>-shm-in-<shard>`, shard = tid % `CLIO_SHM_IN_SHARDS`); the response key is the task's own address; send failure completes the task with an error |
| client, TCP or IPC mode | `IpcCpu2CpuZmq::SendIn` (`ipc/ipc_cpu2cpu_zmq_impl.h:16-90`) | serialized and sent over ZeroMQ: TCP uses a DEALER socket to `server_addr:port+3`, responses come back on an ephemeral ROUTER; IPC uses a unix-domain socket at `GetMemfdPath("clio_<port>.ipc")` (`src/ipc_manager.cc:314-384`) |
| inside the runtime (a handler sending a subtask) | `IpcCpu2Self::SendIn` | enqueued by pointer, no serialization |

**Bulk data**: `SerializeIn/Out` call `ar.bulk(ptr, size, flags)`. In SHM
transports a bulk is written as its `ShmPtr` (allocator id + offset); a
`BULK_XFER` bulk whose `alloc_id_` is null (private memory) is inlined as raw
bytes and copied out on the receiver
(`CTP lightbeam/.../shm_mpsc_transport.h:530-592`). ZeroMQ sends `BULK_XFER`
bulks as separate frames (`zmq_transport.h:560ff`). Device pointers are staged
through a host copy (`src/task_archive.cc:72-106`).

### 4.3 Transport selection: `CLIO_IPC_MODE`

`CLIO_IPC_MODE` = `SHM` | `TCP` | `IPC` forces a transport
(`src/ipc_manager.cc:254-268`). Unset, the client probes, fastest first
(`SelectBestIpcMode`, `:197-245`): SHM if the server's main segment exists on
this host, else IPC if the unix socket exists, else TCP. Comments record SHM
round trips ~5x faster than the unix socket and ~190x faster than TCP
(`:256-258`).

- **SHM**: same machine, lowest latency; client attaches to the server's
  segments.
- **TCP** (ZeroMQ): works across machines; the only option for remote clients.
- **IPC** (unix domain socket, ZeroMQ): same machine, no TCP stack.

Older notes call this variable `CLIO_X`; no code reads that name.

### 4.4 Server side: route, schedule, execute

1. Workers pull client SHM requests from their shard
   (`IpcCpu2Cpu::RecvIn`, called from `Worker::DrainMyShard`,
   `src/worker.cc:898,1079-1127`) and execute them inline; tasks from lanes go
   through `Worker::ProcessNewTask` (`:717-769`) -> `RouteAndExec`
   (`:1052-1077`).
2. `IpcManager::RouteTask` (`src/ipc_manager.cc:5183`): collectives first;
   **Dynamic queries are resolved by the pool's static container
   `ScheduleTask`** (`:5280-5287`); `ResolvePoolQuery` turns the query into
   targets (`:5737-5911`); local targets go to `RouteLocal`, which asks the
   scheduler (`RuntimeMapTask`) for a worker; remote targets go to the network
   queues (`RouteGlobal`, `:5688-5697`). Details: `create-and-ids.md` section 6.
3. `Worker::ExecTask` (`src/worker.cc:1418`) checks the per-RPC ACL, starts or
   resumes the task's coroutine; the coroutine is the generated
   `Container::Run(method, task)` switch, which awaits your handler.
4. When the handler finishes, `Worker::EndTask` (`:1587`) trains the cost
   model, then completes the task: locally it sets the task's completion flag
   and wakes the waiter; for a remote origin it queues the response on the
   network (`kSendOutIO` / `kSendOutLatency`, `:1724-1728`).

`ExecTask`, `EndTask` and the other worker execution methods return `void`;
they handle flow internally.

### 4.5 Completion and waiting

- Completion state lives **inside the task**: `FutureInfo fut_` with
  `is_complete_`, `task_size_`, `waiter_pid_`, `waiter_tid_`
  (`task.h:311-334`). (Older text calls this `FutureShm`; the separate object
  is gone except as the GPU completion bit `FUTURE_COMPLETE = 1`,
  `task.h:1127`.)
- Client: `future.Wait(float max_sec = -1, bool reuse_task = false)`
  (`include/clio_runtime/future.h:433`): -1 waits forever, 0 polls once,
  >0 is a timeout in seconds; returns `bool`. SHM responses are received by
  `IpcCpu2Cpu::RecvOut`; ZMQ responses by `IpcCpu2CpuZmq::RecvOut`, which
  waits on the thread's EventManager and deserializes the response archive
  (`ipc_manager.h:629-640`, `ipc_cpu2cpu_zmq_impl.h:150-207`). Then
  `SerializeOut` fields are valid: read `future->return_code_` and your OUT
  fields through the future.
- When the future is destroyed after `Wait`, the task's `PostWait` hook runs
  (e.g. the create task copies the new pool id into the client,
  `ipc_manager.h:2686-2690`).
- Inside a handler, never call `Wait()`: `CLIO_CO_AWAIT(future)` suspends the
  coroutine and the worker runs other tasks until the subtask completes (see
  `locking-and-coroutines.md`).

### 4.6 Runtime-to-runtime (remote) tasks

`IpcManagerRun2Run` (`include/clio_runtime/ipc/ipc_run2run.h:196`,
`src/ipc/ipc_run2run.cc`):

- `SendIn` serializes the task with the **container's** `SaveTask` (the
  generated switch over method ids, `ipc_run2run.cc:224`) and ships it to the
  target node; the receiver allocates the task with `AllocLoadTask`, runs it,
  and `SendOut` returns the OUT fields.
- A task sent to several containers (Broadcast, Range) creates one replica per
  target; replies are merged into the origin with the container's
  `AggregateOut` (`RecvOutAggregate`, `ipc_run2run.cc:1059-1093`), which
  dispatches to your task's `AggregateOut` (rules in `module-anatomy.md`).
- `completer_` records which container finished the task (0 = unset,
  `task.h:358,533`); it is not used for routing.
- Network send lanes are `NetQueuePriority`: `kSendInLatency=0`,
  `kSendInIO=1`, `kSendOutLatency=2`, `kSendOutIO=3`, `kClientSendTcp=4`,
  `kClientSendIpc=5`; threshold `kNetQueueIoThreshold = 4096` bytes
  (`ipc_manager.h:100-124`). See `scheduling-and-performance.md`.
- A task whose query resolves to nothing completes with
  `kRouteNullPoolRc = -1001` (`ipc_run2run.h:61`).

## 5. Shared memory, IPC paths and cleanup

### Segments

`MemorySegment` = `kMainSegment 0`, `kClientDataSegment 1`,
`kQueueSegment 2`, `kMetadataSegment 3` (`include/clio_runtime/types.h:670-680`).
Defaults (`src/config_manager.cc:443-453`): main size 0 (auto), client data
512 MB, metadata 0 (auto); names `chi_main_segment_${USER}`,
`chi_client_data_segment_${USER}`, `chi_metadata_segment_${USER}`.
`CLIO_MAIN_SEGMENT_SIZE` overrides the main size; `CLIO_IPC_NAMESPACE` adds a
suffix to segment names (to run several runtimes per user). Container
`shm_size` (Docker) must cover all segments.

Modules may create their own segments (the CTE shm cache does,
`cte-data-path.md`) and register them in the metadata segment's directory so
clients can find them.

### Paths: always use `SystemInfo`

Never hardcode `/tmp/clio_*`. Use (`CTP introspect/src/system_info.cc:524-573`):

- `ctp::SystemInfo::GetMemfdDir()` -> `$CLIO_MEMFD_DIR` if set, else
  `/tmp/clio_$USER` (Windows: `%TEMP%/clio_<USERNAME>`).
- `ctp::SystemInfo::GetMemfdPath(name)` -> `<dir>/<name>` (a leading `/` is
  stripped).
- `ctp::SystemInfo::EnsureMemfdDir()` -> creates the directory (0700) and
  returns false unless it is a real directory owned by this user that no
  other user can write (`CheckMemfdDir()`). The runtime refuses to start and
  SHM/IPC clients refuse to attach otherwise.
- IPC socket: `ctp::SystemInfo::GetMemfdPath("clio_" + std::to_string(port) + ".ipc")`.

### Automatic cleanup at runtime start

`IpcManager::ServerInit` (`src/ipc_manager.cc:493ff`) calls
`ClearUserIpcs()` (`:528`, implementation `:3838-3972`) before creating
segments. It scans `GetMemfdDir()` and removes leftover memfd symlinks, pid
records and IPC socket files from previous or crashed runs, **except**:
everything is kept if this port's pid record names another live process;
memfd symlinks into a live process's `/proc/<pid>`; pid records of live
processes; files suffixed with another port whose runtime is alive. Removal
failures (e.g. another user's file) are logged at kDebug and skipped; the
count is logged at kInfo and returned. So stale state from crashed runs needs
no manual cleanup, and other users' / other ports' live runtimes are not
disturbed.

### Prefaulting

`ctp::SystemInfo::BulkFault(ptr, size)` (`CTP introspect/src/system_info.cc:946ff`)
populates pages up front: `madvise(MADV_POPULATE_WRITE)` on Linux (cached as
unsupported on EINVAL/ENOSYS), `PrefetchVirtualMemory` on Windows, else a
touch loop. Prefault every large segment you create; otherwise every first
write takes a page fault in the writer (~256 faults per MiB of 4 KiB pages).

## 6. Configuration

### Where the server config comes from

`ConfigManager` (`src/config_manager.cc:318-360`): `CLIO_SERVER_CONF` first;
with `CLIO_TEST_MODE` set (not "0"), no config; otherwise
`${HOME}/.clio/clio.yaml`. `config/clio_default.yaml` is the documented
default file; CMake installs it into the pip wheel's `data/`
(`CMakeLists.txt:2624-2626`) and the pip launcher falls back to it
(`installers/pip/iowarp_core/_config.py:35-58`); the C++ runtime does not read
it directly.

Top-level sections: `networking:`, `runtime:`, `swim:`, `compose:` (plus
`gpu:` and `viz:`). Keys (`src/config_manager.cc:484-613`):
`runtime.{num_threads, queue_depth, local_sched, client_reap_grace_s,
first_busy_wait, task_progress_interval_ms, conf_dir, learning_rate,
ipc_namespace, metadata_segment_size, main_segment_size}`,
`networking.{port (default 9413), neighborhood_size, hostfile,
wait_for_restart, wait_for_restart_poll_period}`,
`gpu.{blocks, threads_per_block, queue_depth}`. Empty hostfile = single node.
The runtime binds ports base, base+1 and base+3 (`src/ipc_manager.cc:657-693`).

### Environment overrides that exist in code

| variable | effect |
|---|---|
| `CLIO_SERVER_CONF` | server YAML path |
| `CLIO_PORT` | base port (default 9413) |
| `CLIO_SERVER_ADDR` | server address for clients (default 127.0.0.1) |
| `CLIO_IPC_MODE` | client transport `SHM`/`TCP`/`IPC` (unset = probe) |
| `CLIO_WITH_RUNTIME` | start/attach an in-process runtime (see section 2) |
| `CLIO_NUM_THREADS` | worker count |
| `CLIO_MAIN_SEGMENT_SIZE` | main segment size |
| `CLIO_IPC_NAMESPACE` | segment-name suffix |
| `CLIO_MEMFD_DIR` | memfd/socket directory |
| `CLIO_SHM_IN_SHARDS`, `CLIO_SHM_ASYNC_SEND`, `CLIO_SHM_CLIENT_SPIN_US` | SHM request path tuning |
| `CLIO_EPHEMERAL`, `CLIO_BIND_ADDR`, `CLIO_ZMQ_LOCAL_IPC`, `CLIO_CLIENT_RETRY_TIMEOUT`, `CLIO_CLIENT_TRY_NEW_SERVERS`, `CLIO_WAIT_SERVER`, `CLIO_FORCE_NET`, `CLIO_TEST_MODE` | see `src/config_manager.cc:168-281`, `src/ipc_manager.cc:128-507` |
| `CTP_LOG_LEVEL` | log level (`CTP util/include/clio_ctp/util/logging.h:172-173`) |
| `CLIO_VIZ_*` | web dashboard |

The hostfile is set only by YAML `networking.hostfile` (no env var). Variables
that appear in older notes but are read nowhere: `CLIO_X`,
`CLIO_SCHED_WORKERS`, `CLIO_CLIENT_DATA_SEGMENT_SIZE`,
`CLIO_RUNTIME_DATA_SEGMENT_SIZE`, `CLIO_SHM_SIZE`, `CLIO_HOSTFILE`,
`CLIO_LOG_LEVEL`.

## 7. Known stale comments in the code (do not copy them)

- `pool_query.h:53,155`: Dynamic described as routing to Monitor with
  `kGlobalSchedule` (it is `ScheduleTask` now).
- `ipc_manager.h:1590-1598`: `ClearUserIpcs` "cleans up all entries" (it keeps
  live ones).
- `ipc_manager.cc:540-548`: says the port claim precedes `ClearUserIpcs`; the
  code calls `ClearUserIpcs` first (`:528`).
- `run_inline.h:36`: "kill switch `CLIO_DISABLE_INLINE_RUN`" (inline run is
  opt-in via `CLIO_ENABLE_INLINE_RUN`).
- `work_orchestrator.cc:86-96` log text "task + 1 dedicated network" (the
  DefaultScheduler layout differs).
- `scheduler.h:84`, `default_sched.h:98`: monitor "~500 ms" (it is 100 ms).
- `config/clio_default.yaml:100` documents `first_busy_wait: 10000`; the code
  default is 1000 us (`src/config_manager.cc:464`).
