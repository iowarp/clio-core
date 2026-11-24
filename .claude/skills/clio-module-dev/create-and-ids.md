# Create, pools, ids and routing

How a pool comes to exist, how it is named and numbered, and how every task
finds the container that runs it. Paths are relative to `context-runtime/`
unless they name another component. Line numbers were checked on branch
`cte-shm-cache`; search for the quoted symbol if one has drifted.

## 1. Vocabulary

| term | meaning |
|---|---|
| **module** (chimod) | a shared library `lib<name>_runtime.so` (+ `_client`) implementing a `Container` subclass |
| **pool** | one named, numbered instance of a module across the cluster; identified by a `PoolId` |
| **container** | the pool's per-node object (`Container` subclass instance) that executes tasks; identified within the pool by a `ContainerId` |
| **static container** | an extra, state-less container per pool (id `kStaticContainerId`), used only for routing decisions (`ScheduleTask`) |
| **method** | a numbered entry point of a module (`Method::kFoo`), dispatched by `Container::Run(method, task)` |
| **task** | a struct derived from `clio::run::Task` carrying a method's inputs and outputs |
| **PoolQuery** | the routing instruction a task carries: which container(s) should run it |

## 2. Ids

- `PoolId = UniqueId{u32 major_, u32 minor_}` (`include/clio_runtime/types.h:228-290`);
  null is `(0,0)`, string form `"major.minor"` (`src/types.cc:45-59`).
- Reserved: `kAdminPoolId = (1,0)` (`types.h:579-580`). Existing conventions
  in `config/clio_default.yaml`: the default RAM bdev composes at `301.0`
  (`:218-227`), the CTE core at `512.0` (`:450-455`). Pick a fixed, unused id
  for a pool you compose; grep the configs for collisions.
- **Pool ids are chosen by the caller** (client `custom_pool_id` or compose
  `pool_id`). `PoolManager::GeneratePoolId()` exists
  (`src/pool_manager.cc:649-659`) but nothing calls it. `CreatePool` rejects a
  null id with EINVAL unless a pool with the requested name already exists
  (`src/pool_manager.cc:937-960`).
- `ContainerId` is a `u32`. `kInvalidContainerId = (u32)-1`
  (`types.h:452`), `kStaticContainerId = (u32)-2` (`types.h:458`).
- **Container id == node id.** `CreatePool` reads
  `node_id = CLIO_IPC->GetNodeId()` (`src/pool_manager.cc:1011-1012`) and
  passes it to `Init`/`Restart` and `RegisterContainer` (`:1046-1063`). The
  address map is initialized as identity, `address_map_[i] = i`
  (`InitAddressMap`, `:689-719`), and the number of containers is the
  hostfile size, not the live host count (`:903-914`). The node id is the
  node's line offset in the hostfile (`src/ipc_manager.cc:1854-1860`). This
  is what makes `DirectHash` work: `container = hash % num_containers`, then
  container -> node via the address table. Migration / recovery change the
  map through admin `ChangeAddressTable` (WAL first, then
  `UpdateContainerNodeMapping`, `modules/admin/src/admin_runtime.cc:1530-1560`).
  (`ModuleManager::CreateContainer(chimod_name, pool_id, pool_name)` itself
  takes no container id, `src/module_manager.cc:191-200`; the id arrives via
  `Init`.)
- `MethodId`: the numbers in `clio_mod.yaml`. 0-9 are lifecycle methods
  (`kCreate: 0`, `kDestroy: 1`, `kRestart: 2` if the module supports it,
  `kMonitor: 9`; `-1` = not implemented), custom methods start at 10. See
  `module-anatomy.md`.
- `TaskId` (`task_id_`) identifies a task: pid, tid, major, node id and more.
  Subtasks share `[pid, tid, major, node_id]` with their parent, which is how
  lock reentrancy is detected (`types.h:368-371`). Clients make one with
  `clio::run::CreateTaskId()`.

## 3. Module discovery and the module name

- At startup `ModuleManager` **scans directories** for libraries whose names
  end in `_runtime` + the platform extension
  (`HasModuleNamingConvention`, `src/module_manager.cc:321-332`). Search order
  (`GetScanDirectories`, `:262-311`): the directory of `clio_run_cxx`,
  `CLIO_REPO_PATH` entries, `LD_LIBRARY_PATH`, `./lib`, `../lib`,
  `/usr/local/lib`.
- A valid module exports `alloc_chimod`, `new_chimod`, `get_chimod_name`,
  `destroy_chimod` (`ValidateChiMod`, `:341-350`). `CLIO_TASK_CC(Class)` at
  the end of the runtime `.cc` defines all four
  (`include/clio_runtime/container.h:970-990`).
- The registry key is the string from `get_chimod_name()`, which
  `CLIO_TASK_CC` takes from `Class::CreateParams::chimod_lib_name`. The first
  library loaded under a name wins; duplicates are skipped
  (`src/module_manager.cc:101-129`).
- So **`CreateParams::chimod_lib_name` is the module's identity**. It must
  equal the library basename without `_runtime` (e.g. `"clio_admin"`,
  `"clio_bdev"` (`modules/bdev/include/clio_runtime/bdev/bdev_tasks.h:228`),
  `"clio_cte_core"` (`context-transfer-engine/core/include/clio_cte/core/core_tasks.h:219`),
  `"clio_MOD_NAME"`), it is the `mod_name:` in compose, and clients pass it in
  every create. Never hardcode a module name string in a client: use
  `CreateParams::chimod_lib_name`. If you pass `LIB_NAME` to the CMake
  functions, it must produce the same basename.

## 4. The Create flow, end to end

### 4.1 Client side

Every module's `CreateTask` is an alias of the admin's
`GetOrCreatePoolTask<CreateParams>`
(`modules/admin/include/clio_runtime/admin/admin_tasks.h:520-522`; e.g.
`modules/MOD_NAME/include/clio_runtime/MOD_NAME/MOD_NAME_tasks.h:81`):

- `CreatePoolFields : Task` (`admin_tasks.h:110-256`) holds the
  module-independent fields: `chimod_name_` (INOUT), `pool_name_` (IN),
  `chimod_params_` (INOUT, the serialized `CreateParams`), `new_pool_id_`
  (INOUT), `error_message_` (OUT), flags `is_admin_`, `do_compose_`,
  `persist_`, and a non-serialized `ContainerClient* client_`.
- `BaseCreateTask<CreateParamsT, MethodId, IS_ADMIN, DO_COMPOSE>`
  (`:270-500`) adds no storage; `CreateTask` (admin's own, method `kCreate`),
  `GetOrCreatePoolTask<P>` (method `kGetOrCreatePool`) and `ComposeTask<P>`
  (compose mode) are aliases of it (`:506-533`).
- The emplace constructor builds `CreateParamsT(args...)` and serializes it
  into `chimod_params_` with `Task::Serialize` (a `priv::string` written by
  `ctp::ipc::GlobalSerialize`; `admin_tasks.h:307-312`,
  `include/clio_runtime/task.h:639-662`). `CreateParams` therefore needs a
  `template <class Ar> void serialize(Ar &ar)` covering every field.

Client code (`MOD_NAME_client.h:66-84`):

```cpp
Future<CreateTask> AsyncCreate(const PoolQuery &pool_query,
                               const std::string &pool_name,
                               const PoolId &custom_pool_id /*, params... */) {
  auto *ipc = CLIO_CPU_IPC;
  auto task = ipc->NewTask<CreateTask>(
      CreateTaskId(),
      kAdminPoolId,                    // ALWAYS the admin pool
      pool_query,                      // use PoolQuery::Dynamic()
      CreateParams::chimod_lib_name,   // never a literal string
      pool_name,                       // ALWAYS user-provided
      custom_pool_id,                  // fixed id for this pool
      this /*, params... */);          // client pointer for PostWait
  return ipc->Send(task);
}
```

Rules:

1. **Send to `kAdminPoolId`**, never to the client's `pool_id_`: the create is
   a `GetOrCreatePool` executed by the admin module.
2. **Use `PoolQuery::Dynamic()`** for creates (section 4.2 explains why).
3. **Pool names are user-provided.** Never generate a pool name from
   `pool_id_`: it is not set until the create completes.
4. **Use `CreateParams::chimod_lib_name`.**
5. **Check the return code** after `Wait()`; 0 is success.

When the future is destroyed after `Wait()`, `PostWait`
(`admin_tasks.h:478-499`, called from `Future::Destroy`,
`include/clio_runtime/ipc_manager.h:2686-2690`) copies
`new_pool_id_` into `client_->pool_id_` and `return_code_` into
`client_->return_code_` (it also forgives the run-to-run network-timeout rc
when a pool id came back). Tests also set it explicitly:
`client.pool_id_ = create->new_pool_id_;`.

### 4.2 Routing the create (Dynamic)

Dynamic is resolved by the **target pool's `ScheduleTask`** -- for a create
the target pool is the admin pool. `admin::Runtime::ScheduleTask`
(`modules/admin/src/admin_runtime.cc:193-219`), for `kGetOrCreatePool`:

- a pool with this **name** already exists locally -> `Local()`;
- the requested fixed **pool id** already exists locally -> `Local()` (added
  because a client asking for `clio_cte_core` at `512.0` while compose named
  the same pool `cte_core` otherwise broadcast a no-op create from every
  rank: N per node, N^2 cluster-wide);
- otherwise -> `Broadcast()` over the admin pool, so every node creates its
  container.

So with Dynamic, re-creating an existing pool (every client init does this)
costs one local task instead of a broadcast. (Older documents describe this as
"Dynamic routes to Monitor with `MonitorModeId::kGlobalSchedule`"; that
mechanism is gone -- `MonitorModeId` in `container.h:76-80` is unreferenced.)

### 4.3 Admin `GetOrCreatePool`

`modules/admin/src/admin_runtime.cc:241-315` (a coroutine):

1. If the target is a pool defined in the server's `compose:` section and
   this is not the compose task itself, wait (yield every 2000 us, up to
   300 s) until compose has created it (`:271-283`).
2. `CLIO_CO_AWAIT(pool_manager->CreatePool(task))` (`:288-289`). An exception
   yields rc 99 plus `error_message_`.
3. Broadcast replies are merged by `CreatePoolFields::AggregateOut`
   (`admin_tasks.h:222-255`): first non-null `new_pool_id_`, first error.

### 4.4 `PoolManager::CreatePool` (coroutine)

`src/pool_manager.cc:873-1138`:

1. Validate module name and pool name (EINVAL; `:661-687`, `:931-934`).
2. **Existing pool**: `FindPoolByName` (`:937-948`) or `HasPool(id)`
   (`:964-972`) -> `FillExistingCreateOut` (`:860-871`) -> calls the module's
   `Container::FillExistingCreate(task)` hook (`container.h:651`, default
   no-op) if the module name matches and this node has a local container;
   then return. On the by-name path `new_pool_id_` is overwritten with the
   existing id (`:940`).
3. Reject a null pool id (`:954-960`).
4. Record metadata; `InitAddressMap` (identity, section 2).
5. `EnsureStaticContainer` (`:276-333`): a container with id
   `kStaticContainerId`, on which only `Init` runs -- **never Create**.
6. Construct the container through `ModuleManager::CreateContainer`.
7. `Restart(pool_id, pool_name, node_id)` if compose says `restart: true` or
   the pool log is being replayed, else `Init(...)` (`:1033-1049`); then the
   ACL is configured.
8. `RegisterContainer` **before** Create (`:1061-1069`), so Create can submit
   tasks that find the container.
9. `CLIO_CO_AWAIT(container->Run(0, task))` -- the module's `kCreate` handler
   runs on the same task (`:1079`). Because this is an await, a Create (or
   any handler) can itself create pools and `CLIO_CO_AWAIT` them (nested
   creates). Example: the CTE's `RegisterTarget` creates a bdev pool and
   awaits it (`context-transfer-engine/core/src/core_runtime.cc:1496-1507`).
10. Non-zero rc from Create: unregister, destroy, erase metadata
    (`:1083-1091`).
11. Durable pools (non-admin with `persist_`, or compose `restart`) are
    written to the pool log (`:1116-1130`).

### 4.5 What your `Create` handler must do

- `Init` (autogenerated, see `module-anatomy.md`) already ran: base fields,
  `client_ = Client(pool_id)`, per-method model. Read your parameters with
  `task->GetParams()` / `GetParamsAs<CreateParams>()` (compose mode gives you
  a `PoolConfig` whose YAML is passed to `CreateParams::LoadConfig`,
  `admin_tasks.h:186-198`).
- Set `task->return_code_` (0 success) and `error_message_` on failure. A
  failed Create unregisters the container; it must never leave half-built
  state that later tasks dereference.
- **Optional resources must be optional.** If something can be absent (a
  cache with no segment, "serving tasks only"), every path that touches it
  checks it first. A failed optional setup must never become a segfault later.
- Create sub-pools you need with `CLIO_CO_AWAIT` on their `AsyncCreate`
  (Dynamic query, explicit pool ids/names) and check their return codes.
- If clients need output from Create (e.g. shared-memory keys), write it into
  the task's `chimod_params_` AND implement `FillExistingCreate` to write the
  same output for an existing pool. The CTE does exactly this:
  `WriteCreateOut` reserializes `CreateParams{config_, cache_keys_}` into
  `chimod_params_` at the end of Create and in `FillExistingCreate`
  (`context-transfer-engine/core/src/core_runtime.cc:947-961`); the client
  reads `Task::Deserialize<CreateParams>(task->chimod_params_).cache_keys_`
  (`core/include/clio_cte/core/core_client.h:669-683`). Without the
  existing-pool half, every client after the first silently gets nothing.
- `Restart` (generated when `kRestart: 2` is in `clio_mod.yaml`) sets
  `is_restart_ = true` and calls `Init`
  (`util/clio_run_cmd_refresh_repo.cc:387-395`); Create then recovers
  persistent state instead of starting empty.

## 5. Compose: pools from the server YAML

The server config's `compose:` list creates pools at runtime start
(`src/config_manager.cc:665-721` parses into `PoolConfig`,
`include/clio_runtime/config_manager.h:50-85`):

```yaml
compose:
  - mod_name: clio_bdev               # == CreateParams::chimod_lib_name
    pool_name: "ram::chi_default_bdev"
    pool_query: local                 # FromString: local, broadcast, dynamic, direct_id:N, ...
    pool_id: "301.0"
    restart: false                    # true -> Restart() path on start
    bdev_type: ram                    # every other key is module config:
    capacity: "0g"                    #   passed to CreateParams::LoadConfig
```

- Each node walks its own compose list serially at start and fails startup on
  a non-zero rc (`src/manager.cc:402-445`). On a plain `clio-run start`
  (recovery) compose pools are forced to `restart` and the address-table WAL
  and pool log are replayed (`src/manager.cc:417-455`); `start --fresh`
  discards the state.
- `admin::Client::AsyncCompose` sends `ComposeTask<PoolConfig>` to the admin
  pool with the entry's `pool_query` (`modules/admin/include/clio_runtime/admin/admin_client.h:200-212`).
- CLI: `clio-run compose start|stop|rm|list` (`util/clio_run_cmd_compose.cc`).
- Your `CreateParams` needs `LoadConfig(const PoolConfig&)` (or equivalent)
  to read its keys from the YAML node. **Every key it reads needs a default**
  and must be documented in `config/clio_default.yaml` (see SKILL.md, config
  duty).
- Interposer stacks (CTE caches etc.) are composed bottom-up: the lower pool
  first, then the interposer naming it as its next pool (see
  `cte-data-path.md`).
- Admin pool: created synchronously in `PoolManager::ServerInit` with pool
  name `"admin"` and `kAdminPoolId` (`src/pool_manager.cc:74-142`, name at
  `:106`). Exactly one admin pool per runtime; always name it `admin`
  (nothing enforces this; code that looks it up by name assumes it).

## 6. PoolQuery: every routing mode

Factories in `include/clio_runtime/pool_query.h` (each non-local factory takes
an optional `net_timeout`):

| mode | factory | resolves to (`IpcManager::ResolvePoolQuery`, `src/ipc_manager.cc:5737-5911`) |
|---|---|---|
| Local | `PoolQuery::Local()` (`:104`) | this node's container. **Default when unsure.** |
| Dynamic | `PoolQuery::Dynamic()` (`:163`) | the pool's `ScheduleTask` decides (6.1); unresolved -> Local |
| DirectId | `DirectId(container_id)` | Local if the container lives here, else `Physical(node of container)` |
| DirectHash | `DirectHash(hash)` | `container = hash % num_containers`; Local if here, else DirectId |
| Range | `Range(offset, count)` | one `Range(i,1)` per container (count 1 -> Local or DirectId) |
| Broadcast | `Broadcast()` | `Range(0, num_containers)`: every container |
| Physical | `Physical(node_id)` | that node, as is |
| ManyToOne | `ManyToOne(container_hash, batch_key, window_ns)` | collective: members batched at the neighborhood leader within a time window, folded with `AggregateIn`, run once, result broadcast back |
| AllToOne | `AllToOne(container_hash, batch_key)` | collective: leader waits until every container contributed (allreduce / barrier) |
| Null | `PoolQuery::Null()` (`:220`) | documented "do nothing", but no code treats it so: it resolves to itself and runs locally |

Rules:

- **Never use a null PoolQuery.** If you do not know, use `Local()`.
- A default-constructed `PoolQuery` is Local with `kInvalidContainerId`
  (`pool_query.h:83-88`).
- String forms (`FromString`, `src/pool_query.cc:139-226`): `local`,
  `broadcast`, `dynamic`, `direct_id:N`, `direct_hash:N`, `range:o:c`,
  `physical:N`, `many_to_one:h[:k[:ns]]`, `all_to_one:h[:k]`.
- A task whose pool cannot be resolved completes with
  `kRouteNullPoolRc = -1001` (`include/clio_runtime/ipc/ipc_run2run.h:61`).
- Broadcasts gather replies with your task's `AggregateOut` (see
  `module-anatomy.md`), one per container: an N-node broadcast is N messages
  and N aggregations.

### 6.1 How Dynamic is resolved

`IpcManager::RouteTask` (`src/ipc_manager.cc:5183`), called by workers and the
local submit path:

1. Collectives (ManyToOne/AllToOne) go to `RouteManyToOne` first (`:5216`).
2. A task already routed executes here (`:5221-5270`).
3. **Only for Dynamic**: `GetStaticContainer(pool_id)->ScheduleTask(task)`,
   and the result overwrites `task->pool_query_` (`:5280-5287`). Concrete
   modes are never re-scheduled; this prevents re-broadcast loops when the
   task arrives at a remote node.
4. `ResolvePoolQuery` turns the concrete mode into target containers/nodes;
   every resolved query records the return node.
5. Local targets run via `RouteLocal`; others go to the network queues.

Consequences for module authors:

- **`ScheduleTask` runs on the static container**, which only had `Init`
  called on it -- never Create. It must decide from the task, static config
  and global managers (pool manager, IPC), not from state Create builds.
- **The routing decision runs before your handler.** If a policy depends on
  *where* a task executes (e.g. write-back buffering on the writer's node vs
  executing on the blob's owner), put it in `ScheduleTask`. A "run locally"
  switch inside the handler is a no-op when routing already sent the task to
  the owner.
- The default `ScheduleTask` returns the query unchanged
  (`container.h:488-491`), so Dynamic on a module without an override runs
  Local. Overrides exist in admin, CTE core, the CTE interposers (cache,
  cache_base, shm cache, compressor) and the CAE core.
- A Dynamic task for a pool that has no static container on this node is not
  resolved and falls into the retry/"does not exist" path (`:5284`,
  `:5329-5345`).
- Leftover Dynamic on the send path maps to the local node
  (`src/ipc/ipc_run2run.cc:158-164`).

## 7. Method ids and dispatch

`clio_mod.yaml` assigns numbers; `clio-run refresh` turns them into
`Method::kFoo` constants and a generated `Runtime::Run` switch that casts the
task to `FooTask` and `CLIO_CO_AWAIT`s your `Foo(task)` handler
(`modules/MOD_NAME/src/autogen/MOD_NAME_lib_exec.cc:33-110`). The method id
travels in `task->method_`; on the network, `SaveTask`/`LoadTask` switch on the
same id to (de)serialize the right type. A mismatch between client and runtime
method numbering (a stale build, or a drifted yaml) dispatches the bytes to the
wrong handler. Details: `module-anatomy.md`.

## 8. Destroy, Monitor

- `admin::Client::AsyncDestroyPool(pool_query, target, flags)`
  (`admin_client.h:99-111`); `kDestroyPoolKeepRestartable = 0x1`
  (`admin_tasks.h:540`); a module's `DestroyTask` is an alias of
  `DestroyPoolTask` (`:795`). `PoolManager::DestroyPool`
  (`src/pool_manager.cc:1140-1174`) unregisters the pool's containers (saving
  their learned models), erases metadata and writes a pool-log removal.
  **Unverified / likely gap:** it does not call the module's `kDestroy`
  handler, so do not rely on `Destroy` running to release resources; make the
  container destructor safe too.
- `kMonitor = 9` is an ordinary method (no Container virtual). `MonitorTask`
  has IN `query_` and OUT `results_` (`std::map<ContainerId, std::string>`,
  `admin_tasks.h:1360-1387`); a module writes
  `task->results_[container_id_] = <msgpack bytes>`
  (`modules/MOD_NAME/src/MOD_NAME_runtime.cc` `Monitor`). The admin answers
  `worker_stats`, `pool_stats://`, `system_stats`, `bdev_stats`,
  `container_stats`, `get_host_info` (`admin_runtime.cc:807-827`).
