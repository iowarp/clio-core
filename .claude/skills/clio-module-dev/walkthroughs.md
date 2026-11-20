# Walkthroughs

Two step-by-step recipes, each ending with a checklist. Start from the
template module `context-runtime/modules/MOD_NAME/` (small, complete, tested)
and read `module-anatomy.md` and `create-and-ids.md` alongside.

## A. Add a new module

Example: a module `kvs` in the CTE repository, namespace `clio::cte`, with one
data method `Put` and one query method `Get`.

### 1. Decide the shape first

- What are the pool's containers responsible for? (Per-node state? Keys
  hashed to owners?) This decides the routing: `Local`, `DirectHash` via a
  `ScheduleTask` that hashes the key, or `Broadcast`.
- Is it an **interposer** in front of an existing pool (the CTE core)? Then
  derive from `CoreInterposer`, reuse the core's method ids for interposed
  verbs, and number your own verbs >= 100 (`cte-data-path.md` section 3).
- Which config keys does it need, and what are their defaults?
- Does it already exist? Grep for the mechanism first: the scheduler, cost
  model, network lanes, shm cache and flush paths exist; most "missing
  features" are a missing input to one of them (a size, a config key, a
  routing case), not a missing subsystem.

### 2. Create the directory

```
context-transfer-engine/kvs/
  clio_mod.yaml
  CMakeLists.txt
  include/clio_cte/kvs/{kvs_tasks.h, kvs_client.h, kvs_runtime.h}
  src/{kvs_client.cc, kvs_runtime.cc}
  test/{CMakeLists.txt, test_kvs.cc}
```

Every new `.h`/`.cc` starts with the BSD 3-clause header (copy it from any
existing source file, or run `CI/update_headers.sh`).

### 3. `clio_mod.yaml`

```yaml
module_name: kvs
namespace: clio::cte
version: 1.0.0
kCreate: 0
kDestroy: 1
kNodeFailure: -1
kRecover: -1
kMigrate: -1
kUpgrade: -1
kMonitor: 9
kPut: 10
kGet: 11
```

Add `- kvs` to `modules:` in `context-transfer-engine/clio_repo.yaml`.

### 4. Tasks header

- `CreateParams` with `chimod_lib_name = "clio_cte_kvs"` (=
  `<package>_<module>`, the runtime library basename without `_runtime`),
  defaulted fields, `serialize()`, and `LoadConfig(const PoolConfig&)` if it
  is composed from YAML.
- `using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;`
- `PutTask`, `GetTask` (names must be `<Method>Task`): default constructor,
  emplace constructor `(TaskId, PoolId, PoolQuery, inputs...)` setting
  `method_ = Method::kPut`, `SerializeIn`/`SerializeOut` calling the base
  first, payloads via `ar.bulk(...)`, `Copy`, and `AggregateOut` touching OUT
  fields only. Use `MonitorTask`/`DestroyTask` aliases as MOD_NAME does.

### 5. Generate autogen files

Build `clio_run` (or use an installed one) and run:

```bash
<build>/bin/clio-run refresh context-transfer-engine
```

This regenerates **every** module listed in that `clio_repo.yaml`. Before
running it, check for manifest drift (`module-anatomy.md` section 2: the CTE
core and shm cache yamls lack methods that their generated headers contain)
and afterwards review `git diff` of all `autogen/` files -- revert anything
you did not intend.

### 6. Client

`kvs_client.h`: `class Client : public clio::run::ContainerClient` with
`AsyncCreate` (to `kAdminPoolId`, `CreateParams::chimod_lib_name`,
user-provided pool name, `custom_pool_id`, `this`) and `AsyncPut`/`AsyncGet`
on `pool_id_` with sensible default queries (`Dynamic()` if `ScheduleTask`
picks the owner). `kvs_client.cc` defines
`constexpr const char *CreateParams::chimod_lib_name;`.

### 7. Runtime

`kvs_runtime.h/.cc`: `class Runtime : public clio::run::Container` with
`using CreateParams = ...;`, handler declarations matching the generated
`Run` switch, the generated-method overrides declared (copy the block from
`MOD_NAME_runtime.h`), and:

- `Create`: parse params, build state, create sub-pools with awaited creates,
  set `return_code_`; tolerate absent optional resources; if clients need
  outputs, write them to `chimod_params_` and implement
  `FillExistingCreate`.
- `Put`/`Get` handlers as coroutines (`CLIO_TASK_BODY_BEGIN`,
  `CLIO_CO_AWAIT`, `CLIO_CO_RETURN`, `CLIO_TASK_BODY_END`).
- `ScheduleTask` if Dynamic must resolve to an owner (decide from the task
  and static config only -- it runs on the static container).
- **`GetTaskStats` with a case for every method** (`cost::Data(size, ...)`
  for `Put`/`Get`, presets for Create/Destroy/Monitor, a deliberate default).
- `Monitor`, `Destroy`, `GetWorkRemaining`.
- `CLIO_TASK_CC(clio::cte::kvs::Runtime)` at the end of `kvs_runtime.cc`.

Keep every function <= 100 lines (split into helpers), give every new
function a doxygen docstring, follow the Google C++ style guide.

### 8. CMake

```cmake
add_clio_module_client(SOURCES src/kvs_client.cc)
add_clio_module_runtime(SOURCES src/kvs_runtime.cc src/autogen/kvs_lib_exec.cc)
if(CLIO_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
```

Add `add_subdirectory(kvs)` to `context-transfer-engine/CMakeLists.txt`
(optionally behind an `option(CLIO_CTE_ENABLE_KVS ... ON)` like the other CTE
modules). Targets: `clio_cte_kvs_client` / `clio_cte_kvs_runtime`, aliases
`clio::cte::kvs_client` / `clio::cte::kvs_runtime`. No absolute paths.

### 9. Config and docs

If the module is composed or has config keys: add a commented compose entry
with every key and its default to `context-runtime/config/clio_default.yaml`
(pick an unused fixed `pool_id`), and document the keys on
the module's page under `site/pages/modules/` in the same change.

### 10. Unit test

`test/test_kvs.cc` with `simple_test.h`: `CLIO_INIT(kClient, true)` once,
create the pool with `PoolQuery::Dynamic()` and a unique pool id/name,
`REQUIRE` the create rc == 0, set `client.pool_id_ = create->new_pool_id_`,
exercise Put/Get, `SIMPLE_TEST_MAIN()`. Register it once with `add_test(...)`
and a `TIMEOUT` (`build-and-test.md` section 7).

### 11. Build and verify

```bash
cmake --build <build> -j "$(nproc)"                # no --target
ctest --test-dir <build> -R kvs --output-on-failure
```

Confirm `Built target clio_cte_kvs_runtime` and that the runtime finds the
module (it scans for `*_runtime` libraries next to `clio_run_cxx`, in
`CLIO_REPO_PATH`, `LD_LIBRARY_PATH`, `./lib`, `../lib`, `/usr/local/lib`).

### Checklist: new module

- [ ] Searched for an existing mechanism; the module is needed
- [ ] `clio_mod.yaml` (ids 0-9 lifecycle, customs >= 10, interposer verbs >= 100) and `clio_repo.yaml` entry
- [ ] Autogen regenerated with `clio-run refresh`; autogen diff reviewed; no hand edits in `autogen/`
- [ ] `CreateParams::chimod_lib_name` == library basename; `CreateTask = GetOrCreatePoolTask<CreateParams>`
- [ ] Every task: default + emplace ctor, `SerializeIn/Out` (base first, every field that must travel, `bulk` for payloads), `Copy`, `AggregateOut` (OUT fields only, never `Copy()`)
- [ ] Client: `AsyncCreate` to `kAdminPoolId`, `Dynamic()`, user pool name, `chimod_lib_name`; other methods on `pool_id_`; no null queries
- [ ] Runtime: handlers are coroutines with the `CLIO_*` macros; `CLIO_TASK_CC` present
- [ ] `GetTaskStats` covers every method (payload-scaled data, explicit constants otherwise)
- [ ] `ScheduleTask` (if any) uses no Create-built state
- [ ] Create checks optional resources; outputs mirrored in `FillExistingCreate`
- [ ] No lock held across an await; yields use positive delays; independent subtasks issued before awaiting
- [ ] Named constants (no raw queue ids, priorities, flags); singleton pointers stored before use
- [ ] BSD header, doxygen docstrings, functions <= 100 lines, no stubs or mock code
- [ ] CMake via `add_clio_module_client/_runtime`, `add_subdirectory` added, no absolute paths
- [ ] `clio_default.yaml` + the module's page under `site/pages/modules/` updated for every key/default
- [ ] `simple_test.h` unit test checking every Create rc; registered in ctest
- [ ] Full build (no `--target`), warning-free, target confirmed built, tests pass
- [ ] Multi-node behaviour checked (Aurora suites) if it routes across nodes or affects caching/consistency

## B. Add a method to an existing module

Example: add `kStatKey` to `kvs`.

1. **yaml**: add `kStatKey: 12` to `clio_mod.yaml` (next free id; never
   renumber existing methods -- ids are wire ABI and persisted model keys are
   by name). For a CTE module, first reconcile any manifest drift (section 2
   of `module-anatomy.md`).
2. **Task**: add `StatKeyTask` to the tasks header (constructors,
   Serialize In/Out, `Copy`, `AggregateOut`).
3. **Regenerate**: `clio-run refresh <repo>`; review the autogen diff (the new
   case appears in `Run`, `SaveTask`, `LoadTask`, `AllocLoadTask`,
   `LocalLoad/Save`, `NewCopyTask`, `NewTask`, `AggregateOut/In`, and
   `Method::kStatKey` in the methods header).
4. **Runtime**: declare and implement `TaskResume StatKey(shared_ptr<StatKeyTask> &task)`;
   add a `case Method::kStatKey:` to `GetTaskStats`; if Dynamic routing
   applies, add a case to `ScheduleTask`. For an interposer: forward core
   verbs with `ForwardToCore`, give your verb its own stats.
5. **Client**: `AsyncStatKey(...)` (and a synchronous wrapper only if callers
   need one; not for use inside handlers). If it can be served from a shared
   segment, add the cache path with a task-path fallback.
6. **Interposers above this module**: if other modules interpose on this
   pool and must handle (or pass) the new method, add it to their yaml with
   the same id and regenerate them too; their default cases forward to the
   next pool.
7. **Config/docs**: any new key or default -> `clio_default.yaml` and the docs
   repo page.
8. **Test**: a `simple_test.h` case; under write-back, verify after fsync.
9. **Build everything** (task headers changed -> no `--target`), including
   benchmarks and adapters that include the header; run the module's tests.

### Checklist: new method

- [ ] New id appended in `clio_mod.yaml`; existing ids unchanged; drift reconciled
- [ ] `<Method>Task` complete (ctors, Serialize In/Out with bulk, Copy, AggregateOut)
- [ ] Autogen regenerated and diff reviewed
- [ ] Handler is a coroutine; issues-then-awaits; no locks across awaits; positive yields
- [ ] `GetTaskStats` case added (never zeros)
- [ ] `ScheduleTask` case added if Dynamic; routing-dependent policy lives there, not in the handler
- [ ] Client `Async*` method with a deliberate default query (never null; unsure -> Local)
- [ ] Interposers updated if they must see the method
- [ ] Broadcasts are not re-broadcast by receivers (mark fanned-out requests)
- [ ] Config + docs updated for any new key or default
- [ ] Unit test added; full rebuild; tests pass; multi-node suites if routing/caching changed

## C. Change a config key or default

1. Change the code default and the parse.
2. Update `context-runtime/config/clio_default.yaml` (the key, its default,
   and a comment saying what it does and its options -- even if it stays
   commented out).
3. Update the module's compose-key table under `site/pages/modules/`.
4. Grep every config in the repo (`*.yaml` under `context-*/config`,
   `benchmark`, `test`, `CI`, `docker`) that relies on the default and check it
   still makes sense (sizes that must fit a segment, counts that must fit a
   table).
