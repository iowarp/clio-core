---
name: clio-module-dev
description: Use when writing, changing, building, testing or optimizing Clio runtime modules (chimods) or any CTE / runtime code -- adding a module, method, task, client API or interposer (cache, shm cache, write-back/write-through), touching Create / pool ids / PoolQuery routing / ScheduleTask, the scheduler or cost model (GetTaskStats), coroutines, locks, GPU task submission, IPC transports, shared memory, a config key, CMake for modules, unit tests, or a performance fix. The primary reference for how clients, runtimes, tasks, pools and containers fit together, with the rules learned from real defects.
---

# Developing Clio modules and runtime / data-path code

This skill is the reference for everything module-related in this repository:
what a module is, how its client and runtime halves relate, how pools are
created and tasks routed, how the scheduler places work, how the CTE data path
works, and how to build, test and measure. Every rule here comes from the code
or from a real defect; reference files give `file:line` evidence.

Follow the checklists in order. Do not skip steps because a change "looks
small".

## Reference files (read the one you need)

| file | read it when |
|---|---|
| [architecture.md](architecture.md) | you need the big picture: processes, managers, how a task travels client -> worker -> handler -> back (SHM/TCP/IPC, futures, remote tasks), shared memory, IPC paths, cleanup, config and env vars |
| [module-anatomy.md](module-anatomy.md) | writing or reading a module: `clio_mod.yaml`, autogen (`clio-run refresh`), task structs (IN/OUT, Serialize, bulk, Copy/AggregateOut), client class, runtime class (Container API), `GetTaskStats`, CMake targets/aliases/linking |
| [create-and-ids.md](create-and-ids.md) | anything about Create / GetOrCreatePool, `kAdminPoolId`, pool ids, container ids (== node ids), module discovery and `chimod_lib_name`, compose, `PoolQuery` modes, how Dynamic is resolved (`ScheduleTask`), method dispatch, Destroy/Monitor |
| [scheduling-and-performance.md](scheduling-and-performance.md) | performance: worker layout, hold-time placement, the learned cost model, network lanes, worker loop and yield queues, inline ops, collectives, measured costs, fast-handler rules, measurement tools |
| [cte-data-path.md](cte-data-path.md) | CTE blobs/tags/targets, the two client paths, interposers, the shm cache (segment, flusher, invalidation, zero copy), write-back vs write-through, fsync |
| [locking-and-coroutines.md](locking-and-coroutines.md) | coroutine macros, awaiting futures, `yield`, `CoMutex`/`CoRwLock` semantics and rules |
| [gpu.md](gpu.md) | GPU producer-only model, kernel submission, GPU macro rules |
| [build-and-test.md](build-and-test.md) | building (presets, options, build-dir rules, rebuild-all), local build traps, `simple_test.h` unit tests, ctest, Aurora suites, Docker, wheels |
| [walkthroughs.md](walkthroughs.md) | step-by-step: add a module, add a method, change a config key -- each with a checklist |

## The model in one page

- A **module** (chimod) is two libraries from one directory: a **client**
  (`<pkg>_<mod>_client`) that applications link to build and send tasks, and a
  **runtime** (`<pkg>_<mod>_runtime`) that the runtime process loads to
  execute them. Both compile the same **tasks header**: the task structs are
  the contract (and the ABI) between them.
- A **pool** is a named, numbered (`PoolId` major.minor) instance of a module
  across the cluster; it has one **container** (your `Container` subclass)
  per node, whose **container id equals the node id**, plus a state-less
  **static container** used only for routing.
- A **method** is a numbered entry point (`clio_mod.yaml`, 0-9 lifecycle:
  `kCreate 0`, `kDestroy 1`, `kRestart 2`, `kMonitor 9`; custom >= 10;
  interposer verbs >= 100). `clio-run refresh` generates `Method::k*`, the
  `Run(method, task)` dispatch switch and all (de)serialization switches.
- The client does `NewTask<FooTask>(CreateTaskId(), pool_id_, query, ...)` and
  `Send`. The runtime routes the task by its **PoolQuery** (Dynamic is
  resolved by the pool's `ScheduleTask` before any handler runs), the
  scheduler picks a worker from the method's **predicted cost**
  (`GetTaskStats`), and the generated `Run` awaits your **coroutine handler**.
  `SerializeIn`/`SerializeOut` carry inputs/outputs across processes and
  nodes; `AggregateOut` merges multi-container replies; the client's `Future`
  completes.
- **Create** is a `GetOrCreatePoolTask<CreateParams>` sent to the **admin
  pool** (`kAdminPoolId`) with `PoolQuery::Dynamic()`; the admin's
  `ScheduleTask` makes it Local if the pool exists here, else Broadcast;
  `PoolManager::CreatePool` (a coroutine) builds the container, `Init`s it and
  awaits your `Create` handler -- which may itself create and await sub-pools.
  The module is found by `CreateParams::chimod_lib_name`.

## Core rules

### Before writing code

1. **Find the existing mechanism before adding one.** Grep for it. The
   scheduler, cost model, network lanes, shm cache, flush paths and
   collectives exist; most "missing features" are a missing *input* to them
   (a size, a config key, a routing case). Do not add flags, workers,
   threads or layers the design does not call for.
2. **Go through the abstraction you were asked to use.** If the request says
   "use the shm cache module", the change goes through it, even if a bypass
   measures faster today.
3. **Implement real code.** No mock or stub code unless explicitly asked. No
   commenting code out to make a build pass.
4. Work incrementally: one class or unit at a time, compile after each, then
   move on.

### Code style (all C/C++)

- Google C++ style. Keep code simple; **no function longer than 100 lines** --
  split into logically named helpers.
- **Every new function gets a doxygen docstring** (purpose, every `@param`,
  `@return`).
- **Every new `.h/.hpp/.cc/.cpp` starts with the IOWarp BSD 3-Clause header**
  (copy it verbatim from any existing source file; `CI/update_headers.sh
  --dry-run` previews and `CI/update_headers.sh` applies it to files missing
  it). Autogenerated files (`*autogen*`, `*_generated.*`, `*.pb.*`) and
  third-party code (`third_party/`, `external/`, `_deps/`) are exempt.
- **Store singleton pointers before use**: `auto *ipc = CLIO_IPC;
  ipc->Foo();` / `auto *x = ctp::Singleton<T>::GetInstance(); x->var_;` --
  never `GetInstance()->var_` or `*GetInstance()`.
- **Never a null PoolQuery.** If you do not know, use `PoolQuery::Local()`.
- **Name queue ids, priorities, flags and method ids** (`constexpr` /
  enums). Never raw integers.
- **All timing output in milliseconds, with units** (`"took 3.2 ms"`).
- No logging inside lock acquire/release paths.
- Fields shared between threads are atomic (`return_code_` via
  `GetReturnCode()`/`SetReturnCode()`).
- Warning-free builds: `size_t` for sizes, `(void)x;` for intentionally unused
  variables, no implicit narrowing.
- GPU code uses `CTP_IS_GPU`, `CTP_IS_HOST`, `CTP_IS_GPU_COMPILER`,
  `CTP_IS_CUDA_COMPILER`, `CTP_IS_ROCM_COMPILER` -- never raw `__CUDACC__`,
  `__HIPCC__`, `__HIP__`, `__CUDA_ARCH__`, `__HIP_DEVICE_COMPILE__` outside
  `context-transport-primitives/constants/include/clio_ctp/constants/macros.h`.
- Never hardcode `/tmp/clio_*`: use `ctp::SystemInfo::GetMemfdDir()`,
  `GetMemfdPath(name)`, `EnsureMemfdDir()`; IPC sockets are
  `GetMemfdPath("clio_" + std::to_string(port) + ".ipc")`.
- `CTP_MCTX` is now a no-op kept for compatibility
  (`macros.h:470-471`); do not add new uses, and do not remove an mctx that GPU
  code constructs explicitly.

### Module checklist

1. Methods live in `clio_mod.yaml` (modules listed in `clio_repo.yaml`).
   After adding a method run `clio-run refresh <repo dir>`; never hand-edit
   `autogen/`; never re-implement `SaveIn/LoadIn/SaveOut/LoadOut`,
   `NewCopy`, `Aggregate`, `Run`, `Init` or the Save/Load family in runtime
   sources. **Check manifest drift first** (the CTE core and shm cache yamls
   lack methods their generated headers have; refresh would delete them).
2. **Task structs are ABI**: after adding, removing or reordering a field,
   rebuild everything (no `--target`), including benchmarks and adapters that
   compile runtime headers.
3. Every field that must travel is in `SerializeIn`/`SerializeOut` (base
   first); payloads use `ar.bulk(ptr, len, BULK_XFER)` (outputs:
   `BULK_EXPOSE` in, `BULK_XFER` out). `IN/OUT/INOUT` are only annotations.
4. `AggregateOut` merges OUT fields only, never calls `Copy()`, never
   reassigns client-owned shared-memory members.
5. **Every method has a cost model** (`GetTaskStats`): data tasks scale
   `io_size_`, `compute_` and `wall_time_` with the payload
   (`cost::Data(bytes, ...)`; ~10 KB per CPU us, ~500 MB/s storage wall);
   control tasks return explicit constants (`cost::Metadata()`, ...), never
   zeros. Zeros make a multi-MB task look free: it piles onto one worker and
   rides the latency network lane. Interposers add cases for their own verbs.
6. **Create**: client sends to `kAdminPoolId` with `PoolQuery::Dynamic()`,
   `CreateParams::chimod_lib_name`, a **user-provided pool name** (never
   derived from `pool_id_`), a fixed pool id; check the rc. The admin pool is
   always named `admin` (one per runtime). The handler must work when optional
   resources are absent; outputs clients need must also be written for an
   existing pool (`FillExistingCreate`).
7. End the runtime `.cc` with `CLIO_TASK_CC(Runtime)`.

### Routing and coroutines

1. **Dynamic is resolved by the target pool's `ScheduleTask` before the
   handler**, on the static container (no Create-built state). Policies that
   depend on where a task executes (e.g. write-back buffering on the writer's
   node) belong in `ScheduleTask`; a handler-side "run locally" switch is a
   no-op once routing sent the task to the owner.
2. **Never hold a lock across a remote await.** Two nodes doing it to each
   other deadlock. Snapshot and release, or split work so remote waits hold
   only locks nobody remote needs. `CoMutex`/`CoRwLock` block the worker
   thread on contention (they do not suspend the coroutine), so avoid holding
   them across any await.
3. `CLIO_CO_AWAIT(clio::run::yield(us))` needs **us > 0**; `yield()` /
   `yield(0)` waits for an event and may sleep forever.
4. **Issue, then await**: start independent subtasks, then await their
   futures; never await inside the issuing loop. Never call `Wait()` inside a
   handler.
5. **Avoid broadcast-of-broadcast** (N^2 messages; one fsync once became
   32x32). Mark fanned-out requests (`kSyncFannedOut`) so receivers do only
   their local part. Batch per destination, never one message per page/item.
6. No blocking syscalls on workers (offload and poll); keep worker holds short
   (~100 us is the scheduler's budget).

### Data-path rules (CTE)

1. The client has **two paths**: the shared-memory cache path (when the shm
   cache is connected: copy between the caller's buffer and the segment -- no
   task, no runtime allocation, nobody waits) and the task path. `PutBlob`
   always tries the cache first. The client gets the segment keys from the
   CTE Create (also for an existing pool).
2. **No staging copies**: pass pointers to bytes already in shared memory and
   pin them until the send completes; wrap co-located caller pointers.
3. **Prefault** shared segments at creation (`SystemInfo::BulkFault`,
   `MADV_POPULATE_WRITE`); otherwise ~256 faults per MiB in the writer.
4. **Size defaults for the working set** and check every config that inherits
   a default you change (a `max_blobs` that did not fit a 256 MiB segment
   disabled the cache and crashed the runtime).

### Consistency policies

- **Write-back = eventual consistency.** Writes are buffered on the node that
  makes them; background flushes (timer, dirty bytes, fill level) send them
  on; invalidations go out per flush (batched per home), not per write and
  not per fsync. A read before a flush is not expected to be coherent.
  **fsync makes data visible and durable and must wait for the flush.**
- **Write-through = immediate visibility**, for data that needs it.
- Shared read-modify-write **metadata** (directory/inode blocks, `kMetaBlob`)
  is applied on its owner even under write-back.
- Tests match the policy: under write-back, verify after fsync.

### Config and docs duty

Every new or changed config key, environment variable or default -- for the
runtime, CTE, CAE, bdev or any module -- is documented in the same change:

1. `context-runtime/config/clio_default.yaml`: the key with its default (as a
   comment if it is not set there) and a comment describing it and its
   options.
2. The module's page under `site/pages/modules/` (its compose-key table).

### Building and testing

1. Build only inside a build directory; never in-source; never `rm -rf` a
   build directory (configure a new one alongside); no absolute paths in
   CMake; third-party builds go outside the repo and are removed after
   install. Use the `debug` preset by default.
2. A build directory that cluster jobs load binaries from is **live** while
   any of them is queued or running: check the queue first, compile-check
   elsewhere, rebuild it only between jobs.
3. `cmake --build <dir> -j N`; keep-going goes after `--` (`-- -k`). Confirm
   targets were built ("Built target X"). Install before re-running anything
   that uses the installed tree (`sudo` in the container). RPATHs mean
   `LD_LIBRARY_PATH` does not override libraries.
4. Unit tests that start the runtime use `simple_test.h` (never Catch2) and
   `clio::run::CLIO_INIT(RuntimeMode::kClient, true)`; check every Create's
   return code.
5. Locally red but CI green: stale build tree (`PutBlob` rc=11, no targets)
   -> new build dir; root-owned `/tmp` leftovers (`Permission denied`) ->
   chown. CI's `--repeat until-pass:3` and exclusions make green weaker
   evidence than it looks.
6. Multi-node correctness: `CI/aurora/run.sh --suite cache|posix --nodes 32`
   after cache, routing or consistency changes.

### Performance work

1. **Measure before explaining.** Do not present a hypothesis as the cause;
   separate what the data proves from what is unknown and add the measurement
   that settles it: per-method remote counts (`clio-evremote`, printed with
   `CLIO_EVLAT=1`), latency channels (`CLIO_EVLAT`, `evlat.h`), module
   counters printed at stop (e.g. `[shm-cache stats]`), and stack samples that
   cover the measured phase (make the phase long enough).
2. Change one thing per run, compare against a baseline in the same job,
   report single-run numbers with their noise.
3. Compare like with like (e.g. both with fsync when the baseline fsyncs).

## Further reading (optional, outside this skill)

The site has a longer module development guide
(`site/pages/docs/runtime/usage.html` and `extend.html`, published at
https://core.iowarp.ai/develop/). This skill takes precedence where they
differ: it was verified against the code.
