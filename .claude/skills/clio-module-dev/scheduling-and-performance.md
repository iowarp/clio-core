# Scheduling and performance

How the runtime decides where and when a task runs, what that costs, how to
write handlers that are fast, and how to measure instead of guess. Paths are
relative to `context-runtime/` unless they name another component
(`CTE/` = `context-transfer-engine/`). Line numbers were checked on branch
`cte-shm-cache`.

## 1. Workers and their roles

- `runtime.num_threads` (code default 4, `include/clio_runtime/config_manager.h:492`;
  env `CLIO_NUM_THREADS`) workers are created by `WorkOrchestrator::Init`
  (`src/work_orchestrator.cc:88-113`). One task lane per worker. Threads are
  spawned through `CTP_THREAD_MODEL->Spawn`; **workers are not pinned to
  cores** (no affinity calls in the runtime).
- `runtime.local_sched`: `"default"` (DefaultScheduler) or `"local"`
  (LocalScheduler); unknown names fall back to default
  (`src/scheduler/scheduler_factory.cc:43-53`).
- **DefaultScheduler layout** (`DivideWorkers`,
  `src/scheduler/default_sched.cc:88-171`), N workers:
  - N = 1: worker 0 does everything.
  - N = 2: worker 1 is the single network worker (send + recv).
  - N >= 3: network send = worker N-2, network recv = worker N-1.
  - N >= 5: GPU worker = N-3 (otherwise worker 0).
  - The **routable pool** (where ordinary tasks run) is worker 0 plus workers
    1 .. (N-3 or N-2, exclusive). With the default 4 workers that is workers
    0 and 1.
- The admin network periodics are pinned by method id: `kSend` (14) on the
  send worker; `kRecv` (15), `kClientRecv` (20), `kClientSend` (21) on the
  recv worker (`default_sched.cc:211-224`).
- Extra network threads outside the pool: `chi-peer-recv`, one
  `chi-peer-tier` per tier transport (`src/ipc/ipc_run2run.cc:1992-2019`),
  optional `chi-net-send` threads with `CLIO_NET_SEND_THREADS=N` (default 0).
- **Elastic workers**: when the pool is saturated the monitor grows it
  (`GrowPool`, `default_sched.cc:359-392`) up to `min(ncpu, 16)` extra
  workers; workers are never retired.

## 2. How a task is placed: `DefaultScheduler::RuntimeMapTask`

`src/scheduler/default_sched.cc:178-307`, called by `IpcManager::RouteLocal`
after routing has decided the task runs on this node:

1. **Task group affinity**: if `task_group_` is set and the container already
   bound that group to a worker, use it. First placement binds the group
   forever (`:182-194`, `:287-301`). Use groups to keep ordered work (e.g. one
   stream's operations) on one worker.
2. Admin network periodics -> their fixed workers.
3. `CLIO_SCHED_FUNNEL=1` (A/B testing only): everything to worker 0.
4. **Hold-time routing** (`:240-280`):
   - `hold = PredictHold(task, container)` = the container's learned
     `InferWallClockTime` of the task's `GetTaskStats`, capped at
     `kMaxBookedHoldUs = 1e6` us (`:315-328`,
     `include/clio_runtime/scheduler/default_sched.h:142-143`).
   - If the submitting worker is in the pool and has room -- its queued work
     (excluding its own running task) + hold <= `kMaxWorkerHoldUs = 100` us
     ("about one OS context switch", `default_sched.h:135-140`) -- run it
     there (no hand-off).
   - Otherwise `PickPoolWorker`: the pool worker with the lowest
     `RealtimeLoad` (`worker.h:345-362`: booked hold of executing tasks +
     queued hold + elapsed time of the current execution), scanning from a
     rotating index (`:330-347`). If even that worker lacks room the task
     still goes there and a new worker is requested.
   - The hold is booked onto the chosen worker's queue
     (`Worker::ReserveOn`, `src/worker.cc:1384-1390`).
5. Fallback: the current worker.

Consequences:

- **The prediction is the placement.** A 10 MB task that predicts 1 us is
  run inline on the submitting worker or queued behind others as if it were
  free; a burst of them lands on one worker. A tiny task that predicts 50 ms
  forces hand-offs and pool growth. Report real costs (section 3).
- A **parked task is not load**: while a coroutine waits on a future its hold
  is unbooked (`src/worker.cc:1546-1557`), so awaiting remote work does not
  make a worker look busy.
- **A non-yielding long task blocks its worker**: everything already queued on
  that lane waits behind it. The monitor (period 100 ms,
  `work_orchestrator.h:216`) detects stalled workers (> `kStallThresholdSec =
  1` s) and rescues their lanes, but that is a last resort measured in
  hundreds of milliseconds. Split long work or yield.
- LocalScheduler (`src/scheduler/local_sched.cc`) is simpler: group
  affinity, net periodics to one net worker, tasks submitted from net/GPU
  workers spread round-robin, otherwise stay on the current worker. No load
  balancing (stubs).

## 3. The learned cost model

- `TaskStat{io_size_, compute_, wall_time_}` (`include/clio_runtime/task.h:301-305`)
  from your `GetTaskStats` (see `module-anatomy.md` section 6 for how to write
  it and the `task_cost.h` rates/presets).
- Per container, per method (`include/clio_runtime/container.h:168-420`):
  - CPU prediction `InferCpuTime = a[m] * (compute_ + 1)` (`:291-297`).
  - Wall (hold) prediction `InferWallClockTime = b[m] * (wall_time_ + 1)`
    (`:305-311`). **This is what the scheduler books.**
  - Coefficients start at 1.0 and learn by SGD with
    `runtime.learning_rate` (default 0.2) from measured CPU time and from
    measured wall time per execution slice (suspensions excluded) in
    `Worker::EndTask` (`src/worker.cc:1615-1637`).
- `io_size_` is **not** an input to either prediction; it only selects the
  network lane (section 4). A data task must therefore scale `wall_time_` (and
  `compute_`) with its bytes -- `cost::Data(bytes, ...)` does all three.
- If `GetTaskStats` returns zeros (the default), both features become 1 and
  the coefficient degenerates into a running average of the method's hold: a
  single constant per method, blind to size. That is the defect class behind
  "a multi-MB task piled onto one worker".
- Learned weights persist per method name to
  `<conf_dir>/models/<chimod>.<pool>.<node>.<container>.yaml`
  (`src/task_stat_model.cc:81-92`), flushed at most every 30 s and at
  shutdown, restored when the pool is re-created (`src/pool_manager.cc:1224-1291`).
  If you change the *scale* of a method's features, old profiles mis-predict
  until relearned (this is why `kStorageBytesPerWallUs` must not change).

## 4. Network lanes

- Send priorities (`include/clio_runtime/ipc_manager.h:100-124`):
  `kSendInLatency`, `kSendInIO`, `kSendOutLatency`, `kSendOutIO`,
  `kClientSendTcp`, `kClientSendIpc`; `kNetQueueIoThreshold = 4096` bytes;
  per-tick IO byte budget `kNetQueueIoByteBudget = 8 MiB`.
- Requests: `RouteGlobal` puts a task with `PredictedStat().io_size_ >= 4096`
  on `kSendInIO`, otherwise `kSendInLatency` (`src/ipc_manager.cc:5688-5697`).
  Responses: `EndTask` uses `kSendOutIO` vs `kSendOutLatency`
  (`src/worker.cc:1724-1728`).
- The admin `Send` periodic (every 25 us) drains the latency lanes first,
  then alternates the IO lanes under the byte budget
  (`modules/admin/src/admin_runtime.cc:446-610`). So small control messages
  are not stuck behind 1 MiB payloads -- **if** the sizes are reported.
- **Caveat (found by reading the code, not verified at runtime):**
  `PredictedStat()` is only filled when a task starts executing on a node
  (`src/task.cc:158-162`). A request routed off-node before it ever executed
  locally appears to carry `io_size_ = 0` and ride the latency lane. Responses
  are classified correctly. If you are diagnosing lane behaviour, check this
  first.

## 5. Worker loop, queues, and what yield does

`Worker::Run` (`src/worker.cc:370-514`) repeats:

1. drain its client SHM shard (requests executed inline, up to
   `CLIO_SHARD_DRAIN_BUDGET`, default 256);
2. drain SHM sends;
3. `ProcessEventQueue` (resume coroutines whose awaited future completed;
   only the exact awaited future resumes a task, `:2034-2066`);
4. pop up to 16 new tasks from its lane, processing the event queue between
   tasks;
5. GPU tasks (one per poll, GPU worker only);
6. `ContinueBlockedTasks`: periodic, blocked and retry queues;
7. worker 0: flush due collective batches;
8. idle: spin-yield for `runtime.first_busy_wait` us (code default 1000),
   then sleep in epoll for up to the next periodic deadline (min 1000 us,
   doubling backoff, max 50 ms) (`SuspendMe`, `:1168-1350`).

Suspension rules (`src/worker.cc:1546-1570`, `task.h:1958-2030`):

- `CLIO_CO_AWAIT(future)`: the task parks with yield time 0 and is resumed
  only by the completion event. Correct for awaiting subtasks.
- `CLIO_CO_AWAIT(clio::run::yield(us))` with **us > 0**: the task goes to a
  periodic queue chosen by delay -- <= 50 us (scanned every 4 loop
  iterations, up to 1024 tasks per pass), <= 200 us (every 8, up to 1024),
  <= 50 ms (every 64, up to 8), > 50 ms (every 128, up to 8)
  (`:2121-2140`, `:2195-2213`, `:1870-1878`); a task fires when within 2 ms of
  its deadline (`:1913-1914`). A yielded task resumes on the same worker.
- `yield()` / `yield(0)`: **waits for an event and may never resume.** It is
  not "yield and come back next pass" -- that is `yield(1.0)`.
- Periodic tasks (`TASK_PERIODIC`, `period_ns_`) are re-routed every period;
  their cost seed should be `cost::IdlePeriodic()` when the common case is
  idle.
- Blocking syscalls (fsync, fallocate, long reads) must not run on a worker:
  offload them and poll. bdev's `BlockingOffload`
  (`modules/bdev/include/clio_runtime/bdev/blocking_offload.h`) runs them on
  16 helper threads and the task loops `CLIO_CO_AWAIT(yield(50))` until done
  (comment: blocking fsync/fallocate took 5-76 s on workers).

## 6. Inline execution

- `Container::InlineOp(method, in, out)` (`container.h:250-275`): a caller
  already on a worker, holding the in-process container, may run a pure,
  synchronous, any-thread-safe method directly, skipping a ~20 us task round
  trip. Only bdev implements it (`kAllocateBlocks`,
  `modules/bdev/src/bdev_runtime.cc:449`), used by the CTE core; skipping
  block allocation took 4 KiB puts from 39k to 91k IOPS (comment
  `container.h:255-262`). Never wire a method that awaits, does I/O, or
  touches per-worker state.
- `CLIO_RUN_INLINE(task)` (`include/clio_runtime/run_inline.h:73-110`): opt-in
  with `CLIO_ENABLE_INLINE_RUN`; requires the process to be the runtime, a
  single host, and the caller on a worker; disabled under `CLIO_UNIT_TEST`
  (set by `SIMPLE_TEST_MAIN`).

## 7. Collectives and batching

- `PoolQuery::ManyToOne(container_hash, batch_key = 0, batch_for_ns = 10000)`:
  submissions with the same (pool, method, hash, key) arriving within the
  window are parked at the neighborhood leader, folded with `AggregateIn`,
  executed once, and the result is copied back to every member.
- `PoolQuery::AllToOne(container_hash, batch_key = 0)`: the batch flushes when
  every container in the pool has contributed (allreduce / barrier
  semantics).
- (`include/clio_runtime/pool_query.h:178-214`,
  `include/clio_runtime/batch_manager.h:33-110`,
  `src/ipc_manager.cc:5442`; flushes run on worker 0, `src/worker.cc:448-453`.)
  Examples: `MOD_NAME` `ManyToOneSum`, `AllReduce`, `Barrier`.

## 8. Costs recorded in code comments (orders of magnitude)

| fact | where |
|---|---|
| 100 us worker hold ~ one OS context switch | `default_sched.h:136-140` |
| task round trip ~20 us | `container.h:255-262` |
| CTE put ~11-14 us server side, mostly task round trips | `run_inline.h:25-29` |
| SHM round trip ~5x faster than unix socket, ~190x faster than TCP | `src/ipc_manager.cc:256-258` |
| libzmq 64 B round trip 253 us vs blocking TCP 79 us | `src/ipc_manager.cc:2452-2453` |
| network send poll at 500 us: RTT ~650 us; at 25 us: ~310 us | `modules/admin/src/admin_runtime.cc:121-131` |
| worker loop ~1.3 us/iteration on a scheduler worker vs 82-157 us on net workers | `src/worker.cc:464-466` |
| event-queue wait averaged 20 ms (max 140) before the between-task drain was added | `src/worker.cc:687-694` |
| a 10 us yield loop that hopped workers made a 64 KB bounce take 17-22 ms | `src/worker.cc:1935-1943` |
| remote tasks waited 3-5 ms in lane 0 when all ran on worker 0 | `src/ipc_manager.cc:5238-5245` |
| `cudaMallocHost(1 GiB)` first write ~550 ms; pageable cold 12 / warm 18 / pinned 27 GB/s | `modules/bdev/include/clio_runtime/bdev/bdev_tasks.h:98-103` |
| GPU page fault ~110 us round trip vs ~6 us for the 256 KB copy it performs | `CTE/core/clio_mod.yaml` (batched POD paging comment) |
| ~256 page faults per MiB on first touch of an un-prefaulted segment | `CTE/core/include/clio_cte/core/shm_wtc_impl.h:76-79` |

No code comment gives a per-coroutine frame cost; the Boost backend exists to
avoid the per-task heap frame of stackless coroutines (`CMakeLists.txt:126`).

## 9. Writing fast handlers: rules

1. **Report real costs** for every method (`GetTaskStats`): payload-scaled for
   data tasks, deliberate constants for control tasks, never zeros.
2. **Issue, then await.** Start all independent subtasks, then await their
   futures. Awaiting inside the issuing loop serializes N round trips:

   ```cpp
   std::vector<Future<GetTask>> futs;
   for (auto &k : keys) futs.emplace_back(client_.AsyncGet(PoolQuery::Dynamic(), k));
   for (auto &f : futs) { CLIO_CO_AWAIT(f); /* use f->out_ */ }
   ```
3. **Batch per destination, not per item.** One message per page/range/key is
   the most common slowdown at scale. Group by owner node and send one task
   per owner (the CTE shm cache's per-home flushes and invalidation lanes are
   the model, `cte-data-path.md`).
4. **Avoid broadcast-of-broadcast.** A broadcast whose handler broadcasts
   again is N^2 messages (one fsync once became 32x32). Mark requests that
   were already fanned out (e.g. a `kSyncFannedOut` flag) so receivers do only
   their local part. Likewise use Dynamic + a `ScheduleTask` that turns
   "already exists here" into Local, so repeated creates do not broadcast.
5. **No staging copies.** Pass pointers to bytes that already live in shared
   memory; pin them until the send completes. A copy inside a single-threaded
   flusher or forwarder is a serial bottleneck.
6. **Prefault** every large shared segment at creation
   (`ctp::SystemInfo::BulkFault`).
7. **Never hold a lock across a remote await** (see
   `locking-and-coroutines.md`); never block a worker in a syscall; yield with
   a positive delay when polling.
8. **Keep handlers short and non-blocking** -- a worker that holds a task for
   more than ~100 us delays everything queued behind it. Split large work into
   subtasks or yield between chunks.
9. **Size defaults for the working set**: caches, logs, slot tables. Too small
   becomes refusals, retries and faults; too large for another configuration
   breaks it (a `max_blobs` that no longer fit a 256 MiB segment disabled the
   cache and crashed the runtime). Check every config that inherits a default
   you change.
10. **Use task groups** when ordering matters, not a lock.

## 10. Measuring

**Method:** do not present a hypothesis as the cause. State what the data
proves and what is unknown, then add the measurement that settles it. Change
one thing per run, compare against a baseline in the same job, report
single-run numbers with their noise, and compare like with like (e.g. both
with fsync when the baseline fsyncs). Make the measured phase long enough for
profiles to cover it.

| tool | how | what you get |
|---|---|---|
| `CLIO_EVLAT=1` | env var on the runtime/clients (`include/clio_runtime/evlat.h:226-232`) | per-channel latency at exit: `clio-evchan` (avg/max), `clio-evhist` (13 log buckets <10 us .. >=50 ms), `clio-evbusy` (net worker busy fraction per second), `clio-evlat S-C/C-P/P-F` (GPU submit phases), `clio-evqw` (per-worker event-queue wait). Channels (names in `src/ipc/ipc_gpu2cpu.cc:98-131`): remote RTT, event wait, net queue stages, client end-to-end, filesystem metadata (`fs_*`), data path / read-ahead (`rd_*`, `wr_*`, `gb_*`, `pb_*`, `ra_*`), DAOS, write-back sync (`wb_sync_*`), shm cache (`wtc_*`). `kill -USR1 <pid>` dumps mid-run (`:340-376`). Add a channel in `evlat.h` for a new stage. |
| `clio-evremote` lines (printed with `CLIO_EVLAT`) | `clio-evremote pool=X method=Y n=N` | per (pool major, method) count of tasks sent to other nodes (`src/ipc/ipc_gpu2cpu.cc:331-335`, recorded at `ipc_run2run.cc:374`); and per worker `clio-evremote worker=N n= exec_sum=ms` (`src/worker.cc:102-138`). The quickest way to find per-item messaging and broadcast storms. |
| module stats at stop | `RuntimeManager::AddStopHook` (`include/clio_runtime/manager.h:163-189`) | e.g. `[shm-cache stats] puts= refused: log= slot= buf= ... flush_batches= get_hits= get_misses= ...` at warning level. Add counters + a stop hook to any module you are optimizing. |
| scheduler logs | debug level, every ~1 s | `[#781 PDF]`, `[#781 METHODS]` (top methods by wall time), `[sched]` per-worker load/queued |
| `CLIO_WORKER_RATE=1`, `CLIO_NET_TRACE=1`, `CLIO_COLL_PROF` | env | worker loop rate, network send tick rate, collective profile |
| `clio-run monitor [-i N] [-j]` | CLI (`util/clio_run_cmd_monitor.cc`) | per-worker queued/blocked/periodic/retry/processed |
| viz dashboard | `CLIO_VIZ_ENABLE`, `viz:` YAML (default 127.0.0.1:8080) | pools, workers, config |
| stack sampling | `benchmarks/io500/sample_stacks.sh <outdir> [delay] [samples] [gap]` (`top -H` + `eu-stack` of `clio-run`/`clio_cte_fuse`; `PROFILE=1` in `pbs_io500_aurora.sh`); `CLIO_SUITE_GDB=1` runs suite daemons under gdb | where workers actually spend time |
