# Coroutines, waiting, and locking in runtime code

Paths are relative to `context-runtime/`. Line numbers were checked on branch
`cte-shm-cache`.

## 1. Handlers are coroutines

Every method handler returns `clio::run::TaskResume` and is written with the
portable macros (`include/clio_runtime/task.h:2122-2137`):

| macro | C++20 stackless (default) | Boost.Context fibers (`CLIO_CORE_ENABLE_BOOST_COROUTINES=ON`, forced for NVHPC) |
|---|---|---|
| `CLIO_TASK_BODY_BEGIN` / `CLIO_TASK_BODY_END` | empty | empty |
| `CLIO_CO_AWAIT(x)` | `co_await (x)` | `boost_await(x)`: suspend the task's fiber |
| `CLIO_CO_RETURN` | `co_return` | `return TaskResume{}` |

Always use the macros (never raw `co_await`/`co_return`) so the module builds
under both backends. Under the Boost backend each task runs on its own fiber
stack (`CLIO_BOOST_STACK_SIZE` KiB, default 256, min 64; `task.h:107-115`) and
nested helper coroutines run inline on it.

Helper coroutines: a helper that awaits must itself return `TaskResume` and be
awaited by the caller (`CLIO_CO_AWAIT(Helper(args))`). The generated
`Runtime::Run` does exactly this for each handler.

Stackless coroutine pitfalls:

- Parameters are references (`shared_ptr<FooTask> &task`); the coroutine frame
  holds the reference, which stays valid because the worker keeps the task
  alive while it runs. Do not capture references to locals of a caller frame
  that may finish before the awaited work.
- Locals live in the heap-allocated frame across awaits; large stack arrays
  become large frames.

`include/clio_runtime/co/` is a different thing: the GPU device coroutine
runtime ("clio-coroc"), not the CPU task machinery.

## 2. Waiting

### In client code (not on a worker)

```cpp
auto fut = client.AsyncFoo(PoolQuery::Local(), ...);
fut.Wait();                      // bool Wait(float max_sec = -1, bool reuse_task = false)
if (fut->GetReturnCode() != 0) { ... }
use(fut->result_);
```

`Future::Wait` (`include/clio_runtime/future.h:433`; implementation
`include/clio_runtime/ipc_manager.h:2539ff`): `max_sec = -1` waits forever,
`0` polls once and returns false if not complete, `> 0` is a timeout in
seconds. On the client it busy-waits/parks on the completion flag
(`fut_.is_complete_`, `task.h:311-334`) and then deserializes the outputs.
After a successful `Wait`, destroying the future runs the task's `PostWait`.

There is no `Task::Wait(...)` or `Task::Yield(...)` member any more (older
documents describe them); use the `Future` API and the awaitables below.

### In runtime handlers (on a worker)

Never call `Wait()` in a handler -- it would block the worker thread that has
to run the very tasks you wait for. Await instead:

```cpp
auto sub = client_.AsyncBar(PoolQuery::Dynamic(), ...);
CLIO_CO_AWAIT(sub);          // park until sub completes; worker runs others
```

- An awaited future parks the task with yield time 0; the completion event of
  **that exact future** resumes it (`src/worker.cc:2034-2066`; issues #705,
  #856 fixed resumption by unrelated events).
- A parked task's predicted hold is unbooked, so it does not count as worker
  load (`src/worker.cc:1546-1557`).
- **Issue, then await**: submit every independent subtask first, then await
  them all. Awaiting in the issuing loop turns N parallel requests into N
  serial round trips.

### Cooperative yield (polling)

```cpp
while (!Ready()) {
  CLIO_CO_AWAIT(clio::run::yield(50.0));   // positive delay in microseconds
}
```

- `clio::run::yield(double us = 0.0)` (`task.h:2030`). With `us > 0` the task
  is placed in a periodic/blocked queue and resumed after roughly `us`
  (queues by delay: <= 50 us, <= 200 us, <= 50 ms, > 50 ms; see
  `scheduling-and-performance.md` section 5).
- **`yield()` / `yield(0)` waits for an event and may sleep forever**
  (`task.h:1958-1962`, `src/worker.cc:1558-1564`). "Resume on the next pass"
  is `yield(1.0)`.
- Typical delays in the tree: 10 us barrier polls (shm cache), 25 us
  network send poll, 50 us offload polls, 2000 us waits for compose.

## 3. Locks: `CoMutex` and `CoRwLock`

### What they are (verified)

- `clio::run::CoMutex` (`include/clio_runtime/comutex.h:57-108`): a
  **reentrant wrapper around `ctp::Mutex`**, a fair ticket lock
  (`context-transport-primitives/thread/include/clio_ctp/thread/lock/mutex.h:79-135`).
  API: `Lock()`, `TryLock()`, `Unlock()`, RAII `ScopedCoMutex`.
- `clio::run::CoRwLock` (`include/clio_runtime/corwlock.h:64-150`): a
  reentrant wrapper around `ctp::RwLock`. API: `ReadLock`, `ReadUnlock`,
  `WriteLock`, `WriteUnlock`, RAII `ScopedCoRwReadLock`,
  `ScopedCoRwWriteLock`. Write->write and write->read reentrancy are allowed.
  Copy is deleted, move allowed.
- Reentrancy: a lock is re-acquired without blocking only when the calling
  **OS thread** already holds it **and** the current task's `LockOwnerId`
  matches (`comutex.h:71-80`). `LockOwnerId` = (worker id, pid, tid, major,
  node id) of the current task (`include/clio_runtime/types.h:368-410`,
  `src/types.cc:65-77`); subtasks share pid/tid/major/node with their parent,
  so a subtask of the holder running on the same worker can re-enter. The
  `holder_tid_` gate fixed a data race (#807).
- **Contention does not suspend the coroutine.** `ctp::Mutex::Lock` spins
  with an escalating host backoff (CPU pause, then `std::this_thread::yield`
  for ~2 ms, then sleep; `mutex.h:186-250`). The waiting *worker thread* is
  blocked; no other task runs on it meanwhile. (Older documents describe these
  locks as "an `atomic<bool>` + `Task::Yield`"; that is no longer the code.)

### Rules

1. **Never hold a lock across a remote await.** If node A holds lock X and
   awaits a task on node B, while B holds lock Y and awaits a task on A that
   needs X, both deadlock. Snapshot what you need, release, then send; or
   split the work so remote waits only hold locks nobody remote can need.
   (Real case: the write-back cache flushes foreign-owned blobs before local
   ones so that a lock on a locally-owned blob is never held while waiting on
   another container, `context-transfer-engine/cache_base/src/cache_base.cc:1002-1006`.)
2. **Avoid holding a `CoMutex`/`CoRwLock` across any `CLIO_CO_AWAIT`.**
   Reasoning from the implementation above (not separately tested): while the
   holder is parked, another task that contends the lock blocks its whole
   worker thread; if that worker is the one that must resume the holder (a
   parked task is resumed on the worker it ran on), nothing makes progress.
   When a critical section must span an await, use a coroutine-friendly lock
   that yields instead (e.g. a per-key "busy" flag polled with
   `yield(us > 0)`, as the CTE cache's per-blob lock does with
   `kLockPollUs`, `cache_base.cc:219-230`), or restructure.
3. **Keep critical sections short and copy-free.** Payload memcpys should
   hold a pin (a reference count that blocks eviction), never a lock (the shm
   cache's rule, `context-transfer-engine/core/include/clio_cte/core/shm_wtc.h:50-55`).
4. **No logging in lock paths.** Do not add `HLOG`/print statements inside
   `Lock`, `Unlock`, `TryLock`, `ReadLock`, `ReadUnlock`, `WriteLock`,
   `WriteUnlock` (or your own lock implementations): it degrades performance
   severely and can deadlock when the logger takes a lock.
5. Lock ordering: when you must take several per-key locks, take them in
   sorted key order (the write-back cache's land path does,
   `cache_base.cc:1396-1480`).
6. Prefer **task groups** (`task_group_`, `scheduling-and-performance.md`
   section 2) to serialize ordered work on one worker without locks.

## 4. Shared state and thread safety

- Several workers run handlers of the same container concurrently. Counters
  and flags touched from handlers must be atomic (MOD_NAME's `custom_count_`
  was a TSan-reported race until made `std::atomic`,
  `modules/MOD_NAME/include/clio_runtime/MOD_NAME/MOD_NAME_runtime.h:125-128`).
- Task fields read by other threads are atomic in the base class
  (`return_code_`, `completer_`). Use `GetReturnCode()`/`SetReturnCode()`
  (or `.load()`/`.store()`), and initialize with `store(0)` in constructors.
- `ScheduleTask` runs on the static container, possibly concurrently on many
  threads, and must not depend on state built in Create.

## 5. Futures across processes

- A `Future<T>` owns a reference to the task. Read OUT fields only after
  `Wait()` returned true (or after the await resumed).
- Fire-and-forget: a future that is never waited returns immediately from
  `Wait` (`ipc_manager.h:2548`); do not use it for anything whose completion
  or result matters.
- Streaming outputs use the task's `is_new_data_` flag (`task.h`) -- see
  `MOD_NAME` `TestLargeOutput` and `test_streaming.cc` for the pattern.
