# Project: Migrating Blocked / In-Flight Tasks Off a Stalled Worker (issue #785)

Follow-up to **#781** (merged in #782). Branch `785-blocked-task-stealing`, based on
`origin/dev` @ `6b2913a3`.

---

## 1. Where #781 left us

#781 stopped a mislabeled / non-yielding task from wedging the runtime, but only for **new**
work: `RuntimeMapTask` routes on measured `RealtimeLoad()`, and the `WorkOrchestrator` monitor
thread calls `Scheduler::LoadBalance()` every 500 ms and detects a worker stuck > 1 s inside one
`ExecTask` (`default_sched.cc:331`).

Still stubs in the tree:

* `default_sched.cc:388` — `SpawnAdditionalWorker()` + steal the stalled lane.
* `default_sched.cc:404` — `StealWork()` returns `false`.
* `work_orchestrator.cc:432/439` — `SpawnAdditionalWorker()` / `RetireWorker()`.

So today a bad task strands **one thread plus everything already committed to it**. New tasks flow
around the wedge; in-flight work behind it does not.

---

## 2. What is stranded (verified code map)

Five worker structures are reachable only from that worker's own thread:

| # | State | Where | Why it is stuck |
|---|---|---|---|
| 1 | Lane backlog | `assigned_lane_` (`worker.h:475`), SHM MPSC ring | Wedged worker is the ring's only consumer |
| 2 | Blocked queues | `blocked_queues_[4]` (`worker.h:490`), `std::queue` | No lock; owner thread only |
| 3 | Periodic queues | `periodic_queues_[4]` (`worker.h:524`), `std::queue` | Same — **and this is the most severe row**, see §2.3 |
| 4 | Retry queue | `retry_queue_` (`worker.h:497`), `std::queue` | Same |
| 5 | Completion events | `event_queue_` (`worker.h:508`), process-local MPSC ring | Producers push from anywhere, but only the wedged worker pops (`ProcessEventQueue`, `worker.cc:1017`) |

Plus a sixth category that is not merely stranded but **invisible**: a task suspended on `co_await`
(`IsYielded() && YieldTimeUs() == 0`) is placed in **no queue at all** — `ExecTask`
(`worker.cc:720-733`) deliberately skips `AddToBlockedQueue` for it. Its only owner is the subtask
future's `parent_task_` handle (`future.h:138`, set in `ipc_cpu2self.cc:60`). The runtime cannot
enumerate its own suspended tasks. **D4 fixes this by parking them in the blocked queue**, which
folds category 6 into category 2 and makes the whole set enumerable by one mechanism.

### 2.1 Why they are pinned

`IpcCpu2Self::SendOut` (`ipc_cpu2self.cc:109-123`):

```cpp
if (!parent_task.IsNull() && parent_task->EventQueue()) {
  auto *parent_event_queue = reinterpret_cast<...>(parent_task->EventQueue());
  parent_event_queue->Emplace(task_ptr->RunFuture());
  if (parent_task->Lane()) CLIO_IPC->AwakenWorker(parent_task->Lane());
}
```

`EventQueue()` / `Lane()` are raw addresses stamped into the parent's `RunContext` the first time
it was popped off a lane (`worker.cc:429-432`) and never revisited. A completed subtask therefore
cannot wake its parent when that worker is wedged, even though the subtask ran to completion on a
healthy worker. Same design already produced the #680 lost-wakeup class — hence the two
`AwakenAllWorkers()` fallbacks in `IpcManager::AwakenWorker` (`ipc_manager.cc:750-793`).

### 2.2 `blocked_queues_` is currently vestigial

Worth stating plainly, because D4 depends on it. `AddToBlockedQueue`'s `YieldTimeUs() == 0` branch
(`worker.cc:1151`) is the one that feeds `blocked_queues_`, and essentially nothing reaches it:

* `ExecTask` (`worker.cc:725`) only calls it when `YieldTimeUs() > 0` — i.e. always the *periodic*
  branch.
* `ProcessBlockedQueue`'s own re-add (`worker.cc:951`) is **dead code**, unreachable behind a
  `continue` at line 947.
* `ReschedulePeriodicTask` (`worker.cc:1268`) reaches it only for a periodic task whose period is 0.

So `blocked_queues_[4]`, its four-bucket backoff, and `WorkerStats::num_blocked_tasks_` are
scaffolding for a case that never populates them. D4 puts them to their intended use rather than
adding a parallel registry.

`periodic_queues_[4]`, by contrast, is very much alive — and is where the runtime's own
infrastructure lives.

### 2.3 Periodic tasks are the highest-stakes stranded state

The periodic queue holds the runtime's **network and client-IPC pollers**. Both schedulers
hard-route admin periodic methods to specific worker roles (`default_sched.cc:167-195`,
`local_sched.cc:126-136`):

```
14 = kSend       peer DEALER pool          -> net_send_worker_
15 = kRecv       peer ROUTER (9413)        -> net_recv_worker_
20 = kClientRecv client-facing ROUTER      -> net_recv_worker_
21 = kClientSend same client-facing ROUTER -> net_recv_worker_
```

Consequences the plan has to answer for:

* **Severity.** If `net_recv_worker_` stalls, its stranded pollers mean the node stops receiving
  peer traffic *and* stops servicing client IPC. One bad task becomes a whole-node outage. This is
  worse than a stalled compute worker, and the plan previously treated periodic as a lesser sibling
  of blocked.
* **The routing table itself is being deleted.** D8 replaces these roles with dedicated-worker
  periodic tasks placed by load, so the pollers become migratable like anything else. The
  socket-ownership constraint the table encodes does *not* go away — it has to be re-expressed, and
  that is most of what D8 is about.

### 2.4 Measured, not assumed *(P0 complete — commit 05b7a357)*

`context-runtime/modules/MOD_NAME/test/test_stall_migration.cc` reproduces the failure against
unmodified code, using only existing primitives (`Custom` with `spin_us_` is already a non-yielding
spin; `WaitTest` with `depth > 1` already self-sends a subtask and `CLIO_CO_AWAIT`s it).

**Result — 4 I/O workers, 2 spinners at 8 s, 3 s deadline:**

```
control (flat, dependency-free)   0/8 pending    <- runtime demonstrably alive
chained (co_await)                6/8 pending    <- the bug, isolated
tasks dropped (30 s deadline)     0              <- nothing lost, only stalled
```

Three things this settles:

1. **The bug is real and is about dependencies, not load.** Flat tasks submitted at the same instant
   all complete; chained ones do not. #781's routing works exactly as designed and is simply not
   sufficient for anything that blocks.
2. **Nothing is dropped.** Given a generous deadline every task eventually completes. This is a
   *liveness/latency* defect, not a task-loss defect — which narrows the whole issue. The
   correctness machinery in D3 (no lost wakeups) is guarding against a regression we would otherwise
   introduce, not repairing existing loss.
3. **A control group is mandatory in any test here.** The first run of this test used 8 spinners and
   showed 8/8 chained pending — which looked like a dramatic repro and was actually just saturation.
   Without the control it is impossible to tell the two apart, and the wrong conclusion is the
   flattering one.

### 2.5 The default configuration has ONE general-purpose worker

Found while making the repro valid, and it reframes P1. `DivideWorkers` at the default
`num_threads = 4` produces:

```
worker 0  scheduler worker
worker 1  I/O worker        <- the ONLY general-purpose compute worker
worker 2  net_send_worker
worker 3  net_recv_worker
```

So in the default configuration **a single non-yielding task wedges 100% of compute capacity**, and
*migration cannot help at all* — there is nowhere to migrate to. Every task-moving mechanism in this
document is inert until a spare compute worker exists.

That makes `WorkOrchestrator::SpawnAdditionalWorker()` — still a `return nullptr` stub from #781 —
the load-bearing piece, not a supporting one. **P1 is not "backlog rescue"; it is the precondition
for the entire issue.** It also means any test here must raise `CLIO_NUM_THREADS` (the fixture sets
8) or it measures saturation regardless of what the scheduler does.

---

## 3. Decisions taken

Recorded because they shape everything below:

* **D-a — `LoadBalance()` is the only migrator.** No idle-thief work stealing in this issue.
  Migration happens on the monitor thread, at 500 ms cadence, triggered by stall detection.
* **D-b — Migration is triggered by stall, and moves everything.** When a worker is detected
  stalled, all of its pending and blocked tasks move to a replacement worker.
* **D-c — Migrated tasks get their event-queue pointer updated.** Signal redirection is done by
  rewriting the address in the task, not by making the address unnecessary.
* **D-d — The event queue gets a consumer mutex**, so a foreign thread can drain it.
* **D-e — Net-worker roles are deleted in favour of `TaskGroup` affinity.** Tasks sharing a ZMQ
  socket share a group; the group binds to one worker, and that worker is dedicated to the group.
  Three groups — peer-send, peer-recv, client — following socket ownership. Reuses the existing group
  machinery rather than adding a task flag. See D8.

This is a deliberately smaller design than a lock-free late-binding scheduler, and the
single-migrator constraint is what makes it tractable: **there is exactly one writer**, so no
migrator/migrator races exist and no per-task CAS state machine is needed. What remains is a
two-party problem (migrator vs. signalling producer), solved by D3 below.

---

## 4. Design

### D1 — Consumer mutex on the event queue: use `mpmc_ring_buffer`

The runtime already has the exact primitive. `ctp::ipc::mpmc_ring_buffer`
(`ring_buffer.h:790`) is *identical* to `mpsc_ring_buffer` except `RING_BUFFER_LOCK_POP`
serialises `Pop()` behind a `ctp::Mutex`. So D-d is a one-line type change:

```cpp
// worker.h:508
ctp::ipc::mpsc_ring_buffer<Future<Task, ...>, ctp::ipc::MallocAllocator> *event_queue_;
// becomes
ctp::ipc::mpmc_ring_buffer<Future<Task, ...>, ctp::ipc::MallocAllocator> *event_queue_;
```

Cost is one uncontended lock per **drain batch**, not per element, provided `ProcessEventQueue`
keeps its `while (Pop(...))` loop. The event queue is process-local (`MallocAllocator`), so an
in-process mutex here carries none of the SHM robustness concerns that apply to the lane (D5).

The lock is safe to take from the monitor thread because a wedged worker is inside `ExecTask` and
therefore provably not inside `Pop()`. **Invariant: never hold this lock across `ExecTask`** — pop
a batch, release, then execute.

### D2 — Park mutex on the blocked / periodic / retry queues

One `std::mutex park_mtx_` per worker guarding `blocked_queues_[4]`, `periodic_queues_[4]`,
`retry_queue_`. Taken by the owning worker in `AddToBlockedQueue` / `AddToRetryQueue` /
`ContinueBlockedTasks`, and by the migrator. Contention is negligible — these are touched at
park/unpark boundaries and a migration is a 500 ms-cadence event.

Same invariant: the owner pops a task under the lock, **releases**, then calls `ExecTask`. This is
what guarantees a wedged worker never holds a lock the migrator needs.

Migration splices whole `std::queue`s rather than moving element-by-element.

### D3 — The producer-side handshake (the piece the consumer mutex alone does not cover)

**A consumer mutex makes the drain safe, but it does not make the pointer swap safe.** The
signalling producer runs on a third thread and does an unsynchronised read of
`parent_task->EventQueue()`. Interleaving:

```
producer (worker B)                    migrator (monitor thread)
-------------------------------------  ------------------------------------
q = parent->EventQueue()   // OLD
                                       lock old->event_mtx_
                                       drain OLD -> NEW
                                       parent->SetEventQueue(NEW)
                                       unlock
q->Emplace(future)         // into OLD, now orphaned
AwakenWorker(parent->Lane())
```

The event lands in a queue nobody will ever drain again → the parent sleeps forever. This is the
#680 failure mode reintroduced by a different route, so it must be closed explicitly.

**Protocol.** Give each task a `sig_mtx_` (or a spinlock — it is per-task and essentially
uncontended) plus a `sig_gen_` counter incremented on every re-point.

*Producer* (`IpcCpu2Self::SendOut`):

```cpp
u32 g0; EventQ* q; TaskLane* lane;
{ std::lock_guard lk(parent->SigMtx()); q = parent->EventQueue();
  lane = parent->Lane(); g0 = parent->SigGen(); }

q->Emplace(future);                      // OUTSIDE the lock — see below

{ std::lock_guard lk(parent->SigMtx());
  if (parent->SigGen() != g0) {          // migrated under us
    q = parent->EventQueue(); lane = parent->Lane(); repush = true; } }
if (repush) q->Emplace(future);          // duplicate is harmless — §6.3

CLIO_IPC->AwakenWorker(lane);
```

*Migrator*, per task, **before** draining the old queue:

```cpp
{ std::lock_guard lk(task->SigMtx());
  task->SetEventQueue(new_q); task->Lane() = new_lane; task->BumpSigGen(); }
```

Order matters: **re-point every task first, then drain the old queue.** Anything pushed to the old
queue before its task was re-pointed is picked up by the drain; anything pushed after is caught by
the generation re-check. Between them the window is closed.

The `Emplace` deliberately happens **outside** the lock. `mpsc_ring_buffer::Emplace` with
`RING_BUFFER_WAIT_FOR_SPACE` claims a tail slot and then **spins forever** if the ring is full
(`ring_buffer.h:509-521`) — holding `sig_mtx_` across that would let a full orphaned queue
deadlock the migrator, converting a one-thread stall into a two-thread stall. See §7.2.

### D4 — Park `co_await`-suspended tasks in the blocked queue

**Decision:** the `YieldTimeUs() == 0` yield path in `ExecTask` (`worker.cc:720-733`) now calls
`AddToBlockedQueue`, so an event-waiting task is parked in `blocked_queues_[]` instead of vanishing
into the subtask future's `parent_task_` handle. The blocked queue becomes the enumerable set of
suspended tasks — for migration (D-b), for `WorkerStats`, and for diagnostics. No parallel registry.

The one-line version of this is wrong, though. `ProcessBlockedQueue` (`worker.cc:902-953`) today is
a *resume-once-and-forget* scan, and both halves of that are hostile to event-waiting tasks:

1. **It resumes unconditionally.** It pops a task and calls `ExecTask(task, is_started)` with no
   check that what the task is waiting on has actually happened (`worker.cc:944`). An event-waiting
   task resumed mid-`co_await` returns from the await while the subtask is still running and reads
   its outputs early — precisely the #705 failure (short reads/writes, freed buffers).
2. **It never re-queues.** The re-add at `worker.cc:951` is dead code behind the `continue` at 947,
   so a popped task leaves the queue permanently. Tracking would be lost on first scan.

So `ProcessBlockedQueue` is rewritten as a **readiness scan**:

```
for each parked task (bounded batch):
  if (task->IsCoroCompleted())            -> erase        // already finished elsewhere
  else if (awaited future is complete)    -> erase, resume, LOG(kWarning)
  else                                    -> leave parked, bump miss count / backoff bucket
```

The middle branch is a genuine bonus: it is a **self-healing net for lost wakeups**. Normally
`ProcessEventQueue` resumes the parent long before the scan sees it, so reaching that branch means
an event was dropped — which is exactly the #680 class, and it should be loud rather than silent.
The existing four buckets (`% 2/4/8/16`) become the backoff for how often a long-waiter is
re-checked, which is what they were shaped for.

Supporting changes:

* **`std::queue` → `std::list` + an iterator stored on the task**, so `ProcessEventQueue` can erase
  a task in O(1) before resuming it. Without O(1) erase, a resumed task lingers as a stale entry
  and can be double-parked when it suspends again.
* **`AddToBlockedQueue`'s `wait_for_task` parameter is deleted.** It currently means "do not add"
  (`worker.cc:1145`) — the exact behaviour being reversed — and no caller passes `true`.
* **Delete the dead `continue` + unreachable re-add** at `worker.cc:946-951`.
* **Awaited-future lifetime.** The scan needs to test completeness of what the task awaits.
  `RunContext::awaited_fshm_` is a raw `const void*` (`task.h:885`). Dereferencing it is safe *while
  the parent is suspended* (the parent's coroutine frame holds the owning `Future`, which holds the
  `shared_ptr<FutureShm>`), but that is an invariant worth making explicit — consider storing an
  owning `Future` instead of the raw pointer. See open decision 1.
* **Ownership.** Parking adds a second owning `shared_ptr<Task>` reference. A task that completes
  through a path which never erases its entry is retained forever by the queue. The
  `IsCoroCompleted()` branch above makes this self-correcting, but given #620/#680 were both
  reference-lifetime bugs, the leak-detection tests should be run against this specifically.

### D4b — Periodic tasks: the migration silently undoes itself unless the role moves too

`ProcessPeriodicQueue` (`worker.cc:955-1015`) does not merely resume a due task — it **re-routes**
it every period: `CLIO_IPC->RouteTask(task->RunFuture())`, executing only on `ExecHere`. That has a
consequence the rest of the plan has to respect:

> Migrate a role-pinned periodic task to a replacement worker, and on its very next period
> `RouteTask` → `RuntimeMapTask` sends it **straight back to the stalled worker's lane**, because
> the routing table keys on role, not on health. The rescue reverts itself within one period, and
> the symptom is a rescue that appears to succeed and then quietly stops working.

This splits periodic tasks in two, and they need opposite treatment:

**Non-pinned periodic tasks — migrate the queue, placement self-heals.** `RuntimeMapTask` routes
these by measured `RealtimeLoad`, so they will naturally avoid the stalled worker. All they need is
*somebody to scan them*. Moving `periodic_queues_[]` to a worker that runs `ProcessPeriodicQueue` is
sufficient; the existing per-period re-route does the placement. This is the cheapest correct fix in
the whole plan.

**Role-pinned periodic tasks (admin 14/15/20/21) — do not migrate; re-point the role, or neither.**
Moving the task without moving the role reverts (above); moving the role without moving the socket
is UB (§2.3). So a stalled net worker is *not* rescuable by task migration at all — see D7.

Periodic-specific mechanics for whichever tasks do move:

* **`Lane()` must stay valid.** `ReschedulePeriodicTask` (`worker.cc:1247-1251`) **silently drops**
  a periodic task whose `Lane()` is null — `if (!lane) return;`, no log, task gone. Lane transfer
  (D5) keeps the lane object alive and valid, which is another argument for it over lane draining.
* **Reset `BlockStart()` on migration.** A periodic task stranded 10 s behind a wedged worker has
  `elapsed >> yield_time`, so it fires the instant it is scanned — and a whole migrated queue fires
  simultaneously. `AddToBlockedQueue` already has staleness handling for this
  (`worker.cc:1179-1185`, the >10 ms reset); migration should reuse it rather than invent a rule.
* **`ProcessPeriodicQueue` is the model for D4's readiness scan.** It already does exactly the right
  shape — check whether the task is due, resume if so, **re-queue if not** (`worker.cc:1010-1013`).
  D4's rewrite of `ProcessBlockedQueue` is that same loop with "awaited future is complete"
  substituted for "deadline reached". Framing it as *make blocked look like periodic* keeps the
  change small and well-precedented.

### D5 — The lane: transfer it, do not lock its pop path

Symmetry would suggest making the lane `mpmc` too. Recommend **not**:

* `TaskLane` is `multi_mpsc_ring_buffer<Future<Task>, CLIO_QUEUE_ALLOC_T>::ring_buffer_type`
  (`task.h:766`) — it lives in **shared memory** and external client processes push into it.
  `LOCK_POP` puts a `ctp::Mutex` in SHM on the hottest path in the runtime, where a killed holder
  wedges the lane permanently.
* Transferring the lane costs nothing on the hot path and is simpler:
  1. `Worker *fresh = work_orch_->SpawnAdditionalWorker();`
  2. `fresh->AdoptLane(stalled->assigned_lane_)` → `lane->SetTid(fresh->GetTid())` so
     `AwakenWorker`'s `tgkill` (`ipc_manager.cc:771`) reaches the new consumer.
  3. Hand the stalled worker a fresh empty lane; quarantine it so the mapper places nothing on it.
  4. Requires `assigned_lane_` → `std::atomic<TaskLane*>`, re-read at the top of each `Run()`
     iteration (`worker.cc:273`), so the wedged worker sees the new lane when it finally returns.

This also **resolves #781's open "steal-safe MPSC pop" question** by making it moot.

Bonus: because the lane *object* moves with its tid, `task->Lane()` stays valid for every migrated
task — only `EventQueue()` genuinely needs re-pointing, halving the mutable state in D3.

### D6 — Load accounting must move with the tasks

Migrated tasks carry `sched_reserved_us_` reservations counted in the stalled worker's
`queued_load_us_` (#781). If they are not released from the old worker and re-reserved on the new
one, the stalled worker looks permanently loaded (harmless) and the replacement looks idle
(harmful — the mapper will pile new work onto it). Also update `SetRunWorkerId`.

### D7 — Migratability is a property of the worker's role

Not every worker can be rescued by moving its tasks. This has to be explicit or the rescue path
will do something unsafe.

| Role | Migratable? | Why | Rescue strategy |
|---|---|---|---|
| `scheduler_worker_` | Yes | General purpose | Lane transfer + task migration |
| `io_workers_` | Yes | General purpose | Lane transfer + task migration |
| `net_send_worker_` | **Deleted** | Becomes a dedicated-worker task — D8 | Migratable once D8 lands |
| `net_recv_worker_` | **Deleted** | Same | Migratable once D8 lands |
| GPU worker | **No** (probably) | Bound to `gpu_lanes_`; `SuspendMe` early-returns because "GPU workers must never sleep" (`worker.cc:483-486`) | Spike §7.6 |
| Worker 0 | Special | Hardcoded `worker_id_ == 0` drives `BatchManager::FlushDue` (`worker.cc:288`) | Make it a role pointer so the duty can move |

`LoadBalance` branches on this before rescuing. After D8 the non-migratable set shrinks to the GPU
worker (pending §7.6) and worker 0's batch duty, which is the point: **role gating is a temporary
scaffold, not the destination.**

Note this also dissolves what was an open decision — `DefaultScheduler::PickAltWorker`
(`default_sched.cc:302-319`) deliberately falls back to the net workers, which made "protect them by
prevention" unworkable. With no net worker to protect, the hole closes by construction.

### D8 — Delete the net-worker roles; express socket ownership as a TaskGroup

**Decision:** remove `net_send_worker_` / `net_recv_worker_` as scheduler roles. The pollers declare
a **TaskGroup**; the existing group-affinity machinery pins the group to one worker, and that worker
is dedicated to the group.

This mechanism already exists and is already half-wired for exactly this purpose:

* `TaskGroup { int64_t id_ }` on every task (`task.h:205`), documented as *"Tasks in the same group
  are pinned to the same worker once routed."*
* `Container::task_group_map_ : unordered_map<int64_t, Worker*>` + `task_group_lock_`
  (`container.h:107-109`).
* `RuntimeMapTask` checks it **first** and returns the bound worker early (`default_sched.cc:152-165`,
  `local_sched.cc:117-122`), then inserts the binding after routing (`default_sched.cc:284-291`).
* `SendTask` and `RecvTask` **already set** `task_group_ = TaskGroup(0)` — *"Network tasks in
  affinity group 0"* (`admin_tasks.h:660,729`).

So most of what the previous `TASK_DEDICATED` draft proposed is already in the tree. Groups give us
sticky binding and socket co-location for free, which were the two hard parts. What is missing is
exclusivity, rebinding, and pointer hygiene.

#### D8-1 — The group map already silently defeats the role table *(bug, present on dev)*

Worth fixing regardless of this issue. `RuntimeMapTask` consults the group map **before** the
periodic role table, and the insert is first-writer-wins (`if (it == end || it->second == nullptr)`).
`SendTask` and `RecvTask` are both in group 0. Therefore:

> Whichever of `kSend` / `kRecv` is routed first binds group 0 to *its* role worker. The other one
> then hits the group early-return and follows it there — **bypassing its own role assignment**.

The documented send/recv split — *"keeping [outbound sends] off the recv worker means inbound SWIM
probe responses can still be polled"* (`default_sched.cc:176-181`) — is therefore already not
happening. That makes deleting the role table much less risky than it looks: we are removing a
mechanism that the group map has already overridden. It also means any "regression" measured against
today's behaviour is a comparison against an accident, not against the design.

#### D8-2 — Three groups, following socket ownership *(decided)*

```c
// admin: group ids are Container-scoped, so these only need to be unique
// within the admin pool. Named constants, not the bare TaskGroup(0) in use today.
kGroupNetPeerSend = 0   // kSend        — peer DEALER pool
kGroupNetPeerRecv = 1   // kRecv        — peer ROUTER (9413)
kGroupNetClient   = 2   // kClientRecv + kClientSend — client-facing ROUTER
```

Groups follow **socket ownership**, which is the actual constraint
(`default_sched.cc:169-171`). Three groups, not four: `kClientRecv` and `kClientSend` share the
client-facing ROUTER, so they must stay in one group — that is a correctness requirement, not a
packing preference.

Consequences:

* A wedged peer poller leaves client IPC alive, and vice versa. Under a single shared group, one
  wedge would have been the §2.3 whole-node outage in concentrated form.
* Restores the send/recv split that D8-1 shows is currently broken — outbound DEALER sends can
  back-pressure on ZMQ HWM without starving inbound SWIM probe polling.
* Costs one thread more than today (3 vs 2), same shape as the role table it replaces. Degrades
  under core pressure via D8-5.

**`ClientConnect` stays ungrouped.** It is spawned as a periodic alongside the four pollers
(`admin_runtime.cc:156`) but it is a request *handler*, not a socket poller — it fills response
fields (`GetServerGeneration`, allocator ids, worker tids) and touches no transport. It routes
normally and needs no affinity. Verified rather than assumed, because a fifth periodic quietly
sharing a socket would be a latent race under any grouping.

#### D8-3 — Exclusivity: the group binding needs a dedicated worker

Groups pin tasks *to* a worker; they do not keep other work *off* it. Add exclusivity to the binding
rather than to the task:

```cpp
struct GroupBinding { Worker *worker_; bool exclusive_; };
std::unordered_map<int64_t, GroupBinding> task_group_map_;
```

`RuntimeMapTask` skips exclusive-bound workers in its least-loaded scan; `LoadBalance` spawns a
worker when an exclusive group has none of its own. Decide `exclusive_` **once, at first binding**,
from the group's declared `TaskStat` — which is precisely what `TaskStat`'s own doc comment says it
is for: *"when a task belongs to a TaskGroup, these stats describe the characteristics of the entire
group"* (`task.h:228-231`), and `Runtime::GetTaskStats` already reports the net methods with
`io_size_ = 1 MiB` and an inflated `compute_` (`admin_runtime.cc:1870-1886`).

Note this also **retires my earlier objection to cost-driven exclusivity**. That objection was to
*continuous* repulsion via `RealtimeLoad`, which the SGD-fit `coef` in `InferCpuTime` erases within
minutes as an `EAGAIN`-returning poller measures ~0 CPU. A **one-time bind-time** decision reading
the declared `TaskStat` is immune — the learner never gets a vote. So "set the compute high enough
and let it get its own worker" works, exactly as you framed it, provided the decision is made once
at bind time and recorded in the binding.

#### D8-4 — The binding must become mutable (this is the trap)

`task_group_map_` is **insert-if-absent and never updated**. Nothing anywhere rebinds a group. That
collides head-on with migration, and in the worst way:

> The group early-return is the **first** thing `RuntimeMapTask` does. Migrate a group's periodic
> tasks off a stalled worker and, on the very next period, `RouteTask` → `RuntimeMapTask` reads the
> stale binding and sends them **straight back to the stalled worker** — before any load, role or
> health check runs.

Same self-undoing failure as D4b, but harder to spot because the early return precedes everything.
`LoadBalance` must therefore own the binding: rebind on stall, rebind on retire, and take
`task_group_lock_` in write mode to do it. This is now a first-class part of the migration design,
not a detail.

**Also: `Worker*` in that map is raw.** With an elastic pool that retires workers (P1), a retired
worker leaves a dangling pointer in every container's group map. `RetireWorker` must sweep the group
maps of all containers, or bindings must hold a worker id rather than a pointer. Worker id is safer
and is the recommendation.

#### D8-5 — Degradation and limits

* **Core pressure.** N exclusive groups means N threads. Let `LoadBalance` collapse exclusive groups
  onto a shared worker when workers exceed cores. Safe by inspection: it is today's behaviour, where
  `net_recv_worker_` hosts several pollers — provided a *group* is never split across workers, which
  is exactly the invariant the group map already enforces.
* **Group ids are container-scoped.** The map lives on the `Container`, so ids only need to be unique
  within a pool. Give them named constants rather than the bare `TaskGroup(0)` in use today.
* **Honest limitation — this contains a stall, it does not cure one.** If a task wedges *inside its
  own handler* (say `Recv` blocking on a socket), it cannot be migrated: migration moves only
  *parked* tasks, and starting a second instance is forbidden by D8-6. What is gained is that no
  other work queues behind it, and other groups survive. Curing a wedged poller needs handler-level
  timeout with provably sane socket state — out of scope. Nobody should read "LoadBalance gives it a
  worker" as "networking always recovers".

#### D8-6 — "Exactly one live instance" becomes load-bearing

ØMQ permits moving a socket between threads given a full barrier and no concurrent use. Groups give
us the "no concurrent use" half — one group, one worker — and rebinding happens between periods, so
period N finishes before period N+1 starts elsewhere. **But this now rests on an invariant that is
currently true and unasserted**, and two pieces of code depend on it silently, both justified by the
pinning we are deleting:

* `Runtime::Recv`: *"No socket lock needed - single net worker processes all Recv tasks."*
* `Runtime::ClientSend` (`admin_runtime.cc:652`) keeps a function-local
  `static std::vector<shared_ptr<Task>> deferred_deletes`, mutated with no lock. Two concurrent
  instances would be a data race on a `shared_ptr` vector — a double-free, the #680 shape.

Assert one live instance per net periodic method, and audit both sites as part of the change.

**What this buys.** Deletes the hardcoded method-id routing from *both* schedulers; removes the
non-migratable role class from D7; fixes D8-1; and converts the §2.3 failure — a stalled net worker
taking the node's networking down with no rescue possible — into an ordinary migratable stall.

---

## 5. Phases

### P0 — Deterministic repro + telemetry *(prereq, no behaviour change)*

* Extend `clio_run_thrpt_bench --test-case sched_variety` with a **dependency-chain** class: a
  `Custom` task that self-sends a subtask and `co_await`s it, while a sibling `spin_us_ = 10s`
  task wedges a worker. Today's `sched_variety` is flat (independent tasks), which is exactly why
  the in-flight stall does not appear in the #781 numbers.
* Success criterion for the whole issue: **chained-task p99 must not track the spin duration.**
  Measure before touching anything.
* Counters: parked / suspended per worker, event-queue depth, stalls detected, rescues performed,
  tasks migrated. Wire into `WorkerStats` + the `LoadBalance` 5 s telemetry dump.

### P1 — Replacement workers + lane transfer (D5)

* Real `WorkOrchestrator::SpawnAdditionalWorker()`: construct `Worker` + lane, register in
  `all_workers_` / `worker_threads_`, `thread_model_->Spawn(Run)`. `Worker::Run` already does its
  own per-thread setup (`CLIO_IPC->GetTls()`, `AddSignalEvent`, `lane->SetTid`,
  `worker.cc:247-260`), so a late-spawned worker is self-initialising.
* `assigned_lane_` → atomic; `AdoptLane`; quarantine flag; `RetireWorker` with idle-cooldown
  hysteresis.
* Wire the rescue into `LoadBalance`'s existing stall branch (`default_sched.cc:388`).
* **Clears stranded category 1.** Unstarted lane tasks have no coroutine state, so this phase
  needs no cross-thread-resume guarantee beyond what the runtime already does (§7.1).

**STATUS: P1 DONE — commit 35fba7a6.** `SpawnAdditionalWorker` is real; `LoadBalance` transfers a
stalled worker's lane to a fresh worker. Verified 5/5 runs: chained tasks 6/8 stranded → 0/8, 4
rescues fired, no regression in `wait_functionality` (4/4) or `comutex` (13/13).

Implementation notes worth carrying forward:

* **No lane headroom exists.** The TaskQueue is constructed with `num_lanes = num_threads`
  (`ipc_manager.cc:1021`) — the `+1` in `CalculateQueueSegmentSize` is sizing only. So an elastic
  worker cannot be given a lane. It does not need one: the rescue is a *swap*. This is what makes
  D5 the right call rather than a convenience.
* **Append safety without a lock.** `all_workers_`/`workers_`/`worker_threads_` reserve
  `kElasticHeadroom` at Init so a spawn is a pure append that cannot reallocate under readers on
  other threads; `GetWorker`/`GetWorkerCount` are bounded by an atomic published count rather than
  `size()`. Hitting the cap is a hard stop, never a reallocation.
* **Rescue once per stalled worker**, guarded on `GetLane() != nullptr`. Without that guard the
  500 ms monitor spawns a thread per tick for the entire life of the bad task.
* **`RetireWorker` stays a deliberate no-op**, per §7.7 — retiring a worker whose tid
  `ClientConnect` published leaves clients addressing a dead mailbox, which is task loss strictly
  worse than the thread it reclaims.

**Scope check — what P1 did NOT fix.** The passing test exercises the **lane backlog** category
only: the chained tasks' subtasks were queued behind spinners, and moving the lane freed them. The
event-queue, blocked, periodic and retry categories are untouched, and no test isolates them yet.
The next test must suspend a parent on a worker and wedge that worker *afterwards*, so the stall
lands on the completion path rather than the lane. That needs a subtask slow enough to keep the
parent suspended — `CustomTask` needs a `chain_depth_` so a Custom can self-send a spinning Custom
child and `co_await` it.

### P2 — Migration of parked + suspended state (D1–D4, D6)

Split into two mergeable steps, because D4 is independently testable and carries the #705 risk:

**P2a — park event-waiting tasks (D4), no migration yet.** `ExecTask` parks them;
`blocked_queues_` → `std::list` + task-held iterator; `ProcessBlockedQueue` becomes the readiness
scan; `ProcessEventQueue` erases before resuming; delete `wait_for_task` and the dead re-add.
Behaviour-preserving by design — nothing should resume differently — so any change in the FUSE /
stress suites here is a real regression and is easy to attribute. Lands the lost-wakeup safety net
and fixes `WorkerStats::num_blocked_tasks_` on its own.

**P2b — dissolve the net-worker roles into TaskGroups (D8).** Give `kClientRecv`/`kClientSend` a
group (they have none today); add `exclusive_` to the group binding and key it off the declared
`TaskStat` at bind time; make the binding **mutable** and hold a worker *id* not a pointer (D8-4);
delete the method-id routing tables from `default_sched.cc` and `local_sched.cc` along with
`net_send_worker_`/`net_recv_worker_`; drop the net fallbacks from `PickAltWorker`; assert
one-live-instance per net periodic. Standalone and independently valuable — it fixes the live D8-1
bug and simplifies #781's routing, and it happens to unblock the rescue. Land it before P2c so
periodic migration never has to special-case a role or fight a stale binding.

**P2c — periodic queue migration (D4b).** Migrate `periodic_queues_[]` under `park_mtx_`; reset
`BlockStart()`; D7 gating for whatever remains non-migratable (GPU, worker 0). Needs no event-queue
changes at all — the per-period `RouteTask` self-heals placement — so it is the cheapest large win
available and it clears the most severe stranded category (§2.3).

**P2d — blocked/suspended migration.** `event_queue_` → `mpmc_ring_buffer`; `sig_mtx_`/`sig_gen_`
and the D3 producer handshake; `LoadBalance::MigrateAllFrom(stalled, fresh)` + re-drain of
quarantined queues (§7.2); retry queue; load accounting moves (D6).

* **Clears stranded categories 2–6.** This is what stops one bad task from stalling an unbounded
  dependency tree.
* Requires §7.1 closed, since started fibers now move deliberately.

### P3 — Observability + quarantine policy

* Suspended-duration histogram, migration counters surfaced through `clio_run_cmd_monitor`.
* Wedged-worker readmission policy (open decision 3).

---

## 6. Correctness

### 6.1 Why the monitor thread can take a wedged worker's locks

Both `event_mtx_` (inside the mpmc ring) and `park_mtx_` are, by construction, never held across
`ExecTask`. A wedged worker is by definition *inside* `ExecTask`, so it holds neither. The
migrator therefore acquires promptly. Belt and braces: use `try_lock` with a bounded attempt and
skip to the next `LoadBalance` tick on failure — a missed rescue tick costs 500 ms, a blocked
monitor thread costs the whole safety net.

### 6.2 Ordering proof for the re-point

Let *T* be a migrated task and *P* a producer signalling *T*'s completion.

* If *P* releases `sig_mtx_` (first critical section) **before** the migrator acquires it, *P*
  read the OLD queue. The migrator's re-point happens after, and the drain happens after that, so
  *P*'s `Emplace` — which completes before *P*'s second critical section, which itself must wait
  for the migrator's release — is either already in the old queue when drained, or *P* observes
  `sig_gen_` changed and re-pushes to the new queue. Either way the event is delivered.
* If *P* acquires **after** the migrator's re-point, it reads the NEW queue directly.

No third case exists, because both parties serialise on the same per-task lock.

### 6.3 Duplicate events are already tolerated

The D3 re-push can deliver the same future twice (once via the drain, once via the re-push).
`ProcessEventQueue` already skips events whose future is not the parent's `AwaitedFshm()`
(`worker.cc:1056`, the #705 guard) and events whose parent `IsCoroCompleted()`
(`worker.cc:1042`, the "orphan events from parallel subtasks" case). A duplicate hits one of those
two guards. **Verify this explicitly with a test** rather than relying on the reading — it is now
load-bearing where before it was defensive.

### 6.4 Double execution after migration

A task can be simultaneously in a blocked queue *and* the target of an event — today that is
benign because both resumes happen on one thread and the second finds `IsCoroCompleted()`. After
migration both still happen on the *new* worker's single thread, so the property is preserved.
The migrator must move (not copy) under `park_mtx_`, so the old worker finds its queues empty when
it un-wedges.

### 6.5 `EndTask`'s #680 ordering

`EndTask` (`worker.cc:864-899`) orders `break_self_cycle()` against `SendOut` differently for the
in-process-client case vs. the subtask case, specifically to avoid a `shared_ptr<Task>`
double-release found by TSan. P2 changes what the subtask branch of `SendOut` does, so that
ordering must be re-derived, not assumed. **TSan run required on the P2 diff.**

### 6.6 Lost wakeups

`AwakenWorker` stays unconditional-`tgkill` (`ipc_manager.cc:759-769` — do **not** reintroduce a
park-flag gate; that regression is documented). Its two `AwakenAllWorkers()` fallbacks should
become unreachable once lanes republish their tid on transfer; keep them but log at `kWarning`, so
they act as a bug detector for this issue.

---

## 7. Risks and spikes

### 7.1 Cross-thread resume of a started fiber — *probably already happens* (gates P2)

`BoostStackPool()` (`boost_stack_allocator.h:65`) is a `SlabAllocator` with a **per-thread reuse
cache**; a stack allocated on worker A and freed on worker B lands in B's cache. Establish:

1. Does `ctp::ipc::SlabAllocator` tolerate a foreign-thread `Free` (no owner assertion, no
   intrusive header written by the owning thread)? **Spike this first.**
2. Is cross-thread resume already happening? Strong evidence it is: `ProcessRetryQueue`
   (`worker.cc:1225`) re-routes with `force_enqueue=true`, which can land a **started** task on
   another worker's lane, where `ProcessNewTask` (`worker.cc:440`) calls
   `ExecTask(task, is_started=true)`. If confirmed, the audit validates existing behaviour rather
   than introducing new risk. Confirm with a targeted test; do not assume.

The C++20 stackless backend heap-allocates frames and is thread-agnostic; only Boost needs this.

### 7.2 Full event queue — accepted, keep blocking `Emplace` *(closed)*

`Emplace` under `WAIT_FOR_SPACE` claims a tail slot then busy-loops until the consumer advances
(`ring_buffer.h:509-521`). Decision: **leave it as is.** The argument that makes it safe:

* The push is **outside** `sig_mtx_` (D3), so a spinning producer can never block the migrator.
  This is the constraint that must not be relaxed; if the push is ever moved inside the lock, a full
  orphaned queue converts a one-thread stall into a two-thread stall.
* The migrator drains the old queue as part of the re-point sequence, so a producer that arrives
  with a stale pointer finds space and completes, then catches the generation bump and re-pushes.
* Residual window: enough producers to refill a just-drained queue before any of them re-checks the
  generation. Closed cheaply by having `LoadBalance` **re-drain a quarantined worker's event queue
  on every subsequent tick**, not just at rescue time. No protocol change, ~10 lines.

### 7.3 TLS captured across a suspend point

Any handler caching `CLIO_CUR_WORKER` / a `Worker*` / a `GetCurrentTask()` reference in a local
across a `co_await` is already fragile and becomes wrong under migration. Grep the ChiMods and
audit each use for suspend-point crossing — cheap and mechanical, do it in P0.

### 7.4 Per-thread SHM server affinity

Each worker `ServerInit`s a named segment `clio-<pid>-<tid>` on its own thread
(`worker.cc:242-247`). If any response path is keyed to the *worker's* tid rather than the
*client's*, migration breaks it. Read `IpcManager::GetTls()` / `SendRuntime` before P2.

### 7.5 ZMQ socket thread-ownership (gates D8)

*"ZeroMQ sockets are not safe to share across threads, so each socket has exactly one owner
thread"* (`default_sched.cc:170-171`). D8 relies on the weaker, documented ØMQ property — a socket
*may* migrate between threads given a full memory barrier and no concurrent use — rather than on
socket recreation. **Verify that reading of ØMQ against the version in use before building on it**;
if it does not hold, D8 needs socket teardown/recreate on handoff, which is a much larger change and
would push the net pollers back to non-migratable.

Second-order: `IpcCpu2CpuZmq` and the admin recv threads cache `GetMainTransport()`
(`ipc_manager.h:280`) on their own background threads. Those are unaffected by worker placement, but
confirm none of them shares a socket with the periodic pollers.

### 7.6 GPU worker rescue (spike)

`gpu_lanes_` are polled by their owning worker, and `SuspendMe` (`worker.cc:483-486`) early-returns
for GPU workers because they must never sleep. Determine whether `SetGpuLanes` can transfer lanes to
a replacement thread, or whether device-context affinity makes GPU workers non-migratable like net
workers. Only needed if we intend to rescue them at all.

### 7.7 Published worker tids go stale under an elastic pool *(gates P1)*

`Runtime::ClientConnect` publishes the runtime's worker OS thread ids to every connecting client
(`admin_runtime.cc`, #642) so SHM clients can address each worker's `clio-<server_pid>-<worker_tid>`
mailbox directly. That list is a **snapshot taken at connect time**, and P1 makes the worker set
mutable for the first time:

* A client that connected before a spawn never learns the new worker's tid — benign, it just never
  uses it.
* A client holding a **retired** worker's tid addresses a dead mailbox, and tasks sent there are
  never consumed. Not benign.
* `ClientConnectTask::kMaxWorkerTids` is a fixed cap; an unbounded elastic pool can exceed it.

So `RetireWorker` cannot simply join and drop a worker whose tid has been published. Options: never
retire a published worker (simplest — retirement is an optimisation, not a correctness requirement);
bump `GetServerGeneration()` and make clients re-fetch; or have the replacement keep draining the
retired worker's mailbox. Recommendation: **do not retire published workers** in this issue. Note
this constrains P1 independently of any migration work.

### 7.8 CI noise

`force_net` / stress tests were already flaky on `dev` at the #782 merge (different test each run,
clear on retry). Do not read one red run as a regression; re-run and compare against a same-day
`dev` baseline.

---

## 8. Open decisions

*Closed:* suspended-task tracking → D4 (park in the blocked queue). Event-queue overflow → §7.2
(keep blocking `Emplace`, re-drain quarantined queues). How exclusivity is expressed → D8-3
(a property of the group binding, decided once at bind time from the declared `TaskStat`). How many
networking groups → D8-2 (three, by socket ownership).

1. **Awaited-future handle (D4).** Keep `awaited_fshm_` as a raw `const void*` (`task.h:885`) and
   document the "safe while suspended" invariant, or promote it to an owning `Future`? The raw
   pointer is sound today only because the suspended parent's frame keeps the `FutureShm` alive —
   an invariant the readiness scan now depends on, where before it only did a pointer comparison.
   Recommendation: promote it; it is a small change and removes a lifetime argument from the
   critical path.
2. **Readiness-scan backoff.** Reuse the existing `% 2/4/8/16` buckets keyed on yield count, or
   re-key them on *time* parked? Yield count no longer means much for a task that suspends once and
   waits a long time.
3. **Wedged-worker fate.** When the bad task finally returns, does the worker rejoin the pool or
   retire? Rejoining risks re-wedging on the same task class; retiring leaks a thread per bad task
   until exit. Recommendation: rejoin with exponential-backoff quarantine.
4. **All-workers-stalled.** `LoadBalance` already detects it (`default_sched.cc:383`). With an
   unbounded elastic pool the answer is "spawn as many replacements as needed" — confirm there is
   no cap, and that the loud warning stays.
5. **Do the periodic buckets stay four-deep?** The `% 2/4/8/16` scheme is a hand-rolled timer wheel.
   Optional cleanup, unrelated to correctness.

---

## 9. Test plan

* **Unit:** D3 handshake under a stress harness (N producers signalling, 1 migrator re-pointing in
  a loop, 1 parent) — assert every signal is delivered exactly once *or* duplicated-and-skipped,
  never lost. This is the piece worth proving outside the runtime.
* **Integration** (self-launching runtime tests):
  * chained task + wedged worker → parent completes within a bound independent of spin length;
  * rescue: lane transfer republished the tid, backlog drained, blocked queues emptied;
  * migration racing a completion → no lost wakeup, no double-execute (per-task exec counter);
  * duplicate-event tolerance (§6.3) as an explicit case.
* **P2a-specific** (the #705 risk surface):
  * **no spurious resume** — a task parked awaiting a slow subtask must not be resumed by the
    readiness scan before that subtask completes. Assert with a counter on the await return path;
    this is the regression that would silently corrupt reads/writes rather than hang.
  * parked task is resumed exactly once when its event arrives, and its blocked-queue entry is
    gone afterwards (no stale entry, no double-park on re-suspend);
  * the lost-wakeup branch fires and logs when an event is deliberately dropped;
  * `num_blocked_tasks_` now tracks the real suspended count;
  * leak-detection suite against the new owning reference (D4, ownership note).
* **P2b-specific** (D8, net roles):
  * **the stale-binding test** — stall a group's worker, let `LoadBalance` rebind, then run a full
    period and assert the group did **not** return to the stalled worker (D8-4). This is the trap:
    the group early-return precedes every health check in `RuntimeMapTask`.
  * **the learner test** — run a poller long enough for SGD to converge and assert its group still
    holds a worker to itself. Bind-time exclusivity (D8-3) should make this structurally impossible;
    the test exists to prove nobody reintroduced continuous cost-based repulsion.
  * a group keeps the **same** worker across many periods — its socket must not hop threads per tick;
  * every member of a group lands on the same worker — `kClientRecv`/`kClientSend` especially, since
    they share the ROUTER socket;
  * the three groups land on three **different** workers (D8-2), so a wedge in one leaves the others
    polling;
  * no non-group task is ever routed onto an exclusive worker; if all workers are exclusive, an
    ordinary task causes a spawn rather than landing on one;
  * `RetireWorker` leaves no dangling binding in any container's `task_group_map_` (D8-4);
  * degraded mode: with workers > cores, exclusive groups collapse onto shared workers **without**
    splitting a group across workers;
  * `kClientRecv`/`kClientSend` work never runs on two workers at once (assert one live instance);
  * kill the worker hosting a poller mid-flight → the poller migrates and networking recovers, which
    is the §2.3 outage becoming survivable and is the headline result for this phase;
  * throughput/latency A/B vs. the static net workers — dedicated-by-load must not be worse than
    dedicated-by-role.
* **P2c-specific** (periodic, D4b):
  * **the self-undo test** — migrate a periodic task, let one full period elapse, assert it did
    *not* route back onto the stalled worker's lane. This is the failure that looks like a working
    rescue for 500 ms and then silently stops.
  * a role-pinned admin poller (14/15/20/21) is **not** migrated, and the rescue path says so;
  * a queue of long-stranded periodic tasks does not all fire in the same tick after migration
    (`BlockStart()` reset);
  * a migrated periodic task whose `Lane()` went null is never silently dropped
    (`worker.cc:1247-1251`) — add the missing log there regardless.
* **Node-level:** stall a compute worker under network load and assert peer traffic and client IPC
  keep flowing — i.e. the §2.3 outage does not occur. Worth having even though compute workers are
  not where that risk lives, because it is the regression test for D7's role gating.
* **TSan** on the P2 diff (§6.5).
* **Benchmark:** `sched_variety` chained-class p50/p99/max per phase.
* **Regression:** the embedded-FUSE xfstests named in `SuspendMe`'s comment
  (`generic/006/007/011/013/089/100/113/127/286/363/438/471`) are the historical canary for
  worker-starvation changes; run on a 2-core-constrained config, which is where they hang.

---

## 9b. Implementation status (measured, 2026-07-22)

**Done and verified.** `test_stall_migration.cc` — 5 tests, 3/3 consecutive runs green at a flat
61-63 s, TSan-clean in every changed file, no regression (wait 4/4, comutex 13/13).

| Category (§2) | Mechanism | Status |
|---|---|---|
| 1 lane backlog | lane transfer (D5) | done |
| 2 blocked queues | `park_mtx_` + `MigrateParkedTo` | done |
| 3 periodic queues | same | done |
| 4 retry queue | same | done |
| 5 event queue | queue-object transfer, drained by adopter | done |
| 6 `co_await`-suspended | follows category 5 | done |

**The design got simpler than §4 assumed.** Transferring the event queue OBJECT rather than draining
it redirects both queued events and every event a still-running subtask pushes later, because a
parked task's raw queue pointer stays correct. No per-task re-pointing and no producer-side
handshake — so D1's `mpmc` and D3's `sig_mtx_`/`sig_gen_` were never needed on this path. Same
insight as the lane transfer.

**Corrections to earlier decisions, forced by measurement:**

* §2.5 stands and is the load-bearing fact: at the default `num_threads=4` there is exactly ONE
  general-purpose worker, so migration has nowhere to go. `SpawnAdditionalWorker` is the
  precondition for everything else.
* #781's **unbounded** elastic pool is wrong once rescue cascades. Spawning per rescue makes the pool
  track CUMULATIVE stalls; it reached 17 workers on a 6-core box and the suite slowed until it looked
  like a deadlock. Fixed by reusing idle replacements (`FindIdleElasticWorker`) plus a ceiling of
  baseline + one thread per core. Soak now shows 20 rescues from 6 spawns.
* `RetireWorker` stays a no-op — §7.7.

**Bugs found in this work, all by running repeatedly rather than once:**

1. Elastic workers were invisible to stall detection, so cascades stopped at level one.
2. `check()` rescued net workers — the exact D7 ZMQ hazard, violated in the implementation of the
   document that describes it.
3. `AdoptEventQueue` stored over the rescuer's queue pointer, orphaning events for tasks that still
   pointed at it. **Task loss**, and the true cause of the intermittency previously misdiagnosed as
   a hang, oversubscription, and rescue latency in turn.
4. Six data races TSan found in the new code (unguarded `queue.size()`, `GetSuspendPeriod`,
   `Finalize`, `tid_`, `load_`, `is_running_`).
5. Pre-existing: `ReschedulePeriodicTask` silently dropped a periodic task with no lane — fixed.

**Still open.** `Container::Reinforce{Cpu,Wall}Model` races on its float vectors (~30 TSan reports).
Pre-existing #781; the fix either breaks the public const-ref getters or puts a mutex on the
per-task-completion hot path of every worker, so it is recorded rather than changed here.

**What is NOT proven.** The tests cover two stall shapes — CPU spin and blocking sleep — on one
6-core box, single node, no CI. There is no coverage of dependency cycles (mutual `co_await`, lock
cycles), which migration cannot fix by design and the runtime does not detect. A task wedged inside
its own handler is still unrescuable. Net and GPU workers are never migrated (D7), so a wedge there
remains a node-level outage until D8 lands. "Deadlock is impossible" is therefore NOT established;
what is established is that a worker wedged by a non-yielding task no longer strands the work
committed to it.

---

## 9c. The invariant this design creates — read before extending it

The implementation collapsed most of §4: transferring a queue OBJECT rather than draining it
redirects both queued entries and every future arrival, because the tasks' raw pointers stay valid.
That is why D1's `mpmc_ring_buffer` and D3's `sig_mtx_`/`sig_gen_` handshake were never needed.

The price is a new invariant, and it is not enforced by any type:

> **Every transferable structure has exactly ONE owner that drains it, and no owner ever BLOCKS on a
> structure another thread owns.**

Three bugs came from breaking it, none caught by local tests:

| Break | Symptom | Found by |
|---|---|---|
| `AdoptEventQueue(w->GetEventQueue())` left the donor holding the same pointer — two consumers on a single-consumer ring | `cr_all_safe_bdev_tests` SEGFAULT on all 3 Windows configs, clean on Linux | **CI** |
| `adopted_event_queues_` appended and drained but never transferred — a cascading rescue stranded inherited queues | silent; needs a rescue chain two deep | audit |
| `TaskLane::Push` is `WAIT_FOR_SPACE`, so pushing from the MONITOR thread into a full lane spins forever | would freeze stall detection, rescues AND the watchdog — silent, because the alarm lives on the frozen thread | audit |

So the review question for any change here is not "is it correct?" but **"who owns this, and can this
block?"** Every transfer point must answer both. The current points are: `ReleaseLane`/`AdoptLane`,
`ReplaceEventQueue`/`AdoptEventQueue`, `TransferAdoptedEventQueuesTo`, `MigrateParkedTo`, and the
per-worker-id lane reservation in `SpawnAdditionalWorker`.

Corollary worth stating: **local green means little for this class.** The Windows segfault was a race
whose window this 6-core Linux box never opened, and it survived a full TSan pass — TSan only sees
what actually executes concurrently. Platform CI is not a formality here; it is the only tool that
found the worst bug in the branch.

---

## 10. Deferred / out of scope

* **General work stealing by idle workers.** Explicitly deferred per D-a. `StealWork()` stays a
  stub. This is the efficiency half; P1–P2 are the safety half, and keeping them apart means a
  perf regression there cannot un-fix liveness.
* **Late-bound completion signalling** (resume target chosen at signal time instead of rewriting
  stored addresses). This removes D3 and D4 entirely and makes idle-thief stealing nearly free,
  but needs a per-task CAS state machine. Revisit if per-migration re-pointing cost or the
  suspended registry proves awkward.
* **Preemption.** A non-yielding task still owns its thread until it returns; we bound the blast
  radius, we do not interrupt it.
* **Cross-node stealing.** Everything here is intra-process.
* **Poison-task quarantine** (a method that stalls N times gets confined to a blocking pool).
  Separate issue — it changes placement policy, not mechanism.
