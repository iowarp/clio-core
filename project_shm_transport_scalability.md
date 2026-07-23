# Project: Scalable, Low-Latency SHM Transport (issue #807)

Branch `807-shm-transport-scalability`, based on `origin/dev` @ `f679f31c`.

Follow-up to the SHM transport refactor (branch `debug-delta-transport-deadlock`) that reshaped the
transport into the ZMQ model — one inbound ring + one outbound ring, each drained by one dedicated
background thread. That fixed the #768/#774 bidirectional-deadlock class but made the transport
deliberately single-threaded. This is the deferred second half: keep the deadlock fix, remove the
serialization.

---

## 1. Where we are (verified current architecture)

The SHM path today, as reshaped by the refactor:

**Inbound (client → runtime):**
* ONE ring `clio-<pid>-shm-in` (MPSC), created in `ServerInitShm` (`IpcManager::shm_in_server_`).
* Drained by ONE dedicated `IpcManager::RecvShmServerThread` (`ipc_manager.cc:3098`), which loops
  `IpcCpu2Cpu::RecvIn(this)` (`ipc_cpu2cpu.cc:14`) — deserialize one task, push onto **lane 0**,
  `AwakenWorker`. `RuntimeMapTask` then redistributes off lane 0.

**Outbound (runtime → client):**
* SHM `IpcCpu2Cpu::SendOut` (`ipc_cpu2cpu.cc:92`) runs **INLINE on the executing worker**: it
  serializes the result and does an MPSC transfer into the client's single `clio-<pid>-shm-out`
  ring (`conn->Send`). It can block if that ring is full.
* Contrast the ZMQ path: `IpcCpu2CpuZmq::EnqueueSendOut` (`ipc_cpu2cpu_zmq.cc:229`) enqueues an
  OWNING future onto `net_queue_` (nonblocking) and a background drainer (`ClientSend` periodic →
  `IpcCpu2CpuZmq::SendOut`) serializes and sends. **The ZMQ path already does what #807 asks for; the
  SHM path does not.**

**Client outbound recv:**
* ONE ring `clio-<pid>-shm-out` per client process (`IpcManager::shm_out_server_`), drained by ONE
  `RecvShmClientThread` (`ipc_manager.cc:401`) that demuxes by net_key and wakes the waiting thread
  via `EventManager::Signal`. `RecvOut` blocks on the EventManager.

**The pivotal primitive constraint.** `ShmMpscTransport` is **MPSC**: *"multiple producers … push
concurrently; a single consumer RecvBytes"* (`shm_mpsc_transport.h:12,18`). One ring cannot be
drained by two threads. **Therefore parallel draining requires N rings, not N consumers on one ring.**
`RegisterConsumer()` is liveness tracking (is anyone draining?), not multi-consumer support.

---

## 2. The serialization points (the bottlenecks)

| # | Point | Cost |
|---|---|---|
| 1 | Single inbound ring | Every client contends on one MPSC tail atomic |
| 2 | Single inbound drainer | One core does ALL task deserialization + routing |
| 3 | Lane-0 ingress funnel | `RecvIn` deposits every task on lane 0 before routing |
| 4 | Inline blocking `SendOut` | Worker thread pays serialize+transfer; blocks on a full client ring — stalling task processing, the exact thing the design meant to avoid |
| 5 | Single client out-ring + drainer | A client receiving many concurrent responses is bottlenecked on one ring + one thread |

**Measured impact (mixed 1µs–1s workload, 12 client threads, SHM forced, this box):** Clio's median
latency for sub-100µs tasks is ~1.2 ms vs ~250 µs for a self-RPC libthallium na+sm baseline — 4.8×
on median, on a workload where our *scheduler* wins the p99 (25 ms vs 31 ms). Much of the median gap
is transport overhead, and points 1–5 are where it lives. (Bench: `thallium_variety_bench.cc` on the
785 branch; reproduce with the same distribution.)

---

## 3. Design — combine the two approaches

Neither "single dedicated drain thread" (deadlock-safe but serial) nor "workers do their own I/O"
(parallel but deadlock-prone, the pre-refactor model) is right alone. Combine them:

* **Parallel sharded rings** replace each single ring, so producers spread across tails and each
  ring has its own drainer.
* **A background drain-thread pool** keeps every ring constantly drained regardless of whether app
  or worker threads are blocked — this is what preserves the #768 deadlock fix.
* **Nonblocking rings** — no producer (client submit, or worker response) ever blocks on a full
  ring; the drainers guarantee forward progress.
* **Nonblocking `SendOut` + an ordered pending queue** — the worker enqueues a pending response and
  returns immediately; a background sender drains the queue in order, mirroring `EnqueueSendOut`.

### D1 — Inbound: N sharded in-rings, N drain threads

Replace the single `clio-<pid>-shm-in` with `clio-<pid>-shm-in-<k>` for `k in [0, S)`. Each ring has
one dedicated drain thread (`RecvShmServerThread(k)`), satisfying MPSC single-consumer per ring.

* **Client → ring assignment**: shard by client `(pid,tid)` hash so a given client thread always
  targets the same ring (cache locality; preserves per-thread request ordering). Round-robin is the
  alternative — better balance, no per-thread ordering. See open Q1.
* **Removes points 1 and 2.** S clients-worth of deserialization now runs on S cores.
* **Point 3 (lane-0 funnel)**: each drainer can deposit on a *different* ingress lane (e.g. lane
  `k`) instead of all on lane 0, so `RuntimeMapTask` starts from a spread rather than a funnel.
  Cheap and worth doing here.
* **Sharding factor S**: tie to a config knob, default ~= number of I/O workers, capped. Too many
  drain threads on a small box just steals cores from workers (the #785 lesson).

### D2 — Outbound `SendOut`: nonblocking enqueue + ordered pending queue + sender pool

Make SHM `SendOut` match the ZMQ model instead of running inline:

1. The worker calls `SendOut`, which now **enqueues an OWNING future** (as `EnqueueSendOut` does —
   the non-owning self-handle hazard at `ipc_cpu2cpu_zmq.cc:229` applies identically) onto a pending
   SHM-send queue, nonblocking, and returns. The worker never serializes and never touches the
   client ring.
2. A **background SHM-sender pool** drains the pending queue: serialize the result, MPSC-transfer
   into the destination client's out-ring. Processed **in order** per destination.
3. Task lifetime: the owning future keeps the task alive until the response is transferred; then the
   pending-queue entry drops. `SetComplete` / `ClearFlags(TASK_DATA_OWNER)` move from the worker to
   the sender (they currently run at the end of the inline `SendOut`).

**Sharding the send side**: partition the pending queue by destination client so responses to
different clients proceed in parallel and in-order-per-client. A single global pending queue with a
pool draining it loses per-destination ordering unless entries are keyed — see open Q2.

**Do NOT reuse the `net_queue_` / `ClientSend`-periodic machinery.** That path runs on a *worker*
(the net worker), and #785 is removing net-worker roles. The SHM sender must be its own background
thread pool, independent of the worker pool.

### D3 — Nonblocking rings + the full-ring problem

Producers must never block. Today `conn->Send` into a full out-ring can spin/block on the worker.
Options, in preference order:

* **(a) Size + constant drain.** With a dedicated drainer per ring always running, a correctly sized
  ring rarely fills; make `Send` fail-fast (`SHM_MPSC_DONTWAIT`-style) and the *sender thread* (never
  the worker) retries/backs off. The worker is already decoupled by D2, so back-pressure lands on the
  sender pool, not on task processing.
* **(b) Spill list.** On a full ring, the sender parks the entry on an in-process overflow list
  drained ahead of the ring next tick. Bounds memory to in-flight responses.
* Never busy-spin a *worker* on a full ring (the #785 monitor-thread lesson: a `WAIT_FOR_SPACE` push
  from the wrong thread freezes it).

Symmetric on the client's inbound submit: a client submitting into a full in-ring must not block a
latency-critical app thread indefinitely — fail-fast + brief spin + park, never an unbounded spin.

### D4 — Client outbound: N sharded out-rings, N drain threads (optional, phase 3)

Mirror D1 on the client: `clio-<pid>-shm-out-<k>`, each drained by its own thread, demuxed by
net_key. Only helps a single client with high response concurrency; lower priority than D1/D2 because
most latency wins come from the runtime side. Gate on measurement.

---

## 4. Correctness constraints (must not regress)

* **#768/#774 deadlock fix**: every out-ring stays drained by a dedicated thread even when app
  threads block in submit. Preserved by construction — D1–D4 only *add* drain threads.
* **net_key demux**: the `client_net_key_` save (`RecvIn`) / restore (`SendOut`) must survive the
  move of `SendOut` to the sender pool (`ipc_cpu2cpu.cc:55,116`). It is per-task state, so it rides
  the pending-queue entry — verify it is restored in the sender, not the worker.
* **`TASK_DATA_OWNER` payload adoption** on the consumed SHM path (the refactor's bug #2:
  `CleanupResponseArchive` must not blanket-free a CHI-allocated `bulk.data.ptr_`). Untouched here,
  but the sender now owns the archive lifetime — re-verify.
* **Per-destination response ordering**: a client that submitted A then B on one thread and awaits
  both must still observe correct completion; out-of-order *delivery* is fine (net_key demux) but
  the pending queue must not drop or reorder within a destination in a way that breaks the demux.
* **Single-consumer per ring**: never point two drain threads at one ring. The sharding is the only
  parallelism; enforce it (assert `RegisterConsumer` returns exclusive).

---

## 5. Phases

**P0 — Repro + baseline.** Port `thallium_variety_bench` alongside `clio_run_thrpt_bench
--test-case sched_variety`; capture current SHM median/p99/throughput at 1/4/12/24 client threads on
this box as the before-number. Add transport counters (per-ring drained counts, pending-queue depth,
full-ring events). Success metric for the whole issue: **sub-100µs median latency and per-client
throughput both improve materially with client-thread count**, closing the gap to the thallium
baseline without regressing the p99 the scheduler already wins.

**P1 — Nonblocking queued `SendOut` (D2 + D3).** Highest value, smallest blast radius, no SHM-layout
change. Move SHM `SendOut` off the worker to a background sender pool with an ordered per-destination
pending queue; make the ring push fail-fast on the sender. This alone should cut the median (removes
point 4, the inline serialize+block on the worker).

**P2 — Sharded inbound rings (D1).** N in-rings + N drain threads + per-ring ingress lane. This is
the scalability half — measure throughput vs client-thread count before/after. Changes SHM segment
layout, so it carries the ABI/segment-sizing risk.

**P3 — Sharded client out-rings (D4).** Only if P1/P2 measurements show the client's single out-ring
+ drainer is still a bottleneck at high response concurrency.

Each phase independently benchmarkable against the P0 baseline.

---

## 6. Open questions

1. **Inbound shard key** (D1): hash by `(pid,tid)` (locality + per-thread order) vs round-robin
   (balance)? Leaning hash, since per-thread request order is the property clients expect.
2. **Send-side ordering vs sharding** (D2): per-destination pending queues (clean ordering, more
   queues) vs one global queue with keyed entries drained by a pool (fewer structures, ordering
   needs care)? Leaning per-destination — ordering is a correctness constraint, not a tuning knob.
3. **Sharding factor S and thread budget**: how many drain + sender threads before they contend with
   workers for cores? #785 showed unbounded threads hurt on small boxes. Tie to core count with a
   cap; measure the knee.
4. **One pool or two?** Separate inbound-drain and outbound-send thread pools, or a unified transport
   pool that does both? Separate is simpler to reason about; unified may pack better on small boxes.
5. **Does the client submit path (SendIn) also need sharding**, or is a client thread's own submit
   already parallel enough (each thread pushes independently into its shard's ring)? Likely fine
   once D1 shards the rings; confirm by measurement.

---

## 7. Test / benchmark plan

* **Correctness**: all 4 transport modes still pass; SHM transport-mode integration (real bdev 1MB
  write+read round-trip); `per_process_shm`; the #768/#774 deadlock repro (bidirectional submit
  while app threads blocked) must stay green.
* **Ordering**: a client thread submitting a dependent sequence over SHM observes correct
  completions under the sharded/queued path.
* **Scalability**: `sched_variety` + `thallium_variety_bench` at 1/4/12/24 client threads;
  report median + p99 + throughput per phase; ratios only within a session (this box is ±20%
  cross-session, per [[clio-vs-thallium-latency]]).
* **Deadlock-safety**: full in-ring / full out-ring under sustained load must not hang — the
  fail-fast + drainer path (D3) keeps producers live.
* **No worker stalls**: with `SendOut` moved off the worker (P1), a full client ring must show up as
  sender-pool back-pressure, never as a stalled task-processing worker.

---

## 8. Out of scope

* The ZMQ/TCP/IPC transports (already queue their sends via `net_queue_`); this issue is SHM-only,
  though it should converge the SHM path toward the ZMQ path's shape.
* Cross-node run2run transport.
* The #785 net-worker-role removal — related (the SHM sender must NOT depend on a net worker) but a
  separate change.
