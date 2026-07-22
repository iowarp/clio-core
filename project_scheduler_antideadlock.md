# Anti-deadlock scheduler (issue #781)

Tracking issue: https://github.com/iowarp/clio-core/issues/781

## Thesis
The runtime's liveness must **not** depend on tasks being honestly labeled. A single
**non-yielding** task (CPU spin, blocking syscall, or an infinite loop inside a
coroutine that never yields) permanently wedges the worker it lands on. Workers are
coroutine-based, so an expensive-but-*cooperative* task is already fine — the lethal
case is the non-cooperative one. Today every externally-ingressed task funnels to
lane 0 (`DefaultScheduler::ClientMapTask` returns 0), the pool is fixed at
`num_threads`, there is no work-stealing, and `RebalanceWorker` is a dead no-op. A
few mislabeled blocking tasks on lane 0 starve the ingress path — the likely Delta
hang.

## Goal
Run a group of arbitrarily complex operations with the **minimum number of threads**,
keeping **low p99 for quick tasks** and **scaling for large tasks**, while
**guaranteeing forward progress even for adversarial/mislabeled tasks**.

## The algorithm — elastic, latency-aware, work-conserving pool with hazard isolation

Two independent layers. Layer 1 is best-effort efficiency; Layer 2 is the hard
liveness guarantee. Correctness never depends on Layer 1 being smart.

### State (per worker)
- `load_` (µs, exists today) — sum of `PredictedLoad()` of in-flight/queued tasks.
- `last_exec_start_` (**new**) — timestamp stamped at the top of `ExecTask`, cleared on
  return. `now() − last_exec_start_` = current task's elapsed wall time.
- `realtime_load(w) = w.load_ + (now() − w.last_exec_start_)` — measured, never predicted.

### Layer 1 — mapping (RuntimeMapTask), keeps p99 low with few threads
```
C = incoming task coarse tolerance   # declared/default; NOT a learned prediction
candidates = workers, in ring order from a rotating anchor
pick argmin_w realtime_load(w) subject to realtime_load(w) <= C
if none qualify: pick argmin_w realtime_load(w)          # least loaded overall
```
A 5 µs task (small C) refuses workers already deep in work, so it is not queued behind
a 500 ms task. Group-affinity and periodic-net routing are preserved from today.

### Layer 1 — work stealing (StealWork), keeps the pool work-conserving
```
when thief.lane is empty:
    for neighbor in {left, right}:      # ring neighbours
        move up to 4 tasks neighbor.lane -> thief.lane
    if still empty:
        victim = worker with max realtime_load
        move up to 4 tasks victim.lane -> thief.lane
```
Work-conservation is what lets us hit throughput with *few* threads (kept busy) rather
than many (mostly idle) — it directly serves "minimize threads."

### Layer 2 — the liveness guarantee (LoadBalance, monitor thread every 500 ms)
```
for w in workers:
    if w.IsExecuting() and (now() - w.last_exec_start_) > STALL = 1s:
        w2 = work_orch.SpawnAdditionalWorker()      # unbounded; see Observability
        move ALL of w.lane -> w2.lane                # w stays stuck on its 1 bad task
        metrics.stalls++; WARN("worker {} stalled {}s on task {}")
    ingest w's recently-completed exec-times into the perf-bin PDF   # telemetry
if aggregate pending load high and >=1 worker idle:
    kick StealWork / spawn
retire elastic workers idle > COOLDOWN = 5s          # hysteresis -> back to minimum
```
A pathological task strands exactly one thread; its backlog is migrated and continues;
new tasks are mapped away from it (its `realtime_load` is enormous). The runtime cannot
fully wedge. **Con:** stalls are only caught on the 500 ms tick — bounded rescue
latency, but simple and robust.

### Why threads stay minimal
Baseline = `num_threads`. We grow *only* on a demonstrated stall or sustained
imbalance, and retire elastic threads after an idle cooldown. Steady state returns to
the baseline; growth is proportional to concurrent *hazards*, not to load.

## Component changes (grounded in current code)

| Area | File(s) | Change |
|---|---|---|
| Remove `ClientMapTask` | `scheduler.h`, `default_sched.*`, `local_sched.*` | delete the virtual + impls + `MapByPidTid`; ingress maps in the runtime only |
| Rewire ingress | `ipc_cpu2cpu_zmq.cc:201`, `ipc_cpu2cpu.cc:72`, `ipc_run2run.cc:461` | push to shared ingress lane 0 (behavior-preserving), runtime `RuntimeMapTask` places it |
| `LoadBalance` | `scheduler.h` (+ both schedulers) | new pure virtual, replaces dead `RebalanceWorker`; called every 500 ms |
| Monitor thread + elastic pool | `work_orchestrator.*` | dedicated monitor thread; `SpawnAdditionalWorker()` / `RetireWorker()` (today the pool is fixed at Init) |
| Worker instrumentation | `worker.h/.cc` | add `last_exec_start_` (stamp in `ExecTask` @ worker.cc:605); `RealtimeLoad()`, `IsStalled()` |
| Measured-load mapping | `default_sched.cc RuntimeMapTask` | route by `realtime_load`, coarse tolerance; no PDF |
| Work stealing | `default_sched.cc StealWork` | neighbours then most-loaded; needs steal-safe lane pop |
| Perf-bin PDF (telemetry) | `default_sched.*` | histogram `<10µs…≥1s` from `EndTask` timings; metrics + LoadBalance signal only |
| Observability | metrics + logs | `sched.threads.{live,spawned,stalled,retired}`; WARN when live > N×num_threads |
| Benchmark | `MOD_NAME_*`, `clio_run_thrpt_bench.cc` | `spin_us` param on `Custom`; mixed 1µs–1s workload; avg + p99 by class |

## Observability / accepted risk
The elastic pool is **unbounded** (always spawn to guarantee progress). The safeguard is
loudness, not a cap: a genuine flood of non-yielding tasks grows threads without bound,
and we make that observable (`kWarning` past `N×num_threads`, live/spawned/stalled
metrics) instead of silently wedging.

## Benchmark — scheduler variety
`MOD_NAME::Custom` (a no-op today at `MOD_NAME_runtime.cc:71`) gains a `spin_us`
busy-spin parameter. `clio_run_thrpt_bench --test-case latency` drives a mixed workload
with per-request compute drawn 1 µs … 1 s (mostly quick, occasional huge) and reports
avg + **p99 latency segmented by task class**. Quick-task p99 under expensive-task
pressure is the starvation metric. Old (fixed pool, lane-0 funnel) vs new (elastic +
steal + stall-rescue).

## Measured results (first implementation)

`clio_run_thrpt_bench --test-case sched_variety --threads 12 --duration 12`, runtime
`num_threads: 8` (→ 4 I/O workers), mixed 1µs..1s workload (90% quick / 9% medium /
1% heavy), Release. `CLIO_SCHED_FUNNEL=1` forces the legacy funnel for A/B:

| metric | funnel (old) | measured-load (new) |
|---|---|---|
| **quick-task p99** | **519 ms** | **62 ms** (8.4× better) |
| quick-task avg | 29 ms | 19 ms |
| throughput | 375 ops/s | 560 ops/s (+50%) |

Measured-load routing (Layer 1) alone cuts quick-task p99 by **8.4×** by steering quick
tasks onto the least-loaded I/O worker and off any worker running a heavy/non-yielding
task. The residual 62 ms comes from two not-yet-wired pieces: (a) `RealtimeLoad` counts
only the *executing* task (`load_` bumps at exec-start), not tasks *queued* behind a
heavy one on the same lane — a prediction-free fix is to add a per-worker
assigned-but-not-done counter (or lane depth) to the load signal; (b) the backlog-steal
(LoadBalance stall→spawn/steal) that rescues tasks already queued on a stalled worker,
which needs the steal-safe MPSC pop. Both are the next PRs.

## Open questions
- Steal-safe pop on a single-consumer MPSC lane (or a dedicated steal channel).
- `WAIT_FOR_SPACE` blocking `Push` must not block a worker during steal/migration.
- Keep ingress from re-forming the lane-0 funnel under load (StealWork spreads it).
