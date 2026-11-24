# Collective latency: Clio `PoolQuery::AllToOne` vs MPI

A 4-node Docker benchmark that times our collectives head-to-head against the
implementations they are modelled on, so "how slow are we?" has a number
attached to it instead of an intuition.

## What it measures

Four arms, run by the same four ranks, over the same cluster and the same TCP
network, in one process each:

| arm              | what it is                                        |
|------------------|---------------------------------------------------|
| `mpi_barrier`    | `MPI_Barrier` — the reference barrier              |
| `mpi_allreduce`  | `MPI_Allreduce(1 × u64, MPI_SUM)` — the reference reduce |
| `clio_barrier`   | MOD_NAME `BarrierTask`, routed `PoolQuery::AllToOne` |
| `clio_allreduce` | MOD_NAME `AllReduceTask`, routed `PoolQuery::AllToOne` |

One rank per physical node (`mpi_hostfile`), each attached as a client to its
**own** local clio daemon (`CLIO_WITH_RUNTIME=0`). MPI is used for the two MPI
arms and, in the clio arms, only to align the start of a phase and to reduce
per-rank statistics at the end — never inside a timed clio region, so no MPI
cost leaks into the clio numbers.

**Why `AllToOne` is the allreduce analogue.** An `AllToOne` task parks at the
neighborhood leader until a task from every container in the pool has arrived
(the pool has one container per node); the batch is then folded into a single
aggregate via `AggregateIn`, that aggregate runs once, and its OUT is broadcast
1→N back to every participant. All contribute, all block until the last one
has, and all observe the same combined result — the defining properties of
`MPI_Allreduce`. The barrier arm is the same path with a payload-free task, so
the gap between the two clio arms isolates the cost of the reduction from the
cost of the synchronization.

**The allreduce arm self-checks.** Rank *r*'s contribution at iteration *i* is
`(i+1)*1000 + (r+1)`, so the expected total encodes both the iteration and the
full membership. A batch that mixed two iterations, dropped a member, or
double-counted one cannot match the closed form by accident, and the run exits
non-zero. Without that check the benchmark could happily time a collective that
had silently stopped being collective (which is exactly what it caught — see
below).

## Are the barriers actually barriers?

Yes — verified, not assumed. This matters because a return code cannot tell you:
a "barrier" that completed every request independently would hand `rc=0` to
everyone and look perfectly healthy. That is not hypothetical here — it is
exactly the failure the cross-node path already had (see below). And unlike the
allreduce, `BarrierTask` has no payload, so there is no result to check.

So the run forces the question before it times anything. Arrivals are staggered
by `COLL_BENCH_STAGGER_MS`, rotating which rank arrives last each round, and
every rank's exit is compared against the **latest entry across all ranks** on a
clock the ranks share (all containers are on one host and Docker does not
virtualize `CLOCK_MONOTONIC`; the check measures the actual clock spread first
and declares itself inconclusive if the clocks disagree by more than 5 ms).

12 rounds, 50 ms stagger, measured rank clock spread 22.7 µs:

| check                      | verdict | earliest exit vs. last arrival |
|----------------------------|---------|-------------------------------:|
| `MPI_Barrier` (control)    | HELD    | +48.8 µs |
| clio barrier               | HELD    | +121.3 µs |
| clio allreduce             | HELD    | +168.9 µs |
| plain local task (neg ctrl)| BROKEN  | −150 050.6 µs |

No participant ever returned before the last one arrived. The margins are also
the right size: clio exits ~120-170 µs after the last arrival because that is
roughly what it costs to propagate the release, against MPI's ~49 µs.

The last row is the point. A plain local task is definitively not a barrier, and
the check catches it by −150 ms — exactly `(ranks-1) × stagger`, with 36
violations = 12 rounds × 3 ranks. Without that negative control, "0 violations"
would only prove the check was inert, so the run **fails** if the negative
control ever comes back clean.

The allreduce is held to the same standard *and* verifies its arithmetic: each
rank's contribution encodes the iteration and its rank, so the expected total
cannot be matched by a batch that mixed iterations, dropped a member, or
double-counted one.

### Where the release time goes

The margins above are **release propagation**: last participant arrives → that
participant gets out. Reporting only the minimum flatters the collective, so the
check reports min/median/max. On a quiet host, 40 rounds at 20 ms stagger:

| check                   |    min |   median |     max |
|-------------------------|-------:|---------:|--------:|
| `MPI_Barrier` (control) |  58.2  |   93.6   |  156.0  |
| clio barrier            | 119.5  |  459.9   | 1643.8  |
| clio allreduce          |  71.2  |  443.5   | 2006.8  |

The minimum is the participant co-located with the leader — it needs no network
hop to be told the barrier opened. The median includes the remote ones.

The leader's own contribution is measured directly (`CLIO_COLL_PROF=1`), in µs:

| stage  | what it is                                            | hot | staggered |
|--------|-------------------------------------------------------|----:|----------:|
| notice | flusher spotting that the count was reached           |   0 |     0 |
| build  | constructing the aggregate + folding in `AggregateIn` |   0 |     2 |
| exec   | `Send` the aggregate → schedule → run → `EndTask`     |  20 |    37 |
| bcast  | serialize OUT once, load into all 4 members, complete |   3 |    19 |
|        | **leader total**                                      |  **23** | **58** |

So of a ~444 µs median release, the leader accounts for ~58 µs. **The other
~385 µs is the two cross-node legs** — the last participant's request reaching
the leader, and the release reaching each participant — at roughly 150-190 µs
per one-way leg, which is the same per-hop cost broken down in the table above
(send-queue wait ~45-65 µs, receiving-node residency ~30-60 µs, and wakeups;
serialization ~0.5 µs and the wire ~5-10 µs are noise).

Two things follow. First, `notice` and `build` are ~0: the count-based release
and the aggregation are free, so there is nothing to win in the `BatchManager`.
Second, the largest item that *is* the collective's own is `exec` — 20 µs hot,
37 µs staggered, to schedule and run an aggregate whose body does nothing. That
is a full `ipc_->Send` → route → lane → worker-pickup → `ExecTask` → `EndTask`
round for a no-op. Running the aggregate inline on the flushing worker, rather
than submitting it as an ordinary task, is the one optimization the collective
path itself still offers — and it is worth ~20-37 µs of ~444, so it is a
rounding error next to the network legs.

`skew` (first member arriving at the leader → last) is 305 µs even with the
ranks looping back-to-back, and 60 ms by construction in the staggered test. It
is not a cost of this implementation — it is how far apart the callers actually
are — but it is what every early arriver waits through, so a real application's
barrier cost is its own skew plus the release above.

### One real caveat: the release counts arrivals, not participants

`BatchManager::FlushDue` releases on

```cpp
ready = (num_containers > 0) && (it->second.members.size() >= num_containers);
```

and `BatchManager::Add` appends every arrival unconditionally. Nothing checks
that the N tasks came from N *distinct* containers. So the guarantee actually
implemented is "wait for N arrivals sharing this (pool, method, hash, key)",
which coincides with a barrier only while each participant keeps at most one
request outstanding on a given key.

Every caller today satisfies that — a participant waits for its own completion
before issuing the next — which is why the checks above pass. But a caller that
issued two barriers concurrently on one key would let the group reach N while a
peer had not arrived, and the barrier would release early. That is a latent
sharp edge, not a bug being hit; fixing it means keying membership on the
contributing container rather than counting.

## Results

4 nodes, 1 rank each, Docker bridge network on one host, 1000 iterations after
100 warmup, Release build. Latency in microseconds, **max across ranks** (a
collective is not done until its slowest participant returns):

| arm               |    mean |     p50 |     p99 |     max |
|-------------------|--------:|--------:|--------:|--------:|
| `mpi_barrier`     |   27.02 |   26.35 |   34.29 |   62.37 |
| `mpi_allreduce`   |   22.73 |   20.22 |   40.34 |   63.18 |
| `clio_barrier`    |  320.89 |  305.41 |  897.69 | 1734.59 |
| `clio_allreduce`  |  345.31 |  321.15 |  787.13 | 1505.10 |
| `clio_local_rtt`  |   22.32 |   11.87 |  188.91 |  427.32 |
| `clio_remote_rtt` |  316.63 |  302.95 |  633.80 | 1669.57 |

- `clio_barrier` / `mpi_barrier`: **11.9×**
- `clio_allreduce` / `mpi_allreduce`: **15.2×**

The last two rows are not collectives. They are ordinary empty `Custom` tasks —
one to the caller's own daemon, one to a peer — and they are what makes the
collective numbers interpretable:

```
  plain local task round trip           22.32
  plain remote task round trip         316.63  (+294.31 for the network hop)
  collective barrier                   320.89  (+4.26 for the collective machinery)
  MPI barrier, for scale                27.02
```

**The collective machinery costs ~4 µs.** Parking in the `BatchManager`,
waiting for the count, running the aggregate, and broadcasting the result back
to four nodes is essentially free. A `clio` barrier costs what it costs because
*one cross-node task round trip* costs 317 µs — against a local round trip of
22 µs and an entire 4-rank MPI barrier of 27 µs. The reduction is free too:
`clio_allreduce` and `clio_barrier` differ by less than run-to-run noise.

So the headline ratio is not a statement about collectives. It is a statement
about the cross-node task path, and that is where the remaining work is.

### Where the 317 µs goes

Measured with the env-gated profilers below (`CLIO_NET_QPROF=1`,
`CLIO_NET_TRACE=1`), per one-way hop:

| stage                                    |   cost |
|------------------------------------------|-------:|
| wait on the outbound net queue           | ~45-65 µs |
| serialize the task                       |  ~0.5 µs |
| ZMQ transmit                             |    ~5 µs |
| wire (docker bridge)                     |   ~5-10 µs |
| receiving node: arrival → response queued |  ~30-60 µs |

Serialization and the wire are noise. The cost is queueing and wakeup at both
ends — the task waiting to be picked up for sending, and the receiving node
waiting to run it and hand back a response.

### The 3.1× already recovered

`clio_barrier` was **987 µs** and is now **321 µs**; a cross-node round trip was
652 µs and is now 317 µs. One line did it:

```cpp
client_.AsyncSendPoll(clio::run::PoolQuery::Local(), 0, 500);  // was
client_.AsyncSendPoll(clio::run::PoolQuery::Local(), 0, 25);   // now
```

The period is not a delay — it is a **bucket selector**.
`Worker::AddToBlockedQueue` files a periodic task by its period, and
`ContinueBlockedTasks` services the buckets at very different rates: `<=50 µs`
every 4 worker-loop iterations, `<=200 µs` every 8, `<=50 ms` every **64**. This
poll drives every cross-node send, so at 500 µs the entire outbound network path
was serviced every 64 iterations and tasks sat ~300 µs on the send queue. At
25 µs it lands in the fastest bucket and waits ~45 µs. A `static_assert` at the
call site now pins it below the 50 µs boundary, because the failure mode is
silent: nothing breaks, everything just gets 16× less responsive.

### What did NOT work (measured, not assumed)

Each of these was implemented and benchmarked, and changed nothing:

- **Servicing the fast periodic bucket every iteration** instead of every 4.
- **Waking the net worker on every enqueue** instead of only on empty→non-empty.
- **Decoupling the per-tick maintenance scans** (`ProcessRetryQueues`,
  `ScanSendMapTimeouts`, `ScanTaskProgress`) from the drain. Kept anyway: after
  the period change those liveness scans would otherwise run 16× more often
  than before, which is a CPU regression the latency numbers do not show.
- **Never parking** (`first_busy_wait` huge). Much *worse* — 3798 µs. Four
  containers × 8 spinning workers oversubscribes a 16-core host and starves the
  client processes and ZMQ threads. Do not benchmark this configuration.

That the poll rate no longer matters is itself the finding: the net send and
recv workers are **parked** most of the time (`CLIO_WORKER_RATE=1` shows ~82 µs
and ~157 µs of wall time per loop iteration for workers 6 and 7, versus ~1.3 µs
for the scheduler worker). Their *wakeup*, not their poll cadence, is what sets
send latency now.

### The remaining lever

Move the outbound send off the worker-periodic and onto a dedicated thread
woken on enqueue — exactly what commit `c6c1ef7d` already did for the *inbound*
path, and for the same reason ("the worker scheduler's yield/wake/epoll cycle
added too much latency"). That commit explicitly left the outbound side on the
periodic. Doing so should remove most of the ~50 µs per-hop queue wait, worth
roughly 100 µs of the 317.

It is not a small change: `IpcCpu2CpuZmq::SendOut` currently relies on running
on the net worker's thread so the transport's deferred-completion callbacks
(`DelTask`) can touch coroutine-aware container state, and ZMQ sockets must be
used from the thread that created them. Both constraints have to be carried
over deliberately.

## Running it

```bash
# Requires MPI and Docker CI:
cmake -S . -B build -DCLIO_CORE_ENABLE_MPI=ON -DCLIO_CORE_ENABLE_DOCKER_CI=ON ...
cmake --build build --target clio_run clio_collective_bench

# Full run (defaults: 1000 iterations, 100 warmup)
./run_tests.sh all

# Quick run
COLL_BENCH_ITERS=100 COLL_BENCH_WARMUP=10 ./run_tests.sh all

# Or via ctest (short: 200 iterations, so it works as a regression gate)
ctest -R cr_collective_bench_docker
```

Sub-commands `setup` / `run` / `clean` bring the cluster up, wait on it, and
tear it down separately — useful when iterating, since `all` always tears down.

Results are written to `results.csv` in this directory. Exit codes: `2`
CLIO_INIT, `3` pool create, `4` allreduce mismatch, `5` failed iterations.

## The bug this found

Cross-node collectives did not work at all, and failed *silently*. Two defects,
both fixed on this branch:

1. **`SendIn` overwrote the forwarded member's `pool_query_`** with the
   `Physical(leader)` envelope it was wrapped in
   (`context-runtime/src/ipc/ipc_run2run.cc`). That erased the collective
   routing mode *and* the `container_hash` / `batch_key` that decide which
   group a member joins, so a member forwarded from a non-leader node arrived
   at the leader looking like an ordinary task.
2. **`RouteTask`'s `IsRouted()` early-return preceded the collective check**
   (`context-runtime/src/ipc_manager.cc`). `RecvIn` marks every net-received
   task routed, so even with its query intact a forwarded member would have
   returned `ExecHere` instead of reaching the `BatchManager`.

The combined effect: a collective whose members did not all originate on the
leader node did not happen. Remote members each ran standalone and returned an
**un-combined result with `rc=0`** — an `AllReduce` handed every caller back its
own value — while any leader-local member waited forever for a count that could
never be reached. A hang and a silent wrong answer, depending on where you
stood.

The pre-existing `alltoone` integration test cannot see this: it issues all four
`AllToOne` requests from a *single* client on the leader node, so every member
is leader-local and neither defect is on the path. This benchmark is the first
thing in the tree that submits one member per node, which is both what MPI does
and what a collective is for.

A third fix accompanies them: `BatchManager::OnAggregateComplete` now restores
an owning `RunFuture` handle before `EndTask` on a remote member, because
completing one is an asynchronous `SendOut` and the batch's owning reference
drops as soon as the broadcast loop ends.

## Profiling knobs

All are env-gated and cost nothing when unset. Pass them to `run_tests.sh`;
`docker-compose` forwards them and the daemon logs land in `logs/`.

| var | what it reports |
|-----|-----------------|
| `CLIO_COLL_PROF=1` | `[COLLPROF]` collective stages: member skew, flusher notice delay, aggregate build/exec, broadcast |
| `CLIO_NET_QPROF=1` | `[NETQPROF]` net-queue enqueue→pop wait per priority, and server-side residency (arrival → response queued) |
| `CLIO_NET_TRACE=1` | `[NETTRACE]` serialize/transmit/recv timings and the send-poll tick rate |
| `CLIO_WORKER_RATE=1` | `[WORKERRATE]` wall time per worker-loop iteration (includes parked time — compare workers, don't read as CPU) |

`CLIO_NET_QPROF` is the heavy one: its server-residency tracker takes a mutex
and hits an `unordered_map` on every task arrival and every response enqueue.
That is fine for attributing a stage, but it distorts absolute latency — do not
read headline numbers off a run with it enabled. `CLIO_COLL_PROF` is cheap (a
few timestamps per collective) and safe to leave on while measuring.

Note all four test for a non-EMPTY value, not a non-null pointer. A harness that
forwards a knob as `"${VAR:-}"` sets it to the empty string when unset, and a
null check leaves profiling permanently on — which silently happened here and
put clock reads into "clean" measurements.

## Notes

- Node 1 is the mpirun launcher and sets up passwordless SSH; nodes 2-4 serve
  `sshd` and wait for node 1's "done" flag. A worker must not exit while a peer
  is still in a collective — its daemon would vanish mid-barrier and the run
  would hang rather than fail, which is the harder failure to read.
- The daemons log to `/tmp/clio_daemon.log` inside each container rather than
  stdout; at `info` level the periodic scheduler report buries the harness's own
  progress in `docker logs`.
- Containers carry `SYS_PTRACE` so a hung rank can be inspected with `gdb` via
  `docker exec` — which is how the bug above was found.
