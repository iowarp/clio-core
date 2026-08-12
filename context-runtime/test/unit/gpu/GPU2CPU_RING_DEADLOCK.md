# gpu2cpu ring deadlock: the consumer stops advancing `head_`

**Status:** root-caused at the ring level, **not fixed**. The precise memory-model mechanism
behind the lost store is **not** proven — see [Not proven](#what-is-not-proven).

**Reproduced by:** `test_gpu2cpu_backpressure_gpu.cc`, case 2 (`gpu2cpu: producer overruns the
ring`), gated behind `CLIO_TEST_GPU2CPU_OVERRUN=1`.

**Symptom in the wild:** the Gray-Scott `[gsbench_async]` arm
(`context-transfer-engine/adapter/kvhdf5/test/e2e/gray_scott_threeway_bench.cu`) hangs forever
whenever `chunks * snaps > queue_depth`.

---

## One-paragraph summary

The gpu2cpu lane is a `RING_BUFFER_WAIT_FOR_SPACE` ring. When a GPU producer outruns it, the
producer does not fail — it **parks on the device**, spinning in `Emplace()` until the CPU
consumer frees a slot. The bug is that **the CPU consumer stops advancing `head_` the moment a
producer parks.** The parked producers then spin forever on a `head_` that will never move, and
the consumer, whose `Pop` keys off that same `head_`, re-reads one already-consumed slot forever.
Circular wait. Nothing drains, ever.

`chunks * snaps > queue_depth` is not incidental — it is exactly the condition for a producer to
park, and therefore **the bug's precondition**.

---

## The decisive evidence

The repro is deliberately minimal: **no bdev, no D2H copy, and no CUDA at all on the consumer
side** — the chimod it submits to is pure arithmetic (`result = value * 2 + gpu_id`). This strips
away every subsystem that earlier (wrong) theories blamed. Submitting 3072 producer-only tasks
into a depth-1025 ring wedges in exactly this state:

```
head=0  tail=3072  depth=1025
slots the CPU sees READY            = 1023      (not ready: slot 0 and slot 1024)
tasks actually executed by the CPU  = exactly ONE
```

Read that carefully, because each number rules something out:

| Observation | What it proves |
|---|---|
| `tail=3072` | All 3072 producers claimed a slot. `tail_.fetch_add_system` **is** visible to the CPU — system-scope atomics work. |
| 1023 slots READY | Those producers landed their entries and the CPU **can see them**. The data is there. It is not a visibility problem on the entries. |
| slot 1024 not ready | **Correct, not a bug.** Its producer is legitimately parked waiting for space. |
| slot 0 not ready | The consumer popped it and cleared its ready bit via the CAS in `PopUnlocked`. |
| executed == 1 | The consumer ran `PopUnlocked` to completion **exactly once** — so it did reach `head_.store_system(head + 1)`. |
| **`head` still 0** | …and yet the head it stored (`1`) is **gone**. The store was lost. |

That last line is the bug. After one successful pop, `head_` reads back as `0`. Every subsequent
`Pop` recomputes `idx = head % size` → slot 0, finds `IsReadySystem() == 0` (it already consumed
that slot), returns `false`, and gives up. 1023 ready entries sit there forever.

**Control:** submit `depth - 1` tasks, so no producer *ever* has to wait for space. All 1024 drain
cleanly and `head_` advances perfectly. The failure appears **precisely** when the first producer
parks — which is why case 1 of the test passes and case 2 does not.

---

## The code path

`context-transport-primitives/include/clio_ctp/data_structures/ipc/ring_buffer.h`

Producer, `Emplace()` (~line 500) — note it claims the slot **before** checking for space:

```cpp
u64 head = head_.load();
u64 tail = tail_.fetch_add_system(1);   // slot claimed unconditionally
if constexpr (WaitForSpace) {
  size_t size = tail - head + 1;
  while (size >= queue.size()) {
    head = head_.load_device();          // spin. head_ is advanced by the CPU.
    size = tail - head + 1;
  }
}
```

Consumer, `PopUnlocked()` (~line 574):

```cpp
u64 head = head_.load_system();
u64 tail = tail_.load_system();
if (head >= tail) return false;
entry_type& entry = queue_[head % queue_.size()];
if (!entry.IsReadySystem()) return false;              // <-- where it gives up forever
if (!entry.flags_.bits_.compare_exchange_strong_system(expected, 0u)) return false;
val = entry.data_;
entry.data_ = T();
head_.store_system(head + 1);                          // <-- the store that goes missing
```

### The strongest lead for *why* the store is lost

`ring_buffer.h` lines 218–220:

```cpp
entry_vector queue_;   /**< Internal vector storing entries */
head_type head_;       /**< Consumer head pointer */
tail_type tail_;       /**< Producer tail pointer */
```

`head_` and `tail_` are **adjacent, unpadded 64-bit atomics — guaranteed to share a cache line.**

Meanwhile the GPU is hammering that exact line: thousands of threads doing `tail_.fetch_add_system`
*and* `head_.load()` / `head_.load_device()`. The plain `head_.load()` on line 504 is a
**device-scope** load, which is free to cache the line in a GPU L2 that is **not coherent** with
the host. The working hypothesis is that a GPU-L2 writeback of that shared line clobbers the CPU's
`head_.store_system(head + 1)` with a stale `head_ == 0`.

This is a hypothesis with a cheap experiment attached: **pad `head_` and `tail_` onto separate
cache lines** and re-run case 2. It is *not* yet proven — see below.

---

## Why `chunks=64` works and `chunks=128` hangs

`chunks * snaps` **is** the number of gpu2cpu submissions.

| config | submissions | vs `queue_depth` (1024) | result |
|---|---|---|---|
| 64 chunks × 12 snaps | 768 | fits | works — **no producer ever parks** |
| 128 chunks × 12 snaps | 1536 | overruns | **hangs** — a producer parks in `Emplace` |

Raising `queue_depth` to 16384 "fixes" 128 and 256 chunks for the same reason: nothing ever parks.
**This is not a fix.** It only avoids arming the hazard, and leaves it armed for any workload that
outruns the consumer.

---

## The drain-kernel contradiction, resolved

The obvious story — *"a resident spinning GPU kernel starves the CPU consumer"* — is **false**, and
it had to be killed because `TwDrainKernel` also spins on the GPU while the CPU drains, and *that*
works fine.

Timestamping both on a common clock in the **working** run (chunks=128, queue_depth=16384):

```
HOST: launching 12 drain kernels     t = T
memcpy#0 ENTER                       t = T + 6.8 ms
memcpy#0 EXIT                        t = T + 101.5 ms
... all 1536 copies EXIT between T+101ms and T+670ms
copies that ENTER before the drain launch = 0
copies that EXIT  before the drain launch = 0
```

**Every single copy completes *after* the drain kernels are launched and spinning**, with 11 more
queued behind the resident one. A spinning kernel demonstrably does **not** starve the CPU.

The fire kernel is fatal not because it spins, but because **it spins holding the ring hostage.**

---

## Refuted theories — do not re-derive these

1. **"A resident spinning kernel starves the CPU's D2H copy."** *False.*
   `context-transport-primitives/test/unit/gpu/test_gpu_spin_kernel_copy_gpu.cc` exists specifically
   to keep this dead: it proves a spinning kernel permits **28k+ concurrent D2H copies/sec** across
   device/managed × pinned/pageable, using `DeviceAwareMemcpy`'s exact pattern. Keep it green.
2. **OOM / too many GPU backends.** Device bytes are *constant* across the chunk sweep (total
   payload fixed at 1875 MB), and 256 chunks works fine at `queue_depth=16384`.
3. **`--default-stream=per-thread`** on the benchmark TU. Still hangs.
4. **`CUDA_DEVICE_MAX_CONNECTIONS=32`** (hardware work-queue aliasing). Still hangs.
5. **Pageable-vs-pinned staging**, and CLIO's registered backend being special. Both ruled out: a
   probe copy from a *fresh, unrelated `cudaMalloc` buffer* into pinned memory on a fresh
   non-blocking stream also hangs.

### Explicitly NOT settled

The `head_.load_device()` → `head_.load_system()` change in `Emplace`'s spin was tried against
*gsbench* and the hang persisted, so it was reverted. But gsbench carries a second, confounding
symptom (below). **Re-test that change against `test_gpu2cpu_backpressure_gpu.cc` case 2** — a far
cleaner instrument — before concluding anything about it.

---

## What is NOT proven

- **Why the store is lost.** The GPU-L2-writeback / `head_`+`tail_` false-sharing story above is a
  *hypothesis*, not a demonstrated mechanism. It fits every observation and has an obvious
  experiment (pad the two fields apart), but nobody has isolated it.
- **An unexplained side-symptom in gsbench.** There, the worker's *proximate* blocker is different:
  it pops one task and parks **inside `cuMemcpyAsync`** (1.28 MB D2H, `FsBdevTransport::WriteBlocks`
  → `DeviceAwareMemcpy`) and never returns. `cuda-gdb` shows a single active kernel
  (`TwSnapFireKernel`, grid (1,1,1), on **1 SM** — 127 SMs idle), GPU utilization 100% but
  **memory utilization 0%: no DMA in flight at all.** This was **not** reproducible standalone and
  the driver mechanism is **unnamed**. It only ever occurs when a producer is parked in `Emplace`,
  so it is gated behind the same precondition — but it is not explained.

---

## The fix

**A GPU kernel must never park unboundedly waiting on a CPU consumer.** That is the invariant the
current design violates.

Switch the gpu2cpu lane to `RING_BUFFER_ERROR_ON_NO_SPACE` and make the producer yield the device:

```cpp
// `next` lives in pinned host memory.
__global__ void Fire(GpuDatasetHandle h, uint32_t* next) {
  for (uint32_t c = *next; c < h.Count(); ++c)
    if (!h.WriteAsync(c)) { *next = c; return; }   // ring full -> EXIT, don't spin
  *next = h.Count();
}
```

The host re-fires from `*next` until it reaches `Count()`. The kernel exits, the ring never enters
the broken state, and `queue_depth` goes back to being a pure performance knob.

This removes the precondition for **both** the ring wedge and the unexplained memcpy stall, so it
resolves the observed hang regardless of which mechanism is really behind the lost store.

### Two things that must be handled with it

1. **`IpcGpu2Cpu::SendIn` discards `Push`'s return value.**
   `context-runtime/include/clio_runtime/ipc/ipc_gpu2cpu_impl.h`: `qlane.Push(task_future);`
   Under `WAIT_FOR_SPACE` this is latent (Push cannot fail). The instant you switch to
   `ErrorOnNoSpace` it becomes a **silent task drop** — and a dropped task hangs `WriteWait`
   forever. The bool must be propagated up through `SubmitAsync` / `WriteAsync`.
2. **`TwDrainKernel` is the same unbounded GPU-waits-for-CPU pattern.** It is safe today only
   because the ring never fills. It is not independently safe.

### Prefer fixing the ring, not just routing around it

The producer-side fix above stops *this* caller from tripping the bug, but a blocking producer that
cannot be drained is a landmine for any future user of the GPU lane. The ring itself should either
be made correct under backpressure, or `WAIT_FOR_SPACE` should be rejected outright for
GPU-producer lanes.

---

## Tests

| Test | Status |
|---|---|
| `context-runtime/test/unit/gpu/test_gpu2cpu_backpressure_gpu.cc` case 1 — *producer fills the ring exactly* | **passes**; guards the normal path |
| `context-runtime/test/unit/gpu/test_gpu2cpu_backpressure_gpu.cc` case 2 — *producer overruns the ring* | **reproduces this bug**; opt-in via `CLIO_TEST_GPU2CPU_OVERRUN=1`. Fails cleanly with a ring-state dump rather than hanging. |
| `context-transport-primitives/test/unit/gpu/test_gpu_spin_kernel_copy_gpu.cc` | **passes**; pins the CUDA invariant so refuted theory #1 cannot be re-derived |

Case 2 is gated only because `simple_test.h` has no Catch2 `[.]` hidden-tag support, so an env var
was the only way to keep CI green. **When the ring is fixed, delete the gate** so the overrun case
runs by default: a blocking producer MUST be drainable.

Reproduce:

```sh
CLIO_TEST_GPU2CPU_OVERRUN=1 ./bin/test_gpu2cpu_backpressure_gpu
```
