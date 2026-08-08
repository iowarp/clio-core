# gpu2cpu queue in device memory, batch-copied by the CPU

Branch: gpu2cpu-devicemem-queue (off gpu-vector-rewrite). Merge back when the
full gpu_vector + compressor + bdev suites pass AND the flush/read benches show
equal-or-better numbers than the managed-memory baseline recorded below.

## Why

The queue currently lives in managed memory (after ad07a2e9; it was
cudaMallocHost, where this GPU cannot legally perform the system-scoped ring
atomics at all -- cudaDevAttrHostNativeAtomicSupported=0). Every device Send
still crosses PCIe for the push, and every Wait() polls across the bus.
Measured overhead is ~50-66us per put, which caps the GPU flush plateau at
~6.5 GB/s against a 12.9 GB/s PCIe ceiling.

## Design

- Ring buffers (requests and responses) live in DEVICE memory.
  * Device push = device-scope atomics only (atomicAdd on head). No .sys, no
    PCIe round trip. Legal on every CUDA GPU.
  * gpu::Future<T>::Wait() polls a completion word in DEVICE memory: local
    reads, no bus traffic.
- CPU worker drains by BATCH:
  * cudaMemcpyAsync the ring header (head/tail words) on a DEDICATED stream,
    then one copy for the whole span of new entries [tail, head). One bus
    round trip amortizes N requests.
  * Executes tasks as today (RecvIn wrapping unchanged downstream).
  * Responses: write completions into a host staging array, one
    cudaMemcpyAsync H2D for the batch, then a second small H2D that publishes
    the response-ring head AFTER the payload copy completes (stream order
    guarantees payload-before-publish; the device sees the head move only
    when the completions are readable). Preserves the "completion flag last"
    invariant per task.
- STREAMS: the queue's copies use their own stream (or small pool), NEVER the
  bdev I/O stream pool (GpuApi::BorrowStream). Two reasons:
  * a queue copy stuck behind a long data copy would add latency to every
    task on the GPU;
  * the resident-kernel deadlock class (see cuda-stream-create-deadlock
    memory + WarmStreamPool in gpu2cpu_init_hip.cc): create the queue stream
    at ServerInitGpuQueues time, before any user kernel can be resident.
- Poll cadence: keep the worker's existing poll loop; each poll now costs one
  small D2H (header) instead of a managed-memory read. Batch size = however
  many entries arrived since last poll (bounded by ring capacity).

## Invariants that MUST survive (each was a real bug)

1. Any thread may Send and any thread may Wait (f691c304) -- no threadIdx
   guards.
2. fetching published BEFORE page_num on slot claim (vector side, unchanged).
3. Completion visible to device only after the full task POD writeback
   (payload-before-publish ordering above).
4. Multi-producer safety on the device ring: atomicAdd claim, then write
   entry, then publish a per-entry ready flag (device-scope release); the CPU
   batch reader skips entries whose ready flag is unset and re-reads next poll.
5. Task PODs themselves stay where they are (registered client backends);
   only the QUEUE moves to device memory.

## Baseline to beat (this machine, frosty_wu)

- flush bench, 8 blocks, 1MB pages, 16MB/block, spin 0: 6.0-6.5 GB/s,
  ~50-66us/put overhead
- read path: 492 MB/s demand faults (1MB pages, sequential per block)
- sync fault round trip ~2ms (1MB); CPU-side GetBlob is 1-5us of that
- suites: 23/23 ctest (gpu_vector+compressor+bdev), compute-sanitizer 0 errors

## Files

- context-runtime/src/gpu/gpu2cpu_init_hip.cc  (queue allocation -> device
  mem; create dedicated queue stream at init)
- context-runtime/include/clio_runtime/ipc/ipc_gpu2cpu_impl.h (SendIn push,
  Wait poll -- device-local now)
- context-runtime/src/ipc/ipc_gpu2cpu.cc + worker.cc (CPU drain -> batched
  copies; keep ONE-batch-per-poll bounded so the single-lane worker still
  yields -- do NOT reintroduce the d265bdb3 batch-drain deadlock shape)

## Status: IMPLEMENTED AND DEFAULT (ce118908)

Gate cleared:
- 23/23 ctest (gpu_vector + compressor + bdev) with the ring active
- compute-sanitizer 0 errors on test_gpu_vector_smoke
- correct through 64 blocks and multi-oversub-64
- faster on every workload measured:

| workload                | managed | ring | |
|-------------------------|---------|------|-|
| sync demand faults      | 493 MB/s | 801 MB/s | 1.6x |
| read async overlap      | 175 ms   | 138 ms   | 1.27x |
| flush 8 blocks          | 5859 MB/s | 6051 MB/s | +3% |
| flush 64 blocks         | 6541 MB/s | 6558 MB/s | flat |

The read path gains and the flush path does not, which is the design working
as predicted: a flush already pipelines many puts so its per-submission
crossing hides behind data transfer, while a demand fault is a BLOCKING round
trip whose latency is exactly the submit atomic plus the completion poll --
the two PCIe crossings the ring removes.

CLIO_GPU_DEVRING=0 restores the managed queue for bisecting.

### Not yet done
- Response batching. Completions are still written per task by SendOut; the
  design's batched H2D response path is unimplemented. Reads would gain again
  from it (the completion poll is now local, but the CPU still crosses per
  completion).
- Flush is bounded by per-put TASK cost (~50-66us), not the transport, so it
  needs 8MB pages or a region-level put to approach the 12.9 GB/s PCIe ceiling.
