# Separate-CUDA-context probes

Two standalone A100 probes that established *why* running the compressor in a
dedicated CUDA context fixes the on-device page-fault deadlock (they answer the
two questions the fix depends on). Build with
`nvcc -arch=sm_80 <probe>.cu -o <probe> -lcuda` and run on a GPU.

- **`preemption_probe.cu`** — launches a spin-wait kernel in the PRIMARY context
  (never signalled), then a worker kernel in a SEPARATE context. Result: the
  worker completes while the primary kernel spins => **A100 compute preemption
  lets a second context run while the first spin-waits**. (Streams within one
  context do NOT preempt — that is why the earlier stream-only attempts
  deadlocked.)

- **`cross_context_mem_probe.cu`** — a kernel/`cudaMemcpy` in a SEPARATE context
  writes to and reads from a buffer `cudaMalloc`'d in the PRIMARY context; the
  host reads back the written value. Result: **cross-context UVA works**, so the
  compressor's decompress (its own context) can write the decompressed page
  straight into the vector's HBM slot (allocated in the app's primary context).

Together these justify `cuszp.h`'s `ContextScope`: the compressor's cuSZp work
runs in a dedicated context, preempting the app's spin-waiting fault kernel and
writing results into the app's HBM pages via UVA.
