# cuSZp patch: stream-ordered allocations

`cuszp-stream-ordered-alloc.patch` makes cuSZp's 1D-f32 **outlier**
compress/decompress entry functions fully stream-ordered:
`cudaMalloc`/`cudaMemset`/`cudaFree` → `cudaMallocAsync`/`cudaMemsetAsync`/
`cudaFreeAsync` on the caller-supplied stream, and the compressed-size read-back
uses `cudaMemcpyAsync` + `cudaStreamSynchronize(stream)` instead of a synchronous
`cudaMemcpy`.

**Why:** stock cuSZp device-synchronizes (plain `cudaMalloc`/`cudaMemcpy`) inside
every (de)compress, which blocks the host worker until the whole device is idle.
That deadlocks the compressed `gpu_vector` fault/eviction path, where a caller
kernel spin-waits on-device for the very (de)compress being serviced. Stream-
ordered allocation removes those device syncs so the (de)compress runs on its own
non-blocking stream, concurrently with the spin-waiting kernel.

Apply against the cuSZp source tree (the `CLIO_CUSZP_ROOT` build):
    cd <cuSZp> && git apply <this-dir>/cuszp-stream-ordered-alloc.patch && \
      cmake --build build && cmake --install build

NOTE: necessary but not sufficient on its own to fix the on-device fault
deadlock — the compressed bytes are still staged host→device from pageable
memory (a synchronous copy) and first-use CUDA allocations block during the spin.
Fully closing the on-device fault needs the preallocation work (device-resident
compressed region + pre-warmed scratch passed via the Context object). The
host-orchestrated `Vector::FaultAllSync()` path is unaffected and works today.
