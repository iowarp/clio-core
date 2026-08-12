#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // expose O_DIRECT from <fcntl.h> (raw-arm cache-bypass parity)
#endif
/*
 * THREE-WAY Gray-Scott I/O benchmark (the paper figure the advisor asked for):
 * ONE shared computation, THREE storage paths, wall-clock compared:
 *
 *   - raw   : no CLIO. GPU compute -> D2H to pinned host -> pwrite + fsync to disk.
 *   - sync  : CLIO, fused submit-AND-WAIT snapshot (GPU blocks on each PutBlob).
 *   - async : CLIO, fire-all snapshot + drain-at-end (server I/O overlaps GPU compute).
 *
 * SAME COMPUTATION across all three by construction: every arm runs the identical
 * `GsStepKernel` (one thread per cell) through the identical timed `RunSim` loop;
 * only the per-snapshot "sink" differs. So the comparison isolates the I/O + storage
 * backend, not the math.
 *
 * ROUTING AROUND THE iowarp ~16-large-backend ceiling (ADVISOR-REPORT §6a; the limit
 * is by COUNT not bytes, and dataset REUSE hangs — see refire probe): each arm reaches
 * the ~2 GB target with FEWER, BIGGER snapshots (<=~12 distinct datasets, each large),
 * keeping the proven fresh-dataset-per-snapshot recipe. The sync and async arms run in
 * SEPARATE processes (one arm per TEST_CASE invocation), so neither exceeds the ceiling.
 *
 * FAIR STORAGE: the CLIO arms can target a kFile bdev (O_DIRECT to a real file, cache-
 * bypassing) to match the raw arm's disk, or a kRam bdev for a RAM baseline — selected
 * by GSBENCH_BDEV. The raw arm always writes real files + fsync.
 *
 * All knobs are ENV VARS so scaling to 2 GB / switching to disk needs NO recompile:
 *   GSBENCH_N            grid dim (NxN float32)              default 512
 *   GSBENCH_CHUNKS       chunks per snapshot dataset         default 4
 *   GSBENCH_SNAPS        number of snapshots (<= ~12!)       default 4
 *   GSBENCH_STEPS_PER    sim steps between snapshots         default 8
 *   GSBENCH_BDEV         ram | pinned | file  (CLIO arms)    default ram
 *   GSBENCH_BDEV_CAP_MB  bdev capacity (MB)                  default 512
 *   GSBENCH_BDEV_PATH    kFile path (CLIO arms)              default ./gsbench_bdev.dat
 *   GSBENCH_DISK_DIR     raw-arm output dir                  default ./gsbench_raw_out
 *
 * Each arm prints ONE machine-parseable RESULT line (ms, MB, MB/s, checksum). A wrapper
 * runs all three processes and builds the relative table; the shared checksum proves all
 * three computed identical bytes. Cases HIDDEN ([.]); CLIO arms RUN AT num_threads=1.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/layout.h>           // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>
#include <clio_cte/kvhdf5/tag_path.h>         // CanonicalTag

#include <cuda_runtime.h>
#include <cooperative_groups.h>   // Option A: grid.sync() across a resident cooperative grid

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#ifndef O_DIRECT
#define O_DIRECT 040000  // Linux x86/arm value; fallback if _GNU_SOURCE didn't expose it
#endif
#if GSBENCH_HAVE_HDF5
#include <hdf5.h>
#endif
#endif

using kvhdf5::byte_t;

namespace {
struct GsParams { float Du, Dv, F, k, dt; };
}  // namespace

// ---- shared computation: identical for every arm ---------------------------

// One Gray-Scott step, one thread per cell, periodic BCs. Pure CUDA.
__global__ void GsStepKernel(const float* u, const float* v, float* un, float* vn,
                             GsParams p, unsigned N) {
    unsigned cells = N * N;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= cells) return;
    unsigned x = gid % N, y = gid / N;
    unsigned xm = (x == 0) ? (N - 1) : (x - 1);
    unsigned xp = (x == N - 1) ? 0u : (x + 1);
    unsigned ym = (y == 0) ? (N - 1) : (y - 1);
    unsigned yp = (y == N - 1) ? 0u : (y + 1);
    float uc = u[gid], vc = v[gid];
    float lap_u = u[y*N+xm] + u[y*N+xp] + u[ym*N+x] + u[yp*N+x] - 4.f*uc;
    float lap_v = v[y*N+xm] + v[y*N+xp] + v[ym*N+x] + v[yp*N+x] - 4.f*vc;
    float uvv = uc * vc * vc;
    un[gid] = uc + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - uc));
    vn[gid] = vc + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vc);
}

// ---- CLIO snapshot kernels (device-facing handle) --------------------------

// Bulk stage a snapshot into its registered per-chunk device backends with a FULL grid
// (gridDim.y = chunk, gridDim.x = blocks/chunk), saturating HBM. This is split OUT of the
// submit kernels: the old design fused the copy into the single <<<1,256>>> CLIO-producer
// block, so 156 MB moved at ~1 GB/s (~160 ms/snapshot) and dominated BOTH arms — dwarfing
// the actual compute and masking any async advantage. `src` is the flat masked grid; chunk
// c's bytes are src[c*size .. c*size+size). Word-wise (float-grid sizes are 4-byte
// multiples). Being its OWN completed kernel gives kernel-boundary ordering, so the staged
// writes are visible to the subsequent submit kernel and the server's readback — no
// __threadfence_system needed (that fence was only for the old same-kernel copy+enqueue).
__global__ void TwCopyKernel(kvhdf5::GpuDatasetHandle h, const byte_t* src) {
    uint32_t c = blockIdx.y;
    uint64_t n = h.Size(c);
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src + uint64_t(c) * n);
    uint32_t* d = reinterpret_cast<uint32_t*>(h.Data(c));
    uint64_t words = n >> 2;
    uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = gid; i < words; i += stride) d[i] = s[i];
}

// SYNC submit: per chunk, fused Write(c) = submit-AND-WAIT (GPU blocks). Data already
// staged by TwCopyKernel. GRID-STRIDE over chunks: block b submits chunks b, b+gridDim,
// ... (each block's thread-0 enqueues via its own per-block IpcManager — the framework's
// documented one-block-per-chunk pattern). gridDim.x == 1 reproduces the old single-block
// serial loop byte-for-byte; gridDim.x == Count() gives one CUDA block per chunk. This is
// the "number of GPU blocks" axis. __syncthreads is per-block, so unequal per-block trip
// counts (Count() not divisible by gridDim.x) are safe.
template <bool kProbing>
__global__ __launch_bounds__(256) void TwSnapSyncKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.Write<kProbing>(c);
        __syncthreads();
    }
}

// ASYNC FIRE only: fire every chunk's PutBlob, no wait. Data already staged by
// TwCopyKernel. Drained later so the puts run on the server WHILE the subsequent sim
// steps run on the GPU. Grid-stride over chunks (see TwSnapSyncKernel).
template <bool kProbing>
__global__ __launch_bounds__(256) void TwSnapFireKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.WriteAsync<kProbing>(c);
        __syncthreads();
    }
}

// Explicitly instantiate both submit-kernel variants. Without this nvcc does not
// emit/register the device code for a __global__ template instantiation that is
// only implicitly ODR-used (via <<<>>> launch / address-of), so a launch of the
// missing variant fails at runtime with cudaErrorInvalidDeviceFunction (98).
template __global__ void TwSnapSyncKernel<false>(kvhdf5::GpuDatasetHandle);
template __global__ void TwSnapSyncKernel<true>(kvhdf5::GpuDatasetHandle);
template __global__ void TwSnapFireKernel<false>(kvhdf5::GpuDatasetHandle);
template __global__ void TwSnapFireKernel<true>(kvhdf5::GpuDatasetHandle);

// ---- READER: GPU-initiated read-back + PDF --------------------------------------------
// The reader case: read every snapshot back OUT of storage and compute the probability
// density of the v concentration field. That is a real Gray-Scott post-processing step (the
// distribution's shape is what tells you which pattern regime you are in), and it is a read
// workload that cannot be faked -- computing it must touch every byte that was written.
//
// ONE GLOBAL histogram is accumulated across ALL snapshots, so its bin counts are a cross-arm
// READ correctness gate exactly like the write checksum: every reader must observe the same
// bytes, therefore the same PDF. A reader that silently short-reads shows up as a mismatch.
//
// Read and histogram are SEPARATE kernels on purpose. h.Read(c) is the proven device Get (the
// same pattern as multichunk_gpu_putget_test's McReadKernel); consuming the freshly-landed
// bytes inside that same kernel would need acquire semantics against the CPU worker's DMA, so
// we take the kernel boundary as the visibility barrier instead.
constexpr unsigned kHistBins = 256;

// GPU-initiated GetBlob for every chunk, grid-stride. Thread-0 submits AND waits (the
// handle's internal producer guard); __syncthreads keeps the block together per chunk.
__global__ __launch_bounds__(256) void TwReadKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.Read(c);        // thread-0 only (internal guard)
        __syncthreads();
    }
}

// ASYNC read: device-stamp this snapshot's tag (SetTag covers the GET slot too), then FIRE the
// Get for every chunk WITHOUT waiting. Stamped on the DEVICE, not by a host Rearm, because the
// host runs ahead here: a host re-tag would race the CPU worker still reading tag_id_ for the
// previous in-flight Get -- the same hazard the write path hit.
__global__ __launch_bounds__(256) void TwReadFireKernel(kvhdf5::GpuDatasetHandle h,
                                                        const clio::cte::core::TagId* tags,
                                                        unsigned si) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.SetTag(c, tags[si]);
        __syncthreads();
        h.ReadAsync(c);   // thread-0 only (internal guard); no wait
        __syncthreads();
    }
}

// Drain a fired snapshot's outstanding gets (the read-side TwDrainKernel).
__global__ __launch_bounds__(32) void TwReadDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.ReadWait(c);
        __syncthreads();
    }
}

// Bin one chunk of floats into the global histogram. Shared-memory privatisation first, so
// the global atomics are per-bin-per-block instead of per-element. Gray-Scott v lives in
// [0,1]; anything outside clamps into the end bins rather than being dropped.
__global__ __launch_bounds__(256) void HistKernel(const float* __restrict__ data, uint64_t n,
                                                  unsigned long long* __restrict__ hist) {
    __shared__ unsigned long long smem[kHistBins];
    for (unsigned i = threadIdx.x; i < kHistBins; i += blockDim.x) smem[i] = 0ULL;
    __syncthreads();
    const uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = gid; i < n; i += stride) {
        int b = int(data[i] * float(kHistBins));
        b = (b < 0) ? 0 : ((b >= int(kHistBins)) ? int(kHistBins) - 1 : b);
        atomicAdd(&smem[b], 1ULL);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < kHistBins; i += blockDim.x)
        if (smem[i]) atomicAdd(&hist[i], smem[i]);
}

// Drain a fired snapshot's outstanding puts. Grid-stride over chunks (see TwSnapSyncKernel).
__global__ __launch_bounds__(32) void TwDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.WriteWait(c);
        __syncthreads();
    }
}

// REUSE arm's fire kernel: like TwSnapFireKernel (async fire, no wait), but first
// device-stamps each chunk's destination tag from a host-pre-built table. This is
// the tag-table addressing that lets a FIXED set of reused buffer groups serve all
// snapshots — 2 groups instead of one dataset per snapshot — while each snapshot
// still lands in its OWN dataset (tag_table[snap]) under the same chunk-coordinate
// key (no `_pi` suffix). SetTag writes tag_id_ on the pinned task POD; WriteAsync's
// __threadfence_system publishes it before the task is enqueued (see SetTag). The
// group's buffers were just refilled by TwCopyKernel; the caller drains this group's
// PRIOR snapshot out of it first (drain-before-refill) so the reuse is race-free.
// Non-probing only — the reuse arm is not on the submit-probe path. Grid-stride over
// chunks (see TwSnapSyncKernel).
__global__ __launch_bounds__(256) void ReuseFireKernel(
    kvhdf5::GpuDatasetHandle h, const clio::cte::core::TagId* tag_table, uint32_t snap) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.SetTag(c, tag_table[snap]);
        h.WriteAsync</*Probing=*/false>(c);
        __syncthreads();
    }
}

// ---- OPTION A: the PERSISTENT (resident, cooperative) producer kernel --------
//
// ONE cudaLaunchCooperativeKernel runs the ENTIRE snapshot loop in-kernel — the "move the
// snapshot loop into a persistent kernel" design (DESIGN §7 "Option A"). The relaunch arms get a
// grid-wide barrier between timesteps for free from the kernel-launch boundary; a resident kernel
// must supply it explicitly with cooperative-groups grid.sync(), because the global Gray-Scott
// stencil reads neighbours that live in other blocks (periodic BCs). At each snapshot boundary it
// device-stamps tag_table[s] and fires the Put into reused group s%G (G = ngroups, bounded),
// draining that group's snapshot s-G first (the acquire). Same bounded memory + device
// self-addressing as the reuse arm; the difference under test is resident-cooperative vs
// relaunched.
//
// ACQUIRE PLACEMENT (why the drain is INSIDE the fill loop, not above the compute). The obvious
// shape — drain group s%G, grid.sync(), then compute — is what this kernel used to do, and it
// costs almost all of the overlap the async path exists to buy. Measured at N=4096 / 32 chunks /
// 8 snaps / steps_per=20: persistent 83.6 ms against relaunch 65.5 ms, where a per-step-cost
// decomposition says a properly overlapped persistent kernel should land at ~60 ms (the I/O
// floor). The drain gated compute even though COMPUTE NEVER TOUCHES THE I/O BUFFERS — it works
// on the u/v grids, while the Puts read h.Data(c). So the wait had no reason to precede it.
//
// Two things follow, and both are safety arguments, not just perf:
//   1. Compute first, drain later. Moving the acquire below the compute gives each in-flight
//      snapshot a FULL extra compute phase to land in before anyone waits on it.
//   2. No grid.sync() around the acquire. Block b drains chunks {b, b+G_grid, ...} and refills
//      exactly that same set (both loops stride by gridDim.x), so the drain/refill dependency is
//      BLOCK-LOCAL. A grid-wide barrier was never required for it; the load-bearing sync is the
//      __syncthreads() after the thread-0-only WriteWait, which is what stops a non-zero thread
//      racing into the fill while chunk c's prior Put is still draining (same invariant
//      WritePipelined documents).
// The grid.sync() AFTER the fire is still required and stays: the next snapshot's ping-pong
// swap makes step 2 write into the buffer holding snapshot s, which the fill may still be reading.
//
// DEADLOCK SAFETY (DESIGN §8): the in-kernel WriteWait (acquire + tail drain) spins on the CPU
// server. A resident cooperative grid occupies the whole device, so the server MUST be able to
// complete Puts WITHOUT any device operation — i.e. the data backend MUST be kPinnedHost (server
// reads pinned data + sets the pinned completion flag, no D2H). With kDeviceMem the server's D2H
// cannot run while the grid is resident and the acquire deadlocks. The launcher enforces pinned
// data. Prewarm (cudaFuncGetAttributes) before launch is a correctness requirement (cold-launch).
namespace cg = cooperative_groups;
// Async == true: fire the snapshot's Puts and defer draining to the next reuse / the tail (the
// double-buffered producer, the persistent analog of gpuh5). Async == false: fire-AND-WAIT each
// chunk in-kernel (h.Write), so the GPU blocks on every PutBlob and there is no overlap — the
// persistent analog of gpuh5_sync. The acquire + tail drain stay in both (idempotent / instant
// under sync, since each snapshot's Put is already complete before its group is reused).
template <bool Async>
__global__ void GsPersistentKernel(
    float* u0, float* v0, float* u1, float* v1,
    const kvhdf5::GpuDatasetHandle* groups, unsigned ngroups,
    const clio::cte::core::TagId* tag_table, const uint32_t* mask,
    GsParams p, unsigned N, unsigned num_snaps, unsigned steps_per) {
    CLIO_GPU_INIT(groups[0].info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    cg::grid_group grid = cg::this_grid();
    const unsigned cells = N * N;
    const unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gstride = gridDim.x * blockDim.x;

    float* uc = u0; float* vc = v0; float* un = u1; float* vn = v1;
    for (unsigned s = 0; s < num_snaps; ++s) {
        // COMPUTE steps_per Gray-Scott steps in-kernel; grid.sync() is the grid-wide barrier
        // between steps (identical math to GsStepKernel). Ping-pong via local pointer swap
        // (every thread swaps identically, so uc/vc stay consistent grid-wide).
        for (unsigned t = 0; t < steps_per; ++t) {
            for (unsigned gid = gtid; gid < cells; gid += gstride) {
                unsigned x = gid % N, y = gid / N;
                unsigned xm = (x == 0) ? (N - 1) : (x - 1);
                unsigned xp = (x == N - 1) ? 0u : (x + 1);
                unsigned ym = (y == 0) ? (N - 1) : (y - 1);
                unsigned yp = (y == N - 1) ? 0u : (y + 1);
                float ucv = uc[gid], vcv = vc[gid];
                float lap_u = uc[y*N+xm] + uc[y*N+xp] + uc[ym*N+x] + uc[yp*N+x] - 4.f*ucv;
                float lap_v = vc[y*N+xm] + vc[y*N+xp] + vc[ym*N+x] + vc[yp*N+x] - 4.f*vcv;
                float uvv = ucv * vcv * vcv;
                un[gid] = ucv + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - ucv));
                vn[gid] = vcv + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vcv);
            }
            grid.sync();
            float* tu = uc; uc = un; un = tu;
            float* tv = vc; vc = vn; vn = tv;
        }

        // SUBMIT: vc now holds snapshot s. Drain-then-fill group (s%G)'s chunk buffers (masked,
        // matching the relaunch arms' MaskKernel), stamp tag_table[s], fire. Grid-strided over
        // chunks; one block owns each chunk (block-local __syncthreads), thread-0 fires. The
        // acquire is FUSED here, per chunk, for the reasons in the header comment: it is
        // block-local, and doing it here (rather than above the compute) hands the in-flight
        // snapshot a whole extra compute phase to complete in.
        const kvhdf5::GpuDatasetHandle h = groups[s % ngroups];
        const unsigned chunk_cells = cells / h.Count();
        const uint32_t* vsrc = reinterpret_cast<const uint32_t*>(vc);
        for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
            // ACQUIRE this chunk's buffer back from snapshot s-G. Thread-0 only, so the
            // __syncthreads() below is what actually protects the fill.
            if (s >= ngroups) h.WriteWait(c);
            __syncthreads();   // <-- LOAD-BEARING: no thread may fill until the drain is done
            uint32_t* dst = reinterpret_cast<uint32_t*>(h.Data(c));
            for (unsigned i = threadIdx.x; i < chunk_cells; i += blockDim.x) {
                unsigned gid = c * chunk_cells + i;
                uint32_t val = vsrc[gid];
                if (mask) val ^= mask[gid];
                dst[i] = val;
            }
            __threadfence_system();
            __syncthreads();
            h.SetTag(c, tag_table[s]);
            if (Async) h.WriteAsync</*Probing=*/false>(c);   // fire, drain later (thread-0)
            else       h.Write</*Probing=*/false>(c);        // fire-AND-WAIT (thread-0)
            __syncthreads();
        }
        grid.sync();   // every block's fire done before the next snapshot's compute reuses vc
    }
    // TAIL: drain every group still in flight (already complete under Async==false). Covers at
    // most min(G, num_snaps) groups; WriteWait on a never-fired chunk would spin forever, so the
    // loop is bounded by num_snaps as well as by G.
    const unsigned tail_groups = (num_snaps < ngroups) ? num_snaps : ngroups;
    for (unsigned gi = 0; gi < tail_groups; ++gi) {
        const kvhdf5::GpuDatasetHandle h = groups[gi];
        for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) h.WriteWait(c);
    }
}
template __global__ void GsPersistentKernel<true>(
    float*, float*, float*, float*, const kvhdf5::GpuDatasetHandle*, unsigned,
    const clio::cte::core::TagId*, const uint32_t*, GsParams, unsigned, unsigned, unsigned);
template __global__ void GsPersistentKernel<false>(
    float*, float*, float*, float*, const kvhdf5::GpuDatasetHandle*, unsigned,
    const clio::cte::core::TagId*, const uint32_t*, GsParams, unsigned, unsigned, unsigned);

// ---- COMPUTE-ONLY variant of the persistent kernel (register-decomposition probe) ------
//
// Splits GsPersistentKernel<true>'s +14 registers over GsStepKernel (40 vs 26) into two
// deltas for the paper: "cost of going resident/cooperative" vs "cost of the I/O path".
// This kernel is GsPersistentKernel with EVERY CLIO call stripped — no CLIO_GPU_INIT, no
// per-chunk submit loop (WriteWait/fill/SetTag/Write/WriteAsync), no tail drain — keeping
// ONLY the resident cooperative compute: the snapshot/step loop, grid.sync() between
// steps, the identical Gray-Scott stencil math, and the ping-pong pointer swap. It keeps
// GsPersistentKernel's full parameter list (the I/O-only params go unused) so the two
// kernels are argument-for-argument comparable. It is never launched; it exists solely to
// be measured by cudaFuncGetAttributes / cudaOccupancyMaxActiveBlocksPerMultiprocessor in
// the register-report TEST_CASE below.
__global__ void GsPersistentComputeOnlyKernel(
    float* u0, float* v0, float* u1, float* v1,
    const kvhdf5::GpuDatasetHandle* groups, unsigned ngroups,
    const clio::cte::core::TagId* tag_table, const uint32_t* mask,
    GsParams p, unsigned N, unsigned num_snaps, unsigned steps_per) {
    (void)groups; (void)ngroups; (void)tag_table; (void)mask;
    cg::grid_group grid = cg::this_grid();
    const unsigned cells = N * N;
    const unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gstride = gridDim.x * blockDim.x;

    float* uc = u0; float* vc = v0; float* un = u1; float* vn = v1;
    for (unsigned s = 0; s < num_snaps; ++s) {
        // COMPUTE steps_per Gray-Scott steps in-kernel; identical to GsPersistentKernel's
        // compute phase. No SUBMIT section here — that is the whole point of this probe.
        for (unsigned t = 0; t < steps_per; ++t) {
            for (unsigned gid = gtid; gid < cells; gid += gstride) {
                unsigned x = gid % N, y = gid / N;
                unsigned xm = (x == 0) ? (N - 1) : (x - 1);
                unsigned xp = (x == N - 1) ? 0u : (x + 1);
                unsigned ym = (y == 0) ? (N - 1) : (y - 1);
                unsigned yp = (y == N - 1) ? 0u : (y + 1);
                float ucv = uc[gid], vcv = vc[gid];
                float lap_u = uc[y*N+xm] + uc[y*N+xp] + uc[ym*N+x] + uc[yp*N+x] - 4.f*ucv;
                float lap_v = vc[y*N+xm] + vc[y*N+xp] + vc[ym*N+x] + vc[yp*N+x] - 4.f*vcv;
                float uvv = ucv * vcv * vcv;
                un[gid] = ucv + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - ucv));
                vn[gid] = vcv + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vcv);
            }
            grid.sync();
            float* tu = uc; uc = un; un = tu;
            float* tv = vc; vc = vn; vn = tv;
        }
    }
}

// ---- POOLED arm: fill + fire fused into ONE kernel --------------------------
//
// The sync/async arms are a THREE-kernel shape: TwCopyKernel stages every chunk's bytes
// into its own device buffer, then TwSnapFire/SyncKernel submits, then (async) TwDrainKernel
// waits. That shape is only correct at M == N — with a bounded pool (M < N) the copy kernel
// would refill chunk c+M's shared buffer before chunk c's PutBlob has drained. So the pooled
// arm replaces all three with GpuDatasetHandle::WritePipelined: per block, wait(c-M) -> fill
// -> fence -> fireAsync(c), with a tail that drains the <= D still in flight.
//
// The fill now runs on G blocks (the dataset's producer grid), not on the copy_bpc * chunks
// blocks TwCopyKernel used, so G must be large enough to saturate HBM or the arm is
// memory-parallelism-bound — which has nothing to do with pooling.
//
// TAIL DRAIN IS DEFERRED (TailDrain=false). The in-loop WriteWait(c-M) is what enforces the
// buffer-reuse invariant; the tail wait never did. Keeping the tail IN-kernel only forces the
// kernel to stay resident until this snapshot's last <= D Puts land, which holds the NEXT
// snapshot's compute behind this snapshot's I/O — at M == N it degenerates to "fire all, drain
// all, in-kernel", i.e. exactly the synchronous arm. (Measured: pooled ~2.1 s == sync, vs async
// ~1.7 s.) So the pooled arm defers the drain to finalize() and drains with TwDrainKernel per
// snapshot dataset, precisely as the async arm does. Safe because each launch owns its own
// snapshot dataset — nothing refills those buffers before the drain.
struct TwPooledFill {
    const byte_t* src;   // masked snapshot grid; chunk c is src[c*n .. c*n+n)
    // BLOCK-WIDE (every thread), word-wise, block-strided — the same copy TwCopyKernel does
    // for one chunk, minus the extra gridDim.x blocks. Uniform across the block.
    __device__ void operator()(uint32_t c, byte_t* dst, uint64_t n) const {
        const uint32_t* s = reinterpret_cast<const uint32_t*>(src + uint64_t(c) * n);
        uint32_t* d = reinterpret_cast<uint32_t*>(dst);
        const uint64_t words = n >> 2;
        for (uint64_t i = threadIdx.x; i < words; i += blockDim.x) d[i] = s[i];
    }
};

// NOTE (2026-07-20, measured via cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor,
// RTX 4090 / sm_89): the persistent snapshot kernel (GsPersistentKernel; loops over snapshots
// in-kernel, double-buffers across them, WriteWait to reclaim a group) measures 40 registers (Async)
// / 42 (fire-and-wait) with a 136 B/thread local-memory spill, yet reaches 6 blocks/SM = 100%
// occupancy at blockDim=256 -- the SAME tier as the stock stencil (26 regs), because the thread-count
// ceiling (1536/256=6) binds before the 64K register file. The register pressure is the Submit/
// SubmitWait machinery (Future<TaskT>, IpcManager pointer-chasing) + loop-carried state held resident,
// not the stencil. __launch_bounds__(256,6) recovers nothing (occupancy is already 100%) and only
// grows the spill, so it is NOT used. (An earlier "~55 regs -> 66.7%" estimate here was static-ptxas
// on a prototype and is superseded by the numbers above.)
template <bool kProbing>
__global__ __launch_bounds__(256) void TwSnapPooledKernel(kvhdf5::GpuDatasetHandle h,
                                                          const byte_t* src) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    h.WritePipelined<kProbing, /*TailDrain=*/false>(TwPooledFill{src});
}
template __global__ void TwSnapPooledKernel<false>(kvhdf5::GpuDatasetHandle, const byte_t*);
template __global__ void TwSnapPooledKernel<true>(kvhdf5::GpuDatasetHandle, const byte_t*);

// XOR the snapshot with a fixed random mask (word-wise) into a scratch buffer, so the
// PERSISTED bytes are high-entropy / incompressible — otherwise the Gray-Scott field is
// mostly zeros and a compressing filesystem (e.g. btrfs zstd) makes the "disk I/O" nearly
// free, voiding any disk comparison. Deterministic (fixed mask) => byte-identical across
// arms => checksums still match.
__global__ void MaskKernel(uint32_t* dst, const uint32_t* src, const uint32_t* mask,
                           unsigned words) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) dst[i] = src[i] ^ mask[i];
}

// ---- Submit-path hop breakdown (GSBENCH_SUBMIT_PROBE=1) ---------------------
//
// Splits one GPU-initiated PutBlob into its hops (see clio_runtime/gpu/submit_probe.h).
// The device half stamps %globaltimer into DEVICE global memory; the host half stamps
// CLOCK_MONOTONIC. The two halves are joined offline on the task POD's address, which
// requires putting them on a common clock — hence ProbeClockOffset below.
//
// Both dumps are raw per-submit CSVs. Nothing is averaged here on purpose: a busy-spin
// poll latency reported without its distribution is not a measurement.
//
// NOTE: these two ping kernels must live at file (not anonymous-namespace) scope, like
// every other __global__ above — nvcc's device-link step (-rdc=true, needed for this
// target) cannot resolve a __global__ function's host-side launch stub when the kernel
// is defined inside an unnamed namespace, and fails at the final host link instead.

// Device sees the host's flag write, stamps immediately. Bounds the offset from below.
__global__ void ProbePingA(volatile unsigned* flag, unsigned long long* out) {
    while (*flag == 0u) {}
    *out = clio::run::gpu::ProbeNowNs();
}

// Device stamps, then publishes to pinned memory the host is spinning on. Bounds the
// offset from above. The stamp is taken BEFORE the store so the store's own PCIe latency
// falls inside the bound rather than corrupting it.
__global__ void ProbePingB(volatile unsigned long long* out) {
    unsigned long long t = clio::run::gpu::ProbeNowNs();
    *out = t;
}

// Characterizes the two device clocks against each other.
//   out[0] = smallest nonzero step %globaltimer takes — its true resolution. On Ada this
//            comes back ~1024 ns, which is why no fine hop may be timed with it.
//   out[1] / out[2] = wall ns and SM cycles over the same spin -> the SM clock.
// The cycles->ns ratio used in the analysis is derived PER RECORD from the long
// enter->wait span instead (it sees the real boost state under load); this kernel is the
// independent cross-check on that, and the fallback where no long span exists.
__global__ void ProbeClockCalKernel(unsigned long long* out) {
    const unsigned long long t0 = clio::run::gpu::ProbeNowNs();
    const unsigned long long c0 = clio::run::gpu::ProbeCycles();
    unsigned long long prev = t0, mn = ~0ull, t = t0;
    while (t - t0 < 20000000ull) {          // 20 ms
        t = clio::run::gpu::ProbeNowNs();
        const unsigned long long d = t - prev;
        if (d > 0 && d < mn) mn = d;
        prev = t;
    }
    const unsigned long long c1 = clio::run::gpu::ProbeCycles();
    out[0] = mn;
    out[1] = t - t0;
    out[2] = c1 - c0;
}

#if !CTP_IS_DEVICE_PASS

namespace {

constexpr GsParams kGs{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

// ---- env-var config --------------------------------------------------------

unsigned EnvU(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x > 0 ? static_cast<unsigned>(x) : dflt;
}
// Like EnvU but 0 is a MEANINGFUL value (EnvU coerces <=0 to the default). The HDF5
// knobs need this: "chunk cache = 0 bytes" is a real, and as it turns out the fastest,
// setting — see Hdf5Sink.
unsigned EnvU0(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x >= 0 ? static_cast<unsigned>(x) : dflt;
}
std::string EnvS(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(dflt);
}

struct Cfg {
    unsigned N       = EnvU("GSBENCH_N", 512);
    unsigned chunks  = EnvU("GSBENCH_CHUNKS", 4);
    unsigned snaps   = EnvU("GSBENCH_SNAPS", 4);
    unsigned steps_per = EnvU("GSBENCH_STEPS_PER", 8);
    std::string bdev = EnvS("GSBENCH_BDEV", "ram");
    unsigned cap_mb  = EnvU("GSBENCH_BDEV_CAP_MB", 512);
    std::string bdev_path = EnvS("GSBENCH_BDEV_PATH", "./gsbench_bdev.dat");
    std::string disk_dir  = EnvS("GSBENCH_DISK_DIR", "./gsbench_raw_out");
    // Raw arm O_DIRECT (cache-bypass) for the disk comparison; set 0 for buffered writes
    // when the raw target is a RAM tier (tmpfs, which rejects O_DIRECT) — the fair
    // software-path comparison vs CLIO's kRam bdev.
    unsigned raw_odirect  = EnvU("GSBENCH_RAW_ODIRECT", 1);
    // Raw arm fsync per snapshot (also gates the hdf5 arms' per-snapshot H5Fflush+fdatasync):
    // DURABILITY PARITY with CLIO (whose bdev waits for each write to commit). Without it, a
    // buffered raw/hdf5 write just lands in RAM/page-cache and isn't really persisted — doing
    // less work than CLIO. Default on. EnvU0 (not EnvU) so GSBENCH_RAW_FSYNC=0 actually turns
    // it OFF: EnvU coerces 0 to the default, which would silently keep raw/hdf5 durable while a
    // matched-non-durable comparison drops CLIO's flush — flattering CLIO. 0 must mean 0.
    unsigned raw_fsync    = EnvU0("GSBENCH_RAW_FSYNC", 1);
    // DURABILITY PARITY FOR THE CLIO ARMS. CLIO's kFile bdev opens its backing file with a
    // plain O_RDWR|O_CREAT (fs_bdev_transport.cc) — no O_DIRECT, no O_SYNC — and NOTHING in
    // the bdev or AsyncIO stack ever calls fsync/fdatasync. So a PutBlob that "completed"
    // has only reached the OS page cache, while the raw and hdf5 arms fdatasync every
    // snapshot and are therefore timed against the actual device.
    //
    // That is not a small effect, it is a measurement-invalidating one: on this box the
    // durable write ceiling is ~1145 MB/s (dd conv=fdatasync, incompressible) and the
    // buffered ceiling ~1810 MB/s, yet the async arm was reporting up to 2263 MB/s —
    // i.e. FASTER THAN THE DISK CAN PHYSICALLY ACCEPT BYTES, durable or not. Those bytes
    // were still in RAM when the clock stopped. Any "N GB/s to disk" claim built on that
    // is fiction, and the arm that skips the flush looks fastest precisely because it did
    // less work.
    //
    // On: fdatasync the bdev's backing file inside the timed region, so every arm is held
    // to one standard. Off (0): the historical, non-durable behaviour — kept only so the
    // artifact can be reproduced and quantified.
    unsigned clio_fsync = EnvU0("GSBENCH_CLIO_FSYNC", 1);
    // PER-SNAPSHOT durability for the CLIO arms (checkpoint semantics): drain THIS snapshot's
    // Puts + fdatasync the bdev file INSIDE the snapshot loop, so a crash loses at most the
    // in-flight snapshot — the same guarantee raw/hdf5 give with per-snapshot fdatasync
    // (GSBENCH_RAW_FSYNC). This deliberately SERIALIZES the async/reuse arms (their whole point
    // is to fire-and-drain-in-background; forcing durability each snapshot removes that overlap),
    // so it is the honest apples-to-apples DURABLE comparison vs the HDF5 arms — and it shows the
    // real cost of checkpoint durability under the async design. Needs a file bdev (fdatasync
    // target) and clio_fsync on. 0 (default) = the end-of-run flush only. EnvU0 so 0 means 0.
    unsigned clio_persnap = EnvU0("GSBENCH_CLIO_PERSNAP", 0);
    // cudaHostRegister the hostclio arm's shm staging buffer.
    //
    // hostclio D2Hs into a CLIO shm buffer, which CUDA sees as ordinary pageable memory, so
    // the copy goes through CUDA's bounce-buffer path: 17.2 GB/s instead of the 25.0 GB/s a
    // registered buffer gets (measured; ~109 ms vs ~75 ms for the 1875 MB payload).
    //
    // The GPU-producer arms are on a different footing: the kRam bdev's pages are pageable
    // `new char[]` (mem_bdev_transport.cc), but they are now allocated AND pre-faulted in
    // Init() whenever the bdev has an explicit capacity, so its D2H at least lands on warm
    // pageable memory rather than the cold-fault floor; GSBENCH_BDEV=pinned selects the
    // kPinned bdev, whose pages ARE cudaMallocHost'd and thus on the true DMA fast path.
    // Either way that is a property of the bdev, not of the staging buffer this knob is
    // about. hostclio exists precisely to isolate
    // "what does GPU-initiation buy?" from "what does the CLIO backend buy?", so leaving its
    // one D2H on the slow path would inflate OUR OWN headline number by handicapping the
    // control. It is worth ~34 ms/run, which moved the GPU-initiation claim from 1.84x to
    // 2.00x — i.e. most of the difference between an honest result and a flattering one.
    // Default on; set 0 to reproduce the handicapped measurement.
    unsigned hostclio_pin = EnvU0("GSBENCH_HOSTCLIO_PIN", 1);
    // XOR each snapshot with a random mask so persisted bytes are incompressible (else the
    // mostly-zero Gray-Scott field compresses away on btrfs zstd, voiding disk comparisons).
    unsigned incompressible = EnvU("GSBENCH_INCOMPRESSIBLE", 1);
    // Raw writer structure: 0 = background thread (I/O overlaps compute); 1 = inline
    // synchronous (GPU idle during the write, matching host-CLIO / sync-CLIO). Use inline
    // for a storage-path comparison free of this box's GPU-concurrent-I/O throttle.
    unsigned raw_inline   = EnvU("GSBENCH_RAW_INLINE", 0);
    // Place the CLIO snapshot DATA backend in pinned host memory (kPinnedHost)
    // instead of on-GPU (kDeviceMem). The in-process bdev server's device->host
    // readback does not overlap the producer's compute (its first per-run D2H
    // stalls for the whole compute window), so with kDeviceMem the async drain
    // serializes after compute and barely beats sync. Pinned data removes the
    // server D2H, letting disk writes pipeline under compute: at steps_per=192,
    // N=6400 async goes ~836 -> ~955 MB/s. Trade-off: the producer kernel writes
    // mapped host over PCIe (slower submit) and it regresses the disk-bound /
    // low-compute regime, so it is off by default.
    unsigned data_pinned  = EnvU("GSBENCH_DATA_PINNED", 0);
    // READER benchmark (GSBENCH_READ=1, default off so the writer numbers are untouched):
    // after the write phase, read every snapshot back and compute the PDF of the v field.
    // Timed separately and reported on its own GSBENCH_READ line.
    unsigned read_pdf     = EnvU0("GSBENCH_READ", 0);
    // GSBENCH_READ_ASYNC=1: use the PIPELINED reader (2 reused read buffers, snapshot s+1's
    // Gets fired before s is drained+binned) instead of the synchronous one. Arms with only a
    // single dataset fall back to the sync reader; the emitted reader= field says which ran.
    unsigned read_async   = EnvU0("GSBENCH_READ_ASYNC", 0);
    // Number of CUDA thread blocks that submit PutBlob tasks (grid dim of the
    // fire/sync/drain kernels). Each block grid-strides over chunks, so block b's
    // thread-0 enqueues chunks b, b+gridDim, ... This is the advisor's "number of
    // GPU blocks" axis. Default 1 = the historical single-block serial-submit
    // baseline; set it to `chunks` for the framework's one-block-per-chunk pattern.
    // (EnvU coerces <=0 to the default, so the effective value is always >=1; it is
    // clamped to `chunks` below since extra blocks would just idle.)
    unsigned submit_blocks = EnvU("GSBENCH_SUBMIT_BLOCKS", 1);
    // POOLED arm only: M, the number of RESIDENT per-chunk data buffers a snapshot's
    // dataset holds. N chunks stream through M buffers (chunk c uses buffer c % M), so the
    // device-side payload footprint is M * chunk_bytes instead of N * chunk_bytes. 0 (the
    // default) means M == N: no pooling, one buffer per chunk — the CONTROL that isolates
    // the cost of FUSING fill+fire from the cost of POOLING. The pipeline depth (in-flight
    // Puts per block) is D = M / G, with G = GSBENCH_SUBMIT_BLOCKS; the dataset ctor throws
    // unless G divides M and G <= M <= N.
    unsigned pool = EnvU0("GSBENCH_POOL", 0);
    // Force the CLIO kernels' device code resident BEFORE the timed region
    // (cudaFuncGetAttributes). CUDA 12 loads a kernel's module on its FIRST launch and that
    // load is a DEVICE-WIDE sync, which lands behind the async arm's queued compute and
    // pins the whole device — killing the compute/IO overlap (162 ms -> 94 ms once warm).
    // Default 1 (the fix ON). Set 0 to measure the cold-launch penalty deliberately.
    unsigned prewarm = EnvU0("GSBENCH_PREWARM", 1);

    // ---- hdf5 arm ----------------------------------------------------------
    // Output dir for the HDF5 arm's single checkpoint file (one dataset per snapshot,
    // /v/step_NNNN, chunked to the SAME {N/chunks, N} geometry as the CLIO arms).
    std::string hdf5_dir = EnvS("GSBENCH_HDF5_DIR", "./gsbench_hdf5_out");
    // Leave EVERY tuning knob alone (stock HDF5: 1 MB chunk cache, late alloc, default
    // fill). This is the "naive baseline" we deliberately do NOT publish — it exists so
    // the tuning can be shown to be worth something rather than asserted.
    unsigned hdf5_stock = EnvU0("GSBENCH_HDF5_STOCK", 0);
    // Per-dataset chunk cache (H5Pset_chunk_cache), in MB. 0 = size the cache to ZERO,
    // which makes HDF5 take its cache-bypass path: a whole-chunk write with no filters
    // goes straight from the user buffer to the file, no staging memcpy. Our writes are
    // always whole-chunk and unfiltered, so the cache can only ever cost us a copy.
    unsigned hdf5_rdcc_mb = EnvU0("GSBENCH_HDF5_RDCC_MB", 0);
    // H5Pset_alloc_time(H5D_ALLOC_TIME_EARLY): allocate the dataset's file space up front
    // instead of chunk-by-chunk during the write.
    unsigned hdf5_early_alloc = EnvU0("GSBENCH_HDF5_EARLY_ALLOC", 1);
    // Use H5Dwrite_chunk() (direct chunk write: skips the cache AND the filter pipeline)
    // instead of H5Dwrite(). Legal here because chunk c is exactly rows
    // [c*N/chunks, (c+1)*N/chunks) x N, which is contiguous in the row-major host buffer.
    unsigned hdf5_direct_chunk = EnvU0("GSBENCH_HDF5_DIRECT_CHUNK", 0);
    // The "typical non-expert user" baseline (arm `hdf5_naive`): a default HDF5 setup with
    // NO optimizations. Overrides the tuned knobs above — high-level H5Dwrite of the whole
    // array, DEFAULT (late/incremental) alloc, default chunk cache, default fill. Structurally
    // identical to hdf5_inline (synchronous, GPU idle during the write, same per-snapshot
    // durability); only the HDF5 dataset config differs. Exists so the tuning can be shown to be
    // worth something against a representative untuned reference.
    unsigned hdf5_naive = EnvU0("GSBENCH_HDF5_NAIVE", 0);
    // Layout for the naive arm: 0 = CONTIGUOUS (no H5Pset_chunk — the most default thing a naive
    // user gets); 1 = CHUNKED at the same {N/chunks, N} geometry as the tuned/CLIO arms, but
    // still otherwise untuned (default alloc/cache/fill, high-level H5Dwrite). Lets the naive
    // baseline choose the pure-default layout or the milder "chunked-but-untuned" middle ground.
    // Only consulted when hdf5_naive is set.
    unsigned hdf5_naive_chunked = EnvU0("GSBENCH_HDF5_NAIVE_CHUNKED", 0);
    // Create all `snaps` datasets BEFORE the timed region rather than one per snapshot
    // inside it. The motivation is PARITY: the CLIO arms call MakeSnapDatasets() (tag +
    // per-chunk backend registration) before RunSim, and the raw arm ftruncate()s its file
    // in the BgWriter ctor before RunSim — so both get their setup off the clock, and
    // charging HDF5 for its setup inside the loop would be an unearned handicap, worst
    // exactly at the small-chunk end that the sweep exists to measure.
    //
    // Except it MEASURES SLOWER (~5%, at every chunk size). Holding 12 datasets open with
    // ALLOC_TIME_EARLY means the whole 1.9 GB is allocated and all 12 chunk indices are
    // resident and dirty in the metadata cache before the first write, and every subsequent
    // per-snapshot H5Fflush then has to walk all of that. So the default is OFF: it is both
    // faster for HDF5 and the simpler code. Kept as a knob because the fairness argument for
    // ON is real and the numbers are the only reason to overrule it.
    unsigned hdf5_precreate = EnvU0("GSBENCH_HDF5_PRECREATE", 0);
    // File driver: "sec2" (default; unbuffered pwrite, the same kernel path the raw arm
    // takes) or "stdio" (buffered FILE*). At the small-chunk end of the sweep sec2 emits
    // ONE write() per chunk — 6400 x 25 KiB per snapshot — and stdio's FILE buffer
    // coalesces those into far fewer, larger syscalls. Whether that actually pays is an
    // empirical question, so it is a knob and not an assumption.
    std::string hdf5_vfd = EnvS("GSBENCH_HDF5_VFD", "sec2");
    // Pinned host buffers in the threaded `hdf5` arm's writer pool (raw uses 3). Exposed so
    // the `hdf5_async` arm's much larger pinned footprint — it needs one buffer PER SNAPSHOT
    // (snaps * 156 MB = 1875 MB), because the async VOL reads the buffer after the call
    // returns — can be ruled in or out as the cause of that arm's slowdown: run `hdf5` at
    // nbuf=snaps and see whether it collapses too.
    unsigned hdf5_nbuf = EnvU("GSBENCH_HDF5_NBUF", 3);
    // Pinned (cudaMallocHost) vs pageable (malloc) staging buffers for the hdf5 arm.
    //
    // Pinned looks like the obvious choice — it is what raw uses, and it makes the D2H
    // ~2x faster. But raw ALSO splits its writes into <=1 MiB pwrites, and its comment says
    // why: "a single huge pwrite from CUDA-pinned memory hit a ~5x-slow kernel path on this
    // box". HDF5's sec2 driver issues ONE write() per chunk — 39 MiB at the default geometry
    // — straight out of whatever buffer we hand it, so with a pinned buffer the hdf5 arm
    // walks into exactly the kernel path raw was written to dodge, and no HDF5-side knob can
    // reach it. Pageable trades a slower D2H for a write that runs at full speed.
    // Which wins is an empirical question at each geometry, hence the knob.
    unsigned hdf5_pinned = EnvU0("GSBENCH_HDF5_PINNED", 1);
    // H5Pset_meta_block_size: how much file space HDF5 grabs at a time for metadata. The
    // chunk index for a 6400-chunk dataset is not small and the 2 KB default makes it
    // dribble out; 2 MB is worth ~6% at 6400 chunks and costs nothing at 4. 0 = leave
    // HDF5's default alone.
    unsigned hdf5_meta_block_kb = EnvU0("GSBENCH_HDF5_META_BLOCK_KB", 2048);
    // PAGE-FAULT PARITY between the file-backed arms (raw/hdf5) and the CLIO arms.
    //
    // raw and hdf5 write into a FRESH file. ftruncate() and H5D_ALLOC_TIME_EARLY set the
    // file SIZE but leave it SPARSE, so the first store to each page faults it in and the
    // kernel zeroes it — INSIDE the timed region. The CLIO arms do not pay this: the kRam
    // bdev's pages are pageable `new char[]`, but a bdev created with an explicit capacity
    // allocates and pre-faults them all in Init(), at server startup, i.e. off the clock.
    // (The kPinned bdev, GSBENCH_BDEV=pinned, likewise cudaMallocHost's its pages in Init.)
    //
    // Measured with dd on /dev/shm at 1875 MB: fresh pages 505 ms (~3700 MB/s) vs
    // already-allocated pages 256 ms (~7300 MB/s). ~250 ms — ~40% of those arms' runtime —
    // was pure page-allocation cost charged to raw/hdf5 and to nobody else. That is not a
    // storage-path difference, it is an artifact of who allocated first.
    //
    // On (default): write real zeros over the output file's full extent BEFORE RunSim, so
    // the timed writes land on pages that already exist — the steady state CLIO's bdev is
    // handed for free. Off (0): the historical, unfair behaviour, kept so the artifact can
    // be reproduced and quantified. EnvU0, not EnvU: 0 must mean 0.
    unsigned prefault = EnvU0("GSBENCH_PREFAULT", 1);
    // Effective submit grid: clamp the knob to chunks (more blocks than chunks idle).
    unsigned submit_grid() const {
        return submit_blocks < chunks ? submit_blocks : chunks;
    }

    unsigned steps() const { return snaps * steps_per; }
    uint64_t cells() const { return uint64_t(N) * N; }
    uint64_t grid_bytes() const { return cells() * sizeof(float); }
    // total bytes persisted by the whole run (one v-grid per snapshot).
    uint64_t total_bytes() const { return grid_bytes() * snaps; }
};

// ---- shared sim scaffolding ------------------------------------------------

struct Grids { float *u_curr, *u_next, *v_curr, *v_next; };
Grids MakeGrids(unsigned N) {
    uint64_t cells = uint64_t(N) * N, bytes = cells * sizeof(float);
    Grids g{};
    REQUIRE(cudaMalloc(&g.u_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.u_next, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_next, bytes) == cudaSuccess);
    std::vector<float> u0(cells, 1.0f), v0(cells, 0.0f);
    unsigned lo = N/2 - 3, hi = N/2 + 3;
    for (unsigned y = lo; y < hi; ++y)
        for (unsigned x = lo; x < hi; ++x) v0[uint64_t(y)*N + x] = 1.0f;
    ctp::GpuApi::Memcpy(g.u_curr, u0.data(), bytes);
    ctp::GpuApi::Memcpy(g.v_curr, v0.data(), bytes);
    return g;
}
void FreeGrids(Grids& g) {
    cudaFree(g.u_curr); cudaFree(g.u_next); cudaFree(g.v_curr); cudaFree(g.v_next);
}

// Makes each snapshot INCOMPRESSIBLE: Apply(v) XORs the grid with a fixed random mask into
// a scratch device buffer and returns it, so the persisted bytes are high-entropy (the
// mostly-zero Gray-Scott field would otherwise compress away on btrfs zstd). Mask is fixed
// (seed) => byte-identical across arms => checksums still match. If disabled, Apply is a
// pass-through. One shared scratch is safe: the MaskKernel and each arm's subsequent
// copy/D2H are serialized on the default stream, so scratch is consumed before the next
// Apply overwrites it.
struct Masker {
    bool on_;
    unsigned cells_;
    uint32_t* d_mask_ = nullptr;
    uint32_t* d_scratch_ = nullptr;
    Masker(unsigned N, bool on) : on_(on), cells_(N * N) {
        if (!on_) return;
        uint64_t bytes = uint64_t(cells_) * sizeof(uint32_t);
        REQUIRE(cudaMalloc(&d_mask_, bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_scratch_, bytes) == cudaSuccess);
        std::vector<uint32_t> mask(cells_);
        std::mt19937 rng(0xC0FFEEu);            // fixed => identical across arms
        for (auto& x : mask) x = rng();
        ctp::GpuApi::Memcpy(d_mask_, mask.data(), bytes);
    }
    ~Masker() { if (on_) { cudaFree(d_mask_); cudaFree(d_scratch_); } }
    // Returns an incompressible view of v (scratch), or v itself if disabled.
    float* Apply(float* v) {
        if (!on_) return v;
        unsigned t = 256, b = (cells_ + t - 1) / t;
        MaskKernel<<<b, t>>>(d_scratch_, reinterpret_cast<uint32_t*>(v), d_mask_, cells_);
        return reinterpret_cast<float*>(d_scratch_);
    }
    // Raw device mask pointer for arms that XOR inline (the persistent kernel), or nullptr when
    // incompressibility is off. Same fixed-seed mask Apply() uses, so persisted bytes match.
    const uint32_t* MaskPtr() const { return on_ ? d_mask_ : nullptr; }
};

// FNV-1a over a host byte buffer (cross-arm "identical computation" proof).
uint64_t Fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// GPU-timeline phase tracer (GSBENCH_TRACE=1), used by the sync & async CLIO arms to
// answer the advisor's question: is async's I/O actually OVERLAPPING compute, or is
// compute so cheap there's nothing to overlap? It splits each snapshot interval on the
// GPU's own timeline into:
//   compute_ms  — the steps_per GsStepKernels leading up to the snapshot,
//   submit_ms   — the snapshot's mask + PutBlob-task emplace kernel (async: fire only;
//                 sync: fire-AND-device-wait, so sync's submit bucket carries the wait),
//   drain_ms    — (async only) the single TAIL TwDrainKernel spin-wait at the very end.
// So the hypothesis reads directly off the buckets: async should have near-zero submit_ms
// and a drain_ms that is SMALL vs total compute if I/O kept up (good overlap), or LARGE if
// the server fell behind (backlog => async collapses toward sync). sync has no drain; its
// per-snapshot wait is inside submit_ms.
//
// Mechanics: everything runs on the default stream, so cudaEvents recorded at the phase
// boundaries measure GPU-serialized work in order. We ONLY cudaEventRecord in the hot loop
// (async, ~1us, no stall) and read every cudaEventElapsedTime AFTER the closing Synchronize
// — a mid-loop cudaEventSynchronize would serialize the stream and destroy the overlap we
// are measuring. NOTE: submit_ms includes the per-snapshot MaskKernel (identical work in
// both arms, so it cancels in the sync-vs-async comparison).
struct PhaseTrace {
    bool on_;
    unsigned snaps_;
    bool has_drain_;
    std::vector<cudaEvent_t> cs_, ce_, se_;   // compute-start, compute-end(=submit-start), submit-end
    cudaEvent_t ds_ = nullptr, de_ = nullptr; // drain-start, drain-end (async only)
    PhaseTrace(bool on, unsigned snaps, bool async)
        : on_(on), snaps_(snaps), has_drain_(async) {
        if (!on_) return;
        cs_.resize(snaps); ce_.resize(snaps); se_.resize(snaps);
        for (unsigned i = 0; i < snaps; ++i) {
            cudaEventCreate(&cs_[i]); cudaEventCreate(&ce_[i]); cudaEventCreate(&se_[i]);
        }
        if (has_drain_) { cudaEventCreate(&ds_); cudaEventCreate(&de_); }
    }
    ~PhaseTrace() {
        if (!on_) return;
        for (auto e : cs_) cudaEventDestroy(e);
        for (auto e : ce_) cudaEventDestroy(e);
        for (auto e : se_) cudaEventDestroy(e);
        if (has_drain_) { cudaEventDestroy(ds_); cudaEventDestroy(de_); }
    }
    void CompStart(unsigned si)  { if (on_) cudaEventRecord(cs_[si]); }
    void CompEnd(unsigned si)    { if (on_) cudaEventRecord(ce_[si]); }
    void SubmitEnd(unsigned si)  { if (on_) cudaEventRecord(se_[si]); }
    void DrainStart()            { if (on_ && has_drain_) cudaEventRecord(ds_); }
    void DrainEnd()              { if (on_ && has_drain_) cudaEventRecord(de_); }
    // Read the deltas + print the per-snapshot series and totals. MUST be called after the
    // caller's RunSim returns (i.e. after the closing Synchronize) so every event is done.
    void Report(const char* arm) {
        if (!on_) return;
        double tot_c = 0, tot_s = 0;
        std::fprintf(stderr, "GSBENCH_TRACE arm=%s per-snapshot (GPU-timeline ms):\n", arm);
        for (unsigned i = 0; i < snaps_; ++i) {
            float c = 0, s = 0;
            cudaEventElapsedTime(&c, cs_[i], ce_[i]);
            cudaEventElapsedTime(&s, ce_[i], se_[i]);
            tot_c += c; tot_s += s;
            std::fprintf(stderr, "GSBENCH_TRACE   snap=%2u compute_ms=%8.3f submit_ms=%8.3f\n",
                         i, c, s);
        }
        float drain = 0;
        if (has_drain_) cudaEventElapsedTime(&drain, ds_, de_);
        std::fprintf(stderr,
            "GSBENCH_TRACE arm=%s TOTALS compute_ms=%.3f submit_ms=%.3f drain_ms=%.3f "
            "(sum=%.3f)\n",
            arm, tot_c, tot_s, double(drain), tot_c + tot_s + double(drain));
    }
};

// Direct measurement of the PRODUCER-BLOCKING device-to-host copy (GSBENCH_D2H_TRACE=1),
// i.e. the stall GPUH5 eliminates: a host-mediated writer cannot start until the D2H that
// produces its buffer has landed. Used by every host-mediated arm (raw / hdf5 / hdf5_async
// / hostclio) at its single cudaMemcpy site.
//
// WHY EVENTS, AND NOT A HOST CLOCK AROUND THE cudaMemcpy. RunSim issues steps_per
// GsStepKernels with NO synchronize before calling snap(), so the host runs ahead and the
// GPU still holds most of the snapshot's compute when snap() is entered. A cudaMemcpy on
// the default stream blocks the host until that whole backlog drains AND THEN copies, so a
// host clock around it measures `compute_tail + mask + copy` — it grows with steps_per and
// is NOT a measurement of the copy. (This is exactly the bug the old hdf5_async t_d2h had;
// both numbers are reported below so the conflation is visible rather than assumed.)
//
// cudaEventRecord on the default stream is ordered IN STREAM, so the opening event fires
// only once the mask kernel has retired, and the closing one once the copy has. Their delta
// is the copy alone, on the GPU's own timeline, with zero perturbation: we only record in
// the hot loop (~1us, no stall) and read every cudaEventElapsedTime after RunSim's closing
// Synchronize. A cudaDeviceSynchronize-then-host-clock would give the same answer but would
// drain the pipeline mid-loop; events do not.
//
// Reported per copy:
//   copy_ms  — GPU-timeline duration of the D2H alone. THE NUMBER. Bandwidth is bytes/copy_ms.
//   block_ms — host wall-clock across the cudaMemcpy call: what the producer thread actually
//              loses, = compute_tail + copy. Reported only to expose the difference.
struct D2HTrace {
    bool on_;
    uint64_t bytes_;                       // bytes moved per copy (constant across snapshots)
    std::vector<cudaEvent_t> a_, b_;       // copy-start / copy-end, one pair per snapshot
    std::vector<double> block_ms_;         // host-observed block, one per snapshot
    unsigned i_ = 0;                       // next slot

    D2HTrace(bool on, unsigned snaps, uint64_t bytes) : on_(on), bytes_(bytes) {
        if (!on_) return;
        a_.resize(snaps); b_.resize(snaps); block_ms_.assign(snaps, 0.0);
        for (unsigned i = 0; i < snaps; ++i) {
            cudaEventCreate(&a_[i]); cudaEventCreate(&b_[i]);
        }
    }
    ~D2HTrace() {
        if (!on_) return;
        for (auto e : a_) cudaEventDestroy(e);
        for (auto e : b_) cudaEventDestroy(e);
    }

    // Drop-in for `cudaMemcpy(dst, src, bytes_, cudaMemcpyDeviceToHost)`. Untraced when off.
    void Copy(void* dst, const void* src) {
        if (!on_ || i_ >= a_.size()) {
            cudaMemcpy(dst, src, bytes_, cudaMemcpyDeviceToHost);
            return;
        }
        const unsigned i = i_++;
        auto h0 = std::chrono::steady_clock::now();
        cudaEventRecord(a_[i]);
        cudaMemcpy(dst, src, bytes_, cudaMemcpyDeviceToHost);
        cudaEventRecord(b_[i]);
        block_ms_[i] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - h0).count();
    }

    // MUST be called after RunSim returns (i.e. after the closing Synchronize).
    void Report(const char* arm) {
        if (!on_ || i_ == 0) return;
        double tot_copy = 0, tot_block = 0;
        std::fprintf(stderr, "GSBENCH_D2H arm=%s per-snapshot:\n", arm);
        for (unsigned i = 0; i < i_; ++i) {
            float c = 0;
            cudaEventElapsedTime(&c, a_[i], b_[i]);
            tot_copy += c; tot_block += block_ms_[i];
            std::fprintf(stderr,
                "GSBENCH_D2H   snap=%2u copy_ms=%8.3f block_ms=%8.3f GBps=%6.2f\n",
                i, c, block_ms_[i],
                double(bytes_) / (double(c) / 1e3) / 1e9);
        }
        const double mean_copy = tot_copy / double(i_);
        std::fprintf(stderr,
            "GSBENCH_D2H arm=%s copies=%u bytes_per_copy=%llu total_MB=%.1f "
            "copy_ms=%.3f block_ms=%.3f mean_copy_ms=%.3f GBps=%.2f\n",
            arm, i_, (unsigned long long)bytes_,
            double(bytes_) * double(i_) / (1024.0 * 1024.0),
            tot_copy, tot_block, mean_copy,
            double(bytes_) / (mean_copy / 1e3) / 1e9);
    }
};

// host_ns ~= device_globaltimer_ns + offset. Neither one-way latency is observable on its
// own, so we bracket: ping A gives a lower bound on the offset, ping B an upper bound, and
// the true offset lies between. The bracket's half-width IS the measurement's uncertainty
// and is reported alongside every cross-domain hop — a poll latency quoted to a tighter
// precision than this would be a lie.
// NOTE: %globaltimer's ~1024 ns quantization can make this bracket come back INVERTED
// (hi < lo, i.e. a negative half-width). That is not a bug in the bounds — it is the two
// bounds landing inside a single timer tick. The honest uncertainty on a cross-domain hop
// is therefore max(err_ns, globaltimer_tick_ns), and the analysis applies that floor.
//
// The offset also DRIFTS: %globaltimer and CLOCK_MONOTONIC are independent oscillators and
// run ~20 ppm apart on this box, which is tens of microseconds over a single run — larger
// than the poll latency being measured. A one-shot offset therefore silently corrupts both
// cross-domain hops (it showed up as a physically impossible NEGATIVE completion-visibility
// time that grew with run length). So the offset is measured TWICE, at the run's start and
// end, each stamped with the host clock, and the analysis interpolates between them.
struct ProbeClockOffset {
    long long est_ns = 0;   // midpoint of the bracket
    long long err_ns = 0;   // half-width: the honest error bar on hops 3 and 8
    long long lo_ns = 0;    // raw bounds, kept so the analysis can see the inversion
    long long hi_ns = 0;
    long long at_host_ns = 0;  // host clock when this anchor was taken (for the lerp)

    static ProbeClockOffset Measure(unsigned reps = 200) {
        ProbeClockOffset o;
        unsigned* flag = nullptr;
        unsigned long long* stamp = nullptr;
        cudaHostAlloc(&flag, sizeof(unsigned), cudaHostAllocMapped);
        cudaHostAlloc(&stamp, sizeof(unsigned long long), cudaHostAllocMapped);

        long long lo = LLONG_MIN;   // max over A: offset >= h - d
        long long hi = LLONG_MAX;   // min over B: offset <= h - d
        for (unsigned i = 0; i < reps; ++i) {
            // --- A: host -> device
            *flag = 0; *stamp = 0;
            ProbePingA<<<1, 1>>>(flag, stamp);
            // Let the kernel reach its spin before releasing it, so the kernel-launch
            // latency is not charged to the one-way trip we are bounding.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            long long h = (long long)clio::run::gpu::SubmitProbe::NowNs();
            *flag = 1;
            cudaDeviceSynchronize();
            long long d = (long long)*stamp;
            if (d > 0) lo = std::max(lo, h - d);

            // --- B: device -> host
            *stamp = 0;
            ProbePingB<<<1, 1>>>(stamp);
            unsigned long long seen = 0;
            while ((seen = *(volatile unsigned long long*)stamp) == 0ull) {}
            long long h2 = (long long)clio::run::gpu::SubmitProbe::NowNs();
            cudaDeviceSynchronize();
            hi = std::min(hi, h2 - (long long)seen);
        }
        cudaFreeHost(flag);
        cudaFreeHost(stamp);
        o.lo_ns = lo;
        o.hi_ns = hi;
        o.est_ns = (lo + hi) / 2;
        o.err_ns = (hi - lo) / 2;
        o.at_host_ns = (long long)clio::run::gpu::SubmitProbe::NowNs();
        return o;
    }
};

struct SubmitProbeHarness {
    bool on_ = false;
    unsigned cap_ = 0;
    clio::run::gpu::SubmitProbeRec* d_recs_ = nullptr;
    unsigned* d_counter_ = nullptr;
    ProbeClockOffset off_;             // anchor 1: taken at arm time
    ProbeClockOffset off_end_;         // anchor 2: taken at dump time (drift correction)
    unsigned long long tick_ns_ = 0;   // %globaltimer resolution (the fine-hop floor)
    double sm_ghz_ = 0.0;              // SM clock, for the cycles->ns cross-check

    // Arms both halves and hands the device half to the kernel via gpu_info. MUST run
    // before the datasets are built — they snapshot gpu_info by value into every handle.
    SubmitProbeHarness(bool on, unsigned cap, clio::run::IpcManagerGpuInfo* gpu_info)
        : on_(on), cap_(cap) {
        if (!on_) return;
        cudaMalloc(&d_recs_, size_t(cap_) * sizeof(clio::run::gpu::SubmitProbeRec));
        cudaMemset(d_recs_, 0, size_t(cap_) * sizeof(clio::run::gpu::SubmitProbeRec));
        cudaMalloc(&d_counter_, sizeof(unsigned));
        cudaMemset(d_counter_, 0, sizeof(unsigned));
        gpu_info->probe_.recs = d_recs_;
        gpu_info->probe_.counter = d_counter_;
        gpu_info->probe_.cap = cap_;
        clio::run::gpu::SubmitProbe::Get().Enable(cap_);
        off_ = ProbeClockOffset::Measure();

        unsigned long long* cal = nullptr;
        cudaMalloc(&cal, 3 * sizeof(unsigned long long));
        ProbeClockCalKernel<<<1, 1>>>(cal);
        cudaDeviceSynchronize();
        unsigned long long h[3] = {0, 0, 0};
        cudaMemcpy(h, cal, sizeof(h), cudaMemcpyDeviceToHost);
        cudaFree(cal);
        tick_ns_ = h[0];
        sm_ghz_ = h[1] ? double(h[2]) / double(h[1]) : 0.0;
        std::fprintf(stderr,
            "GSBENCH_PROBE clock_offset_ns=%lld err_ns=%lld (host = device + offset) "
            "globaltimer_tick_ns=%llu sm_ghz=%.3f\n",
            off_.est_ns, off_.err_ns, tick_ns_, sm_ghz_);
    }

    void Dump(const char* arm, const char* dir) {
        if (!on_) return;
        // Second clock anchor, taken as soon after the timed region as possible: the
        // device<->host offset drifts, and every cross-domain hop is interpolated between
        // this anchor and the one taken at arm time.
        off_end_ = ProbeClockOffset::Measure();
        unsigned n = 0;
        cudaMemcpy(&n, d_counter_, sizeof(unsigned), cudaMemcpyDeviceToHost);
        if (n > cap_) {
            std::fprintf(stderr,
                "GSBENCH_PROBE WARNING arm=%s: %u submits but capacity %u — "
                "%u records DROPPED; raise the cap before trusting this run\n",
                arm, n, cap_, n - cap_);
            n = cap_;
        }
        std::vector<clio::run::gpu::SubmitProbeRec> recs(n);
        cudaMemcpy(recs.data(), d_recs_,
                   size_t(n) * sizeof(clio::run::gpu::SubmitProbeRec),
                   cudaMemcpyDeviceToHost);

        char path[512];
        std::snprintf(path, sizeof(path), "%s/probe_dev_%s.csv", dir, arm);
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "task_ptr,seq,d_enter,d_pushed,d_wait_begin,d_wait_end,"
                            "c_enter,c_fields,c_prefence,c_postfence,c_pushed,"
                            "c_wait_begin,c_wait_end\n");
            for (unsigned i = 0; i < n; ++i) {
                const auto& r = recs[i];
                std::fprintf(f,
                             "%llu,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                             r.task_ptr, i, r.d_enter, r.d_pushed, r.d_wait_begin,
                             r.d_wait_end, r.c_enter, r.c_fields, r.c_prefence,
                             r.c_postfence, r.c_pushed, r.c_wait_begin, r.c_wait_end);
            }
            std::fclose(f);
        }
        std::snprintf(path, sizeof(path), "%s/probe_host_%s.csv", dir, arm);
        clio::run::gpu::SubmitProbe::Get().Dump(path);

        std::snprintf(path, sizeof(path), "%s/probe_meta_%s.csv", dir, arm);
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "arm,dev_records,host_records,clock_offset_ns,clock_err_ns,"
                            "clock_at_ns,clock_offset_end_ns,clock_err_end_ns,"
                            "clock_at_end_ns,globaltimer_tick_ns,sm_ghz\n");
            std::fprintf(f, "%s,%u,%u,%lld,%lld,%lld,%lld,%lld,%lld,%llu,%.4f\n", arm, n,
                         clio::run::gpu::SubmitProbe::Get().Count(), off_.est_ns,
                         off_.err_ns, off_.at_host_ns, off_end_.est_ns, off_end_.err_ns,
                         off_end_.at_host_ns, tick_ns_, sm_ghz_);
            std::fclose(f);
        }
        std::fprintf(stderr,
            "GSBENCH_PROBE arm=%s dev_records=%u host_records=%u -> %s/probe_*_%s.csv\n",
            arm, n, clio::run::gpu::SubmitProbe::Get().Count(), dir, arm);
    }

    ~SubmitProbeHarness() {
        if (!on_) return;
        cudaFree(d_recs_);
        cudaFree(d_counter_);
    }
};

// The ONE timed sim loop shared by every arm. `snap(si, v_curr)` persists snapshot si;
// `finalize()` runs inside the timed region (async drain). No per-step Synchronize so
// the stream can overlap; one Synchronize closes the region. Returns wall-clock ms and
// accumulates a checksum of every snapshot's v-grid (host-read) for cross-arm equality.
// `trace` (optional) records GPU-timeline phase events; nullptr = off.
// ---- Peak device-memory measurement (GSBENCH_MEM=1, default off) -----------
// A background thread polls cudaMemGetInfo during the timed loop and records the minimum
// free bytes seen. The reported peak is (baseline_free - min_free): device memory the arm
// brought resident ON TOP OF whatever was already allocated when the server came up. The
// baseline is captured at the END of BenchEnv's ctor (server up, bdev allocated) but BEFORE
// the arm allocates its compute grids and client buffer groups, so:
//   * other GPU apps' steady usage and the fixed bdev cancel out of the delta;
//   * what remains is THIS arm's device working set = compute grids (identical across arms)
//     + client buffer groups + any in-flight device staging.
// kRam/kFile bdevs and pinned-host data live in HOST memory, so pinned/persistent arms show
// a LOW device peak by design (their footprint is host-pinned, not device) -- that asymmetry
// is exactly the point of the measurement, not a bug.
std::atomic<bool> g_mem_run{false};
size_t g_mem_baseline_free = 0;   // set in BenchEnv ctor, before arm allocations
size_t g_mem_min_free = SIZE_MAX; // min free seen during the timed loop
double g_last_peak_device_mb = -1.0;
double g_last_peak_host_mb = -1.0;

// Deterministic I/O staging-buffer footprint: the bytes each arm holds resident for moving
// snapshot data between compute and storage -- NOT the compute grids, server, or page cache.
// Computed from the known allocation (Sum of GpuCteDataset payloads for the CLIO/GPUH5 arms;
// nbuf * snapshot_bytes for the writer arms), so it is exact and adds ZERO timing overhead --
// it goes on the perf line. This is the design's real buffering cost / memory-vs-overlap knob.
// (One exception: hdf5_async's VOL makes hidden internal copies we cannot count here; its true
// buffer cost comes from the sampled host RSS, flagged where reported.)
uint64_t g_io_buf_bytes = 0;

// The ACTUAL data backend the arm ran on. cfg.data_pinned is only the env KNOB, and the
// persistent arms force kPinnedHost regardless of it -- reporting the knob mislabelled them
// as pinned=0 and made their read numbers look inexplicable (they sit in the pinned
// performance band). -1 => no override, report the knob.
int g_actual_pinned = -1;
unsigned ReportedPinned(const Cfg& cfg) {
    return (g_actual_pinned >= 0) ? unsigned(g_actual_pinned) : cfg.data_pinned;
}

// Read one "Key: <n> kB" field from /proc/self/status (host RSS accounting). Returns kB, or 0.
long ReadProcStatusKB(const char* key) {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long kb = 0;
    const size_t klen = std::strlen(key);
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, klen) == 0) { kb = std::strtol(line + klen, nullptr, 10); break; }
    }
    std::fclose(f);
    return kb;
}
// Captured at program load (static init), i.e. before ANY arm allocation -- the host-RSS
// baseline. cudaMalloc'd device memory does NOT count toward RSS, so the peak-host delta below
// isolates HOST buffers: HDF5 write buffers + chunk cache, the async VOL's per-write heap copies,
// and cudaMallocHost'd pinned data (persistent / *_pinned arms). This is the cross-arm-comparable
// "memory utilization" metric for the arms whose footprint is host-side (all the HDF5 arms).
long g_host_baseline_kb = ReadProcStatusKB("VmRSS:");

struct MemSampler {
    std::thread th;
    void Start() {
        if (!EnvU0("GSBENCH_MEM", 0)) return;
        g_mem_min_free = SIZE_MAX;
        g_mem_run.store(true, std::memory_order_relaxed);
        th = std::thread([] {
            while (g_mem_run.load(std::memory_order_relaxed)) {
                size_t f = 0, t = 0;
                if (cudaMemGetInfo(&f, &t) == cudaSuccess && f < g_mem_min_free)
                    g_mem_min_free = f;
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        });
    }
    void Stop() {
        if (!th.joinable()) return;
        g_mem_run.store(false, std::memory_order_relaxed);
        th.join();
        // baseline_free >= min_free normally; clamp so noise/other-app frees can't go negative.
        const double peak_bytes = (g_mem_baseline_free > g_mem_min_free)
                                      ? double(g_mem_baseline_free - g_mem_min_free) : 0.0;
        g_last_peak_device_mb = peak_bytes / (1024.0 * 1024.0);
        // Peak HOST RSS above the program-load baseline (VmHWM = kernel high-water-mark, so no
        // sampling needed). Valid for EVERY arm -- gives the HDF5 arms a real number.
        const long hwm_kb = ReadProcStatusKB("VmHWM:");
        g_last_peak_host_mb = (hwm_kb > g_host_baseline_kb)
                                  ? double(hwm_kb - g_host_baseline_kb) / 1024.0 : 0.0;
    }
};

template <class SnapFn, class FinalizeFn>
double RunSim(const Cfg& cfg, Grids& g, SnapFn snap, FinalizeFn finalize,
              uint64_t* checksum_out, PhaseTrace* trace = nullptr) {
    unsigned N = cfg.N;
    uint64_t cells = cfg.cells();
    unsigned threads = 256, blocks = unsigned((cells + threads - 1) / threads);
    ctp::GpuApi::Synchronize();  // settle the seed before timing
    MemSampler mem;
    mem.Start();                 // no-op unless GSBENCH_MEM=1
    auto t0 = std::chrono::steady_clock::now();
    unsigned si = 0;
    for (unsigned step = 1; step <= cfg.steps(); ++step) {
        // First step of a snapshot interval => mark where this snapshot's compute begins.
        if (trace && (step - 1) % cfg.steps_per == 0) trace->CompStart(si);
        GsStepKernel<<<blocks, threads>>>(g.u_curr, g.v_curr, g.u_next, g.v_next,
                                          kGs, N);
        std::swap(g.u_curr, g.u_next);
        std::swap(g.v_curr, g.v_next);
        if (step % cfg.steps_per != 0) continue;
        if (trace) trace->CompEnd(si);   // compute done; snap() is the submit phase
        snap(si, g.v_curr);
        if (trace) trace->SubmitEnd(si);
        ++si;
    }
    if (trace) trace->DrainStart();
    finalize();                          // async: tail-drain all outstanding puts
    if (trace) trace->DrainEnd();
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    mem.Stop();          // records g_last_peak_device_mb (no-op unless GSBENCH_MEM=1)
    (void)checksum_out;  // checksum computed by arms post-run from persisted bytes
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintResult(const char* arm, const Cfg& cfg, double ms, uint64_t checksum) {
    double mb = double(cfg.total_bytes()) / (1024.0 * 1024.0);
    // durable= is the flush state THIS arm ran under, so a result line can never be
    // mistaken for a durable one when it isn't: raw/hdf5 fdatasync per snapshot
    // (GSBENCH_RAW_FSYNC), the CLIO arms fdatasync the bdev at the end of the timed region
    // (GSBENCH_CLIO_FSYNC). A RAM bdev / tmpfs target is never durable by construction.
    // Prefix-matched, NOT exact: the arms are named raw_inline/raw_threaded/async_VOL, so an
    // exact "raw"/"hdf5" test would misclassify them as CLIO arms and report the wrong
    // durability source (CLIO end-flush instead of their per-snapshot fsync).
    const bool clio_arm = (std::strncmp(arm, "raw", 3) != 0 &&
                           std::strncmp(arm, "hdf5", 4) != 0 &&
                           std::strcmp(arm, "async_VOL") != 0);
    const unsigned durable = clio_arm ? (cfg.clio_fsync && cfg.bdev == "file")
                                      : cfg.raw_fsync;
    std::fprintf(stderr,
        "GSBENCH_RESULT arm=%s N=%u chunks=%u blocks=%u snaps=%u steps=%u bdev=%s "
        "pinned=%u durable=%u MB=%.1f ms=%.2f MBps=%.1f io_buf_mb=%.1f checksum=%llu\n",
        arm, cfg.N, cfg.chunks, cfg.submit_grid(), cfg.snaps, cfg.steps(),
        cfg.bdev.c_str(), ReportedPinned(cfg), durable,
        mb, ms, mb / (ms / 1000.0),
        double(g_io_buf_bytes) / (1024.0 * 1024.0), (unsigned long long)checksum);
    // Peak device-memory line (only when GSBENCH_MEM=1 populated it). Separate line so the
    // GSBENCH_RESULT format is unchanged for existing parsers; associate by the arm= field.
    if (g_last_peak_device_mb >= 0.0 || g_last_peak_host_mb >= 0.0) {
        std::fprintf(stderr,
            "GSBENCH_MEM arm=%s N=%u chunks=%u snaps=%u bdev=%s pinned=%u "
            "peak_device_mb=%.1f peak_host_mb=%.1f\n",
            arm, cfg.N, cfg.chunks, cfg.snaps, cfg.bdev.c_str(), ReportedPinned(cfg),
            g_last_peak_device_mb, g_last_peak_host_mb);
    }
}

// ---- CLIO env bring-up (configurable bdev) ---------------------------------

struct BenchEnv {
    clio::cte::core::TagId probe_tag;
    BenchEnv(const Cfg& cfg) {
        using namespace std::chrono_literals;
        namespace bdev = clio::run::bdev;
        std::fprintf(stderr, "[bench] bringing up server (bdev=%s cap=%uMB)\n",
                     cfg.bdev.c_str(), cfg.cap_mb);
        if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer))
            throw std::runtime_error("CLIO_INIT(kServer) failed");
        if (!clio::cte::core::CLIO_CTE_CLIENT_INIT())
            throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");
        auto* cte = CLIO_CTE_CLIENT;
        cte->Init(clio::cte::core::kCtePoolId);
        clio::cte::core::CreateParams params;
        auto ct = cte->AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                   clio::cte::core::kCtePoolName,
                                   clio::cte::core::kCtePoolId, params);
        ct.Wait();
        if (ct->GetReturnCode() != 0) throw std::runtime_error("CTE create failed");
        std::this_thread::sleep_for(50ms);

        const clio::run::u64 cap = clio::run::u64(cfg.cap_mb) << 20;
        const bool is_file = (cfg.bdev == "file");
        // "pinned" is kRam's page-locked sibling: same in-memory bdev, but its pages are
        // cudaMallocHost'd, so the server's device->host copy is a true DMA instead of a
        // driver-staged bounce through pageable memory.
        const bdev::BdevType type = is_file     ? bdev::BdevType::kFile
                                    : (cfg.bdev == "pinned") ? bdev::BdevType::kPinned
                                                             : bdev::BdevType::kRam;
        // For kFile the bdev name IS the on-disk file path (O_DIRECT). For kRam/kPinned it
        // is just an identifier.
        const std::string name = is_file ? cfg.bdev_path : std::string("gsbench_ram");
        clio::run::PoolId bdev_pool_id(960, 0);
        bdev::Client bclient(bdev_pool_id);
        auto bc = bclient.AsyncCreate(clio::run::PoolQuery::Dynamic(), name, bdev_pool_id,
                                      type, cap);
        bc.Wait();
        if (bc->GetReturnCode() != 0) throw std::runtime_error("bdev create failed");
        std::this_thread::sleep_for(50ms);
        auto rt = cte->AsyncRegisterTarget(name, type, cap, clio::run::PoolQuery::Local(),
                                           bdev_pool_id);
        rt.Wait();
        if (rt->GetReturnCode() != 0) throw std::runtime_error("RegisterTarget failed");
        std::this_thread::sleep_for(50ms);
        std::fprintf(stderr, "[bench] server ready\n");
        // Peak-memory baseline (GSBENCH_MEM=1): captured now -- server + bdev are up, but the
        // arm has NOT yet allocated its compute grids or client buffer groups. The sampler's
        // reported peak is the free-memory drop below this point, i.e. the arm's own device
        // working set. See MemSampler.
        if (EnvU0("GSBENCH_MEM", 0)) {
            size_t f = 0, t = 0;
            if (cudaMemGetInfo(&f, &t) == cudaSuccess) g_mem_baseline_free = f;
        }
    }
};

// Flush CLIO's kFile bdev to the device (see Cfg::clio_fsync for why this must exist).
// fdatasync flushes the INODE's dirty pages, so any fd on the file works — we do not need
// the bdev's own descriptor. No-op for a kRam bdev (nothing to flush) or when disabled.
// MUST be called inside the timed region, after the GPU work has landed.
void ClioBdevSync(const Cfg& cfg) {
    if (!cfg.clio_fsync || cfg.bdev != "file") return;
    int fd = open(cfg.bdev_path.c_str(), O_RDONLY);
    if (fd < 0) return;                      // bdev file gone => nothing to flush
    fdatasync(fd);                           // Linux permits fsync on a read-only fd
    close(fd);
}

clio::cte::core::TagId MakeTag(const char* name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

std::vector<byte_t> HostReadBlob(clio::cte::core::TagId tag, const std::string& name,
                                 uint64_t size) {
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(size);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0, size);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, clio::run::u64(0), size,
                                           clio::run::u32(0), shm);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

// Pre-create `snaps` snapshot datasets (distinct path => distinct tag). Kept well under
// the ~16-large-backend ceiling by the caller (snaps <= ~12).
//
// pool_size (M) / grid_size (G) carry the bounded-pool contract to the dataset: M resident
// data buffers, G producer blocks, pipeline depth D = M/G. 0/0 (the sync and async arms)
// means M == N and G == M, i.e. one buffer per chunk — the historical shape those two arms
// require, since their fill (TwCopyKernel) and their fire are SEPARATE kernels and would
// otherwise clobber a shared buffer. Only the pooled arm passes nonzero values.
void MakeSnapDatasets(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                      const char* prefix, const Cfg& cfg,
                      std::vector<kvhdf5::GpuCteDataset>& out,
                      std::vector<clio::cte::core::TagId>& tags,
                      unsigned pool_size = 0, unsigned grid_size = 0) {
    kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                          /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                          /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    const auto data_kind = cfg.data_pinned
        ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
        : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;
    out.clear(); tags.clear(); out.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        out.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            pool_size, data_kind, grid_size));
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
}

// The reader's global PDF accumulator: ONE histogram over every snapshot every arm reads
// back. Its fold (below) is the cross-arm READ correctness gate -- see the HistKernel note.
struct Pdf {
    unsigned long long* d_ = nullptr;
    uint64_t bytes_ = 0;                 // total bytes binned (the reader's I/O volume)
    Pdf() {
        REQUIRE(cudaMalloc(&d_, kHistBins * sizeof(unsigned long long)) == cudaSuccess);
        REQUIRE(cudaMemset(d_, 0, kHistBins * sizeof(unsigned long long)) == cudaSuccess);
    }
    ~Pdf() { if (d_) cudaFree(d_); }
    Pdf(const Pdf&) = delete;
    Pdf& operator=(const Pdf&) = delete;

    // Bin `bytes` of float data at a DEVICE-ADDRESSABLE pointer. Device-backed datasets pass
    // their chunk buffer straight through; host arms stage into a device scratch first.
    void Add(const void* data, uint64_t bytes) {
        const uint64_t n = bytes / sizeof(float);
        unsigned blocks = unsigned((n + 255) / 256);
        if (blocks > 2048) blocks = 2048;
        if (blocks < 1) blocks = 1;
        HistKernel<<<blocks, 256>>>(static_cast<const float*>(data), n, d_);
        bytes_ += bytes;
    }

    // FNV over the bin counts: one number that MUST match across every reader arm.
    uint64_t Checksum() const {
        std::vector<unsigned long long> h(kHistBins);
        ctp::GpuApi::Synchronize();
        REQUIRE(cudaMemcpy(h.data(), d_, kHistBins * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        uint64_t x = 1469598103934665603ull;
        for (auto v : h) { x ^= v; x *= 1099511628211ull; }
        return x;
    }
    // Total binned elements — must equal snaps*cells, i.e. proof nothing was short-read.
    uint64_t Count() const {
        std::vector<unsigned long long> h(kHistBins);
        ctp::GpuApi::Synchronize();
        REQUIRE(cudaMemcpy(h.data(), d_, kHistBins * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        uint64_t t = 0;
        for (auto v : h) t += v;
        return t;
    }
};

// Time a read-back pass. `read_snap(si)` is the arm's reader: it must make snapshot si's
// bytes available and bin them into the Pdf. Mirrors RunSim's shape so the read number is
// produced the same way the write number is.
template <typename ReadSnapFn>
double RunReadPhase(const Cfg& cfg, ReadSnapFn read_snap) {
    ctp::GpuApi::Synchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned si = 0; si < cfg.snaps; ++si) read_snap(si);
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintReadResult(const char* arm, const Cfg& cfg, double ms, const Pdf& pdf,
                     const char* reader) {
    const double mb = double(cfg.total_bytes()) / (1024.0 * 1024.0);
    std::fprintf(stderr,
        "GSBENCH_READ arm=%s reader=%s N=%u chunks=%u snaps=%u bdev=%s pinned=%u MB=%.1f "
        "ms=%.2f MBps=%.1f pdf=%llu cells=%llu\n",
        arm, reader, cfg.N, cfg.chunks, cfg.snaps, cfg.bdev.c_str(), ReportedPinned(cfg),
        mb, ms, mb / (ms / 1000.0),
        (unsigned long long)pdf.Checksum(), (unsigned long long)pdf.Count());
}

// GPU-initiated reader shared by every CLIO/GPUH5 arm: ONE reused dataset, re-tagged per
// snapshot (Rearm sets the GET slot's tag too), Get lands straight in device memory, PDF
// binned in place. The reader's own footprint is a single chunk set regardless of snaps.
double GpuReadPdf(const Cfg& cfg, kvhdf5::GpuCteDataset& rd,
                  const std::vector<clio::cte::core::TagId>& tags,
                  unsigned submit_grid, Pdf& pdf) {
    const uint64_t rchunk = cfg.grid_bytes() / cfg.chunks;
    return RunReadPhase(cfg, [&](unsigned si) {
        rd.Rearm(tags[si]);
        TwReadKernel<<<submit_grid, 256>>>(rd.Handle());
        ctp::GpuApi::Synchronize();     // reads landed => safe to bin, then to re-arm
        for (unsigned c = 0; c < cfg.chunks; ++c) pdf.Add(rd.DeviceData(c), rchunk);
        ctp::GpuApi::Synchronize();
    });
}

// ASYNC (pipelined) GPU reader: 2 reused read buffers. Snapshot s+1's Gets are FIRED before
// snapshot s is drained + histogrammed, so the fetch of s+1 overlaps the reduction of s -- a
// PDF is a streaming reduction, so nothing forces the serialization the sync reader imposes.
// No host sync inside the loop: stream order alone guarantees group g's histogram kernel has
// completed before the next TwReadFireKernel refills g. Costs 2 read buffers vs the sync
// reader's 1 -- the read-side analog of the write-side double buffer.
double GpuReadPdfAsync(const Cfg& cfg, std::vector<kvhdf5::GpuCteDataset>& groups,
                       const std::vector<clio::cte::core::TagId>& tags,
                       unsigned submit_grid, Pdf& pdf) {
    REQUIRE(groups.size() >= 2);
    const uint64_t rchunk = cfg.grid_bytes() / cfg.chunks;
    // Device tag table so the fire kernel can self-stamp. Setup, outside the timed region.
    clio::cte::core::TagId* d_tags = nullptr;
    REQUIRE(cudaMalloc(&d_tags, cfg.snaps * sizeof(clio::cte::core::TagId)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tags, tags.data(), cfg.snaps * sizeof(clio::cte::core::TagId),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    ctp::GpuApi::Synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    TwReadFireKernel<<<submit_grid, 256>>>(groups[0].Handle(), d_tags, 0);   // prime the pipe
    for (unsigned si = 0; si < cfg.snaps; ++si) {
        if (si + 1 < cfg.snaps)   // fetch the NEXT snapshot while this one reduces
            TwReadFireKernel<<<submit_grid, 256>>>(groups[(si + 1) % 2].Handle(), d_tags, si + 1);
        kvhdf5::GpuCteDataset& g = groups[si % 2];
        TwReadDrainKernel<<<submit_grid, 32>>>(g.Handle());
        for (unsigned c = 0; c < cfg.chunks; ++c) pdf.Add(g.DeviceData(c), rchunk);
    }
    ctp::GpuApi::Synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    cudaFree(d_tags);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Host-side arms (raw / hdf5 / hostclio) read into a HOST buffer, then stage H2D through this
// scratch so the PDF is computed on the GPU for EVERY arm. That keeps the analysis identical
// across arms and mirrors the write side, where those same arms D2H before writing.
struct HostReadStage {
    byte_t* d_ = nullptr;
    uint64_t bytes_ = 0;
    explicit HostReadStage(uint64_t bytes) : bytes_(bytes) {
        REQUIRE(cudaMalloc(&d_, bytes) == cudaSuccess);
    }
    ~HostReadStage() { if (d_) cudaFree(d_); }
    HostReadStage(const HostReadStage&) = delete;
    HostReadStage& operator=(const HostReadStage&) = delete;
    void Bin(Pdf& pdf, const void* host_src) {
        REQUIRE(cudaMemcpy(d_, host_src, bytes_, cudaMemcpyHostToDevice) == cudaSuccess);
        pdf.Add(d_, bytes_);
    }
};

// Read every snapshot back and fold into one checksum (proves what was persisted).
uint64_t ChecksumSnapshots(const std::vector<clio::cte::core::TagId>& tags,
                           const Cfg& cfg) {
    const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    uint64_t h = 1469598103934665603ull;
    for (unsigned s = 0; s < cfg.snaps; ++s)
        for (unsigned c = 0; c < cfg.chunks; ++c) {
            auto got = HostReadBlob(tags[s], std::to_string(c), chunk_bytes);
            h = Fnv1a(got.data(), got.size(), h);
        }
    return h;
}

// The three GPU-producer submission shapes.
//   kSync   — TwCopyKernel, then Write(c) = fire-AND-wait per chunk. GPU blocks on every put.
//   kAsync  — TwCopyKernel, then WriteAsync(c) for every chunk, drained by a host-issued
//             TwDrainKernel per snapshot at the END of the run. Needs one buffer per chunk
//             (M == N), which is exactly what makes its device payload footprint N * S.
//   kPooled — ONE fused TwSnapPooledKernel per snapshot: fill+fire+drain inside
//             WritePipelined, streaming N chunks through M buffers at depth D = M/G.
enum class ClioMode { kSync, kAsync, kPooled };

const char* ClioModeName(ClioMode m) {
    switch (m) {
        case ClioMode::kSync:   return "gpuh5_sync_relaunch";
        case ClioMode::kAsync:  return "gpuh5_noreuse";
        default:                return "pooled";
    }
}

// Run a CLIO arm. Returns ms; fills checksum from persisted bytes.
double RunClioArm(const Cfg& cfg, ClioMode mode, const char* prefix, uint64_t* checksum) {
    const bool async = (mode == ClioMode::kAsync);
    const bool pooled = (mode == ClioMode::kPooled);
    const bool sync = (mode == ClioMode::kSync);
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    // Arm the submit probe BEFORE the datasets are built: MakeSnapDatasets copies
    // gpu_info by value into every GpuDatasetHandle, so a probe attached afterwards
    // would never reach the kernel. Capacity covers one submit per chunk per snapshot
    // with slack; Dump() shouts if the cap was exceeded rather than silently truncating.
    SubmitProbeHarness probe(EnvU0("GSBENCH_SUBMIT_PROBE", 0) != 0,
                             cfg.snaps * cfg.chunks + 64, &gpu_info);

    // The pooled arm is the only one that carries a (M, G) pool contract into the dataset;
    // sync/async keep the historical M == N, G == M shape (0/0). G is the SAME
    // GSBENCH_SUBMIT_BLOCKS knob the other arms use for their submit grid — for pooled it is
    // also the fill grid, since fill and fire are fused into one launch.
    const unsigned pool_m = pooled ? cfg.pool : 0u;
    const unsigned pool_g = pooled ? cfg.submit_grid() : 0u;

    std::vector<kvhdf5::GpuCteDataset> ds;
    std::vector<clio::cte::core::TagId> tags;
    if (sync || pooled) {
        // ONE reused dataset, re-tagged per snapshot (Rearm(tags[si])), bounded regardless of
        // snaps. Both arms drain + Synchronize per snapshot (sync: fused submit-and-wait;
        // pooled: fused kernel + a tail TwDrainKernel below), so only ONE snapshot is ever in
        // flight and the host re-arm is race-free -- no device tag-stamp needed (unlike the
        // overlapped reuse arm). The old one-dataset-per-snapshot shape was an allocation
        // artifact these serialized arms never needed. sync => 1 full snapshot (64 MB); pooled
        // => M pooled chunk buffers. Distinct snapshots still persist to distinct tags, so the
        // readback checksum is unchanged.
        tags.reserve(cfg.snaps);
        for (unsigned s = 0; s < cfg.snaps; ++s) {
            char p[160];
            std::snprintf(p, sizeof(p), "%s/v/step_%04u", prefix, s);
            tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(p).c_str()));
        }
        kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                              /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                              /*elem_size=*/sizeof(float)};
        const auto data_kind = cfg.data_pinned
            ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
            : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/reuse_sync", prefix);
        ds.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            pool_m, data_kind, pool_g));
    } else {
        MakeSnapDatasets(ipc, gpu_info, prefix, cfg, ds, tags, pool_m, pool_g);
    }
    {   // deterministic I/O buffer footprint: (sync/pooled) 1 reused dataset, async = snaps
        uint64_t io = 0;
        for (auto& d : ds) io += d.DeviceDataBytes();
        g_io_buf_bytes = io;
    }
    if (pooled) {
        const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
        std::fprintf(stderr,
            "GSBENCH_POOLED N=%u M=%u G=%u D=%u chunk_bytes=%llu "
            "resident_data_bytes=%llu (unpooled M=N would be %llu)\n",
            ds[0].ChunkCount(), ds[0].PoolSize(), ds[0].GridSize(), ds[0].Depth(),
            (unsigned long long)chunk_bytes,
            (unsigned long long)(uint64_t(ds[0].PoolSize()) * chunk_bytes),
            (unsigned long long)(uint64_t(cfg.chunks) * chunk_bytes));
    }

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);
    // pooled, like async, defers its outstanding Puts to a tail drain in finalize(), so it
    // gets a drain bucket too.
    PhaseTrace trace(EnvU("GSBENCH_TRACE", 0) != 0, cfg.snaps, async || pooled);
    // Grid sizing for the decoupled bulk copy: one gridDim.y per chunk, enough blocks/chunk
    // (each thread copies one 4-byte word, grid-strided) to saturate HBM.
    const uint64_t copy_chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    const unsigned copy_words = unsigned(copy_chunk_bytes / sizeof(uint32_t));
    unsigned copy_bpc = (copy_words + 255) / 256;
    if (copy_bpc < 1) copy_bpc = 1;
    if (copy_bpc > 2048) copy_bpc = 2048;
    const unsigned submit_grid = cfg.submit_grid();  // "number of GPU blocks" axis
    // CUDA 12 loads a kernel's device code lazily, ON ITS FIRST LAUNCH
    // (CUDA_MODULE_LOADING=LAZY is the default). That load is a DEVICE-WIDE
    // synchronizing operation: until everything already queued on the GPU has
    // drained, no work on ANY other stream can be dispatched. The async arm
    // queues ~86 ms of compute and then, without ever synchronizing, issues the
    // first-ever TwDrainKernel launch from finalize() — so the lazy load lands
    // behind the whole queue and pins the entire device for its duration. The
    // server's D2H copies (on their own non-blocking streams) cannot run, so
    // async degenerates to compute + io instead of max(compute, io). Measured:
    // 162 ms -> 94 ms once the kernels are resident, with drain_ms 75 -> 6.
    //
    // cudaFuncGetAttributes forces the load without launching anything, so the
    // one-time cost stays out of the timed region where it belongs. Any CLIO GPU
    // application that queues a deep kernel pipeline and expects the server's I/O
    // to overlap it needs the same warm-up (or CUDA_MODULE_LOADING=EAGER) — a
    // cold first launch anywhere in that pipeline serializes the whole device.
    //
    // GSBENCH_PREWARM=0 skips the whole thing, so the cold-launch penalty can be measured
    // rather than merely asserted. Default 1 (fix on).
    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwCopyKernel));
        // Warm whichever submit-kernel instantiation this run actually launches
        // (see the probe.on_ branch at the launch site below).
        if (probe.on_) {
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapFireKernel<true>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapSyncKernel<true>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapPooledKernel<true>));
        } else {
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapFireKernel<false>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapSyncKernel<false>));
            cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwSnapPooledKernel<false>));
        }
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwDrainKernel));
    }
    // ---- Bound host run-ahead for the POOLED arm (GSBENCH_PACE) --------------
    // The sync arm's Synchronize() below guards a deadlock the pooled arm ALSO has. The
    // taxonomy there — "sync waits per snapshot so it needs this; the async arm must NOT
    // do this ... it never wedges (its waits are deferred to the end drain)" — sorts arms
    // into waits-now vs defers-all. POOLED IS A HYBRID and fell through that gap: its
    // in-loop WriteWait(c-M) (gpu_dataset_handle.h:197, reached ONLY when M < N) leaves
    // the producer RESIDENT and spinning on the in-process server exactly like sync's
    // SubmitWait — so it CAN wedge — while its deferred tail drain made it look like
    // async, so it inherited async's exemption. It has sync's hazard and async's guard.
    // Measured: every M < N wedges once (snaps-1)*(steps_per+2) crosses ~1024 (990
    // completes, 1034 wedges), independent of M, D, chunks and gpu_queue_depth.
    //
    // It cannot use sync's per-snapshot Synchronize(): that caps in-flight work at ONE
    // snapshot and would destroy the cross-snapshot overlap this arm exists to measure.
    // So: a sliding window instead of a barrier. cudaEventSynchronize blocks the HOST
    // only — it adds no stream dependency (that would be cudaStreamWaitEvent) and never
    // stalls the GPU, which keeps consuming whatever is already queued. Waiting on the
    // event K snapshots back therefore leaves K snapshots in flight BY CONSTRUCTION, and
    // costs nothing: the GPU is the bottleneck, so run-ahead past a few snapshots buys no
    // throughput — the queue only has to stay non-empty to keep the GPU fed. (This is NOT
    // the mid-loop cudaEventSynchronize the PhaseTrace comment warns about: that one waits
    // on the JUST-recorded event, i.e. K == 0, which does drain the pipe. These are
    // separate, timing-disabled events and K >= 2.)
    //
    // K is DERIVED from one snapshot's launch cost (steps_per GsStepKernels + MaskKernel +
    // producer) against HALF the ~1024-deep queue, so raising steps_per cannot silently
    // walk it into the cliff. GSBENCH_PACE overrides it; GSBENCH_PACE=0 disables pacing and
    // restores the wedge — that switch is how the no-perf-penalty A/B is measured. EnvU0,
    // not EnvU: EnvU coerces 0 to the default, which would silently pace BOTH sides of that
    // A/B and fake the result.
    const unsigned pace_dflt = 512u / (cfg.steps_per + 2u);
    const unsigned pace_k = EnvU0("GSBENCH_PACE", pace_dflt < 2u ? 2u : pace_dflt);
    std::vector<cudaEvent_t> pace_ev;
    if (pooled && pace_k) {
        pace_ev.resize(cfg.snaps);
        for (auto& e : pace_ev) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    }
    auto pace = [&](unsigned si) {
        if (pace_ev.empty()) return;          // not pooled, or GSBENCH_PACE=0
        cudaEventRecord(pace_ev[si]);
        if (si >= pace_k) cudaEventSynchronize(pace_ev[si - pace_k]);
    };
    // PER-SNAPSHOT durability (GSBENCH_CLIO_PERSNAP): drain THIS snapshot's Puts to the bdev,
    // then fdatasync the bdev file — inside the loop. Deliberately serializes the async/pooled
    // arms (removes the fire-and-defer overlap) so the DURABLE comparison vs raw/hdf5 is
    // apples-to-apples: crash loses at most the in-flight snapshot. No-op unless armed.
    auto durable_persnap = [&](kvhdf5::GpuDatasetHandle h) {
        if (!cfg.clio_persnap) return;
        TwDrainKernel<<<submit_grid, 32>>>(h);   // land this snapshot's writes into the bdev
        ctp::GpuApi::Synchronize();
        ClioBdevSync(cfg);                        // fdatasync the bdev backing file
    };

    auto snap = [&](unsigned si, float* v_curr) {
        float* src = masker.Apply(v_curr);   // incompressible view (or v_curr if disabled)
        const byte_t* bsrc = reinterpret_cast<const byte_t*>(src);
        if (pooled) {
            // ONE fused kernel: fill + fire + bounded intra-snapshot drain over M buffers.
            // Now reuses a SINGLE dataset (M chunk buffers), re-tagged per snapshot and
            // drained+synced before the next re-arm -- so pooled is bounded in BOTH axes
            // (M chunks x 1 snapshot) instead of pre-allocating one dataset per snapshot. The
            // per-snapshot drain also removes the launch-queue wedge the old deferred-drain
            // pooled path could hit (so GSBENCH_PACE is no longer needed here).
            kvhdf5::GpuCteDataset& d = ds[0];
            d.Rearm(tags[si]);
            if (probe.on_)
                TwSnapPooledKernel<true><<<d.GridSize(), 256>>>(d.Handle(), bsrc);
            else
                TwSnapPooledKernel<false><<<d.GridSize(), 256>>>(d.Handle(), bsrc);
            TwDrainKernel<<<d.GridSize(), 32>>>(d.Handle());   // land this snapshot's tail
            ctp::GpuApi::Synchronize();                         // free the buffer before re-arm
            durable_persnap(d.Handle());   // per-snapshot durable (no-op unless armed)
            return;
        }
        // sync reuses ONE dataset, re-tagged per snapshot (race-free: it drains below before
        // the next re-arm); async keeps one dataset per snapshot (all its writes are in flight).
        kvhdf5::GpuCteDataset& d = sync ? ds[0] : ds[si];
        if (sync) d.Rearm(tags[si]);
        TwCopyKernel<<<dim3(copy_bpc, cfg.chunks), 256>>>(d.Handle(), bsrc);  // multi-block stage
        if (async) {
            // Instantiate the non-probing kernel unless the submit probe is armed
            // this run — the <false> path drops SendIn's probe registers.
            if (probe.on_) TwSnapFireKernel<true><<<submit_grid, 256>>>(d.Handle());
            else TwSnapFireKernel<false><<<submit_grid, 256>>>(d.Handle());
        } else {
            if (probe.on_) TwSnapSyncKernel<true><<<submit_grid, 256>>>(d.Handle());
            else TwSnapSyncKernel<false><<<submit_grid, 256>>>(d.Handle());
            // Bound the CUDA pending-launch queue. The sync arm's in-kernel SubmitWait
            // spins waiting for the IN-PROCESS server to flip the completion flag. If the
            // host races ahead and fills the ~1024-deep launch queue (once the run's total
            // kernels, 12*(steps_per+2), exceed it) it blocks inside cudaLaunchKernel while
            // the GPU is stalled on that spin — and the server can't make forward progress
            // from the same process → DEADLOCK (repros at steps_per>=96). A per-snapshot
            // sync caps in-flight work to one interval. Free for the sync arm (it already
            // waits on every put); the async arm must NOT do this — racing ahead IS its
            // overlap, and it never wedges (its waits are deferred to the end drain).
            ctp::GpuApi::Synchronize();
        }
        durable_persnap(d.Handle());   // per-snapshot durable (no-op unless armed)
    };
    auto finalize = [&]() {
        // Both fire-and-defer arms drain here, at the END of the timed region, so each
        // snapshot's I/O overlaps the NEXT snapshot's compute instead of blocking it.
        //
        // async  : TwCopyKernel + TwSnapFireKernel left every chunk in flight.
        // pooled : WritePipelined<TailDrain=false> left the last <= D per block in flight.
        //          TwDrainKernel is reused as-is rather than adding a kernel: it strides
        //          c = blockIdx.x; c < Count(); c += gridDim.x, so launched at grid == G it
        //          walks exactly the same chunk partition the pooled producer did, and
        //          WriteWait on an already-drained chunk just re-polls a completion flag
        //          that is already set (idempotent) — so the ones the in-loop wait already
        //          drained cost nothing.
        if (async) for (auto& d : ds) TwDrainKernel<<<submit_grid, 32>>>(d.Handle());
        if (pooled) for (auto& d : ds) TwDrainKernel<<<d.GridSize(), 32>>>(d.Handle());
        // The drain kernels must land before we can flush what they wrote: only then has
        // the server issued every PutBlob's write() into the page cache. (Harmless for the
        // sync arm — it already synchronizes per snapshot.)
        ctp::GpuApi::Synchronize();
        ClioBdevSync(cfg);   // durability parity with raw/hdf5 — see Cfg::clio_fsync
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr, &trace);
    // finalize() closed with a Synchronize, so the device is idle and these are dead.
    for (auto& e : pace_ev) cudaEventDestroy(e);

    // A PutBlob that does not fit in the bdev FAILS, and until recently did so
    // SILENTLY (the runtime set task->return_code_ but the GPU submit path threw
    // it away — see backend_ceiling_test.cu). For a benchmark that is not merely
    // data loss: we would report I/O throughput for bytes that never reached
    // storage, and the more we dropped the FASTER we would look. Those numbers
    // would be fiction. So a lost write is fatal here — abort rather than publish
    // a fraudulent measurement. Size the bdev: GSBENCH_BDEV_CAP_MB must be >=
    // snaps * snapshot_bytes (default: 12 * 156.25 MiB, so 3072 MB is ample).
    ctp::GpuApi::Synchronize();  // async arm's drain kernels must land first
    for (unsigned s = 0; s < ds.size(); ++s)
        ds[s].ThrowIfIoFailed(
            ("gsbench snapshot " + std::to_string(s)).c_str());

    FreeGrids(g);
    trace.Report(ClioModeName(mode));   // GSBENCH_TRACE=1: emit phase breakdown
    // Dumped only after ThrowIfIoFailed above: a breakdown from a run with failed puts
    // would be timing an I/O that never happened.
    probe.Dump(ClioModeName(mode), EnvS("GSBENCH_PROBE_DIR", ".").c_str());
    // READER: GPU-initiated read-back + PDF. Uses ONE reused dataset re-tagged per snapshot,
    // so the reader's own buffer footprint is a single chunk set regardless of snaps, and the
    // Get lands bytes straight in device memory where HistKernel consumes them -- no host
    // round trip anywhere in the path (which is the whole point of measuring it this way).
    if (cfg.read_pdf) {
        Pdf pdf;
        const bool amode = (cfg.read_async && ds.size() >= 2);
        const double rms = amode ? GpuReadPdfAsync(cfg, ds, tags, submit_grid, pdf)
                                 : GpuReadPdf(cfg, ds[0], tags, submit_grid, pdf);
        PrintReadResult(ClioModeName(mode), cfg, rms, pdf, amode ? "async" : "sync");
    }
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// REUSE arm (DESIGN §7 "Option B", relaunched): the async arm's well-parallelized
// separate kernels (GsStepKernel for every step, TwCopyKernel to stage, WriteAsync to
// fire), with three deltas that make snapshot payload memory CONSTANT in the number of
// snapshots and keep addressing on the device:
//   1. TWO reused buffer GROUPS (round-robin snap % 2) replace the one-dataset-per-
//      snapshot array, so resident device payload is ~2 x snapshot, flat in snaps.
//   2. DRAIN-BEFORE-REFILL: before a group is reused at snapshot s, an in-stream
//      TwDrainKernel waits for snapshot s-2's Puts out of that group to drain, so
//      refilling its buffers cannot race the server's read (the load-bearing buffer-
//      availability barrier; dropping it corrupts — proven by pooled_double_buffer_test
//      and snapshot_reuse_test, which own the negative-control teeth).
//   3. DEVICE TAG-STAMP: ReuseFireKernel sets task->tag_id_ = tag_table[s] from a host-
//      pre-built device-visible TagId table, so each snapshot still persists to its own
//      dataset (path .../step_000s, key = chunk coordinate, no `_pi`).
// NOT fused and NOT cooperative — compute stays the plain full-grid GsStepKernel (26
// regs / 100%), and the kernel-launch boundary is the free grid-wide barrier between
// timesteps. (Option A makes the whole loop resident with cooperative grid.sync(); this
// relaunched arm is the baseline it is measured against.) Its checksum MUST equal
// sync's/async's — identical computation and persisted bytes.
double RunReuseArm(const Cfg& cfg, const char* prefix, uint64_t* checksum) {
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    const kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                                /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                                /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    const auto data_kind = cfg.data_pinned
        ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
        : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;

    // One tag per snapshot (distinct dataset), resolved off the hot path, plus a
    // device-visible TagId table the fire kernel indexes by snapshot.
    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
    clio::cte::core::TagId* d_tag_table = nullptr;
    REQUIRE(cudaMalloc(&d_tag_table,
                       cfg.snaps * sizeof(clio::cte::core::TagId)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tag_table, tags.data(),
                       cfg.snaps * sizeof(clio::cte::core::TagId),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // The buffer groups (GSBENCH_GROUPS, default 2; capped at snaps), constructed ONCE and
    // reused round-robin (snap % ngroups) with drain-before-refill. Their construction tag is
    // irrelevant (the fire kernel overrides it per snapshot via SetTag), but the layout —
    // chunk coordinates + chunk bytes — is shared. EnvU floors any value <=0 to the default,
    // so ngroups is always >=1 and the modulus below is safe. This sweeps the inter-snapshot
    // depth (snapshots in flight); 1 = fully serialized (drain every snapshot), 2 = double
    // buffer. Persistent arm stays fixed at 2 (its kernel takes exactly two handles).
    const unsigned ngroups = std::min(EnvU("GSBENCH_GROUPS", 2), cfg.snaps);
    std::vector<kvhdf5::GpuCteDataset> groups;
    groups.reserve(ngroups);
    for (unsigned gi = 0; gi < ngroups; ++gi) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/group_%u", prefix, gi);
        groups.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            /*pool_size=*/0, data_kind, /*grid_size=*/0));
    }
    {
        const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
        uint64_t resident = 0;
        for (auto& d : groups) resident += d.DeviceDataBytes();
        g_io_buf_bytes = resident;   // deterministic I/O buffer footprint (2 reused groups)
        std::fprintf(stderr,
            "GSBENCH_REUSE groups=%u resident_data_bytes=%llu (async M=N x snaps "
            "would be %llu)\n",
            ngroups, (unsigned long long)resident,
            (unsigned long long)(uint64_t(cfg.chunks) * chunk_bytes * cfg.snaps));
    }

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);
    PhaseTrace trace(EnvU("GSBENCH_TRACE", 0) != 0, cfg.snaps, /*async=*/true);

    const uint64_t copy_chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    const unsigned copy_words = unsigned(copy_chunk_bytes / sizeof(uint32_t));
    unsigned copy_bpc = (copy_words + 255) / 256;
    if (copy_bpc < 1) copy_bpc = 1;
    if (copy_bpc > 2048) copy_bpc = 2048;
    const unsigned submit_grid = cfg.submit_grid();

    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwCopyKernel));
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(ReuseFireKernel));
        cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(TwDrainKernel));
    }

    // Bound host run-ahead (GSBENCH_PACE). Like the pooled arm, this arm has a RESIDENT
    // in-loop wait — the drain-before-refill TwDrainKernel spins on the in-process server
    // — so if the host floods the ~1024-deep CUDA launch queue while that drain is parked,
    // the server can't make progress from the same process and it DEADLOCKS. A sliding
    // cudaEventSynchronize window K snapshots back leaves K snapshots in flight and never
    // stalls the GPU (host-only wait). Free: the GPU is the bottleneck, so run-ahead past a
    // few snapshots buys no throughput. GSBENCH_PACE=0 disables it (to measure the wedge).
    const unsigned pace_dflt = 512u / (cfg.steps_per + 2u);
    const unsigned pace_k = EnvU0("GSBENCH_PACE", pace_dflt < 2u ? 2u : pace_dflt);
    std::vector<cudaEvent_t> pace_ev;
    if (pace_k) {
        pace_ev.resize(cfg.snaps);
        for (auto& e : pace_ev) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    }
    auto pace = [&](unsigned si) {
        if (pace_ev.empty()) return;
        cudaEventRecord(pace_ev[si]);
        if (si >= pace_k) cudaEventSynchronize(pace_ev[si - pace_k]);
    };

    auto snap = [&](unsigned si, float* v_curr) {
        const unsigned gi = si % ngroups;
        float* src = masker.Apply(v_curr);
        const byte_t* bsrc = reinterpret_cast<const byte_t*>(src);
        // DRAIN-BEFORE-REFILL: reclaim this group's buffers from snapshot si-ngroups,
        // in-stream (the WriteWait completes before the copy below refills). Without it
        // the copy races the server's read of the prior snapshot => corruption.
        if (si >= ngroups)
            TwDrainKernel<<<submit_grid, 32>>>(groups[gi].Handle());
        TwCopyKernel<<<dim3(copy_bpc, cfg.chunks), 256>>>(groups[gi].Handle(), bsrc);
        ReuseFireKernel<<<submit_grid, 256>>>(groups[gi].Handle(), d_tag_table, si);
        // PER-SNAPSHOT durability (GSBENCH_CLIO_PERSNAP): drain THIS snapshot's Puts to the
        // bdev + fdatasync, inside the loop. Serializes the arm (removes the fire-and-defer
        // overlap) so the DURABLE comparison vs raw/hdf5 is apples-to-apples. No-op unless armed.
        if (cfg.clio_persnap) {
            TwDrainKernel<<<submit_grid, 32>>>(groups[gi].Handle());
            ctp::GpuApi::Synchronize();
            ClioBdevSync(cfg);
        }
        pace(si);
    };
    auto finalize = [&]() {
        for (auto& d : groups) TwDrainKernel<<<submit_grid, 32>>>(d.Handle());
        ctp::GpuApi::Synchronize();
        ClioBdevSync(cfg);   // durability parity with raw/hdf5 — see Cfg::clio_fsync
    };

    double ms = RunSim(cfg, g, snap, finalize, nullptr, &trace);
    for (auto& e : pace_ev) cudaEventDestroy(e);

    ctp::GpuApi::Synchronize();
    // Best-effort I/O-failure check on the reused groups (each holds only its LAST
    // snapshot's per-slot status; a mid-run failure also shows as a read-back checksum
    // mismatch, which is the primary cross-arm correctness gate).
    for (unsigned gi = 0; gi < ngroups; ++gi)
        groups[gi].ThrowIfIoFailed(("gsbench reuse group " + std::to_string(gi)).c_str());

    FreeGrids(g);
    cudaFree(d_tag_table);
    trace.Report("reuse");
    if (cfg.read_pdf) {   // GPU-initiated read-back + PDF (one reused group, re-tagged)
        Pdf pdf;
        const bool amode = (cfg.read_async && groups.size() >= 2);
        const double rms = amode ? GpuReadPdfAsync(cfg, groups, tags, submit_grid, pdf)
                                 : GpuReadPdf(cfg, groups[0], tags, submit_grid, pdf);
        PrintReadResult("gpuh5_relaunch", cfg, rms, pdf, amode ? "async" : "sync");
    }
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// PERSISTENT arm (DESIGN §7 "Option A"): the whole snapshot loop in ONE resident cooperative
// kernel (GsPersistentKernel), head-to-head against the relaunched reuse/async arms. Same 2 reused
// groups + device tag-stamp + bounded memory; the difference is resident-cooperative (grid.sync())
// vs relaunched. FORCES kPinnedHost data — the in-kernel WriteWait would deadlock a resident grid
// against the server's D2H otherwise (see GsPersistentKernel). Its checksum MUST equal the other
// arms'.
double RunPersistentArm(const Cfg& cfg, const char* prefix, uint64_t* checksum,
                        bool async_submit = true) {
    // Select the resident-kernel submit mode: async fire (double-buffered producer) or
    // fire-AND-wait per snapshot (the persistent analog of gpuh5_sync).
    const void* kfn = async_submit
        ? reinterpret_cast<const void*>(GsPersistentKernel<true>)
        : reinterpret_cast<const void*>(GsPersistentKernel<false>);
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    const kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                                /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                                /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    // MANDATORY no-device-op data backend: the server must complete Puts WITHOUT any device
    // operation while the cooperative grid is resident (see GsPersistentKernel's deadlock note).
    // kDeviceMem is therefore forbidden here -- it needs a server-side D2H that cannot schedule.
    //
    // That forced choice is this arm's single largest cost, and it is NOT a property of the
    // persistent design. Measured at N=4096/32 chunks/8 snaps/steps_per=20 on a RAM bdev:
    //   gpuh5_relaunch (kDeviceMem)  65.6 ms
    //   gpuh5           (kPinnedHost) 83.6 ms   <- this arm
    //   gpuh5_relaunch_pinned         85.7 ms
    // i.e. on a MATCHED backend the persistent kernel is already marginally FASTER than the
    // relaunched one; the whole apparent gap is the +20 ms of in-kernel PCIe stores that
    // kPinnedHost staging costs over 512 MB (25.6 GB/s == PCIe gen4 x16). Those stores are
    // synchronous with the kernel, so no amount of in-kernel restructuring can hide them.
    //
    // GSBENCH_PERSIST_UVM=1 selects kManagedUvm instead. It satisfies the SAME no-device-op
    // invariant (IpcGpu2Cpu::RecvIn treats kManagedUvm exactly like kPinnedHost -- the host
    // mapping is authoritative, no D2H writeback), but its pages may be DEVICE-resident, so in
    // principle the in-kernel fill runs at device bandwidth and the host migration is driven by
    // the driver's fault path rather than by the SMs.
    //
    // MEASURED: it does NOT pay off. Same config as above, kManagedUvm = 172 ms against 83.6 ms
    // for kPinnedHost -- 2x WORSE. The fill gets cheaper but the server's host-side read then
    // faults every page back across PCIe one migration at a time, which is far slower than the
    // single streaming write kPinnedHost does. It is correct (checksum matches, no deadlock),
    // just slow. Kept as an opt-in knob so the obvious "why not just use UVM?" question has a
    // number attached instead of being re-tried; the default stays on the proven pinned path.
    const bool uvm = EnvU0("GSBENCH_PERSIST_UVM", 0) != 0;
    const auto data_kind = uvm ? kvhdf5::GpuCteDataset::MemKind::kManagedUvm
                               : kvhdf5::GpuCteDataset::MemKind::kPinnedHost;
    g_actual_pinned = 1;   // FORCED off kDeviceMem regardless of GSBENCH_DATA_PINNED

    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
    clio::cte::core::TagId* d_tag_table = nullptr;
    REQUIRE(cudaMalloc(&d_tag_table,
                       cfg.snaps * sizeof(clio::cte::core::TagId)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tag_table, tags.data(),
                       cfg.snaps * sizeof(clio::cte::core::TagId),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Reused buffer groups. The ASYNC variant needs >= 2 (compute snapshot s+1 into another
    // group while s is still draining); GSBENCH_GROUPS raises that to deepen the pipeline, which
    // is the knob that buys overlap slack when per-snapshot I/O exceeds per-snapshot compute.
    // Memory stays BOUNDED in snaps either way (G groups, not one dataset per snapshot). The SYNC
    // variant submits-AND-waits in-kernel, so only ONE snapshot is ever in flight and any extra
    // group is dead weight -- allocate 1, matching gpuh5_sync's footprint.
    const unsigned ngroups = async_submit
        ? std::min(std::max(2u, EnvU("GSBENCH_GROUPS", 2)), cfg.snaps)
        : 1u;
    std::vector<kvhdf5::GpuCteDataset> groups;
    groups.reserve(ngroups);
    for (unsigned gi = 0; gi < ngroups; ++gi) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/group_%u", prefix, gi);
        groups.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout,
            /*pool_size=*/0, data_kind, /*grid_size=*/0));
    }
    // The kernel indexes a DEVICE ARRAY of handles (groups[s % G]) rather than taking them as
    // by-value params, so the group count is a runtime value and only the active handle is live
    // in the kernel at a time.
    std::vector<kvhdf5::GpuDatasetHandle> h_handles;
    h_handles.reserve(ngroups);
    for (auto& d : groups) h_handles.push_back(d.Handle());
    kvhdf5::GpuDatasetHandle* d_handles = nullptr;
    REQUIRE(cudaMalloc(&d_handles,
                       ngroups * sizeof(kvhdf5::GpuDatasetHandle)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_handles, h_handles.data(),
                       ngroups * sizeof(kvhdf5::GpuDatasetHandle),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    {
        const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
        uint64_t resident = 0;
        for (auto& d : groups) resident += d.DeviceDataBytes();
        g_io_buf_bytes = resident;   // deterministic I/O buffer footprint (2 pinned groups)
        std::fprintf(stderr,
            "GSBENCH_PERSISTENT groups=%u backend=%s resident_data_bytes=%llu (async M=N x snaps "
            "would be %llu)\n", ngroups, uvm ? "uvm" : "pinned",
            (unsigned long long)resident,
            (unsigned long long)(uint64_t(cfg.chunks) * chunk_bytes * cfg.snaps));
    }

    Masker masker(cfg.N, cfg.incompressible != 0);
    Grids g = MakeGrids(cfg.N);

    // Occupancy-sized grid: cooperative launch requires the whole grid co-resident. Grid-stride
    // (compute over cells, I/O over chunks) makes any grid <= that limit correct.
    int dev = 0; cudaGetDevice(&dev);
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
    const int block = 256;
    int blocks_per_sm = 0;
    REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kfn, block, 0) == cudaSuccess);
    REQUIRE(blocks_per_sm > 0);   // 0 => the kernel can't be co-resident (cooperative launch fails)
    const int grid = num_sm * blocks_per_sm;
    std::fprintf(stderr, "GSBENCH_PERSISTENT grid=%d (%d SMs x %d blocks/SM) block=%d\n",
                 grid, num_sm, blocks_per_sm, block);

    // Prewarm is a CORRECTNESS requirement (cold-launch deadlock): force the module resident
    // before the timed cooperative launch.
    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, kfn);
    }

    unsigned N = cfg.N, snaps = cfg.snaps, steps_per = cfg.steps_per;
    const uint32_t* d_mask = masker.MaskPtr();
    unsigned ngroups_arg = ngroups;
    void* args[] = {&g.u_curr, &g.v_curr, &g.u_next, &g.v_next,
                    &d_handles, &ngroups_arg,
                    &d_tag_table, &d_mask, const_cast<GsParams*>(&kGs),
                    &N, &snaps, &steps_per};

    ctp::GpuApi::Synchronize();   // settle the seed before timing
    MemSampler mem;
    mem.Start();                  // no-op unless GSBENCH_MEM=1 (persistent bypasses RunSim)
    auto t0 = std::chrono::steady_clock::now();
    cudaError_t lerr = cudaLaunchCooperativeKernel(
        const_cast<void*>(kfn), dim3(grid), dim3(block), args, 0, 0);
    REQUIRE(lerr == cudaSuccess);
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    mem.Stop();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ClioBdevSync(cfg);   // durability parity (end flush) — matches the other CLIO arms

    for (unsigned gi = 0; gi < ngroups; ++gi)
        groups[gi].ThrowIfIoFailed(("gsbench persistent group " + std::to_string(gi)).c_str());

    FreeGrids(g);
    cudaFree(d_tag_table);
    cudaFree(d_handles);
    if (cfg.read_pdf) {   // GPU-initiated read-back + PDF (one reused group, re-tagged)
        Pdf pdf;
        const bool amode = (cfg.read_async && groups.size() >= 2);
        const double rms = amode ? GpuReadPdfAsync(cfg, groups, tags, cfg.submit_grid(), pdf)
                                 : GpuReadPdf(cfg, groups[0], tags, cfg.submit_grid(), pdf);
        PrintReadResult(async_submit ? "gpuh5" : "gpuh5_sync", cfg, rms, pdf,
                        amode ? "async" : "sync");
    }
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// COMPUTE-ONLY isolation of the persistent kernel (fig_write_decomp2.tex's compute-floor
// probe). GsPersistentComputeOnlyKernel (defined above, "never launched") is argument-for-
// argument compatible with GsPersistentKernel but has every CLIO/I/O call stripped, so it
// can be launched directly with null/zero I/O args -- no CLIO server, no dataset, no tag
// table needed. Same grid-sizing/prewarm/timing shape as RunPersistentArm, minus everything
// that isn't the resident cooperative compute loop. Gives a DIRECT measured compute floor
// for the gpuh5/gpuh5_sync bars, instead of the small-scale ratio probe in
// sweep_stacked128.sh Part A (which only bounds the gap at 1.1%/9.2%, N=4096/5792).
double RunPersistentFloorArm(const Cfg& cfg) {
    Grids g = MakeGrids(cfg.N);

    const void* kfn = reinterpret_cast<const void*>(GsPersistentComputeOnlyKernel);
    int dev = 0; cudaGetDevice(&dev);
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
    const int block = 256;
    int blocks_per_sm = 0;
    REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kfn, block, 0) == cudaSuccess);
    REQUIRE(blocks_per_sm > 0);
    const int grid = num_sm * blocks_per_sm;
    std::fprintf(stderr, "GSBENCH_PERSISTENT_FLOOR grid=%d (%d SMs x %d blocks/SM) block=%d\n",
                 grid, num_sm, blocks_per_sm, block);

    // Prewarm is a CORRECTNESS requirement for cooperative launches (cold-launch deadlock),
    // same as RunPersistentArm.
    if (cfg.prewarm) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, kfn);
    }

    unsigned N = cfg.N, snaps = cfg.snaps, steps_per = cfg.steps_per;
    // Every I/O arg is unused inside GsPersistentComputeOnlyKernel (see its definition) --
    // null/zero is safe and skips CLIO/dataset/tag-table setup entirely.
    const kvhdf5::GpuDatasetHandle* d_handles = nullptr;
    unsigned ngroups_arg = 0;
    const clio::cte::core::TagId* d_tag_table = nullptr;
    const uint32_t* d_mask = nullptr;
    void* args[] = {&g.u_curr, &g.v_curr, &g.u_next, &g.v_next,
                    &d_handles, &ngroups_arg,
                    &d_tag_table, &d_mask, const_cast<GsParams*>(&kGs),
                    &N, &snaps, &steps_per};

    ctp::GpuApi::Synchronize();   // settle the seed before timing
    auto t0 = std::chrono::steady_clock::now();
    cudaError_t lerr = cudaLaunchCooperativeKernel(
        const_cast<void*>(kfn), dim3(grid), dim3(block), args, 0, 0);
    REQUIRE(lerr == cudaSuccess);
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    FreeGrids(g);
    return ms;
}

// Host-driven CLIO arm: the SAME host flow as the raw arm (synchronous D2H into a host
// buffer, then a host-side write call), but the sink is a CLIO PutBlob instead of a
// file write+fsync. It does NOT use the GPU-producer model — no device-side task
// submission, no GpuCteDataset / registered device backends. So comparing it against:
//   - the RAW arm      isolates the STORAGE-PATH cost (CLIO server+bdev vs a plain file);
//   - the SYNC arm     isolates the SUBMISSION MODEL (host-orchestrated vs GPU-producer).
// Durable like CLIO-sync: each PutBlob is waited (bdev write completion).
double RunHostClioArm(const Cfg& cfg, const char* prefix, uint64_t* checksum) {
    const uint64_t gbytes = cfg.grid_bytes();
    const uint64_t chunk_bytes = gbytes / cfg.chunks;

    // One tag per snapshot (distinct path -> tag), same key scheme as the CLIO arms so
    // ChecksumSnapshots reads them back identically. No GPU backends => not bound by the
    // ~16-large-backend ceiling, but we keep cfg.snaps equal for a matched comparison.
    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }

    // Host staging buffer (shm) that PutBlob DMAs from — the host-side counterpart of raw's
    // pinned D2H buffer.
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(gbytes);
    REQUIRE(!buf.IsNull());
    g_io_buf_bytes = gbytes;   // single host staging buffer

    // Put this arm's D2H on the same pinned fast path the GPU-producer arms already enjoy
    // (see Cfg::hostclio_pin). Done OUTSIDE the timed region, matching raw/hdf5, whose
    // pinned buffers are likewise allocated before RunSim.
    bool registered = false;
    if (cfg.hostclio_pin) {
        cudaError_t rc = cudaHostRegister(buf.ptr_, gbytes, cudaHostRegisterDefault);
        registered = (rc == cudaSuccess);
        if (!registered)
            std::fprintf(stderr,
                "[hostclio] WARNING: cudaHostRegister failed (%s); D2H stays on the slow "
                "pageable path and this arm is handicapped vs sync/async\n",
                cudaGetErrorString(rc));
    }
    std::fprintf(stderr, "[hostclio] shm staging buffer registered=%d\n", int(registered));

    Masker masker(cfg.N, cfg.incompressible != 0);
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        // Same host D2H as raw: pull the (incompressible) current grid into the host buffer.
        d2h.Copy(buf.ptr_, masker.Apply(v_curr));
        // Then persist each chunk to CLIO from the host (synchronous wait == durable).
        for (unsigned c = 0; c < cfg.chunks; ++c) {
            ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
            shm.off_ += uint64_t(c) * chunk_bytes;   // point at chunk c within the buffer
            auto t = CLIO_CTE_CLIENT->AsyncPutBlob(tags[si], std::to_string(c),
                                                   clio::run::u64(0), chunk_bytes, shm);
            t.Wait();
            REQUIRE(t->GetReturnCode() == 0);
        }
    };
    auto finalize = [&]() {
        ClioBdevSync(cfg);   // durability parity with raw/hdf5 — see Cfg::clio_fsync
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("hostclio");
    // Host-side CLIO reader: GetBlob each chunk back into the SAME reused shm staging buffer
    // the writer used, then one H2D of the whole snapshot into the PDF. Reusing the buffer
    // matters — the checksum path allocates per chunk, which would measure the allocator
    // rather than the read.
    if (cfg.read_pdf) {
        Pdf pdf;
        HostReadStage stage(gbytes);
        const double rms = RunReadPhase(cfg, [&](unsigned si) {
            for (unsigned c = 0; c < cfg.chunks; ++c) {
                ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
                shm.off_ += uint64_t(c) * chunk_bytes;
                auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tags[si], std::to_string(c),
                                                       clio::run::u64(0), chunk_bytes,
                                                       clio::run::u32(0), shm);
                t.Wait();
                REQUIRE(t->GetReturnCode() == 0);
            }
            stage.Bin(pdf, buf.ptr_);
        });
        PrintReadResult("hostclio", cfg, rms, pdf, "host");
    }
    if (registered) cudaHostUnregister(buf.ptr_);
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// ---- raw (no-CLIO) disk arm -----------------------------------------------

void EnsureDir(const std::string& d) {
    if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("mkdir " + d);
}
// Write in <=1 MiB O_DIRECT chunks (matches the CLIO bdev's block loop; a single huge
// O_DIRECT pwrite from CUDA-pinned memory hit a ~5x-slow kernel path on this box).
void WriteAllAt(int fd, off_t off, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    constexpr size_t kBlk = 1u << 20;
    while (bytes) {
        size_t want = bytes < kBlk ? bytes : kBlk;
        ssize_t n = pwrite(fd, p, want, off);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("pwrite"); }
        p += n; off += n; bytes -= size_t(n);
    }
}

// Fault in the whole [0, bytes) extent of an ALREADY-CREATED file by writing real zeros
// over it (see Cfg::prefault). Call BEFORE RunSim — never inside the timed region.
//
// Deliberately opens its OWN fd, WITHOUT O_DIRECT and WITHOUT O_TRUNC: the arm's real fd may
// be O_DIRECT (GSBENCH_RAW_ODIRECT=1), whose alignment rules a plain zero buffer would
// violate, and O_TRUNC would throw away the ftruncate the caller just did. The arm's fd is
// left completely undisturbed; this fd is closed before the run starts.
void PrefaultFile(const std::string& path, uint64_t bytes) {
    int fd = open(path.c_str(), O_WRONLY);   // no O_DIRECT, no O_TRUNC, no O_CREAT
    REQUIRE(fd >= 0);
    constexpr size_t kBlk = 4u << 20;
    std::vector<uint8_t> zeros(kBlk, 0);
    off_t off = 0;
    while (uint64_t(off) < bytes) {
        size_t want = size_t(std::min<uint64_t>(kBlk, bytes - uint64_t(off)));
        ssize_t n = pwrite(fd, zeros.data(), want, off);
        if (n < 0) { if (errno == EINTR) continue; close(fd); throw std::runtime_error("prefault pwrite"); }
        off += n;
    }
    fdatasync(fd);
    close(fd);
}

// A competent (deliberately NOT maximally-tuned) decoupled writer: ONE background thread
// + a small pinned-buffer pool (standard double-buffered checkpoint I/O). It lets disk
// I/O OVERLAP the subsequent sim steps instead of stalling inline — the same structural
// benefit CLIO gets by offloading I/O to its server. No libaio / queue-depth / thread
// fan-out (that would be an "expert" baseline; we keep it fair). O_DIRECT for cache-bypass
// parity with CLIO's kFile bdev.
class BgWriter {
public:
    BgWriter(std::string path, uint64_t gbytes, uint64_t wbytes, unsigned nbuf,
             unsigned snaps, bool odirect, bool do_fsync)
        : gbytes_(gbytes), wbytes_(wbytes), fsync_(do_fsync) {
        // ONE pre-allocated checkpoint file, snapshots written at distinct offsets — same
        // as CLIO's bdev (single truncated file, offset per blob). Avoids the per-snapshot
        // file create/O_TRUNC + async-discard churn that throttled the 12-fresh-files
        // pattern ~5x when the writes were spaced out by GPU work.
        int flags = O_WRONLY | O_CREAT | O_TRUNC | (odirect ? O_DIRECT : 0);
        fd_ = open(path.c_str(), flags, 0644);
        REQUIRE(fd_ >= 0);
        REQUIRE(ftruncate(fd_, off_t(snaps) * off_t(wbytes_)) == 0);  // preallocate size
        bufs_.resize(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&bufs_[i]), wbytes_)
                    == cudaSuccess);                 // pinned: fast D2H
            std::memset(bufs_[i] + gbytes_, 0, wbytes_ - gbytes_);  // O_DIRECT pad tail
            free_.push_back(int(i));
        }
        th_ = std::thread([this] { Run(); });
    }
    // Producer (CUDA thread): grab a free buffer to D2H into (blocks if all in flight).
    uint8_t* Acquire(int* idx) {
        std::unique_lock<std::mutex> lk(m_);
        cv_free_.wait(lk, [this] { return !free_.empty(); });
        int i = free_.back(); free_.pop_back();
        *idx = i; return bufs_[i];
    }
    // Producer: buffer filled (D2H complete) -> hand to the writer.
    void Submit(int idx, unsigned s) {
        { std::lock_guard<std::mutex> lk(m_); work_.push_back({idx, s}); }
        cv_work_.notify_one();
    }
    // Drain + join (call INSIDE the timed region). Yields writer busy-ms. Checksum is
    // computed by the caller AFTER timing (readback), matching the CLIO arms.
    void Finish(double* writer_ms) {
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        cv_work_.notify_one();
        th_.join();
        if (fd_ >= 0) close(fd_);
        for (auto* b : bufs_) cudaFreeHost(b);
        *writer_ms = writer_ms_;
        REQUIRE(!err_);
    }
private:
    void Run() {
        for (;;) {
            std::pair<int, unsigned> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [this] { return !work_.empty() || done_; });
                if (work_.empty() && done_) return;
                job = work_.front(); work_.pop_front();
            }
            auto t0 = std::chrono::steady_clock::now();
            uint8_t* buf = bufs_[job.first];
            // NB: checksum is computed AFTER timing (post-run readback, like the CLIO arms)
            // — folding a scalar FNV over 1.9 GB here would add ~6 s to the timed region and
            // unfairly penalize raw vs CLIO (whose ChecksumSnapshots runs post-timing).
            WriteAllAt(fd_, off_t(job.second) * off_t(wbytes_), buf, wbytes_);  // offset slot
            if (fsync_ && fdatasync(fd_) != 0) err_ = true;  // durability parity with CLIO
            writer_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            { std::lock_guard<std::mutex> lk(m_); free_.push_back(job.first); }
            cv_free_.notify_one();
        }
    }
    int fd_ = -1; uint64_t gbytes_, wbytes_; bool fsync_ = true;
    std::vector<uint8_t*> bufs_;
    std::mutex m_; std::condition_variable cv_free_, cv_work_;
    std::vector<int> free_; std::deque<std::pair<int, unsigned>> work_;
    bool done_ = false, err_ = false; std::thread th_;
    double writer_ms_ = 0;
};

// Read the persisted checkpoint back and fold FNV in snapshot order — AFTER timing (a
// scalar FNV over 1.9 GB is ~6 s and must NOT sit in the timed region; the CLIO arms
// likewise checksum post-timing).
uint64_t RawReadbackChecksum(const std::string& path, const Cfg& cfg,
                             uint64_t gbytes, uint64_t wbytes) {
    uint64_t h = 1469598103934665603ull;
    int fd = open(path.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    std::vector<uint8_t> rb(gbytes);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        off_t base = off_t(s) * off_t(wbytes);
        size_t got = 0;
        while (got < gbytes) {
            ssize_t n = pread(fd, rb.data() + got, gbytes - got, base + off_t(got));
            REQUIRE(n > 0);
            got += size_t(n);
        }
        h = Fnv1a(rb.data(), gbytes, h);
    }
    close(fd);
    return h;
}

// raw reader: pread each snapshot back, stage H2D, bin into the PDF (same GPU histogram
// every other arm uses, so only the READ path differs between arms).
double RawReadPdf(const std::string& path, const Cfg& cfg,
                  uint64_t gbytes, uint64_t wbytes, Pdf& pdf) {
    int fd = open(path.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    std::vector<uint8_t> rb(gbytes);
    HostReadStage stage(gbytes);
    const double ms = RunReadPhase(cfg, [&](unsigned s) {
        const off_t base = off_t(s) * off_t(wbytes);
        size_t got = 0;
        while (got < gbytes) {
            ssize_t n = pread(fd, rb.data() + got, gbytes - got, base + off_t(got));
            REQUIRE(n > 0);
            got += size_t(n);
        }
        stage.Bin(pdf, rb.data());
    });
    close(fd);
    return ms;
}

// Raw arm (no CLIO): identical sim + timed loop; each snapshot D2Hs the (incompressible)
// v-grid to host and persists it into one pre-allocated file. Two structures selected by
// GSBENCH_RAW_INLINE:
//   0 = background writer thread — I/O overlaps the next sim steps (the natural design, but
//       this box throttles GPU-concurrent disk I/O ~5x, penalizing the overlap);
//   1 = inline synchronous — GPU idle during the write, matching host-CLIO / sync-CLIO, for
//       a storage-path comparison free of that throttle.
double RunRawArm(const Cfg& cfg, uint64_t* checksum) {
    EnsureDir(cfg.disk_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    constexpr uint64_t kAlign = 4096;
    const uint64_t wbytes = (gbytes + kAlign - 1) & ~(kAlign - 1);  // O_DIRECT length
    const std::string path = cfg.disk_dir + "/checkpoint.bin";
    Masker masker(cfg.N, cfg.incompressible != 0);
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);

    if (cfg.raw_inline) {
        int flags = O_WRONLY | O_CREAT | O_TRUNC | (cfg.raw_odirect ? O_DIRECT : 0);
        int fd = open(path.c_str(), flags, 0644);
        REQUIRE(fd >= 0);
        REQUIRE(ftruncate(fd, off_t(cfg.snaps) * off_t(wbytes)) == 0);
        // ftruncate only sizes the file; the pages are still sparse. Fault them in now, off
        // the clock, so we time the write and not the kernel's page allocator (Cfg::prefault).
        if (cfg.prefault) PrefaultFile(path, uint64_t(cfg.snaps) * wbytes);
        uint8_t* buf = nullptr;
        REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&buf), wbytes) == cudaSuccess);
        std::memset(buf + gbytes, 0, wbytes - gbytes);
        g_io_buf_bytes = wbytes;   // single inline staging buffer
        Grids g = MakeGrids(cfg.N);
        double write_ms = 0;
        auto snap = [&](unsigned si, float* v_curr) {
            d2h.Copy(buf, masker.Apply(v_curr));
            auto a = std::chrono::steady_clock::now();          // GPU idle during this write
            WriteAllAt(fd, off_t(si) * off_t(wbytes), buf, wbytes);
            if (cfg.raw_fsync) fdatasync(fd);
            write_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - a).count();
        };
        auto finalize = [&]() {};
        double ms = RunSim(cfg, g, snap, finalize, nullptr);
        FreeGrids(g);
        d2h.Report("raw_inline");
        close(fd);
        cudaFreeHost(buf);
        std::fprintf(stderr, "[raw] total=%.1f ms  write=%.1f ms  (inline, GPU idle)\n",
                     ms, write_ms);
        if (cfg.read_pdf) {
            Pdf pdf;
            const double rms = RawReadPdf(path, cfg, gbytes, wbytes, pdf);
            PrintReadResult("raw_inline", cfg, rms, pdf, "host");
        }
        *checksum = RawReadbackChecksum(path, cfg, gbytes, wbytes);
        return ms;
    }

    g_io_buf_bytes = uint64_t(3) * wbytes;   // BgWriter's nbuf=3 pinned staging buffers
    BgWriter writer(path, gbytes, wbytes, /*nbuf=*/3, cfg.snaps,
                    /*odirect=*/cfg.raw_odirect != 0, /*do_fsync=*/cfg.raw_fsync != 0);
    // The ctor has open()ed + ftruncate()d the file; fault its pages in through a separate
    // non-O_DIRECT fd, still BEFORE RunSim starts the clock (Cfg::prefault).
    if (cfg.prefault) PrefaultFile(path, uint64_t(cfg.snaps) * wbytes);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        int idx;
        uint8_t* buf = writer.Acquire(&idx);
        // SYNCHRONOUS D2H on the DEFAULT stream (ordered after the step / mask kernel), then
        // hand the buffer to the writer, which writes it while the NEXT sim steps run.
        d2h.Copy(buf, masker.Apply(v_curr));
        writer.Submit(idx, si);
    };
    double writer_ms = 0;
    auto finalize = [&]() { writer.Finish(&writer_ms); };  // drain inside timed region
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("raw_threaded");
    std::fprintf(stderr,
        "[raw] total=%.1f ms  writer-busy=%.1f ms (overlapped with compute)  nbuf=3\n",
        ms, writer_ms);
    if (cfg.read_pdf) {
        Pdf pdf;
        const double rms = RawReadPdf(path, cfg, gbytes, wbytes, pdf);
        PrintReadResult("raw_threaded", cfg, rms, pdf, "host");
    }
    *checksum = RawReadbackChecksum(path, cfg, gbytes, wbytes);
    return ms;
}

// ---- hdf5 arm (the conventional path: GPU -> D2H -> HDF5) -------------------
#if GSBENCH_HAVE_HDF5

#define H5CHK(expr)                                                        \
    do {                                                                   \
        if ((expr) < 0) {                                                  \
            H5Eprint2(H5E_DEFAULT, stderr);                                \
            throw std::runtime_error("HDF5 call failed: " #expr);          \
        }                                                                  \
    } while (0)

// The HDF5 side of the arm: one file, one CHUNKED dataset per snapshot at /v/step_NNNN,
// chunk dims {N/chunks, N} — byte-for-byte the same chunk geometry the CLIO arms use, so
// the comparison is about the I/O path and not about how the data was carved up.
//
// This is deliberately a TUNED HDF5, not a naive one (a weak baseline would be a bug, not
// a win). What is tuned, and why:
//   - chunk cache sized to 0 (H5Pset_chunk_cache): every write here is a WHOLE chunk with
//     no filters, which is precisely the case HDF5's cache-bypass path exists for. Left at
//     the stock 1 MB the small-chunk end of the sweep stages every chunk through the cache
//     for no reason. GSBENCH_HDF5_RDCC_MB>0 sizes it instead; GSBENCH_HDF5_STOCK=1 leaves
//     HDF5's defaults entirely alone.
//   - H5D_ALLOC_TIME_EARLY: allocate the dataset's 156 MB of file space in one go rather
//     than growing it chunk-by-chunk mid-write.
//   - H5D_FILL_TIME_NEVER: we overwrite every byte, so the default fill pass would zero
//     156 MB immediately before we write over it. With EARLY alloc that is not free.
//   - H5Dwrite_chunk (opt-in): skips the cache AND the filter pipeline outright.
//
// HDF5 here is built WITHOUT thread-safety, so every HDF5 call in an arm must come from a
// single thread. Inline mode: the main thread. Threaded mode: the writer thread owns the
// file for its whole lifetime (Hdf5Writer). Readback happens after the writer has joined.
class Hdf5Sink {
public:
    explicit Hdf5Sink(const Cfg& cfg)
        : cfg_(cfg),
          rows_(cfg.N / cfg.chunks),
          chunk_bytes_(uint64_t(cfg.N / cfg.chunks) * cfg.N * sizeof(float)) {}

    void Open(const std::string& path) {
        // sec2 (the default driver): one buffered fd — the same kernel path the raw arm's
        // pwrite() takes. Swapping in core/direct would change what is being measured.
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        H5CHK(fapl);
        stdio_ = (cfg_.hdf5_vfd == "stdio");
        if (stdio_) H5CHK(H5Pset_fapl_stdio(fapl));
        else        H5CHK(H5Pset_fapl_sec2(fapl));
        if (cfg_.hdf5_meta_block_kb)
            H5CHK(H5Pset_meta_block_size(fapl, hsize_t(cfg_.hdf5_meta_block_kb) << 10));
        file_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5CHK(file_);
        H5CHK(H5Pclose(fapl));
        grp_ = H5Gcreate2(file_, "/v", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5CHK(grp_);
        // DURABILITY PARITY. H5Fflush only pushes HDF5's own buffers into the OS page
        // cache; it does not reach the device. The raw arm fdatasyncs. Both drivers hand
        // back the underlying handle, so we can hold HDF5 to the same standard rather than
        // quietly letting it win by doing less work: sec2 gives an fd, stdio a FILE* (which
        // must be fflush'd first — its buffer is in userspace, one level further from disk).
        void* h = nullptr;
        H5CHK(H5Fget_vfd_handle(file_, H5P_DEFAULT, &h));
        // H5Fget_vfd_handle yields a pointer TO the driver's handle, for both drivers:
        // int* for sec2, FILE** for stdio. (Casting the latter straight to FILE* gets you
        // "glibc detected an invalid stdio handle" at the first fflush.)
        if (stdio_) fp_ = h ? *static_cast<FILE**>(h) : nullptr;
        else        fd_ = h ? *static_cast<int*>(h) : -1;
    }

    // Push this snapshot all the way to the device, matching raw's per-snapshot fdatasync.
    void Sync() {
        H5CHK(H5Fflush(file_, H5F_SCOPE_GLOBAL));   // HDF5's caches -> the driver
        if (stdio_) {
            if (fp_) { std::fflush(fp_); fdatasync(fileno(fp_)); }
        } else if (fd_ >= 0) {
            fdatasync(fd_);
        }
    }

    // Create all `snaps` datasets up front and hold them open. Called OUTSIDE the timed
    // region (see Cfg::hdf5_precreate for why that is parity rather than a favour).
    void PrecreateAll() {
        if (!cfg_.hdf5_precreate) return;
        dsets_.resize(cfg_.snaps, -1);
        for (unsigned s = 0; s < cfg_.snaps; ++s) dsets_[s] = CreateDataset(s);
    }

    // PAGE-FAULT PARITY (Cfg::prefault). H5D_ALLOC_TIME_EARLY reserves the dataset's file
    // space but leaves the file sparse, so the timed H5Dwrite would fault in and zero every
    // page — a cost the CLIO arms' preallocated bdev never pays. Write a zero-filled host
    // buffer once to EVERY dataset, through the SAME write path the timed run uses, so the
    // timed run merely OVERWRITES existing pages. Call OUTSIDE the timed region.
    //
    // PREFAULT IMPLIES PRECREATE for the hdf5 arms: WriteSnap creates the dataset lazily when
    // !hdf5_precreate, so a prefault pass would create every dataset and the timed WriteSnap
    // would then H5Dcreate the same name again and fail. The callers force hdf5_precreate on
    // (on a local Cfg copy) whenever prefault is on, and PrecreateAll() runs first.
    //
    // Do NOT pwrite raw zeros at the HDF5 file's fd — that would flatten its metadata.
    void PrefaultAll(const void* zeros) {
        if (!cfg_.prefault || !cfg_.hdf5_precreate) return;
        for (unsigned s = 0; s < cfg_.snaps; ++s) WriteSnap(s, zeros);
    }

    // Persist one snapshot from a host buffer holding the full N*N masked grid.
    void WriteSnap(unsigned si, const void* host) {
        const bool pre = cfg_.hdf5_precreate != 0;
        hid_t dset = pre ? dsets_[si] : CreateDataset(si);

        // Naive forces the high-level H5Dwrite of the whole array (H5Dwrite_chunk is illegal on
        // a contiguous dataset anyway).
        if (cfg_.hdf5_direct_chunk && !cfg_.hdf5_naive) {
            const auto* p = static_cast<const uint8_t*>(host);
            for (unsigned c = 0; c < cfg_.chunks; ++c) {
                hsize_t off[2] = {hsize_t(c) * hsize_t(rows_), 0};
                H5CHK(H5Dwrite_chunk(dset, H5P_DEFAULT, /*filter_mask=*/0, off,
                                     size_t(chunk_bytes_),
                                     p + uint64_t(c) * chunk_bytes_));
            }
        } else {
            H5CHK(H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, host));
        }
        if (!pre) H5CHK(H5Dclose(dset));
        if (cfg_.raw_fsync) Sync();   // durability parity with raw's per-snapshot fdatasync
    }

    // ---- async-VOL variant (arm `hdf5_async`) ------------------------------
    // Issue snapshot si's create+write+close onto an HDF5 event set instead of executing
    // them inline. With the Asynchronous I/O VOL connector loaded (HDF5_VOL_CONNECTOR=async)
    // these return immediately and an Argobots thread performs the I/O while the GPU runs
    // the next sim steps — the host-side analogue of what our async arm does from the
    // device. Without the connector the same calls silently execute synchronously, so this
    // arm is only meaningful when Hdf5AsyncVolActive() says the connector is really loaded.
    //
    // The buffer contract is the reason the caller must hand us a DISTINCT buffer per
    // snapshot: the connector reads `host` at some point in the future, so recycling a small
    // pool would let the next D2H overwrite bytes that have not been written out yet. That
    // is silent corruption, and it would show up as a checksum mismatch (it did, before this
    // comment existed).
    //
    // When the datasets were precreated (which prefault forces — see PrefaultAll) we write
    // into the already-open handle and let Close() close it, exactly as the sync WriteSnap
    // does; only the create+close move off the event set, the write itself is still async.
    void WriteSnapAsync(unsigned si, const void* host, hid_t es) {
        const bool pre = cfg_.hdf5_precreate != 0;
        hid_t dset = pre ? dsets_[si] : CreateDatasetAsync(si, es);
        H5CHK(H5Dwrite_async(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, host,
                             es));
        if (!pre) H5CHK(H5Dclose_async(dset, es));
    }

    void Close() {
        for (hid_t d : dsets_) if (d >= 0) H5Dclose(d);
        dsets_.clear();
        if (grp_ >= 0) { H5Gclose(grp_); grp_ = -1; }
        if (file_ >= 0) { H5Fclose(file_); file_ = -1; }
    }

    // Is the async VOL actually the connector on this file, or did we silently fall back to
    // native (which would make the arm a mislabelled copy of the inline one)?
    bool AsyncVolActive() const {
        char name[64] = {0};
        ssize_t n = H5VLget_connector_name(file_, name, sizeof(name));
        return n > 0 && std::strcmp(name, "async") == 0;
    }

    // vol-async's OWN drain API (H5Fwait, "gov.lbl.async.file.wait"), reached through the
    // public VOL optional-op interface so we do not have to link the connector's static
    // libasynchdf5 into this binary.
    //
    // Why not just H5ESwait? Because H5ESwait is the connector's SLOW drain. Its
    // H5VL_async_request_wait() releases the HDF5 global mutex, waits on the task, and then
    // re-acquires the mutex in a retry loop whose backoff is a hard-coded usleep(1000000) —
    // ONE SECOND per retry (h5_async_vol.c:22827). Every H5ESwait that finds work still
    // outstanding therefore costs a whole-second-quantised ~2 s of pure sleep, regardless of
    // how many bytes it drained (measured: 1 wait -> 2.00 s, 12 waits -> 24.00 s, same 1875
    // MB either way). H5Fwait reaches H5VL_async_file_wait(), which does the same job and
    // re-acquires the mutex with NO backoff sleep at all. Same semantics, no fixed tax.
    //
    // We still H5ESwait afterwards, for error reporting: by then every task is done, so each
    // request takes request_wait's is_done early-out and never touches the mutex.
    //
    // NOT the default, though (GSBENCH_HDF5_ASYNC_FWAIT), because it trades one pathology for
    // another: H5VL_async_file_wait re-acquires the mutex in a loop with NO sleep at all
    // (h5_async_vol.c:2945), so under CPU oversubscription it busy-spins against the very
    // Argobots thread it is waiting on and effectively livelocks.
    // Returns false if the op is unavailable (i.e. connector not loaded) -> caller falls back.
    bool FileWait() const {
        static int op = -2;   // -2 = not looked up yet, -1 = unavailable
        if (op == -2) {
            if (H5VLfind_opt_operation(H5VL_SUBCLS_FILE, "gov.lbl.async.file.wait", &op) < 0)
                op = -1;
        }
        if (op < 0) return false;
        H5VL_optional_args_t args;
        args.op_type = op;
        args.args = nullptr;
        return H5VLfile_optional_op(file_, &args, H5P_DEFAULT, H5ES_NONE) >= 0;
    }

private:
    // Property lists for snapshot si's dataset. Shared by the sync and async create paths so
    // both arms get byte-identical chunk geometry and tuning.
    void MakeDatasetPlists(hid_t* space, hid_t* dcpl, hid_t* dapl) {
        const hsize_t dims[2] = {hsize_t(cfg_.N), hsize_t(cfg_.N)};
        const hsize_t cdims[2] = {hsize_t(rows_), hsize_t(cfg_.N)};

        *space = H5Screate_simple(2, dims, nullptr);
        H5CHK(*space);
        *dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5CHK(*dcpl);
        // Naive contiguous: leave the DCPL at its default -> CONTIGUOUS layout, no fill/alloc
        // tuning. Naive chunked (GSBENCH_HDF5_NAIVE_CHUNKED): chunk at the tuned geometry but
        // still skip the fill/alloc tuning. Non-naive: chunk + full tuning (gated by stock).
        const bool chunk_it = !cfg_.hdf5_naive || cfg_.hdf5_naive_chunked;
        if (chunk_it)
            H5CHK(H5Pset_chunk(*dcpl, 2, cdims));   // geometry parity with the CLIO arms
        if (!cfg_.hdf5_stock && !cfg_.hdf5_naive) {
            H5CHK(H5Pset_fill_time(*dcpl, H5D_FILL_TIME_NEVER));
            if (cfg_.hdf5_early_alloc)
                H5CHK(H5Pset_alloc_time(*dcpl, H5D_ALLOC_TIME_EARLY));
        }

        *dapl = H5P_DEFAULT;
        if (!cfg_.hdf5_stock && !cfg_.hdf5_naive) {
            *dapl = H5Pcreate(H5P_DATASET_ACCESS);
            H5CHK(*dapl);
            // nslots wants to be prime-ish and comfortably > the chunk count (HDF5 hashes
            // chunk index -> slot). Irrelevant when nbytes==0, but the sweep also runs this
            // at GSBENCH_HDF5_RDCC_MB>0.
            size_t nslots = size_t(cfg_.chunks) * 10 + 1;
            if (nslots < 521) nslots = 521;
            const size_t nbytes = size_t(cfg_.hdf5_rdcc_mb) << 20;
            // w0=1.0: fully-written chunks are the preferred eviction victims — we never
            // read a chunk back during the run, so keeping one resident buys nothing.
            H5CHK(H5Pset_chunk_cache(*dapl, nslots, nbytes, 1.0));
        }
    }

    hid_t CreateDatasetAsync(unsigned si, hid_t es) {
        hid_t space, dcpl, dapl;
        MakeDatasetPlists(&space, &dcpl, &dapl);
        char name[64];
        std::snprintf(name, sizeof(name), "step_%04u", si);
        hid_t dset = H5Dcreate_async(grp_, name, H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl,
                                     dapl, es);
        H5CHK(dset);
        if (dapl != H5P_DEFAULT) H5CHK(H5Pclose(dapl));
        H5CHK(H5Pclose(dcpl));
        H5CHK(H5Sclose(space));
        return dset;
    }

    hid_t CreateDataset(unsigned si) {
        hid_t space, dcpl, dapl;
        MakeDatasetPlists(&space, &dcpl, &dapl);
        char name[64];
        std::snprintf(name, sizeof(name), "step_%04u", si);
        // File type IEEE_F32LE == the native x86 float, so HDF5 takes its no-conversion
        // (memcpy) path and the persisted bytes are the masked bytes verbatim. That is what
        // makes the cross-arm checksum comparable at all.
        hid_t dset = H5Dcreate2(grp_, name, H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, dapl);
        H5CHK(dset);
        if (dapl != H5P_DEFAULT) H5CHK(H5Pclose(dapl));
        H5CHK(H5Pclose(dcpl));
        H5CHK(H5Sclose(space));
        return dset;
    }

    std::vector<hid_t> dsets_;   // empty unless precreate
    const Cfg& cfg_;
    unsigned rows_;
    uint64_t chunk_bytes_;
    hid_t file_ = -1, grp_ = -1;
    bool stdio_ = false;
    int fd_ = -1;        // sec2
    FILE* fp_ = nullptr; // stdio
};

// Background HDF5 writer: the structural twin of the raw arm's BgWriter (one thread, a
// small pinned-buffer pool, double-buffered checkpoint I/O) so that `hdf5` threaded vs
// `raw` threaded differs ONLY in the sink. The thread owns the HDF5 file end to end —
// opening it in Run() rather than in the constructor keeps every HDF5 call on one thread,
// which a non-threadsafe libhdf5 requires. It also means dataset creation overlaps compute,
// which is what a real background-checkpoint code gets too.
// Allocate / free an hdf5 staging buffer, pinned or pageable (see Cfg::hdf5_pinned).
uint8_t* Hdf5AllocBuf(const Cfg& cfg, uint64_t bytes) {
    uint8_t* p = nullptr;
    if (cfg.hdf5_pinned) {
        REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&p), bytes) == cudaSuccess);
    } else {
        p = static_cast<uint8_t*>(std::malloc(bytes));
        REQUIRE(p != nullptr);
    }
    return p;
}
void Hdf5FreeBuf(const Cfg& cfg, uint8_t* p) {
    if (!p) return;
    if (cfg.hdf5_pinned) cudaFreeHost(p); else std::free(p);
}

class Hdf5Writer {
public:
    Hdf5Writer(const Cfg& cfg, std::string path, uint64_t gbytes, unsigned nbuf)
        : cfg_(cfg), path_(std::move(path)), gbytes_(gbytes) {
        bufs_.resize(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            bufs_[i] = Hdf5AllocBuf(cfg_, gbytes_);
            free_.push_back(int(i));
        }
        th_ = std::thread([this] { Run(); });
        // BLOCK until the thread has opened the file and pre-created the datasets. The
        // caller constructs us before RunSim, so this keeps HDF5's setup out of the timed
        // region — same as the CLIO arms' MakeSnapDatasets and raw's ftruncate. Without the
        // latch the setup would race into the timing window instead.
        std::string err;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_ready_.wait(lk, [this] { return ready_; });
            err = err_;
        }
        if (!err.empty()) {
            th_.join();   // else ~thread on a joinable thread => std::terminate
            throw std::runtime_error("hdf5 writer open: " + err);
        }
    }
    uint8_t* Acquire(int* idx) {
        std::unique_lock<std::mutex> lk(m_);
        cv_free_.wait(lk, [this] { return !free_.empty(); });
        int i = free_.back(); free_.pop_back();
        *idx = i; return bufs_[i];
    }
    void Submit(int idx, unsigned s) {
        { std::lock_guard<std::mutex> lk(m_); work_.push_back({idx, s}); }
        cv_work_.notify_one();
    }
    // Drain + join + close the file. Called INSIDE the timed region (raw does the same).
    void Finish(double* writer_ms) {
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        cv_work_.notify_one();
        th_.join();
        for (auto* b : bufs_) Hdf5FreeBuf(cfg_, b);
        *writer_ms = writer_ms_;
        if (!err_.empty()) throw std::runtime_error("hdf5 writer thread: " + err_);
    }
private:
    // Signal the constructor that setup finished (successfully or not) exactly once.
    void SignalReady() {
        { std::lock_guard<std::mutex> lk(m_); ready_ = true; }
        cv_ready_.notify_all();
    }
    void Run() {
        Hdf5Sink sink(cfg_);
        try {
            sink.Open(path_);
            sink.PrecreateAll();   // outside the timed region: ctor blocks until we get here
            // Same window, same reason: fault in the file's pages (Cfg::prefault). Must run on
            // THIS thread (libhdf5 is not thread-safe) and it is safe to borrow bufs_[0] — the
            // producer cannot have Acquire()d anything yet, it is still blocked in the ctor.
            if (cfg_.prefault && !bufs_.empty()) {
                std::memset(bufs_[0], 0, gbytes_);
                sink.PrefaultAll(bufs_[0]);
            }
            SignalReady();
            for (;;) {
                std::pair<int, unsigned> job;
                {
                    std::unique_lock<std::mutex> lk(m_);
                    cv_work_.wait(lk, [this] { return !work_.empty() || done_; });
                    if (work_.empty() && done_) break;
                    job = work_.front(); work_.pop_front();
                }
                auto t0 = std::chrono::steady_clock::now();
                sink.WriteSnap(job.second, bufs_[job.first]);
                writer_ms_ += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                { std::lock_guard<std::mutex> lk(m_); free_.push_back(job.first); }
                cv_free_.notify_one();
            }
            sink.Close();   // file close is part of the timed region, as spec'd
        } catch (const std::exception& e) {
            { std::lock_guard<std::mutex> lk(m_); err_ = e.what();
              // Never strand the producer waiting on a buffer that will never come back.
              for (size_t i = 0; i < bufs_.size(); ++i) free_.push_back(int(i)); }
            cv_free_.notify_all();
            SignalReady();   // no-op if setup already succeeded; unblocks the ctor if not
        }
    }
    const Cfg& cfg_;
    std::string path_;
    uint64_t gbytes_;
    std::vector<uint8_t*> bufs_;
    std::mutex m_; std::condition_variable cv_free_, cv_work_, cv_ready_;
    std::vector<int> free_; std::deque<std::pair<int, unsigned>> work_;
    bool done_ = false, ready_ = false; std::string err_; std::thread th_;
    double writer_ms_ = 0;
};

// Read every snapshot dataset back out of the file and fold FNV in snapshot order —
// AFTER the timed region, exactly like RawReadbackChecksum and ChecksumSnapshots. (An
// earlier version of this benchmark timed one arm's checksum and not another's and
// produced a completely fictional speedup. Not again.)
uint64_t Hdf5ReadbackChecksum(const std::string& path, const Cfg& cfg, uint64_t gbytes) {
    uint64_t h = 1469598103934665603ull;
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    H5CHK(file);
    std::vector<uint8_t> rb(gbytes);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char name[80];
        std::snprintf(name, sizeof(name), "/v/step_%04u", s);
        hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
        H5CHK(dset);
        H5CHK(H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rb.data()));
        H5CHK(H5Dclose(dset));
        h = Fnv1a(rb.data(), gbytes, h);
    }
    H5CHK(H5Fclose(file));
    return h;
}

// hdf5 reader: H5Dread each snapshot back, stage H2D, bin into the PDF.
double Hdf5ReadPdf(const std::string& path, const Cfg& cfg, uint64_t gbytes, Pdf& pdf) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    H5CHK(file);
    std::vector<uint8_t> rb(gbytes);
    HostReadStage stage(gbytes);
    const double ms = RunReadPhase(cfg, [&](unsigned s) {
        char name[80];
        std::snprintf(name, sizeof(name), "/v/step_%04u", s);
        hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
        H5CHK(dset);
        H5CHK(H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rb.data()));
        H5CHK(H5Dclose(dset));
        stage.Bin(pdf, rb.data());
    });
    H5CHK(H5Fclose(file));
    return ms;
}

// The conventional path, and the paper's real baseline: GPU compute -> cudaMemcpy D2H ->
// HDF5 write. Same GsStepKernel, same RunSim, same Masker as every other arm; only the
// sink differs. Two structures, selected by the SAME GSBENCH_RAW_INLINE knob the raw arm
// uses, so `hdf5` and `raw` are always compared under matched semantics:
//   0 = background writer thread — the HDF5 write overlaps the next sim steps (pairs with
//       raw-threaded and clio-async);
//   1 = inline synchronous — GPU idle during the write (pairs with raw-inline, hostclio
//       and clio-sync).
double RunHdf5Arm(const Cfg& cfg_in, uint64_t* checksum) {
    // Local copy: PREFAULT IMPLIES PRECREATE for this arm (Hdf5Sink::PrefaultAll explains
    // why — the lazy H5Dcreate in WriteSnap would collide with the prefault pass). Hdf5Sink /
    // Hdf5Writer hold a Cfg&, so this copy must outlive them: it does, it is this frame.
    Cfg cfg = cfg_in;
    if (cfg.prefault) cfg.hdf5_precreate = 1;

    EnsureDir(cfg.hdf5_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    const std::string path = cfg.hdf5_dir + "/checkpoint.h5";
    Masker masker(cfg.N, cfg.incompressible != 0);
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);

    const char* mode = cfg.hdf5_naive ? (cfg.hdf5_naive_chunked ? "NAIVE (chunked, untuned)"
                                                                : "NAIVE (contiguous, untuned)")
                     : cfg.hdf5_stock ? "STOCK (untuned)" : "tuned";
    std::fprintf(stderr,
        "[hdf5] %s: vfd=%s rdcc=%uMB early_alloc=%u direct_chunk=%u precreate=%u "
        "metablk=%uKB fsync=%u chunk=%ux%u\n",
        mode, cfg.hdf5_vfd.c_str(), cfg.hdf5_rdcc_mb, cfg.hdf5_early_alloc,
        cfg.hdf5_direct_chunk, cfg.hdf5_precreate, cfg.hdf5_meta_block_kb,
        cfg.raw_fsync, cfg.N / cfg.chunks, cfg.N);

    if (cfg.raw_inline) {
        Hdf5Sink sink(cfg);
        sink.Open(path);
        sink.PrecreateAll();      // outside the timed region (parity — see Cfg::hdf5_precreate)
        uint8_t* buf = Hdf5AllocBuf(cfg, gbytes);
        g_io_buf_bytes = gbytes;   // single inline staging buffer
        if (cfg.prefault) {       // also outside the timed region (Cfg::prefault)
            std::memset(buf, 0, gbytes);
            sink.PrefaultAll(buf);
        }
        Grids g = MakeGrids(cfg.N);
        double write_ms = 0;
        auto snap = [&](unsigned si, float* v_curr) {
            d2h.Copy(buf, masker.Apply(v_curr));
            auto a = std::chrono::steady_clock::now();      // GPU idle during this write
            sink.WriteSnap(si, buf);
            write_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - a).count();
        };
        auto finalize = [&]() { sink.Close(); };
        double ms = RunSim(cfg, g, snap, finalize, nullptr);
        FreeGrids(g);
        d2h.Report("hdf5_inline");
        Hdf5FreeBuf(cfg, buf);
        std::fprintf(stderr, "[hdf5] total=%.1f ms  write=%.1f ms  (inline, GPU idle)\n",
                     ms, write_ms);
        if (cfg.read_pdf) {
            Pdf pdf;
            const double rms = Hdf5ReadPdf(path, cfg, gbytes, pdf);
            PrintReadResult("hdf5_inline", cfg, rms, pdf, "host");
        }
        *checksum = Hdf5ReadbackChecksum(path, cfg, gbytes);
        return ms;
    }

    g_io_buf_bytes = uint64_t(cfg.hdf5_nbuf) * gbytes;   // nbuf pinned staging buffers
    Hdf5Writer writer(cfg, path, gbytes, /*nbuf=*/cfg.hdf5_nbuf);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        int idx;
        uint8_t* buf = writer.Acquire(&idx);
        d2h.Copy(buf, masker.Apply(v_curr));
        writer.Submit(idx, si);
    };
    double writer_ms = 0;
    auto finalize = [&]() { writer.Finish(&writer_ms); };   // drain inside timed region
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("hdf5_threaded");
    std::fprintf(stderr,
        "[hdf5] total=%.1f ms  writer-busy=%.1f ms (overlapped with compute)  nbuf=%u\n",
        ms, writer_ms, cfg.hdf5_nbuf);
    if (cfg.read_pdf) {
        Pdf pdf;
        const double rms = Hdf5ReadPdf(path, cfg, gbytes, pdf);
        PrintReadResult("hdf5_threaded", cfg, rms, pdf, "host");
    }
    *checksum = Hdf5ReadbackChecksum(path, cfg, gbytes);
    return ms;
}

// Arm `hdf5_async`: the same conventional path, but through the HDF5 Asynchronous I/O VOL
// connector (Tang et al., TPDS 2021). This is the strongest baseline available, and the one
// a reviewer will ask for, because it attacks the SAME compute/IO overlap we do — only from
// the host side, with an Argobots thread draining an HDF5 event set, rather than from the
// device.
//
// Structure mirrors the CLIO async arm exactly: fire every snapshot's create+write+close
// onto the event set as it is produced, then drain once at the end inside the timed region.
//
// Buffers: by default one per snapshot, NOT a recycled pool. The connector reads the buffer
// later, so reusing one would race the next D2H against an in-flight write. That costs
// snaps * grid_bytes of pinned host memory (1875 MB at defaults) — the direct analogue of the
// CLIO async arm, which likewise holds a device backend per chunk per snapshot.
// GSBENCH_HDF5_ASYNC_POOL=<M> opts into a bounded M-buffer pool instead; see the comment on
// `pool` below for the reuse contract and why it needs a full drain.
//
// Durability: drain + flush + fdatasync ONCE at the end, rather than per snapshot. That
// matches CLIO-async's drain-at-end semantics; per-snapshot fsync would serialise exactly
// the overlap this arm exists to exploit. It is therefore a WEAKER durability guarantee than
// the `hdf5` and `raw` arms, and must be compared against `async`, not against `hdf5`.
//
// Requires the connector to actually be loaded at run time (HDF5_VOL_CONNECTOR=async,
// HDF5_PLUGIN_PATH, and a thread-safe libhdf5). Without it the _async calls silently run
// synchronously, so we ABORT rather than quietly report a mislabelled inline arm as async.
double RunHdf5AsyncArm(const Cfg& cfg_in, uint64_t* checksum) {
    // Prefault implies precreate here too (see RunHdf5Arm / Hdf5Sink::PrefaultAll). The zero
    // pass is a SYNCHRONOUS H5Dwrite to every dataset before the async phase begins, so no
    // async op is in flight while it runs; WriteSnapAsync then writes into the open handles.
    Cfg cfg = cfg_in;
    if (cfg.prefault) cfg.hdf5_precreate = 1;

    EnsureDir(cfg.hdf5_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    const std::string path = cfg.hdf5_dir + "/checkpoint_async.h5";
    Masker masker(cfg.N, cfg.incompressible != 0);

    Hdf5Sink sink(cfg);
    sink.Open(path);
    if (!sink.AsyncVolActive()) {
        sink.Close();
        throw std::runtime_error(
            "hdf5_async: the async VOL connector is NOT loaded (H5VLget_connector_name != "
            "\"async\"). Set HDF5_VOL_CONNECTOR=\"async under_vol=0;under_info={}\", "
            "HDF5_PLUGIN_PATH=<vol-async lib>, and LD_LIBRARY_PATH to a THREAD-SAFE libhdf5 "
            "+ Argobots. Refusing to report a synchronous run as async.");
    }
    std::fprintf(stderr, "[hdf5_async] async VOL connector CONFIRMED loaded\n");

    // GSBENCH_HDF5_ASYNC_POOL=<M>: bound the staging footprint to M buffers, snapshot si
    // writing into slot si % M — the async_VOL analogue of GPUH5's `k mod g` buffer groups, so
    // the bounded-memory claim can be tested on the baseline instead of only asserted about it.
    // 0 (the default, EnvU0 so 0 means 0) keeps the unbounded one-buffer-per-snapshot path
    // BYTE-FOR-BYTE: with nbuf == cfg.snaps the round-robin index si % nbuf collapses to si and
    // the reuse guard `si >= nbuf` is never true, so nothing below executes differently.
    //
    // REUSE CONTRACT (the whole reason this needs care). We link /opt/vol-async-nomemcpy
    // (ENABLE_WRITE_MEMCPY=OFF, because the stock memcpy-on build livelocks in Argobots at
    // scale), so the connector does ZERO internal copying: the pointer handed to H5Dwrite_async
    // is the buffer the Argobots thread reads later. Overwriting slot si % M with snapshot si's
    // D2H before snapshot si-M's write has finished is therefore a silent data race — no crash,
    // no H5 error, just wrong bytes on disk, caught only by Hdf5ReadbackChecksum at the very
    // end. So every reuse MUST be preceded by a drain that provably covers si-M's write.
    //
    // DESIGN CHOICE: one shared event set + a FULL drain before each reuse, NOT M per-slot
    // event sets. Per-slot event sets would be the tighter analogue of GPUH5's per-group
    // completion flag (reusing slot k would wait only on slot k, leaving the other M-1 writes
    // overlapping), but they would buy nothing here: the drain path's fast route is
    // Hdf5Sink::FileWait(), which is H5VLfile_optional_op(file_, "gov.lbl.async.file.wait") —
    // a FILE-scoped wait. It takes the file, not an event set, so it blocks until every
    // outstanding task on the file is done no matter which event set tracked it. With
    // GSBENCH_HDF5_ASYNC_FWAIT on (the default, and worth 3.4x, see below) any per-slot wait
    // would degenerate into a file-wide barrier anyway; splitting the event set would only
    // manufacture the APPEARANCE of partial overlap. Honest full barrier instead.
    //
    // (GSBENCH_HDF5_ASYNC_FWAIT=0 IS event-set scoped — it drains with H5ESwait alone — so
    // per-slot sets would be genuinely granular there. It is still not worth building: that
    // path pays H5ESwait's whole-second-quantised ~2 s mutex backoff on every wait that finds
    // work outstanding, so making the wait per-reuse costs ~2 s x (snaps - M) no matter how
    // finely it is scoped. WARNING for whoever runs this: gsbench_driver's registry pins
    // async_VOL to FWAIT=0, so a pooled campaign run inherits exactly that tax — set
    // GSBENCH_HDF5_ASYNC_FWAIT=1 when pooling, on an otherwise idle box.)
    const unsigned pool = EnvU0("GSBENCH_HDF5_ASYNC_POOL", 0);
    // M >= snaps is pooling that never wraps, i.e. exactly the unpooled arm; clamp so we do not
    // allocate (and report) buffers no snapshot will ever touch.
    const unsigned nbuf = pool ? std::min(pool, cfg.snaps) : cfg.snaps;
    // One pinned buffer per slot (see the buffer contract above).
    std::vector<uint8_t*> bufs(nbuf, nullptr);
    for (unsigned s = 0; s < nbuf; ++s) bufs[s] = Hdf5AllocBuf(cfg, gbytes);
    // Deterministic footprint: our nbuf staging buffers -- unbounded (== snaps, like
    // gpuh5_noreuse) unpooled, bounded at M when pooling. The async VOL adds small internal
    // copies on top -- see the sampled host RSS for the total.
    g_io_buf_bytes = uint64_t(nbuf) * gbytes;

    sink.PrecreateAll();                 // outside the timed region
    if (cfg.prefault) {                  // ditto — synchronous zero pass (Cfg::prefault)
        std::memset(bufs[0], 0, gbytes);
        sink.PrefaultAll(bufs[0]);
    }

    hid_t es = H5EScreate();
    H5CHK(es);

    // Instrumentation (GSBENCH_HDF5_ASYNC_TRACE=1): where does the time actually go?
    // d2h  = the cudaMemcpy per snapshot,  fire = the _async issue calls (should be ~0 if the
    // connector really is deferring), wait = the tail H5ESwait drain, sync = final fdatasync.
    const bool trace_async = EnvU0("GSBENCH_HDF5_ASYNC_TRACE", 0) != 0;
    // Drain the event set after every snapshot instead of once at the end. Bounds the
    // connector's memory growth and lets each write overlap the NEXT snapshot's compute.
    const unsigned drain_every = EnvU0("GSBENCH_HDF5_ASYNC_DRAIN_EVERY", 0);
    // Drain through the connector's own H5Fwait before H5ESwait (see Hdf5Sink::FileWait).
    // ON by default because it is worth 3.4x on this arm and nothing to any other arm:
    // 2305 ms -> 679 ms at N=6400/snaps=12, by not paying H5ESwait's 2 s of hard-coded
    // mutex backoff. Set GSBENCH_HDF5_ASYNC_FWAIT=0 to drain with H5ESwait alone.
    //
    // CAVEAT, and the reason for the escape hatch: H5VL_async_file_wait re-acquires the
    // HDF5 global mutex by BUSY-SPINNING with no backoff whatsoever, so if the machine is
    // CPU-oversubscribed it spins against the Argobots thread it is waiting on. Measured:
    // a run that takes 0.68 s on an idle box had not finished after 10 MINUTES with 64
    // spinners on 32 cores. The benchmark itself is single-threaded + one Argobots ES, so
    // it never provokes this — but do not run the sweep next to a busy build.
    const bool use_fwait = EnvU0("GSBENCH_HDF5_ASYNC_FWAIT", 1) != 0;
    double t_d2h = 0, t_fire = 0, t_wait = 0, t_sync = 0;
    using Clk = std::chrono::steady_clock;
    auto secs = [](Clk::time_point a, Clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto drain = [&]() {
        if (use_fwait) sink.FileWait();   // no-op + fall through to H5ESwait if unsupported
        size_t n_in_progress = 0;
        hbool_t err_occurred = 0;
        H5CHK(H5ESwait(es, H5ES_WAIT_FOREVER, &n_in_progress, &err_occurred));
        if (err_occurred)
            throw std::runtime_error("hdf5_async: an async HDF5 operation failed");
    };

    // t_d2h below is the ORIGINAL, CONFLATED figure: a host clock around a cudaMemcpy that
    // is queued behind steps_per un-drained GsStepKernels, so it charges the snapshot's
    // compute tail to the copy (see D2HTrace). It is kept, and still printed, so the clean
    // event-measured copy can be compared against it directly rather than silently replacing
    // it. GSBENCH_D2H_TRACE=1 prints the clean one; they should differ by ~compute/snapshot.
    D2HTrace d2h(EnvU0("GSBENCH_D2H_TRACE", 0) != 0, cfg.snaps, gbytes);
    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        // MANDATORY pre-reuse drain (pooled only; never taken when nbuf == cfg.snaps, so the
        // unpooled path does not even read the clock here). Slot si % nbuf currently holds
        // snapshot si - nbuf's payload, which the connector may still be reading, and the D2H
        // below overwrites it. Charged to t_wait: it is the same H5ESwait/H5Fwait work
        // drain_every does, just made compulsory, so the printed breakdown stays additive and
        // ms - t_d2h - t_fire - t_wait - t_sync still approximates compute.
        if (si >= nbuf) {
            auto w0 = Clk::now();
            drain();
            t_wait += secs(w0, Clk::now());
        }
        // Same D2H as raw/hdf5/hostclio, into THIS snapshot's slot.
        uint8_t* buf = bufs[si % nbuf];
        auto a = Clk::now();
        d2h.Copy(buf, masker.Apply(v_curr));
        auto b = Clk::now();
        sink.WriteSnapAsync(si, buf, es);        // returns immediately; VOL drains it
        auto c = Clk::now();
        if (drain_every && (si + 1) % drain_every == 0) drain();
        auto d = Clk::now();
        t_d2h += secs(a, b);
        t_fire += secs(b, c);
        t_wait += secs(c, d);
    };
    auto finalize = [&]() {                      // tail-drain, inside the timed region
        auto a = Clk::now();
        drain();
        auto b = Clk::now();
        if (cfg.raw_fsync) sink.Sync();
        auto c = Clk::now();
        t_wait += secs(a, b);
        t_sync += secs(b, c);
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    d2h.Report("hdf5_async");
    if (trace_async)
        std::fprintf(stderr,
                     "[hdf5_async] BREAKDOWN d2h=%.1f(CONFLATED: includes compute tail) "
                     "fire=%.1f wait=%.1f sync=%.1f other(compute)=%.1f ms\n",
                     t_d2h, t_fire, t_wait, t_sync,
                     ms - t_d2h - t_fire - t_wait - t_sync);

    H5CHK(H5ESclose(es));
    sink.Close();
    for (auto* b : bufs) Hdf5FreeBuf(cfg, b);
    // The GSBENCH_RESULT arm name stays "async_VOL" whether or not pooling is on. Renaming it
    // (async_VOL_poolM) would break two things: PrintResult decides the durable= source with an
    // EXACT strcmp(arm, "async_VOL"), so a suffixed name would be misclassified as a CLIO arm
    // and report the wrong durability; and gsbench_driver aggregates on its registry name, so
    // the parsed arm= would no longer agree with the row it lands in. Pooled and unpooled runs
    // stay distinguishable without touching the line format: GSBENCH_RESULT already carries
    // io_buf_mb, which is nbuf * grid_bytes (M vs snaps). This stderr note names M outright.
    std::fprintf(stderr,
                 "[hdf5_async] total=%.1f ms  (event-set fire-all + drain-at-end, "
                 "bufs=%u%s)\n",
                 ms, nbuf, pool ? " POOLED: full drain before every slot reuse" : "");
    if (cfg.read_pdf) {
        Pdf pdf;
        const double rms = Hdf5ReadPdf(path, cfg, gbytes, pdf);
        PrintReadResult("async_VOL", cfg, rms, pdf, "host");
    }
    *checksum = Hdf5ReadbackChecksum(path, cfg, gbytes);
    return ms;
}

#endif  // GSBENCH_HAVE_HDF5

}  // namespace

// ---- kernel register-usage report (no simulation, no CLIO server) ----------
// numRegs/sharedSizeBytes/localSizeBytes are fixed by the compiled binary, not by
// which arm actually runs, so one process can report every kernel used across all
// arms in a single shot. localSizeBytes > 0 means the compiler spilled registers to
// local memory — a genuine register-pressure signal, not just informational.
TEST_CASE("GSBENCH kernel register report",
          "[.][gpu][gsbench][gsbench_kernel_regs]") {
    struct KernelInfo {
        const char* name;
        const char* arm;
        const void* fn;
    };
    const KernelInfo kernels[] = {
        {"GsStepKernel", "all", reinterpret_cast<const void*>(GsStepKernel)},
        {"MaskKernel", "all", reinterpret_cast<const void*>(MaskKernel)},
        {"TwCopyKernel", "clio-sync,clio-async",
         reinterpret_cast<const void*>(TwCopyKernel)},
        {"TwSnapSyncKernel", "clio-sync",
         reinterpret_cast<const void*>(TwSnapSyncKernel<false>)},
        {"TwSnapFireKernel", "clio-async",
         reinterpret_cast<const void*>(TwSnapFireKernel<false>)},
        {"TwDrainKernel", "clio-async",
         reinterpret_cast<const void*>(TwDrainKernel)},
        {"GsPersistentKernel<true>", "gpuh5",
         reinterpret_cast<const void*>(GsPersistentKernel<true>)},
        {"GsPersistentKernel<false>", "gpuh5_sync",
         reinterpret_cast<const void*>(GsPersistentKernel<false>)},
        {"GsPersistentComputeOnlyKernel", "gpuh5-decomposition-probe (never launched)",
         reinterpret_cast<const void*>(GsPersistentComputeOnlyKernel)},
        {"ReuseFireKernel", "reuse",
         reinterpret_cast<const void*>(ReuseFireKernel)},
        {"TwSnapPooledKernel<false>", "pooled",
         reinterpret_cast<const void*>(TwSnapPooledKernel<false>)},
    };
    const unsigned kNumKernels = sizeof(kernels) / sizeof(kernels[0]);
    int regs[kNumKernels];
    for (unsigned i = 0; i < kNumKernels; ++i) {
        cudaFuncAttributes attr{};
        REQUIRE(cudaFuncGetAttributes(&attr, kernels[i].fn) == cudaSuccess);
        regs[i] = attr.numRegs;
        std::fprintf(stderr,
            "GSBENCH_KERNEL_REGS kernel=%s arm=%s regs=%d smem=%zu lmem=%zu "
            "maxThreadsPerBlock=%d\n",
            kernels[i].name, kernels[i].arm, attr.numRegs, attr.sharedSizeBytes,
            attr.localSizeBytes, attr.maxThreadsPerBlock);
    }

    // ---- theoretical occupancy: persistent kernel vs the stock stencil, @256 threads/block,
    // sm_89 (max 1536 threads/SM -> occupancy% = blocks_per_sm * 256 / 1536). ----
    {
        int blocks = 0;
        REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, GsStepKernel, 256, 0) == cudaSuccess);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_OCCUPANCY kernel=GsStepKernel blockDim=256 blocks_per_sm=%d "
            "theoretical_occupancy_pct=%.1f\n",
            blocks, 100.0 * double(blocks) * 256.0 / 1536.0);

        blocks = 0;
        REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, GsPersistentKernel<true>, 256, 0) == cudaSuccess);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_OCCUPANCY kernel=GsPersistentKernel<true> blockDim=256 "
            "blocks_per_sm=%d theoretical_occupancy_pct=%.1f\n",
            blocks, 100.0 * double(blocks) * 256.0 / 1536.0);

        blocks = 0;
        REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, GsPersistentComputeOnlyKernel, 256, 0) == cudaSuccess);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_OCCUPANCY kernel=GsPersistentComputeOnlyKernel blockDim=256 "
            "blocks_per_sm=%d theoretical_occupancy_pct=%.1f\n",
            blocks, 100.0 * double(blocks) * 256.0 / 1536.0);
    }
    // indices into `kernels`/`regs`: 0=GsStepKernel 1=MaskKernel(excluded, shared test
    // artifact) 2=TwCopyKernel 3=TwSnapSyncKernel 4=TwSnapFireKernel 5=TwDrainKernel
    const int kGsStep = regs[0];
    struct ArmTotal {
        const char* arm;
        int total;
    };
    const ArmTotal arm_totals[] = {
        {"raw/hdf5/hdf5-async/hostclio", kGsStep},
        {"clio-sync", kGsStep + regs[2] + regs[3]},
        {"clio-async", kGsStep + regs[2] + regs[4] + regs[5]},
    };
    const int baseline = arm_totals[0].total;
    for (const auto& a : arm_totals) {
        const double pct_increase = 100.0 * double(a.total - baseline) / double(baseline);
        std::fprintf(stderr,
            "GSBENCH_KERNEL_REGS_ARM_TOTAL arm=\"%s\" total_regs=%d vs_baseline=%+.1f%%\n",
            a.arm, a.total, pct_increase);
    }
}

// ---- the three arm TEST_CASEs (run each in its OWN process) ----------------

// raw arm needs no CLIO server.
TEST_CASE("GSBENCH raw disk (no CLIO)", "[.][integration][gpu][gsbench][gsbench_raw]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunRawArm(cfg, &checksum);
    PrintResult(cfg.raw_inline ? "raw_inline" : "raw_threaded", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// Tag alias [gsbench_sync] kept for back-compat with the older sweep/campaign scripts
// (run_bench_sweep.sh, run_submit_probe_sweep.sh, scaling_campaign/run_campaign.sh); the
// canonical name is gpuh5_sync.
TEST_CASE("GSBENCH gpuh5 sync",
          "[.][integration][gpu][gsbench][gsbench_gpuh5_sync][gsbench_sync]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, ClioMode::kSync, "results/gsbench/gpuh5_sync", &checksum);
    PrintResult("gpuh5_sync_relaunch", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// Host-driven CLIO (raw's flow, CLIO sink): isolates the storage path vs raw, and the
// submission model vs the GPU-producer sync/async arms.
TEST_CASE("GSBENCH clio host-driven", "[.][integration][gpu][gsbench][gsbench_hostclio]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunHostClioArm(cfg, "results/gsbench/hostclio", &checksum);
    PrintResult("hostclio", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// The conventional path (GPU -> D2H -> HDF5), and the baseline the paper is measured
// against. Needs no CLIO server, like raw. Compiled only when HDF5 was found.
#if GSBENCH_HAVE_HDF5
// The row is named for the writer STRUCTURE, not the sink, so that a sweep which runs this
// arm twice (GSBENCH_RAW_INLINE=1 then =0) emits two distinguishable rows:
//   hdf5_inline   — synchronous H5Dwrite; the GPU idles during it. Matched semantics vs
//                   clio-sync and hostclio (the producer blocks on I/O).
//   hdf5_threaded — background writer thread; the write overlaps the next sim steps. Matched
//                   semantics vs clio-async. THIS is the baseline the paper must beat.
TEST_CASE("GSBENCH hdf5", "[.][integration][gpu][gsbench][gsbench_hdf5]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunHdf5Arm(cfg, &checksum);
    PrintResult(cfg.raw_inline ? "hdf5_inline" : "hdf5_threaded", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// The typical-user, unoptimized HDF5 baseline: contiguous layout by default (or chunked via
// GSBENCH_HDF5_NAIVE_CHUNKED), high-level H5Dwrite of the whole array, default alloc/cache/fill.
// Structurally identical to hdf5_inline (synchronous,
// GPU idle during the write, same per-snapshot durability) — only the dataset config differs,
// so it always runs the inline structure regardless of GSBENCH_RAW_INLINE. Reuses Hdf5Sink and
// RunHdf5Arm via the hdf5_naive flag. Its checksum MUST equal the other HDF5 arms' at the same
// config (same logical float data). Expected to land near/below raw_inline.
TEST_CASE("GSBENCH hdf5 naive", "[.][integration][gpu][gsbench][gsbench_hdf5_naive]") {
    Cfg cfg;
    cfg.hdf5_naive = 1;
    cfg.raw_inline = 1;   // synchronous/GPU-idle structure, matching hdf5_inline
    uint64_t checksum = 0;
    double ms = RunHdf5Arm(cfg, &checksum);
    PrintResult("hdf5_naive", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// The HDF5 Asynchronous I/O VOL arm. Needs the connector loaded at run time (it aborts
// rather than silently degrade to synchronous — see RunHdf5AsyncArm), so it is NOT part of
// the default runner sweep; run_threeway_bench.sh includes it only when GSBENCH_HDF5_ASYNC=1
// and the async-VOL env is set up.
TEST_CASE("GSBENCH hdf5 async-VOL",
          "[.][integration][gpu][gsbench][gsbench_hdf5_async]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunHdf5AsyncArm(cfg, &checksum);
    PrintResult("async_VOL", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}
#endif

// Tag alias [gsbench_async] kept for back-compat (canonical name gpuh5_noreuse).
TEST_CASE("GSBENCH gpuh5 noreuse",
          "[.][integration][gpu][gsbench][gsbench_gpuh5_noreuse][gsbench_async]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, ClioMode::kAsync, "results/gsbench/gpuh5_noreuse", &checksum);
    PrintResult("gpuh5_noreuse", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// Bounded-pool double buffering: ONE fused fill+fire kernel per snapshot, N chunks streamed
// through M = GSBENCH_POOL resident buffers by G = GSBENCH_SUBMIT_BLOCKS blocks at pipeline
// depth D = M/G. GSBENCH_POOL=0 (default) => M == N: the CONTROL run, which fuses fill+fire
// but reuses no buffer, so it separates the cost of FUSING from the cost of POOLING.
//
// Its checksum MUST equal sync's and async's — same computation, same persisted bytes. A
// difference is a buffer-reuse race, not a tuning artifact.
TEST_CASE("GSBENCH clio pooled", "[.][integration][gpu][gsbench][gsbench_pooled]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, ClioMode::kPooled, "results/gsbench/pooled", &checksum);
    PrintResult("pooled", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// REUSE arm (DESIGN §7 "Option B", relaunched): async's shape but memory constant in
// snapshots — 2 reused buffer groups (snap % 2) with drain-before-refill and an in-kernel
// device tag-stamp, so each snapshot still lands in its own dataset. GPU-initiated,
// bounded-both-sides memory, O(1) device state; NOT fused/cooperative (that is Option A).
// Its checksum MUST equal sync's/async's — same computation, same persisted bytes.
// Tag alias [gsbench_reuse] kept for back-compat (canonical name gpuh5 — the DEFAULT design).
TEST_CASE("GSBENCH gpuh5 (reuse; default)",
          "[.][integration][gpu][gsbench][gsbench_gpuh5][gsbench_reuse]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunReuseArm(cfg, "results/gsbench/gpuh5", &checksum);
    PrintResult("gpuh5_relaunch", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// PERSISTENT arm (DESIGN §7 "Option A"): the whole snapshot loop in ONE resident cooperative
// kernel (grid.sync() between timesteps), fired against the relaunched reuse/async arms. Same
// bounded 2-group memory + device self-addressing; forces kPinnedHost data (deadlock safety).
// Checksum MUST equal sync's/async's/reuse's.
TEST_CASE("GSBENCH clio persistent", "[.][integration][gpu][gsbench][gsbench_persistent]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunPersistentArm(cfg, "results/gsbench/persistent", &checksum);
    PrintResult("gpuh5", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// persistent_sync: the resident cooperative kernel but with SYNCHRONOUS submit (fire-AND-wait
// each snapshot in-kernel, no double buffering / no overlap) -- the persistent analog of
// gpuh5_sync. Isolates the resident-vs-relaunch axis for the synchronous submission model.
TEST_CASE("GSBENCH clio persistent_sync",
          "[.][integration][gpu][gsbench][gsbench_persistent_sync]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunPersistentArm(cfg, "results/gsbench/persistent_sync", &checksum,
                                 /*async_submit=*/false);
    PrintResult("gpuh5_sync", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

// persistent_floor: COMPUTE-ONLY isolation of GsPersistentKernel via
// GsPersistentComputeOnlyKernel -- see RunPersistentFloorArm. Direct measured compute floor
// for the gpuh5/gpuh5_sync bars in fig_write_decomp2.tex, replacing that figure's single
// GsStepKernel-only floor for those two bars specifically. MB/MBps/io_buf_mb in the printed
// GSBENCH_RESULT line are not meaningful here (no I/O happens); read ms from the
// GSBENCH_PERSISTENT_FLOOR line instead.
TEST_CASE("GSBENCH clio persistent floor (compute-only)",
          "[.][integration][gpu][gsbench][gsbench_persistent_floor]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    double ms = RunPersistentFloorArm(cfg);
    const double total_steps = double(cfg.snaps) * double(cfg.steps_per);
    std::fprintf(stderr,
        "GSBENCH_PERSISTENT_FLOOR arm=gpuh5_floor N=%u snaps=%u steps_per=%u total_steps=%.0f "
        "compute_ms=%.3f ms_per_step=%.6f\n",
        cfg.N, cfg.snaps, cfg.steps_per, total_steps, ms, ms / total_steps);
    PrintResult("gpuh5_floor", cfg, ms, /*checksum=*/0);
    REQUIRE(ms > 0.0);
}

#endif  // !CTP_IS_DEVICE_PASS

#else
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
