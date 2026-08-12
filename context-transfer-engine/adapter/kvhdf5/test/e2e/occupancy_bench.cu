/*
 * GPU-RESOURCE COST OF FUSED DEVICE-INITIATED I/O  (paper contribution C1)
 * ========================================================================
 *
 * The claim under test: the GPU-resource cost of device-initiated I/O is a
 * consequence of device-held STATE, not of initiation. kvhdf5 holds O(1) state
 * per block — independent of dataset size AND of the number of writes in flight
 * — so the I/O path can be fused into a compute kernel without meaningfully
 * reducing occupancy.
 *
 * WHY THREE KERNELS AND NOT TWO
 * -----------------------------
 * The obvious experiment ("stock Gray-Scott vs. our kernel") is INVALID: our
 * kernel differs from stock in *two* ways at once — it carries an I/O path, AND
 * it is restructured so one block owns one chunk (block-scoped __syncthreads is
 * the only barrier a single kernel has, so a chunk's producer must be a single
 * block). A two-way delta charges the register cost of BOTH changes to I/O.
 *
 *   GsStepPlain          stock Gray-Scott, no I/O, no kvhdf5 type in sight
 *   GsStepBlockPerChunk  one block owns one chunk, no I/O  <- isolates RESTRUCTURING
 *   GsStepFused          one block owns one chunk + WriteAsync/WriteWait
 *
 * THE REPORTED I/O COST IS  fused - blockPerChunk.  `fused - plain` conflates
 * restructuring with I/O and must never be reported as the I/O cost.
 *
 * The three share ONE __forceinline__ GsCell() so the arithmetic is identical by
 * construction; only the indexing and the sink differ. GsStepBlockPerChunk
 * mentions no kvhdf5 type at all and stores its chunk into a plain device
 * buffer, so EVERY GPUH5-attributable byte of device state (the by-value handle,
 * the ChunkDesc load, the __shared__ per-block IpcManager, Send, Future) lands on
 * the fused side of the subtraction. That is the conservative attribution.
 *
 * Note the snapshot store is a SUBSTITUTION, not an addition: a Gray-Scott step
 * has to write its output field somewhere, and fusing means it writes it into the
 * registered buffer instead of into a scratch buffer that would then be D2H-copied.
 * blockPerChunk therefore performs the identical store, into plain memory.
 *
 * IN-FLIGHT DEPTH (the state-invariance test, and the thing that separates us
 * from a device-side block driver)
 * ------------------------------------------------------------------------------
 * Register count is a COMPILE-TIME property, so "registers are equal at N=1 and
 * N=6400" is true by construction and proves nothing. The empirical content is
 * that a *single block* can hold D writes in flight for growing D without its
 * device state growing. GsStepFused grid-strides over chunks: with the grid held
 * FIXED and the chunk count varied, in-flight writes PER BLOCK = ceil(chunks/grid)
 * grows while the launch geometry stays constant — so an occupancy change cannot
 * be blamed on the grid. Depth-invariance holds here because SubmitAsync DISCARDS
 * the future and SubmitWait rebuilds it statelessly from the task slot.
 *
 * GsStepFusedStateful<D> is the COUNTERFACTUAL: the naive design that retains a
 * gpu::Future per in-flight write. It is given the best possible case — a fully
 * unrolled, compile-time depth, so the futures can live in registers rather than
 * being forced to local memory — and its device state is still O(D). It exists to
 * show that our flat curve is a property of the DESIGN, not of the measurement.
 * It is not part of the three-variant comparison.
 *
 * All cases are HIDDEN ([.]) — this is a measurement harness, not a correctness
 * test, and it is driven by ncu. Knobs are env vars so the depth sweep needs no
 * recompile:
 *   OCC_N        grid edge (NxN float32)                default 512
 *   OCC_CHUNKS   chunks in the snapshot dataset (<=64!) default 64
 *   OCC_GRID     CUDA blocks (held FIXED across sweep)  default 8
 *   OCC_THREADS  threads per block                      default 256
 *
 * CAUTION: the fire-all path is known to fail to drain above roughly 64 chunks
 * per dataset. OCC_CHUNKS stays at or below 64; if a run hangs, that is a REAL
 * result and must be reported, not worked around.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/layout.h>
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "cte_env.h"

using kvhdf5::byte_t;

// Gray-Scott coefficients. Deliberately NOT the anonymous-namespace `GsParams` of
// the neighbouring benches — this is its own type so the kernels below cannot be
// confused with theirs.
struct OccGsParams {
    float Du, Dv, F, k, dt;
};

// ---------------------------------------------------------------------------
// The shared arithmetic. Every variant calls exactly this, so any register delta
// between variants is attributable to indexing + I/O, never to the math.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void GsCell(const float* u, const float* v, unsigned gid,
                                       unsigned N, OccGsParams p, float& un_out,
                                       float& vn_out) {
    unsigned x = gid % N, y = gid / N;
    unsigned xm = (x == 0) ? (N - 1) : (x - 1);
    unsigned xp = (x == N - 1) ? 0u : (x + 1);
    unsigned ym = (y == 0) ? (N - 1) : (y - 1);
    unsigned yp = (y == N - 1) ? 0u : (y + 1);
    float uc = u[gid], vc = v[gid];
    float lap_u = u[y*N+xm] + u[y*N+xp] + u[ym*N+x] + u[yp*N+x] - 4.f*uc;
    float lap_v = v[y*N+xm] + v[y*N+xp] + v[ym*N+x] + v[yp*N+x] - 4.f*vc;
    float uvv = uc * vc * vc;
    un_out = uc + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - uc));
    vn_out = vc + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vc);
}

// ---- VARIANT 1a: stock Gray-Scott, flat gid. -------------------------------
// The literal textbook form (one thread per cell, early return). Present so the
// static table can show that the grid-stride form below costs nothing extra —
// i.e. that holding the launch geometry constant did not tax the baseline.
__global__ void GsStepPlainFlat(const float* u, const float* v, float* un, float* vn,
                                OccGsParams p, unsigned N) {
    unsigned cells = N * N;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= cells) return;
    float a, b;
    GsCell(u, v, gid, N, p, a, b);
    un[gid] = a;
    vn[gid] = b;
}

// ---- VARIANT 1: stock Gray-Scott, grid-stride. -----------------------------
// The BASELINE that gets profiled. Grid-stride (the canonical CUDA idiom) purely
// so it can be launched at the SAME geometry as the other two — occupancy is a
// function of the launch config, so comparing variants at different grids would
// be meaningless. When gridDim*blockDim >= cells this degenerates to exactly one
// trip, i.e. to GsStepPlainFlat above.
__global__ void GsStepPlain(const float* u, const float* v, float* un, float* vn,
                            OccGsParams p, unsigned N) {
    unsigned cells = N * N;
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; gid < cells;
         gid += stride) {
        float a, b;
        GsCell(u, v, gid, N, p, a, b);
        un[gid] = a;
        vn[gid] = b;
    }
}

// ---- VARIANT 2: one block owns one chunk. NO I/O. --------------------------
// Isolates the cost of RESTRUCTURING. Chunk c is a contiguous row-slice of the
// field; block b grid-strides over chunks {b, b+gridDim, ...}. The evolved v-field
// lands in `snap` — a plain device buffer at exactly the offset the registered
// buffer would sit at — so the store is byte-for-byte the same work the fused
// kernel does, minus the I/O. Not one kvhdf5 type appears in this kernel.
__global__ void GsStepBlockPerChunk(const float* u, const float* v, float* un,
                                    float* snap, OccGsParams p, unsigned N,
                                    uint32_t chunks) {
    const unsigned chunk_cells = (N * N) / chunks;
    for (uint32_t c = blockIdx.x; c < chunks; c += gridDim.x) {
        float* dst = snap + static_cast<uint64_t>(c) * chunk_cells;
        for (unsigned i = threadIdx.x; i < chunk_cells; i += blockDim.x) {
            unsigned gid = c * chunk_cells + i;
            float a, b;
            GsCell(u, v, gid, N, p, a, b);
            un[gid] = a;
            dst[i] = b;
        }
        __syncthreads();
    }
}

// ---- VARIANT 3: THE REAL THING. one block owns one chunk + fused I/O. ------
// Identical to VARIANT 2 except the chunk's output lands directly in that chunk's
// registered device buffer and the block then FIRES its PutBlob from inside the
// same launch. Fire-all, then drain-all: a block that owns k chunks holds k puts
// in flight, and holds NO per-put state while it does (WriteAsync discards the
// future; WriteWait reconstructs it from the task slot).
__global__ void GsStepFused(const float* u, const float* v, float* un,
                            kvhdf5::GpuDatasetHandle h, OccGsParams p, unsigned N,
                            uint32_t chunks) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const unsigned chunk_cells = (N * N) / chunks;
    for (uint32_t c = blockIdx.x; c < chunks; c += gridDim.x) {
        float* dst = reinterpret_cast<float*>(h.Data(c));
        for (unsigned i = threadIdx.x; i < chunk_cells; i += blockDim.x) {
            unsigned gid = c * chunk_cells + i;
            float a, b;
            GsCell(u, v, gid, N, p, a, b);
            un[gid] = a;
            dst[i] = b;
        }
        __threadfence_system();  // server's readback must see the staged bytes
        __syncthreads();
        h.WriteAsync(c);         // fire; retains nothing
        __syncthreads();
    }
    for (uint32_t c = blockIdx.x; c < chunks; c += gridDim.x) {
        h.WriteWait(c);          // drain; future rebuilt statelessly from the slot
        __syncthreads();
    }
}

// ---- COUNTERFACTUAL: the naive design that RETAINS a future per in-flight put.
// Not one of the three variants; it exists to give the depth-invariance result
// something to be measured *against*. Depth is a compile-time constant and the
// fire/drain loops are fully unrolled — the most favourable case possible for
// this design, since it lets the futures stay register-resident instead of being
// spilled to local memory by dynamic indexing. Device state is nonetheless O(D).
//
// !!! KNOWN BROKEN AT RUNTIME FOR kDepth >= 2 — DO NOT TRUST ITS EXECUTION !!!
// Measured 2026-07-13: depth=1 runs and drains fine; depth>=2 HANGS in the drain
// loop, with no profiler attached. Not diagnosed. SendIn() does return a
// well-formed gpu::Future (it stamps task_size_ and clears is_complete_ before
// enqueuing), so retaining it *ought* to be equivalent to GpuDatasetHandle's
// stateless rebuild-from-slot — but empirically it is not, and that gap is
// unexplained.
//
// This kernel is therefore VALID ONLY AS A COMPILE-TIME ARTIFACT: its ptxas /
// cuobjdump resource numbers (stack frame growing 32 B per unit of depth) are a
// sound measurement of what retaining a future per in-flight write COSTS IN
// STORAGE. They are not evidence that the naive design works, and no achieved-
// occupancy number should ever be quoted from it. The [.] tag keeps it out of the
// default suite; run it only with an explicit timeout.
template <int kDepth>
__global__ void GsStepFusedStateful(const float* u, const float* v, float* un,
                                    kvhdf5::GpuDatasetHandle h, OccGsParams p,
                                    unsigned N, uint32_t chunks) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    auto* ipc = clio::run::gpu::IpcManager::GetBlockIpcManager();
    const unsigned chunk_cells = (N * N) / chunks;
    const bool lead = (clio::run::gpu::IpcManager::GetGpuThreadId() == 0);

    clio::run::gpu::Future<kvhdf5::cte::PodPutBlobTask> futs[kDepth];  // O(D) state

#pragma unroll
    for (int j = 0; j < kDepth; ++j) {
        uint32_t c = blockIdx.x * kDepth + j;
        if (c >= chunks) break;
        float* dst = reinterpret_cast<float*>(h.Data(c));
        for (unsigned i = threadIdx.x; i < chunk_cells; i += blockDim.x) {
            unsigned gid = c * chunk_cells + i;
            float a, b;
            GsCell(u, v, gid, N, p, a, b);
            un[gid] = a;
            dst[i] = b;
        }
        __threadfence_system();
        __syncthreads();
        if (lead) futs[j] = ipc->Send(h.chunks_[c].put_fp);  // RETAINED
        __syncthreads();
    }
    if (lead) {
#pragma unroll
        for (int j = 0; j < kDepth; ++j) {
            uint32_t c = blockIdx.x * kDepth + j;
            if (c >= chunks) break;
            futs[j].Wait();
        }
    }
}

// Force codegen for each depth so ptxas/cuobjdump report all of them.
template __global__ void GsStepFusedStateful<1>(const float*, const float*, float*,
                                                kvhdf5::GpuDatasetHandle, OccGsParams,
                                                unsigned, uint32_t);
template __global__ void GsStepFusedStateful<2>(const float*, const float*, float*,
                                                kvhdf5::GpuDatasetHandle, OccGsParams,
                                                unsigned, uint32_t);
template __global__ void GsStepFusedStateful<4>(const float*, const float*, float*,
                                                kvhdf5::GpuDatasetHandle, OccGsParams,
                                                unsigned, uint32_t);
template __global__ void GsStepFusedStateful<8>(const float*, const float*, float*,
                                                kvhdf5::GpuDatasetHandle, OccGsParams,
                                                unsigned, uint32_t);
template __global__ void GsStepFusedStateful<16>(const float*, const float*, float*,
                                                 kvhdf5::GpuDatasetHandle, OccGsParams,
                                                 unsigned, uint32_t);
template __global__ void GsStepFusedStateful<32>(const float*, const float*, float*,
                                                 kvhdf5::GpuDatasetHandle, OccGsParams,
                                                 unsigned, uint32_t);
template __global__ void GsStepFusedStateful<64>(const float*, const float*, float*,
                                                 kvhdf5::GpuDatasetHandle, OccGsParams,
                                                 unsigned, uint32_t);

// ===========================================================================
// REGISTER-PRESSURE SWEEP — generalizing C1 beyond Gray-Scott.
//
// The three-variant result says the I/O path costs +3 registers and 0 pp of
// occupancy. But Gray-Scott's compute kernel needs only 37 registers, and on
// sm_89 the 100%-occupancy cliff sits at 40 — so it had *exactly* 3 registers of
// headroom and the I/O path consumed exactly those 3. "Free" was therefore a
// property of THIS host kernel, not a law, and a reviewer will ask what happens
// to a register-hungry one.
//
// So: a synthetic compute kernel whose register pressure is a COMPILE-TIME knob
// (kLive independent live accumulators, fully unrolled so they stay in registers),
// instantiated across a wide range, in TWO variants that differ ONLY by the I/O
// path — SynthControl (no I/O) and SynthFused (CLIO_GPU_INIT + WriteAsync/Wait).
// Everything else — structure, indexing, memory traffic, the store into the chunk
// buffer — is identical, exactly as GsStepBlockPerChunk is to GsStepFused.
//
// Sweeping kLive walks the host kernel across the register-allocation cliffs, and
// (regs_fused - regs_control, occ_fused - occ_control) at each point answers the
// general question: WHEN does fusing I/O cost occupancy, and when is it free?
//
// Measured with the CUDA occupancy API — a driver query against the compiled
// image. No launch, no server, no profiler. Register count and occupancy ceiling
// are compile-time properties, so this is the right (and exact, and deterministic)
// instrument; running these kernels would add nothing.
// ===========================================================================

// kLive independent accumulators, all live across the loop -> ~kLive registers of
// pressure on top of the kernel's fixed cost. Compile-time indices keep them in
// registers rather than local memory.
template <int kLive>
__device__ __forceinline__ float SynthCompute(const float* src, unsigned base,
                                              unsigned n, unsigned stride) {
    float acc[kLive];
#pragma unroll
    for (int j = 0; j < kLive; ++j)
        acc[j] = src[base + (unsigned(j) % n)];

    for (unsigned i = threadIdx.x; i < n; i += stride) {
        float x = src[base + i];
#pragma unroll
        for (int j = 0; j < kLive; ++j)
            acc[j] = fmaf(acc[j], 1.0000001f, x * float(j + 1));  // all kLive stay live
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < kLive; ++j) s += acc[j];
    return s;
}

// CONTROL: one block owns one chunk, NO I/O. Writes its chunk into a plain buffer.
// No GPUH5 type appears here — same contract as GsStepBlockPerChunk.
template <int kLive>
__global__ void SynthControl(const float* src, float* snap, unsigned N,
                             uint32_t chunks) {
    const unsigned chunk_cells = (N * N) / chunks;
    for (uint32_t c = blockIdx.x; c < chunks; c += gridDim.x) {
        const unsigned base = c * chunk_cells;
        float* dst = snap + base;
        float s = SynthCompute<kLive>(src, base, chunk_cells, blockDim.x);
        for (unsigned i = threadIdx.x; i < chunk_cells; i += blockDim.x)
            dst[i] = s + float(i);
        __syncthreads();
    }
}

// FUSED: identical, except the chunk lands in its registered buffer and the block
// fires + drains its PutBlob from inside the same launch.
template <int kLive>
__global__ void SynthFused(const float* src, kvhdf5::GpuDatasetHandle h, unsigned N,
                           uint32_t chunks) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const unsigned chunk_cells = (N * N) / chunks;
    for (uint32_t c = blockIdx.x; c < chunks; c += gridDim.x) {
        const unsigned base = c * chunk_cells;
        float* dst = reinterpret_cast<float*>(h.Data(c));
        float s = SynthCompute<kLive>(src, base, chunk_cells, blockDim.x);
        for (unsigned i = threadIdx.x; i < chunk_cells; i += blockDim.x)
            dst[i] = s + float(i);
        __threadfence_system();
        __syncthreads();
        h.WriteAsync(c);
        __syncthreads();
    }
    for (uint32_t c = blockIdx.x; c < chunks; c += gridDim.x) {
        h.WriteWait(c);
        __syncthreads();
    }
}

#define OCC_SYNTH_INST(R)                                                        \
    template __global__ void SynthControl<R>(const float*, float*, unsigned,      \
                                             uint32_t);                           \
    template __global__ void SynthFused<R>(const float*, kvhdf5::GpuDatasetHandle,\
                                           unsigned, uint32_t)
OCC_SYNTH_INST(2);
OCC_SYNTH_INST(4);
OCC_SYNTH_INST(6);
OCC_SYNTH_INST(8);
OCC_SYNTH_INST(12);
OCC_SYNTH_INST(16);
OCC_SYNTH_INST(20);
OCC_SYNTH_INST(24);
OCC_SYNTH_INST(28);
OCC_SYNTH_INST(32);
OCC_SYNTH_INST(40);
OCC_SYNTH_INST(48);
OCC_SYNTH_INST(56);
OCC_SYNTH_INST(64);
OCC_SYNTH_INST(80);
OCC_SYNTH_INST(96);

#if !CTP_IS_DEVICE_PASS

namespace {

constexpr OccGsParams kGs{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

unsigned EnvU(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x > 0 ? static_cast<unsigned>(x) : dflt;
}

struct OccCfg {
    unsigned N       = EnvU("OCC_N", 512);
    unsigned chunks  = EnvU("OCC_CHUNKS", 64);
    unsigned grid    = EnvU("OCC_GRID", 8);
    unsigned threads = EnvU("OCC_THREADS", 256);
    // OCC_PINNED=1 places the blob DATA backend in pinned host memory instead of
    // device memory. This exists to make the kernel PROFILABLE under ncu, and it
    // does not change what is being measured: the kernel image is byte-identical
    // either way (same registers, same handle, same Write path) — only where the
    // payload lands differs.
    //
    // Why it matters: with a kDeviceMem data backend, the CPU-side server worker
    // must cudaMemcpy the payload out of HBM to service the put (see
    // gpu_ipc_manager.h: "direct read for kPinnedHost; cudaMemcpy for kDeviceMem").
    // Our fused kernel spins ON THE DEVICE waiting for that server to complete —
    // so the server must issue a CUDA call while the kernel is still resident.
    // ncu serializes the CUDA API during profiling and blocks that concurrent
    // call: kernel waits on server, server waits on the CUDA API, ncu holds it.
    // Deadlock. kPinnedHost removes the server's CUDA call entirely.
    unsigned pinned  = EnvU("OCC_PINNED", 0);
};

kvhdf5::GpuCteDataset::MemKind DataKind(const OccCfg& c) {
    return c.pinned ? kvhdf5::GpuCteDataset::MemKind::kPinnedHost
                    : kvhdf5::GpuCteDataset::MemKind::kDeviceMem;
}

// Seeded u/v fields + a plain snapshot sink for the no-I/O variants.
struct Fields {
    float *u = nullptr, *v = nullptr, *un = nullptr, *vn = nullptr, *snap = nullptr;
    unsigned cells = 0;
};

Fields MakeFields(unsigned N) {
    Fields f;
    f.cells = N * N;
    const size_t bytes = static_cast<size_t>(f.cells) * sizeof(float);
    REQUIRE(cudaMalloc(&f.u, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&f.v, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&f.un, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&f.vn, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&f.snap, bytes) == cudaSuccess);

    std::vector<float> u0(f.cells, 1.0f), v0(f.cells, 0.0f);
    for (unsigned y = N / 2 - 3; y < N / 2 + 3; ++y)
        for (unsigned x = N / 2 - 3; x < N / 2 + 3; ++x) v0[y * N + x] = 1.0f;
    ctp::GpuApi::Memcpy(f.u, u0.data(), bytes);
    ctp::GpuApi::Memcpy(f.v, v0.data(), bytes);
    return f;
}

void FreeFields(Fields& f) {
    cudaFree(f.u); cudaFree(f.v); cudaFree(f.un); cudaFree(f.vn); cudaFree(f.snap);
}

clio::cte::core::TagId MakeTag(const char* name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

void Check(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[occ] CUDA ERROR after %s: %s\n", what,
                     cudaGetErrorString(e));
    REQUIRE(e == cudaSuccess);
}

}  // namespace

// ---------------------------------------------------------------------------
// THEORETICAL OCCUPANCY via the CUDA occupancy API.
//
// This launches NOTHING. cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocks-
// PerMultiprocessor are driver queries against the compiled kernel image, so this
// case needs no CTE server, does no I/O, cannot hang, and cannot disturb a display
// attached to the same GPU. That matters here: ncu CANNOT profile the fused I/O
// kernels on a display-attached GPU (they spin-wait on a CPU round-trip, which
// trips the driver's launch watchdog — see the trace doc), so this is how the
// occupancy CEILING gets measured on such a machine.
//
// Occupancy ceiling is precisely what device STATE determines, which is what C1
// claims. It is not a substitute for achieved occupancy, and the report says so.
// ---------------------------------------------------------------------------
namespace {

struct OccRow {
    const char* name;
    int regs;
    size_t smem;          // static shared bytes/block
    size_t local;         // local (stack) bytes/thread
    int blocks_per_sm;    // max ACTIVE blocks per SM at `threads`
    double occupancy_pct; // active warps / max warps per SM
};

template <typename KernelT>
OccRow QueryOcc(const char* name, KernelT kern, int threads, int warps_per_sm,
                int warp_size) {
    cudaFuncAttributes a{};
    REQUIRE(cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(kern)) ==
            cudaSuccess);
    int blocks = 0;
    REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks, reinterpret_cast<const void*>(kern), threads,
                /*dynamicSMemSize=*/0) == cudaSuccess);
    double active_warps = double(blocks) * (double(threads) / warp_size);
    return OccRow{name,
                  a.numRegs,
                  a.sharedSizeBytes,
                  a.localSizeBytes,
                  blocks,
                  100.0 * active_warps / warps_per_sm};
}

void PrintOcc(const OccRow& r, int threads) {
    std::fprintf(stderr,
                 "[occ-api] %-26s regs=%-3d smem=%-4zuB local=%-5zuB "
                 "blocks/SM=%-2d  theoretical occupancy=%6.2f%%  (@%d thr/blk)\n",
                 r.name, r.regs, r.smem, r.local, r.blocks_per_sm,
                 r.occupancy_pct, threads);
}

}  // namespace

TEST_CASE("OCC theoretical occupancy (driver query, no launch, no server)",
          "[.][gpu][occupancy_api]") {
    OccCfg cfg;
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    const int warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::fprintf(stderr,
                 "[occ-api] %s  sm_%d%d  SMs=%d  maxThreads/SM=%d (=%d warps)  "
                 "regs/SM=%d  smem/SM=%zuB\n",
                 prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                 prop.maxThreadsPerMultiProcessor, warps_per_sm,
                 prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor);

    // Sweep block size so the result is not an artifact of one launch config.
    for (int threads : {128, 256, 512}) {
        std::fprintf(stderr, "[occ-api] ---- %d threads/block ----\n", threads);
        OccRow plain_flat = QueryOcc("GsStepPlainFlat", GsStepPlainFlat, threads,
                                     warps_per_sm, prop.warpSize);
        OccRow plain      = QueryOcc("GsStepPlain", GsStepPlain, threads,
                                     warps_per_sm, prop.warpSize);
        OccRow bpc        = QueryOcc("GsStepBlockPerChunk", GsStepBlockPerChunk,
                                     threads, warps_per_sm, prop.warpSize);
        OccRow fused      = QueryOcc("GsStepFused", GsStepFused, threads,
                                     warps_per_sm, prop.warpSize);
        PrintOcc(plain_flat, threads);
        PrintOcc(plain, threads);
        PrintOcc(bpc, threads);
        PrintOcc(fused, threads);

        // THE HEADLINE: the I/O path's cost is fused - blockPerChunk. Anything
        // measured against `plain` conflates I/O with the restructuring.
        std::fprintf(stderr,
                     "[occ-api] >> I/O COST (fused - blockPerChunk): "
                     "regs %+d, smem %+lldB, local %+lldB, occupancy %+.2f pp "
                     "  [vs. the WRONG two-way number (fused - plain): regs %+d]\n",
                     fused.regs - bpc.regs,
                     (long long)fused.smem - (long long)bpc.smem,
                     (long long)fused.local - (long long)bpc.local,
                     fused.occupancy_pct - bpc.occupancy_pct,
                     fused.regs - plain.regs);
    }

    // Depth invariance. Our fused kernel takes depth as a RUNTIME argument, so a
    // single image serves every depth — there is literally no per-depth kernel to
    // query, which is the claim. The counterfactual takes depth as a COMPILE-TIME
    // parameter, so each depth is a distinct image and CAN differ; show whether it
    // does, and where the growth lands.
    std::fprintf(stderr,
                 "[occ-api] ---- in-flight depth sweep @%u thr/blk ----\n",
                 cfg.threads);
    std::fprintf(stderr,
                 "[occ-api] GsStepFused: depth is a RUNTIME arg -> ONE image for "
                 "all depths; the row above is the answer at every depth.\n");
    const int thr = static_cast<int>(cfg.threads);
    PrintOcc(QueryOcc("GsStepFusedStateful<1>",  GsStepFusedStateful<1>,  thr, warps_per_sm, prop.warpSize), thr);
    PrintOcc(QueryOcc("GsStepFusedStateful<2>",  GsStepFusedStateful<2>,  thr, warps_per_sm, prop.warpSize), thr);
    PrintOcc(QueryOcc("GsStepFusedStateful<4>",  GsStepFusedStateful<4>,  thr, warps_per_sm, prop.warpSize), thr);
    PrintOcc(QueryOcc("GsStepFusedStateful<8>",  GsStepFusedStateful<8>,  thr, warps_per_sm, prop.warpSize), thr);
    PrintOcc(QueryOcc("GsStepFusedStateful<16>", GsStepFusedStateful<16>, thr, warps_per_sm, prop.warpSize), thr);
    PrintOcc(QueryOcc("GsStepFusedStateful<32>", GsStepFusedStateful<32>, thr, warps_per_sm, prop.warpSize), thr);
    PrintOcc(QueryOcc("GsStepFusedStateful<64>", GsStepFusedStateful<64>, thr, warps_per_sm, prop.warpSize), thr);
}

// ---------------------------------------------------------------------------
// REGISTER-PRESSURE SWEEP: when does fusing I/O actually cost occupancy?
//
// For each synthetic host-kernel register pressure, query the control (no I/O) and
// the fused kernel and report BOTH deltas. Emits a CSV block to stdout so the
// result lands in results/ verbatim.
// ---------------------------------------------------------------------------
TEST_CASE("OCC register-pressure sweep (driver query, no launch, no server)",
          "[.][gpu][occupancy_regsweep]") {
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    const int warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    const int thr = static_cast<int>(OccCfg{}.threads);

    std::fprintf(stderr,
                 "[regsweep] %s sm_%d%d  %d warps/SM  %d regs/SM  @%d thr/blk\n",
                 prop.name, prop.major, prop.minor, warps_per_sm,
                 prop.regsPerMultiprocessor, thr);
    // CSV to stdout — pipe straight into results/.
    std::printf("kLive,regs_control,regs_fused,delta_regs,"
                "occ_control_pct,occ_fused_pct,delta_occ_pp,fusion_costs_occupancy\n");

    auto probe = [&](int live, auto ctrl_kern, auto fused_kern) {
        OccRow c = QueryOcc("control", ctrl_kern, thr, warps_per_sm, prop.warpSize);
        OccRow f = QueryOcc("fused", fused_kern, thr, warps_per_sm, prop.warpSize);
        double docc = f.occupancy_pct - c.occupancy_pct;
        std::printf("%d,%d,%d,%d,%.2f,%.2f,%.2f,%s\n", live, c.regs, f.regs,
                    f.regs - c.regs, c.occupancy_pct, f.occupancy_pct, docc,
                    (docc < -0.005) ? "YES" : "no");
        std::fprintf(stderr,
                     "[regsweep] live=%-3d  control: regs=%-3d occ=%6.2f%%   "
                     "fused: regs=%-3d occ=%6.2f%%   | dregs=%+d  docc=%+6.2f pp %s\n",
                     live, c.regs, c.occupancy_pct, f.regs, f.occupancy_pct,
                     f.regs - c.regs, docc, (docc < -0.005) ? "  <== FUSION COSTS" : "");
    };

#define OCC_SYNTH_PROBE(R) probe(R, SynthControl<R>, SynthFused<R>)
    OCC_SYNTH_PROBE(2);
    OCC_SYNTH_PROBE(4);
    OCC_SYNTH_PROBE(6);
    OCC_SYNTH_PROBE(8);
    OCC_SYNTH_PROBE(12);
    OCC_SYNTH_PROBE(16);
    OCC_SYNTH_PROBE(20);
    OCC_SYNTH_PROBE(24);
    OCC_SYNTH_PROBE(28);
    OCC_SYNTH_PROBE(32);
    OCC_SYNTH_PROBE(40);
    OCC_SYNTH_PROBE(48);
    OCC_SYNTH_PROBE(56);
    OCC_SYNTH_PROBE(64);
    OCC_SYNTH_PROBE(80);
    OCC_SYNTH_PROBE(96);
#undef OCC_SYNTH_PROBE
}

// The measurement harness. ncu profiles this binary and filters by kernel name;
// one invocation launches all three variants at the SAME geometry so their
// achieved-occupancy numbers are directly comparable, then the counterfactual.
//
//   PART 2 (three-way occupancy):  OCC_CHUNKS=OCC_GRID=64
//   PART 3 (depth sweep):          OCC_GRID=8 fixed, OCC_CHUNKS in {8,16,32,64}
//                                  -> in-flight puts per block = 1,2,4,8
TEST_CASE("OCC three-variant register/occupancy harness",
          "[.][integration][gpu][cte][occupancy]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    (void)env;
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    OccCfg cfg;
    REQUIRE(cfg.chunks <= 64);                       // fire-all drain ceiling
    REQUIRE((cfg.N * cfg.N) % cfg.chunks == 0);      // uniform chunks
    const unsigned chunk_cells = (cfg.N * cfg.N) / cfg.chunks;
    const unsigned chunk_bytes = chunk_cells * sizeof(float);
    const unsigned depth = (cfg.chunks + cfg.grid - 1) / cfg.grid;

    std::fprintf(stderr,
                 "[occ] N=%u chunks=%u grid=%u threads=%u | chunk=%u B | "
                 "in-flight puts per block = %u\n",
                 cfg.N, cfg.chunks, cfg.grid, cfg.threads, chunk_bytes, depth);

    Fields f = MakeFields(cfg.N);

    kvhdf5::Layout layout{/*dims=*/{cfg.N * cfg.N},
                          /*chunk_dims=*/{chunk_cells},
                          /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0,
                             MakeTag("kvhdf5_occ_tag"), layout,
                             /*pool_size=*/0, DataKind(cfg));
    REQUIRE(ds.ChunkCount() == cfg.chunks);

    // --- VARIANT 1: plain. Same geometry as the others (grid-stride absorbs it).
    GsStepPlain<<<cfg.grid, cfg.threads>>>(f.u, f.v, f.un, f.vn, kGs, cfg.N);
    ctp::GpuApi::Synchronize();
    Check("GsStepPlain");

    // --- VARIANT 1a: plain at its NATURAL saturating grid, for reference only.
    GsStepPlainFlat<<<(f.cells + cfg.threads - 1) / cfg.threads, cfg.threads>>>(
        f.u, f.v, f.un, f.vn, kGs, cfg.N);
    ctp::GpuApi::Synchronize();
    Check("GsStepPlainFlat");

    // --- VARIANT 2: restructured, no I/O.
    GsStepBlockPerChunk<<<cfg.grid, cfg.threads>>>(f.u, f.v, f.un, f.snap, kGs,
                                                   cfg.N, cfg.chunks);
    ctp::GpuApi::Synchronize();
    Check("GsStepBlockPerChunk");

    // --- VARIANT 3: restructured + fused I/O. `depth` puts in flight per block.
    GsStepFused<<<cfg.grid, cfg.threads>>>(f.u, f.v, f.un, ds.Handle(), kGs, cfg.N,
                                           cfg.chunks);
    ctp::GpuApi::Synchronize();
    Check("GsStepFused");
    ds.ThrowIfIoFailed("occupancy harness / GsStepFused");  // a lost put would void the run

    std::fprintf(stderr, "[occ] all variants launched OK (fused drained cleanly)\n");
    FreeFields(f);
}

// Counterfactual sweep, run separately so a spill-heavy launch cannot perturb the
// three-variant numbers. Each depth D launches with grid = chunks/D, so one block
// really does hold D puts in flight — and really does retain D futures.
TEST_CASE("OCC stateful counterfactual (retained futures, O(D) state)",
          "[.][integration][gpu][cte][occupancy_counterfactual]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    OccCfg cfg;
    const unsigned chunk_cells = (cfg.N * cfg.N) / cfg.chunks;
    Fields f = MakeFields(cfg.N);

    kvhdf5::Layout layout{/*dims=*/{cfg.N * cfg.N},
                          /*chunk_dims=*/{chunk_cells},
                          /*elem_size=*/sizeof(float)};
    kvhdf5::GpuCteDataset ds(ipc, gpu_info, 0, MakeTag("kvhdf5_occ_cf_tag"), layout,
                             /*pool_size=*/0, DataKind(cfg));

    auto run = [&](auto kern, unsigned D, const char* label) {
        unsigned grid = cfg.chunks / D;
        if (grid == 0) return;
        kern<<<grid, cfg.threads>>>(f.u, f.v, f.un, ds.Handle(), kGs, cfg.N,
                                    cfg.chunks);
        ctp::GpuApi::Synchronize();
        Check(label);
        std::fprintf(stderr, "[occ-cf] depth=%-2u grid=%-2u OK (%s)\n", D, grid, label);
    };
    run(GsStepFusedStateful<1>,  1,  "stateful<1>");
    run(GsStepFusedStateful<2>,  2,  "stateful<2>");
    run(GsStepFusedStateful<4>,  4,  "stateful<4>");
    run(GsStepFusedStateful<8>,  8,  "stateful<8>");
    run(GsStepFusedStateful<16>, 16, "stateful<16>");
    run(GsStepFusedStateful<32>, 32, "stateful<32>");
    run(GsStepFusedStateful<64>, 64, "stateful<64>");

    ds.ThrowIfIoFailed("occupancy counterfactual");
    FreeFields(f);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to measure here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
