/*
 * Gray-Scott ASYNC end-to-end demo (the paper figure): a realistic GPU-resident
 * reaction-diffusion simulation that persists periodic, multi-chunk, HDF5-like
 * snapshots to iowarp with ASYNC-OVERLAPPED I/O, and measures the wall-clock win
 * of async over synchronous snapshotting IN THE REAL SIM.
 *
 * This is the realistic counterpart to async_overlap_test.cu's micro-benchmark:
 * there the "compute" was a synthetic FLOP burn; here it is the actual sim. The
 * overlap demonstrated is the headline benefit — snapshot N's server-side I/O hides
 * behind the GPU compute of the simulation steps that follow it:
 *
 *   ASYNC: ... [step][step] [snap: copy+fire, NO wait] [step][step]... ; drain at end
 *          └─ the fired snapshot's PutBlobs drain on the server's CPU worker WHILE
 *             the next steps' kernels run on the GPU.
 *   SYNC : ... [step][step] [snap: copy+Write = submit-AND-WAIT  ◄ GPU blocks] ...
 *          └─ the GPU spins in-kernel until the put completes; next steps can't start.
 *
 * Each snapshot is its own dataset PATH (results/.../step_NNNN -> distinct tag),
 * the trustworthy iterative recipe (sidesteps the dataset-reuse hang). We keep the
 * snapshot count under the ~16-large-backend-per-process iowarp ceiling (a surfaced
 * iowarp-core limit; see ADVISOR-REPORT.md / dataset_lifecycle_test.cu): the timed
 * demo pre-creates 6 sync + 6 async datasets = 12 total < 16.
 *
 * Two cases:
 *   - [grayscottasync]          DEFAULT-suite correctness guard: small, 2 chunks,
 *                               within-kernel fire-all/drain async snapshots, byte
 *                               verified. Stays in the nt4-safe window.
 *   - [.][grayscottasyncbench]  HIDDEN; the paper measurement. RUN AT num_threads=1.
 *                               Multi-chunk, large; times sync vs async, and asserts
 *                               async snapshots are byte-identical to sync (so the
 *                               speedup is not from dropping work).
 *
 * Reuses the one-time SharedCteEnv server bring-up.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/cpu_dataset.h>      // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>
#include <clio_cte/kvhdf5/tag_path.h>         // CanonicalTag

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "cte_env.h"

using kvhdf5::byte_t;

namespace {

struct GsParams { float Du, Dv, F, k, dt; };

}  // namespace

// One Gray-Scott step: one thread per cell, periodic BCs. Pure CUDA, touches zero
// kvhdf5 surface (compute on the GPU; I/O is explicit persistence).
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

// Copy the grid slice for chunk c into the handle's chunk-c buffer. Uniform chunk
// size, 1-D float layout => chunk c is the contiguous byte range [c*n, c*n+n).
__device__ inline void CopyChunk(kvhdf5::GpuDatasetHandle& h, uint32_t c,
                                 const byte_t* src) {
    byte_t* dst = h.Data(c);
    uint64_t n = h.Size(c);
    uint64_t off = static_cast<uint64_t>(c) * n;
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[off + i];
}

// SYNC snapshot (baseline): copy each chunk then fused Write(c) = submit-AND-WAIT.
// The GPU blocks in-kernel on each put, so the following sim steps can't overlap it.
__global__ void GsSnapSyncKernel(kvhdf5::GpuDatasetHandle h, const float* src) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const byte_t* s = reinterpret_cast<const byte_t*>(src);
    for (uint32_t c = 0; c < h.Count(); ++c) {
        CopyChunk(h, c, s);
        __threadfence_system();
        __syncthreads();
        h.Write(c);
        __syncthreads();
    }
}

// ASYNC snapshot, within-kernel fire-all/drain (bounded in-flight = Count()). Used
// by the DEFAULT correctness guard (proven-safe window). Overlaps chunk I/O with
// chunk compute inside the snapshot.
__global__ void GsSnapAsyncKernel(kvhdf5::GpuDatasetHandle h, const float* src) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const byte_t* s = reinterpret_cast<const byte_t*>(src);
    for (uint32_t c = 0; c < h.Count(); ++c) {
        CopyChunk(h, c, s);
        __threadfence_system();
        __syncthreads();
        h.WriteAsync(c);
        __syncthreads();
    }
    for (uint32_t c = 0; c < h.Count(); ++c) { h.WriteWait(c); __syncthreads(); }
}

// ASYNC snapshot FIRE only (no drain): copy each chunk, fire its PutBlob, return.
// The puts drain on the server while the SUBSEQUENT sim-step kernels run on the GPU
// — the cross-kernel overlap the timed demo measures. Drained later by GsDrainKernel.
__global__ void GsSnapFireKernel(kvhdf5::GpuDatasetHandle h, const float* src) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const byte_t* s = reinterpret_cast<const byte_t*>(src);
    for (uint32_t c = 0; c < h.Count(); ++c) {
        CopyChunk(h, c, s);
        __threadfence_system();
        __syncthreads();
        h.WriteAsync(c);
        __syncthreads();
    }
}

// Drain a fired snapshot's outstanding puts (thread-0 polls each completion bit).
__global__ void GsDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = 0; c < h.Count(); ++c) { h.WriteWait(c); __syncthreads(); }
}

#if !CTP_IS_DEVICE_PASS

namespace {

constexpr GsParams kGs{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

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
    auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, clio::run::u64(0), size, clio::run::u32(0), shm);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

// Allocate the 4 ping-pong grids and seed the classic Gray-Scott IC (u=1; v=1 in a
// centre square). Caller frees.
struct Grids { float *u_curr, *u_next, *v_curr, *v_next; };
Grids MakeGrids(unsigned N) {
    unsigned cells = N * N, bytes = cells * sizeof(float);
    Grids g{};
    REQUIRE(cudaMalloc(&g.u_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.u_next, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_next, bytes) == cudaSuccess);
    std::vector<float> u0(cells, 1.0f), v0(cells, 0.0f);
    unsigned lo = N/2 - 3, hi = N/2 + 3;
    for (unsigned y = lo; y < hi; ++y)
        for (unsigned x = lo; x < hi; ++x) v0[y*N + x] = 1.0f;
    ctp::GpuApi::Memcpy(g.u_curr, u0.data(), bytes);
    ctp::GpuApi::Memcpy(g.v_curr, v0.data(), bytes);
    return g;
}
void FreeGrids(Grids& g) {
    cudaFree(g.u_curr); cudaFree(g.u_next); cudaFree(g.v_curr); cudaFree(g.v_next);
}

// Pre-create one snapshot dataset per snapshot step (distinct path => distinct tag).
// Returns the datasets and the canonical tag for each (for read-back verify).
void MakeSnapDatasets(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                      const char* prefix, unsigned snaps, const kvhdf5::Layout& layout,
                      std::vector<kvhdf5::GpuCteDataset>& out,
                      std::vector<clio::cte::core::TagId>& tags) {
    out.clear(); tags.clear(); out.reserve(snaps);
    for (unsigned s = 0; s < snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        out.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout));
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
}

}  // namespace

// DEFAULT-suite correctness guard: a small multi-chunk Gray-Scott run whose periodic
// snapshots are persisted via the within-kernel fire-all/drain ASYNC path, then read
// back and byte-verified against the host-recorded grid. Proves the realistic async
// snapshot path is correct; stays in the nt4-safe window (2 chunks, small).
TEST_CASE("GPU Gray-Scott async multi-chunk snapshots round-trip",
          "[integration][gpu][cte][grayscottasync]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    (void)env;
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    const unsigned N = 32, C = 2, cells = N*N, bytes = cells*sizeof(float);
    const int steps = 12, snap = 4;
    const unsigned nsnap = steps / snap;
    kvhdf5::Layout layout{/*dims=*/{cells}, /*chunk_dims=*/{cells / C},
                          /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == C);

    std::vector<kvhdf5::GpuCteDataset> snaps;
    std::vector<clio::cte::core::TagId> tags;
    MakeSnapDatasets(ipc, gpu_info, "results/grayscott/async_guard", nsnap, layout,
                     snaps, tags);

    Grids g = MakeGrids(N);
    std::vector<std::vector<float>> expected;  // host v-grid at each snapshot
    unsigned threads = 256, blocks = (cells + threads - 1) / threads;
    unsigned si = 0;
    for (int step = 1; step <= steps; ++step) {
        GsStepKernel<<<blocks, threads>>>(g.u_curr, g.v_curr, g.u_next, g.v_next, kGs, N);
        ctp::GpuApi::Synchronize();
        std::swap(g.u_curr, g.u_next);
        std::swap(g.v_curr, g.v_next);
        if (step % snap != 0) continue;

        std::vector<float> hostv(cells);
        ctp::GpuApi::Memcpy(hostv.data(), g.v_curr, bytes);
        REQUIRE(std::memcmp(hostv.data(), std::vector<float>(cells, 0.0f).data(),
                            bytes) != 0);   // compute actually ran
        expected.push_back(hostv);

        GsSnapAsyncKernel<<<1, 256>>>(snaps[si].Handle(), g.v_curr);
        ctp::GpuApi::Synchronize();
        ++si;
    }
    REQUIRE(si == nsnap);

    // Read each snapshot's chunks back and verify byte-identical to the host grid.
    const unsigned chunk_bytes = bytes / C;
    for (unsigned s = 0; s < nsnap; ++s) {
        for (unsigned c = 0; c < C; ++c) {
            auto got = HostReadBlob(tags[s], std::to_string(c), chunk_bytes);
            const byte_t* exp = reinterpret_cast<const byte_t*>(expected[s].data())
                                + static_cast<uint64_t>(c) * chunk_bytes;
            REQUIRE(std::memcmp(got.data(), exp, chunk_bytes) == 0);
        }
    }
    FreeGrids(g);
    std::fprintf(stderr, "[ok] gray-scott async: %u snapshots x %u chunks verified\n",
                 nsnap, C);
}

namespace {

// Run the sim, snapshotting every `snap` steps into the pre-created `ds`. `async`:
// fire-and-continue (overlap), drained at the end; else fused submit-and-wait. NO
// per-step Synchronize so the stream can overlap; one Synchronize closes the timed
// region. Returns wall-clock ms.
double RunSimTimed(Grids& g, unsigned N, int steps, int snap,
                   std::vector<kvhdf5::GpuCteDataset>& ds, bool async) {
    unsigned cells = N*N, threads = 256, blocks = (cells + threads - 1) / threads;
    // Settle the seed before timing.
    ctp::GpuApi::Synchronize();
    auto t0 = std::chrono::steady_clock::now();
    unsigned si = 0;
    for (int step = 1; step <= steps; ++step) {
        GsStepKernel<<<blocks, threads>>>(g.u_curr, g.v_curr, g.u_next, g.v_next, kGs, N);
        std::swap(g.u_curr, g.u_next);
        std::swap(g.v_curr, g.v_next);
        if (step % snap != 0) continue;
        if (async) GsSnapFireKernel<<<1, 256>>>(ds[si].Handle(), g.v_curr);
        else       GsSnapSyncKernel<<<1, 256>>>(ds[si].Handle(), g.v_curr);
        ++si;
    }
    if (async) for (auto& d : ds) GsDrainKernel<<<1, 32>>>(d.Handle());
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

// HIDDEN ([.]): the paper measurement. RUN AT SERVER num_threads=1 (the same nt1
// substrate as the async micro-benchmark). Multi-chunk Gray-Scott; snapshots persisted
// sync vs async; reports the wall-clock speedup AND asserts the async snapshots are
// byte-identical to the sync ones (the speedup is real overlap, not dropped work).
// Snapshot count (6) keeps total large-dataset creations (6 sync + 6 async = 12) under
// the ~16 iowarp ceiling.
TEST_CASE("GPU Gray-Scott async-overlapped snapshots beat sync in a real sim (nt1)",
          "[.][integration][gpu][cte][grayscottasyncbench]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    const unsigned N = 512, C = 4, cells = N*N, bytes = cells*sizeof(float);
    const unsigned nsnap = 6;
    const int snap = 48, steps = static_cast<int>(nsnap) * snap;  // 288
    kvhdf5::Layout layout{/*dims=*/{cells}, /*chunk_dims=*/{cells / C},
                          /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == C);

    std::vector<kvhdf5::GpuCteDataset> sync_ds, async_ds;
    std::vector<clio::cte::core::TagId> sync_tags, async_tags;
    MakeSnapDatasets(ipc, gpu_info, "results/grayscott/bench_sync", nsnap, layout,
                     sync_ds, sync_tags);
    MakeSnapDatasets(ipc, gpu_info, "results/grayscott/bench_async", nsnap, layout,
                     async_ds, async_tags);

    Grids gs = MakeGrids(N);
    double sync_ms = RunSimTimed(gs, N, steps, snap, sync_ds, /*async=*/false);
    FreeGrids(gs);

    Grids ga = MakeGrids(N);
    double async_ms = RunSimTimed(ga, N, steps, snap, async_ds, /*async=*/true);
    FreeGrids(ga);

    // Correctness: identical deterministic sims => the ASYNC snapshot must byte-match
    // the proven SYNC baseline at every chunk (so the speedup is real overlap, not
    // dropped work). Plus sanity: real data was produced (some chunk non-zero — note
    // a single chunk CAN be all-zero, since Gray-Scott activity is a centre patch far
    // from the edge rows) and the field evolved between the first and last snapshot.
    const unsigned chunk_bytes = bytes / C;
    const std::vector<byte_t> zero(chunk_bytes, byte_t{});
    std::vector<byte_t> first_full, last_full;
    bool any_nonzero = false;
    for (unsigned s = 0; s < nsnap; ++s) {
        std::vector<byte_t> full;
        for (unsigned c = 0; c < C; ++c) {
            auto sv = HostReadBlob(sync_tags[s], std::to_string(c), chunk_bytes);
            auto av = HostReadBlob(async_tags[s], std::to_string(c), chunk_bytes);
            REQUIRE(av == sv);                       // async ≡ sync (the real proof)
            if (av != zero) any_nonzero = true;
            full.insert(full.end(), av.begin(), av.end());
        }
        if (s == 0) first_full = full;
        if (s == nsnap - 1) last_full = full;
    }
    REQUIRE(any_nonzero);                            // produced real data
    REQUIRE(first_full != last_full);                // the sim evolved across snapshots

    double speedup = sync_ms / async_ms;
    std::fprintf(stderr,
                 "[bench] gray-scott %ux%u, %u chunks x %u KiB, %d steps, %u snapshots\n"
                 "[bench]   sync  = %8.2f ms\n"
                 "[bench]   async = %8.2f ms\n"
                 "[bench]   speedup = %.2fx\n",
                 N, N, C, chunk_bytes/1024u, steps, nsnap, sync_ms, async_ms, speedup);
    // Real overlap win in a real sim (modest, repeatable margin; single-worker 2-stage
    // pipeline ceiling is ~2x at the compute~IO balance).
    REQUIRE(speedup > 1.05);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
