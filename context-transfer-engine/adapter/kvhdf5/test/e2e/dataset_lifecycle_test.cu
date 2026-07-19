/*
 * Iterative dataset-lifecycle GPU integration test (de-risk for the E2E demo).
 *
 * The end-to-end Gray Scott demo is iterative: it persists a snapshot every K
 * steps over many steps. Two known fragilities live exactly there:
 *   1. Sequential REUSE of one GpuCteDataset HANGS (the one-round-trip-per-dataset
 *      invariant) -> snapshots cannot reuse a dataset.
 *   2. The workaround is a FRESH GpuCteDataset per snapshot, which churns three
 *      registered GPU backends per snapshot (alloc + free). Earlier async-bench
 *      evidence (a corruption on the ~10th dataset of a run) HINTED that cumulative
 *      churn might corrupt -- but that run also had concurrency (4-8 in-flight
 *      puts) and a heavy kernel, so the variables were tangled.
 *
 * This test settles a TRUSTWORTHY snapshot recipe and DISCRIMINATES the churn
 * signal. The recipe adopts the HDF5-like model: each snapshot is its own dataset
 * PATH (results/snapshots/2026/temperature/step_NNNN), built through the real
 * GpuCteDataset::FromPath surface. Distinct path => distinct CanonicalTag =>
 * distinct CTE tag / blob namespace, which sidesteps the reuse-hang BY
 * CONSTRUCTION.
 *
 * The discriminator: writes are SEQUENTIAL + fused-sync (<= 1 put in flight at a
 * time), which isolates dataset CHURN from concurrency. If it corrupts as the
 * snapshot count grows -> churn is real. If it stays clean over many snapshots ->
 * pure churn is NOT the cause (the earlier corruption was concurrency-/heavy-kernel
 * specific). Because writes are sequential, this is safe at ANY num_threads, so it
 * runs in the DEFAULT integration suite. Verification is host-side CTE read-back
 * AFTER the dataset destructs (the bytes persist in the store, not in the freed GPU
 * buffer -- a stronger check). Reuses SharedCteEnv.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/cpu_dataset.h>      // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>
#include <clio_cte/kvhdf5/tag_path.h>         // CanonicalTag (verify the same tag FromPath made)

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

// chunk c, byte i -> ((seed + c) ^ i) & 0xFF; distinct seed per snapshot so a
// stale buffer/blob would MISMATCH rather than silently coincide.
constexpr byte_t LifePattern(unsigned seed, unsigned i) {
    return static_cast<byte_t>((seed ^ i) & 0xFFu);
}

}  // namespace

// Sequential producer: one block fills each chunk then fused-Write(c) = Send().Wait(),
// so at most one put is ever in flight (isolates churn from concurrency).
__global__ void LifecycleFillWriteKernel(kvhdf5::GpuDatasetHandle h, unsigned seed) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = 0; c < h.Count(); ++c) {
        byte_t* dst = h.Data(c);
        uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = static_cast<byte_t>(((seed + c) ^ i) & 0xFFu);
        __threadfence_system();
        __syncthreads();
        h.Write(c);        // fused Send().Wait(): <= 1 in-flight
        __syncthreads();
    }
}

// Data-dependent FLOP burn so the async snapshot has real compute to overlap (and
// to mirror the heavy-kernel factor in the original corruption regime). `wb`
// (always 0) keeps `acc` live without ever writing the data back.
__device__ void LifeCompute(byte_t* dst, uint64_t n, unsigned iters, unsigned wb) {
    if (iters == 0) return;
    float acc = 1.0f;
    for (unsigned k = 0; k < iters; ++k)
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            acc = acc * 1.0000001f + static_cast<float>(dst[i]) * 1e-6f;
    if (wb && acc < 0.0f) dst[threadIdx.x % n] = static_cast<byte_t>(acc);
}

// ASYNC producer: one block fires every chunk's PutBlob without waiting, then
// drains at the end (fire-all/drain). This is the E2E's actual snapshot path —
// multiple in-flight puts per snapshot — so looping it over many fresh datasets
// hits the original corruption intersection (async + size + many datasets).
__global__ void LifecycleAsyncFillWriteKernel(kvhdf5::GpuDatasetHandle h,
                                              unsigned seed, unsigned iters) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = 0; c < h.Count(); ++c) {
        byte_t* dst = h.Data(c);
        uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = static_cast<byte_t>(((seed + c) ^ i) & 0xFFu);
        LifeCompute(dst, n, iters, /*wb=*/0u);
        __threadfence_system();
        __syncthreads();
        h.WriteAsync(c);   // fire; do NOT wait
        __syncthreads();
    }
    for (uint32_t c = 0; c < h.Count(); ++c) {
        h.WriteWait(c);    // drain
        __syncthreads();
    }
}

#if !CTP_IS_DEVICE_PASS

namespace {

clio::cte::core::TagId MakeTag(const char* name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

std::vector<byte_t> HostReadBlob(clio::cte::core::TagId tag,
                                 const std::string& name, uint64_t size) {
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(size);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0, size);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, /*offset=*/clio::run::u64(0), size,
                                           /*flags=*/clio::run::u32(0), shm);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

// Run `snapshots` fresh dataset-per-path snapshots and verify every blob clean.
// `prefix` keeps distinct cases on disjoint tag namespaces. Sequential fused-sync
// writes => the churn discriminator (no concurrency variable).
void RunLifecycle(unsigned snapshots, unsigned chunks, unsigned chunk_bytes,
                  const char* prefix) {
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    auto* cte_client = CLIO_CTE_CLIENT;

    kvhdf5::Layout layout{/*dims=*/{chunks * chunk_bytes},
                          /*chunk_dims=*/{chunk_bytes}, /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == chunks);

    for (unsigned s = 0; s < snapshots; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/temperature/step_%05u", prefix, s);
        const unsigned seed = 0x40u + s;  // distinct per snapshot

        {
            // Fresh dataset (3 fresh registered GPU backends); destructs at block
            // end -> frees them. The churn we want to stress.
            kvhdf5::GpuCteDataset ds = kvhdf5::GpuCteDataset::FromPath(
                ipc, gpu_info, /*gpu_id=*/0, cte_client, path, layout);
            REQUIRE(ds.ChunkCount() == chunks);
            // Diagnostic (large runs only): log the data-backend device address to
            // see whether the 17th registration aliases an earlier freed one.
            if (chunk_bytes >= 65536u)
                std::fprintf(stderr, "[addr] snapshot %u data0=%p task-chunkbytes=%u\n",
                             s, (void*)ds.DeviceData(0), chunk_bytes);
            LifecycleFillWriteKernel<<<1, 32>>>(ds.Handle(), seed);
            ctp::GpuApi::Synchronize();
        }

        // Verify AFTER destruct: data lives in the CTE store keyed by (tag, coord),
        // independent of the freed GPU buffer. Resolve the SAME tag FromPath made.
        clio::cte::core::TagId tag =
            MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str());
        for (unsigned c = 0; c < chunks; ++c) {
            std::vector<byte_t> expected(chunk_bytes);
            for (unsigned i = 0; i < chunk_bytes; ++i)
                expected[i] = LifePattern(seed + c, i);
            auto got = HostReadBlob(tag, std::to_string(c), chunk_bytes);
            if (got != expected)
                std::fprintf(stderr, "[lifecycle] snapshot %u chunk %u mismatch\n",
                             s, c);
            REQUIRE(got == expected);
        }
    }
    std::fprintf(stderr, "[ok] lifecycle %s: %u snapshots x %u chunks x %u B clean\n",
                 prefix, snapshots, chunks, chunk_bytes);
}

// Iterative ASYNC-at-size variant: every snapshot fires all its chunks fire-all/
// drain (multiple in-flight puts) at demo chunk size, over many fresh datasets —
// the E2E's real snapshot path AND the original corruption intersection. MUST run
// at server num_threads=1 (the concurrent-put-safe substrate). `iters` adds a
// per-chunk compute burn to mirror the heavy-kernel factor.
void RunLifecycleAsync(unsigned snapshots, unsigned chunks, unsigned chunk_bytes,
                       unsigned iters, const char* prefix) {
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    auto* cte_client = CLIO_CTE_CLIENT;

    kvhdf5::Layout layout{/*dims=*/{chunks * chunk_bytes},
                          /*chunk_dims=*/{chunk_bytes}, /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == chunks);

    for (unsigned s = 0; s < snapshots; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/temperature/step_%05u", prefix, s);
        const unsigned seed = 0x40u + s;

        {
            kvhdf5::GpuCteDataset ds = kvhdf5::GpuCteDataset::FromPath(
                ipc, gpu_info, /*gpu_id=*/0, cte_client, path, layout);
            REQUIRE(ds.ChunkCount() == chunks);
            LifecycleAsyncFillWriteKernel<<<1, 32>>>(ds.Handle(), seed, iters);
            ctp::GpuApi::Synchronize();
        }

        clio::cte::core::TagId tag =
            MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str());
        for (unsigned c = 0; c < chunks; ++c) {
            std::vector<byte_t> expected(chunk_bytes);
            for (unsigned i = 0; i < chunk_bytes; ++i)
                expected[i] = LifePattern(seed + c, i);
            auto got = HostReadBlob(tag, std::to_string(c), chunk_bytes);
            if (got != expected)
                std::fprintf(stderr,
                             "[lifecycle-async] snapshot %u chunk %u mismatch\n", s, c);
            REQUIRE(got == expected);
        }
    }
    std::fprintf(stderr,
                 "[ok] lifecycle-async %s: %u snapshots x %u chunks x %u KiB "
                 "(iters=%u) clean\n",
                 prefix, snapshots, chunks, chunk_bytes / 1024u, iters);
}

}  // namespace

// DEFAULT-suite characterization/discriminator: prove a fresh dataset-per-snapshot
// path stays byte-clean across many snapshots (the churn discriminator). A failure
// that appears as the snapshot count grows would mean churn corruption; staying
// clean clears churn as the cause and gives the E2E a trustworthy snapshot recipe.
// 64 snapshots covers the E2E's 50-100 range; sequential writes keep it nt-agnostic.
TEST_CASE("GPU fresh-dataset-per-snapshot lifecycle stays clean",
          "[integration][gpu][cte][lifecycle]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunLifecycle(/*snapshots=*/64, /*chunks=*/4, /*chunk_bytes=*/4096,
                 "results/snapshots/2026");
}

// HIDDEN ([.]): push the churn bound well past the E2E range (200 snapshots =>
// 600 backend alloc/free cycles) to confirm there is no churn cliff. Hidden only
// to keep the default suite fast; still sequential, so safe at any num_threads.
TEST_CASE("GPU dataset-per-snapshot lifecycle has no churn cliff (stress)",
          "[.][integration][gpu][cte][lifecyclestress]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunLifecycle(/*snapshots=*/200, /*chunks=*/4, /*chunk_bytes=*/4096,
                 "results/stress/2026");
}

// HIDDEN ([.]): the REAL E2E de-risk — iterative ASYNC snapshots at demo chunk
// size over many fresh datasets, the original corruption intersection (async +
// large + many datasets). RUN AT SERVER num_threads=1 (the concurrent-put-safe
// substrate; the same driver/env as [asyncbench]). 100 snapshots x 4 chunks x
// 256 KiB, fire-all/drain per snapshot. Clean => the async snapshot lifecycle the
// E2E depends on is genuinely de-risked; a mismatch reproduces the corruption in a
// controlled iterative setting.
TEST_CASE("GPU iterative async snapshot lifecycle stays clean at size (nt1)",
          "[.][integration][gpu][cte][lifecycleasync]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunLifecycleAsync(/*snapshots=*/100, /*chunks=*/4,
                      /*chunk_bytes=*/256u * 1024u, /*iters=*/2u,
                      "results/asynclife/2026");
}

// DISCRIMINATOR D1 — ASYNC at SMALL size (4 KiB), many snapshots. Isolates whether
// chunk SIZE is required for the corruption: clean => size-dependent (async+large);
// corrupt => async + churn alone corrupts (size irrelevant). Run at nt1.
TEST_CASE("DISC async-small iterative lifecycle (nt1)",
          "[.][integration][gpu][cte][lifecycleasyncsmall]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunLifecycleAsync(/*snapshots=*/100, /*chunks=*/4, /*chunk_bytes=*/4096u,
                      /*iters=*/2u, "results/asyncsmall/2026");
}

// DISCRIMINATOR D2 — SYNC at LARGE size (256 KiB), many snapshots. Isolates whether
// ASYNC (multiple in-flight) is required: clean => async required (size alone is
// fine); corrupt => size + churn alone corrupts (concurrency irrelevant). Run at
// nt1 to match the async case's substrate (sync is safe at any nt regardless).
TEST_CASE("DISC sync-large iterative lifecycle (nt1)",
          "[.][integration][gpu][cte][lifecyclesynclarge]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunLifecycle(/*snapshots=*/100, /*chunks=*/4, /*chunk_bytes=*/256u * 1024u,
                 "results/synclarge/2026");
}

// DISCRIMINATOR D3 — sync-large at DOUBLE the chunk size (512 KiB => 2 MiB/dataset).
// If the cause is a ~16 MiB cumulative non-reclaiming device-backend arena, this
// must fail at ~HALF the snapshot index of D2 (~snapshot 8 vs ~16). Confirms the
// failure tracks CUMULATIVE BYTES, not dataset count. Run at nt1.
TEST_CASE("DISC sync-large-2x iterative lifecycle (nt1)",
          "[.][integration][gpu][cte][lifecyclesynclarge2x]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunLifecycle(/*snapshots=*/40, /*chunks=*/4, /*chunk_bytes=*/512u * 1024u,
                 "results/synclarge2x/2026");
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
