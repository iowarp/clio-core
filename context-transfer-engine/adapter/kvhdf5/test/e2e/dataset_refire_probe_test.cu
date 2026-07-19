/*
 * DE-RISK PROBE for the three-way Gray-Scott I/O benchmark (see
 * agents/refactor/three-way-bench-plan.md, Step 0).
 *
 * The benchmark's CLIO arms must stream ~2 GB through a SMALL fixed ring of
 * datasets, because iowarp-core has a fixed ~16 large-GPU-backend registrations
 * per process and frees do NOT reclaim (ADVISOR-REPORT §6a). So we cannot create
 * a fresh dataset per snapshot; we must create K datasets ONCE and RE-FIRE each
 * dataset's Write many times.
 *
 * A stale note in dataset_lifecycle_test.cu says "sequential REUSE of one
 * GpuCteDataset HANGS (one-round-trip-per-dataset)". That predates the current
 * *stateless* FutureShm wait (GpuDatasetHandle::SubmitWait rebuilds the future
 * from the task slot). Whether re-firing still hangs / goes stale / corrupts is
 * now the OPEN QUESTION this probe settles. It is the linchpin of the whole plan.
 *
 * Method: create ONE GpuCteDataset (single chunk => one blob key "0"), then in a
 * HOST loop launch a fill+fused-Write kernel N times, each with a DISTINCT seed,
 * and read the blob back after each iteration asserting it equals the LATEST
 * pattern. Fused-sync (<= 1 put in flight) => nt-agnostic, safe in the default
 * substrate. Per-iteration stderr progress makes a hang visible at its iteration
 * (run the binary under `timeout` so a hang aborts).
 *
 *   - CLEAN            -> ring reuse is viable as-is.
 *   - STALE read-back  -> re-fire returns an earlier write's bytes; FutureShm /
 *                         task-id needs a reset before re-Send (cheap host fix).
 *   - HANG at iter 2   -> re-fire genuinely unsupported; fall back to K-ring one-
 *                         fire-each capped at the ceiling, or small-chunk high-count.
 *
 * The LARGE case (1 MiB x 200 = 200 MiB moved through ONE 1 MiB blob, overwritten)
 * ALSO probes whether the 64 MiB kRam bdev reclaims storage on overwrite (if it
 * leaks, it trips around the 64th iteration). Both cases HIDDEN ([.]).
 *
 * Reuses SharedCteEnv.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/cpu_dataset.h>      // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>
#include <clio_cte/kvhdf5/tag_path.h>         // CanonicalTag

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

// chunk byte i for a given seed -> (seed ^ i) & 0xFF. Distinct seed per iteration
// so a stale buffer/blob MISMATCHES rather than silently coinciding.
constexpr byte_t RefirePattern(unsigned seed, uint64_t i) {
    return static_cast<byte_t>((seed ^ static_cast<unsigned>(i)) & 0xFFu);
}

}  // namespace

// Fill chunk 0 with the seed's pattern, then fused-Write (Send().Wait()): at most
// one put in flight. Re-launched each iteration on the SAME handle == the reuse.
__global__ void RefireFillWriteKernel(kvhdf5::GpuDatasetHandle h, unsigned seed) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    byte_t* dst = h.Data(0);
    uint64_t n = h.Size(0);
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = static_cast<byte_t>((seed ^ static_cast<unsigned>(i)) & 0xFFu);
    __threadfence_system();
    __syncthreads();
    h.Write(0);
    __syncthreads();
}

#if !CTP_IS_DEVICE_PASS

namespace {

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

// Create ONE single-chunk dataset, re-fire its Write `iters` times with distinct
// seeds, verify the store holds the LATEST pattern after each fire. `prefix` keeps
// cases on disjoint tag namespaces.
void RunRefire(unsigned iters, uint64_t chunk_bytes, const char* prefix) {
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    auto* cte_client = CLIO_CTE_CLIENT;

    // Single chunk => blob key "0". dims == chunk_dims == chunk_bytes, elem_size 1.
    kvhdf5::Layout layout{/*dims=*/{chunk_bytes}, /*chunk_dims=*/{chunk_bytes},
                          /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == 1);

    std::string path = std::string(prefix) + "/refire/blob";
    // ONE dataset for the whole run — this is the reuse under test.
    kvhdf5::GpuCteDataset ds = kvhdf5::GpuCteDataset::FromPath(
        ipc, gpu_info, /*gpu_id=*/0, cte_client, path, layout);
    REQUIRE(ds.ChunkCount() == 1);
    clio::cte::core::TagId tag = MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str());

    for (unsigned s = 0; s < iters; ++s) {
        const unsigned seed = 0x100u + s * 7u + 1u;  // distinct, non-trivial
        std::fprintf(stderr, "[refire] iter %u/%u fire (seed=%u)\n", s + 1, iters, seed);
        RefireFillWriteKernel<<<1, 64>>>(ds.Handle(), seed);
        ctp::GpuApi::Synchronize();

        auto got = HostReadBlob(tag, "0", chunk_bytes);
        // Full compare would be O(N) per iter; check a strided sample + endpoints.
        bool ok = got.size() == chunk_bytes;
        for (uint64_t i = 0; ok && i < chunk_bytes;
             i += (chunk_bytes > 4096 ? 997 : 1))
            ok = (got[i] == RefirePattern(seed, i));
        if (chunk_bytes >= 2)
            ok = ok && got[0] == RefirePattern(seed, 0) &&
                 got[chunk_bytes - 1] == RefirePattern(seed, chunk_bytes - 1);
        if (!ok)
            std::fprintf(stderr, "[refire] iter %u MISMATCH (stale or corrupt)\n", s + 1);
        REQUIRE(ok);
    }
    std::fprintf(stderr, "[ok] refire %s: %u re-fires x %llu B on ONE dataset clean\n",
                 prefix, iters, (unsigned long long)chunk_bytes);
}

}  // namespace

// HIDDEN ([.]): small re-fire — isolates "does re-fire hang / go stale?" from size.
// Run under `timeout`; nt-agnostic (sequential fused-sync).
TEST_CASE("PROBE dataset re-fire small (4 KiB x 50)",
          "[.][integration][gpu][cte][refireprobe]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunRefire(/*iters=*/50, /*chunk_bytes=*/4096u, "results/probe/small");
}

// HIDDEN ([.]): large re-fire — proves the ~16-backend ceiling is about registration
// COUNT not writes (200 fires on ONE backend), and probes 64 MiB kRam overwrite
// reclaim (200 MiB moved through one 1 MiB blob). Run under `timeout`; nt-agnostic.
TEST_CASE("PROBE dataset re-fire large (1 MiB x 200)",
          "[.][integration][gpu][cte][refireprobelarge]") {
    (void)kvhdf5::itest::SharedCteEnv();
    RunRefire(/*iters=*/200, /*chunk_bytes=*/1024u * 1024u, "results/probe/large");
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
