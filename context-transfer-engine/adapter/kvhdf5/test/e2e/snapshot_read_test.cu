/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/*
 * GPU-INITIATED SNAPSHOT READ-BACK: the Get side of the reuse model.
 *
 * The write path (snapshot_reuse_test) is well covered. The READ path was not: before
 * this test, `h.Read()` appeared only in a few single-shot put/get tests, and
 * `h.ReadAsync()` / `h.ReadWait()` were NOT EXERCISED ANYWHERE IN THE TREE. That is
 * exactly the shape of untested path where the write side hid the reuse hang (a 2nd
 * fire into a reused task slot mis-driven as a completed coroutine), so it gets the
 * same treatment here rather than being trusted because the benchmark "looked right".
 *
 * WHAT IT GUARDS:
 *  1. SYNC read (h.Read = submit-AND-wait), one dataset RE-ARMED to each snapshot's
 *     tag in turn, must return each snapshot's bytes exactly.
 *  2. ASYNC PIPELINED read (SetTag + h.ReadAsync, drained by h.ReadWait) over 2 reused
 *     groups, with snapshot s+1's Gets FIRED BEFORE snapshot s is drained/copied --
 *     the ordering the reader benchmark actually uses. This is where a re-tag race or
 *     a missing drain-before-refill would corrupt, and it is checked per snapshot, not
 *     just in aggregate.
 *  3. TEETH: the read buffers are POISONED (0xEE) before every read pass, so a Get that
 *     silently does nothing fails instead of passing on stale/zero content. Content is
 *     keyed on the SNAPSHOT index, so cross-snapshot mixups show up as changed bytes
 *     rather than as a self-consistent wrong answer.
 *  4. Both read paths must agree with the host-computed expectation AND with each other.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/layout.h>
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>

#include <cstdint>
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
constexpr unsigned kChunks     = 4;
constexpr unsigned kChunkBytes = 64u * 1024u;
constexpr unsigned kThreads    = 128;
constexpr unsigned kGroups     = 2;      // read-side double buffer
constexpr unsigned kSnaps      = 8;      // > kGroups, so groups really do get reused
constexpr int      kPoison     = 0xEE;   // pre-read fill: a no-op Get must NOT pass
}  // namespace

// Content keyed on SNAPSHOT and chunk (never on the buffer/group), so any cross-snapshot
// mixup changes bytes. Host + device identical.
__host__ __device__ inline byte_t RdSnapByte(uint32_t snap, uint32_t c, uint64_t i) {
    return static_cast<byte_t>(
        ((snap * 2246822519u) ^ (c * 2654435761u) ^
         (static_cast<uint32_t>(i) * 40503u) ^ 0x5Au) & 0xFFu);
}

// ---- write side (to produce something to read) -----------------------------------
__global__ void RdFillFireKernel(kvhdf5::GpuDatasetHandle h, uint32_t snap) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        byte_t* dst = h.Data(c);
        uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = RdSnapByte(snap, c, i);
        __threadfence_system();
        __syncthreads();
        h.WriteAsync</*Probing=*/false>(c);
        __syncthreads();
    }
}
__global__ void RdWriteDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.WriteWait(c);
        __syncthreads();
    }
}

// ---- read side -------------------------------------------------------------------
// SYNC: submit-AND-wait per chunk. The destination tag is set by the HOST (Rearm)
// before launch, which is safe here only because the caller synchronizes per snapshot.
__global__ void RdReadSyncKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.Read(c);
        __syncthreads();
    }
}
// ASYNC: the DEVICE stamps this snapshot's tag (SetTag covers the GET slot) and fires the
// Get without waiting. Device-stamped, not host-Rearm'd, because the pipelined caller runs
// ahead of the GPU -- a host re-tag would race the worker reading tag_id_ for the previous
// still-in-flight Get.
__global__ void RdReadFireKernel(kvhdf5::GpuDatasetHandle h,
                                 const clio::cte::core::TagId* tag_table, uint32_t snap) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.SetTag(c, tag_table[snap]);
        __syncthreads();
        h.ReadAsync(c);
        __syncthreads();
    }
}
__global__ void RdReadDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.ReadWait(c);
        __syncthreads();
    }
}

#if !CTP_IS_DEVICE_PASS

namespace {

kvhdf5::Layout RdLayout() {
    return kvhdf5::Layout{/*dims=*/{kChunks * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes}, /*elem_size=*/1};
}

clio::cte::core::TagId MakeTag(const std::string& name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

std::vector<clio::cte::core::TagId> MakeSnapTags(const char* prefix, unsigned snaps) {
    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(snaps);
    for (unsigned s = 0; s < snaps; ++s)
        tags.push_back(MakeTag(std::string(prefix) + std::to_string(s)));
    return tags;
}

// Fill kSnaps snapshots' worth of blobs, one dataset re-armed per snapshot (fully
// synchronized between snapshots -- the write side is not what this test is probing).
void WriteSnapshots(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                    const std::vector<clio::cte::core::TagId>& tags) {
    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, tags[0], RdLayout());
    for (unsigned s = 0; s < tags.size(); ++s) {
        ds.Rearm(tags[s]);
        RdFillFireKernel<<<kChunks, kThreads>>>(ds.Handle(), s);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        RdWriteDrainKernel<<<kChunks, 32>>>(ds.Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        ctp::GpuApi::Synchronize();
        ds.ThrowIfIoFailed("snapshot_read(write)");
    }
}

void PoisonBuffers(kvhdf5::GpuCteDataset& ds) {
    for (unsigned c = 0; c < kChunks; ++c)
        REQUIRE(cudaMemset(ds.DeviceData(c), kPoison, kChunkBytes) == cudaSuccess);
}

// Compare one snapshot's kChunks buffers against the expectation. Returns mismatch count.
uint64_t VerifySnap(const std::vector<std::vector<byte_t>>& got, uint32_t snap) {
    uint64_t bad = 0;
    for (unsigned c = 0; c < kChunks; ++c)
        for (uint64_t i = 0; i < kChunkBytes; ++i)
            if (got[c][i] != RdSnapByte(snap, c, i)) ++bad;
    return bad;
}

}  // namespace

TEST_CASE("GPU snapshot read-back: SYNC Get returns every snapshot exactly",
          "[integration][gpu][cte][read]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    auto tags = MakeSnapTags("kvhdf5_read_sync/snap_", kSnaps);
    WriteSnapshots(ipc, gpu_info, tags);

    // One dataset, re-armed per snapshot. Safe with a host Rearm because we synchronize
    // after each snapshot's read (the sync reader's contract).
    kvhdf5::GpuCteDataset rd(ipc, gpu_info, /*gpu_id=*/0, tags[0], RdLayout());
    uint64_t bad_total = 0;
    for (unsigned s = 0; s < kSnaps; ++s) {
        PoisonBuffers(rd);                       // a no-op Get must fail, not pass
        rd.Rearm(tags[s]);
        RdReadSyncKernel<<<kChunks, kThreads>>>(rd.Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        ctp::GpuApi::Synchronize();
        rd.ThrowIfIoFailed("snapshot_read(sync)");

        std::vector<std::vector<byte_t>> got(kChunks, std::vector<byte_t>(kChunkBytes));
        for (unsigned c = 0; c < kChunks; ++c)
            REQUIRE(cudaMemcpy(got[c].data(), rd.DeviceData(c), kChunkBytes,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
        const uint64_t bad = VerifySnap(got, s);
        if (bad) {
            // ThrowIfIoFailed only inspects PUTs -- a failed Get is silent -- so surface the
            // GET return code and the actual bytes before concluding anything.
            std::fprintf(stderr,
                "[read] SYNC snap=%u mismatched bytes=%llu  GetStatus(c0)=%d "
                "got=%02x %02x %02x %02x  expect=%02x %02x %02x %02x  (poison=%02x)\n",
                s, (unsigned long long)bad, rd.GetStatus(0),
                (unsigned)got[0][0], (unsigned)got[0][1], (unsigned)got[0][2], (unsigned)got[0][3],
                (unsigned)RdSnapByte(s, 0, 0), (unsigned)RdSnapByte(s, 0, 1),
                (unsigned)RdSnapByte(s, 0, 2), (unsigned)RdSnapByte(s, 0, 3),
                (unsigned)kPoison);
        }
        bad_total += bad;
    }
    std::fprintf(stderr, "[read] SYNC path: snaps=%u chunks=%u -> %s\n",
                 kSnaps, kChunks, bad_total == 0 ? "OK" : "MISMATCH");
    REQUIRE(bad_total == 0);
}

TEST_CASE("GPU snapshot read-back: ASYNC pipelined Get over 2 reused groups",
          "[integration][gpu][cte][read]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    auto tags = MakeSnapTags("kvhdf5_read_async/snap_", kSnaps);
    WriteSnapshots(ipc, gpu_info, tags);

    // Device-resident tag table: the fire kernel stamps its own destination tag.
    clio::cte::core::TagId* d_tags = nullptr;
    REQUIRE(cudaMalloc(&d_tags, kSnaps * sizeof(clio::cte::core::TagId)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tags, tags.data(), kSnaps * sizeof(clio::cte::core::TagId),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    std::vector<kvhdf5::GpuCteDataset> groups;
    groups.reserve(kGroups);
    for (unsigned g = 0; g < kGroups; ++g)
        groups.emplace_back(ipc, gpu_info, /*gpu_id=*/0, tags[g], RdLayout());
    for (auto& g : groups) PoisonBuffers(g);

    // Per-snapshot host landing buffers, PINNED. This is load-bearing, not a nicety:
    // cudaMemcpyAsync into PAGEABLE memory is actually SYNCHRONOUS, so it blocks the host
    // mid-pipeline while the drain kernel is spinning on the GPU waiting for the CPU server
    // -- and the server must itself issue a device copy to satisfy the Get. Host blocked +
    // GPU spinning + server unable to make progress = deadlock (observed: a hang at 100% GPU).
    // Pinned memory keeps the D2H genuinely asynchronous, so the host never blocks in the loop.
    byte_t* h_land = nullptr;
    const size_t kLandBytes = size_t(kSnaps) * kChunks * kChunkBytes;
    REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&h_land), kLandBytes) == cudaSuccess);
    auto land = [&](unsigned s, unsigned c) {
        return h_land + (size_t(s) * kChunks + c) * kChunkBytes;
    };

    // Prime, then: fire s+1 BEFORE draining/copying s. This is the ordering that would
    // expose a re-tag race or a missing drain-before-refill.
    RdReadFireKernel<<<kChunks, kThreads>>>(groups[0].Handle(), d_tags, 0);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    for (unsigned s = 0; s < kSnaps; ++s) {
        if (s + 1 < kSnaps) {
            RdReadFireKernel<<<kChunks, kThreads>>>(
                groups[(s + 1) % kGroups].Handle(), d_tags, s + 1);
            REQUIRE(cudaGetLastError() == cudaSuccess);
        }
        kvhdf5::GpuCteDataset& g = groups[s % kGroups];
        RdReadDrainKernel<<<kChunks, 32>>>(g.Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        for (unsigned c = 0; c < kChunks; ++c)
            REQUIRE(cudaMemcpyAsync(land(s, c), g.DeviceData(c), kChunkBytes,
                                    cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    ctp::GpuApi::Synchronize();
    for (auto& g : groups) g.ThrowIfIoFailed("snapshot_read(async)");

    uint64_t bad_total = 0;
    for (unsigned s = 0; s < kSnaps; ++s) {
        uint64_t bad = 0;
        for (unsigned c = 0; c < kChunks; ++c) {
            const byte_t* p = land(s, c);
            for (uint64_t i = 0; i < kChunkBytes; ++i)
                if (p[i] != RdSnapByte(s, c, i)) ++bad;
        }
        if (bad) std::fprintf(stderr, "[read] ASYNC snap=%u mismatched bytes=%llu\n",
                              s, (unsigned long long)bad);
        bad_total += bad;
    }
    std::fprintf(stderr,
                 "[read] ASYNC pipelined: snaps=%u groups=%u chunks=%u -> %s\n",
                 kSnaps, kGroups, kChunks, bad_total == 0 ? "OK" : "MISMATCH");
    cudaFree(d_tags);
    cudaFreeHost(h_land);
    REQUIRE(bad_total == 0);
}

#endif  // !CTP_IS_DEVICE_PASS
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
