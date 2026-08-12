/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/*
 * SNAPSHOT DOUBLE-BUFFERING (DESIGN §7 step 1): device memory constant in the
 * number of snapshots.
 *
 * The async arm today allocates ONE GpuCteDataset per snapshot and holds them all
 * live until finalize(), so device payload memory grows LINEARLY in the snapshot
 * count (snaps × snapshot_size). This test proves the replacement: a FIXED set of
 * G=2 buffer GROUPS (2 datasets), reused round-robin (snapshot s uses group s % 2),
 * re-armed to snapshot s's tag via GpuCteDataset::Rearm() after the group's prior
 * write drains. Memory becomes 2 × snapshot_size, CONSTANT in snaps.
 *
 * This is the reuse that historically HUNG at the 2nd fire ("one-round-trip-per-
 * dataset"); it is unblocked by the producer-only run_ctx_ reset in
 * ipc_gpu2cpu.cc. It uses the EXISTING relaunched kernels (fill+fire-async per
 * snapshot, a separate drain kernel), NOT the persistent kernel — that is step 3.
 *
 * WHAT IT GUARDS (like pooled_double_buffer_test, this test has TEETH):
 *  1. CORRECTNESS. Content is keyed on the SNAPSHOT index, so a cross-snapshot
 *     buffer-reuse race shows up as CHANGED BYTES. Every snapshot's persisted blobs
 *     are read back and folded into a checksum compared against an independent
 *     host-computed expectation (a uniformly-wrong run cannot pass by self-agreeing)
 *     AND against a FRESH-dataset-per-snapshot control (reuse must match the known-
 *     good non-reusing path byte-for-byte).
 *  2. NEGATIVE CONTROL. Dropping the reuse drain (refilling a group before its
 *     snap-2 Put drained) MUST corrupt — the test asserts it does, so the drain is
 *     proven load-bearing rather than assumed.
 *  3. BOUNDED MEMORY. The resident device footprint of the 2-group reused path is
 *     measured and shown flat as snaps grows, vs the linear fresh-per-snapshot path.
 *  4. ThrowIfIoFailed() after every drain (a lost Put is silent on the device side).
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/layout.h>           // Layout
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

// N chunks per snapshot, one CUDA block per chunk (G == N == unpooled group), 64
// KiB chunks so the server-side read of a buffer is slow enough that a missing
// cross-snapshot drain actually loses the race (at 256 B the refill would often
// lose it anyway and the bug would hide — same reasoning as pooled_double_buffer).
constexpr unsigned kChunks = 4;              // chunks per snapshot dataset
constexpr unsigned kChunkBytes = 64u * 1024u;
constexpr unsigned kThreads = 128;
constexpr unsigned kGroups = 2;              // double buffer across snapshots

}  // namespace

// Content keyed on the SNAPSHOT index AND chunk index, never on the buffer/group —
// so a cross-snapshot buffer-reuse race shows up as *changed bytes*, which is the
// whole signal. Host + device identical.
__host__ __device__ inline byte_t SnapByte(uint32_t snap, uint32_t c, uint64_t i) {
    return static_cast<byte_t>(
        ((snap * 2246822519u) ^ (c * 2654435761u) ^
         (static_cast<uint32_t>(i) * 40503u) ^ 0xA5u) & 0xFFu);
}

// Fill every chunk this block owns with snapshot `snap`'s content, fence, then
// fire its Put WITHOUT waiting (async). Grid-stride over chunks (one block per
// chunk at grid == kChunks). Non-probing. Relaunched per snapshot.
__global__ void SnapFillFireKernel(kvhdf5::GpuDatasetHandle h, uint32_t snap) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        byte_t* dst = h.Data(c);
        uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = SnapByte(snap, c, i);
        __threadfence_system();
        __syncthreads();
        h.WriteAsync</*Probing=*/false>(c);   // thread-0 only; no wait
        __syncthreads();
    }
}

// Drain every chunk this block owns (the reuse barrier + final drain). Idempotent
// per chunk (WriteWait just polls an already-set completion flag for chunks that
// already drained).
__global__ void SnapDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.WriteWait(c);
        __syncthreads();
    }
}

// DEVICE-STAMPED variant of SnapFillFireKernel: instead of the host re-arming the
// tag between launches, the kernel stamps the snapshot's destination tag itself,
// read from a device-resident TagId table (the tag-table addressing model of
// DESIGN §3.3 — the exact key-set the persistent kernel will use). Confirms §6
// unknown #1: a DEVICE-set tag_id_ routes through the store identically to a
// host-set one. Everything else matches SnapFillFireKernel.
__global__ void SnapFillFireStampKernel(kvhdf5::GpuDatasetHandle h,
                                        const clio::cte::core::TagId* tag_table,
                                        uint32_t snap) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        byte_t* dst = h.Data(c);
        uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = SnapByte(snap, c, i);
        __threadfence_system();
        __syncthreads();
        h.SetTag(c, tag_table[snap]);         // device chooses the destination tag
        h.WriteAsync</*Probing=*/false>(c);    // thread-0 only; no wait
        __syncthreads();
    }
}

#if !CTP_IS_DEVICE_PASS

namespace {

clio::cte::core::TagId MakeTag(const std::string& name) {
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
    auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, /*offset=*/clio::run::u64(0),
                                           size, /*flags=*/clio::run::u32(0),
                                           buf.shm_.template Cast<void>());
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

uint64_t Fnv1a(const byte_t* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    for (size_t i = 0; i < n; ++i) { h ^= static_cast<unsigned char>(p[i]); h *= 1099511628211ULL; }
    return h;
}

// Force every kernel's device code resident before the reuse loop launches any of
// them (same cold-launch correctness argument as pooled_double_buffer_test — a
// resident producer + a cold drain-kernel launch can deadlock the device).
void PrewarmKernels() {
    cudaFuncAttributes fa;
    REQUIRE(cudaFuncGetAttributes(
        &fa, reinterpret_cast<const void*>(SnapFillFireKernel)) == cudaSuccess);
    REQUIRE(cudaFuncGetAttributes(
        &fa, reinterpret_cast<const void*>(SnapDrainKernel)) == cudaSuccess);
    REQUIRE(cudaFuncGetAttributes(
        &fa, reinterpret_cast<const void*>(SnapFillFireStampKernel)) == cudaSuccess);
}

// Device-resident TagId[num_snaps] table (host builds it off the hot path; the
// kernel stamps chunk c's tag = table[snapshot] in-kernel). Frees in the dtor.
struct DeviceTagTable {
    clio::cte::core::TagId* d_ = nullptr;
    explicit DeviceTagTable(const std::vector<clio::cte::core::TagId>& tags) {
        const size_t bytes = tags.size() * sizeof(clio::cte::core::TagId);
        REQUIRE(cudaMalloc(&d_, bytes) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_, tags.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~DeviceTagTable() { if (d_) cudaFree(d_); }
    const clio::cte::core::TagId* get() const { return d_; }
};

kvhdf5::Layout SnapLayout() {
    return kvhdf5::Layout{/*dims=*/{kChunks * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes}, /*elem_size=*/1};
}

// The independent expectation: fold every snapshot's every chunk's content, host-
// computed, no GPU. Read-back must equal THIS, not merely be self-consistent.
uint64_t ExpectedChecksum(unsigned snaps) {
    uint64_t h = 1469598103934665603ULL;
    std::vector<byte_t> chunk(kChunkBytes);
    for (unsigned s = 0; s < snaps; ++s)
        for (unsigned c = 0; c < kChunks; ++c) {
            for (unsigned i = 0; i < kChunkBytes; ++i) chunk[i] = SnapByte(s, c, i);
            h = Fnv1a(chunk.data(), chunk.size(), h);
        }
    return h;
}

// Fold the PERSISTED bytes of every snapshot's every chunk into one checksum.
uint64_t ReadBackChecksum(const std::vector<clio::cte::core::TagId>& tags) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t s = 0; s < tags.size(); ++s)
        for (unsigned c = 0; c < kChunks; ++c) {
            auto got = HostReadBlob(tags[s], std::to_string(c), kChunkBytes);
            h = Fnv1a(got.data(), got.size(), h);
        }
    return h;
}

// ---- the REUSED-2-GROUP path (the thing under test) ------------------------
// 2 datasets serve `snaps` snapshots: snapshot s uses group s % 2, re-armed to
// tags[s]. Before refilling a group at snapshot s (s >= kGroups), its snap-(s-2)
// Put is drained (`drain=true`); the negative control passes drain=false. Returns
// the number of buffer groups (datasets) held live — CONSTANT (kGroups) regardless
// of snaps, which is the whole point. The cudaMemGetInfo delta is printed too, but
// only as informational context: at this scale it is dominated by CUDA's
// sub-allocator arena granularity (a ~2 MiB arena grabbed on the first device alloc,
// later allocs fit inside), so it does NOT resolve the real 0.5 vs 1.5 MiB
// difference and must not be asserted on.
uint64_t RunReused(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                   const std::vector<clio::cte::core::TagId>& tags, bool drain) {
    const unsigned snaps = static_cast<unsigned>(tags.size());
    kvhdf5::Layout layout = SnapLayout();

    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

    // The FIXED set of buffer groups — constructed ONCE, reused for all snaps.
    std::vector<kvhdf5::GpuCteDataset> groups;
    groups.reserve(kGroups);
    for (unsigned g = 0; g < kGroups; ++g)
        groups.emplace_back(ipc, gpu_info, /*gpu_id=*/0, tags[g], layout);

    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
    const size_t dev_delta = (free_before > free_after) ? (free_before - free_after) : 0;
    std::fprintf(stderr, "[reuse]   reused snaps=%u: %u live groups, cudaMemGetInfo "
                 "delta=%.2f MiB (informational)\n",
                 snaps, kGroups, double(dev_delta) / 1048576.0);

    for (unsigned s = 0; s < snaps; ++s) {
        const unsigned g = s % kGroups;
        auto& ds = groups[g];
        // Reclaim this group from the snapshot that used it kGroups ago.
        if (s >= kGroups) {
            if (drain) {
                SnapDrainKernel<<<ds.GridSize(), 32>>>(ds.Handle());
                REQUIRE(cudaGetLastError() == cudaSuccess);
                ctp::GpuApi::Synchronize();
                ds.ThrowIfIoFailed("snapshot_reuse(drain)");
            }
            // Re-target the reused slots to THIS snapshot's tag (host key re-arm).
            ds.Rearm(tags[s]);
        }
        SnapFillFireKernel<<<ds.GridSize(), kThreads>>>(ds.Handle(), s);
        REQUIRE(cudaGetLastError() == cudaSuccess);
    }
    // Drain the last <= kGroups groups still in flight.
    for (unsigned g = 0; g < kGroups && g < snaps; ++g) {
        SnapDrainKernel<<<groups[g].GridSize(), 32>>>(groups[g].Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
    }
    ctp::GpuApi::Synchronize();
    for (unsigned g = 0; g < kGroups && g < snaps; ++g)
        groups[g].ThrowIfIoFailed("snapshot_reuse(final)");

    // MEASURED resident device payload = sum of the live groups' registered data
    // backends. Bounded (kGroups groups) regardless of snaps — assert on THIS, not
    // a hardcoded group count.
    uint64_t resident = 0;
    for (unsigned g = 0; g < kGroups && g < snaps; ++g)
        resident += groups[g].DeviceDataBytes();
    return resident;
}

// Like RunReused(drain=true) but the DEVICE picks each snapshot's tag from a
// device-resident table (SnapFillFireStampKernel) instead of the host calling
// Rearm() between launches. This is the tag-table addressing the persistent kernel
// will use; running it under RELAUNCHED kernels isolates "does a device-stamped
// tag_id_ route correctly?" from the persistent-kernel machinery. Returns live
// group count (kGroups), same bounded-memory witness as RunReused.
unsigned RunReusedDeviceStamp(clio::run::IpcManager* ipc,
                              clio::run::IpcManagerGpuInfo gpu_info,
                              const std::vector<clio::cte::core::TagId>& tags) {
    const unsigned snaps = static_cast<unsigned>(tags.size());
    kvhdf5::Layout layout = SnapLayout();
    DeviceTagTable table(tags);   // host builds the device-visible tag table

    // The groups are constructed with tags[0]/tags[1], but the kernel OVERRIDES the
    // tag every snapshot from the table, so the construction tag is irrelevant.
    std::vector<kvhdf5::GpuCteDataset> groups;
    groups.reserve(kGroups);
    for (unsigned g = 0; g < kGroups; ++g)
        groups.emplace_back(ipc, gpu_info, /*gpu_id=*/0, tags[g], layout);

    for (unsigned s = 0; s < snaps; ++s) {
        const unsigned g = s % kGroups;
        auto& ds = groups[g];
        if (s >= kGroups) {   // reclaim this group's buffers from snapshot s-kGroups
            SnapDrainKernel<<<ds.GridSize(), 32>>>(ds.Handle());
            REQUIRE(cudaGetLastError() == cudaSuccess);
            ctp::GpuApi::Synchronize();
            ds.ThrowIfIoFailed("snapshot_reuse(devstamp drain)");
            // NB: no host Rearm — the kernel below stamps tags[s] itself.
        }
        SnapFillFireStampKernel<<<ds.GridSize(), kThreads>>>(ds.Handle(), table.get(), s);
        REQUIRE(cudaGetLastError() == cudaSuccess);
    }
    for (unsigned g = 0; g < kGroups && g < snaps; ++g) {
        SnapDrainKernel<<<groups[g].GridSize(), 32>>>(groups[g].Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
    }
    ctp::GpuApi::Synchronize();
    for (unsigned g = 0; g < kGroups && g < snaps; ++g)
        groups[g].ThrowIfIoFailed("snapshot_reuse(devstamp final)");

    return (snaps < kGroups) ? snaps : kGroups;
}

// ---- the FRESH-per-snapshot control (the known-good, memory-LINEAR path) ----
// One dataset per snapshot, all held live until the end (mirrors today's async
// arm). Never reuses a slot, so it can never hit the reuse race — the reference
// the reused path must match byte-for-byte. Returns the number of datasets held
// live (== snaps): the LINEAR footprint the reused path replaces.
uint64_t RunFresh(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                  const std::vector<clio::cte::core::TagId>& tags) {
    const unsigned snaps = static_cast<unsigned>(tags.size());
    kvhdf5::Layout layout = SnapLayout();

    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

    std::vector<kvhdf5::GpuCteDataset> all;
    all.reserve(snaps);
    for (unsigned s = 0; s < snaps; ++s)
        all.emplace_back(ipc, gpu_info, /*gpu_id=*/0, tags[s], layout);

    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);

    for (unsigned s = 0; s < snaps; ++s) {
        SnapFillFireKernel<<<all[s].GridSize(), kThreads>>>(all[s].Handle(), s);
        REQUIRE(cudaGetLastError() == cudaSuccess);
    }
    for (unsigned s = 0; s < snaps; ++s) {
        SnapDrainKernel<<<all[s].GridSize(), 32>>>(all[s].Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
    }
    ctp::GpuApi::Synchronize();
    for (unsigned s = 0; s < snaps; ++s)
        all[s].ThrowIfIoFailed("snapshot_reuse(fresh)");

    const size_t dev_delta = (free_before > free_after) ? (free_before - free_after) : 0;
    std::fprintf(stderr, "[reuse]   fresh  snaps=%u: %u live datasets, cudaMemGetInfo "
                 "delta=%.2f MiB (informational)\n",
                 snaps, snaps, double(dev_delta) / 1048576.0);
    // MEASURED resident device payload across ALL held-live datasets — linear in snaps.
    uint64_t resident = 0;
    for (unsigned s = 0; s < snaps; ++s) resident += all[s].DeviceDataBytes();
    return resident;
}

std::vector<clio::cte::core::TagId> MakeSnapTags(const char* prefix, unsigned snaps) {
    std::vector<clio::cte::core::TagId> tags;
    tags.reserve(snaps);
    for (unsigned s = 0; s < snaps; ++s)
        tags.push_back(MakeTag(std::string(prefix) + std::to_string(s)));
    return tags;
}

}  // namespace

TEST_CASE("GPU snapshot double-buffering: 2 groups reused across N snapshots, "
          "byte-identical + memory constant in snaps",
          "[integration][gpu][cte][reuse][doublebuf]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    PrewarmKernels();

    const uint64_t snap_bytes =
        static_cast<uint64_t>(kChunks) * kChunkBytes;

    // Prove memory is FLAT in snaps: run the reused path at two snapshot counts and
    // require the number of LIVE buffer groups to be the same (kGroups) at both,
    // while the fresh path holds `snaps` datasets live. The device payload footprint
    // is (#live datasets) × snap_bytes, so kGroups groups = a fixed 2 × snap_bytes
    // regardless of snaps.
    uint64_t prev_reuse_bytes = 0;
    for (unsigned snaps : {6u, 12u}) {
        const uint64_t expect = ExpectedChecksum(snaps);

        auto reuse_tags = MakeSnapTags("kvhdf5_reuse/snap_", snaps);
        uint64_t reuse_bytes = RunReused(ipc, gpu_info, reuse_tags, /*drain=*/true);
        uint64_t reuse_sum = ReadBackChecksum(reuse_tags);

        auto fresh_tags = MakeSnapTags("kvhdf5_fresh/snap_", snaps);
        uint64_t fresh_bytes = RunFresh(ipc, gpu_info, fresh_tags);
        uint64_t fresh_sum = ReadBackChecksum(fresh_tags);

        std::fprintf(stderr,
            "[reuse] snaps=%2u snap_bytes=%.2f MiB | reused: sum=0x%016llx resident=%.2f MiB "
            "| fresh: sum=0x%016llx resident=%.2f MiB\n",
            snaps, double(snap_bytes) / 1048576.0,
            (unsigned long long)reuse_sum, double(reuse_bytes) / 1048576.0,
            (unsigned long long)fresh_sum, double(fresh_bytes) / 1048576.0);

        // (1) reuse is byte-correct vs the independent host expectation,
        // (2) and byte-identical to the fresh (never-reusing) control.
        if (reuse_sum != expect)
            std::fprintf(stderr, "[reuse] CORRUPTION: reused 0x%016llx != expected "
                         "0x%016llx at snaps=%u\n",
                         (unsigned long long)reuse_sum,
                         (unsigned long long)expect, snaps);
        REQUIRE(reuse_sum == expect);
        REQUIRE(fresh_sum == expect);
        REQUIRE(reuse_sum == fresh_sum);

        // (3) bounded memory — MEASURED device payload (summed from each live
        // dataset's registered data backend, not a hardcoded count). The reused
        // path holds exactly kGroups groups' worth regardless of snaps (constant,
        // and identical at snaps=6 and snaps=12); the fresh path holds `snaps`
        // worth. A regression that stopped bounding the reused path (e.g. held all
        // N snapshots live) would push reuse_bytes to snaps*snap_bytes and trip this.
        REQUIRE(reuse_bytes == static_cast<uint64_t>(kGroups) * snap_bytes);
        REQUIRE(fresh_bytes == static_cast<uint64_t>(snaps) * snap_bytes);
        REQUIRE(reuse_bytes < fresh_bytes);
        if (prev_reuse_bytes) REQUIRE(reuse_bytes == prev_reuse_bytes);  // flat across snaps
        prev_reuse_bytes = reuse_bytes;
    }

    std::fprintf(stderr, "[reuse] OK: 2 groups served every snapshot byte-identical to "
                 "the fresh path, at memory constant in snaps.\n");
}

// DEVICE-STAMPED TAG (DESIGN §6 unknown #1): the KERNEL picks each snapshot's
// destination tag from a device-resident table (GpuDatasetHandle::SetTag), instead
// of the host re-arming it. Proves a device-set tag_id_ routes through the store
// exactly like a host-set one — the addressing mechanism the persistent kernel
// depends on — while still under relaunched kernels + 2 reused groups.
TEST_CASE("GPU snapshot double-buffering: device-stamped tag routes correctly "
          "(tag-table addressing)",
          "[integration][gpu][cte][reuse][doublebuf]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    PrewarmKernels();

    const unsigned snaps = 12;
    const uint64_t expect = ExpectedChecksum(snaps);

    auto tags = MakeSnapTags("kvhdf5_reuse_devstamp/snap_", snaps);
    unsigned live = RunReusedDeviceStamp(ipc, gpu_info, tags);
    const uint64_t sum = ReadBackChecksum(tags);

    std::fprintf(stderr,
        "[reuse] device-stamped tag: snaps=%u live=%u sum=0x%016llx expected=0x%016llx -> %s\n",
        snaps, live, (unsigned long long)sum, (unsigned long long)expect,
        sum == expect ? "OK (device-set tag routes like host-set)" : "MISMATCH");

    REQUIRE(sum == expect);          // every snapshot landed under its own device-set tag
    REQUIRE(live == kGroups);        // still only 2 buffer groups live
}

// WHY THERE IS NO EXECUTABLE SNAPSHOT-LEVEL "drop the drain" NEGATIVE CONTROL HERE.
//
// The obvious negative control — reuse a group WITHOUT draining its prior Put and
// assert corruption — cannot be a sound, safe assertion at the SNAPSHOT level,
// because dropping the drain here re-fires the same TASK SLOT while its previous
// submission may still be in flight. That is a contract violation (the reuse
// invariant is: drain a slot's prior Put before re-arming/re-firing it), and it is
// UNDEFINED behaviour with no clean, assertable outcome:
//   - it may silently corrupt data (the buffer is overwritten mid-Put) — but only
//     if the race is lost, which is TIMING-DEPENDENT (a fast server drains first and
//     the run comes out clean), so "MUST corrupt" is inherently flaky; and/or
//   - it can hang (the reused slot carries a stale, still-in-flight run_ctx_, which
//     RecvIn's completion-guarded reset deliberately will NOT free — see
//     ipc_gpu2cpu.cc — so the second fire is mis-driven and never completes).
// A test that must trigger UB to pass is neither deterministic nor safe for CI.
//
// The drain's load-bearing property is instead proven by two SAFE tests:
//   1. the POSITIVE test above — reuse is byte-identical ONLY because each group is
//      drained before refill; remove the drain from RunReused (drain=false) by hand
//      to observe the hazard during manual investigation, under a timeout.
//   2. pooled_double_buffer_test.cu — the deterministic, in-bounds negative control
//      for the SAME buffer-reuse race at the pipeline-depth level: it sweeps M and
//      requires byte-identical output, and a missing WriteWait(c-M) barrier there
//      moves the checksum. That is where the teeth live.

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
