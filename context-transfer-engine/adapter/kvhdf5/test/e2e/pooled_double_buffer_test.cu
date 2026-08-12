/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/*
 * Bounded-pool double buffering: a per-block pipeline-depth knob.
 *
 * THIS TEST GUARDS AGAINST SILENT DATA CORRUPTION. With N chunks streaming
 * through M < N resident data buffers, chunk c and chunk c+M share one buffer.
 * If the kernel refills that buffer before chunk c's PutBlob has finished
 * draining on the CPU side, chunk c's blob silently persists chunk c+M's bytes
 * (or a torn mix of both). Nothing errors. The only way to see it is to read the
 * persisted blobs back and compare.
 *
 * So: fix N and G, sweep the pipeline depth D = M/G over {1, 2, 4, ...} up to
 * N/G (i.e. M from G to N), produce the SAME deterministic content every time
 * through the ONE fused GpuDatasetHandle::WritePipelined kernel, and require the
 * read-back checksum to be BYTE-IDENTICAL across every M — and equal to the
 * independently host-computed expectation, so a uniformly-wrong run cannot pass
 * by agreeing with itself.
 *
 *   D == 1      (M == G): synchronous. Every block drains its previous chunk
 *                         before refilling. The reference behaviour.
 *   D == N/G    (M == N): fire-all. No in-loop wait; everything drains in the
 *                         tail. Today's async arm.
 *   1 < D < N/G          : the actual double-buffered pipeline. This is the range
 *                         where a missing barrier corrupts.
 * All three run through the same code path (see WritePipelined) — that is the
 * point of the design: the endpoints must collapse for free.
 *
 * A differing checksum at any M is a buffer-reuse race, NOT a test bug.
 *
 * Also asserted after every run: ThrowIfIoFailed(). A failed Put is invisible on
 * the device side (the submit path discards the return code), so a checksum test
 * that skips this can "pass" while dropping every single write.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/dataset_meta.h>  // Layout
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

// N chunks over G blocks. N/G == 8, so the depth sweep is D in {1,2,4,8} and
// M = D*G in {4,8,16,32} — from "one buffer per block" all the way to "one buffer
// per chunk". 64 KiB chunks keep the server-side read of a buffer slow enough
// that a missing barrier actually loses the race (at 256 B the refill would often
// lose it anyway and the bug would hide).
constexpr unsigned kChunks = 32;         // N
constexpr unsigned kGrid = 4;            // G
constexpr unsigned kChunkBytes = 64u * 1024u;
constexpr unsigned kThreads = 128;

}  // namespace

// Deterministic content keyed on the CHUNK index, never on the buffer index — a
// buffer-reuse race therefore shows up as *changed bytes*, which is the whole
// signal this test is built around.
__host__ __device__ inline byte_t PdbByte(uint32_t c, uint64_t i) {
    return static_cast<byte_t>(
        ((c * 2654435761u) ^ (static_cast<uint32_t>(i) * 40503u) ^ 0x5Au) & 0xFFu);
}

// Block-wide fill of one chunk's buffer. A plain functor rather than a device
// lambda so the file needs no --extended-lambda.
struct PdbFill {
    __device__ void operator()(uint32_t c, byte_t* dst, uint64_t n) const {
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = PdbByte(c, i);
    }
};

/**
 * The ONE fused pipelined producer. Identical for every M — the pool size and
 * the grid ride in on the handle, and WritePipelined derives the depth and the
 * wait schedule from them. Non-probing instantiation (the submit probe's
 * registers are pure overhead here).
 */
__global__ void PdbPipelinedKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    h.WritePipelined</*Probing=*/false>(PdbFill{});
}

/**
 * The DEFERRED-DRAIN producer: identical, but TailDrain=false leaves the last <= D Puts
 * per block in flight past kernel exit, for the caller to drain separately.
 *
 * This is the PRODUCTION shape — the Gray-Scott bench's pooled arm runs exactly this, so
 * that a snapshot's trailing I/O overlaps the NEXT snapshot's compute instead of pinning the
 * kernel (an in-kernel tail drain made the pooled arm perform like the synchronous one).
 * It is therefore the shape that MUST be proven not to corrupt: the in-loop WriteWait(c-M)
 * is what enforces the buffer-reuse invariant, and the claim is that the tail wait was never
 * load-bearing for correctness. If that claim is wrong, the checksum moves.
 */
__global__ void PdbPipelinedDeferredKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    h.WritePipelined</*Probing=*/false, /*TailDrain=*/false>(PdbFill{});
}

/** Drains everything the deferred producer left in flight. Idempotent per chunk. */
__global__ void PdbDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    h.WriteDrainAll();
}

#if !CTP_IS_DEVICE_PASS

namespace {

clio::cte::core::TagId MakeTag(const std::string& name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

// Read one persisted blob back off the CTE server (the GPU cannot Get all N
// chunks back — with M < N there aren't N buffers to land them in).
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

/** FNV-1a over a byte range, chained so it can fold N chunks in order. */
uint64_t Fnv1a(const byte_t* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<unsigned char>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

/**
 * Force every producer/drain kernel's device code RESIDENT before any of them is launched.
 *
 * This is a CORRECTNESS requirement for the deferred-drain path, not a warm-up nicety.
 * CUDA 12 loads a kernel's module on its FIRST launch, and that load is a DEVICE-WIDE
 * SYNCHRONIZING driver operation. The deferred path launches the drain kernel WHILE the
 * producer is still resident and spinning in WriteWait — and that producer can only make
 * forward progress if the CPU-side CTE server completes its Puts, which requires the server
 * to issue its own CUDA calls. A cold drain-kernel launch blocks exactly those calls, so the
 * producer never advances and the drain never starts: a REAL DEADLOCK, not a slowdown.
 *
 * (Verified: without this, D=1 deferred hangs indefinitely; inserting a Synchronize between
 * the two launches also "fixes" it, by retiring the producer before the cold load — but that
 * sync is precisely the overlap the deferral exists to buy, so pre-warming is the right fix.
 * The Gray-Scott bench pre-warms for the same reason; see GSBENCH_PREWARM.)
 */
void PrewarmKernels() {
    cudaFuncAttributes fa;
    REQUIRE(cudaFuncGetAttributes(
        &fa, reinterpret_cast<const void*>(PdbPipelinedKernel)) == cudaSuccess);
    REQUIRE(cudaFuncGetAttributes(
        &fa, reinterpret_cast<const void*>(PdbPipelinedDeferredKernel)) == cudaSuccess);
    REQUIRE(cudaFuncGetAttributes(
        &fa, reinterpret_cast<const void*>(PdbDrainKernel)) == cudaSuccess);
}

struct RunResult {
    uint64_t checksum;
    size_t dev_bytes;      // cudaMemGetInfo delta across dataset construction
    uint64_t buf_bytes;    // computed resident-buffer figure: M * chunk_bytes
};

/**
 * Build a dataset at (N=kChunks, M=pool, G=kGrid), produce every chunk through
 * the fused pipelined kernel, assert no Put was lost, and fold the *persisted*
 * bytes into a checksum.
 */
RunResult RunDepth(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                   unsigned pool, unsigned depth, bool deferred = false) {
    const std::string tag_name = "kvhdf5_pdb_m" + std::to_string(pool) +
                                 (deferred ? "_deferred" : "");
    clio::cte::core::TagId tag = MakeTag(tag_name);

    kvhdf5::Layout layout{/*dims=*/{kChunks * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes}, /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == kChunks);

    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, tag, layout,
                             /*pool_size=*/pool,
                             kvhdf5::GpuCteDataset::MemKind::kDeviceMem,
                             /*grid_size=*/kGrid);

    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);

    REQUIRE(ds.ChunkCount() == kChunks);
    REQUIRE(ds.PoolSize() == pool);
    REQUIRE(ds.GridSize() == kGrid);
    REQUIRE(ds.Depth() == depth);

    if (deferred) {
        // The production shape: the producer RETIRES with its trailing <= D Puts still in
        // flight and a SEPARATE kernel drains them. Deliberately NOT synchronized between the
        // two launches — a sync here would defeat the point (the deferral is what lets the
        // next snapshot's compute overlap this snapshot's I/O).
        //
        // This is only safe because both kernels are already resident: see PrewarmKernels.
        PdbPipelinedDeferredKernel<<<ds.GridSize(), kThreads>>>(ds.Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        PdbDrainKernel<<<ds.GridSize(), 32>>>(ds.Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
    } else {
        PdbPipelinedKernel<<<ds.GridSize(), kThreads>>>(ds.Handle());
        REQUIRE(cudaGetLastError() == cudaSuccess);
    }
    ctp::GpuApi::Synchronize();

    // A lost Put is SILENT on the device side. Without this, every checksum below
    // could be folded from stale/absent blobs and the test would still be green.
    ds.ThrowIfIoFailed("pooled_double_buffer");
    REQUIRE(ds.FirstFailedPut() == -1);

    uint64_t h = 1469598103934665603ULL;
    for (unsigned c = 0; c < kChunks; ++c) {
        std::vector<byte_t> got = HostReadBlob(tag, std::to_string(c), kChunkBytes);
        h = Fnv1a(got.data(), got.size(), h);
    }

    RunResult r;
    r.checksum = h;
    r.dev_bytes = (free_before > free_after) ? (free_before - free_after) : 0;
    r.buf_bytes = static_cast<uint64_t>(pool) * kChunkBytes;
    return r;
}

}  // namespace

TEST_CASE("GPU bounded-pool double buffering: byte-identical across every depth",
          "[integration][gpu][cte][pool][doublebuf]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    PrewarmKernels();   // MANDATORY for the deferred path — see PrewarmKernels

    // The independent expectation: what the content MUST be, computed on the host
    // with no GPU involved. Requiring only "all M agree" would pass a run that is
    // uniformly wrong.
    uint64_t expect = 1469598103934665603ULL;
    {
        std::vector<byte_t> chunk(kChunkBytes);
        for (unsigned c = 0; c < kChunks; ++c) {
            for (unsigned i = 0; i < kChunkBytes; ++i) chunk[i] = PdbByte(c, i);
            expect = Fnv1a(chunk.data(), chunk.size(), expect);
        }
    }

    std::fprintf(stderr,
                 "\n[pdb] N=%u chunks x %u B, G=%u blocks, %u threads/block\n"
                 "[pdb] %-4s %-4s %-18s %-14s %-14s\n",
                 kChunks, kChunkBytes, kGrid, kThreads,
                 "D", "M", "checksum", "resident-buf", "dev-alloc");

    bool have_ref = false;
    uint64_t ref = 0;
    unsigned ref_m = 0;

    for (unsigned depth = 1; depth * kGrid <= kChunks; depth *= 2) {
        const unsigned pool = depth * kGrid;  // M = D * G
        RunResult r = RunDepth(ipc, gpu_info, pool, depth);

        std::fprintf(stderr, "[pdb] %-4u %-4u 0x%016llx %10.2f MiB %10.2f MiB%s\n",
                     depth, pool, static_cast<unsigned long long>(r.checksum),
                     static_cast<double>(r.buf_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(r.dev_bytes) / (1024.0 * 1024.0),
                     (pool == kGrid) ? "   (D=1: synchronous)"
                                     : (pool == kChunks) ? "   (M=N: fire-all)" : "");

        // vs. the host's independent expectation.
        if (r.checksum != expect)
            std::fprintf(stderr,
                         "[pdb] CORRUPTION at M=%u (D=%u): checksum 0x%016llx != "
                         "expected 0x%016llx. A chunk's buffer was refilled while "
                         "its Put was still draining.\n",
                         pool, depth, static_cast<unsigned long long>(r.checksum),
                         static_cast<unsigned long long>(expect));
        REQUIRE(r.checksum == expect);

        // vs. every other M. Same content, same code path, different pool depth:
        // the bytes MUST NOT move.
        if (have_ref && r.checksum != ref)
            std::fprintf(stderr,
                         "[pdb] M=%u disagrees with M=%u (0x%016llx vs 0x%016llx)\n",
                         pool, ref_m, static_cast<unsigned long long>(r.checksum),
                         static_cast<unsigned long long>(ref));
        if (have_ref) REQUIRE(r.checksum == ref);
        ref = r.checksum;
        ref_m = pool;
        have_ref = true;
    }
    REQUIRE(have_ref);

    // ---- the DEFERRED-DRAIN (TailDrain=false) path, over the same depth sweep ----
    // This is what the bench's pooled arm actually runs. The producer kernel RETIRES with up
    // to D Puts per block still in flight and a separate kernel drains them. If the tail wait
    // was secretly load-bearing for the buffer-reuse invariant, these checksums move.
    std::fprintf(stderr, "[pdb] --- deferred drain (TailDrain=false, production shape) ---\n");
    for (unsigned depth = 1; depth * kGrid <= kChunks; depth *= 2) {
        const unsigned pool = depth * kGrid;
        RunResult r = RunDepth(ipc, gpu_info, pool, depth, /*deferred=*/true);
        std::fprintf(stderr, "[pdb] %-4u %-4u 0x%016llx %10.2f MiB   (deferred)\n",
                     depth, pool, static_cast<unsigned long long>(r.checksum),
                     static_cast<double>(r.buf_bytes) / (1024.0 * 1024.0));
        if (r.checksum != expect)
            std::fprintf(stderr,
                         "[pdb] CORRUPTION at M=%u (D=%u, deferred drain): 0x%016llx != "
                         "0x%016llx. Deferring the tail drain broke the buffer-reuse "
                         "invariant — the in-loop WriteWait(c-M) is NOT sufficient.\n",
                         pool, depth, static_cast<unsigned long long>(r.checksum),
                         static_cast<unsigned long long>(expect));
        REQUIRE(r.checksum == expect);
        REQUIRE(r.checksum == ref);   // identical to the TailDrain=true runs
    }

    std::fprintf(stderr,
                 "[pdb] OK: every pipeline depth produced byte-identical output.\n"
                 "[pdb] resident data buffers scale M*S (D*G*S), not N*S: the D=2\n"
                 "[pdb] double-buffered arm holds %.2f MiB vs %.2f MiB for M=N.\n\n",
                 static_cast<double>(2ull * kGrid * kChunkBytes) / (1024.0 * 1024.0),
                 static_cast<double>(1ull * kChunks * kChunkBytes) / (1024.0 * 1024.0));
}

// The (M, G) contract is enforced at CONSTRUCTION, loudly. A bad pair does not
// degrade — it corrupts an already-submitted chunk's blob with no error anywhere,
// so it must never be constructible in the first place.
TEST_CASE("GPU bounded-pool: illegal (M, G) pairs throw at construction",
          "[integration][gpu][cte][pool][doublebuf]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);

    kvhdf5::Layout layout{/*dims=*/{8u * 256u}, /*chunk_dims=*/{256u},
                          /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == 8u);
    clio::cte::core::TagId tag = MakeTag("kvhdf5_pdb_bad");
    using MK = kvhdf5::GpuCteDataset::MemKind;

    // M > N: more buffers than chunks is a caller mistake, not something to clamp.
    REQUIRE_THROWS(kvhdf5::GpuCteDataset(ipc, gpu_info, 0, tag, layout,
                                         /*pool=*/9, MK::kDeviceMem, /*grid=*/1));
    // G does not divide M: block b's buffer set {b, b+G, ...} would overlap another
    // block's, so two blocks could write the same buffer concurrently.
    REQUIRE_THROWS(kvhdf5::GpuCteDataset(ipc, gpu_info, 0, tag, layout,
                                         /*pool=*/6, MK::kDeviceMem, /*grid=*/4));
    // G > M: some block would own zero buffers.
    REQUIRE_THROWS(kvhdf5::GpuCteDataset(ipc, gpu_info, 0, tag, layout,
                                         /*pool=*/2, MK::kDeviceMem, /*grid=*/4));
    // Legal, and the default (grid==0 => G==M => D==1, the historical contract).
    kvhdf5::GpuCteDataset ok(ipc, gpu_info, 0, tag, layout, /*pool=*/4);
    REQUIRE(ok.GridSize() == 4u);
    REQUIRE(ok.Depth() == 1u);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
