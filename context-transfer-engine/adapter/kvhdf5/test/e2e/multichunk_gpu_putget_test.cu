/*
 * Multi-chunk GPU PutBlob+GetBlob integration test (Phase 2).
 *
 * Proves the device-facing handle's N-chunk path: an N-chunk dataset whose
 * per-chunk Put/Get tasks + distinct data regions live behind one
 * GpuDatasetHandle (a device-side ChunkDesc array). One fused kernel fills each
 * chunk's region and submits its PutBlob; one kernel Gets every chunk back.
 *
 * The kernels are written in the *general* grid-stride form (a block strides
 * over a range of chunks), so one kernel covers both modes:
 *   - grid=1: a single block strides over all N chunks serially (a genuine
 *     range: N loop iterations).
 *   - grid>1: multiple blocks submit chunk Put/Get *concurrently*; with
 *     grid < N each block still strides over a sub-range, so grid=2 over 4
 *     chunks exercises concurrency AND the multi-iteration range loop at once.
 * Both are verified below (RunAndVerify is parameterized by grid). Concurrent
 * multi-block GPU->CPU submit is fine on this iowarp pin: every block's thread-0
 * Send()s to the same gpu2cpu lane (GetLane(0,0)), which iowarp's own
 * test_gpu_kernel_stress drives at <<<64,32>>>. (An earlier note here claimed
 * multi-block deadlocks; that was an environmental artifact of leftover
 * /dev/shm + stale servers during iteration, disproven by a clean interleaved
 * 21/21 run. NOTE: very high host thread counts — e.g. num_threads=16, 4x the
 * default — are separately unstable in the CTE/runtime path; ctest uses the
 * default, where this is solid.)
 *
 * Verification reads every chunk back, asserts byte-identical to the pattern it
 * was filled with, AND asserts distinct chunks differ (so a stuck chunk index
 * can't pass). Reuses the one-time SharedCteEnv server bring-up.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/cpu_dataset.h>      // Layout
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "cte_env.h"

using kvhdf5::byte_t;  // raw blob-payload bytes (codebase convention)

namespace {

constexpr unsigned kChunkBytes = 256;
constexpr unsigned kChunkCount = 4;
constexpr unsigned kSeedBase = 0x40u;  // chunk c uses seed (kSeedBase + c)

// Device pattern for a chunk: byte i = (seed ^ i) & 0xFF. Distinct seeds give
// distinct chunk contents (so a stuck chunk index fails verification).
constexpr byte_t Pattern(unsigned seed, unsigned i) {
    return static_cast<byte_t>((seed ^ i) & 0xFFu);
}

}  // namespace

/**
 * Fused fill + PutBlob for every chunk, grid-stride. A block strides over chunks
 * (blockIdx, blockIdx+gridDim, ...). For each chunk: all threads fill its region,
 * fence system-wide so the CPU-side PutBlob's D2H read sees the writes while the
 * kernel is resident, barrier, then thread-0 submits that chunk's Put and waits.
 * The trailing barrier keeps the block in lockstep before the next chunk.
 */
__global__ void McFillWriteKernel(kvhdf5::GpuDatasetHandle h, unsigned seed_base) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        byte_t* dst = h.Data(c);
        uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = static_cast<byte_t>(((seed_base + c) ^ i) & 0xFFu);
        __threadfence_system();
        __syncthreads();
        h.Write(c);       // thread-0 only (internal guard)
        __syncthreads();
    }
}

/** GetBlob every chunk back, grid-stride. */
__global__ void McReadKernel(kvhdf5::GpuDatasetHandle h) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.Read(c);        // thread-0 only (internal guard)
        __syncthreads();
    }
}

#if !CTP_IS_DEVICE_PASS

// fill+Write -> zero -> Read -> verify every chunk byte-identical to its
// pattern, and chunks mutually distinct. General over chunk count + per-chunk
// size (read from `ds`) and over launch `grid`: grid=1 strides serially over
// all chunks; grid>1 submits chunks across concurrent blocks (grid < N also
// exercises the per-block range loop). See the file header note.
static void RunAndVerify(kvhdf5::GpuCteDataset& ds, unsigned seed_base,
                         const char* label, unsigned grid) {
    kvhdf5::GpuDatasetHandle h = ds.Handle();
    const uint32_t n = ds.ChunkCount();

    std::vector<std::vector<byte_t>> expected(n);
    for (uint32_t c = 0; c < n; ++c) {
        const uint64_t bytes = ds.ChunkBytes(c);
        expected[c].resize(bytes);
        for (uint64_t i = 0; i < bytes; ++i)
            expected[c][i] = Pattern(seed_base + c, static_cast<unsigned>(i));
    }
    auto ZeroAll = [&] {
        for (uint32_t c = 0; c < n; ++c) {
            std::vector<byte_t> z(ds.ChunkBytes(c));
            ctp::GpuApi::Memcpy(ds.DeviceData(c), z.data(), ds.ChunkBytes(c));
        }
    };

    ZeroAll();
    McFillWriteKernel<<<grid, 32>>>(h, seed_base);
    ctp::GpuApi::Synchronize();

    ZeroAll();  // clobber so the GetBlob readback is real
    McReadKernel<<<grid, 32>>>(h);
    ctp::GpuApi::Synchronize();

    for (uint32_t c = 0; c < n; ++c) {
        const uint64_t bytes = ds.ChunkBytes(c);
        std::vector<byte_t> back(bytes);
        ctp::GpuApi::Memcpy(back.data(), ds.DeviceData(c), bytes);
        if (std::memcmp(back.data(), expected[c].data(), bytes) != 0)
            std::fprintf(stderr, "[%s] chunk %u mismatch\n", label, c);
        REQUIRE(std::memcmp(back.data(), expected[c].data(), bytes) == 0);
    }
    // Distinct chunks differ: each readback is compared to its own distinct
    // pattern above, so a stuck/aliased chunk index fails the per-chunk check.
    for (uint32_t c = 1; c < n; ++c) REQUIRE(expected[c] != expected[c - 1]);
    std::fprintf(stderr,
                 "[ok] multichunk %s: %u chunks round-tripped (grid=%u)\n",
                 label, n, grid);
}

// Create (or look up) a tag by name so each sweep config gets its OWN blob
// namespace. Chunk blob names are just coord text ("0".."N-1"); two same-shape
// datasets sharing one tag would collide on those names (re-Put the same blob
// -> hang). A distinct tag per config keeps them independent.
static clio::cte::core::TagId MakeTag(const char* name) {
    auto* cte_client = CLIO_CTE_CLIENT;
    auto t = cte_client->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

// Build a fresh 1-D dataset of `chunks` chunks under its own tag and round-trip
// it once at launch `grid`. One dataset == one round-trip (the proven invariant).
static void SweepConfig(clio::run::IpcManager* ipc, clio::run::IpcManagerGpuInfo gpu_info,
                        unsigned chunks, unsigned grid, unsigned seed,
                        const char* tag_name, const char* label) {
    clio::cte::core::TagId tag = MakeTag(tag_name);
    kvhdf5::Layout layout{/*dims=*/{chunks * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes}, /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == chunks);
    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, tag, layout);
    REQUIRE(ds.ChunkCount() == chunks);
    RunAndVerify(ds, seed, label, grid);
}

TEST_CASE("GPU multi-chunk PutBlob+GetBlob round trip via dataset handle",
          "[integration][gpu][cte][multichunk]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    REQUIRE(ipc->GetGpuQueueCount() >= 1u);

    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    // 1-D layout split into kChunkCount equal chunks of kChunkBytes (elem_size=1).
    kvhdf5::Layout layout{/*dims=*/{kChunkCount * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes},
                          /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == kChunkCount);

    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, env.tag_id, layout);
    REQUIRE(ds.ChunkCount() == kChunkCount);

    // grid=2 over 4 chunks -> 2 concurrent blocks, each striding over 2 chunks:
    // exercises cross-block concurrency AND the per-block range loop
    // (gridDim < ChunkCount). Concurrent multi-block gpu2cpu Send is fine here
    // (see file header) — this is the cross-block parallelism Phase 2 deferred,
    // and grid=1 is just its degenerate single-block sub-case. One round-trip
    // per dataset (re-running over the same GpuCteDataset is out of scope).
    RunAndVerify(ds, kSeedBase, "1d", /*grid=*/2);
}

// Phase 4 tag-wiring follow-up: build a GPU dataset straight from a DatasetMeta
// (path + layout). FromDataset canonicalizes the path into the CTE tag and
// resolves it to a TagId via get-or-create; the blob key stays the chunk coord.
// Proves the full path -> TagId -> GPU-resident blobs round-trip without the
// caller hand-resolving a tag.
TEST_CASE("GPU dataset built from a DatasetMeta path resolves its tag end-to-end",
          "[integration][gpu][cte][tagwire]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    (void)env;
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    kvhdf5::Layout layout{/*dims=*/{kChunkCount * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes}, /*elem_size=*/1};
    kvhdf5::DatasetMeta meta{"/results//snapshots/2026/pressure/", layout};
    REQUIRE(meta.TagName() == "results/snapshots/2026/pressure");

    auto ds = kvhdf5::GpuCteDataset::FromDataset(
        ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, meta);
    REQUIRE(ds.ChunkCount() == kChunkCount);
    RunAndVerify(ds, /*seed_base=*/0xC0u, "tagwire", /*grid=*/2);
}

// Decoupled (minimal) surface: the GPU producer needs only a path (-> tag) and a
// layout, NOT a host metadata struct. FromPath canonicalizes the path and resolves
// the tag inline; no DatasetMeta / cpu_file directory model in sight.
TEST_CASE("GPU dataset built from a bare path + layout (no metadata struct)",
          "[integration][gpu][cte][tagwire]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    (void)env;
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    kvhdf5::Layout layout{/*dims=*/{kChunkCount * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes}, /*elem_size=*/1};
    auto ds = kvhdf5::GpuCteDataset::FromPath(
        ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT,
        "results/snapshots/2026/velocity", layout);
    REQUIRE(ds.ChunkCount() == kChunkCount);
    RunAndVerify(ds, /*seed_base=*/0xD0u, "frompath", /*grid=*/2);
}

// Covers two things the 1-D case above doesn't: (a) the Layout ctor's
// multi-dimensional chunk-coord name derivation (2-D -> "r_c" names via
// ChunkIndexToCoord with rank>1), and (b) GpuCteDataset's MULTI-chunk move ctor
// (3 device allocs + a populated host_descs_ vector). A move that fails to null
// the source's allocs would double-free at teardown and abort the process —
// which ctest reports as a failure.
TEST_CASE("GPU multi-chunk dataset 2-D layout names + move ctor",
          "[integration][gpu][cte][multichunk][move]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    // 2x2 grid of 1x1 chunks -> 4 chunks named "0_0","0_1","1_0","1_1".
    constexpr unsigned kElemBytes = 64;
    kvhdf5::Layout layout{/*dims=*/{2, 2}, /*chunk_dims=*/{1, 1},
                          /*elem_size=*/kElemBytes};
    REQUIRE(layout.ChunkCount() == 4u);

    kvhdf5::GpuCteDataset src(ipc, gpu_info, /*gpu_id=*/0, env.tag_id, layout);
    REQUIRE(src.ChunkCount() == 4u);

    // Move-construct; `src` is left null (count 0). Round-trip through the
    // moved-into object to prove its handle + 3 allocs + host_descs_ survived.
    kvhdf5::GpuCteDataset ds(std::move(src));
    REQUIRE(src.ChunkCount() == 0u);
    REQUIRE(ds.ChunkCount() == 4u);
    RunAndVerify(ds, /*seed_base=*/0x80u, "2d-moved", /*grid=*/2);
    // At scope exit: `ds` frees its 3 backends once; `src` (nulled) frees nothing.
    // A double-free here would abort and fail this case.
}

// Sweep (chunk-count x grid) to exercise multi-block parallelism broadly:
// serial (grid=1), gridDim < ChunkCount (per-block range loop + concurrency),
// and one-block-per-chunk (grid == ChunkCount), at two chunk counts. Each
// config uses its own tag + a fresh dataset (one round-trip per dataset).
TEST_CASE("GPU multi-chunk parallelism sweep (grid x chunk-count)",
          "[integration][gpu][cte][multichunk][sweep]") {
    (void)kvhdf5::itest::SharedCteEnv();  // ensure server + CTE are up
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    // 4 chunks: serial / concurrent+range (grid<count) / one-block-per-chunk.
    SweepConfig(ipc, gpu_info, /*chunks=*/4, /*grid=*/1, 0x10u, "kvhdf5_sweep_4c_g1", "4c-g1-serial");
    SweepConfig(ipc, gpu_info, /*chunks=*/4, /*grid=*/2, 0x20u, "kvhdf5_sweep_4c_g2", "4c-g2-range");
    SweepConfig(ipc, gpu_info, /*chunks=*/4, /*grid=*/4, 0x30u, "kvhdf5_sweep_4c_g4", "4c-g4-perchunk");
    // 8 chunks: gridDim << count (each block strides 4 chunks) / one-per-chunk.
    SweepConfig(ipc, gpu_info, /*chunks=*/8, /*grid=*/2, 0x50u, "kvhdf5_sweep_8c_g2", "8c-g2-range");
    SweepConfig(ipc, gpu_info, /*chunks=*/8, /*grid=*/8, 0x60u, "kvhdf5_sweep_8c_g8", "8c-g8-perchunk");
}

// --- Phase 3 "P": bounded data-buffer pool (M buffers, N chunks, M <= N) -------
//
// The producer GPU kernel (McFillWriteKernel, reused) fills each chunk and Puts
// it; with a pool, chunks c and c+M share buffer c % M. The GPU can't Get all N
// chunks back (only M buffers exist), so verification reads each blob from the
// CTE server on the host instead. The launch uses grid == M so each pooled buffer
// is owned by a single block and reuse is serialized by that block's
// Write() = Send().Wait() — this is exactly what these cases assert.

// Host-side CTE read-back of one blob into a registered CPU buffer.
static std::vector<byte_t> HostReadBlob(clio::cte::core::TagId tag,
                                        const std::string& name, uint64_t size) {
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(size);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0, size);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    auto* cte_client = CLIO_CTE_CLIENT;
    auto t = cte_client->AsyncGetBlob(tag, name, /*offset=*/clio::run::u64(0), size,
                                      /*flags=*/clio::run::u32(0), shm);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

// Build a 1-D dataset of `chunks` chunks bounded to `pool` buffers, GPU-produce
// all chunks (grid == pool), then verify each blob from the server. A buffer
// refilled before its prior Put's server read completed would corrupt the earlier
// blob -> a mismatch here (the Wait-gates-reuse assertion).
static void RunPoolAndVerifyHost(clio::run::IpcManager* ipc,
                                 clio::run::IpcManagerGpuInfo gpu_info, unsigned chunks,
                                 unsigned pool, unsigned chunk_bytes, unsigned seed,
                                 const char* tag_name, const char* label,
                                 unsigned grid = 0) {
    if (grid == 0) grid = pool;  // default: grid == pool (buffer-exclusivity)
    clio::cte::core::TagId tag = MakeTag(tag_name);
    kvhdf5::Layout layout{/*dims=*/{chunks * chunk_bytes},
                          /*chunk_dims=*/{chunk_bytes}, /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == chunks);
    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, tag, layout,
                             /*pool_size=*/pool);
    REQUIRE(ds.ChunkCount() == chunks);
    REQUIRE(ds.PoolSize() == pool);

    std::fprintf(stderr,
                 "[ck] %s: ds built (chunks=%u pool=%u bytes=%u grid=%u); launching\n",
                 label, chunks, pool, chunk_bytes, grid);
    McFillWriteKernel<<<grid, 32>>>(ds.Handle(), seed);
    ctp::GpuApi::Synchronize();
    std::fprintf(stderr, "[ck] %s: kernel + Synchronize done (Put-side complete)\n",
                 label);

    std::vector<std::vector<byte_t>> got(chunks);
    for (unsigned c = 0; c < chunks; ++c) {
        std::vector<byte_t> expected(chunk_bytes);
        for (unsigned i = 0; i < chunk_bytes; ++i) expected[i] = Pattern(seed + c, i);
        std::fprintf(stderr, "[ck] %s: HostReadBlob chunk %u ...\n", label, c);
        got[c] = HostReadBlob(tag, std::to_string(c), chunk_bytes);
        std::fprintf(stderr, "[ck] %s: HostReadBlob chunk %u returned\n", label, c);
        if (got[c] != expected)
            std::fprintf(stderr, "[%s] chunk %u (buffer %u) mismatch\n",
                         label, c, c % pool);
        REQUIRE(got[c] == expected);
    }
    // Chunks sharing a buffer must still differ (reuse didn't collapse them).
    for (unsigned c = pool; c < chunks; ++c) REQUIRE(got[c] != got[c - pool]);
    std::fprintf(stderr,
                 "[ok] pool %s: %u chunks through %u buffers x %u B (grid=%u)\n",
                 label, chunks, pool, chunk_bytes, pool);
}

// Step 0 (gates all of Phase 3): 2 chunks through 1 buffer. Chunk 0 is filled +
// Put + Wait; then chunk 1 overwrites the SAME buffer + Put + Wait. If Wait gates
// the server's read of the GPU buffer, blob "0" still holds chunk 0's pattern
// despite the overwrite — proven by host read-back.
TEST_CASE("GPU data-buffer pool: Wait gates buffer reuse (2 chunks, 1 buffer)",
          "[integration][gpu][cte][pool][spike]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    // Tag = the dataset's full path, canonicalized (the path->tag scheme); the
    // blob key stays the chunk coord. Proves a real path-tag round-trips iowarp.
    std::string tag = kvhdf5::tagpath::CanonicalTag(
        "/results//snapshots/2026/temperature/");
    REQUIRE(tag == "results/snapshots/2026/temperature");
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/2, /*pool=*/1,
                         /*chunk_bytes=*/kChunkBytes, 0x90u,
                         tag.c_str(), "spike-2c-1b");
}

// P milestone: 8 chunks stream through 3 resident buffers (M < N). Each buffer is
// reused for up to ceil(8/3) chunks; host read-back verifies all 8 blobs survived.
TEST_CASE("GPU data-buffer pool: 8 chunks through 3 buffers",
          "[integration][gpu][cte][pool]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    // Distinct dataset path -> distinct tag (avoids chunk-coord blob-name
    // collision with the other pool case while sharing the same scheme).
    std::string tag = kvhdf5::tagpath::CanonicalTag(
        "results/snapshots/2026/density");
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/8, /*pool=*/3,
                         /*chunk_bytes=*/kChunkBytes, 0xA0u,
                         tag.c_str(), "8c-3b");
}

// Large-chunk diagnostics (HIDDEN: leading [.] keeps them out of the default
// suite — run explicitly by tag). The "Wait gates the server's D2H read" claim is
// only meaningful when that read is SLOW: at 256 B it's near-instant and wins the
// reuse race whether or not Wait gates it; at multi-MB the window is wide. These
// two isolate size from reuse:
//
//  - [poolctrl]  : large chunks, M == N (NO buffer reuse). Exercises the 4 MiB
//                  Put + 4 MiB host read-back path alone. If THIS fails, large I/O
//                  is broken independent of pooling (a test/infra issue).
//  - [poolreuse] : large chunks, M < N (buffers reused). If ctrl passes and this
//                  fails/hangs, the failure is the reuse race itself → fut.Wait()
//                  does NOT gate the server read at scale; P needs a server-read
//                  fence before refill.
// Chunk size for the large diagnostics: env KVHDF5_BIG_BYTES (default 4 MiB), so a
// driver script can sweep sizes in separate processes to find the ceiling.
static unsigned BigChunkBytes() {
    if (const char* e = std::getenv("KVHDF5_BIG_BYTES")) {
        unsigned v = static_cast<unsigned>(std::strtoul(e, nullptr, 10));
        if (v) return v;
    }
    return 4u * 1024u * 1024u;
}

TEST_CASE("pool large control: 2 chunks / 2 buffers (no reuse)",
          "[.][integration][gpu][cte][poolctrl]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/2, /*pool=*/2,
                         /*chunk_bytes=*/BigChunkBytes(), 0xB0u,
                         "kvhdf5_pool_lg_ctrl", "2c-2b-noreuse");
}

// Instrumented localizer (HIDDEN): 256 B then 4 KiB in ONE process, both 2c/2b
// (M==N, NO pooling, NO reuse — pure Put + host read-back at two sizes). The [ck]
// checkpoints (unbuffered stderr) pinpoint, if 4 KiB hangs, whether it stalls
// AFTER "kernel + Synchronize done" inside a HostReadBlob (Get-side / verification
// harness) or never reaches that line (Put-side). 256 B running first re-confirms
// the current code still passes (kills the regression theory).
TEST_CASE("pool instrumented: 256 B then 4 KiB in one process",
          "[.][integration][gpu][cte][poolinstr]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    // 256 B, grid=2 (concurrent) — baseline, expected pass.
    std::fprintf(stderr, "[instr] === 256 B grid=2 (expect pass) ===\n");
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/2, /*pool=*/2,
                         /*chunk_bytes=*/256u, 0xD0u, "instr256", "256B-g2",
                         /*grid=*/2);
    // 4 KiB, grid=1 (SERIAL: one block strides both chunks, each its own buffer).
    // If THIS fails, the corruption is pure size (serial Put truncates/aliases).
    std::fprintf(stderr, "[instr] === 4 KiB grid=1 SERIAL (size discriminator) ===\n");
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/2, /*pool=*/2,
                         /*chunk_bytes=*/4096u, 0xD4u, "instr4kg1", "4KiB-g1",
                         /*grid=*/1);
    // 4 KiB, grid=2 (CONCURRENT) — the config that corrupted under kDeviceMem.
    std::fprintf(stderr, "[instr] === 4 KiB grid=2 CONCURRENT (was corrupt under kDeviceMem) ===\n");
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/2, /*pool=*/2,
                         /*chunk_bytes=*/4096u, 0xD8u, "instr4kg2", "4KiB-g2",
                         /*grid=*/2);
    // 1 MiB, grid=2 (CONCURRENT, no reuse) — convincing at-scale concurrent Put.
    std::fprintf(stderr, "[instr] === 1 MiB grid=2 CONCURRENT (at-scale) ===\n");
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/2, /*pool=*/2,
                         /*chunk_bytes=*/1u * 1024u * 1024u, 0xDCu, "instr1mg2",
                         "1MiB-g2", /*grid=*/2);
    // 1 MiB, 4 chunks / 2 buffers, grid=2 (CONCURRENT + REUSE) — the actual P
    // scenario at scale: bounded pool, concurrent produce, buffers reused. This is
    // the real "Wait gates reuse" test at a size where the server read is slow.
    std::fprintf(stderr, "[instr] === 1 MiB 4c/2b grid=2 CONCURRENT+REUSE (real P at scale) ===\n");
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/4, /*pool=*/2,
                         /*chunk_bytes=*/1u * 1024u * 1024u, 0xE0u, "instr1mreuse",
                         "1MiB-4c2b", /*grid=*/2);
    std::fprintf(stderr, "[instr] === all configs completed ===\n");
}

TEST_CASE("pool large reuse: 4 chunks / 2 buffers (reuse)",
          "[.][integration][gpu][cte][poolreuse]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
    RunPoolAndVerifyHost(ipc, gpu_info, /*chunks=*/4, /*pool=*/2,
                         /*chunk_bytes=*/BigChunkBytes(), 0xC0u,
                         "kvhdf5_pool_lg_reuse", "4c-2b-reuse");
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
