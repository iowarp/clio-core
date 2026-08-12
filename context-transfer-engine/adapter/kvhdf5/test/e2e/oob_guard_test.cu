/*
 * Device-side chunk-coordinate bounds guard (GpuDatasetHandle).
 *
 * Every device accessor (Data/Size/Write/Read/WriteAsync/ReadAsync/WriteWait/
 * ReadWait/SetTag and the WritePipelined c-m path) subscripts the device
 * ChunkDesc array chunks_[c]. An out-of-range c used to read past the array and
 * then dereference whatever pointer it found — undefined behaviour that could
 * silently corrupt unrelated device memory. The guard turns any out-of-range
 * coordinate into a no-op that the HOST can observe through the existing I/O
 * status API (OutOfRangeChunk / IoFailed / ThrowIfIoFailed), never a memory
 * access.
 *
 * This binary is compiled with KVHDF5_GPU_OOR_TRAP=0 (see CMakeLists) so the
 * guard reports-and-returns instead of trapping — a trap would destroy the CUDA
 * context and take the shared server down with it. That is exactly the release
 * behaviour this test asserts; the debug trap is a separate development tripwire.
 *
 * Asserts, all in one process against the live runtime:
 *   (a) the host status API reports the offending coordinate;
 *   (b) NO write lands anywhere — every chunk's data buffer keeps its canary,
 *       and a neighbouring device allocation keeps its checksum;
 *   (c) Data(bad) is nullptr in-kernel and every void accessor is a no-op;
 *   (d) an IN-RANGE coordinate is completely unaffected (no false positive), and
 *       a clean dataset reports OutOfRangeChunk() == -1.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/dataset_meta.h>
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
constexpr unsigned kChunkBytes = 256;
constexpr unsigned kChunkCount = 4;
constexpr byte_t Canary(unsigned seed, unsigned i) {
    return static_cast<byte_t>((seed ^ i) & 0xFFu);
}
}  // namespace

// Hammer every device accessor with an out-of-range coordinate. Records, into a
// device-visible result array, what each accessor returned so the host can prove
// the no-op (rather than only trusting the sticky report word):
//   r[0] = (Data(bad) == nullptr) ? 1 : 0     (must be 1)
//   r[1] = (Size(bad) == 0)       ? 1 : 0     (must be 1)
//   r[2] = 1 once the block finished all void calls without faulting
// The void calls (Write/Read/WriteAsync/ReadAsync/WriteWait/ReadWait/SetTag)
// must simply not touch memory; their effect is checked host-side via the
// canaries + status word.
__global__ void OobHammerKernel(kvhdf5::GpuDatasetHandle h, uint32_t bad,
                                int* r) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        r[0] = (h.Data(bad) == nullptr) ? 1 : 0;
        r[1] = (h.Size(bad) == 0) ? 1 : 0;
    }
    __syncthreads();
    // Block-wide (Data) and thread-0 (the rest) accessors, all out of range.
    (void)h.Data(bad);
    (void)h.Size(bad);
    h.SetTag(bad, clio::cte::core::TagId{});
    h.Write(bad);
    h.Read(bad);
    h.WriteAsync(bad);
    h.ReadAsync(bad);
    h.WriteWait(bad);
    h.ReadWait(bad);
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) r[2] = 1;
}

// In-range control: fill chunk 0 and Write it. Proves the guard leaves the valid
// path byte-for-byte identical.
__global__ void InRangeWriteKernel(kvhdf5::GpuDatasetHandle h, unsigned seed) {
    CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    byte_t* dst = h.Data(0);
    uint64_t n = h.Size(0);
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = static_cast<byte_t>((seed ^ i) & 0xFFu);
    __threadfence_system();
    __syncthreads();
    h.Write(0);
    __syncthreads();
}

#if !CTP_IS_DEVICE_PASS

static clio::cte::core::TagId MakeTag(const char* name) {
    auto* cte_client = CLIO_CTE_CLIENT;
    auto t = cte_client->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

static kvhdf5::GpuCteDataset MakeDataset(clio::run::IpcManager* ipc,
                                         clio::run::IpcManagerGpuInfo gpu_info,
                                         const char* tag_name) {
    clio::cte::core::TagId tag = MakeTag(tag_name);
    kvhdf5::Layout layout{/*dims=*/{kChunkCount * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes}, /*elem_size=*/1};
    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, tag, layout);
    REQUIRE(ds.ChunkCount() == kChunkCount);
    return ds;
}

TEST_CASE("GPU dataset handle: out-of-range chunk coordinate is a reported no-op",
          "[integration][gpu][cte][oob]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    kvhdf5::GpuCteDataset ds = MakeDataset(ipc, gpu_info, "kvhdf5_oob_guard");

    // Seed every chunk buffer with a distinct canary so any wild write shows up.
    for (uint32_t c = 0; c < kChunkCount; ++c) {
        std::vector<byte_t> pat(kChunkBytes);
        for (unsigned i = 0; i < kChunkBytes; ++i) pat[i] = Canary(0x50u + c, i);
        ctp::GpuApi::Memcpy(ds.DeviceData(c), pat.data(), kChunkBytes);
    }

    // Neighbouring device allocation with its own checksum — stands in for
    // "memory adjacent to the descriptor array": if the guard let a wild store
    // through, an unrelated allocation is where it would land.
    constexpr size_t kNeighborBytes = 4096;
    byte_t* neighbor = ctp::GpuApi::Malloc<byte_t>(kNeighborBytes);
    std::vector<byte_t> nb_pat(kNeighborBytes);
    for (size_t i = 0; i < kNeighborBytes; ++i)
        nb_pat[i] = static_cast<byte_t>((0xA5u ^ i) & 0xFFu);
    ctp::GpuApi::Memcpy(neighbor, nb_pat.data(), kNeighborBytes);

    // Device result slots for the in-kernel observations. GpuApi::Malloc takes a
    // BYTE count, not an element count.
    int* d_r = ctp::GpuApi::Malloc<int>(3 * sizeof(int));
    ctp::GpuApi::Memset(d_r, 0, 3 * sizeof(int));

    const uint32_t bad = kChunkCount + 5;  // well past the end
    OobHammerKernel<<<2, 32>>>(ds.Handle(), bad, d_r);
    ctp::GpuApi::Synchronize();

    // (c) in-kernel accessors were no-ops.
    int r[3] = {0, 0, 0};
    ctp::GpuApi::Memcpy(r, d_r, 3 * sizeof(int));
    REQUIRE(r[0] == 1);  // Data(bad) == nullptr
    REQUIRE(r[1] == 1);  // Size(bad) == 0
    REQUIRE(r[2] == 1);  // block completed all void calls without faulting

    // (a) host status reports the offending coordinate.
    REQUIRE(ds.OutOfRangeChunk() == static_cast<int>(bad));
    REQUIRE(ds.IoFailed());
    REQUIRE_THROWS_AS(ds.ThrowIfIoFailed(), std::runtime_error);

    // (b) no write landed: every chunk buffer keeps its canary...
    for (uint32_t c = 0; c < kChunkCount; ++c) {
        std::vector<byte_t> back(kChunkBytes), pat(kChunkBytes);
        ctp::GpuApi::Memcpy(back.data(), ds.DeviceData(c), kChunkBytes);
        for (unsigned i = 0; i < kChunkBytes; ++i) pat[i] = Canary(0x50u + c, i);
        REQUIRE(std::memcmp(back.data(), pat.data(), kChunkBytes) == 0);
    }
    // ...and the neighbouring allocation keeps its checksum.
    std::vector<byte_t> nb_back(kNeighborBytes);
    ctp::GpuApi::Memcpy(nb_back.data(), neighbor, kNeighborBytes);
    REQUIRE(std::memcmp(nb_back.data(), nb_pat.data(), kNeighborBytes) == 0);
    // No in-range Put ever fired, so no chunk reports a Put failure.
    REQUIRE(ds.FirstFailedPut() == -1);

    ctp::GpuApi::Free(neighbor);
    ctp::GpuApi::Free(d_r);
    std::fprintf(stderr, "[ok] oob guard: coord %u rejected, no write landed\n", bad);
}

TEST_CASE("GPU dataset handle: in-range coordinate is unaffected by the guard",
          "[integration][gpu][cte][oob]") {
    (void)kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    clio::run::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    kvhdf5::GpuCteDataset ds = MakeDataset(ipc, gpu_info, "kvhdf5_oob_inrange");

    constexpr unsigned kSeed = 0x33u;
    InRangeWriteKernel<<<1, 32>>>(ds.Handle(), kSeed);
    ctp::GpuApi::Synchronize();

    // Clean status: no out-of-range access, no failed Put.
    REQUIRE(ds.OutOfRangeChunk() == -1);
    REQUIRE_FALSE(ds.IoFailed());
    ds.ThrowIfIoFailed();  // must not throw

    // The write landed correctly on chunk 0.
    std::vector<byte_t> back(kChunkBytes), pat(kChunkBytes);
    ctp::GpuApi::Memcpy(back.data(), ds.DeviceData(0), kChunkBytes);
    for (unsigned i = 0; i < kChunkBytes; ++i)
        pat[i] = static_cast<byte_t>((kSeed ^ i) & 0xFFu);
    REQUIRE(std::memcmp(back.data(), pat.data(), kChunkBytes) == 0);
    std::fprintf(stderr, "[ok] oob guard: in-range write unaffected\n");
}

#endif  // !CTP_IS_DEVICE_PASS

#else
// Non-GPU build: nothing to test here.
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
