/*
 * GPU Dataset-handle integration test — PutBlob + GetBlob round trip.
 *
 * Exercises kvhdf5's device-facing producer surface (Slice 2): the host
 * GpuCteDataset preallocates the registered device task + data backends and
 * hands out a trivially-copyable GpuDatasetHandle; a kernel fills the buffer and
 * submits the pre-built task via handle.Write()/Read() (thread-0 Send+Wait).
 *
 * This proves the handle with the *reference's proven structure*: a separate
 * fill kernel + host Synchronize before the submit kernel, so the buffer's
 * contents are visible to the CPU-side PutBlob without relying on intra-kernel
 * fill->Send visibility (the fused fill+Send path Gray Scott wants is a
 * follow-on that needs an explicit device fence — deliberately not tested here).
 *
 * Reuses the one-time SharedCteEnv server bring-up via the shared header.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/chunking.h>
#include <clio_cte/kvhdf5/gpu_cte_dataset.h>

#include <cstdio>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "cte_env.h"

namespace {

constexpr clio::run::u32 kBlobBytes = 256;
constexpr clio::run::u32 kPatternSeed = 0x5Au;

}  // namespace

using kvhdf5::byte_t;  // raw blob-payload bytes (codebase convention)

/** Fill the device blob buffer with the byte pattern (separate kernel). */
__global__ void DsFillKernel(byte_t *buf, clio::run::u32 size, clio::run::u32 seed) {
  clio::run::u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  buf[i] = static_cast<byte_t>((seed ^ i) & 0xFFu);
}

/** Submit the pre-built PutBlob task from the kernel via the handle. */
__global__ void DsWriteKernel(kvhdf5::GpuDatasetHandle h) {
  CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);  // block-wide, has __syncthreads
  (void)g_ipc_manager;
  h.Write();  // internally thread-0 only
}

__global__ void DsReadKernel(kvhdf5::GpuDatasetHandle h) {
  CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
  (void)g_ipc_manager;
  h.Read();
}

// Fused compute+IO: fill the buffer AND submit PutBlob in one launch, with no
// host Synchronize between. This is the path Gray Scott wants. Each thread
// fences its own fills system-wide before the barrier so the CPU-side PutBlob
// (which D2H-copies the device buffer while this kernel is still resident,
// thread-0 spinning in Wait) observes them.
__global__ void DsFillAndWriteKernel(kvhdf5::GpuDatasetHandle h, clio::run::u32 seed) {
  CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
  (void)g_ipc_manager;
  for (uint64_t i = threadIdx.x; i < h.Size(); i += blockDim.x)
    h.Data()[i] = static_cast<byte_t>((seed ^ i) & 0xFFu);
  __threadfence_system();  // flush each thread's fills to system scope
  __syncthreads();         // all fills done before the producer Sends
  h.Write();               // thread-0 only (internal guard)
}

#if !CTP_IS_DEVICE_PASS

/** Verify the device buffer holds the pattern (host reads it back via D2H). */
static clio::run::u32 DsVerifyDevicePattern(const byte_t *device_buf, clio::run::u32 size,
                                      clio::run::u32 seed) {
  std::vector<byte_t> host(size);
  ctp::GpuApi::Memcpy(host.data(), device_buf, size);
  for (clio::run::u32 i = 0; i < size; ++i) {
    byte_t want = static_cast<byte_t>((seed ^ i) & 0xFFu);
    if (host[i] != want) return i;
  }
  return size;  // all match
}

TEST_CASE("GPU Dataset handle PutBlob+GetBlob round trip",
          "[integration][gpu][cte][dataset]") {
  auto &env = kvhdf5::itest::SharedCteEnv();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  REQUIRE(ipc->GetGpuQueueCount() >= 1u);

  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

  // Single-chunk blob name from the shared chunking helper (coord {0} -> "0").
  char name[kvhdf5::chunking::kMaxBlobNameLen + 1];
  uint64_t coord[1] = {0};
  auto chunk_name = kvhdf5::chunking::ChunkCoordToName({coord, 1}, name);
  REQUIRE(!chunk_name.empty());

  // Host control plane preallocates the registered task + data backends.
  kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, env.tag_id, name,
                           kBlobBytes);
  kvhdf5::GpuDatasetHandle h = ds.Handle();
  byte_t *data = ds.DeviceData();

  // ---- Fill the buffer (separate kernel) then sync so PutBlob sees it. ----
  clio::run::u32 threads = 256;
  clio::run::u32 blocks = (kBlobBytes + threads - 1) / threads;
  DsFillKernel<<<blocks, threads>>>(data, kBlobBytes, kPatternSeed);
  ctp::GpuApi::Synchronize();

  // ---- Submit PutBlob from a kernel via the handle. ----
  std::fprintf(stderr, "[put] launching DsWriteKernel\n");
  DsWriteKernel<<<1, 32>>>(h);
  ctp::GpuApi::Synchronize();

  // ---- Zero the buffer so the GetBlob readback is verifiable. ----
  std::vector<byte_t> zeros(kBlobBytes);
  ctp::GpuApi::Memcpy(data, zeros.data(), kBlobBytes);

  // ---- Submit GetBlob from a kernel via the handle. ----
  std::fprintf(stderr, "[get] launching DsReadKernel\n");
  DsReadKernel<<<1, 32>>>(h);
  ctp::GpuApi::Synchronize();

  // ---- Verify the original pattern came back. ----
  clio::run::u32 first_bad = DsVerifyDevicePattern(data, kBlobBytes, kPatternSeed);
  if (first_bad != kBlobBytes)
    std::fprintf(stderr, "[verify] mismatch at index %u (of %u)\n", first_bad,
                 kBlobBytes);
  REQUIRE(first_bad == kBlobBytes);
  std::fprintf(stderr, "[ok] gpu dataset-handle round trip ok (%u bytes)\n",
               kBlobBytes);
}

// The capability the device-facing handle exists for: fill + submit in ONE
// kernel launch (no host Synchronize between the compute and the Put), proving
// the producer sees the just-written bytes via the in-kernel system fence.
TEST_CASE("GPU Dataset handle fused fill+write round trip",
          "[integration][gpu][cte][dataset][fused]") {
  auto &env = kvhdf5::itest::SharedCteEnv();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

  // Distinct blob name (coord {1}) so this case doesn't alias the first.
  char name[kvhdf5::chunking::kMaxBlobNameLen + 1];
  uint64_t coord[1] = {1};
  auto chunk_name = kvhdf5::chunking::ChunkCoordToName({coord, 1}, name);
  REQUIRE(!chunk_name.empty());

  kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, env.tag_id, name,
                           kBlobBytes);
  kvhdf5::GpuDatasetHandle h = ds.Handle();
  byte_t *data = ds.DeviceData();

  // ---- Fill + submit PutBlob in a single launch (no host sync between). ----
  std::fprintf(stderr, "[put] launching DsFillAndWriteKernel (fused)\n");
  DsFillAndWriteKernel<<<1, 32>>>(h, kPatternSeed);
  ctp::GpuApi::Synchronize();

  // ---- Zero, read back via the handle, verify. ----
  std::vector<byte_t> zeros(kBlobBytes);
  ctp::GpuApi::Memcpy(data, zeros.data(), kBlobBytes);
  DsReadKernel<<<1, 32>>>(h);
  ctp::GpuApi::Synchronize();

  clio::run::u32 first_bad = DsVerifyDevicePattern(data, kBlobBytes, kPatternSeed);
  if (first_bad != kBlobBytes)
    std::fprintf(stderr, "[verify] fused mismatch at index %u (of %u)\n",
                 first_bad, kBlobBytes);
  REQUIRE(first_bad == kBlobBytes);
  std::fprintf(stderr, "[ok] gpu dataset-handle fused round trip ok (%u bytes)\n",
               kBlobBytes);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
