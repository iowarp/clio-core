/*
 * Gray-Scott end-to-end integration test on kvhdf5's new producer surface
 * (Slice 3). Proves the device-facing handle in a realistic compute loop:
 *
 *   - Working grids (u/v, ping-pong) live in plain device buffers; a pure-CUDA
 *     kernel evolves them each step (Laplacian + reaction, periodic BCs). The
 *     compute touches zero kvhdf5 surface — that's the point of the producer
 *     model: compute on the GPU, I/O is *explicit persistence*.
 *   - Every kSnap steps a snapshot is persisted via the Slice-2 *fused* PutBlob
 *     path: a kernel copies the evolved grid into a fresh GpuCteDataset's
 *     registered buffer, fences system-wide, and submits handle.Write() in one
 *     launch. Each snapshot is its own per-step-named GpuCteDataset stored in a
 *     std::vector — which exercises GpuCteDataset's move ctor (vector growth) and
 *     so the GPU-backend double-free risk for free.
 *   - Verification reads each snapshot back via GetBlob, asserts byte-identical
 *     to the evolved grid, AND asserts the grid differs from the seed (so a
 *     no-op compute kernel can't pass).
 *
 * Deliberately does NOT use the old orchestrator pause/resume + LaunchAndPoll
 * harness — the producer model dropped it; we just launch and Synchronize().
 *
 * Reuses the one-time SharedCteEnv server bring-up via the shared header.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <clio_cte/kvhdf5/gpu_cte_dataset.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "cte_env.h"

using kvhdf5::byte_t;  // raw blob-payload bytes (codebase convention)

namespace {

constexpr unsigned kN = 32;                       // grid edge (single chunk)
constexpr unsigned kCells = kN * kN;              // 1024
constexpr unsigned kBytes = kCells * sizeof(float);  // 4096
constexpr int kSteps = 12;
constexpr int kSnap = 4;                          // snapshot every kSnap steps

struct GsParams {
  float Du, Dv, F, k, dt;
};

}  // namespace

/** One Gray-Scott step: one thread per cell, periodic BCs. Pure CUDA. */
__global__ void GsStepKernel(const float *u, const float *v, float *un,
                             float *vn, GsParams p) {
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= kCells) return;
  unsigned x = gid % kN, y = gid / kN;
  unsigned xm = (x == 0) ? (kN - 1) : (x - 1);
  unsigned xp = (x == kN - 1) ? 0u : (x + 1);
  unsigned ym = (y == 0) ? (kN - 1) : (y - 1);
  unsigned yp = (y == kN - 1) ? 0u : (y + 1);

  float uc = u[gid], vc = v[gid];
  float lap_u =
      u[y * kN + xm] + u[y * kN + xp] + u[ym * kN + x] + u[yp * kN + x] - 4.f * uc;
  float lap_v =
      v[y * kN + xm] + v[y * kN + xp] + v[ym * kN + x] + v[yp * kN + x] - 4.f * vc;
  float uvv = uc * vc * vc;
  un[gid] = uc + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - uc));
  vn[gid] = vc + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vc);
}

/**
 * Fused snapshot: copy the evolved grid into the handle's registered buffer and
 * submit PutBlob in one launch (the path Gray Scott wants). Each thread fences
 * its copy system-wide before the barrier so the CPU-side PutBlob's D2H read
 * sees it while the kernel is still resident (Slice 2 proved the fence suffices).
 */
__global__ void GsSnapKernel(kvhdf5::GpuDatasetHandle h, const float *src) {
  CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
  (void)g_ipc_manager;
  const byte_t *s = reinterpret_cast<const byte_t *>(src);
  byte_t *dst = h.Data();
  for (uint64_t i = threadIdx.x; i < h.Size(); i += blockDim.x) dst[i] = s[i];
  __threadfence_system();
  __syncthreads();
  h.Write();  // thread-0 only (internal guard)
}

/** Submit the pre-built GetBlob task from the kernel via the handle. */
__global__ void GsReadKernel(kvhdf5::GpuDatasetHandle h) {
  CLIO_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
  (void)g_ipc_manager;
  h.Read();
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("GPU Gray-Scott end-to-end snapshots via dataset handle",
          "[integration][gpu][cte][grayscott]") {
  auto &env = kvhdf5::itest::SharedCteEnv();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  REQUIRE(ipc->GetGpuQueueCount() >= 1u);

  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

  // Pin THIS host thread to the same GPU the datasets and snapshot kernels use
  // (gpu 0). The shared CLIO runtime initialises every GPU on the node and
  // leaves the current device set to the last one, so on a multi-GPU host the
  // working grids below would otherwise be cudaMalloc'd on a different device
  // than where GsSnapKernel launches — the dataset ctor's SetDevice(0) only
  // runs later, mid-loop — making the kernel's read of `src` an illegal access
  // (CUDA Error 700). Setting it up front keeps grids, dataset buffers, and
  // launches all on gpu 0. (Single-GPU CI never exposed this.)
  ctp::GpuApi::SetDevice(/*gpu_id=*/0);

  // ---- Working grids: 4 plain device buffers, ping-pong u/v. ----
  float *u_curr = nullptr, *u_next = nullptr, *v_curr = nullptr, *v_next = nullptr;
  REQUIRE(cudaMalloc(&u_curr, kBytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&u_next, kBytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&v_curr, kBytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&v_next, kBytes) == cudaSuccess);

  // ---- Seed: classic Gray-Scott IC (u=1 everywhere; v=1 in a centre square). ----
  std::vector<float> u0(kCells, 1.0f), v0(kCells, 0.0f);
  for (unsigned y = kN / 2 - 3; y < kN / 2 + 3; ++y)
    for (unsigned x = kN / 2 - 3; x < kN / 2 + 3; ++x) v0[y * kN + x] = 1.0f;
  ctp::GpuApi::Memcpy(u_curr, u0.data(), kBytes);
  ctp::GpuApi::Memcpy(v_curr, v0.data(), kBytes);

  GsParams params{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

  // ---- Snapshot archive: one GpuCteDataset per snapshot (move-ctor exercise). ----
  std::vector<kvhdf5::GpuCteDataset> snaps;  // intentionally NOT reserved
  std::vector<std::vector<float>> expected;  // host copy of each snapshotted v

  // ---- Step loop. ----
  unsigned threads = 256;
  unsigned blocks = (kCells + threads - 1) / threads;
  for (int step = 1; step <= kSteps; ++step) {
    GsStepKernel<<<blocks, threads>>>(u_curr, v_curr, u_next, v_next, params);
    ctp::GpuApi::Synchronize();
    std::swap(u_curr, u_next);
    std::swap(v_curr, v_next);

    if (step % kSnap != 0) continue;

    // Record the evolved v grid (host) and assert it diverged from the seed.
    std::vector<float> got(kCells);
    ctp::GpuApi::Memcpy(got.data(), v_curr, kBytes);
    REQUIRE(std::memcmp(got.data(), v0.data(), kBytes) != 0);  // compute ran
    expected.push_back(got);

    // Fresh per-step-named dataset; emplace into the (growing) vector.
    std::string name = "snap_v_" + std::to_string(step);
    REQUIRE(name.size() <= kvhdf5::chunking::kMaxBlobNameLen);
    snaps.emplace_back(ipc, gpu_info, /*gpu_id=*/0, env.tag_id, name.c_str(),
                       kBytes);

    // Fused copy-from-grid + PutBlob in one launch.
    GsSnapKernel<<<1, 32>>>(snaps.back().Handle(), v_curr);
    ctp::GpuApi::Synchronize();
    std::fprintf(stderr, "[snap] step %d persisted (%s)\n", step, name.c_str());
  }

  REQUIRE(snaps.size() == expected.size());
  REQUIRE(snaps.size() == static_cast<size_t>(kSteps / kSnap));

  // ---- Read each snapshot back via GetBlob and verify. ----
  std::vector<byte_t> zeros(kBytes);
  for (size_t s = 0; s < snaps.size(); ++s) {
    byte_t *buf = snaps[s].DeviceData();
    ctp::GpuApi::Memcpy(buf, zeros.data(), kBytes);  // clobber so readback is real
    GsReadKernel<<<1, 32>>>(snaps[s].Handle());
    ctp::GpuApi::Synchronize();

    std::vector<byte_t> back(kBytes);
    ctp::GpuApi::Memcpy(back.data(), buf, kBytes);
    if (std::memcmp(back.data(), expected[s].data(), kBytes) != 0)
      std::fprintf(stderr, "[verify] snapshot %zu mismatch\n", s);
    REQUIRE(std::memcmp(back.data(), expected[s].data(), kBytes) == 0);
  }
  std::fprintf(stderr, "[ok] gray-scott %d steps, %zu snapshots round-tripped\n",
               kSteps, snaps.size());

  cudaFree(u_curr);
  cudaFree(u_next);
  cudaFree(v_curr);
  cudaFree(v_next);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
