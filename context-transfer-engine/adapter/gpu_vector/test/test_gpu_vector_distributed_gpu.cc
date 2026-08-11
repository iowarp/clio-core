/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Distributed compressed GPU vector, checkpointing a REAL Gray-Scott field:
 * evolve a Gray-Scott reaction-diffusion simulation on one GPU, checkpoint the
 * field by compressing its pages IN HBM and storing them through the compressor
 * into a cte_core pool whose storage spans MULTIPLE physical nodes, then read
 * them back to the GPU, decompress, and verify against the pre-checkpoint field.
 *
 * Gray-Scott output is not analytic, so verification compares the decompressed
 * readback to a host copy of the field taken right before the checkpoint (the
 * honest check for lossy compression of real data) -- and reports the ACTUAL
 * compression ratio on evolved simulation data (lower than a smooth synthetic
 * field, which is the point of using GS).
 *
 * This is the single-GPU compressed vector's PutBlob/GetBlob path (compressor
 * methods 15/16) run against a distributed cte_core. It works only because the
 * compressor now HASH-ROUTES its forward store (CLIO_CTE_COMPRESS_DISTRIBUTE=1,
 * see compressor_runtime.cc ForwardQuery) -- otherwise every compressed page
 * would pile up on this node's local container.
 *
 * Runs in SERVER mode as node 0 of the cluster (so the compressor is in-process
 * and a device pointer allocated here is valid when it reads it -- a raw device
 * address is not shareable to a separate daemon process). Peer nodes run plain
 * daemons hosting cte_core storage. Driven by
 * test/integration/distributed_slurm/run_slurm_gpu.sbatch (CTE_CLIENT_MODE=
 * server_rank0). With CTE_NUM_NODES=1 (no hostfile) the same binary is a
 * standalone single-GPU checkpoint -- used by run_4gpu.sbatch, which runs one
 * instance per GPU (CUDA_VISIBLE_DEVICES) to exercise all 4 GPUs on a node.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <clio_ctp/util/gpu_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {

// Compressor pool (600) wraps the distributed cte_core (512). Tag ops go to the
// core; per-page Put/Get go to the compressor, which compresses in HBM and
// forwards (hash-routed) to the core.
inline clio::run::PoolId CompressorPool() { return clio::run::PoolId(600, 0); }

// A device pointer wrapped as a null-alloc ShmPtr: off_ carries the raw device
// address, so the compressor reads/writes HBM directly (zero DRAM staging).
ctp::ipc::ShmPtr<> DevPtr(void *dev) {
  ctp::ipc::ShmPtr<> p;
  p.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  p.off_ = reinterpret_cast<clio::run::u64>(dev);
  return p;
}

int EnvInt(const char *k, int d) { const char *v = std::getenv(k); return v ? std::atoi(v) : d; }

}  // namespace

// ---- Gray-Scott reaction-diffusion (same kernels as clio_gs_checkpoint_bench) --
__global__ void GsInit(float *u, float *v, int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int idx = row * cols + col;
    unsigned int hu = (unsigned int)idx * 2654435761u + 12345u;
    unsigned int hv = (unsigned int)idx * 40503u + 6789u;
    const float nu = (float)((hu >> 9) & 0xffff) / 65535.0f - 0.5f;
    const float nv = (float)((hv >> 9) & 0xffff) / 65535.0f - 0.5f;
    u[idx] = 0.5f + 0.40f * nu;
    v[idx] = 0.25f + 0.20f * nv;
  }
}

__global__ void GsStep(const float *__restrict__ u, const float *__restrict__ v,
                       float *__restrict__ u2, float *__restrict__ v2, int rows,
                       int cols, float Du, float Dv, float F, float k, float dt) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int idx = row * cols + col;
    const int up = (row > 0 ? row - 1 : row) * cols + col;
    const int dn = (row < rows - 1 ? row + 1 : row) * cols + col;
    const int lf = row * cols + (col > 0 ? col - 1 : col);
    const int rt = row * cols + (col < cols - 1 ? col + 1 : col);
    const float uc = u[idx], vc = v[idx];
    const float lap_u = u[up] + u[dn] + u[lf] + u[rt] - 4.0f * uc;
    const float lap_v = v[up] + v[dn] + v[lf] + v[rt] - 4.0f * vc;
    const float uvv = uc * vc * vc;
    u2[idx] = uc + (Du * lap_u - uvv + F * (1.0f - uc)) * dt;
    v2[idx] = vc + (Dv * lap_v + uvv - (F + k) * vc) * dt;
  }
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: Gray-Scott checkpoint compressed pages distribute across "
          "a multi-node cte_core and round-trip within the error bound",
          "[gpu_vector][compress][distributed]") {
  const int nnodes = EnvInt("CTE_NUM_NODES", 1);
  const int rank = EnvInt("CTE_RANK", 0);
  const int steps = EnvInt("CLIO_GS_STEPS", 200);
  std::fprintf(stderr, "[DIST] rank=%d/%d gs_steps=%d\n", rank, nnodes, steps);

  // SERVER mode: this process IS node 0 of the cluster and hosts the compressor
  // in-process, so a device pointer we allocate here is valid when the
  // compressor reads it (a raw device address is NOT shareable to a separate
  // daemon process without CUDA IPC handles). The compose + hostfile come from
  // CLIO_SERVER_CONF (written by run_slurm_gpu.sbatch); the hostfile makes
  // cte_core span all nodes, so the compressor's hash-routed forward distributes
  // the compressed pages across the cluster.
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);

  // Report the physical GPU this instance is bound to (via CUDA_VISIBLE_DEVICES).
  // In the 4-GPU-per-node run this makes each of the 4 instances identifiable.
  {
    int dev = 0; char pci[32] = {0}; cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    cudaDeviceGetPCIBusId(pci, sizeof(pci), dev);
    std::fprintf(stderr, "[DIST] GPU: %s pci=%s (CUDA_VISIBLE_DEVICES=%s)\n",
                 prop.name, pci,
                 std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "?");
  }

  const clio::run::u64 page_size = 256ULL * 1024;      // >=256 KiB: cuSZp-correct
  const clio::run::u64 epp = page_size / sizeof(float);
  // Pages per GPU (dataset size). Env-configurable so a weak-scaling study can
  // grow the per-GPU checkpoint (default 64 pages = 16 MiB keeps the original
  // correctness-test size). cols stays 2048; rows scales with K.
  const clio::run::u32 K = (clio::run::u32)EnvInt("CLIO_DIST_PAGES", 64);
  const clio::run::u64 total = (clio::run::u64)K * epp; // field elements
  const int cols = 2048;                                // 2048x2048 = 4M = K*epp
  const int rows = (int)(total / cols);
  const char *tag = "dist";

  // ---- Evolve a real Gray-Scott field in HBM ----
  float *u = nullptr, *v = nullptr, *u2 = nullptr, *v2 = nullptr;
  const clio::run::u64 field_bytes = total * sizeof(float);
  REQUIRE(cudaMalloc(&u, field_bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&v, field_bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&u2, field_bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&v2, field_bytes) == cudaSuccess);
  GsInit<<<rows, 256>>>(u, v, rows, cols);
  ctp::GpuApi::Synchronize();
  for (int s = 0; s < steps; ++s) {
    GsStep<<<rows, 256>>>(u, v, u2, v2, rows, cols, 0.16f, 0.08f, 0.060f,
                          0.062f, 1.0f);
    std::swap(u, u2);
    std::swap(v, v2);
  }
  ctp::GpuApi::Synchronize();

  // Retain the pre-checkpoint field on the host: GS is not analytic, so this is
  // the reference the lossy readback is verified against.
  auto *ref = static_cast<float *>(std::malloc(field_bytes));
  REQUIRE(ref != nullptr);
  cudaMemcpy(ref, u, field_bytes, cudaMemcpyDeviceToHost);
  std::fprintf(stderr, "[DIST] evolved Gray-Scott %dx%d for %d steps\n", rows,
               cols, steps);

  // Tag lives on the core (kCtePoolId == 512).
  clio::cte::core::Client core;
  core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag(tag);
  tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;

  clio::cte::core::Client comp;
  comp.Init(CompressorPool());
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;      // static-pin the compressor (env picks cuszp)
  ctx.compress_preset_ = 2;       // BALANCED -> abs error bound 1e-3

  // ---- Checkpoint: compress each page of U IN HBM, store (fan out) ----
  auto store_t0 = std::chrono::steady_clock::now();
  for (clio::run::u32 P = 0; P < K; ++P) {
    std::string name = std::string(tag) + "_b0_pi" + std::to_string(P);
    auto pf = comp.AsyncPutBlob(tag_id, name, (clio::run::u64)0, page_size,
                                DevPtr(u + (clio::run::u64)P * epp), 0.5f, ctx, 0,
                                clio::run::PoolQuery::Local());
    pf.Wait();
    REQUIRE(pf->GetReturnCode() == 0);
  }
  double store_dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - store_t0).count();
  std::fprintf(stderr,
               "[DIST] checkpointed %u compressed GS pages (%lluMiB) across %d "
               "nodes\n",
               K, (unsigned long long)(field_bytes >> 20), nnodes);

  // ---- Read each page back to HBM, decompress, verify vs the host reference --
  float *readbuf = nullptr;
  REQUIRE(cudaMalloc(&readbuf, page_size) == cudaSuccess);
  auto *host = static_cast<float *>(std::malloc(page_size));
  double max_err = 0.0;
  double read_io_dt = 0.0;  // GetBlob (fetch+decompress) time only, excl. verify
  for (clio::run::u32 P = 0; P < K; ++P) {
    cudaMemset(readbuf, 0, page_size);
    std::string name = std::string(tag) + "_b0_pi" + std::to_string(P);
    auto rd_t0 = std::chrono::steady_clock::now();
    auto gf = comp.AsyncGetBlob(tag_id, name, (clio::run::u64)0, page_size,
                                /*flags=*/0, DevPtr(readbuf),
                                clio::run::PoolQuery::Local());
    gf.Wait();
    read_io_dt += std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - rd_t0).count();
    REQUIRE(gf->GetReturnCode() == 0);
    cudaMemcpy(host, readbuf, page_size, cudaMemcpyDeviceToHost);
    for (clio::run::u64 j = 0; j < epp; ++j)
      max_err = std::max(max_err, std::fabs((double)host[j] -
                                            (double)ref[(clio::run::u64)P * epp + j]));
  }
  std::fprintf(stderr,
               "[DIST] read back %u GS pages from the distributed store; "
               "max_abs_err=%.3e vs pre-checkpoint field (eb=1e-3)\n",
               K, max_err);
  // Per-GPU throughput (this rank). An external harness sums these across GPUs
  // for aggregate/scaling-efficiency numbers.
  const double mib = (double)(field_bytes) / (1024.0 * 1024.0);
  std::fprintf(stderr,
               "[DIST][THROUGHPUT] rank=%d nnodes=%d pages=%u field_mib=%.1f "
               "store_s=%.4f store_mibps=%.1f read_io_s=%.4f read_mibps=%.1f\n",
               rank, nnodes, K, mib, store_dt,
               store_dt > 0 ? mib / store_dt : 0.0, read_io_dt,
               read_io_dt > 0 ? mib / read_io_dt : 0.0);
  REQUIRE(max_err <= 2.0e-3);

  std::free(host);
  std::free(ref);
  cudaFree(readbuf);
  cudaFree(u); cudaFree(v); cudaFree(u2); cudaFree(v2);
  std::fprintf(stderr,
               "[DIST] PASS: Gray-Scott field checkpointed (%u pages compressed "
               "in HBM), stored across %d nodes, read back + decompressed within "
               "the error bound.\n",
               K, nnodes);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
