/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Gray-Scott checkpoint benchmark: traditional storage path vs. compressed GPU
 * vector.
 *
 * Mirrors the canonical GPU Gray-Scott loop (iterate N steps, checkpoint the
 * field every `interval` steps -- like the iowarp-gray-scott / HDF5 checkpoint
 * pattern), and checkpoints each snapshot TWO ways, timing both:
 *
 *   TRADITIONAL (what an HDF5/kvhdf5 checkpoint does): cudaMemcpy the field
 *     device->host (pinned), then write the full uncompressed field to a file on
 *     the parallel filesystem. Data leaves the GPU; footprint = full size on
 *     disk.
 *
 *   COMPRESSED GPU VECTOR: a CLIO PutBlob through the compressor chimod
 *     (CLIO_CTE_COMPRESS_LIB, e.g. cuszp) whose storage tier is a kHbm bdev.
 *     The field is compressed IN HBM (zero host copy) and the compressed blob
 *     stays in GPU memory. Footprint = compressed size in HBM.
 *
 * Reports per-checkpoint latency, effective checkpoint bandwidth (logical
 * bytes / time), bytes written, and where the data lives -- so the two paths
 * compare apples-to-apples on the same evolving simulation snapshots.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---- Gray-Scott kernels (real reaction-diffusion) ----------------------------
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

// Zero-copy ShmPtr wrapping a raw device address (null alloc id).
ctp::ipc::ShmPtr<> DevPtr(void *dev) {
  ctp::ipc::ShmPtr<> p;
  p.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  p.off_ = reinterpret_cast<clio::run::u64>(dev);
  return p;
}

constexpr clio::run::u64 kHbmBudgetMiB = 512;

void WriteCompose(int port) {
  std::string cfg = "/tmp/gs_ckpt_" + std::to_string(port) + ".yaml";
  std::ofstream f(cfg);
  f << "networking:\n  port: " << port << "\n\n"
    << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
    << "compose:\n"
    << "  - mod_name: clio_bdev\n"
    << "    pool_name: \"hbm::chi_default_bdev\"\n"
    << "    pool_query: local\n    pool_id: \"301.0\"\n"
    << "    bdev_type: hbm\n    capacity: \"" << kHbmBudgetMiB << "MB\"\n\n"
    << "  - mod_name: clio_cte_compressor\n"
    << "    pool_name: cte_compressor\n    pool_query: local\n"
    << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
    << "  - mod_name: clio_cte_core\n"
    << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
    << "    storage:\n      - path: \"hbm::cte_hbm_tier\"\n"
    << "        bdev_type: \"hbm\"\n        capacity_limit: \"" << kHbmBudgetMiB
    << "MB\"\n        score: 1.0\n"
    << "    dpe:\n      dpe_type: \"max_bw\"\n";
  f.close();
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
}

}  // namespace

int main(int argc, char **argv) {
#if !CTP_IS_DEVICE_PASS
  // Field geometry: rows x cols floats. Default ~64 MiB per field.
  int rows = (argc > 1) ? std::atoi(argv[1]) : 4096;
  int cols = (argc > 2) ? std::atoi(argv[2]) : 4096;
  int steps = (argc > 3) ? std::atoi(argv[3]) : 200;
  int interval = (argc > 4) ? std::atoi(argv[4]) : 25;   // checkpoint every N steps
  const char *trad_dir = (argc > 5) ? argv[5] : "/u/rpawar/gsbench/ckpt";

  const clio::run::u64 n = (clio::run::u64)rows * cols;
  const clio::run::u64 field_bytes = n * sizeof(float);
  const int nckpt = steps / interval;

  std::fprintf(stderr,
      "[CKPT] field=%dx%d (%.1f MiB)  steps=%d  interval=%d  checkpoints=%d\n",
      rows, cols, field_bytes / (1024.0 * 1024.0), steps, interval, nckpt);

  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10580;
  WriteCompose(port);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "[CKPT] CLIO_INIT failed\n");
    return 2;
  }
  std::this_thread::sleep_for(1s);

  const clio::run::PoolId kCompressor(600, 0), kCore(512, 0), kHbmTier(512, 1);
  clio::cte::core::Client comp;
  comp.Init(kCompressor);
  clio::cte::core::Client core;
  core.Init(kCore);
  auto tagf = core.AsyncGetOrCreateTag("gs_ckpt");
  tagf.Wait();
  auto tag_id = tagf->tag_id_;

  // Which compressor library (env pin), preset best-effort.
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  const char *lib = std::getenv("CLIO_CTE_COMPRESS_LIB");
  ctx.compress_preset_ = 2;

  // GS device buffers (double-buffered).
  float *u, *v, *u2, *v2;
  cudaMalloc(&u, field_bytes);
  cudaMalloc(&v, field_bytes);
  cudaMalloc(&u2, field_bytes);
  cudaMalloc(&v2, field_bytes);
  GsInit<<<rows, 256>>>(u, v, rows, cols);
  cudaDeviceSynchronize();

  float *host_pinned = nullptr;
  cudaMallocHost(&host_pinned, field_bytes);

  clio::run::bdev::Client hbm(kHbmTier);
  auto sb = hbm.AsyncGetStats(); sb.Wait();
  clio::run::u64 hbm_remaining0 = sb->remaining_size_;

  // Accumulators.
  double trad_d2h_ms = 0, trad_write_ms = 0, comp_ms = 0;
  clio::run::u64 trad_bytes = 0, comp_bytes = 0;
  int ck = 0;

  for (int s = 0; s < steps; ++s) {
    GsStep<<<rows, 256>>>(u, v, u2, v2, rows, cols, 0.16f, 0.08f, 0.060f,
                          0.062f, 1.0f);
    std::swap(u, u2);
    std::swap(v, v2);
    if ((s + 1) % interval != 0) continue;
    cudaDeviceSynchronize();

    // ---- TRADITIONAL: D2H + write full field to a PFS file ----
    auto t0 = Clock::now();
    cudaMemcpy(host_pinned, u, field_bytes, cudaMemcpyDeviceToHost);
    trad_d2h_ms += ms_since(t0);
    auto t1 = Clock::now();
    std::string path = std::string(trad_dir) + "_" + std::to_string(port) +
                       "_" + std::to_string(ck) + ".bin";
    {
      std::ofstream of(path, std::ios::binary | std::ios::trunc);
      of.write(reinterpret_cast<const char *>(host_pinned),
               (std::streamsize)field_bytes);
      of.flush();
    }
    trad_write_ms += ms_since(t1);
    trad_bytes += field_bytes;
    std::remove(path.c_str());  // don't fill the FS; we only need the timing

    // ---- COMPRESSED GPU VECTOR: PutBlob through compressor -> kHbm ----
    std::string bname = "gs_ckpt_" + std::to_string(ck);
    auto t2 = Clock::now();
    auto pf = comp.AsyncPutBlob(tag_id, bname, (clio::run::u64)0, field_bytes,
                                DevPtr(u), 0.5f, ctx, 0,
                                clio::run::PoolQuery::Local());
    pf.Wait();
    comp_ms += ms_since(t2);
    ++ck;
  }

  auto sa = hbm.AsyncGetStats(); sa.Wait();
  clio::run::u64 hbm_used = (hbm_remaining0 >= sa->remaining_size_)
                                ? (hbm_remaining0 - sa->remaining_size_) : 0;
  comp_bytes = hbm_used;

  auto GBs = [](clio::run::u64 bytes, double ms) {
    return ms > 0 ? (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0) : 0.0;
  };
  double trad_ms = trad_d2h_ms + trad_write_ms;
  clio::run::u64 logical_total = (clio::run::u64)ck * field_bytes;

  std::fprintf(stderr, "\n==================== CHECKPOINT COMPARISON ==========="
                       "=======\n");
  std::fprintf(stderr, "checkpoints: %d   logical per ckpt: %.1f MiB   lib: %s\n",
               ck, field_bytes / (1024.0 * 1024.0), lib ? lib : "(none)");
  std::fprintf(stderr,
      "TRADITIONAL (D2H + PFS file, uncompressed, OFF-GPU):\n"
      "  d2h=%.1fms  write=%.1fms  total=%.1fms  (%.2f ms/ckpt)\n"
      "  effective ckpt BW = %.2f GiB/s   wrote %.1f MiB to disk\n",
      trad_d2h_ms, trad_write_ms, trad_ms, trad_ms / (ck ? ck : 1),
      GBs(logical_total, trad_ms), trad_bytes / (1024.0 * 1024.0));
  std::fprintf(stderr,
      "COMPRESSED GPU VECTOR (compress in HBM, ON-GPU):\n"
      "  total=%.1fms  (%.2f ms/ckpt)\n"
      "  effective ckpt BW = %.2f GiB/s   HBM footprint %.1f MiB  (ratio %.1fx)\n",
      comp_ms, comp_ms / (ck ? ck : 1), GBs(logical_total, comp_ms),
      comp_bytes / (1024.0 * 1024.0),
      comp_bytes ? (double)logical_total / (double)comp_bytes : 0.0);
  std::fprintf(stderr,
      "SPEEDUP (traditional/compressed): %.2fx   FOOTPRINT REDUCTION: %.1fx   "
      "location: disk -> HBM\n",
      comp_ms > 0 ? trad_ms / comp_ms : 0.0,
      comp_bytes ? (double)trad_bytes / (double)comp_bytes : 0.0);
  std::fprintf(stderr, "======================================================="
                       "=====\n");

  cudaFree(u); cudaFree(v); cudaFree(u2); cudaFree(v2);
  cudaFreeHost(host_pinned);
  return 0;
#else
  (void)argc; (void)argv;
  return 0;
#endif
}

#else
int main() { return 0; }
#endif
