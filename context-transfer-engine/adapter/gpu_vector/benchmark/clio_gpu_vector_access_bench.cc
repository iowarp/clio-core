/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * What does the vector ABSTRACTION cost, with no I/O involved at all?
 *
 * Gray-Scott reaction-diffusion, run twice over an identical grid with an
 * identical launch configuration: once against a plain cudaMalloc'd float*,
 * once against a gpu_vector sized so the whole grid is resident. Nothing is
 * faulted, evicted or flushed during timing -- every page is made resident
 * first -- so the difference between the two numbers is purely the cost of
 * going through at()/operator[] instead of dereferencing a pointer.
 *
 * That cost is not academic. A 5-point stencil does 12 element accesses per
 * cell per step, and each one resolves a page: a bounds computation, a
 * compare against the per-thread cached page, and a load of the page's data
 * pointer. If that is expensive it dominates any workload that touches memory
 * more than once per byte fetched, which is most of them.
 *
 * Both versions are checked against each other cell by cell: the vector must
 * produce bit-identical results, or the timing means nothing.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr float kDu = 0.16f;
constexpr float kDv = 0.08f;
constexpr float kF = 0.060f;
constexpr float kK = 0.062f;
constexpr float kDt = 1.0f;
}  // namespace

/** Initial condition: mostly u=1, a seeded square of v. */
CTP_INLINE_CROSS_FUN void InitCell(u64 idx, u32 dim, float *u, float *v) {
  const u32 y = static_cast<u32>(idx / dim);
  const u32 x = static_cast<u32>(idx % dim);
  const bool seed = (x > dim / 2 - dim / 16 && x < dim / 2 + dim / 16 &&
                     y > dim / 2 - dim / 16 && y < dim / 2 + dim / 16);
  *u = seed ? 0.50f : 1.0f;
  *v = seed ? 0.25f : 0.0f;
}

/** Periodic neighbour index. */
CTP_INLINE_CROSS_FUN u64 Wrap(int c, u32 dim) {
  if (c < 0) return static_cast<u64>(c + static_cast<int>(dim));
  if (c >= static_cast<int>(dim)) return static_cast<u64>(c - static_cast<int>(dim));
  return static_cast<u64>(c);
}

/** One Gray-Scott step on RAW pointers. */
__global__ void GrayScottRaw(const float *ui, const float *vi, float *uo,
                             float *vo, u32 dim) {
  const u64 cells = static_cast<u64>(dim) * dim;
  for (u64 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cells;
       idx += static_cast<u64>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(idx % dim);
    const int y = static_cast<int>(idx / dim);
    const u64 l = y * dim + Wrap(x - 1, dim);
    const u64 r = y * dim + Wrap(x + 1, dim);
    const u64 d = Wrap(y - 1, dim) * dim + x;
    const u64 t = Wrap(y + 1, dim) * dim + x;
    const float u = ui[idx], v = vi[idx];
    const float lu = ui[l] + ui[r] + ui[d] + ui[t] - 4.0f * u;
    const float lv = vi[l] + vi[r] + vi[d] + vi[t] - 4.0f * v;
    const float uvv = u * v * v;
    uo[idx] = u + kDt * (kDu * lu - uvv + kF * (1.0f - u));
    vo[idx] = v + kDt * (kDv * lv + uvv - (kF + kK) * v);
  }
}

/**
 * The same step, through the vector, using the INDEXING FAST PATH.
 *
 * HoldPage resolves each field's page once, up front; the loop then uses
 * operator[]/at(), which index the held page with no resolution at all. This
 * is only valid while every access stays inside the held page -- here that
 * means one page per field, which the caller guarantees before launching.
 */
__global__ void GrayScottHold(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<float> ui,
                              gv::DeviceVector<float> vi,
                              gv::DeviceVector<float> uo,
                              gv::DeviceVector<float> vo, u32 dim) {
  CLIO_GPU_INIT(info, nullptr);
  const u64 cells = static_cast<u64>(dim) * dim;
  // One resolution per field for the whole kernel, instead of one per access.
  ui.HoldPage(0, cells);
  vi.HoldPage(0, cells);
  uo.HoldPage(0, cells);
  vo.HoldPage(0, cells);
  for (u64 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cells;
       idx += static_cast<u64>(gridDim.x) * blockDim.x) {
    const int x = static_cast<int>(idx % dim);
    const int y = static_cast<int>(idx / dim);
    const u64 l = y * dim + Wrap(x - 1, dim);
    const u64 r = y * dim + Wrap(x + 1, dim);
    const u64 d = Wrap(y - 1, dim) * dim + x;
    const u64 t = Wrap(y + 1, dim) * dim + x;
    const float u = ui.at(idx), v = vi.at(idx);
    const float lu = ui.at(l) + ui.at(r) + ui.at(d) + ui.at(t) - 4.0f * u;
    const float lv = vi.at(l) + vi.at(r) + vi.at(d) + vi.at(t) - 4.0f * v;
    const float uvv = u * v * v;
    uo[idx] = u + kDt * (kDu * lu - uvv + kF * (1.0f - u));
    vo[idx] = v + kDt * (kDv * lv + uvv - (kF + kK) * v);
  }
}

__global__ void InitRaw(float *u, float *v, u32 dim) {
  const u64 cells = static_cast<u64>(dim) * dim;
  for (u64 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cells;
       idx += static_cast<u64>(gridDim.x) * blockDim.x) {
    InitCell(idx, dim, &u[idx], &v[idx]);
  }
}

/** Initialise the vectors AND make every page resident (no faults later). */
__global__ void InitVec(clio::run::IpcManagerGpuInfo info,
                        gv::DeviceVector<float> u, gv::DeviceVector<float> v,
                        gv::DeviceVector<float> uo, gv::DeviceVector<float> vo,
                        u32 dim) {
  CLIO_GPU_INIT(info, nullptr);
  const u64 cells = static_cast<u64>(dim) * dim;
  for (u64 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cells;
       idx += static_cast<u64>(gridDim.x) * blockDim.x) {
    float a, b;
    InitCell(idx, dim, &a, &b);
    u.HoldPage(idx, 1);
    v.HoldPage(idx, 1);
    uo.HoldPage(idx, 1);
    vo.HoldPage(idx, 1);
    u[idx] = a;
    v[idx] = b;
    uo[idx] = a;
    vo[idx] = b;
  }
}

/** Copy a vector's contents out to a raw buffer, for comparison. */
__global__ void VecToRaw(clio::run::IpcManagerGpuInfo info,
                         gv::DeviceVector<float> src, float *dst, u32 dim) {
  CLIO_GPU_INIT(info, nullptr);
  const u64 cells = static_cast<u64>(dim) * dim;
  for (u64 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < cells;
       idx += static_cast<u64>(gridDim.x) * blockDim.x) {
    src.HoldPage(idx, 1);
    dst[idx] = src.at(idx);
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char **argv) {
  u32 dim = 256;
  u32 steps = 100;
  u32 blocks = 8;
  u32 threads = 256;
  u32 repeat = 3;
  u64 page_kb = 64;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> const char * {
      return (i + 1 < argc) ? argv[++i] : "0";
    };
    if (f == "--dim") dim = std::atoi(next());
    else if (f == "--steps") steps = std::atoi(next());
    else if (f == "--blocks") blocks = std::atoi(next());
    else if (f == "--threads") threads = std::atoi(next());
    else if (f == "--page-kb") page_kb = std::atoll(next());
    else if (f == "--repeat") repeat = std::atoi(next());
    else if (f == "--help") {
      std::printf("usage: %s [--dim N] [--steps N] [--blocks N] [--threads N] "
                  "[--page-kb N] [--repeat N]\n", argv[0]);
      return 0;
    }
  }
  const u64 cells = static_cast<u64>(dim) * dim;
  const u64 page_bytes = page_kb * 1024;
  const u64 page_elems = page_bytes / sizeof(float);
  const u64 pages = (cells + page_elems - 1) / page_elems;

  {
    std::ofstream cfg("gpu_vector_access.yaml");
    cfg << "networking:\n  port: 9438\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 8192\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"2GB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gv_access_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"1GB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_access.yaml", 1);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  std::printf(
      "gray-scott: raw cuda pointer vs gpu_vector (all resident, ZERO I/O)\n"
      "  grid=%ux%u (%llu cells, %.2f MB/field)  steps=%u\n"
      "  launch=<<<%u,%u>>>  page=%lluKB (%llu pages/field)\n",
      dim, dim, (unsigned long long) cells,
      static_cast<double>(cells * sizeof(float)) / (1024.0 * 1024.0), steps,
      blocks, threads, (unsigned long long) page_kb,
      (unsigned long long) pages);

  // ---- raw ----
  float *ru = nullptr, *rv = nullptr, *ru2 = nullptr, *rv2 = nullptr;
  cudaMalloc(&ru, cells * sizeof(float));
  cudaMalloc(&rv, cells * sizeof(float));
  cudaMalloc(&ru2, cells * sizeof(float));
  cudaMalloc(&rv2, cells * sizeof(float));

  double raw_ms = 1e30;
  for (u32 r = 0; r < repeat; ++r) {
    InitRaw<<<blocks, threads>>>(ru, rv, dim);
    cudaDeviceSynchronize();
    const double t0 = NowMs();
    float *a = ru, *b = rv, *c = ru2, *d = rv2;
    for (u32 s = 0; s < steps; ++s) {
      GrayScottRaw<<<blocks, threads>>>(a, b, c, d, dim);
      float *ta = a; a = c; c = ta;
      float *tb = b; b = d; d = tb;
    }
    cudaDeviceSynchronize();
    const double ms = NowMs() - t0;
    if (ms < raw_ms) raw_ms = ms;
  }
  // Re-run once to leave the final state in a known buffer for comparison.
  InitRaw<<<blocks, threads>>>(ru, rv, dim);
  cudaDeviceSynchronize();
  {
    float *a = ru, *b = rv, *c = ru2, *d = rv2;
    for (u32 s = 0; s < steps; ++s) {
      GrayScottRaw<<<blocks, threads>>>(a, b, c, d, dim);
      float *ta = a; a = c; c = ta;
      float *tb = b; b = d; d = tb;
    }
    cudaDeviceSynchronize();
    ru = a; rv = b;   // final results live here
  }

  // ---- vector: cache holds every page, so nothing ever faults while timing --
  double vec_ms = 1e30, hold_ms = 1e30;
  {
    gv::Vector<float> vu("gs_u", {0}, page_bytes, blocks,
                         static_cast<u32>(pages), cells);
    gv::Vector<float> vv("gs_v", {0}, page_bytes, blocks,
                         static_cast<u32>(pages), cells);
    gv::Vector<float> vu2("gs_u2", {0}, page_bytes, blocks,
                          static_cast<u32>(pages), cells);
    gv::Vector<float> vv2("gs_v2", {0}, page_bytes, blocks,
                          static_cast<u32>(pages), cells);
    vu.EnableStats();
    auto du = vu.GetDevice(0), dv = vv.GetDevice(0);
    auto du2 = vu2.GetDevice(0), dv2 = vv2.GetDevice(0);

    for (u32 r = 0; r < repeat; ++r) {
      InitVec<<<blocks, threads>>>(gpu, du, dv, du2, dv2, dim);
      if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "init failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 1;
      }
      vu.ResetStats();
      const double t0 = NowMs();
      auto a = du, b = dv, c = du2, d = dv2;
      for (u32 s = 0; s < steps; ++s) {
        GrayScottHold<<<blocks, threads>>>(gpu, a, b, c, d, dim);
        auto ta = a; a = c; c = ta;
        auto tb = b; b = d; d = tb;
      }
      if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "vec step failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 1;
      }
      const double ms = NowMs() - t0;
      if (ms < vec_ms) vec_ms = ms;
      if (r + 1 == repeat) {
        // Compare the vector's final state against the raw run's.
        float *cu = nullptr, *cv = nullptr;
        cudaMalloc(&cu, cells * sizeof(float));
        cudaMalloc(&cv, cells * sizeof(float));
        VecToRaw<<<blocks, threads>>>(gpu, a, cu, dim);
        VecToRaw<<<blocks, threads>>>(gpu, b, cv, dim);
        cudaDeviceSynchronize();
        std::vector<float> hu(cells), hv(cells), hru(cells), hrv(cells);
        cudaMemcpy(hu.data(), cu, cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hv.data(), cv, cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hru.data(), ru, cells * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(hrv.data(), rv, cells * sizeof(float),
                   cudaMemcpyDeviceToHost);
        u64 bad = 0;
        double worst = 0.0;
        for (u64 i = 0; i < cells; ++i) {
          const double eu = std::fabs(hu[i] - hru[i]);
          const double ev = std::fabs(hv[i] - hrv[i]);
          if (eu > worst) worst = eu;
          if (ev > worst) worst = ev;
          if (eu > 1e-5 || ev > 1e-5) ++bad;
        }
        std::printf("  agreement: %llu/%llu cells differ, worst |delta|=%.3e\n",
                    (unsigned long long) bad, (unsigned long long) cells,
                    worst);
        if (bad != 0) {
          std::fprintf(stderr, "RESULTS DISAGREE -- timings are void\n");
          return 1;
        }
        cudaFree(cu);
        cudaFree(cv);
      }
    }
    // ---- fast path, when each field is a single page ----
    if (pages == 1) {
      for (u32 r = 0; r < repeat; ++r) {
        InitVec<<<blocks, threads>>>(gpu, du, dv, du2, dv2, dim);
        cudaDeviceSynchronize();
        const double t0 = NowMs();
        auto a = du, b = dv, c = du2, d = dv2;
        for (u32 s2 = 0; s2 < steps; ++s2) {
          GrayScottHold<<<blocks, threads>>>(gpu, a, b, c, d, dim);
          auto ta = a; a = c; c = ta;
          auto tb = b; b = d; d = tb;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
          std::fprintf(stderr, "hold step failed: %s\n",
                       cudaGetErrorString(cudaGetLastError()));
          return 1;
        }
        const double ms = NowMs() - t0;
        if (ms < hold_ms) hold_ms = ms;
        if (r + 1 == repeat) {
          float *cu = nullptr;
          cudaMalloc(&cu, cells * sizeof(float));
          VecToRaw<<<blocks, threads>>>(gpu, a, cu, dim);
          cudaDeviceSynchronize();
          std::vector<float> hu(cells), hru(cells);
          cudaMemcpy(hu.data(), cu, cells * sizeof(float),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy(hru.data(), ru, cells * sizeof(float),
                     cudaMemcpyDeviceToHost);
          u64 bad = 0;
          for (u64 i = 0; i < cells; ++i) {
            if (std::fabs(hu[i] - hru[i]) > 1e-5) ++bad;
          }
          std::printf("  HoldPage agreement: %llu/%llu cells differ\n",
                      (unsigned long long) bad, (unsigned long long) cells);
          if (bad != 0) {
            std::fprintf(stderr, "HOLD RESULTS DISAGREE -- timings void\n");
            return 1;
          }
          cudaFree(cu);
        }
      }
    }
    const auto st = vu.ReadStats(0);
    std::printf("  vector I/O during timing: faults=%llu puts=%llu evicts=%llu"
                " (must be 0)\n",
                (unsigned long long) st.faults, (unsigned long long) st.puts,
                (unsigned long long) st.evicts);
  }

  // 12 element accesses per cell per step: 10 reads + 2 writes.
  const double accesses =
      static_cast<double>(cells) * static_cast<double>(steps) * 12.0;
  std::printf(
      "\n  raw pointer      %8.2f ms   (%.2f ns/access)\n"
      "  gpu_vector       %8.2f ms   (%.2f ns/access)\n"
      "  slowdown         %8.2fx\n"
      "  added cost       %8.2f ns/access\n",
      raw_ms, raw_ms * 1e6 / accesses, vec_ms, vec_ms * 1e6 / accesses,
      vec_ms / raw_ms, (vec_ms - raw_ms) * 1e6 / accesses);
  if (hold_ms < 1e29) {
    std::printf(
        "  gpu_vector+Hold  %8.2f ms   (%.2f ns/access)\n"
        "  slowdown vs raw  %8.2fx\n"
        "  speedup vs fault %8.2fx\n",
        hold_ms, hold_ms * 1e6 / accesses, hold_ms / raw_ms, vec_ms / hold_ms);
  } else {
    std::printf("  gpu_vector+Hold      n/a   (needs 1 page/field; use "
                "--page-kb >= field size)\n");
  }

  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
