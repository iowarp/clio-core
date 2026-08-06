/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Compressed page-fault path stress benchmark.
 *
 * Emulates the llama.cpp weight-streaming workload in miniature: a "model"
 * of L layers x E slices ("experts") of S bytes; each "token" selects k
 * slices per layer with a skewed (zipf-ish) distribution, fetches them into
 * a device staging ring through Vector::FetchSpanDevice, and runs a reduce
 * kernel over the bytes (the dequant-read stand-in). Correctness is checked
 * against a host-computed checksum.
 *
 * The CTE compose has an hbm:: tier (REAL device memory) capped at
 * --hbm-mb and a ram:: tier below it. The SAME cap serves both modes, so
 * the only variable is compression:
 *   uncompressed : pages stored raw; hbm hits are D2D copies, ram spills
 *                  cross PCIe uncompressed.
 *   compressed   : pages stored nvcomp-lz4; hbm hits DECOMPRESS VRAM->VRAM,
 *                  ram spills cross PCIe compressed.
 *
 * Cases (pick sizes with --case, or override the knobs directly):
 *   a: data fits the tier uncompressed; compress anyway. Shows device bytes
 *      saved vs. the decompress cost.
 *   b: data too large uncompressed, fits compressed. The compressed mode
 *      serves every fault from device memory; uncompressed spills.
 *   c: data too large even compressed. Hot slices are seeded FIRST so the
 *      score-1.0 hbm tier fills with the best pages (prefetch-informed
 *      placement); the skewed access stream then faults mostly from hbm.
 *
 * Reports: device bytes consumed by the tier (cudaMemGetInfo delta across
 * seeding), host RSS delta, seed time, tokens/s, fetched GB/s, checksum.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
using namespace std::chrono_literals;

namespace {

uint64_t kPage = 1ull << 20;   // vector/storage page (--page-mb)

#if !CTP_IS_DEVICE_PASS

struct Cfg {
  char mode_case = 'a';
  bool compressed = true;
  uint64_t layers = 12, experts = 32, slice_bytes = 4ull << 20;
  uint64_t hbm_mb = 2048;
  uint64_t tokens = 64, k_sel = 4;
  double fill_const_frac = 0.5;   // fraction of each 64B chunk that repeats
  uint64_t seed = 12345;
};

uint64_t NowMs() {
  return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint64_t HostRssKb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    unsigned long long kb = 0;
    if (sscanf(line.c_str(), "VmRSS: %llu kB", &kb) == 1) return kb;
  }
  return 0;
}

/** Deterministic slice fill: per 64-byte chunk, the first
 *  fill_const_frac*64 bytes repeat a slice-tag byte (compressible) and the
 *  rest are xorshift noise (incompressible). Checksum is derivable. */
void FillSlice(uint8_t *dst, uint64_t bytes, uint64_t slice_id, double frac) {
  const uint32_t rep = (uint32_t) (frac * 64.0);
  uint64_t x = 0x9E3779B97F4A7C15ull ^ (slice_id * 0xBF58476D1CE4E5B9ull);
  for (uint64_t i = 0; i < bytes; ++i) {
    if ((i & 63u) < rep) {
      dst[i] = (uint8_t) (slice_id * 131u + (i & 63u));
    } else {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      dst[i] = (uint8_t) x;
    }
  }
}

uint64_t SliceSum(uint64_t bytes, uint64_t slice_id, double frac) {
  static std::vector<uint8_t> tmp;
  tmp.resize(bytes);
  FillSlice(tmp.data(), bytes, slice_id, frac);
  uint64_t s = 0;
  for (uint64_t i = 0; i < bytes; ++i) s += tmp[i];
  return s;
}

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace

/** The dequant-read stand-in: block-strided byte reduce into a u64. */
__global__ void ConsumeKernel(const uint8_t *src, uint64_t bytes,
                              unsigned long long *out) {
  unsigned long long local = 0;
  for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
       i < bytes; i += (uint64_t) gridDim.x * blockDim.x) {
    local += src[i];
  }
  for (int off = 16; off > 0; off >>= 1) {
    local += __shfl_xor_sync(0xffffffff, local, off);
  }
  if ((threadIdx.x & 31u) == 0) atomicAdd(out, local);
}

#if !CTP_IS_DEVICE_PASS
int main(int argc, char **argv) {
  Cfg c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&](const char *) { return std::stoull(argv[++i]); };
    if (a == "--case") {
      c.mode_case = argv[++i][0];
    } else if (a == "--uncompressed") {
      c.compressed = false;
    } else if (a == "--hbm-mb") {
      c.hbm_mb = val("hbm");
    } else if (a == "--tokens") {
      c.tokens = val("tokens");
    } else if (a == "--layers") {
      c.layers = val("layers");
    } else if (a == "--experts") {
      c.experts = val("experts");
    } else if (a == "--page-mb") {
      kPage = val("page") << 20;
    }
  }
  // Case presets: dataset vs. tier-cap regimes (see header comment).
  if (c.mode_case == 'a') {
    c.layers = 12;
    c.hbm_mb = 2048;                      // 1.5 GiB fits raw AND compressed
  } else if (c.mode_case == 'b') {
    c.layers = 12;
    c.hbm_mb = 900;                       // raw spills, compressed fits
  } else if (c.mode_case == 'c') {
    c.layers = 24;
    c.hbm_mb = 700;                       // even compressed spills
  }
  const uint64_t slices = c.layers * c.experts;
  const uint64_t data_bytes = slices * c.slice_bytes;
  const uint64_t pages_per_slice = c.slice_bytes / kPage;
  const uint64_t num_pages = data_bytes / kPage;

  // ---- compose: compressor 600 (nvcomp-lz4) -> core 512 [hbm + ram] ----
  {
    std::string cfg_path =
        "/tmp/clio_cfb_compose_" + std::to_string((long) ::getpid()) + ".yaml";
    std::ofstream cfg(cfg_path);
    cfg << "runtime:\n  num_threads: 6\n  queue_depth: 65536\n"
        << "  first_busy_wait: 1000000000\n\n"
        << "compose:\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n"
        << "    compress_lib: \"nvcomp-lz4\"\n    compress_preset: 2\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"hbm::cfb_hbm_tier0\"\n"
        << "        bdev_type: \"hbm\"\n        capacity_limit: \""
        << c.hbm_mb << "MB\"\n        score: 1.0\n"
        << "      - path: \"ram::cfb_ram_tier1\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \""
        << (data_bytes * 2 >> 20) << "MB\"\n        score: 0.5\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CTP_LOG_LEVEL", "error", 0);
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg_path.c_str(), 1);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "CLIO_INIT failed\n");
    return 1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return 1;
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);
  std::this_thread::sleep_for(300ms);

  size_t vfree0 = 0, vtot = 0;
  cudaMemGetInfo(&vfree0, &vtot);
  const uint64_t rss0 = HostRssKb();

  // ---- access-frequency ranking (zipf-ish) shared by seeding order and
  // the access stream: rank r has weight 1/(1+r). Case c seeds HOT slices
  // first so the score-1.0 hbm tier holds the best pages.
  std::vector<uint64_t> rank(slices);
  std::iota(rank.begin(), rank.end(), 0);
  {
    uint64_t x = c.seed;
    for (uint64_t i = slices - 1; i > 0; --i) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      std::swap(rank[i], rank[x % (i + 1)]);
    }
  }
  std::vector<double> cum(slices);
  double acc = 0;
  for (uint64_t r = 0; r < slices; ++r) {
    acc += 1.0 / (double) (1 + r);
    cum[r] = acc;
  }

  // ---- seed pages through the storage pool (hot slices first) ----
  const clio::run::PoolId storage_pool =
      c.compressed ? clio::run::PoolId(600, 0) : clio::run::PoolId(0, 0);
  gv::Vector<uint8_t> vec("cfb_data", /*nblocks=*/1, /*gpu_id=*/0,
                          /*gpu_pages_per_block=*/8,
                          /*host_pages_per_block=*/0, kPage,
                          /*cache_period_us=*/1000000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true, storage_pool,
                          /*family_pages=*/num_pages);
  vec.SetLogicalBytes(data_bytes);
  const uint64_t t_seed0 = NowMs();
  {
    clio::cte::core::Client put_client(c.compressed
                                           ? clio::run::PoolId(600, 0)
                                           : clio::cte::core::kCtePoolId);
    std::vector<uint8_t> host(kPage);
    for (uint64_t r = 0; r < slices; ++r) {   // rank order == hot first
      const uint64_t sl = rank[r];
      for (uint64_t p = 0; p < pages_per_slice; ++p) {
        const uint64_t gp = sl * pages_per_slice + p;
        FillSlice(host.data(), kPage, gp, c.fill_const_frac);
        auto buf = CLIO_CPU_IPC->AllocateBuffer(kPage);
        std::memcpy(buf.ptr_, host.data(), kPage);
        auto t = put_client.AsyncPutBlob(
            vec.TagId(), vec.PageBlobName(gp), 0, kPage,
            buf.shm_.template Cast<void>(), 1.0f, clio::cte::core::Context(),
            0);
        t.Wait();
        CLIO_CPU_IPC->FreeBuffer(buf);
        if (t->return_code_ != 0) {
          std::fprintf(stderr, "seed put failed gp=%llu rc=%d\n",
                       (unsigned long long) gp, (int) t->return_code_);
          return 1;
        }
      }
    }
  }
  const uint64_t seed_ms = NowMs() - t_seed0;
  size_t vfree1 = 0;
  cudaMemGetInfo(&vfree1, &vtot);

  // ---- device staging ring + reduce scratch ----
  if (!vec.ReserveSpanCaches(/*ring_bytes=*/256ull << 20,
                             /*vram_reserve=*/1ull << 30,
                             /*slab_cap_max=*/1)) {   // no pinned slab:
    std::fprintf(stderr, "ReserveSpanCaches failed\n");  // pure fault path
    return 1;
  }
  unsigned long long *d_sum = nullptr;
  cudaMalloc(&d_sum, sizeof(unsigned long long));
  cudaMemset(d_sum, 0, sizeof(unsigned long long));

  // ---- the token loop ----
  uint64_t x = c.seed ^ 0xD1B54A32D192ED03ull;
  uint64_t expected = 0, fetched = 0, fails = 0;
  const uint64_t t_run0 = NowMs();
  gv::Vector<uint8_t>::FetchList fl;
  std::vector<std::pair<uint8_t *, uint64_t>> batch;   // (dst, slice)
  for (uint64_t tok = 0; tok < c.tokens; ++tok) {
    for (uint64_t l = 0; l < c.layers; ++l) {
      // ASYNC BULK PREFETCH: issue every selected slice's page-gets first,
      // wait ONCE for the layer -- the server decompresses the whole batch
      // concurrently. Per-slice blocking waits measured dispatch-bound.
      batch.clear();
      for (uint64_t s = 0; s < c.k_sel; ++s) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        const double u = (double) (x >> 11) / 9007199254740992.0 * acc;
        const uint64_t r =
            std::lower_bound(cum.begin(), cum.end(), u) - cum.begin();
        const uint64_t sl = rank[std::min(r, slices - 1)];
        uint8_t *dst = vec.StageTransient(c.slice_bytes, nullptr);
        if (dst == nullptr ||
            !vec.FetchSpanDeviceAsync(dst, sl * c.slice_bytes, c.slice_bytes,
                                      &fl)) {
          ++fails;
          continue;
        }
        batch.emplace_back(dst, sl);
      }
      if (!vec.WaitFetches(fl)) ++fails;
      for (auto &pr : batch) {
        ConsumeKernel<<<256, 256>>>(pr.first, c.slice_bytes, d_sum);
        for (uint64_t p = 0; p < pages_per_slice; ++p) {
          expected += SliceSum(kPage, pr.second * pages_per_slice + p,
                               c.fill_const_frac);
        }
        fetched += c.slice_bytes;
      }
    }
  }
  cudaDeviceSynchronize();
  const uint64_t run_ms = NowMs() - t_run0;
  unsigned long long got = 0;
  cudaMemcpy(&got, d_sum, sizeof(got), cudaMemcpyDeviceToHost);

  const double dev_used_gib =
      (double) ((int64_t) vfree0 - (int64_t) vfree1) / (1ull << 30);
  std::printf(
      "CFB case=%c mode=%s data=%.2fGiB hbm_cap=%lluMB | seed=%.1fs "
      "dev_used=%.2fGiB rss_delta=%.2fGiB | tokens=%llu fetched=%.2fGiB "
      "run=%.2fs tok/s=%.1f GB/s=%.2f | checksum=%s fails=%llu\n",
      c.mode_case, c.compressed ? "compressed" : "uncompressed",
      (double) data_bytes / (1ull << 30), (unsigned long long) c.hbm_mb,
      seed_ms / 1000.0, dev_used_gib,
      (double) (HostRssKb() - rss0) / (1ull << 20),
      (unsigned long long) c.tokens, (double) fetched / (1ull << 30),
      run_ms / 1000.0, c.tokens * 1000.0 / (double) run_ms,
      (double) fetched / (1ull << 30) / (run_ms / 1000.0),
      got == expected ? "OK" : "MISMATCH", (unsigned long long) fails);
  if (got != expected) {
    std::printf("CFB expected=%llu got=%llu\n", (unsigned long long) expected,
                (unsigned long long) got);
  }
  vec.StopManager();
  return got == expected ? 0 : 2;
}
#endif  // !CTP_IS_DEVICE_PASS
