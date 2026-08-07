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
#include <clio_cte/gpu_vector/gpu_vector_hbm_direct.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
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
  uint64_t slab_mb = 512;    // decompressed span cache (--slab-mb)
  bool in_kernel = false;    // read through the vector's device API
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

/**
 * IN-KERNEL vector read (--in-kernel): the kernel reads the slice THROUGH
 * the vector's device API, faulting pages on demand from inside the kernel,
 * with NO host staging at all. This is the mode where the kernel indexes a
 * logical array larger than device memory; the default mode below has the
 * host stage a slice and hands the kernel a plain pointer, which never
 * exercises the device-side path.
 */
__global__ void ConsumeKernelVec(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceView<uint8_t> view,
                                 const uint64_t *access, uint64_t n_access,
                                 uint64_t slice_bytes,
                                 unsigned long long *out) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  auto fam = [] (const gv::DeviceViewBase &b, uint64_t pg) {
    return gv::FamilyOf(b, pg);
  };
  cte::gpu::dev::vector<uint8_t, decltype(fam)> W(view, g_ipc_manager_ptr,
                                                  fam);
  // read_range is WARP-collective and keeps its resolved page in ONE
  // __shared__ slot per block, so exactly one warp may drive it; a second
  // warp on a different page overwrites that slot mid-read (illegal
  // access). The block-wide pattern is therefore: warp 0 pulls a chunk
  // through the vector into shared memory, the whole block consumes it.
  // The page is sized to the access granularity, so warp 0 never leaves
  // one page during a slice and its faults are one page each.
  constexpr uint32_t kChunk = 16384;
  __shared__ uint8_t stage[kChunk];
  __shared__ unsigned long long block_sum;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  if (tid == 0) block_sum = 0;
  __syncthreads();

  // ONE launch for the WHOLE workload: each CUDA block owns one vector
  // cache block and walks its stride of the access stream, faulting pages
  // in from inside the kernel as it goes.
  for (uint64_t i = blockIdx.x; i < n_access; i += gridDim.x) {
    const uint64_t base = access[i] * slice_bytes;
    for (uint64_t off = 0; off < slice_bytes; off += kChunk) {
      const uint64_t lo = base + off;
      const uint64_t hi = lo + kChunk;
      if (warp == 0) {
        W.read_range(lo, hi, [&](uint64_t idx, uint8_t v) {
          stage[idx - lo] = v;
        });
      }
      __syncthreads();
      unsigned long long local = 0;
      for (uint32_t j = tid; j < kChunk; j += blockDim.x) local += stage[j];
      for (int o = 16; o > 0; o >>= 1) {
        local += __shfl_xor_sync(0xffffffff, local, o);
      }
      if ((tid & 31u) == 0) atomicAdd(&block_sum, local);
      __syncthreads();
    }
  }
  if (tid == 0) atomicAdd(out, block_sum);
}

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
    } else if (a == "--in-kernel") {
      c.in_kernel = true;
    } else if (a == "--slab-mb") {
      c.slab_mb = val("slab");
    } else if (a == "--page-mb") {
      kPage = val("page") << 20;
    } else if (a == "--no-prefetch") {
      c.k_sel |= (1ull << 63);   // flag bit, masked below
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
  // MODE 2 page sizing: read_range keeps its resolved page in ONE
  // __shared__ slot per block, so warps that are on DIFFERENT pages stomp
  // each other's pointer (illegal access). Sizing the page to the access
  // granularity means every warp in a block is inside the SAME page for
  // the whole call -- they all resolve the identical Page*, so the shared
  // slot is written with one value and the block can run wide.
  const bool prefetch_on = !(c.k_sel >> 63);
  c.k_sel &= ~(1ull << 63);
  if (c.in_kernel) kPage = c.slice_bytes;   // one page == one slice
  const uint64_t slices = c.layers * c.experts;
  const uint64_t data_bytes = slices * c.slice_bytes;
  const uint64_t pages_per_slice = c.slice_bytes / kPage;
  const uint64_t num_pages = data_bytes / kPage;

  // ---- compose: compressor 600 -> core 512 [hbm + ram] ----
  // MODE 2 pins a HOST codec. In-kernel faulting parks the user kernel on
  // the SMs spin-waiting for its page, and a GPU codec must LAUNCH a
  // decompress kernel to satisfy that page -- which cannot schedule behind
  // the spinning kernel. Deadlock, reproduced here as a hang. A host codec
  // decompresses on the CPU, so the fault completes without needing the
  // GPU. (Mode 1 never spins on the device, so it keeps nvcomp.)
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
        << "    compress_lib: \""
        << (c.in_kernel ? "lz4" : "nvcomp-lz4")
        << "\"\n    compress_preset: 2\n\n"
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
    // Lane batching (issue #820): lets the compressor's BuildBatch/SmashBatch
    // collapse a burst of page-gets into ONE merged task with a single
    // batched nvcomp decompress.
    ctp::SystemInfo::Setenv("CLIO_BATCH_LANE", "1", 0);
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
  // LAYER-LOCAL experts (the llama shape): layer l selects only among its
  // own E slices, zipf-skewed within the layer. This is the locality the
  // score-driven prefetcher exploits: the hbm tier is a rolling window over
  // the upcoming layers' hot slices.
  std::vector<uint64_t> rank(c.experts);
  std::iota(rank.begin(), rank.end(), 0);
  {
    uint64_t x = c.seed;
    for (uint64_t i = c.experts - 1; i > 0; --i) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      std::swap(rank[i], rank[x % (i + 1)]);
    }
  }
  std::vector<double> cum(c.experts);
  double acc = 0;
  for (uint64_t r = 0; r < c.experts; ++r) {
    acc += 1.0 / (double) (1 + r);
    cum[r] = acc;
  }

  // ---- seed pages through the storage pool (hot slices first) ----
  const clio::run::PoolId storage_pool =
      c.compressed ? clio::run::PoolId(600, 0) : clio::run::PoolId(0, 0);
  // MODE 2 needs a vector configured for ON-DEVICE faulting:
  //  - kAsync: only that mode does the CacheManagerKernel warmup launch,
  //    without which the cache thread's first cross-thread launch races a
  //    spin-waiting read_range.
  //  - several cache blocks: read_range indexes the page table by blockIdx,
  //    so the grid is capped at nblocks -- one block would serialize the
  //    whole workload onto one SM.
  //  - a real per-block cache, since every span faults through it.
  const clio::run::u32 nblk = c.in_kernel ? 8u : 1u;
  const clio::run::u32 ppb =
      c.in_kernel ? (clio::run::u32) ((c.hbm_mb << 20) / kPage / nblk / 2)
                  : 8u;
  gv::Vector<uint8_t> vec("cfb_data", nblk, /*gpu_id=*/0,
                          /*gpu_pages_per_block=*/ppb < 8 ? 8 : ppb,
                          /*host_pages_per_block=*/0, kPage,
                          /*cache_period_us=*/c.in_kernel ? 200 : 1000000,
                          c.in_kernel ? gv::CacheMode::kAsync
                                      : gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true, storage_pool,
                          /*family_pages=*/num_pages);
  vec.SetLogicalBytes(data_bytes);
  std::vector<uint64_t> page_sum(num_pages, 0);
  const uint64_t t_seed0 = NowMs();
  {
    clio::cte::core::Client put_client(c.compressed
                                           ? clio::run::PoolId(600, 0)
                                           : clio::cte::core::kCtePoolId);
    std::vector<uint8_t> host(kPage);
    for (uint64_t sl = 0; sl < slices; ++sl) {   // layer-major order
      for (uint64_t p = 0; p < pages_per_slice; ++p) {
        const uint64_t gp = sl * pages_per_slice + p;
        FillSlice(host.data(), kPage, gp, c.fill_const_frac);
        uint64_t ps = 0;
        for (uint64_t j = 0; j < kPage; ++j) ps += host[j];
        page_sum[gp] = ps;
        auto buf = CLIO_CPU_IPC->AllocateBuffer(kPage);
        std::memcpy(buf.ptr_, host.data(), kPage);
        // Seed COLD (0.3): placement follows the score protocol -- only the
        // prefetcher's promotions (0.95) move pages into the hbm tier, so
        // the tier's contents track the schedule instead of freezing
        // whatever fit first. (Seeding at 1.0 also made promotion a
        // sub-threshold no-op: 1.0 -> 0.95 moves nothing.)
        auto t = put_client.AsyncPutBlob(
            vec.TagId(), vec.PageBlobName(gp), 0, kPage,
            buf.shm_.template Cast<void>(), 0.3f, clio::cte::core::Context(),
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

  // ---- device staging ring + DECOMPRESSED span cache + reduce scratch ----
  // The slab holds slices in their FINAL form, so a hit skips the fetch AND
  // (for a compressed tag) the decompress. Disabling it measured the pure
  // fault path, which understated compression: every read of a hot slice
  // re-decompressed it. Both modes get the same slab, so the comparison
  // stays honest -- it just no longer forbids the caching a real caller
  // (llama) does.
  if (!vec.ReserveSpanCaches(/*ring_bytes=*/256ull << 20,
                             /*vram_reserve=*/1ull << 30,
                             /*slab_cap_max=*/c.slab_mb << 20)) {
    std::fprintf(stderr, "ReserveSpanCaches failed\n");
    return 1;
  }
  unsigned long long *d_sum = nullptr;
  cudaMalloc(&d_sum, sizeof(unsigned long long));
  cudaMemset(d_sum, 0, sizeof(unsigned long long));

  // ---- the token loop ----
  uint64_t x = c.seed ^ 0xD1B54A32D192ED03ull;
  uint64_t expected = 0, fetched = 0, fails = 0;
  const uint64_t t_run0 = NowMs();
  // The access stream is deterministic: materialize it up front so the
  // prefetcher's look-ahead window is 100% ACCURATE.
  std::vector<uint64_t> access;
  access.reserve(c.tokens * c.layers * c.k_sel);
  for (uint64_t tok2 = 0; tok2 < c.tokens; ++tok2) {
    for (uint64_t l = 0; l < c.layers; ++l) {
      for (uint64_t k = 0; k < c.k_sel; ++k) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        const double u = (double) (x >> 11) / 9007199254740992.0 * acc;
        const uint64_t r =
            std::lower_bound(cum.begin(), cum.end(), u) - cum.begin();
        access.push_back(l * c.experts +
                         rank[std::min(r, c.experts - 1)]);
      }
    }
  }
  // SCORE-DRIVEN PREFETCH: for slices we will need soon, RAISE their pages'
  // scores (0.95) so the CTE's organizer migrates them INTO the hbm tier
  // ahead of use -- a metadata op, NOT a data copy; the Get at access time
  // then decompresses from device memory. Slices leaving the window are
  // demoted (0.30) so the tier's capacity follows the schedule. Promotions
  // are fire-and-forget with a bounded drain.
  // Exact next-occurrence of every access position (backward scan): the
  // prefetcher promotes a slice only when it RECURS soon enough to repay
  // the migration -- one-shot tail slices go through the batched task path
  // instead of churning the tier.
  std::vector<uint64_t> next_occ(access.size(), UINT64_MAX);
  {
    std::vector<uint64_t> last(slices, UINT64_MAX);
    for (uint64_t i2 = access.size(); i2-- > 0;) {
      next_occ[i2] = last[access[i2]];
      last[access[i2]] = i2;
    }
  }
  const uint64_t kRecurHorizon = access.size();   // recurs at all
  clio::cte::core::Client core_cli(clio::cte::core::kCtePoolId);
  if (!core_cli.AttachShmCache()) {
    std::fprintf(stderr, "CFB: shm metadata cache not attachable; "
                         "zero-task path disabled\n");
  }
  gv::DirectStats dstats;
  const uint64_t kLookahead = c.k_sel * 10;  // ~10 layers of migration lead
  std::vector<int8_t> promoted(slices, 0);
  // True LRU: entries are (slice, epoch); a pop is valid only when the
  // epoch matches the slice's latest touch, so re-promotion refreshes
  // recency instead of leaving the hottest slice first in line.
  std::deque<std::pair<uint64_t, uint64_t>> promo_lru;
  uint64_t bounces = 0;
  uint64_t t_direct = 0, t_task = 0, t_consume = 0;
  int pending_evicts_ratio = 0;
  std::vector<uint64_t> promo_epoch(slices, 0);
  uint64_t promoted_bytes = 0;
  std::deque<clio::run::Future<clio::cte::core::ReorganizeBlobTask>> reorgs;
  static uint64_t reorg_ok = 0, reorg_fail = 0;
  auto drain_one = [&]() {
    reorgs.front().Wait();
    if (reorgs.front()->return_code_ == 0) ++reorg_ok; else ++reorg_fail;
    reorgs.pop_front();
  };
  auto set_score = [&](uint64_t sl, float score) {
    for (uint64_t pp = 0; pp < pages_per_slice; ++pp) {
      reorgs.push_back(core_cli.AsyncReorganizeBlob(
          vec.TagId(), vec.PageBlobName(sl * pages_per_slice + pp), score,
          clio::run::PoolQuery::Local()));
    }
    while (reorgs.size() > 256) drain_one();
  };
  // In-kernel mode needs the GPU IPC handle and a grid capped at the
  // vector's cache-block count (read_range indexes the page table by
  // blockIdx, so a wider grid would have several CUDA blocks racing one
  // cache block).
  auto *gpu_ipc = CLIO_CPU_IPC->GetGpuIpcManager();
  clio::run::IpcManagerGpuInfo gpu_info{};
  if (gpu_ipc != nullptr) {
    gpu_info = gpu_ipc->GetGpuInfo(0);
  } else if (c.in_kernel) {
    // A default-constructed handle would be dereferenced by the kernel's
    // fault path -- fail loudly rather than as an illegal access.
    std::fprintf(stderr, "CFB: no GPU IPC manager; --in-kernel needs one\n");
    return 1;
  }
  std::fprintf(stderr, "CFB: gpu_ipc=%p nblk=%u ppb=%u pages=%llu\n",
               (void *) gpu_ipc, nblk, (unsigned) (ppb < 8 ? 8 : ppb),
               (unsigned long long) num_pages);
  const unsigned nblocks_grid = nblk;   // one CUDA block per cache block
  if (c.in_kernel) {
    // MODE 2: exactly ONE kernel invocation for the entire workload, with
    // the device vector as its input. Everything the host does here is
    // setup; no staging, no per-slice launches, no host round trips.
    uint64_t *d_access = nullptr;
    if (cudaMalloc(&d_access, access.size() * sizeof(uint64_t)) !=
            cudaSuccess ||
        cudaMemcpy(d_access, access.data(),
                   access.size() * sizeof(uint64_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      std::fprintf(stderr, "CFB: access upload failed\n");
      return 1;
    }
    for (uint64_t idx : access) {
      for (uint64_t p = 0; p < pages_per_slice; ++p) {
        expected += page_sum[idx * pages_per_slice + p];
      }
      fetched += c.slice_bytes;
    }
    const uint64_t t_k0 = NowMs();
    // Wide blocks are safe because the page is sized to the slice: every
    // warp in the block is inside the same page for the whole read_range,
    // so they all resolve the identical Page* into the shared slot.
    ConsumeKernelVec<<<nblocks_grid, 256>>>(gpu_info, vec.Device(), d_access,
                                            access.size(), c.slice_bytes,
                                            d_sum);
    const cudaError_t kerr = cudaDeviceSynchronize();
    const uint64_t k_ms = NowMs() - t_k0;
    if (kerr != cudaSuccess) {
      std::fprintf(stderr, "CFB: in-kernel run failed: %s\n",
                   cudaGetErrorString(kerr));
      return 1;
    }
    cudaFree(d_access);
    unsigned long long got_k = 0;
    cudaMemcpy(&got_k, d_sum, sizeof(got_k), cudaMemcpyDeviceToHost);
    std::printf(
        "CFB[in-kernel] case=%c mode=%s data=%.2fGiB hbm_cap=%lluMB | "
        "seed=%.1fs dev_used=%.2fGiB | launches=1 fetched=%.2fGiB "
        "run=%.2fs GB/s=%.2f | checksum=%s\n",
        c.mode_case, c.compressed ? "compressed" : "uncompressed",
        (double) data_bytes / (1ull << 30), (unsigned long long) c.hbm_mb,
        seed_ms / 1000.0,
        (double) ((int64_t) vfree0 - (int64_t) vfree1) / (1ull << 30),
        (double) fetched / (1ull << 30), k_ms / 1000.0,
        (double) fetched / (1ull << 30) / (k_ms / 1000.0),
        got_k == expected ? "OK" : "MISMATCH");
    if (got_k != expected) {
      std::printf("CFB expected=%llu got=%llu\n",
                  (unsigned long long) expected,
                  (unsigned long long) got_k);
    }
    vec.StopManager();
    return got_k == expected ? 0 : 2;
  }
  gv::Vector<uint8_t>::FetchList fl;
  std::vector<std::pair<uint8_t *, uint64_t>> batch;   // (dst, slice)
  const uint64_t t_run0b = NowMs();
  (void) t_run0b;
  for (uint64_t i = 0; i < access.size(); i += c.k_sel) {
    // Window maintenance: promote [i, i+K), demote what fell out.
    for (uint64_t j = i + c.k_sel;
         prefetch_on && j < std::min<uint64_t>(i + kLookahead,
                                               access.size()); ++j) {
      const uint64_t sl = access[j];
      if (next_occ[j] == UINT64_MAX || next_occ[j] > j + kRecurHorizon) {
        continue;   // one-shot in this horizon: not worth a migration
      }
      if (promoted[sl]) {
        promo_lru.emplace_back(sl, ++promo_epoch[sl]);   // refresh recency
      } else {
        promoted[sl] = 1;
        promo_lru.emplace_back(sl, ++promo_epoch[sl]);
        promoted_bytes += (uint64_t) (c.slice_bytes * (c.compressed ? 0.36 : 1.0));
        // 1.0, not 0.95: MaxBwDpe treats targets with score > blob score
        // as FALLBACK, so anything below the hbm tier's 1.0 puts hbm in
        // the fallback group and the blob lands in ram.
        set_score(sl, 1.0f);
      }

    }
    // CAPACITY-PRESSURE demotion only: hot slices stay hbm-resident across
    // recurrences (window-exit demotion re-migrated every slice every
    // cycle -- 42k reorganizes saturating the workers for a working set
    // that mostly fits). Evict LRU promoted slices only when the promoted
    // footprint nears the tier cap.
    if (prefetch_on) {
      // Measured lz4 ratio on this data is ~0.34; overestimating stored size
      // made the budget refuse promotions at half the tier's true capacity.
      const uint64_t est_slice = (uint64_t) (c.slice_bytes *
          (c.compressed ? 0.36 : 1.0));
      const uint64_t budget = (c.hbm_mb << 20) * 85 / 100;
      // Bounced promotions mean the tier is REALLY full whatever our
      // estimate says: evict to make room so retries converge.
      const uint64_t forced = bounces;
      bounces = 0;
      auto evict_one = [&]() -> bool {
        uint64_t scanned = 0;
        while (!promo_lru.empty() && scanned < promo_lru.size() + 8) {
          auto [victim, epoch] = promo_lru.front();
          promo_lru.pop_front();
          ++scanned;
          if (!promoted[victim] || promo_epoch[victim] != epoch) {
            continue;   // stale: refreshed or already evicted
          }
          bool ahead = false;
          for (uint64_t j2 = i;
               j2 < std::min<uint64_t>(i + kLookahead, access.size());
               ++j2) {
            if (access[j2] == victim) { ahead = true; break; }
          }
          if (ahead) {
            promo_lru.emplace_back(victim, epoch);   // still needed
            continue;
          }
          promoted[victim] = 0;
          promoted_bytes -= est_slice;
          set_score(victim, 0.30f);
          return true;
        }
        return false;
      };
      for (uint64_t e2 = 0; e2 < forced; ++e2) {
        if (!evict_one()) break;
      }
      while (promoted_bytes + est_slice * c.k_sel > budget) {
        if (!evict_one()) break;
      }
      pending_evicts_ratio = 0;   // placeholder anchor
      (void) pending_evicts_ratio;
    }
    batch.clear();
    // ZERO-TASK first: promoted, hbm-resident frames resolve straight to
    // device pointers and decompress client-side in ONE batch; only the
    // leftovers pay the task path.
    std::vector<gv::DirectWant> want;
    std::vector<gv::DirectWant> leftover;
    for (uint64_t s = 0; s < c.k_sel && i + s < access.size(); ++s) {
      const uint64_t sl = access[i + s];
      uint8_t *dst = vec.StageTransient(c.slice_bytes, nullptr);
      if (dst == nullptr) {
        ++fails;
        continue;
      }
      for (uint64_t pp = 0; pp < pages_per_slice; ++pp) {
        want.push_back(gv::DirectWant{sl * pages_per_slice + pp,
                                      dst + pp * kPage, kPage});
      }
      batch.emplace_back(dst, sl);
    }
    const uint64_t tp0 = NowMs();
    gv::HbmDirectFetchBatch(vec, core_cli, want, &leftover, &dstats);
    t_direct += NowMs() - tp0;
    // Self-correcting accounting: a leftover whose record shows a score
    // below threshold is a promotion that BOUNCED off a full tier (the
    // landed-tier score makes that visible). Clear its promoted state and
    // refund the budget so eviction/retry can converge.
    if (prefetch_on) {
      for (const auto &lw : leftover) {
        const uint64_t sl = lw.gp / pages_per_slice;
        if (!promoted[sl]) continue;
        clio::cte::core::ShmBlobRecord r2;
        if (core_cli.TryGetBlobRecordShm(vec.TagId(),
                                         vec.PageBlobName(lw.gp), &r2) &&
            r2.score_ < 0.9f) {
          promoted[sl] = 0;
          promoted_bytes -= (uint64_t) (c.slice_bytes * (c.compressed ? 0.36 : 1.0));
          bounces += 1;   // tier genuinely full: evict below
        }
      }
    }
    for (const auto &lw : leftover) {
      if (!vec.FetchSpanDeviceAsync(lw.dst, lw.gp * kPage, lw.len, &fl)) {
        ++fails;
      }
    }
    const uint64_t tp1 = NowMs();
    if (!vec.WaitFetches(fl)) ++fails;
    // Slab admissions fill on the vector's copy stream; make them visible
    // before the consuming kernels run.
    cudaStreamSynchronize((cudaStream_t) vec.CopyStream());
    t_task += NowMs() - tp1;
    const uint64_t tp2 = NowMs();
    for (auto &pr : batch) {
      ConsumeKernel<<<256, 256>>>(pr.first, c.slice_bytes, d_sum);
      for (uint64_t p = 0; p < pages_per_slice; ++p) {
        expected += page_sum[pr.second * pages_per_slice + p];
      }
      fetched += c.slice_bytes;
    }
    t_consume += NowMs() - tp2;
  }
  while (!reorgs.empty()) drain_one();
  std::printf("CFB reorg: %llu ok, %llu failed\n",
              (unsigned long long) reorg_ok,
              (unsigned long long) reorg_fail);
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
  std::printf("CFB time: direct=%.1fs task=%.1fs consume=%.1fs\n",
              t_direct / 1000.0, t_task / 1000.0, t_consume / 1000.0);
  std::printf("CFB direct: %llu hits %llu misses | no_rec=%llu cold=%llu "
              "trunc=%llu unresolv=%llu moved=%llu\n",
              (unsigned long long) dstats.hits,
              (unsigned long long) dstats.misses,
              (unsigned long long) dstats.no_rec,
              (unsigned long long) dstats.cold,
              (unsigned long long) dstats.trunc,
              (unsigned long long) dstats.unresolv,
              (unsigned long long) dstats.moved);
  if (got != expected) {
    std::printf("CFB expected=%llu got=%llu\n", (unsigned long long) expected,
                (unsigned long long) got);
  }
  vec.StopManager();
  return got == expected ? 0 : 2;
}
#endif  // !CTP_IS_DEVICE_PASS
