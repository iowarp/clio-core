/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GraphSAGE-mean forward whose feature matrix is STREAMED through a real
 * gv::Vector<float> backed by the compressor, and checked bit-for-bit against
 * an in-core baseline.
 *
 * This is the "Option B" counterpart to test_gpu_vector_gnn_gpu.cc. That test
 * moves feature pages through the compressor with explicit Put/Get calls and
 * never involves the vector. Here the vector IS the path: the features are
 * written into it page by page, flushed out through the compressor to the CTE,
 * and then read back with a cache deliberately far smaller than the matrix, so
 * every page is a genuine fault that has to be fetched and decompressed before
 * the kernel can read it.
 *
 * What changed from the original. It was written against gv::SequentialTransaction
 * over a gv::DeviceView, with a ten-argument Vector constructor carrying
 * CacheMode::kLegacy, host_pages_per_block, cache_period_us and friends. The
 * rewrite deleted all of it -- there is no transaction.h any more. The current
 * Vector takes the storage pool, codec and preset directly, which is a better
 * fit for this test than the old windowed-prefetch abstraction was, so the
 * streaming is now expressed as what it always meant: a yieldable kernel
 * walking the vector with HoldPageYield while the cache thrashes underneath it.
 *
 * The kernels MUST yield. The blocking HoldPage deadlocks against itself the
 * moment a page needs a transfer: the kernel holds the SM waiting for it, and a
 * resident kernel blocks every later launch in its context, including the one
 * that would service the transfer. Every lane runs the loop for the same
 * reason -- the fault path is block-collective and ends in __syncthreads, so no
 * lane may return early -- and the slice is keyed off yv.Block() because the
 * driver relaunches a compacted grid of whichever blocks are still pending.
 *
 * Correctness statement: zstd is LOSSLESS, and both the aggregation and the
 * combine are deterministic (one thread owns one output element, fixed-order
 * reduction, no atomics). So identical input bytes give bitwise identical
 * logits, and the assertion is memcmp == 0 rather than a tolerance.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <clio_cte/core/core_tasks.h>

#include "gnn_dataset.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

// ===========================================================================
//  Device kernels. The two GNN kernels are the same deterministic pair the
//  Option A test uses; only the feature SOURCE differs.
// ===========================================================================

/** Copy `count` floats from a plain device buffer INTO the paged vector. */
__global__ void VecFillKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<float> v, const float *src,
                              u64 elems_per_block, gy::YieldableView<> yv,
                              gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, i, 0);
  CLIO_YLOCAL_INIT(u64, run, 0);
  const u64 base = static_cast<u64>(yv.Block()) * elems_per_block;

  CLIO_YBEGIN();
  for (; i < elems_per_block; i += run) {
    CLIO_YCALL(v.HoldPageYield(base + i, elems_per_block - i, &run));
    for (u64 k = threadIdx.x; k < run; k += blockDim.x) {
      v[base + i + k] = src[base + i + k];
    }
  }
  // SubmitPut clears `dirty` as it submits, so a lane still writing when
  // thread 0 flushes would lose its writes and leave the page looking clean.
  __syncthreads();
  if (threadIdx.x == 0) {
    v.BeginFlush(base, elems_per_block);
  }
  __syncthreads();
  CLIO_YIELD_IF(v.AnyTransferInFlight());
  CLIO_YEND();
}

/** Scatter the paged vector back OUT into a plain device buffer. */
__global__ void VecDrainKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> v, float *dst,
                               u64 elems_per_block, gy::YieldableView<> yv,
                               gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, i, 0);
  CLIO_YLOCAL_INIT(u64, run, 0);
  const u64 base = static_cast<u64>(yv.Block()) * elems_per_block;

  CLIO_YBEGIN();
  for (; i < elems_per_block; i += run) {
    CLIO_YCALL(v.HoldPageYield(base + i, elems_per_block - i, &run));
    for (u64 k = threadIdx.x; k < run; k += blockDim.x) {
      dst[base + i + k] = v[base + i + k];
    }
  }
  CLIO_YEND();
}

__global__ void GnnAggMeanKernel(const float *feat, int D,
                                 const std::int64_t *indptr,
                                 const std::int64_t *indices, std::int64_t N,
                                 float *agg) {
  std::int64_t total = N * (std::int64_t)D;
  for (std::int64_t idx = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (std::int64_t)gridDim.x * blockDim.x) {
    std::int64_t i = idx / D;
    int d = (int)(idx - i * D);
    std::int64_t beg = indptr[i], end = indptr[i + 1];
    float acc = 0.0f;
    for (std::int64_t k = beg; k < end; ++k) {
      acc += feat[indices[k] * (std::int64_t)D + d];
    }
    std::int64_t deg = end - beg;
    agg[idx] = (deg > 0) ? acc / (float)deg : 0.0f;
  }
}

__global__ void GnnCombineKernel(const float *self, const float *neigh, int Din,
                                 int Dout, const float *Wself,
                                 const float *Wneigh, std::int64_t N, float *out,
                                 int relu) {
  std::int64_t total = N * (std::int64_t)Dout;
  for (std::int64_t idx = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (std::int64_t)gridDim.x * blockDim.x) {
    std::int64_t i = idx / Dout;
    int o = (int)(idx - i * Dout);
    float acc = 0.0f;
    for (int k = 0; k < Din; ++k) {
      acc += self[i * (std::int64_t)Din + k] * Wself[(std::int64_t)k * Dout + o];
      acc += neigh[i * (std::int64_t)Din + k] *
             Wneigh[(std::int64_t)k * Dout + o];
    }
    out[idx] = (relu && acc < 0.0f) ? 0.0f : acc;
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

using gnn_test::EnvI64;
using gnn_test::NowSec;

bool g_initialized = false;

/** The compressor pool; the vector spills through it to the core pool. */
inline clio::run::PoolId CompressorPool() { return clio::run::PoolId(600, 0); }

constexpr int kHidden = 256;

void EnsureInit() {
  // An externally supplied CLIO_SERVER_CONF WINS. These tests compose their own
  // tiers and used to setenv over the caller's, which silently discarded a
  // configuration the caller had sized deliberately -- and stayed invisible for
  // as long as the test's own defaults happened to be big enough. See the same
  // note in test_gpu_vector_gnn_train_gpu.cc, where it nearly cost a 53 GiB run.
  if (const char *ext = std::getenv("CLIO_SERVER_CONF")) {
    std::ifstream probe(ext);
    if (probe.good()) {
      std::fprintf(stderr, "[GNN-STREAM] using caller's CLIO_SERVER_CONF=%%s\n", ext);
      if (g_initialized) return;
      REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
      std::this_thread::sleep_for(std::chrono::seconds(1));
      g_initialized = true;
      return;
    }
  }
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10596;
  std::string cfg = "/tmp/gpu_vec_gnn_stream_" + std::to_string(port) + ".yaml";
  const char *ram_env = std::getenv("CLIO_GNN_RAM_MIB");
  int ram_mib = ram_env ? std::atoi(ram_env) : 4096;
  {
    std::ofstream f(cfg);
    f << "networking:\n  port: " << port << "\n\n"
      << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
      << "gpu:\n  queue_depth: 4096\n\n"
      << "compose:\n"
      << "  - mod_name: clio_bdev\n"
      << "    pool_name: \"ram::chi_default_bdev\"\n"
      << "    pool_query: local\n    pool_id: \"301.0\"\n"
      << "    bdev_type: ram\n    capacity: \"512MB\"\n\n"
      << "  - mod_name: clio_cte_compressor\n"
      << "    pool_name: cte_compressor\n    pool_query: local\n"
      << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
      << "  - mod_name: clio_cte_core\n"
      << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
      << "    storage:\n"
      << "      - path: \"ram::cte_ram_tier\"\n"
      << "        bdev_type: \"ram\"\n        capacity_limit: \""
      << ram_mib << "MB\"\n        score: 1.0\n";
    // Optional second tier on real storage. The vector is multi-tiered: a RAM
    // tier sized well below the data plus an NVMe tier behind it is how a
    // matrix larger than memory is held at all. Set CLIO_GNN_NVME_DIR to a
    // directory on the device and the CTE will spill to it by score order.
    if (const char *nvme = std::getenv("CLIO_GNN_NVME_DIR")) {
      const int nvme_mib = std::getenv("CLIO_GNN_NVME_MIB")
                               ? std::atoi(std::getenv("CLIO_GNN_NVME_MIB"))
                               : 8192;
      f << "      - path: \"" << nvme << "\"\n"
        << "        bdev_type: \"file\"\n        capacity_limit: \""
        << nvme_mib << "MB\"\n        score: 0.5\n";
    }
    f << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  std::fprintf(stderr, "[GNN-STREAM] compose=%s\n", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(std::chrono::seconds(1));
  g_initialized = true;
}

/**
 * Store a raw float32 stream into `tag` as vector pages, in this process.
 *
 * Same page naming and codec context as the standalone gnn_ingest tool; it
 * exists here too because a GPU process must own the runtime (see the call
 * site), so it cannot delegate the load to the daemon that gnn_ingest targets.
 */
bool IngestFromFile(const char *path, const clio::cte::core::TagId &tag,
                    u64 page_bytes, int codec, int preset, u64 want_pages) {
  std::FILE *f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "[GNN-STREAM] cannot open ingest file %s\n", path);
    return false;
  }
  clio::cte::core::Client comp(CompressorPool());
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_lib_ = codec;
  ctx.compress_preset_ = preset;
  ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer((size_t)page_bytes);
  if (buf.IsNull()) { std::fclose(f); return false; }

  u64 pg = 0;
  bool ok = true;
  for (; pg < want_pages; ++pg) {
    size_t got = std::fread(buf.ptr_, 1, (size_t)page_bytes, f);
    if (got == 0) break;
    if (got < page_bytes) std::memset(buf.ptr_ + got, 0, page_bytes - got);
    char name[32];
    clio::cte::gpu_vector::PageBlobName(pg, name);
    auto pf = comp.AsyncPutBlob(tag, name, (u64)0, page_bytes,
                                buf.shm_.template Cast<void>(), 0.5f, ctx, 0,
                                clio::run::PoolQuery::Local());
    pf.Wait();
    if (pf->GetReturnCode() != 0) {
      std::fprintf(stderr, "[GNN-STREAM] ingest page %llu rc=%d\n",
                   (unsigned long long)pg, pf->GetReturnCode());
      ok = false;
      break;
    }
  }
  CLIO_IPC->FreeBuffer(buf);
  std::fclose(f);
  std::fprintf(stderr, "[GNN-STREAM] ingested %llu/%llu pages from %s\n",
               (unsigned long long)pg, (unsigned long long)want_pages, path);
  return ok && pg == want_pages;
}

/** Run a yieldable kernel to completion (re-launch until no block is pending). */
template <typename LaunchT>
u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  gy::YieldStack stack(nblocks, 32, 256);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000);
}

}  // namespace

TEST_CASE("gpu_vector: GraphSAGE forward over features streamed through a "
          "compressed gv::Vector is bit-identical to the baseline",
          "[gpu_vector][compress][gnn][zstd][stream]") {
  EnsureInit();

  const std::string data_dir = std::getenv("CLIO_GNN_DATA")
                                   ? std::getenv("CLIO_GNN_DATA")
                                   : "/workspace/data/ogb/arxiv";
  gnn_test::Dataset ds = gnn_test::LoadOrSynthDataset(data_dir, "GNN-STREAM");
  REQUIRE(ds.N > 0);

  // In end-to-end mode the in-core baseline must be the bytes that were
  // INGESTED, not the ones this test would have generated -- otherwise the
  // comparison is against a different matrix and the bit-exact assert fails
  // for reasons that have nothing to do with the vector.
  if (const char *ing = std::getenv("CLIO_GNN_INGEST_FILE")) {
    std::FILE *f = std::fopen(ing, "rb");
    REQUIRE(f != nullptr);
    const size_t want = (size_t)ds.FeatBytes();
    ds.feat.assign(want, 0);
    const size_t got = std::fread(ds.feat.data(), 1, want, f);
    std::fclose(f);
    std::fprintf(stderr,
                 "[GNN-STREAM] baseline taken from ingest file (%zu of %zu B)\n",
                 got, want);
    REQUIRE(got == want);
  }
  REQUIRE(ds.F > 0);
  REQUIRE(ds.C > 0);

  const std::int64_t N = ds.N;
  const int F = ds.F, C = ds.C;
  const int H = std::getenv("CLIO_GNN_HIDDEN")
                    ? std::atoi(std::getenv("CLIO_GNN_HIDDEN"))
                    : kHidden;
  const std::int64_t feat_elems = ds.FeatElems();
  const std::int64_t logical_bytes = ds.FeatBytes();
  // When set, the feature matrix is loaded from this raw float32 file instead
  // of being generated and written by a kernel -- the end-to-end path.
  const char *ingest_file = std::getenv("CLIO_GNN_INGEST_FILE");
  const int codec = (int)EnvI64("CLIO_GNN_COMPRESS_LIB", 10);      // ZSTD
  const int preset = (int)EnvI64("CLIO_GNN_COMPRESS_PRESET", 2);   // BALANCED
  std::fprintf(stderr,
               "[GNN-STREAM] source=%s N=%lld F=%d E=%lld C=%d H=%d codec=%s\n",
               ds.SourceName(), (long long)N, F, (long long)ds.E, C, H,
               gnn_test::CodecName(codec));

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // ---- CSR + weights to device (shared by both forward passes) ----
  const std::int64_t csrE = ds.Csr()[1];
  std::int64_t *d_indptr = nullptr, *d_indices = nullptr;
  REQUIRE(cudaMalloc(&d_indptr, (N + 1) * sizeof(std::int64_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_indices, csrE * sizeof(std::int64_t)) == cudaSuccess);
  cudaMemcpy(d_indptr, ds.Indptr(), (N + 1) * sizeof(std::int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, ds.Indices(), csrE * sizeof(std::int64_t),
             cudaMemcpyHostToDevice);

  auto make_weights = [](int din, int dout, unsigned seed) {
    std::mt19937 rng(seed);
    float s = 1.0f / std::sqrt((float)din);
    std::uniform_real_distribution<float> dist(-s, s);
    std::vector<float> w((size_t)din * dout);
    for (auto &x : w) x = dist(rng);
    return w;
  };
  std::vector<float> hW[4] = {make_weights(F, H, 1), make_weights(F, H, 2),
                              make_weights(H, C, 3), make_weights(H, C, 4)};
  float *dW[4] = {nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < 4; ++i) {
    REQUIRE(cudaMalloc(&dW[i], hW[i].size() * sizeof(float)) == cudaSuccess);
    cudaMemcpy(dW[i], hW[i].data(), hW[i].size() * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  float *d_feat = nullptr, *d_src = nullptr, *d_agg1 = nullptr, *d_h1 = nullptr,
        *d_agg2 = nullptr, *d_logits = nullptr;
  REQUIRE(cudaMalloc(&d_feat, feat_elems * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_src, feat_elems * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_agg1, feat_elems * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_h1, N * (std::int64_t)H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_agg2, N * (std::int64_t)H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_logits, N * (std::int64_t)C * sizeof(float)) ==
          cudaSuccess);

  auto forward = [&](std::vector<float> &out_logits) {
    GnnAggMeanKernel<<<256, 256>>>(d_feat, F, d_indptr, d_indices, N, d_agg1);
    GnnCombineKernel<<<256, 256>>>(d_feat, d_agg1, F, H, dW[0], dW[1], N, d_h1, 1);
    GnnAggMeanKernel<<<256, 256>>>(d_h1, H, d_indptr, d_indices, N, d_agg2);
    GnnCombineKernel<<<256, 256>>>(d_h1, d_agg2, H, C, dW[2], dW[3], N, d_logits,
                                   0);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    out_logits.resize((size_t)(N * (std::int64_t)C));
    cudaMemcpy(out_logits.data(), d_logits,
               N * (std::int64_t)C * sizeof(float), cudaMemcpyDeviceToHost);
  };

  // ---- BASELINE: features resident, never compressed ----
  cudaMemcpy(d_feat, ds.Feat(), logical_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_src, ds.Feat(), logical_bytes, cudaMemcpyHostToDevice);
  std::vector<float> logits_base;
  forward(logits_base);

  // ---- STREAMED: the same bytes, but through a compressed paged vector ----
  //
  // pages_per_block is a small fraction of the matrix on purpose. A cache big
  // enough to hold everything would fault each page once and then sit resident,
  // which tests the codec but not the streaming; sized down, the write phase
  // evicts (each eviction a compressed writeback) and the read phase faults
  // every page back in through a decompress.
  const u64 page_bytes = (u64)EnvI64("CLIO_GNN_PAGE_KIB", 256) * 1024;
  const u64 elems_per_page = page_bytes / sizeof(float);
  REQUIRE(elems_per_page > 0);
  const u32 nblocks = (u32)EnvI64("CLIO_GNN_BLOCKS", 4);
  // Give every block a whole number of pages so no page is shared by two
  // blocks: page tables are per block, and a page resident in two of them at
  // once would be two independent copies of the same bytes.
  const u64 pages_total =
      (u64)((feat_elems + (std::int64_t)elems_per_page - 1) / (std::int64_t)elems_per_page);
  const u64 pages_per_blk = (pages_total + nblocks - 1) / nblocks;
  const u64 elems_per_block = pages_per_blk * elems_per_page;
  const u64 padded_elems = elems_per_block * nblocks;
  const u32 slots = (u32)std::max<u64>(2, EnvI64("CLIO_GNN_SLOTS", 2));
  REQUIRE(slots < pages_per_blk + 1);

  std::fprintf(stderr,
               "[GNN-STREAM] vector: %llu pages of %llu KiB over %u blocks, "
               "%u cache slots/block (%.1f%% resident)\n",
               (unsigned long long)pages_total,
               (unsigned long long)(page_bytes / 1024), nblocks, slots,
               100.0 * (double)(slots * nblocks) / (double)pages_total);

  double t_store = 0.0, t_stream = 0.0;
  gv::Vector<float>::Stats st{};
  {
    gv::Vector<float> vec("gnn_stream_feat", {0}, page_bytes, nblocks,
                          slots, padded_elems, CompressorPool(), codec, preset,
                          clio::cte::core::kCtePoolId);
    vec.EnableStats();

    ctp::GpuApi::Synchronize();
    double t0 = NowSec();
    if (ingest_file != nullptr) {
      // END-TO-END MODE. The matrix is loaded from an external raw stream --
      // whatever gnn_stream_npz.py produced -- instead of being written by a
      // kernel, so this exercises the same path the papers100M load takes.
      //
      // It has to happen in THIS process: the GPU queues are only ever created
      // by ServerInitGpuQueues, there is no client-side attach, so a process
      // that faults on the GPU must host the runtime itself. Loading via a
      // separate daemon and then attaching from the GPU side is not possible.
      REQUIRE(IngestFromFile(ingest_file, vec.TagId(), page_bytes, codec,
                             preset, pages_total));
    } else {
      RunYieldable(nblocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                gy::YieldStackView sv) {
        VecFillKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu_info, vec.GetDevice(0), d_src, elems_per_block, vw, sv);
      });
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    }
    t_store = NowSec() - t0;
    if (ingest_file != nullptr) {
      // Stored sizes must be published before a compressed page can be
      // fetched with its true (encoded) length.
      for (int a = 0; a < 50 && vec.PublishStoredSizes() < pages_total; ++a) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

    // Wipe the destination so a drain that silently does nothing cannot pass
    // by leaving the baseline bytes in place.
    REQUIRE(cudaMemset(d_feat, 0, feat_elems * sizeof(float)) == cudaSuccess);

    t0 = NowSec();
    RunYieldable(nblocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                              gy::YieldStackView sv) {
      VecDrainKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu_info, vec.GetDevice(0), d_feat, elems_per_block, vw, sv);
    });
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    t_stream = NowSec() - t0;

    st = vec.ReadStats(0);
  }

  // The point of this test is the STREAMING path. If the cache ever grows
  // enough to hold the whole matrix the test still passes -- it just stops
  // testing anything, because no page is ever evicted or re-faulted. Assert
  // the pressure rather than trusting the slot arithmetic above.
  std::fprintf(stderr,
               "[GNN-STREAM] cache stats: faults=%llu evicts=%llu puts=%llu "
               "put_errors=%llu get_errors=%llu\n",
               (unsigned long long)st.faults, (unsigned long long)st.evicts,
               (unsigned long long)st.puts,
               (unsigned long long)st.put_errors,
               (unsigned long long)st.get_errors);
  REQUIRE(st.evicts > 0);
  REQUIRE(st.put_errors == 0);

  // ---- lossless proof #1: the streamed bytes are the original bytes ----
  std::vector<float> feat_rt((size_t)feat_elems);
  cudaMemcpy(feat_rt.data(), d_feat, feat_elems * sizeof(float),
             cudaMemcpyDeviceToHost);
  int feat_cmp = std::memcmp(feat_rt.data(), ds.Feat(),
                             (size_t)(feat_elems * sizeof(float)));

  // ---- lossless proof #2: so the forward output is bitwise identical ----
  std::vector<float> logits_stream;
  forward(logits_stream);
  REQUIRE(logits_stream.size() == logits_base.size());
  int logit_cmp = std::memcmp(logits_stream.data(), logits_base.data(),
                              logits_base.size() * sizeof(float));
  double max_abs = 0.0;
  for (size_t i = 0; i < logits_base.size(); ++i) {
    max_abs = std::max(max_abs,
                       (double)std::fabs(logits_stream[i] - logits_base[i]));
  }

  const double mib = logical_bytes / 1048576.0;
  std::fprintf(stderr,
      "\n============ GNN zstd Vector<T> STREAM RESULTS ============\n"
      "  dataset   : %s  N=%lld F=%d E=%lld C=%d\n"
      "  path      : gv::Vector<float> -> compressor(600) -> core(512)\n"
      "  codec     : %s  preset=%d  (LOSSLESS)\n"
      "  write     : %.1f MiB/s (%.3fs, %llu pages, evicting)\n"
      "  stream    : %.1f MiB/s (%.3fs, every page a fault + decompress)\n"
      "  bit-exact : feature memcmp=%d  logit memcmp=%d  max_abs_diff=%.1f\n"
      "  RESULT    : %s\n"
      "==========================================================\n",
      ds.SourceName(), (long long)N, F, (long long)ds.E, C,
      gnn_test::CodecName(codec), preset,
      t_store > 0 ? mib / t_store : 0.0, t_store,
      (unsigned long long)pages_total,
      t_stream > 0 ? mib / t_stream : 0.0, t_stream,
      feat_cmp, logit_cmp, max_abs,
      (feat_cmp == 0 && logit_cmp == 0 && max_abs == 0.0) ? "PASS (bit-exact)"
                                                          : "FAIL");

  REQUIRE(feat_cmp == 0);
  REQUIRE(logit_cmp == 0);
  REQUIRE(max_abs == 0.0);

  cudaFree(d_indptr); cudaFree(d_indices);
  for (int i = 0; i < 4; ++i) cudaFree(dW[i]);
  cudaFree(d_feat); cudaFree(d_src); cudaFree(d_agg1); cudaFree(d_h1);
  cudaFree(d_agg2); cudaFree(d_logits);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif  // CUDA/ROCM && !SYCL
