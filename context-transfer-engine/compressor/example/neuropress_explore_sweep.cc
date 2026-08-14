/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_explore_sweep.cc
 * @brief Exhaustive K-way exploration over NeuroPress's full 32-configuration
 *        action space, with the lossy half enabled, logging every candidate.
 *
 * The other NeuroPress examples show SELECTION: one chunk in, one
 * configuration chosen, one blob out. This one shows the sweep behind that
 * choice. With K=32 and a positive error bound, every chunk is compressed with
 * all 31 alternatives to the model's pick -- 8 nvcomp algorithms x byte-shuffle
 * x quantize -- and each candidate's real ratio, kernel time, PSNR and cost is
 * written to a CSV. The point is to see what the model did NOT choose, which
 * the per-chunk selection log cannot express.
 *
 * Three settings make that happen, and none of them is the normal route:
 *
 *   - neuropress_exploration_k_ = 32 opens the whole action space. The shipped
 *     default is 3, matching NeuroPress's own K.
 *   - neuropress_exploration_threshold_ = 0 makes EVERY chunk sweep. Normally
 *     a sweep only runs when the model's prediction for the chunk it just
 *     compressed was off by more than 50%, so on a short run most chunks would
 *     never explore and the log would be nearly empty.
 *   - ctx.error_bound_ > 0 makes the 16 quantize configurations real. At 0 they
 *     are masked and skipped, and the sweep covers only the 16 lossless ones.
 *
 * Two chunks, because the interesting axis here is the 31 candidates within a
 * chunk rather than the number of chunks: 2 is enough to show the sweep repeat
 * on data the model has already learned from once.
 *
 * The blobs this writes are real -- when a candidate beats the model's pick on
 * cost, the sweep adopts it and the stored blob is the alternative's bytes. The
 * round-trip check at the end reads back through the normal decompress path, so
 * an adopted winner has to decode correctly like any other blob.
 *
 * Run: bin/neuropress_explore_sweep
 *      CLIO_NEUROPRESS_EXPLORE_LOG=/tmp/sweep.csv bin/neuropress_explore_sweep
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_ctp/compress/compress_factory.h>

// Defined in neuropress_gpu_direct_kernel.cu -- the same generator the
// device-direct example uses, so a sweep here is over the same regimes that
// example's selections were measured on.
void LaunchFillRegimes(float *d_buf, size_t num_elems, size_t elems_per_chunk);
unsigned long long CountBoundViolationsOnDevice(const float *d_a,
                                                const float *d_b,
                                                size_t num_elems, double bound);

namespace {

constexpr size_t kChunkBytes = 4ull << 20;  // 4 MiB
constexpr size_t kNumChunks = 2;
constexpr size_t kTotalBytes = kChunkBytes * kNumChunks;

const std::string kBackendFile = "/tmp/neuropress_explore_sweep_backend.dat";

/**
 * Absolute error bound. MUST be > 0 for this example to do what it says: it is
 * what makes the 16 quantize actions rankable instead of masked, and so what
 * takes the sweep from 16 configurations to 32. Same meaning as
 * gpucompress_config_t::error_bound.
 */
const double kErrorBound = [] {
  const char *e = std::getenv("CLIO_NEUROPRESS_ERROR_BOUND");
  const double v = e ? std::atof(e) : 1e-3;
  return v > 0.0 ? v : 1e-3;
}();

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t rc_ = (expr);                                                \
    if (rc_ != cudaSuccess) {                                                \
      std::cerr << "CUDA error " << __LINE__ << ": "                         \
                << cudaGetErrorString(rc_) << std::endl;                     \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cerr << "No CUDA device available -- the sweep is GPU-only."
              << std::endl;
    return 1;
  }

  const char *log_path = std::getenv("CLIO_NEUROPRESS_EXPLORE_LOG");
  if (!log_path || !*log_path) {
    std::cout << "note: CLIO_NEUROPRESS_EXPLORE_LOG is unset, so the "
                 "per-candidate CSV will not be written.\n"
                 "      The sweep still runs; set it to a path to capture it.\n"
              << std::endl;
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::cerr << "CLIO_INIT failed" << std::endl;
    return 1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "CLIO_CTE_CLIENT_INIT failed" << std::endl;
    return 1;
  }
  auto *cte_client = CLIO_CTE_CLIENT;

  auto reg = cte_client->AsyncRegisterTarget(
      kBackendFile, clio::run::bdev::BdevType::kFile,
      static_cast<clio::run::u64>(512) << 20, clio::run::PoolQuery::Local(),
      clio::run::PoolId(902, 0));
  reg.Wait();
  if (reg->GetReturnCode() != 0) {
    std::cerr << "RegisterTarget failed rc=" << reg->GetReturnCode()
              << std::endl;
    return 1;
  }

  clio::run::PoolId compressor_pool_id(903, 1);
  clio::cte::compressor::CompressorConfig cfg;
  cfg.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  // Exploration lives inside the online-learning block, so learning has to be
  // on for the sweep to run at all -- upstream nests them the same way.
  cfg.neuropress_online_learning_enabled_ = true;
  cfg.neuropress_exploration_enabled_ = true;
  // The whole remaining action space. Upstream clamps K to 31 (32 configs
  // minus the primary); passing 32 here reaches the same place.
  cfg.neuropress_exploration_k_ = 32;
  // Sweep on EVERY chunk rather than only on a badly mispredicted one. This is
  // the setting that makes the example reproducible: at the 0.50 default,
  // whether a given chunk explores depends on how far that chunk's measured
  // cost drifted from its prediction, which on a 2-chunk run is mostly a
  // function of GPU warm-up.
  //
  // NEGATIVE, not 0. The gate is a strict `error_pct > threshold`, and
  // error_pct is genuinely 0 whenever a chunk's measured cost lands exactly on
  // its prediction -- which is common here rather than exotic, because the cost
  // model floors compression time at 1 ms and clamps ratio at 100, so any fast
  // chunk compressing better than 100:1 produces identical predicted and actual
  // costs. At 0.0 those chunks silently do not sweep, which is precisely the
  // case this example must not miss.
  cfg.neuropress_exploration_threshold_ = -1.0f;
  clio::cte::compressor::Client compressor;
  auto create = compressor.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(), "neuropress_explore_sweep_pool",
      compressor_pool_id, cfg);
  create.Wait();
  if (create->GetReturnCode() != 0) {
    std::cerr << "CreateCompressor failed rc=" << create->GetReturnCode()
              << std::endl;
    return 1;
  }
  compressor.Init(compressor_pool_id);

  auto tag = cte_client->AsyncGetOrCreateTag("neuropress_explore_sweep");
  tag.Wait();
  if (tag->GetReturnCode() != 0) {
    std::cerr << "GetOrCreateTag failed" << std::endl;
    return 1;
  }
  const auto tag_id = tag->tag_id_;

  clio::cte::core::Context ctx;
  ctx.data_type_ = 1;  // float32 -- see neuropress_gpu_direct.cc
  ctx.error_bound_ = kErrorBound;

  std::cout << "Data:        " << (kTotalBytes >> 20) << " MiB on-device, "
            << kNumChunks << " chunks of " << (kChunkBytes >> 20) << " MiB\n"
            << "Error bound: " << kErrorBound
            << "  (quantize actions ENABLED -> 32 configurations)\n"
            << "Exploration: K=" << cfg.neuropress_exploration_k_
            << ", threshold=" << cfg.neuropress_exploration_threshold_
            << " (every chunk sweeps)\n"
            << std::endl;

  // Fill one chunk MORE than we use, and skip the first.
  //
  // LaunchFillRegimes cycles five regimes and its chunk 0 is constant 42.0f.
  // A constant chunk has zero data range, and analytical PSNR is undefined on
  // a zero range (AnalyticalPsnr returns -1 for exactly that reason), so every
  // quantized candidate over it reports no PSNR. That is correct, but on a
  // two-chunk run it would leave half the sweep saying nothing about the lossy
  // half of the action space -- which is the half this example exists to
  // exercise. Starting at regime 1 gives a smooth sine and a stepped ramp,
  // both of which quantize meaningfully.
  float *d_gen = nullptr;
  CUDA_CHECK(cudaMalloc(&d_gen, kTotalBytes + kChunkBytes));
  LaunchFillRegimes(d_gen, (kTotalBytes + kChunkBytes) / sizeof(float),
                    kChunkBytes / sizeof(float));
  CUDA_CHECK(cudaDeviceSynchronize());
  float *d_src =
      reinterpret_cast<float *>(reinterpret_cast<char *>(d_gen) + kChunkBytes);

  auto t0 = std::chrono::steady_clock::now();
  std::vector<clio::run::Future<clio::cte::compressor::DynamicScheduleTask>>
      futures;
  std::vector<ctp::ipc::AllocatorId> staged;
  futures.reserve(kNumChunks);
  staged.reserve(kNumChunks);

  for (size_t i = 0; i < kNumChunks; ++i) {
    char *device_chunk = reinterpret_cast<char *>(d_src) + i * kChunkBytes;
    char *registered = nullptr;
    ctp::ipc::AllocatorId alloc_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
        /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        kChunkBytes, &registered);
    if (alloc_id.IsNull()) {
      std::cerr << "AllocateAndRegisterGpuBackend failed at chunk " << i
                << std::endl;
      return 1;
    }
    CUDA_CHECK(cudaMemcpy(registered, device_chunk, kChunkBytes,
                          cudaMemcpyDeviceToDevice));

    ctp::ipc::ShmPtr<> blob_data;
    blob_data.alloc_id_ = alloc_id;
    blob_data.off_ = reinterpret_cast<clio::run::u64>(registered);

    // Submitted and awaited one at a time. The sweep runs INSIDE this call and
    // its 31 compressions are already overlapped on their own CUDA streams;
    // letting two chunks' sweeps interleave would put 62 in flight and make
    // each candidate's measured kernel time a function of how many others
    // happened to be resident, which is the one number this example exists to
    // report.
    auto fut = compressor.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), tag_id,
        "chunk_" + std::to_string(i), /*offset=*/0, kChunkBytes, blob_data,
        -1.0f, ctx, 0, cte_client->pool_id_);
    fut.Wait();
    if (fut->GetReturnCode() != 0) {
      std::cerr << "chunk " << i << " failed rc=" << fut->GetReturnCode()
                << std::endl;
      return 1;
    }
    std::cout << "chunk " << i << ": stored "
              << fut->context_.actual_compressed_size_ << " B of "
              << kChunkBytes << " B (ratio "
              << std::fixed << std::setprecision(2)
              << fut->context_.actual_compression_ratio_ << ", "
              << fut->context_.actual_compress_time_ms_ << " ms, lib "
              << ctp::CompressionFactory::NameForWireId(
                     fut->context_.compress_lib_)
              << ")" << std::endl;
    futures.push_back(std::move(fut));
    staged.push_back(alloc_id);
  }

  const auto elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
  for (auto &id : staged) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, id);

  std::cout << "\n" << kNumChunks << " chunks swept in " << elapsed_ms
            << " ms (each chunk compressed up to 32 ways)" << std::endl;

  // ---- Read back and check the bound held. ----
  // Every stored blob is lossy here, so an exact comparison is the wrong test:
  // what must hold is that no element moved further than the error bound. That
  // applies to an adopted sweep winner exactly as it does to the model's own
  // pick, which is the property worth checking -- the winner's quantization
  // parameters travel in its header and nothing else knows they changed.
  float *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dst, kTotalBytes));
  size_t read_failed = 0;
  for (size_t i = 0; i < kNumChunks; ++i) {
    char *registered = nullptr;
    ctp::ipc::AllocatorId alloc_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
        /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        kChunkBytes, &registered);
    if (alloc_id.IsNull()) {
      std::cerr << "readback registration failed at chunk " << i << std::endl;
      return 1;
    }
    ctp::ipc::ShmPtr<> blob_data;
    blob_data.alloc_id_ = alloc_id;
    blob_data.off_ = reinterpret_cast<clio::run::u64>(registered);

    auto dec = compressor.AsyncDecompressExplicit(
        clio::run::PoolQuery::Local(), tag_id,
        "chunk_" + std::to_string(i), /*offset=*/0, kChunkBytes, 0, blob_data,
        cte_client->pool_id_);
    dec.Wait();
    if (dec->GetReturnCode() != 0) {
      ++read_failed;
    } else {
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>(d_dst) + i * kChunkBytes,
                            registered, kChunkBytes, cudaMemcpyDeviceToDevice));
    }
    CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, alloc_id);
  }

  if (read_failed != 0) {
    std::cerr << read_failed << " chunk(s) failed to decompress" << std::endl;
    return 1;
  }

  const unsigned long long violations = CountBoundViolationsOnDevice(
      d_src, d_dst, kTotalBytes / sizeof(float), kErrorBound);
  std::cout << (violations == 0
                    ? "VERIFIED: every element within the error bound."
                    : "FAILED: elements outside the error bound.")
            << " (" << violations << " violations)" << std::endl;

  cudaFree(d_gen);
  cudaFree(d_dst);
  if (log_path && *log_path) {
    std::cout << "\nPer-candidate sweep log: " << log_path << std::endl;
  }

  clio::run::CLIO_RUNTIME_FINALIZE();
  return violations == 0 ? 0 : 1;
}
