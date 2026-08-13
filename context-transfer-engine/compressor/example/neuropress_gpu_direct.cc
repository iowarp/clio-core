/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_gpu_direct.cc
 * @brief Clio compresses a GPU-resident buffer with NeuroPress choosing the
 *        algorithm per chunk -- no HDF5 layer in between.
 *
 * The HDF5 VOL demo (adapter/hdf5_vol/example) shows the same capability
 * through a file API. This one hands the compressor a device pointer
 * directly, which is the shorter path and the one that actually exercises:
 *
 *   - a CUDA-IPC-registered device backend, so the ShmPtr the compressor
 *     receives resolves to a REAL device pointer on the runtime side rather
 *     than a host copy (AllocateAndRegisterGpuBackend);
 *   - on-device statistics -- entropy/MAD/second-derivative are computed by
 *     Clio's own CUDA kernels straight off that buffer, so the data is never
 *     staged to the host just to decide how to compress it;
 *   - NeuroPress ranking restricted to its trained GPU action space, which
 *     only holds when the buffer is genuinely device-resident.
 *
 * Round-trip verification is also done on-device, so from generation to
 * check the payload never touches host memory.
 *
 * Run: bin/neuropress_gpu_direct
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

// Defined in neuropress_gpu_direct_kernel.cu.
unsigned long long CountBoundViolationsOnDevice(const float *d_a,
                                                const float *d_b,
                                                size_t num_elems, double bound);
void LaunchFillRegimes(float *d_buf, size_t num_elems, size_t elems_per_chunk);
unsigned long long CountMismatchesOnDevice(const float *d_a, const float *d_b,
                                           size_t num_elems);

namespace {

/**
 * Dataset and chunk size, in MiB. Overridable so the same example can be run
 * at whatever shape a measurement calls for -- the interesting property here
 * is the device-resident path, not one particular size, and hardcoding meant
 * a second near-identical example every time the shape changed.
 *
 * Device memory needed is roughly 2x the dataset (source + readback
 * destination) plus one registered buffer per chunk, so 1 GiB of data wants
 * about 3 GiB free.
 */
size_t EnvMiB(const char *name, size_t fallback_mib) {
  const char *e = std::getenv(name);
  if (!e) return fallback_mib;
  const long v = std::atol(e);
  return v > 0 ? static_cast<size_t>(v) : fallback_mib;
}

const size_t kChunkBytes =
    EnvMiB("CLIO_NEUROPRESS_CHUNK_MIB", 4) << 20;
const size_t kTotalBytes =
    EnvMiB("CLIO_NEUROPRESS_TOTAL_MIB", 256) << 20;
const size_t kNumChunks = kTotalBytes / kChunkBytes;

const std::string kBackendFile = "/tmp/neuropress_gpu_direct_backend.dat";

/** Absolute error bound; 0 = lossless. See ctx.error_bound_ below. */
const double kErrorBound = [] {
  const char *e = std::getenv("CLIO_NEUROPRESS_ERROR_BOUND");
  return e ? std::atof(e) : 0.0;
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
    std::cerr << "No CUDA device available." << std::endl;
    return 1;
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
      static_cast<clio::run::u64>(1) << 30, clio::run::PoolQuery::Local(),
      clio::run::PoolId(900, 0));
  reg.Wait();
  if (reg->GetReturnCode() != 0) {
    std::cerr << "RegisterTarget failed rc=" << reg->GetReturnCode()
              << std::endl;
    return 1;
  }

  // NeuroPress model configured -> DynamicSchedule ranks with the NN.
  // Online learning left OFF (the library default, matching NeuroPress's
  // own g_online_learning_enabled{false}): this example is about selection.
  clio::run::PoolId compressor_pool_id(901, 1);
  clio::cte::compressor::CompressorConfig cfg;
  cfg.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  clio::cte::compressor::Client compressor;
  auto create = compressor.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(), "neuropress_gpu_direct_pool",
      compressor_pool_id, cfg);
  create.Wait();
  if (create->GetReturnCode() != 0) {
    std::cerr << "CreateCompressor failed rc=" << create->GetReturnCode()
              << std::endl;
    return 1;
  }
  compressor.Init(compressor_pool_id);

  auto tag = cte_client->AsyncGetOrCreateTag("neuropress_gpu_direct");
  tag.Wait();
  if (tag->GetReturnCode() != 0) {
    std::cerr << "GetOrCreateTag failed" << std::endl;
    return 1;
  }
  const auto tag_id = tag->tag_id_;

  // Declare the payload's element type. Context::data_type_ defaults to 0
  // (char/uint8), which would have NeuroPress compute MAD and the second
  // derivative over raw BYTES and set the data_type_char feature -- a
  // different input vector, and in practice a different algorithm choice.
  // This data is float32, so say so.
  clio::cte::core::Context ctx;
  ctx.data_type_ = 1;  // float
  // Error bound for the LOSSY half of NeuroPress's action space. 0 keeps the
  // run lossless and ranks 16 configs; anything positive makes the 16
  // quantize configs rankable too, for the full 32 (algorithm x quantize x
  // byte-shuffle). Same meaning as gpucompress_config_t::error_bound
  // ("0 = lossless"). Override with CLIO_NEUROPRESS_ERROR_BOUND.
  ctx.error_bound_ = kErrorBound;

  std::cout << "Data:  " << (kTotalBytes >> 20) << " MiB on-device, "
            << kNumChunks << " chunks of " << (kChunkBytes >> 20) << " MiB\n"
            << "Path:  device buffer -> Clio compressor (NeuroPress picks "
               "per chunk) -> CTE\n"
            << std::endl;

  // ---- Generate the payload directly in device memory. ----
  float *d_src = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, kTotalBytes));
  LaunchFillRegimes(d_src, kTotalBytes / sizeof(float),
                    kChunkBytes / sizeof(float));

  // ---- Compress each chunk from the GPU buffer. ----
  auto t0 = std::chrono::steady_clock::now();
  std::vector<clio::run::Future<clio::cte::compressor::DynamicScheduleTask>>
      futures;
  std::vector<ctp::ipc::AllocatorId> staged;
  futures.reserve(kNumChunks);
  staged.reserve(kNumChunks);

  for (size_t i = 0; i < kNumChunks; ++i) {
    char *device_chunk = reinterpret_cast<char *>(d_src) + i * kChunkBytes;

    // Register this chunk's device memory with the runtime so the ShmPtr
    // below resolves to a real device pointer through the CUDA IPC handle
    // rather than being staged to host. This is what keeps the statistics
    // on-device and the candidate set GPU-only.
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

    // AsyncDynamicSchedule: NeuroPress analyzes the chunk, ranks its trained
    // action space by cost, compresses with the winner, and stores via the
    // core client. Explicit core pool id -- the AsyncPutBlob override sends
    // a null one, which only resolves via compose config.
    futures.push_back(compressor.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), tag_id,
        "chunk_" + std::to_string(i), /*offset=*/0, kChunkBytes, blob_data,
        -1.0f, ctx, 0, cte_client->pool_id_));
    staged.push_back(alloc_id);
  }

  size_t failed = 0;
  for (auto &f : futures) {
    f.Wait();
    if (f->GetReturnCode() != 0) ++failed;
  }
  auto compress_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
  for (auto &id : staged) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, id);

  if (failed != 0) {
    std::cerr << failed << " chunk(s) failed to compress" << std::endl;
    return 1;
  }
  std::cout << std::fixed << std::setprecision(2)
            << "Compressed " << kNumChunks << " chunks in " << compress_ms
            << " ms (" << (kTotalBytes / (1024.0 * 1024.0)) / (compress_ms / 1000.0)
            << " MiB/s)" << std::endl;

  // ---- Report what NeuroPress actually chose, per chunk. ----
  std::cout << "\nPer-chunk selection (NeuroPress dynamic):\n";
  std::vector<size_t> per_lib(64, 0);
  for (size_t i = 0; i < kNumChunks; ++i) {
    const auto &ctx = futures[i]->context_;
    const int wire = ctx.compress_lib_;
    if (wire >= 0 && wire < 64) ++per_lib[static_cast<size_t>(wire)];
    if (i < 10) {
      std::cout << "  chunk " << std::setw(3) << i << "  "
                << ctp::CompressionFactory::NameForWireId(wire)
                << "  ratio=" << std::setprecision(3)
                << ctx.actual_compression_ratio_ << std::endl;
    }
  }
  std::cout << "  ...\n  totals:\n";
  for (size_t w = 0; w < per_lib.size(); ++w) {
    if (per_lib[w] == 0) continue;
    std::cout << "    " << std::setw(18)
              << ctp::CompressionFactory::NameForWireId(static_cast<int>(w))
              << " : " << per_lib[w] << " chunk(s)" << std::endl;
  }

  // ---- Read back into a second GPU buffer and verify on-device. ----
  float *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dst, kTotalBytes));
  auto t1 = std::chrono::steady_clock::now();
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

    auto get = compressor.AsyncDecompressExplicit(
        clio::run::PoolQuery::Local(), tag_id, "chunk_" + std::to_string(i),
        /*offset=*/0, kChunkBytes, 0, blob_data, cte_client->pool_id_);
    get.Wait();
    if (get->GetReturnCode() != 0) {
      std::cerr << "Decompress failed at chunk " << i
                << " rc=" << get->GetReturnCode() << std::endl;
      return 1;
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<char *>(d_dst) + i * kChunkBytes,
                          registered, kChunkBytes, cudaMemcpyDeviceToDevice));
    CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, alloc_id);
  }
  auto decompress_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t1).count();
  std::cout << "\nDecompressed in " << std::setprecision(2) << decompress_ms
            << " ms (" << (kTotalBytes / (1024.0 * 1024.0)) / (decompress_ms / 1000.0)
            << " MiB/s)" << std::endl;

  // Lossless runs must match bit for bit. A lossy run (error bound set)
  // legitimately does not -- what must hold instead is that every element
  // is within the bound the caller asked for, which is the guarantee
  // error-bounded quantization makes.
  const size_t num_elems = kTotalBytes / sizeof(float);
  unsigned long long mismatches = 0;
  if (kErrorBound > 0.0) {
    mismatches = CountBoundViolationsOnDevice(d_src, d_dst, num_elems,
                                              kErrorBound);
    std::cout << (mismatches == 0
                      ? "VERIFIED: every element within the error bound."
                      : "VIOLATION: elements exceeded the error bound.")
              << " (" << mismatches << " / " << num_elems
              << " outside +-" << kErrorBound << ")" << std::endl;
  } else {
    mismatches = CountMismatchesOnDevice(d_src, d_dst, num_elems);
    std::cout << (mismatches == 0 ? "VERIFIED: round trip matches exactly."
                                  : "MISMATCH: round trip corrupted data.")
              << " (" << mismatches << " / " << num_elems
              << " elements differ)" << std::endl;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return mismatches == 0 ? 0 : 1;
}
