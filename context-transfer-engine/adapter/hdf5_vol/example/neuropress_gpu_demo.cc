/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * NeuroPress GPU demo (issue #693).
 *
 * Runs a real Gray-Scott reaction-diffusion simulation on the GPU -- the
 * same model context-transfer-engine/adapter/kvhdf5/test/e2e/
 * gray_scott_gpu_test.cu uses -- for 32 timesteps, each step producing a
 * 2 MiB snapshot of the evolving field directly on-device (see
 * neuropress_gpu_demo_kernel.cu). The resulting 64 MiB run of snapshots is
 * written through the Clio HDF5 VOL connector with a compressor pool
 * configured; with a compressor pool set, clio_dataset_write routes every
 * 2 MiB chunk (one simulation timestep) through
 * compressor::Client::AsyncDynamicSchedule instead of the raw uncompressed
 * core client, so each timestep's snapshot independently goes through
 * NeuroPress's dynamic algorithm selection -- exactly the "simulation
 * produces data on the GPU, hands it to Clio+NeuroPress" scenario, not a
 * synthetic fill pattern.
 *
 * This file is a plain host translation unit (not compiled by nvcc): the
 * simulation kernel lives in its own TU precisely so this one can safely
 * include clio_cte/compressor headers. See neuropress_gpu_demo_kernel.cu
 * for why.
 *
 * Run with: bin/neuropress_gpu_demo
 */

#include "clio_vol.h"

#include <hdf5.h>
#include <cuda_runtime.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/compressor/compressor_client.h>

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Defined in neuropress_gpu_demo_kernel.cu.
void RunGrayScottSimulation(float *d_snapshots, unsigned grid_w,
                            unsigned grid_h, unsigned num_steps);

namespace {

constexpr size_t kChunkBytes = 2ull * 1024 * 1024;       // 2 MiB
constexpr size_t kDatasetBytes = 64ull * 1024 * 1024;    // 64 MiB
constexpr size_t kNumElems = kDatasetBytes / sizeof(float);
constexpr size_t kElemsPerChunk = kChunkBytes / sizeof(float);
constexpr size_t kNumChunks = kDatasetBytes / kChunkBytes;

// One simulation timestep's field per chunk: grid_w * grid_h floats sized to
// exactly match kChunkBytes, and kNumChunks steps to fill the dataset.
constexpr unsigned kGridW = 1024;
constexpr unsigned kGridH = 512;
static_assert(static_cast<size_t>(kGridW) * kGridH == kElemsPerChunk,
             "grid dimensions must produce exactly one chunk per timestep");

const std::string kBackendFile = "/tmp/neuropress_gpu_demo_backend.dat";
const std::string kH5File = "/tmp/neuropress_gpu_demo.h5";

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    cudaError_t _rc = (expr);                                               \
    if (_rc != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << cudaGetErrorString(_rc) << std::endl;                    \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

hid_t SetupVolEnvironment(clio::run::PoolId *compressor_pool_id) {
  setenv("CLIO_VOL_CHUNK_SIZE", "2097152", 1);

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::cerr << "CLIO_INIT failed" << std::endl;
    std::exit(1);
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "CLIO_CTE_CLIENT_INIT failed" << std::endl;
    std::exit(1);
  }

  auto *cte_client = CLIO_CTE_CLIENT;
  auto reg_task = cte_client->AsyncRegisterTarget(
      kBackendFile, clio::run::bdev::BdevType::kFile,
      static_cast<clio::run::u64>(256) * 1024 * 1024,
      clio::run::PoolQuery::Local(), clio::run::PoolId(800, 0));
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    std::cerr << "AsyncRegisterTarget failed: rc="
               << reg_task->GetReturnCode() << std::endl;
    std::exit(1);
  }

  *compressor_pool_id = clio::run::PoolId(801, 1);
  clio::cte::compressor::CompressorConfig compressor_config;
  compressor_config.neuropress_model_path_ = CLIO_CTP_NEUROPRESS_WEIGHTS_DIR;
  // Off by default in the library (matches NeuroPress's own conservative
  // default); this demo opts in so the K-way exploration path (Cycle 4g)
  // actually runs and is visible in the log.
  compressor_config.neuropress_exploration_enabled_ = true;
  clio::cte::compressor::Client compressor_client;
  auto create_task = compressor_client.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(), "neuropress_gpu_demo_pool",
      *compressor_pool_id, compressor_config);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    std::cerr << "AsyncCreateCompressor failed: rc="
               << create_task->GetReturnCode() << std::endl;
    std::exit(1);
  }

  hid_t vol_id = H5VL_clio_register();
  if (vol_id < 0) {
    std::cerr << "H5VL_clio_register failed" << std::endl;
    std::exit(1);
  }
  return vol_id;
}

hid_t MakeCompressedFapl(hid_t vol_id, const clio::run::PoolId &pool_id) {
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  clio_vol_info_t info = {};
  info.under_vol_id = H5VL_NATIVE;
  info.under_vol_info = nullptr;
  info.chunk_size = kChunkBytes;
  info.compressor_pool_major = pool_id.major_;
  info.compressor_pool_minor = pool_id.minor_;
  if (H5Pset_vol(fapl, vol_id, &info) < 0) {
    std::cerr << "H5Pset_vol failed" << std::endl;
    std::exit(1);
  }
  return fapl;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA device available -- nothing to demonstrate."
               << std::endl;
    return 1;
  }

  std::cout << "Dataset:    " << kDatasetBytes / (1024 * 1024) << " MiB ("
            << kNumElems << " floats)\n"
            << "Chunk size: " << kChunkBytes / (1024 * 1024) << " MiB\n"
            << "Chunks:     " << kNumChunks
            << " (each independently routed through NeuroPress)\n"
            << std::endl;

  clio::run::PoolId compressor_pool_id;
  hid_t vol_id = SetupVolEnvironment(&compressor_pool_id);
  hid_t fapl = MakeCompressedFapl(vol_id, compressor_pool_id);

  std::remove(kH5File.c_str());
  hid_t file = H5Fcreate(kH5File.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (file < 0) { std::cerr << "H5Fcreate failed" << std::endl; return 1; }

  hsize_t dims[1] = {kNumElems};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "neuropress_demo_data", H5T_NATIVE_FLOAT,
                          space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) { std::cerr << "H5Dcreate2 failed" << std::endl; return 1; }

  // Run the simulation directly on the GPU -- each of the kNumChunks steps'
  // evolved field lands straight into its chunk slot in d_wbuf, entirely
  // device-to-device. The host never touches simulation state until the
  // verification step below.
  float *d_wbuf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_wbuf, kDatasetBytes));
  RunGrayScottSimulation(d_wbuf, kGridW, kGridH,
                        static_cast<unsigned>(kNumChunks));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Ran " << kNumChunks << " Gray-Scott timesteps on a " << kGridW
            << "x" << kGridH << " grid (" << kDatasetBytes / (1024 * 1024)
            << " MiB of snapshots, entirely on-device)." << std::endl;

  auto write_start = std::chrono::steady_clock::now();
  if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              d_wbuf) < 0) {
    std::cerr << "H5Dwrite failed" << std::endl;
    return 1;
  }
  auto write_elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    write_start)
          .count();
  double write_mb_s = (kDatasetBytes / (1024.0 * 1024.0)) / write_elapsed;
  std::cout << std::fixed << std::setprecision(2)
            << "H5Dwrite (GPU buffer -> NeuroPress compressor pool): "
            << write_elapsed * 1000.0 << " ms (" << write_mb_s << " MiB/s)"
            << std::endl;

  if (H5Dclose(dset) < 0 || H5Sclose(space) < 0 || H5Fclose(file) < 0 ||
      H5Pclose(fapl) < 0) {
    std::cerr << "close after write failed" << std::endl;
    return 1;
  }

  // Reopen fresh so the read genuinely goes through decompression rather
  // than reusing any in-process handle from the write above.
  hid_t fapl2 = MakeCompressedFapl(vol_id, compressor_pool_id);
  hid_t file2 = H5Fopen(kH5File.c_str(), H5F_ACC_RDONLY, fapl2);
  if (file2 < 0) { std::cerr << "H5Fopen failed" << std::endl; return 1; }
  hid_t dset2 = H5Dopen2(file2, "neuropress_demo_data", H5P_DEFAULT);
  if (dset2 < 0) { std::cerr << "H5Dopen2 failed" << std::endl; return 1; }

  float *d_rbuf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rbuf, kDatasetBytes));

  auto read_start = std::chrono::steady_clock::now();
  if (H5Dread(dset2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             d_rbuf) < 0) {
    std::cerr << "H5Dread failed" << std::endl;
    return 1;
  }
  auto read_elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    read_start)
          .count();
  double read_mb_s = (kDatasetBytes / (1024.0 * 1024.0)) / read_elapsed;
  std::cout << "H5Dread  (NeuroPress decompress -> GPU buffer):      "
            << read_elapsed * 1000.0 << " ms (" << read_mb_s << " MiB/s)"
            << std::endl;

  // Verification: the only host round trip in this whole demo, and only to
  // prove correctness -- the write/read paths above stayed GPU-resident.
  std::vector<float> wbuf(kNumElems), rbuf(kNumElems);
  CUDA_CHECK(cudaMemcpy(wbuf.data(), d_wbuf, kDatasetBytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rbuf.data(), d_rbuf, kDatasetBytes,
                        cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (wbuf[i] != rbuf[i]) ++mismatches;
  }
  std::cout << (mismatches == 0
                   ? "VERIFIED: round trip matches exactly."
                   : "MISMATCH: round trip corrupted data.")
            << " (" << mismatches << " / " << kNumElems
            << " elements differ)" << std::endl;

  cudaFree(d_wbuf);
  cudaFree(d_rbuf);
  H5Dclose(dset2);
  H5Fclose(file2);
  H5Pclose(fapl2);

  clio::run::CLIO_RUNTIME_FINALIZE();

  return mismatches == 0 ? 0 : 1;
}
