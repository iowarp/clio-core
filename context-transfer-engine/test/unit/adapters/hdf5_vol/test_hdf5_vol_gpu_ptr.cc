/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GPU device-pointer passthrough for the Clio HDF5 VOL connector.
 *
 * H5Dwrite/H5Dread accept whatever buffer pointer the caller passes. If that
 * pointer is a real CUDA device allocation (cudaMalloc, not managed/UVM
 * memory), the connector's write/read paths must NOT touch it with a plain
 * host std::memcpy -- that dereferences device memory from host code, which
 * is not a "wrong answer" bug, it is undefined behavior/a crash.
 *
 * This is its own executable (not folded into test_hdf5_vol_io.cc) so a
 * regression here fails as its own isolated, clearly-attributed ctest entry
 * instead of taking down the whole test_hdf5_vol_io process with it.
 */

#include "simple_test.h"

#include "clio_vol.h"

#include <hdf5.h>
#include <cuda_runtime.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

const std::string kBackendFile = "/tmp/clio_vol_gpu_ptr_backend.dat";
const std::string kH5File = "/tmp/clio_vol_gpu_ptr_test.h5";
constexpr size_t kNumElems = 4096;  // 16 KiB of ints

/** Mirrors test_hdf5_vol_io.cc's setupVolEnvironment(). */
hid_t setupVolEnvironment() {
  static hid_t vol_id = H5I_INVALID_HID;
  if (vol_id != H5I_INVALID_HID) {
    return vol_id;
  }

#ifdef _WIN32
  _putenv_s("CLIO_VOL_CHUNK_SIZE", "4096");
#else
  setenv("CLIO_VOL_CHUNK_SIZE", "4096", 1);
#endif

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  auto *cte_client = CLIO_CTE_CLIENT;
  auto reg_task = cte_client->AsyncRegisterTarget(
      kBackendFile, clio::run::bdev::BdevType::kFile,
      static_cast<clio::run::u64>(64) * 1024 * 1024, clio::run::PoolQuery::Local(),
      clio::run::PoolId(600, 0));
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);

  vol_id = H5VL_clio_register();
  REQUIRE(vol_id >= 0);
  return vol_id;
}

hid_t makeFapl(hid_t vol_id) {
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  REQUIRE(fapl >= 0);
  clio_vol_info_t info;
  info.under_vol_id = H5VL_NATIVE;
  info.under_vol_info = nullptr;
  info.chunk_size = 0;
  REQUIRE(H5Pset_vol(fapl, vol_id, &info) >= 0);
  return fapl;
}

}  // namespace

TEST_CASE("HDF5 VOL IO - Write/read survives a real GPU device pointer",
          "[hdf5_vol][io][gpu]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    // No GPU in this environment -- nothing to prove either way.
    return;
  }

  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);

  std::remove(kH5File.c_str());

  hid_t file = H5Fcreate(kH5File.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);

  hsize_t dims[1] = {kNumElems};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  REQUIRE(space >= 0);

  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);

  // Known content, built on the host, then staged into a REAL device
  // allocation (not managed/UVM memory -- that would already be
  // host-dereferenceable and wouldn't exercise the bug at all).
  std::vector<int> wbuf(kNumElems);
  for (size_t i = 0; i < kNumElems; ++i) {
    wbuf[i] = static_cast<int>(i * 3 + 1);
  }

  int *d_wbuf = nullptr;
  REQUIRE(cudaMalloc(&d_wbuf, kNumElems * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_wbuf, wbuf.data(), kNumElems * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  // THE ACTUAL CHECK: H5Dwrite with a device pointer as the buffer. If the
  // connector's write path does a raw std::memcpy on this pointer, this
  // either segfaults (killing this isolated test process, which ctest
  // reports as a failure) or silently corrupts data.
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   d_wbuf) >= 0);

  int *d_rbuf = nullptr;
  REQUIRE(cudaMalloc(&d_rbuf, kNumElems * sizeof(int)) == cudaSuccess);
  REQUIRE(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  d_rbuf) >= 0);

  std::vector<int> rbuf(kNumElems, 0);
  REQUIRE(cudaMemcpy(rbuf.data(), d_rbuf, kNumElems * sizeof(int),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  bool match = true;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (rbuf[i] != wbuf[i]) { match = false; break; }
  }
  REQUIRE(match);

  cudaFree(d_wbuf);
  cudaFree(d_rbuf);

  REQUIRE(H5Dclose(dset) >= 0);
  REQUIRE(H5Sclose(space) >= 0);
  REQUIRE(H5Fclose(file) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
}

SIMPLE_TEST_MAIN()
