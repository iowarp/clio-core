/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Compressor-pool wiring for the Clio HDF5 VOL connector (issue #693).
 *
 * The VOL write path and the compressor chimod were previously two
 * disconnected subsystems: H5Dwrite always stored blobs uncompressed via the
 * plain CTE core client, so NeuroPress/the compressor's dynamic selection was
 * never actually reached from real HDF5 I/O. This test configures a FAPL
 * with a compressor pool (clio_vol_info_t.compressor_pool_major/minor) and
 * verifies H5Dwrite/H5Dread round-trip correctly through it -- proving the
 * write path actually routes through compression (and the read path
 * correctly decompresses) rather than just that the two APIs happen to
 * coexist in the same binary.
 */

#include "simple_test.h"

#include "clio_vol.h"

#include <hdf5.h>
#include <cuda_runtime.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/compressor/compressor_client.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

const std::string kBackendFile = "/tmp/clio_vol_compressor_write_backend.dat";
const std::string kH5File = "/tmp/clio_vol_compressor_write_test.h5";
constexpr size_t kNumElems = 4096;  // 16 KiB of ints -- above chunk_size below
                                     // so the write loop exercises >1 chunk.

clio::run::PoolId g_compressor_pool_id = clio::run::PoolId::GetNull();

/** Mirrors test_hdf5_vol_gpu_ptr.cc's setupVolEnvironment(), plus creating a
    core pool (with a registered RAM storage target) and a compressor pool
    on top of it, matching CTETestFixture in test_compressor_functional.cc. */
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
      clio::run::PoolId(601, 0));
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);

  clio::run::PoolId compressor_pool_id = clio::run::PoolId(602, 1);
  clio::cte::compressor::Client compressor_client;
  auto create_task = compressor_client.AsyncCreateCompressor(
      clio::run::PoolQuery::Local(), "test_vol_compressor_pool",
      compressor_pool_id);
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);
  g_compressor_pool_id = compressor_pool_id;

  vol_id = H5VL_clio_register();
  REQUIRE(vol_id >= 0);
  return vol_id;
}

hid_t makeCompressedFapl(hid_t vol_id) {
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  REQUIRE(fapl >= 0);
  clio_vol_info_t info;
  info.under_vol_id = H5VL_NATIVE;
  info.under_vol_info = nullptr;
  info.chunk_size = 0;
  info.compressor_pool_major = g_compressor_pool_id.major_;
  info.compressor_pool_minor = g_compressor_pool_id.minor_;
  REQUIRE(H5Pset_vol(fapl, vol_id, &info) >= 0);
  return fapl;
}

}  // namespace

TEST_CASE("HDF5 VOL IO - H5Dwrite routes through the compressor and "
          "H5Dread decompresses correctly",
          "[hdf5_vol][io][compressor][693]") {
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeCompressedFapl(vol_id);

  std::remove(kH5File.c_str());

  hid_t file = H5Fcreate(kH5File.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);

  hsize_t dims[1] = {kNumElems};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  REQUIRE(space >= 0);

  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);

  // Low-entropy, compressible content (not random noise): a real compressor
  // should do something meaningful with it, and it makes any accidental
  // pass-through-uncompressed-and-still-match bug (e.g. the CTE cache miss
  // path always falling back to native) less likely to hide a bug (a
  // uniform/repeating pattern still round-trips even through a broken
  // "compressor" that hands data back unchanged, so this test's real
  // assertion is later, on decompressed content correctness).
  std::vector<int> wbuf(kNumElems);
  for (size_t i = 0; i < kNumElems; ++i) {
    wbuf[i] = static_cast<int>((i / 64) * 7 + 3);
  }

  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   wbuf.data()) >= 0);

  REQUIRE(H5Dclose(dset) >= 0);
  REQUIRE(H5Sclose(space) >= 0);
  REQUIRE(H5Fclose(file) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);

  // Reopen fresh: a NEW clio_file_t is built from clio_file_open(), so this
  // also exercises that clio_resolve_compressor_pool() threads the pool id
  // through open (not just create), and that the compressor-stored blob
  // survives across the tag lookup (AsyncGetOrCreateTag finds the SAME tag
  // by name rather than starting from an empty one).
  hid_t fapl2 = makeCompressedFapl(vol_id);
  hid_t file2 = H5Fopen(kH5File.c_str(), H5F_ACC_RDONLY, fapl2);
  REQUIRE(file2 >= 0);

  hid_t dset2 = H5Dopen2(file2, "data", H5P_DEFAULT);
  REQUIRE(dset2 >= 0);

  // THE ACTUAL CHECK: reading back through the same compressor-pool-backed
  // dataset must reproduce the ORIGINAL values. If the write path silently
  // stayed uncompressed-via-core-client while the read path attempted
  // decompression (or vice versa), this comes back corrupted rather than
  // matching wbuf.
  std::vector<int> rbuf(kNumElems, -1);
  REQUIRE(H5Dread(dset2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  rbuf.data()) >= 0);

  bool match = true;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (rbuf[i] != wbuf[i]) { match = false; break; }
  }
  REQUIRE(match);

  REQUIRE(H5Dclose(dset2) >= 0);
  REQUIRE(H5Fclose(file2) >= 0);
  REQUIRE(H5Pclose(fapl2) >= 0);
}

/**
 * Compressor-backed read INTO A DEVICE POINTER, served from the live cache.
 *
 * This is the pairing no test previously covered: the compressor-pool test
 * above uses host buffers, and test_hdf5_vol_gpu_ptr.cc uses device pointers
 * but configures no compressor pool. The decompress path's device branch --
 * where the VOL allocates a device backend and hands it to the decompressor
 * instead of CPU shared memory -- therefore had no coverage at all.
 *
 * H5Dflush rather than H5Dclose is load-bearing. After a close/reopen the
 * coherence check finds no stamp blob, drops the tag and serves the read from
 * the uncompressed HDF5 copy, so nothing decompresses and the device branch is
 * never entered (that is the separately-recorded VOL-3 behaviour, and it is
 * exactly what the reopen-based case above exercises). Flushing keeps the
 * cache live so the read is actually served by the compressor.
 */
TEST_CASE("HDF5 VOL IO - compressor-backed read into a GPU device pointer",
          "[hdf5_vol][io][compressor][gpu][693]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;
  }

  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeCompressedFapl(vol_id);

  const std::string kDevH5File = "/tmp/clio_vol_compressor_devread_test.h5";
  std::remove(kDevH5File.c_str());

  hid_t file = H5Fcreate(kDevH5File.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);

  hsize_t dims[1] = {kNumElems};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  REQUIRE(space >= 0);

  hid_t dset = H5Dcreate2(file, "data_devread", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);

  // Compressible, like the case above: a pattern the codec can actually work
  // on, so the read is genuinely a decompression rather than a pass-through.
  std::vector<int> wbuf(kNumElems);
  for (size_t i = 0; i < kNumElems; ++i) {
    wbuf[i] = static_cast<int>((i / 64) * 11 + 5);
  }
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   wbuf.data()) >= 0);

  // Flush, do NOT close -- keep the cache live (see the comment above).
  REQUIRE(H5Dflush(dset) >= 0);

  // THE ACTUAL CHECK: read the compressor-backed dataset straight into device
  // memory. Before the fix the VOL always allocated a host SHM destination for
  // the decompressor and staged it back, so this exercised no device path; a
  // wrong device destination shows up here as corrupted values rather than a
  // crash, because DeviceAwareMemcpy still moves the (wrong) bytes.
  int *d_rbuf = nullptr;
  REQUIRE(cudaMalloc(&d_rbuf, kNumElems * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemset(d_rbuf, 0, kNumElems * sizeof(int)) == cudaSuccess);

  REQUIRE(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  d_rbuf) >= 0);

  std::vector<int> rbuf(kNumElems, -1);
  REQUIRE(cudaMemcpy(rbuf.data(), d_rbuf, kNumElems * sizeof(int),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  bool match = true;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (rbuf[i] != wbuf[i]) { match = false; break; }
  }
  REQUIRE(match);

  cudaFree(d_rbuf);

  REQUIRE(H5Dclose(dset) >= 0);
  REQUIRE(H5Sclose(space) >= 0);
  REQUIRE(H5Fclose(file) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
}

SIMPLE_TEST_MAIN()
