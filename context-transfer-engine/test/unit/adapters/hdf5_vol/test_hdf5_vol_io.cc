/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * End-to-end I/O tests for the IOWarp HDF5 VOL connector.
 *
 * Unlike test_hdf5_vol_adapter.cc (which only touches the plugin registration
 * and property-list surface), this suite drives *real* HDF5 file, dataset,
 * group, and attribute operations through the connector against an in-process
 * Chimaera/CTE runtime. That exercises the data-path callbacks the base suite
 * cannot reach: iowarp_file_create/open/close, iowarp_dataset_create/open/
 * write/read/close (the chunked AsyncPutBlob / read-through-cache paths),
 * group create/open/close, attr create/write/read/close, and the wrap/unwrap/
 * info callbacks invoked while HDF5 manages object handles.
 *
 * The connector lazily attaches to the running runtime via get_cte_client()
 * (CLIO_CTE_CLIENT_INIT on first use), so we only need to stand up the runtime
 * and register one bdev target before issuing HDF5 calls.
 */

#include "simple_test.h"

#include "iowarp_vol.h"

#include <hdf5.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

const std::string kBackendFile = "/tmp/clio_vol_io_backend.dat";
const std::string kH5File = "/tmp/clio_vol_io_test.h5";
constexpr size_t kNumElems = 4096;  // 16 KiB of ints

/**
 * Stand up the in-process runtime + CTE client + a file bdev target exactly
 * once. Mirrors initializeRuntime() in the POSIX adapter test. Returns the VOL
 * connector id (registered once, valid for the process lifetime).
 */
hid_t setupVolEnvironment() {
  static hid_t vol_id = H5I_INVALID_HID;
  if (vol_id != H5I_INVALID_HID) {
    return vol_id;
  }

  // Small chunk size so the multi-chunk AsyncPutBlob loop is exercised even by
  // a modest dataset (16 KiB / 4 KiB = 4 chunks). setenv is POSIX-only.
#ifdef _WIN32
  _putenv_s("IOWARP_VOL_CHUNK_SIZE", "4096");
#else
  setenv("IOWARP_VOL_CHUNK_SIZE", "4096", 1);
#endif

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  // Register a file-backed target into the default CTE pool so the connector's
  // AsyncPutBlob has somewhere to place data. bdev_id must be explicit (a null
  // PoolId is rejected by the pool manager).
  auto *cte_client = CLIO_CTE_CLIENT;
  auto reg_task = cte_client->AsyncRegisterTarget(
      kBackendFile, clio::run::bdev::BdevType::kFile,
      static_cast<clio::run::u64>(64) * 1024 * 1024, clio::run::PoolQuery::Local(),
      clio::run::PoolId(600, 0));
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);

  vol_id = H5VL_iowarp_register();
  REQUIRE(vol_id >= 0);
  return vol_id;
}

/** Build a file-access property list bound to the IOWarp VOL connector. */
hid_t makeFapl(hid_t vol_id) {
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  REQUIRE(fapl >= 0);
  iowarp_vol_info_t info;
  info.under_vol_id = H5VL_NATIVE;
  info.under_vol_info = nullptr;
  info.chunk_size = 0;  // fall back to the env-var chunk size
  REQUIRE(H5Pset_vol(fapl, vol_id, &info) >= 0);
  return fapl;
}

}  // namespace

TEST_CASE("HDF5 VOL IO - Create-Write-Read whole dataset",
          "[hdf5_vol][io]") {
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);

  std::remove(kH5File.c_str());

  // --- Create file + dataset, write a whole contiguous dataset -------------
  hid_t file = H5Fcreate(kH5File.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);

  hsize_t dims[1] = {kNumElems};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  REQUIRE(space >= 0);

  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);

  std::vector<int> wbuf(kNumElems);
  for (size_t i = 0; i < kNumElems; ++i) {
    wbuf[i] = static_cast<int>(i * 3 + 1);
  }
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   wbuf.data()) >= 0);

  // Read back through the same handle (chunks were submitted on write and are
  // flushed on dataset close; on this path the read falls through to native).
  std::vector<int> rbuf(kNumElems, 0);
  REQUIRE(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  rbuf.data()) >= 0);
  bool match = true;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (rbuf[i] != wbuf[i]) { match = false; break; }
  }
  REQUIRE(match);

  REQUIRE(H5Dclose(dset) >= 0);
  REQUIRE(H5Sclose(space) >= 0);
  REQUIRE(H5Fclose(file) >= 0);

  // --- Reopen and read the dataset back ------------------------------------
  hid_t file2 = H5Fopen(kH5File.c_str(), H5F_ACC_RDONLY, fapl);
  REQUIRE(file2 >= 0);
  hid_t dset2 = H5Dopen2(file2, "data", H5P_DEFAULT);
  REQUIRE(dset2 >= 0);

  std::vector<int> rbuf2(kNumElems, 0);
  REQUIRE(H5Dread(dset2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  rbuf2.data()) >= 0);
  match = true;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (rbuf2[i] != wbuf[i]) { match = false; break; }
  }
  REQUIRE(match);

  REQUIRE(H5Dclose(dset2) >= 0);
  REQUIRE(H5Fclose(file2) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
}

TEST_CASE("HDF5 VOL IO - Groups and attributes", "[hdf5_vol][io]") {
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);

  const std::string path = "/tmp/clio_vol_io_grpattr.h5";
  std::remove(path.c_str());

  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);

  // Group create/close through the connector.
  hid_t grp = H5Gcreate2(file, "grp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(grp >= 0);

  // A small dataset inside the group (whole write path again).
  hsize_t dims[1] = {16};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(grp, "inner", H5T_NATIVE_DOUBLE, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);
  std::vector<double> dbuf(16);
  for (size_t i = 0; i < dbuf.size(); ++i) dbuf[i] = i + 0.5;
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   dbuf.data()) >= 0);

  // Attribute create/write/read/close through the connector.
  hid_t ascalar = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(dset, "meta", H5T_NATIVE_INT, ascalar,
                          H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(attr >= 0);
  int aval = 1234;
  REQUIRE(H5Awrite(attr, H5T_NATIVE_INT, &aval) >= 0);
  int aread = 0;
  REQUIRE(H5Aread(attr, H5T_NATIVE_INT, &aread) >= 0);
  REQUIRE(aread == 1234);
  REQUIRE(H5Aclose(attr) >= 0);
  REQUIRE(H5Sclose(ascalar) >= 0);

  REQUIRE(H5Dclose(dset) >= 0);
  REQUIRE(H5Sclose(space) >= 0);
  REQUIRE(H5Gclose(grp) >= 0);
  REQUIRE(H5Fclose(file) >= 0);

  // Reopen group + dataset to drive the open callbacks.
  hid_t file2 = H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl);
  REQUIRE(file2 >= 0);
  hid_t grp2 = H5Gopen2(file2, "grp", H5P_DEFAULT);
  REQUIRE(grp2 >= 0);
  hid_t dset2 = H5Dopen2(grp2, "inner", H5P_DEFAULT);
  REQUIRE(dset2 >= 0);
  std::vector<double> dread(16, 0.0);
  REQUIRE(H5Dread(dset2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  dread.data()) >= 0);
  REQUIRE(dread[0] == 0.5);
  REQUIRE(dread[15] == 15.5);

  REQUIRE(H5Dclose(dset2) >= 0);
  REQUIRE(H5Gclose(grp2) >= 0);
  REQUIRE(H5Fclose(file2) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
}

TEST_CASE("HDF5 VOL IO - Passthrough callbacks", "[hdf5_vol][io]") {
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);

  const std::string path = "/tmp/clio_vol_io_passthru.h5";
  std::remove(path.c_str());

  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);

  // file_get / file_specific passthroughs.
  unsigned intent = 0;
  REQUIRE(H5Fget_intent(file, &intent) >= 0);
  REQUIRE(H5Fflush(file, H5F_SCOPE_GLOBAL) >= 0);

  // Create a dataset, then exercise dataset_get (space/type queries).
  hsize_t dims[1] = {64};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);

  std::vector<int> wbuf(64);
  for (size_t i = 0; i < wbuf.size(); ++i) wbuf[i] = static_cast<int>(i);
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   wbuf.data()) >= 0);

  // dataset_get: H5Dget_space / H5Dget_type route through the connector.
  hid_t dspace = H5Dget_space(dset);
  REQUIRE(dspace >= 0);
  hid_t dtype = H5Dget_type(dset);
  REQUIRE(dtype >= 0);
  H5Sclose(dspace);
  H5Tclose(dtype);

  // Partial (hyperslab) write + read: exercises the non-cacheable passthrough
  // branch in iowarp_dataset_write / iowarp_dataset_read.
  hsize_t mem_dims[1] = {8};
  hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);
  hid_t file_space = H5Dget_space(dset);
  hsize_t start[1] = {4}, count[1] = {8};
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count,
                      nullptr);
  std::vector<int> part(8, 99);
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, mem_space, file_space, H5P_DEFAULT,
                   part.data()) >= 0);
  std::vector<int> partr(8, 0);
  REQUIRE(H5Dread(dset, H5T_NATIVE_INT, mem_space, file_space, H5P_DEFAULT,
                  partr.data()) >= 0);
  REQUIRE(partr[0] == 99);
  H5Sclose(mem_space);
  H5Sclose(file_space);

  // Attribute open + attr_get passthrough.
  hid_t ascalar = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(dset, "meta", H5T_NATIVE_INT, ascalar,
                          H5P_DEFAULT, H5P_DEFAULT);
  int av = 7;
  H5Awrite(attr, H5T_NATIVE_INT, &av);
  H5Aclose(attr);
  hid_t attr2 = H5Aopen(dset, "meta", H5P_DEFAULT);
  REQUIRE(attr2 >= 0);
  hid_t aspace = H5Aget_space(attr2);  // attr_get passthrough
  REQUIRE(aspace >= 0);
  H5Sclose(aspace);
  H5Aclose(attr2);
  H5Sclose(ascalar);

  // Committed (named) datatype: datatype_commit / open / get / close.
  hid_t named = H5Tcopy(H5T_NATIVE_INT);
  REQUIRE(H5Tcommit2(file, "named_type", named, H5P_DEFAULT, H5P_DEFAULT,
                     H5P_DEFAULT) >= 0);
  hid_t opened_type = H5Topen2(file, "named_type", H5P_DEFAULT);
  REQUIRE(opened_type >= 0);
  REQUIRE(H5Tget_size(opened_type) == sizeof(int));
  H5Tclose(opened_type);
  H5Tclose(named);

  // Link create/specific + object open/get passthroughs.
  REQUIRE(H5Lcreate_soft("data", file, "softlink", H5P_DEFAULT,
                         H5P_DEFAULT) >= 0);
  htri_t exists = H5Lexists(file, "softlink", H5P_DEFAULT);
  REQUIRE(exists > 0);

  hid_t obj = H5Oopen(file, "data", H5P_DEFAULT);
  REQUIRE(obj >= 0);
  H5O_info2_t oinfo;
  // object_get passthrough (signature differs across HDF5 minor versions; the
  // call still routes through the connector regardless of return value).
  H5Oget_info3(obj, &oinfo, H5O_INFO_BASIC);
  H5Oclose(obj);

  REQUIRE(H5Dclose(dset) >= 0);
  REQUIRE(H5Sclose(space) >= 0);
  REQUIRE(H5Fclose(file) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
}

SIMPLE_TEST_MAIN()
