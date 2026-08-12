/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * End-to-end I/O tests for the Clio HDF5 VOL connector.
 *
 * Unlike test_hdf5_vol_adapter.cc (which only touches the plugin registration
 * and property-list surface), this suite drives *real* HDF5 file, dataset,
 * group, and attribute operations through the connector against an in-process
 * Chimaera/CTE runtime. That exercises the data-path callbacks the base suite
 * cannot reach: clio_file_create/open/close, clio_dataset_create/open/
 * write/read/close (the chunked AsyncPutBlob / read-through-cache paths),
 * group create/open/close, attr create/write/read/close, and the wrap/unwrap/
 * info callbacks invoked while HDF5 manages object handles.
 *
 * The connector lazily attaches to the running runtime via get_cte_client()
 * (CLIO_CTE_CLIENT_INIT on first use), so we only need to stand up the runtime
 * and register one bdev target before issuing HDF5 calls.
 */

#include "simple_test.h"

#include "clio_vol.h"

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
  _putenv_s("CLIO_VOL_CHUNK_SIZE", "4096");
#else
  setenv("CLIO_VOL_CHUNK_SIZE", "4096", 1);
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

  vol_id = H5VL_clio_register();
  REQUIRE(vol_id >= 0);
  return vol_id;
}

/** Build a file-access property list bound to the Clio VOL connector. */
hid_t makeFapl(hid_t vol_id) {
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  REQUIRE(fapl >= 0);
  // Value-initialized, so every field starts at its documented default and
  // cache_enabled reads as CLIO_VOL_CACHE_UNSET. Leaving this struct
  // uninitialized once let stack garbage disable the CTE tier for the file
  // with nothing logged -- the tests still passed, they just tested nothing.
  clio_vol_info_t info = {};
  info.under_vol_id = H5VL_NATIVE;
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
  // branch in clio_dataset_write / clio_dataset_read.
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

namespace {

/** Size of a chunk blob as the CTE tier holds it, via the file's tag. */
clio::run::u64 cteBlobSize(const std::string &h5_path,
                           const std::string &blob_name) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto tag = cte_client->AsyncGetOrCreateTag(std::string("hdf5:") + h5_path);
  tag.Wait();
  REQUIRE(tag->GetReturnCode() == 0);
  auto sz = cte_client->AsyncGetBlobSize(tag->tag_id_, blob_name);
  sz.Wait();
  return sz->size_;
}

/** Drop one chunk blob out of the tier, standing in for a per-blob eviction. */
void cteDelBlob(const std::string &h5_path, const std::string &blob_name) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto tag = cte_client->AsyncGetOrCreateTag(std::string("hdf5:") + h5_path);
  tag.Wait();
  REQUIRE(tag->GetReturnCode() == 0);
  auto del = cte_client->AsyncDelBlob(tag->tag_id_, blob_name);
  del.Wait();
  REQUIRE(del->GetReturnCode() == 0);
}

}  // namespace

TEST_CASE("HDF5 VOL IO - Chunk blobs hold only their own bytes",
          "[hdf5_vol][io]") {
  // Regression: chunk blobs were written at their absolute image offset, so
  // chunk_i physically allocated (and zero-filled) i+1 chunks -- N(N+1)/2
  // chunks of tier for an N-chunk dataset, invisible to round-trip tests
  // because the read side used the same offsets. Each chunk blob must be
  // exactly its own bytes at blob offset 0.
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);

  const std::string path = "/tmp/clio_vol_io_chunklayout.h5";
  std::remove(path.c_str());

  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);
  hsize_t dims[1] = {kNumElems};  // 16 KiB = 4 chunks at the 4 KiB test chunk
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);
  std::vector<int> wbuf(kNumElems, 7);
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   wbuf.data()) >= 0);
  REQUIRE(H5Dclose(dset) >= 0);  // drains the async puts
  H5Sclose(space);
  REQUIRE(H5Fclose(file) >= 0);

  REQUIRE(cteBlobSize(path, "data/chunk_0") == 4096);
  REQUIRE(cteBlobSize(path, "data/chunk_1") == 4096);  // was 8192 (1 chunk hole)
  REQUIRE(cteBlobSize(path, "data/chunk_3") == 4096);  // was 16384

  REQUIRE(H5Pclose(fapl) >= 0);
}

TEST_CASE("HDF5 VOL IO - Whole rewrite invalidates when staging is skipped",
          "[hdf5_vol][io]") {
  // Regression: under an admission policy that does not stage on write (and
  // equally while back-pressure holds the gate shut), a whole rewrite updated
  // the native file but left the previously staged image in the tier, so the
  // next whole read served pre-rewrite bytes with a success status.
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);
  setenv("CLIO_VOL_ADMIT", "read-miss", 1);

  const std::string path = "/tmp/clio_vol_io_rewrite.h5";
  std::remove(path.c_str());

  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);
  constexpr size_t kN = 1024;  // 4 KiB: one chunk
  hsize_t dims[1] = {kN};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);

  std::vector<int> v1(kN, 1), v2(kN, 2), rbuf(kN, 0);
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   v1.data()) >= 0);
  // First read misses and stages v1 into the tier (read-miss admission).
  REQUIRE(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  rbuf.data()) >= 0);
  REQUIRE(rbuf[0] == 1);
  REQUIRE(cteBlobSize(path, "data/chunk_0") > 0);  // v1 image is cached

  // Whole rewrite: no staging under read-miss, so the v1 image must be
  // invalidated or the read below hits it.
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   v2.data()) >= 0);
  REQUIRE(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  rbuf.data()) >= 0);
  REQUIRE(rbuf[0] == 2);
  REQUIRE(rbuf[kN - 1] == 2);

  REQUIRE(H5Dclose(dset) >= 0);
  H5Sclose(space);
  REQUIRE(H5Fclose(file) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
  unsetenv("CLIO_VOL_ADMIT");
}

TEST_CASE("HDF5 VOL IO - Partial-write-first dataset stays cacheable",
          "[hdf5_vol][io]") {
  // Regression: invalidating a dataset that had nothing cached (DelBlob
  // returns "not found") was treated as an invalidation FAILURE, so any
  // dataset whose first write was partial was marked uncacheable for the rest
  // of the session and never staged again.
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);

  const std::string path = "/tmp/clio_vol_io_partialfirst.h5";
  std::remove(path.c_str());

  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);
  constexpr size_t kN = 1024;  // 4 KiB: one chunk
  hsize_t dims[1] = {kN};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);

  // Partial write first: nothing is cached yet, so the invalidation this
  // triggers must be a no-op, not a failure.
  hsize_t mem_dims[1] = {128};
  hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);
  hid_t file_space = H5Dget_space(dset);
  hsize_t start[1] = {0}, count[1] = {128};
  REQUIRE(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr,
                              count, nullptr) >= 0);
  std::vector<int> part(128, 5);
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, mem_space, file_space, H5P_DEFAULT,
                   part.data()) >= 0);
  H5Sclose(mem_space);
  H5Sclose(file_space);

  // A whole write afterwards must still stage into the tier.
  std::vector<int> whole(kN, 9);
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   whole.data()) >= 0);
  REQUIRE(H5Dclose(dset) >= 0);  // drains the async puts
  H5Sclose(space);
  REQUIRE(H5Fclose(file) >= 0);

  REQUIRE(cteBlobSize(path, "data/chunk_0") == kN * sizeof(int));

  // And the data itself round-trips.
  hid_t file2 = H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl);
  REQUIRE(file2 >= 0);
  hid_t dset2 = H5Dopen2(file2, "data", H5P_DEFAULT);
  REQUIRE(dset2 >= 0);
  std::vector<int> rbuf(kN, 0);
  REQUIRE(H5Dread(dset2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  rbuf.data()) >= 0);
  REQUIRE(rbuf[0] == 9);
  REQUIRE(rbuf[kN - 1] == 9);
  REQUIRE(H5Dclose(dset2) >= 0);
  REQUIRE(H5Fclose(file2) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
}

TEST_CASE("HDF5 VOL IO - A missing chunk is a miss, not garbage",
          "[hdf5_vol][io]") {
  // Regression: the reassembly loop ignored every GetBlob return code, so a
  // chunk that was gone from the tier (per-blob eviction) was memcpy'd from an
  // uninitialized shared-memory buffer and reported as a cache hit. Only
  // chunk_0 is hit-tested, so losing any later chunk was silent.
  hid_t vol_id = setupVolEnvironment();
  hid_t fapl = makeFapl(vol_id);

  const std::string path = "/tmp/clio_vol_io_evicted.h5";
  std::remove(path.c_str());

  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  REQUIRE(file >= 0);
  hsize_t dims[1] = {kNumElems};  // 4 chunks at the 4 KiB test chunk size
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dset >= 0);
  std::vector<int> wbuf(kNumElems);
  for (size_t i = 0; i < kNumElems; ++i) wbuf[i] = static_cast<int>(i * 7 + 3);
  REQUIRE(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   wbuf.data()) >= 0);
  REQUIRE(H5Dclose(dset) >= 0);  // drains the async puts
  H5Sclose(space);
  REQUIRE(H5Fclose(file) >= 0);
  REQUIRE(cteBlobSize(path, "data/chunk_2") > 0);

  // Evict a middle chunk. chunk_0 survives, so the read still hit-tests as a
  // cache hit and must discover the hole during reassembly.
  cteDelBlob(path, "data/chunk_2");

  hid_t file2 = H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl);
  REQUIRE(file2 >= 0);
  hid_t dset2 = H5Dopen2(file2, "data", H5P_DEFAULT);
  REQUIRE(dset2 >= 0);
  std::vector<int> rbuf(kNumElems, -1);
  REQUIRE(H5Dread(dset2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  rbuf.data()) >= 0);

  // Every element comes from the authoritative native file.
  bool match = true;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (rbuf[i] != wbuf[i]) { match = false; break; }
  }
  REQUIRE(match);

  // ...and the connector NOTICED. A failed reassembly invalidates the image,
  // so the hit-test key is gone. Without that, the assertion above could pass
  // on whatever the recycled buffer happened to hold; this is the part that
  // actually distinguishes "detected the hole" from "got lucky".
  REQUIRE(cteBlobSize(path, "data/chunk_0") == 0);

  REQUIRE(H5Dclose(dset2) >= 0);
  REQUIRE(H5Fclose(file2) >= 0);
  REQUIRE(H5Pclose(fapl) >= 0);
}

SIMPLE_TEST_MAIN()
