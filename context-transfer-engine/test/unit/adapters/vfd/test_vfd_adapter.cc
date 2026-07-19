/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * HDF5 VFD test suite for the "authoritative native file" write-through layer.
 *
 * The driver writes every byte through to a real on-disk native HDF5 file at
 * the stripped path, so standard tools read it live and it is byte-identical to
 * what sec2 would produce. Because the driver is byte-agnostic, the meaningful
 * coverage is exercising the *variety of byte-write patterns* HDF5 generates:
 *   1. Rich round-trip through the VFD -- multiple dtypes, a multi-page (>4 MiB)
 *      dataset, 2-D, chunked, chunked+shuffle+gzip, a subgroup, and an
 *      attribute (structural + scattered small metadata writes).
 *   2. Reopen + append a dataset (eof/reopen from fstat).
 *   3. Differential vs sec2 (native oracle): the same rich content written
 *      through sec2 and through the VFD reads back byte-identical, and the VFD's
 *      native file is reopened/read WITHOUT the VFD.
 *   4. Native tool matrix on the VFD's file, NO VFD loaded:
 *      h5dump/h5ls/h5repack/h5diff. Guarded on tool availability, LOUD if it
 *      must skip -- never a silent pass.
 *   5. Partial I/O: hyperslab overwrite-in-place (an interior, non-whole write)
 *      then read the merged result back.
 *   6. Two VFD files open simultaneously -- independent, no cross-talk.
 *
 * Plain main (no Catch2). Returns non-zero on the first failure.
 */
#include <hdf5.h>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_cte/core/core_client.h"
#include "adapter/cfs/cfs_io.h"
#include "adapter/vfd/H5FDclio.h"

namespace {
const char *kBackend = "/tmp/clio_cte_vfd_test.dat";
const char *kClioFile = "clio::/tmp/clio_cte_vfd_suite.h5";
const char *kNativeFile = "/tmp/clio_cte_vfd_suite.h5";

constexpr hsize_t kBig = 512 * 1024;  // multi-page (>4 MiB of doubles)
constexpr hsize_t kSmall = 1000;

#define CHECK(cond, msg)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      std::fprintf(stderr, "[vfd-suite] FAIL: %s\n", (msg)); \
      return 1;                                              \
    }                                                        \
  } while (0)

bool InitRuntime() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) return false;
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return false;
  auto *cte = CLIO_CTE_CLIENT;
  clio::run::PoolId bdev_pool_id(953, 0);
  clio::run::bdev::Client bdev(bdev_pool_id);
  auto ct = bdev.AsyncCreate(clio::run::PoolQuery::Dynamic(), kBackend,
                             bdev_pool_id, clio::run::bdev::BdevType::kFile);
  ct.Wait();
  auto rt = cte->AsyncRegisterTarget(kBackend, clio::run::bdev::BdevType::kFile,
                                     64ULL * 1024 * 1024,
                                     clio::run::PoolQuery::Local(), bdev_pool_id);
  rt.Wait();
  return rt->GetReturnCode() == 0;
}

hid_t ClioFapl(hid_t driver) {
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl < 0 || H5Pset_driver(fapl, driver, nullptr) < 0)
    return H5I_INVALID_HID;
  return fapl;
}

bool DeflateAvail() { return H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0; }

// ---- deterministic fill patterns (distinct per type) ----------------------
std::vector<int32_t> MakeI32(hsize_t n) {
  std::vector<int32_t> v(n);
  for (hsize_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(i * 7 - 3);
  return v;
}
std::vector<int64_t> MakeI64(hsize_t n) {
  std::vector<int64_t> v(n);
  for (hsize_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(i) * 1000003LL - 42;
  return v;
}
std::vector<float> MakeF32(hsize_t n) {
  std::vector<float> v(n);
  for (hsize_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) * 0.25f - 1.5f;
  return v;
}
std::vector<double> MakeF64(hsize_t n) {
  std::vector<double> v(n);
  for (hsize_t i = 0; i < n; ++i) v[i] = static_cast<double>(i) * 1.5 - 7.0;
  return v;
}

// ---- generic writers/readers (reader is rank-agnostic: H5S_ALL is flat) ----
template <typename T>
bool WriteDset(hid_t loc, const char *name, hid_t type, const std::vector<T> &v) {
  hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset =
      H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  bool ok = dset >= 0 &&
            H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) >= 0;
  if (dset >= 0) H5Dclose(dset);
  H5Sclose(space);
  return ok;
}

template <typename T>
bool WriteChunked(hid_t loc, const char *name, hid_t type,
                  const std::vector<T> &v, bool filters) {
  hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
  hsize_t chunk[1] = {std::min<hsize_t>(v.size(), 256)};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl, 1, chunk);
  if (filters) {
    H5Pset_shuffle(dcpl);
    H5Pset_deflate(dcpl, 6);
  }
  hid_t dset =
      H5Dcreate2(loc, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
  bool ok = dset >= 0 &&
            H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) >= 0;
  if (dset >= 0) H5Dclose(dset);
  H5Pclose(dcpl);
  H5Sclose(space);
  return ok;
}

bool Write2D(hid_t loc, const char *name, const std::vector<double> &flat,
             hsize_t rows, hsize_t cols) {
  hsize_t dims[2] = {rows, cols};
  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t dset = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  bool ok = dset >= 0 && H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, flat.data()) >= 0;
  if (dset >= 0) H5Dclose(dset);
  H5Sclose(space);
  return ok;
}

template <typename T>
bool ReadDsetEq(hid_t loc, const char *name, hid_t type,
                const std::vector<T> &expected) {
  hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
  if (dset < 0) return false;
  std::vector<T> got(expected.size());
  bool ok = H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, got.data()) >= 0;
  H5Dclose(dset);
  return ok &&
         std::memcmp(got.data(), expected.data(), got.size() * sizeof(T)) == 0;
}

bool WriteAttr(hid_t file) {
  hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);
  hsize_t d[1] = {4};
  hid_t sp = H5Screate_simple(1, d, nullptr);
  hid_t a = H5Acreate2(root, "meta", H5T_NATIVE_INT32, sp, H5P_DEFAULT,
                       H5P_DEFAULT);
  int vals[4] = {10, 20, 30, 40};
  bool ok = a >= 0 && H5Awrite(a, H5T_NATIVE_INT32, vals) >= 0;
  if (a >= 0) H5Aclose(a);
  H5Sclose(sp);
  H5Gclose(root);
  return ok;
}
bool VerifyAttr(hid_t file) {
  hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);
  hid_t a = H5Aopen(root, "meta", H5P_DEFAULT);
  int got[4] = {0, 0, 0, 0};
  const int exp[4] = {10, 20, 30, 40};
  bool ok = a >= 0 && H5Aread(a, H5T_NATIVE_INT32, got) >= 0 &&
            std::memcmp(got, exp, sizeof(exp)) == 0;
  if (a >= 0) H5Aclose(a);
  H5Gclose(root);
  return ok;
}

// A structurally rich content set: flat dtypes, a multi-page dataset, 2-D,
// chunked, chunked+filters (if deflate is available), a subgroup, an attribute.
bool WriteRich(hid_t file) {
  if (!WriteDset(file, "i32", H5T_NATIVE_INT32, MakeI32(kSmall))) return false;
  if (!WriteDset(file, "i64", H5T_NATIVE_INT64, MakeI64(kSmall))) return false;
  if (!WriteDset(file, "f32", H5T_NATIVE_FLOAT, MakeF32(kSmall))) return false;
  if (!WriteDset(file, "f64_big", H5T_NATIVE_DOUBLE, MakeF64(kBig))) return false;
  if (!Write2D(file, "d2d", MakeF64(64 * 32), 64, 32)) return false;
  if (!WriteChunked(file, "chunked", H5T_NATIVE_INT32, MakeI32(kSmall), false))
    return false;
  if (DeflateAvail() &&
      !WriteChunked(file, "compressed", H5T_NATIVE_INT32, MakeI32(kSmall), true))
    return false;
  hid_t g = H5Gcreate2(file, "/grp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (g < 0) return false;
  bool gok = WriteDset(g, "inner", H5T_NATIVE_INT64, MakeI64(kSmall));
  H5Gclose(g);
  if (!gok) return false;
  return WriteAttr(file);
}

bool VerifyRich(hid_t file) {
  if (!ReadDsetEq(file, "i32", H5T_NATIVE_INT32, MakeI32(kSmall))) return false;
  if (!ReadDsetEq(file, "i64", H5T_NATIVE_INT64, MakeI64(kSmall))) return false;
  if (!ReadDsetEq(file, "f32", H5T_NATIVE_FLOAT, MakeF32(kSmall))) return false;
  if (!ReadDsetEq(file, "f64_big", H5T_NATIVE_DOUBLE, MakeF64(kBig)))
    return false;
  if (!ReadDsetEq(file, "d2d", H5T_NATIVE_DOUBLE, MakeF64(64 * 32))) return false;
  if (!ReadDsetEq(file, "chunked", H5T_NATIVE_INT32, MakeI32(kSmall)))
    return false;
  if (DeflateAvail() &&
      !ReadDsetEq(file, "compressed", H5T_NATIVE_INT32, MakeI32(kSmall)))
    return false;
  if (!ReadDsetEq(file, "/grp/inner", H5T_NATIVE_INT64, MakeI64(kSmall)))
    return false;
  return VerifyAttr(file);
}

bool HasTool(const char *tool) {
  std::string cmd = "command -v ";
  cmd += tool;
  cmd += " >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}
int RunCmd(const std::string &cmd) {
  return std::system((cmd + " >/dev/null 2>&1").c_str());
}

// H5Ewalk callback: set *data if any error on the stack is the driver's own
// push, identified by the message text the driver emits.
herr_t FindClioErr(unsigned n, const H5E_error2_t *err, void *data) {
  (void)n;
  if (err && err->desc && std::strstr(err->desc, "authoritative native file")) {
    *static_cast<bool *>(data) = true;
  }
  return 0;
}
}  // namespace

int main() {
  if (!InitRuntime()) {
    std::fprintf(stderr, "[vfd-suite] FAIL: runtime/CTE init\n");
    return 1;
  }

  hid_t driver = H5FD_clio_init();
  CHECK(driver >= 0, "H5FD_clio_init");
  hid_t fapl = ClioFapl(driver);
  CHECK(fapl >= 0, "build clio FAPL");
  std::remove(kNativeFile);

  // === 1. Rich round-trip through the VFD =================================
  {
    hid_t f = H5Fcreate(kClioFile, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(f >= 0, "1: H5Fcreate via VFD");
    CHECK(WriteRich(f), "1: write rich content");
    CHECK(H5Fclose(f) >= 0, "1: H5Fclose (write)");
    hid_t f2 = H5Fopen(kClioFile, H5F_ACC_RDONLY, fapl);
    CHECK(f2 >= 0, "1: H5Fopen via VFD");
    CHECK(VerifyRich(f2), "1: rich content reads back byte-clean");
    CHECK(H5Fclose(f2) >= 0, "1: H5Fclose (read)");
    std::printf("[vfd-suite] ok 1: rich round-trip (dtypes/2D/chunked/filters/group/attr)\n");
  }

  // === 2. Reopen + append =================================================
  {
    hid_t f = H5Fopen(kClioFile, H5F_ACC_RDWR, fapl);
    CHECK(f >= 0, "2: reopen RDWR");
    CHECK(WriteDset(f, "appended", H5T_NATIVE_INT32, MakeI32(kSmall)),
          "2: append a dataset");
    CHECK(H5Fclose(f) >= 0, "2: H5Fclose (append)");
    hid_t f2 = H5Fopen(kClioFile, H5F_ACC_RDONLY, fapl);
    CHECK(f2 >= 0, "2: reopen RDONLY");
    CHECK(VerifyRich(f2), "2: original content intact after append");
    CHECK(ReadDsetEq(f2, "appended", H5T_NATIVE_INT32, MakeI32(kSmall)),
          "2: appended dataset readable");
    CHECK(H5Fclose(f2) >= 0, "2: H5Fclose");
    std::printf("[vfd-suite] ok 2: reopen + append\n");
  }

  // === 3. Differential vs sec2 (native oracle) ============================
  {
    const char *kSec2 = "/tmp/clio_cte_vfd_sec2.h5";
    const char *kClioDiff = "clio::/tmp/clio_cte_vfd_diff.h5";
    const char *kNativeDiff = "/tmp/clio_cte_vfd_diff.h5";
    std::remove(kSec2);
    std::remove(kNativeDiff);
    hid_t s = H5Fcreate(kSec2, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(s >= 0 && WriteRich(s) && H5Fclose(s) >= 0, "3: write rich via sec2");
    hid_t v = H5Fcreate(kClioDiff, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(v >= 0 && WriteRich(v) && H5Fclose(v) >= 0, "3: write rich via VFD");
    // Reopen BOTH with sec2 (no VFD) and verify identical, correct content.
    hid_t rs = H5Fopen(kSec2, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t rn = H5Fopen(kNativeDiff, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(rs >= 0, "3: reopen sec2 file");
    CHECK(rn >= 0, "3: reopen VFD's native file WITHOUT the VFD");
    CHECK(VerifyRich(rs), "3: sec2 file content correct");
    CHECK(VerifyRich(rn), "3: VFD's native file content correct (read w/o VFD)");
    CHECK(H5Fclose(rs) >= 0 && H5Fclose(rn) >= 0, "3: close diff files");
    std::printf("[vfd-suite] ok 3: differential vs sec2 (rich content)\n");
  }

  // === 5. Partial I/O: hyperslab overwrite-in-place =======================
  // (Section 4 -- the external tool matrix -- runs last, after H5close.)
  {
    const char *kClioPart = "clio::/tmp/clio_cte_vfd_partial.h5";
    const hsize_t N = kSmall;
    std::vector<int32_t> base(N);
    for (hsize_t i = 0; i < N; ++i) base[i] = static_cast<int32_t>(i);

    hid_t f = H5Fcreate(kClioPart, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(f >= 0, "5: H5Fcreate");
    hsize_t dims[1] = {N};
    hid_t fs = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(f, "partial", H5T_NATIVE_INT32, fs, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
    CHECK(dset >= 0, "5: H5Dcreate2");
    CHECK(H5Dwrite(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   base.data()) >= 0,
          "5: write baseline");
    // Overwrite the middle third with a distinct pattern via a file hyperslab.
    hsize_t start[1] = {N / 3};
    hsize_t count[1] = {N / 3};
    CHECK(H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count,
                              nullptr) >= 0,
          "5: select hyperslab");
    hid_t ms = H5Screate_simple(1, count, nullptr);
    std::vector<int32_t> patch(count[0]);
    for (hsize_t j = 0; j < count[0]; ++j)
      patch[j] = static_cast<int32_t>(-1 - static_cast<int>(j));
    CHECK(H5Dwrite(dset, H5T_NATIVE_INT32, ms, fs, H5P_DEFAULT, patch.data()) >= 0,
          "5: hyperslab overwrite-in-place");
    H5Sclose(ms);
    H5Sclose(fs);
    H5Dclose(dset);
    CHECK(H5Fclose(f) >= 0, "5: H5Fclose");

    // Expected merged result.
    std::vector<int32_t> expect = base;
    for (hsize_t j = 0; j < count[0]; ++j)
      expect[start[0] + j] = static_cast<int32_t>(-1 - static_cast<int>(j));
    hid_t f2 = H5Fopen(kClioPart, H5F_ACC_RDONLY, fapl);
    CHECK(f2 >= 0, "5: reopen");
    CHECK(ReadDsetEq(f2, "partial", H5T_NATIVE_INT32, expect),
          "5: merged (baseline + overwritten hyperslab) reads back correct");
    CHECK(H5Fclose(f2) >= 0, "5: H5Fclose (read)");
    std::printf("[vfd-suite] ok 5: partial hyperslab overwrite-in-place\n");
  }

  // === 6. Two VFD files open simultaneously ===============================
  {
    const char *kA = "clio::/tmp/clio_cte_vfd_a.h5";
    const char *kB = "clio::/tmp/clio_cte_vfd_b.h5";
    hid_t fa = H5Fcreate(kA, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t fb = H5Fcreate(kB, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(fa >= 0 && fb >= 0, "6: create two files at once");
    CHECK(WriteDset(fa, "a", H5T_NATIVE_INT32, MakeI32(kSmall)), "6: write A");
    CHECK(WriteDset(fb, "b", H5T_NATIVE_INT64, MakeI64(kSmall)), "6: write B");
    CHECK(H5Fclose(fa) >= 0 && H5Fclose(fb) >= 0, "6: close both");
    hid_t ra = H5Fopen(kA, H5F_ACC_RDONLY, fapl);
    hid_t rb = H5Fopen(kB, H5F_ACC_RDONLY, fapl);
    CHECK(ra >= 0 && rb >= 0, "6: reopen both");
    CHECK(ReadDsetEq(ra, "a", H5T_NATIVE_INT32, MakeI32(kSmall)), "6: A intact");
    CHECK(ReadDsetEq(rb, "b", H5T_NATIVE_INT64, MakeI64(kSmall)), "6: B intact");
    // No cross-talk: A must not contain B's dataset and vice versa.
    CHECK(H5Lexists(ra, "b", H5P_DEFAULT) <= 0, "6: A has no cross-talk from B");
    CHECK(H5Lexists(rb, "a", H5P_DEFAULT) <= 0, "6: B has no cross-talk from A");
    CHECK(H5Fclose(ra) >= 0 && H5Fclose(rb) >= 0, "6: close both (read)");
    std::printf("[vfd-suite] ok 6: two files open simultaneously (no cross-talk)\n");
  }

  // === 7. flush callback + get_handle ====================================
  // After H5Fflush the native file is a valid HDF5 image readable independently
  // of the VFD (the flush callback ran and the write-through reached the fd).
  // NOTE: this observes the *functional* barrier, not the fsync-to-platter --
  // an in-process read hits the same page cache, so durability itself is not
  // unit-observable. get_handle must hand back the authoritative POSIX fd.
  {
    const char *kClioFl = "clio::/tmp/clio_cte_vfd_flush.h5";
    const char *kNativeFl = "/tmp/clio_cte_vfd_flush.h5";
    std::remove(kNativeFl);
    hid_t f = H5Fcreate(kClioFl, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(f >= 0, "7: create");
    CHECK(WriteDset(f, "d", H5T_NATIVE_INT32, MakeI32(kSmall)), "7: write");
    CHECK(H5Fflush(f, H5F_SCOPE_GLOBAL) >= 0, "7: H5Fflush (flush callback)");
    // The HDF5 superblock signature must be on the fd now, read via plain POSIX.
    int rfd = ::open(kNativeFl, O_RDONLY);
    CHECK(rfd >= 0, "7: POSIX-open native file mid-session");
    unsigned char sig[8] = {0};
    ssize_t n = ::pread(rfd, sig, sizeof(sig), 0);
    ::close(rfd);
    const unsigned char kHdf5Sig[8] = {0x89, 'H', 'D', 'F',  '\r',
                                       '\n', 0x1a, '\n'};
    CHECK(n == 8 && std::memcmp(sig, kHdf5Sig, 8) == 0,
          "7: valid HDF5 superblock on disk after H5Fflush");
    // get_handle returns the authoritative POSIX fd -- verify it is THIS file's
    // fd (same device+inode as the native path), not merely some valid fd.
    void *vh = nullptr;
    CHECK(H5Fget_vfd_handle(f, H5P_DEFAULT, &vh) >= 0, "7: H5Fget_vfd_handle");
    CHECK(vh != nullptr, "7: handle non-null");
    int gfd = *static_cast<int *>(vh);
    struct stat gst, nst;
    CHECK(gfd >= 0 && ::fstat(gfd, &gst) == 0, "7: get_handle fd is valid");
    CHECK(::stat(kNativeFl, &nst) == 0 && gst.st_dev == nst.st_dev &&
              gst.st_ino == nst.st_ino,
          "7: get_handle fd points at the authoritative native file");
    CHECK(H5Fclose(f) >= 0, "7: close");
    std::printf("[vfd-suite] ok 7: flush callback + get_handle\n");
  }

  // NOTE: the truncate callback is exercised on every close (HDF5 sizes the
  // file to EOA there) and is covered transitively -- a broken truncate would
  // corrupt the image, which the round-trip / h5diff / tool-matrix sections
  // catch. There is no clean *isolated* truncate assertion at this layer:
  // HDF5's EOA is not exposed independently of the driver's own get_eof, and
  // comparing on-disk size to sec2 is confounded by the metadata-aggregation
  // feature flags (a separate change) that alter HDF5's allocation.

  // === 8. lock excludes a concurrent opener; unlock releases =============
  // Force HDF5 file locking on, so create takes an exclusive flock via the lock
  // callback. flock conflicts across independent open descriptions (even same
  // process), so an independent flock is denied while held and granted after
  // the VFD unlocks on close.
  {
    const char *kClioLk = "clio::/tmp/clio_cte_vfd_lock.h5";
    const char *kNativeLk = "/tmp/clio_cte_vfd_lock.h5";
    std::remove(kNativeLk);
    hid_t fapl_lk = H5Pcopy(fapl);
    CHECK(fapl_lk >= 0, "8: H5Pcopy fapl");
    CHECK(H5Pset_file_locking(fapl_lk, /*use*/ true, /*ignore_disabled*/ false) >= 0,
          "8: force file locking on");
    hid_t f = H5Fcreate(kClioLk, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_lk);
    CHECK(f >= 0, "8: create (VFD takes exclusive lock)");
    int p = ::open(kNativeLk, O_RDWR);
    CHECK(p >= 0, "8: POSIX-open native file");
    errno = 0;
    int held = ::flock(p, LOCK_EX | LOCK_NB);
    CHECK(held < 0 && (errno == EWOULDBLOCK || errno == EAGAIN),
          "8: independent flock denied while the VFD holds the lock");
    CHECK(H5Fclose(f) >= 0, "8: close (VFD unlocks)");
    int freed = ::flock(p, LOCK_EX | LOCK_NB);
    CHECK(freed == 0, "8: independent flock granted after the VFD unlocks");
    ::flock(p, LOCK_UN);
    ::close(p);
    H5Pclose(fapl_lk);
    std::printf("[vfd-suite] ok 8: lock excludes a concurrent opener; unlock releases\n");
  }

  // === 9. fail-closed error reporting ====================================
  // A failed operation must fail closed AND leave a diagnosable driver error on
  // the HDF5 error stack, not fail silently. Open a non-existent file; suppress
  // the auto-printer so the expected stack doesn't clutter output, and walk the
  // stack to confirm the driver's own error (not just HDF5's) was recorded.
  {
    H5E_auto2_t old_func = nullptr;
    void *old_data = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    H5Eclear2(H5E_DEFAULT);
    hid_t missing = H5Fopen("clio::/tmp/clio_cte_vfd_absent_xyz.h5",
                            H5F_ACC_RDONLY, fapl);
    bool found_clio_err = false;
    H5Ewalk2(H5E_DEFAULT, H5E_WALK_UPWARD, FindClioErr, &found_clio_err);
    H5Eset_auto2(H5E_DEFAULT, old_func, old_data);
    CHECK(missing < 0, "9: H5Fopen of a missing file must fail closed");
    CHECK(found_clio_err,
          "9: the driver pushed a diagnosable error onto the HDF5 stack");
    std::printf("[vfd-suite] ok 9: fail-closed error reporting\n");
  }

  // === 10. feature-flag effect: on-disk size matches sec2 =================
  // With the metadata-aggregation feature flags advertised by query(), HDF5
  // lays the file out the same way it does for sec2, so identical content yields
  // an identical on-disk byte size. Before those flags the VFD produced a
  // smaller, differently-aggregated file -- so this both proves the flags took
  // effect and pins truncate/close sizing against the sec2 oracle. (Byte content
  // isn't compared: HDF5 stamps object modification times, which differ by
  // instant but not in size.)
  {
    const char *kSec2 = "/tmp/clio_cte_vfd_flagsec2.h5";
    const char *kClioFf = "clio::/tmp/clio_cte_vfd_flagvfd.h5";
    const char *kNativeFf = "/tmp/clio_cte_vfd_flagvfd.h5";
    std::remove(kSec2);
    std::remove(kNativeFf);
    hid_t s = H5Fcreate(kSec2, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(s >= 0 && WriteRich(s) && H5Fclose(s) >= 0, "10: sec2 write+close");
    hid_t v = H5Fcreate(kClioFf, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(v >= 0 && WriteRich(v) && H5Fclose(v) >= 0, "10: VFD write+close");
    struct stat ss, vs;
    CHECK(::stat(kSec2, &ss) == 0 && ::stat(kNativeFf, &vs) == 0,
          "10: stat both files");
    CHECK(ss.st_size == vs.st_size,
          "10: VFD on-disk size == sec2 (metadata-aggregation flags in effect)");
    std::printf("[vfd-suite] ok 10: feature-flag effect (size parity with sec2)\n");
  }

  // === 11. SWMR I/O (validates the SUPPORTS_SWMR_IO feature flag) =========
  // SWMR needs the latest file format + an extendible dataset. HDF5 accepts an
  // SWMR-write transition only because the driver advertises SUPPORTS_SWMR_IO;
  // then write-through + flush must make appended data visible to a *concurrent*
  // SWMR reader (a separate handle on the same native file) after H5Drefresh.
  // File locking is disabled on the FAPL (standard for SWMR): the writer's
  // advisory flock would otherwise block the in-process reader -- same as sec2.
  {
    const char *kClioSwmr = "clio::/tmp/clio_cte_vfd_swmr.h5";
    const char *kNativeSwmr = "/tmp/clio_cte_vfd_swmr.h5";
    std::remove(kNativeSwmr);
    hid_t fapl_sw = H5Pcopy(fapl);
    CHECK(fapl_sw >= 0, "11: H5Pcopy");
    CHECK(H5Pset_libver_bounds(fapl_sw, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) >= 0,
          "11: set latest libver (required for SWMR)");
    CHECK(H5Pset_file_locking(fapl_sw, /*use*/ false, /*ignore*/ true) >= 0,
          "11: disable file locking for SWMR");

    // Writer: create + unlimited chunked dataset seeded with 3 rows.
    hid_t f = H5Fcreate(kClioSwmr, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_sw);
    CHECK(f >= 0, "11: H5Fcreate (latest format)");
    hsize_t dims[1] = {3}, maxd[1] = {H5S_UNLIMITED}, chunk[1] = {4};
    hid_t sp = H5Screate_simple(1, dims, maxd);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk);
    hid_t d = H5Dcreate2(f, "swmr", H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl,
                         H5P_DEFAULT);
    int seed[3] = {10, 11, 12};
    CHECK(d >= 0 && H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                             seed) >= 0,
          "11: seed write");
    H5Pclose(dcpl);
    H5Sclose(sp);

    // Transition to SWMR write -- accepted only because the VFD advertises SWMR.
    CHECK(H5Fstart_swmr_write(f) >= 0,
          "11: H5Fstart_swmr_write (VFD accepts SWMR I/O)");

    // Append 2 rows and flush (the SWMR barrier).
    hsize_t newsize[1] = {5};
    CHECK(H5Dset_extent(d, newsize) >= 0, "11: extend to 5 rows");
    hid_t fsp = H5Dget_space(d);
    hsize_t start[1] = {3}, cnt[1] = {2};
    H5Sselect_hyperslab(fsp, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
    hid_t msp = H5Screate_simple(1, cnt, nullptr);
    int more[2] = {13, 14};
    CHECK(H5Dwrite(d, H5T_NATIVE_INT32, msp, fsp, H5P_DEFAULT, more) >= 0,
          "11: append write");
    H5Sclose(msp);
    H5Sclose(fsp);
    CHECK(H5Dflush(d) >= 0, "11: H5Dflush (SWMR barrier)");

    // Concurrent SWMR reader (separate handle) must see all 5 rows after refresh.
    hid_t rf = H5Fopen(kClioSwmr, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, fapl_sw);
    CHECK(rf >= 0, "11: SWMR reader open");
    hid_t rd = H5Dopen2(rf, "swmr", H5P_DEFAULT);
    CHECK(rd >= 0 && H5Drefresh(rd) >= 0, "11: reader open + refresh");
    hid_t rsp = H5Dget_space(rd);
    hsize_t cur[1] = {0};
    H5Sget_simple_extent_dims(rsp, cur, nullptr);
    int got[5] = {0, 0, 0, 0, 0};
    const int exp[5] = {10, 11, 12, 13, 14};
    bool ok = (cur[0] == 5) &&
              H5Dread(rd, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      got) >= 0 &&
              std::memcmp(got, exp, sizeof(exp)) == 0;
    H5Sclose(rsp);
    H5Dclose(rd);
    H5Fclose(rf);
    CHECK(ok, "11: SWMR reader sees appended data via write-through + flush");

    H5Dclose(d);
    H5Fclose(f);
    H5Pclose(fapl_sw);
    std::printf("[vfd-suite] ok 11: SWMR I/O (write-through visible to reader)\n");
  }

  // === 12. driver FAPL: H5Pset_fapl_clio round-trip ======================
  // The supported way to select the driver and carry config. Set the driver via
  // the CLIO setter with the cache DISABLED (native-only path), confirm the FAPL
  // actually carries the CLIO driver + a driver-info block (fapl_size/copy round
  // the config), and that a full rich round-trip works through it.
  {
    hid_t fapl_nc = H5Pcreate(H5P_FILE_ACCESS);
    CHECK(fapl_nc >= 0, "12: H5Pcreate");
    CHECK(H5Pset_fapl_clio(fapl_nc, /*cache_enabled*/ 0) >= 0,
          "12: H5Pset_fapl_clio (cache off)");
    CHECK(H5Pget_driver(fapl_nc) == H5FD_clio_init(),
          "12: FAPL driver is the CLIO VFD");
    // The driver-info block must carry the exact config we set -- proving
    // fapl_copy round-trips the value, not merely that a block exists. The probe
    // layout matches H5FD_clio_fapl_t's first field; both values must survive.
    struct ClioFaplProbe {
      hbool_t cache_enabled;
    };
    const void *di0 = H5Pget_driver_info(fapl_nc);
    CHECK(di0 != nullptr &&
              static_cast<const ClioFaplProbe *>(di0)->cache_enabled == 0,
          "12: FAPL round-trips cache_enabled=false");
    hid_t fapl_c = H5Pcreate(H5P_FILE_ACCESS);
    CHECK(fapl_c >= 0 && H5Pset_fapl_clio(fapl_c, /*cache_enabled*/ 1) >= 0,
          "12: H5Pset_fapl_clio (cache on)");
    const void *di1 = H5Pget_driver_info(fapl_c);
    CHECK(di1 != nullptr &&
              static_cast<const ClioFaplProbe *>(di1)->cache_enabled == 1,
          "12: FAPL round-trips cache_enabled=true");
    H5Pclose(fapl_c);
    const char *kClioFapl = "clio::/tmp/clio_cte_vfd_fapl.h5";
    std::remove("/tmp/clio_cte_vfd_fapl.h5");
    hid_t f = H5Fcreate(kClioFapl, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_nc);
    CHECK(f >= 0 && WriteRich(f) && H5Fclose(f) >= 0,
          "12: write+close via setter");
    hid_t f2 = H5Fopen(kClioFapl, H5F_ACC_RDONLY, fapl_nc);
    CHECK(f2 >= 0 && VerifyRich(f2) && H5Fclose(f2) >= 0,
          "12: rich round-trip via H5Pset_fapl_clio (cache off)");
    H5Pclose(fapl_nc);

    // The setter rejects a plist that is not a file access property list
    // (suppress the auto-printer -- the rejection pushes an expected error).
    hid_t not_fapl = H5Pcreate(H5P_DATASET_CREATE);
    H5E_auto2_t of = nullptr;
    void *od = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &of, &od);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    herr_t bad = H5Pset_fapl_clio(not_fapl, 1);
    H5Eset_auto2(H5E_DEFAULT, of, od);
    CHECK(bad < 0, "12: H5Pset_fapl_clio rejects a non-FAPL plist");
    H5Pclose(not_fapl);
    std::printf("[vfd-suite] ok 12: driver FAPL (H5Pset_fapl_clio round-trip)\n");
  }

  // === 13. vectored I/O (read_vector / write_vector) =====================
  // Force HDF5 down the vector path: request selection I/O on the transfer
  // plist. With the driver's selection callbacks NULL but read_vector/
  // write_vector implemented, HDF5 translates selection I/O to vector I/O.
  // Confirm the vector callbacks actually ran (exported counters advanced) AND
  // the data round-trips byte-clean.
  {
    extern unsigned long H5FDclio_read_vector_calls_g;
    extern unsigned long H5FDclio_write_vector_calls_g;
    const char *kClioVec = "clio::/tmp/clio_cte_vfd_vec.h5";
    std::remove("/tmp/clio_cte_vfd_vec.h5");
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    CHECK(dxpl >= 0 && H5Pset_selection_io(dxpl, H5D_SELECTION_IO_MODE_ON) >= 0,
          "13: request selection I/O on the transfer plist");

    const unsigned long w0 = H5FDclio_write_vector_calls_g;
    const unsigned long r0 = H5FDclio_read_vector_calls_g;
    std::vector<int32_t> w = MakeI32(kSmall);

    hid_t f = H5Fcreate(kClioVec, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(f >= 0, "13: create");
    hsize_t dims[1] = {kSmall};
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t d = H5Dcreate2(f, "vec", H5T_NATIVE_INT32, sp, H5P_DEFAULT, H5P_DEFAULT,
                         H5P_DEFAULT);
    CHECK(d >= 0 && H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, dxpl,
                             w.data()) >= 0,
          "13: H5Dwrite (selection/vector I/O)");
    H5Dclose(d);
    H5Sclose(sp);
    CHECK(H5Fclose(f) >= 0, "13: close");

    std::vector<int32_t> got(kSmall, 0);
    hid_t f2 = H5Fopen(kClioVec, H5F_ACC_RDONLY, fapl);
    hid_t d2 = H5Dopen2(f2, "vec", H5P_DEFAULT);
    CHECK(d2 >= 0 && H5Dread(d2, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, dxpl,
                             got.data()) >= 0,
          "13: H5Dread (selection/vector I/O)");
    H5Dclose(d2);
    H5Fclose(f2);
    H5Pclose(dxpl);

    CHECK(std::memcmp(got.data(), w.data(), kSmall * sizeof(int32_t)) == 0,
          "13: vectored round-trip byte-clean");
    CHECK(H5FDclio_write_vector_calls_g > w0,
          "13: write_vector was actually exercised");
    CHECK(H5FDclio_read_vector_calls_g > r0,
          "13: read_vector was actually exercised");
    std::printf("[vfd-suite] ok 13: vectored I/O (read_vector/write_vector exercised)\n");
  }

  // === 14. del callback: H5Fdelete removes BOTH stores ====================
  // A correct delete must leave NEITHER store orphaned: the authoritative native
  // file AND the CTE cache tag. Create+write a file (populating both), confirm
  // both exist, H5Fdelete through the VFD, then confirm both are gone (verified
  // on the HDF5 side via stat/reopen and on the CLIO side via the CFS tag) so a
  // deleted file leaves nothing behind in either the filesystem or CTE.
  {
    const char *kClioDel = "clio::/tmp/clio_cte_vfd_del.h5";
    const char *kNativeDel = "/tmp/clio_cte_vfd_del.h5";
    std::remove(kNativeDel);
    hid_t f = H5Fcreate(kClioDel, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(f >= 0, "14: create");
    CHECK(WriteDset(f, "d", H5T_NATIVE_INT32, MakeI32(kSmall)), "14: write");
    CHECK(H5Fclose(f) >= 0, "14: close");

    // Precondition: BOTH stores now exist.
    struct stat nst;
    CHECK(::stat(kNativeDel, &nst) == 0, "14: native file exists before delete");
    struct stat cst;
    CHECK(CLIO_CTE_CFS->StatPath(kClioDel, &cst) == 0,
          "14: CTE cache tag exists before delete");

    // Delete through the VFD (drives H5FD__clio_del via H5Fdelete).
    CHECK(H5Fdelete(kClioDel, fapl) >= 0, "14: H5Fdelete succeeds");

    // Postcondition (HDF5 side): native file gone; it no longer opens.
    errno = 0;
    CHECK(::stat(kNativeDel, &nst) != 0 && errno == ENOENT,
          "14: native file removed after H5Fdelete");
    H5E_auto2_t of = nullptr;
    void *od = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &of, &od);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t reopened = H5Fopen(kClioDel, H5F_ACC_RDONLY, fapl);
    H5Eset_auto2(H5E_DEFAULT, of, od);
    CHECK(reopened < 0, "14: deleted file no longer opens");

    // Postcondition (CLIO side): the CTE cache tag is gone, not orphaned.
    CHECK(CLIO_CTE_CFS->StatPath(kClioDel, &cst) != 0,
          "14: CTE cache tag removed after H5Fdelete (not orphaned)");
    std::printf("[vfd-suite] ok 14: del removes both native file and CTE tag\n");
  }

  // === 15. no-pending-dirty-state barrier: H5Fflush finalizes the image ======
  // An independent sec2 reader (no VFD; the writer stays open and is never
  // closed) reads the dataset back from the native file only AFTER H5Fflush: the
  // read fails before the flush (HDF5 metadata still buffered) and succeeds
  // after, so it is the flush -- not write-through alone -- that leaves a
  // complete, consistent on-disk image. Consistency only, not fsync-to-platter:
  // an in-process read hits the page cache. Locking is off so the reader is not
  // blocked by the writer.
  {
    const char *kClioDur = "clio::/tmp/clio_cte_vfd_durable.h5";
    const char *kNativeDur = "/tmp/clio_cte_vfd_durable.h5";
    std::remove(kNativeDur);
    std::vector<int32_t> w = MakeI32(kSmall);

    hid_t wfapl = H5Pcopy(fapl);
    CHECK(wfapl >= 0 && H5Pset_file_locking(wfapl, false, true) >= 0,
          "15: writer FAPL (locking off)");
    hid_t rfapl = H5Pcreate(H5P_FILE_ACCESS);
    CHECK(rfapl >= 0 && H5Pset_file_locking(rfapl, false, true) >= 0,
          "15: reader FAPL (locking off)");

    hid_t f = H5Fcreate(kClioDur, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
    CHECK(f >= 0, "15: create");
    CHECK(WriteDset(f, "durable", H5T_NATIVE_INT32, w), "15: write");

    // Open the native file with sec2 (no VFD) and read the dataset back
    // byte-clean; false if it is not yet a complete HDF5 image. The pre-flush
    // call is expected to error, so suppress the auto-printer around it.
    auto independent_read_ok = [&]() -> bool {
      H5E_auto2_t af = nullptr;
      void *ad = nullptr;
      H5Eget_auto2(H5E_DEFAULT, &af, &ad);
      H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
      hid_t rf = H5Fopen(kNativeDur, H5F_ACC_RDONLY, rfapl);
      bool ok = rf >= 0 && ReadDsetEq(rf, "durable", H5T_NATIVE_INT32, w);
      if (rf >= 0) H5Fclose(rf);
      H5Eset_auto2(H5E_DEFAULT, af, ad);
      return ok;
    };

    CHECK(!independent_read_ok(),
          "15: pre-flush image is incomplete to an independent reader");
    CHECK(H5Fflush(f, H5F_SCOPE_GLOBAL) >= 0, "15: H5Fflush (the barrier)");
    CHECK(independent_read_ok(),
          "15: post-flush data fully persisted + readable without the VFD");

    CHECK(H5Fclose(f) >= 0, "15: close writer");
    H5Pclose(rfapl);
    H5Pclose(wfapl);
    std::printf("[vfd-suite] ok 15: no-pending-dirty-state barrier (flush finalizes the image)\n");
  }

  H5Pclose(fapl);

  // === 4. Native tool matrix on the VFD's file (NO VFD loaded) ============
  {
    if (!HasTool("h5dump") || !HasTool("h5ls") || !HasTool("h5repack") ||
        !HasTool("h5diff")) {
      std::fprintf(stderr,
                   "[vfd-suite] WARN 4: native HDF5 CLI tools not on PATH; "
                   "SKIPPING the tool matrix (native-compat NOT verified here). "
                   "Install hdf5-tools to exercise this.\n");
    } else {
      std::string f = kNativeFile;
      CHECK(RunCmd("h5dump -H '" + f + "'") == 0, "4: h5dump -H");
      CHECK(RunCmd("h5ls -r '" + f + "'") == 0, "4: h5ls -r");
      std::string rp = "/tmp/clio_cte_vfd_repacked.h5";
      std::remove(rp.c_str());
      CHECK(RunCmd("h5repack '" + f + "' '" + rp + "'") == 0, "4: h5repack");
      CHECK(RunCmd("h5diff '" + f + "' '" + rp + "'") == 0,
            "4: h5diff repacked == original");
      std::printf("[vfd-suite] ok 4: native tool matrix (h5dump/h5ls/h5repack/h5diff)\n");
    }
  }

  std::printf("[vfd-suite] PASS: native write-through verified\n");
  return 0;
}
