/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * The CLIO VFD without a CLIO runtime.
 *
 * The authoritative store is a plain on-disk native HDF5 file, so the driver is
 * a complete and correct HDF5 driver whether or not a CLIO runtime is running.
 * The CTE tier is an accelerator layered on top of that, not a prerequisite.
 * This asserts the guarantee directly: with no runtime reachable, applications
 * must still be able to create, write, close, reopen and read their files.
 *
 * Two configurations are covered:
 *   1. Cache explicitly disabled -- the documented native-only mode, which must
 *      never touch CLIO at all.
 *   2. Cache requested but unavailable -- the accident case. The open must
 *      SUCCEED and fall back to native-only rather than failing the
 *      application, because nothing about the file's correctness depends on the
 *      tier.
 *
 * This deliberately lives in its own binary. The main VFD suite brings a
 * runtime up before its first case, so it structurally cannot express "no
 * runtime"; a test of the no-runtime path has to be a process that never
 * starts one.
 */
#include <hdf5.h>

#include <cstdio>
#ifdef _WIN32
#include <crtdbg.h>
#endif
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "adapter/vfd/H5FDclio.h"

namespace {

/** setenv(3) is POSIX; MSVC spells it _putenv_s and has no "don't overwrite"
 *  mode, so honor overwrite=0 by checking first. Kept local rather than shared
 *  with test_vfd_adapter.cc on purpose: this test deliberately links nothing
 *  beyond the driver, which is the whole point of "no runtime". */
inline void TestSetenv(const char *name, const char *value, int overwrite) {
#ifdef _WIN32
  if (!overwrite && std::getenv(name) != nullptr) return;
  (void)_putenv_s(name, value);
#else
  (void)setenv(name, value, overwrite);
#endif
}

constexpr hsize_t kN = 4096;

#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "[vfd-no-runtime] FAIL: %s\n", (msg));     \
      return 1;                                                       \
    }                                                                 \
  } while (0)

std::vector<double> MakeF64(hsize_t n) {
  std::vector<double> v(n);
  for (hsize_t i = 0; i < n; ++i) v[i] = static_cast<double>(i) * 3.5 - 11.0;
  return v;
}

bool WriteDset(hid_t loc, const char *name, const std::vector<double> &v) {
  hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  bool ok = dset >= 0 && H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, v.data()) >= 0;
  if (dset >= 0) H5Dclose(dset);
  H5Sclose(space);
  return ok;
}

bool ReadDsetEq(hid_t loc, const char *name, const std::vector<double> &want) {
  hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
  if (dset < 0) return false;
  std::vector<double> got(want.size());
  bool ok = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    got.data()) >= 0;
  H5Dclose(dset);
  return ok && std::memcmp(got.data(), want.data(),
                           want.size() * sizeof(double)) == 0;
}

// Full lifecycle through the driver, then an independent native read of the
// same file with no driver loaded -- which is what proves the bytes really
// landed in a valid native HDF5 image rather than anywhere CLIO-specific.
int RoundTrip(const char *label, hbool_t cache_enabled, const char *path) {
  const std::vector<double> w = MakeF64(kN);
  std::remove(path);

  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  CHECK(fapl >= 0, "H5Pcreate");
  CHECK(H5Pset_fapl_clio(fapl, cache_enabled) >= 0, "H5Pset_fapl_clio");

  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  CHECK(f >= 0, "H5Fcreate with no runtime running");
  CHECK(WriteDset(f, "d", w), "write with no runtime running");
  CHECK(H5Fflush(f, H5F_SCOPE_GLOBAL) >= 0, "flush with no runtime running");
  CHECK(H5Fclose(f) >= 0, "close with no runtime running");

  hid_t f2 = H5Fopen(path, H5F_ACC_RDONLY, fapl);
  CHECK(f2 >= 0, "reopen with no runtime running");
  CHECK(ReadDsetEq(f2, "d", w), "data reads back intact with no runtime");
  CHECK(H5Fclose(f2) >= 0, "close after read");
  H5Pclose(fapl);

  hid_t fn = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK(fn >= 0, "the file opens with no driver at all");
  CHECK(ReadDsetEq(fn, "d", w), "a plain native reader sees the same data");
  CHECK(H5Fclose(fn) >= 0, "close native");

  std::printf("[vfd-no-runtime] ok: %s\n", label);
  return 0;
}
}  // namespace

int main() {
  // Make a hard failure in this binary say something. Every one of these was
  // load-bearing while diagnosing an abort in the driver's close path, and they
  // are cheap enough to keep permanently: without them the process exited 3
  // having printed NOTHING, because abort() discards buffered stdout and takes
  // every case marker with it. Unbuffer both streams so the last thing printed
  // survives, and name the terminating exception rather than leaving a silent
  // exit code.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
#ifdef _WIN32
  // A Debug HDF5 has its internal assert()s live, and the debug CRT reports
  // those through a MODAL DIALOG once a debugger is attached -- which blocks an
  // unattended run forever instead of failing it. Route them to stderr so the
  // assert text is captured and the process proceeds to abort().
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _set_error_mode(_OUT_TO_STDERR);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
  std::set_terminate([] {
    if (auto e = std::current_exception()) {
      try {
        std::rethrow_exception(e);
      } catch (const std::exception &ex) {
        std::fprintf(stderr, "[vfd-no-runtime] TERMINATE: %s\n", ex.what());
      } catch (...) {
        std::fprintf(stderr, "[vfd-no-runtime] TERMINATE: non-std exception\n");
      }
    } else {
      std::fprintf(stderr, "[vfd-no-runtime] TERMINATE: no active exception\n");
    }
    std::fflush(stderr);
    std::_Exit(3);
  });

  // Give up on a missing runtime immediately instead of retrying for the
  // default 60s. Without this the second case below spends a minute confirming
  // what it already knows, and the fallback it is checking looks like a hang.
  TestSetenv("CLIO_CLIENT_RETRY_TIMEOUT", "0", /*overwrite*/ 0);

  hid_t driver = H5FD_clio_init();
  CHECK(driver >= 0, "the driver registers without a runtime");

  if (RoundTrip("native-only mode never needs CLIO", /*cache_enabled*/ 0,
                "/tmp/clio_cte_vfd_nort_off.h5")) {
    return 1;
  }
  if (RoundTrip("caching requested but unavailable falls back to native",
                /*cache_enabled*/ 1, "/tmp/clio_cte_vfd_nort_on.h5")) {
    return 1;
  }

  // Deleting a file must also not depend on the runtime: there is no cache
  // entry to orphan when there is no cache.
  {
    const char *path = "/tmp/clio_cte_vfd_nort_del.h5";
    std::remove(path);
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHECK(fapl >= 0 && H5Pset_fapl_clio(fapl, 0) >= 0, "delete: fapl");
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(f >= 0 && WriteDset(f, "d", MakeF64(64)) && H5Fclose(f) >= 0,
          "delete: seed the file");
    CHECK(H5Fdelete(path, fapl) >= 0, "H5Fdelete succeeds with no runtime");
    hid_t gone = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(gone < 0, "the deleted file is really gone");
    H5Pclose(fapl);
    std::printf("[vfd-no-runtime] ok: delete works with no runtime\n");
  }

  std::printf("[vfd-no-runtime] PASS: the driver is complete without CLIO\n");
  return 0;
}
