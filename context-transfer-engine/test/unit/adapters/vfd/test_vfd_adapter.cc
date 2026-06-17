/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * HDF5 VFD smoke test. Selects the clio_cte_hdf5_vfd driver and writes/reads a
 * dataset to a clio:: file, end-to-end through the context-filesystem chimod
 * (against a co-located runtime). Plain main so it pulls in only HDF5 + the
 * runtime client (no Catch2).
 */
#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_cte/core/core_client.h"
#include "adapter/vfd/H5FDhermes.h"

namespace {
constexpr hsize_t kN = 4096;  // 4096 doubles = 32 KiB dataset
const char *kBackend = "/tmp/clio_cte_vfd_test.dat";
const char *kClioFile = "clio::/tmp/clio_cte_vfd_smoke.h5";

#define CHECK(cond, msg)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      std::fprintf(stderr, "[vfd-smoke] FAIL: %s\n", (msg));   \
      return 1;                                                \
    }                                                          \
  } while (0)

bool InitRuntime() {
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    return false;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    return false;
  }
  auto *cte = CLIO_CTE_CLIENT;
  chi::PoolId bdev_pool_id(953, 0);
  clio::run::bdev::Client bdev(bdev_pool_id);
  auto ct = bdev.AsyncCreate(chi::PoolQuery::Dynamic(), kBackend, bdev_pool_id,
                             clio::run::bdev::BdevType::kFile);
  ct.Wait();
  auto rt = cte->AsyncRegisterTarget(kBackend, clio::run::bdev::BdevType::kFile,
                                     64ULL * 1024 * 1024, chi::PoolQuery::Local(),
                                     bdev_pool_id);
  rt.Wait();
  return rt->GetReturnCode() == 0;
}
}  // namespace

int main() {
  if (!InitRuntime()) {
    std::fprintf(stderr, "[vfd-smoke] FAIL: runtime/CTE init\n");
    return 1;
  }

  // Build a FAPL that uses the clio VFD.
  hid_t driver = H5FD_hermes_init();
  CHECK(driver >= 0, "H5FD_hermes_init");
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  CHECK(fapl >= 0, "H5Pcreate(FAPL)");
  CHECK(H5Pset_driver(fapl, driver, nullptr) >= 0, "H5Pset_driver");

  std::vector<double> w(kN), r(kN, 0.0);
  for (hsize_t i = 0; i < kN; ++i) {
    w[i] = static_cast<double>(i) * 1.5 - 7.0;
  }

  // --- Create + write a dataset through the VFD ---
  hid_t file = H5Fcreate(kClioFile, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  CHECK(file >= 0, "H5Fcreate via clio VFD");
  hsize_t dims[1] = {kN};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  CHECK(dset >= 0, "H5Dcreate2");
  CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 w.data()) >= 0,
        "H5Dwrite");
  H5Dclose(dset);
  H5Sclose(space);
  CHECK(H5Fclose(file) >= 0, "H5Fclose (write)");

  // --- Reopen + read it back ---
  hid_t file2 = H5Fopen(kClioFile, H5F_ACC_RDONLY, fapl);
  CHECK(file2 >= 0, "H5Fopen via clio VFD");
  hid_t dset2 = H5Dopen2(file2, "data", H5P_DEFAULT);
  CHECK(dset2 >= 0, "H5Dopen2");
  CHECK(H5Dread(dset2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                r.data()) >= 0,
        "H5Dread");
  H5Dclose(dset2);
  CHECK(H5Fclose(file2) >= 0, "H5Fclose (read)");

  for (hsize_t i = 0; i < kN; ++i) {
    if (r[i] != w[i]) {
      std::fprintf(stderr, "[vfd-smoke] FAIL: data mismatch at %llu\n",
                   (unsigned long long)i);
      return 1;
    }
  }

  H5Pclose(fapl);
  std::printf("[vfd-smoke] PASS: wrote+read %llu doubles via clio HDF5 VFD\n",
              (unsigned long long)kN);
  return 0;
}
