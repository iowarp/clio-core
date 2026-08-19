/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Helper process for the "file left open at exit" test.
 *
 * The behaviour under test only exists at process exit, so it cannot be
 * reached from inside a test case: an application that never closes its HDF5
 * file has HDF5 close it from H5_term_library(), registered with atexit, and
 * whether the connector can still reach the runtime at that moment depends on
 * the order those handlers were registered in. A thread cannot reproduce that
 * and fork() cannot either -- forking a process that already hosts a runtime
 * gives the child one whose worker threads do not exist.
 *
 * So this runs as its own process, twice:
 *
 *   write   create a file, write a dataset, and exit WITHOUT calling
 *           H5Fclose -- exactly what LAMMPS' h5md dump does.
 *   verify  reopen in a fresh process with CLIO_RESTART=1 and check the tag
 *           survived: if the writer's stamp never landed, the open reads
 *           "stamp absent", drops the tag, re-creates it under a new id, and
 *           every chunk the writer stored becomes unreachable (size 0).
 *
 * Both modes exit non-zero on failure; the test asserts on the exit codes.
 */

#include <hdf5.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kNumElems = 4096;  /* 16 KiB -> 4 chunks at 4 KiB */

int DoWrite(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) return 2;
  hsize_t dims[1] = {kNumElems};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_INT, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) return 3;
  std::vector<int> wbuf(kNumElems);
  for (size_t i = 0; i < kNumElems; ++i) wbuf[i] = static_cast<int>(i * 5 + 2);
  if (H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               wbuf.data()) < 0)
    return 4;
  /* Closing the DATASET is normal -- it is what drains the chunk puts. The
     FILE is left open on purpose: that is the whole point of this process. */
  H5Dclose(dset);
  H5Sclose(space);
  return 0;  /* return from main: atexit handlers run, H5Fclose never called */
}

int DoVerify(const char *path) {
  /* Reopening is what exercises the stamp: a writer whose stamp never landed
     leaves this open unable to trust the tier, and the tag is dropped. */
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) return 2;
  hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);
  if (dset < 0) return 3;
  std::vector<int> rbuf(kNumElems, -1);
  if (H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              rbuf.data()) < 0)
    return 4;
  for (size_t i = 0; i < kNumElems; ++i) {
    if (rbuf[i] != static_cast<int>(i * 5 + 2)) return 5;
  }
  H5Dclose(dset);

  /* The data above is correct either way -- the native file is authoritative
     and answers a miss perfectly well, which is exactly why this bug stayed
     invisible. What distinguishes a live cache from a dead one is whether the
     writer's chunks are still reachable under the tag this open resolved. */
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return 6;
  auto *cte = CLIO_CTE_CLIENT;
  auto tag = cte->AsyncGetOrCreateTag(std::string("hdf5:") + path);
  tag.Wait();
  if (tag->GetReturnCode() != 0) return 7;
  auto sz = cte->AsyncGetBlobSize(tag->tag_id_, "/data/chunk_0");
  sz.Wait();
  if (sz->size_ == 0) {
    std::fprintf(stderr,
                 "exit_stamp_child: /data/chunk_0 is unreachable after "
                 "reopen -- the writer's stamp did not land, so the tag was "
                 "dropped and re-created\n");
    return 8;
  }
  H5Fclose(file);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <write|verify> <h5-path>\n", argv[0]);
    return 1;
  }
  if (std::strcmp(argv[1], "write") == 0) return DoWrite(argv[2]);
  if (std::strcmp(argv[1], "verify") == 0) return DoVerify(argv[2]);
  return 1;
}
