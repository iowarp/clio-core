/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * VOL cache-identity regression tests. Each case here corresponds to a way the
 * CTE blob cache could claim to hold data it does not actually hold, and each
 * one returned WRONG DATA WITH A SUCCESS STATUS before the fix. h5py cannot
 * express these (it picks its own memory types and does not let you re-create a
 * file underneath a live tag), so they live in C.
 *
 *   1. mem/file datatype mismatch (VOL_AUDIT W3). The blob image is sized by
 *      the TRANSFER's datatype, so writing int32 into an int64 dataset cached
 *      4n bytes while a later int64 read asked for 8n -- hit-tested chunk_0,
 *      found it, and reassembled 8n bytes from 4n.
 *   2. cache outliving its file (VOL_AUDIT W4). The tag is keyed on the file
 *      NAME, which H5F_ACC_TRUNC does not change, so blobs from a previous file
 *      of the same name survived a re-create.
 *   3. H5Dflush as a durability barrier (VOL_AUDIT W7). dataset_specific was a
 *      bare pass-through, so H5Dflush returned with async CTE puts in flight.
 *
 * Exits 0 on pass. Run through the VOL (HDF5_VOL_CONNECTOR=clio).
 */
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 512
static const char *kFile = "/tmp/clio_vol_cache_identity.h5";

static int fail(const char *msg) {
  printf("cache_identity: %s FAIL\n", msg);
  return 1;
}

/* Case 1: write with a NARROWER memory type than the dataset stores, then read
 * back with a memory type matching the file. HDF5 converts on both sides, so
 * the values must round-trip exactly; only a cache that confused the two sizes
 * returns anything else. */
static int case_dtype_mismatch(void) {
  hid_t f = H5Fcreate(kFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("case1 create");

  hsize_t dims[1] = {N};
  hid_t sp = H5Screate_simple(1, dims, NULL);
  /* Dataset stores 64-bit ints ... */
  hid_t d = H5Dcreate2(f, "mixed", H5T_STD_I64LE, sp, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT);
  if (d < 0) return fail("case1 create dataset");

  /* ... but the application writes 32-bit ints. */
  int32_t w[N];
  for (int i = 0; i < N; ++i) w[i] = i * 3 - 7;
  if (H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, w) < 0)
    return fail("case1 write int32");
  H5Dclose(d);
  H5Sclose(sp);
  H5Fclose(f);

  /* Read back with a 64-bit memory type -- a different transfer size for the
     same dataset. Then read again with int32 to exercise both orders. */
  f = H5Fopen(kFile, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f < 0) return fail("case1 reopen");
  d = H5Dopen2(f, "mixed", H5P_DEFAULT);
  if (d < 0) return fail("case1 open dataset");

  int64_t got64[N];
  memset(got64, 0, sizeof(got64));
  if (H5Dread(d, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, got64) < 0)
    return fail("case1 read int64");
  for (int i = 0; i < N; ++i) {
    if (got64[i] != (int64_t)w[i]) {
      printf("cache_identity: case1 int64 read mismatch at %d: got %lld want %lld\n",
             i, (long long)got64[i], (long long)w[i]);
      return fail("case1 dtype mismatch served wrong bytes");
    }
  }

  int32_t got32[N];
  memset(got32, 0, sizeof(got32));
  if (H5Dread(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, got32) < 0)
    return fail("case1 read int32");
  for (int i = 0; i < N; ++i) {
    if (got32[i] != w[i]) {
      printf("cache_identity: case1 int32 read mismatch at %d: got %d want %d\n",
             i, got32[i], w[i]);
      return fail("case1 second-type read served wrong bytes");
    }
  }
  H5Dclose(d);
  H5Fclose(f);
  return 0;
}

/* Case 2: re-create the SAME FILENAME with H5F_ACC_TRUNC, re-create a dataset
 * of the SAME NAME, and leave it UNWRITTEN. Native semantics say an unwritten
 * dataset reads as its fill value (0). The blob cache is keyed on the file name
 * and the dataset path -- neither of which the truncate changed -- so a tag that
 * survives the re-create still holds the OLD file's bytes under that exact key
 * and serves them as a hit.
 *
 * Leaving the dataset unwritten is what makes this observable: any write through
 * the connector would re-stage the chunks and paper over the stale tag. */
static int case_truncate_drops_cache(void) {
  hsize_t dims[1] = {N};
  hid_t sp = H5Screate_simple(1, dims, NULL);

  /* Run 1: populate "d" and read it back so it is definitely staged. */
  hid_t f = H5Fcreate(kFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("case2 create #1");
  hid_t d = H5Dcreate2(f, "d", H5T_NATIVE_INT32, sp, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT);
  int32_t w[N];
  for (int i = 0; i < N; ++i) w[i] = 1000 + i;
  if (d < 0 || H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, w) < 0)
    return fail("case2 write #1");
  H5Dclose(d);
  H5Fclose(f);

  f = H5Fopen(kFile, H5F_ACC_RDONLY, H5P_DEFAULT);
  d = H5Dopen2(f, "d", H5P_DEFAULT);
  int32_t warm[N];
  if (d < 0 || H5Dread(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, warm) < 0)
    return fail("case2 warm read");
  H5Dclose(d);
  H5Fclose(f);

  /* Run 2: same filename, same dataset name, never written. */
  f = H5Fcreate(kFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("case2 create #2");
  hid_t d2 = H5Dcreate2(f, "d", H5T_NATIVE_INT32, sp, H5P_DEFAULT, H5P_DEFAULT,
                        H5P_DEFAULT);
  if (d2 < 0) return fail("case2 create dataset #2");
  H5Dclose(d2);
  H5Sclose(sp);
  H5Fclose(f);

  f = H5Fopen(kFile, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f < 0) return fail("case2 reopen");
  d2 = H5Dopen2(f, "d", H5P_DEFAULT);
  int32_t got[N];
  for (int i = 0; i < N; ++i) got[i] = -1;
  if (d2 < 0 || H5Dread(d2, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, got) < 0)
    return fail("case2 read unwritten dataset");
  H5Dclose(d2);
  H5Fclose(f);
  for (int i = 0; i < N; ++i) {
    if (got[i] != 0) {
      printf("cache_identity: case2 unwritten dataset at %d: got %d want 0 "
             "(this is run 1's value %d)\n", i, got[i], w[i]);
      return fail("case2 cache outlived H5F_ACC_TRUNC of its file");
    }
  }
  return 0;
}

/* Case 3: H5Dflush must leave a complete, readable image in the native file for
 * an independent reader while the writer is still open.
 *
 * NOTE on what this does and does not prove. The connector writes through to
 * native synchronously, so native visibility here is not the thing that was
 * broken -- the W7 gap was that H5Dflush returned with async CTE puts still in
 * flight. That drain is internal and has no observable at this layer (the same
 * limitation the VFD suite notes for fsync). This case pins the native-side
 * contract and guards the pass-through wiring around the new drain; it is not a
 * test of the drain itself. */
static int case_dataset_flush_barrier(void) {
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_file_locking(fapl, 0, 1); /* reader must not be blocked by the writer */

  hid_t f = H5Fcreate(kFile, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (f < 0) return fail("case3 create");
  hsize_t dims[1] = {N};
  hid_t sp = H5Screate_simple(1, dims, NULL);
  hid_t d = H5Dcreate2(f, "flushed", H5T_NATIVE_INT32, sp, H5P_DEFAULT,
                       H5P_DEFAULT, H5P_DEFAULT);
  int32_t w[N];
  for (int i = 0; i < N; ++i) w[i] = i ^ 0x5a5a;
  if (d < 0 || H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, w) < 0)
    return fail("case3 write");

  if (H5Dflush(d) < 0) return fail("case3 H5Dflush");

  /* Independent handle, writer still open. */
  hid_t rf = H5Fopen(kFile, H5F_ACC_RDONLY, fapl);
  if (rf < 0) return fail("case3 independent open after H5Dflush");
  hid_t rd = H5Dopen2(rf, "flushed", H5P_DEFAULT);
  int32_t got[N];
  memset(got, 0, sizeof(got));
  int ok = (rd >= 0 &&
            H5Dread(rd, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, got) >= 0 &&
            memcmp(got, w, sizeof(w)) == 0);
  if (rd >= 0) H5Dclose(rd);
  H5Fclose(rf);
  H5Dclose(d);
  H5Sclose(sp);
  H5Fclose(f);
  H5Pclose(fapl);
  if (!ok) return fail("case3 data not visible after H5Dflush");
  return 0;
}

int main(void) {
  /* Run every case even after one fails: these are three independent
     regressions, and knowing which of them a change broke is the whole point. */
  remove(kFile);
  int bad = 0;
  bad += case_dtype_mismatch();
  remove(kFile);
  bad += case_truncate_drops_cache();
  remove(kFile);
  bad += case_dataset_flush_barrier();
  remove(kFile);
  if (bad) {
    printf("cache_identity: %d/3 cases FAIL\n", bad);
    return 1;
  }
  printf("cache_identity: dtype-mismatch + truncate-drop + H5Dflush PASS\n");
  return 0;
}
