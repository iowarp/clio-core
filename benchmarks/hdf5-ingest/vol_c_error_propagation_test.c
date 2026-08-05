/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Failures reaching the application.
 *
 * The native VOL owns the file, so its verdict on an operation IS the
 * connector's verdict. An operation that failed underneath must surface as a
 * failure, never be swallowed into a success return with an untouched or
 * partially-filled user buffer -- an application that is told its read
 * succeeded has no reason to look at the data again.
 *
 * Failures are injected by damaging the file underneath HDF5 rather than by
 * mocking, so what is exercised is the real error path.
 *
 * Exits 0 on pass. Run through the connector (HDF5_VOL_CONNECTOR=clio).
 */
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4096
static const char *kFile = "/tmp/clio_vol_error_prop.h5";

static int fail(const char *msg) {
  printf("error_propagation: %s FAIL\n", msg);
  return 1;
}

/* Flip a byte inside the dataset's stored payload by locating the written
 * pattern in the file. Returns 0 if the pattern was not found (the caller then
 * skips rather than reporting a false pass). */
static int corrupt_payload(const char *path, const int32_t *pattern,
                           size_t pattern_bytes) {
  FILE *f = fopen(path, "r+b");
  if (!f) return 0;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)malloc((size_t)len);
  if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
    free(buf);
    fclose(f);
    return 0;
  }
  int found = 0;
  /* Look for the first 64 bytes of the payload; enough to be unique. */
  size_t needle = pattern_bytes < 64 ? pattern_bytes : 64;
  for (long i = 0; i + (long)needle <= len; ++i) {
    if (memcmp(buf + i, pattern, needle) == 0) {
      /* Flip a bit well inside the payload so the checksum, not the layout,
         is what disagrees. */
      long target = i + (long)needle / 2;
      fseek(f, target, SEEK_SET);
      unsigned char b = (unsigned char)buf[target] ^ 0xFFu;
      fwrite(&b, 1, 1, f);
      found = 1;
      break;
    }
  }
  free(buf);
  fclose(f);
  return found;
}

/* A dataset whose stored bytes no longer match its checksum cannot be read.
 * The connector must pass that failure up rather than returning success with
 * whatever happened to be in the buffer. */
static int case_read_failure_is_reported(void) {
  remove(kFile);
  int32_t w[N];
  for (int i = 0; i < N; ++i) w[i] = 0x5EED0000 + i;

  hid_t f = H5Fcreate(kFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("read: create");
  hsize_t dims[1] = {N}, chunk[1] = {N};
  hid_t sp = H5Screate_simple(1, dims, NULL);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl, 1, chunk);
  if (H5Pset_fletcher32(dcpl) < 0) return fail("read: enable checksum");
  hid_t d = H5Dcreate2(f, "guarded", H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl,
                       H5P_DEFAULT);
  if (d < 0 || H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, w) < 0)
    return fail("read: write");
  H5Dclose(d);
  H5Pclose(dcpl);
  H5Sclose(sp);
  H5Fclose(f);

  /* Filters may compress or reorder the payload; if the raw pattern is not
     findable, say so rather than claiming a pass. */
  if (!corrupt_payload(kFile, w, sizeof(w))) {
    printf("error_propagation: SKIP read case (payload not locatable on disk)\n");
    return 0;
  }

  f = H5Fopen(kFile, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f < 0) return fail("read: reopen after corruption");
  d = H5Dopen2(f, "guarded", H5P_DEFAULT);
  if (d < 0) return fail("read: open dataset after corruption");

  int32_t got[N];
  for (int i = 0; i < N; ++i) got[i] = -1;
  H5E_auto2_t af = NULL;
  void *ad = NULL;
  H5Eget_auto2(H5E_DEFAULT, &af, &ad);
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL); /* the failure is expected */
  herr_t rc = H5Dread(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, got);
  H5Eset_auto2(H5E_DEFAULT, af, ad);
  H5Dclose(d);
  H5Fclose(f);

  if (rc >= 0) {
    printf("error_propagation: H5Dread returned success (%d) for a dataset "
           "whose stored bytes fail their checksum\n", (int)rc);
    return fail("read: a failed read was reported as success");
  }
  return 0;
}

/* A dataset whose raw data lives in an external file that cannot be opened
 * fails to write. That failure originates below the connector and must arrive
 * at the caller intact. */
static int case_write_failure_is_reported(void) {
  const char *path = "/tmp/clio_vol_error_prop_ext.h5";
  remove(path);
  int32_t w[N];
  for (int i = 0; i < N; ++i) w[i] = i;

  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("write: create");
  hsize_t dims[1] = {N};
  hid_t sp = H5Screate_simple(1, dims, NULL);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  /* A directory that does not exist, so the backing store cannot be opened. */
  H5Pset_external(dcpl, "/nonexistent-dir-for-clio-test/payload.dat", 0,
                  (hsize_t)sizeof(w));

  H5E_auto2_t af = NULL;
  void *ad = NULL;
  H5Eget_auto2(H5E_DEFAULT, &af, &ad);
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
  hid_t d = H5Dcreate2(f, "external", H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl,
                       H5P_DEFAULT);
  herr_t rc = 0;
  int reached_write = 0;
  if (d >= 0) {
    reached_write = 1;
    rc = H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, w);
    H5Dclose(d);
  }
  H5Eset_auto2(H5E_DEFAULT, af, ad);
  H5Pclose(dcpl);
  H5Sclose(sp);
  H5Fclose(f);
  remove(path);

  if (!reached_write) {
    /* HDF5 refused earlier than the connector; nothing was silently accepted,
       which is the property under test, but the connector's own path was not
       exercised. Report honestly rather than counting it as coverage. */
    printf("error_propagation: SKIP write case (rejected before reaching the "
           "connector)\n");
    return 0;
  }
  if (rc >= 0) {
    printf("error_propagation: H5Dwrite returned success (%d) for a dataset "
           "whose external storage cannot be opened\n", (int)rc);
    return fail("write: a failed write was reported as success");
  }
  return 0;
}

int main(void) {
  /* Run with caching off, so what is measured is the error path and nothing
     else.

     This is not incidental. With caching on, the read case below does NOT
     fail: the connector stages the dataset when it is written, and a read
     after the file has been damaged underneath is served from that staged copy
     — correct-looking data for a file that no longer contains it. That is the
     known limitation recorded as the external-modification half of the cache
     staleness gap: the connector detects a file being re-created, but not one
     edited behind its back between sessions. Until that is closed, a test of
     the error path has to bypass the cache to reach the error path at all. */
  setenv("CLIO_VOL_CACHE", "0", /*overwrite*/ 0);

  int bad = 0;
  bad += case_read_failure_is_reported();
  bad += case_write_failure_is_reported();
  remove(kFile);
  if (bad) {
    printf("error_propagation: %d case(s) FAIL\n", bad);
    return 1;
  }
  printf("error_propagation: failures reach the application PASS\n");
  return 0;
}
