/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Operations that must work through the connector unchanged.
 *
 * A pass-through connector forwards what it does not itself implement. How an
 * unforwarded operation behaves varies: HDF5 routes some of these through
 * paths that survive a missing callback (a rename can be expressed as a hard
 * link plus a delete; token comparison has a byte-wise fallback), while others
 * fail outright. That variation is the reason to assert on the operations
 * rather than reason about them -- checked against this connector, all of the
 * below work, and this pins that so a future change cannot quietly remove one.
 *
 * These are the operations most likely to be overlooked, because none of them
 * touch the cache path that gets all the attention:
 *
 *   1. Renaming a link.
 *   2. Comparing and serialising object tokens (how H5Oget_info-driven code
 *      establishes object identity).
 *   3. Reading and writing chunks directly, bypassing the filter pipeline.
 *   4. Running with the cache switched off entirely, which must be a correct
 *      (if unaccelerated) mode rather than a broken one.
 *   5. Reading a selection from a dataset larger than the connector is willing
 *      to serve from cache -- the answer must still be right, just fetched
 *      from the file instead.
 *
 * Exits 0 on pass. Run through the connector (HDF5_VOL_CONNECTOR=clio).
 */
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1024
static const char *kFile = "/tmp/clio_vol_passthrough_ops.h5";

static int fail(const char *msg) {
  printf("passthrough_ops: %s FAIL\n", msg);
  return 1;
}

static int32_t g_vals[N];
static void fill(void) {
  for (int i = 0; i < N; ++i) g_vals[i] = i * 11 - 5;
}

static int write_dset(hid_t loc, const char *name) {
  hsize_t dims[1] = {N};
  hid_t sp = H5Screate_simple(1, dims, NULL);
  hid_t d = H5Dcreate2(loc, name, H5T_NATIVE_INT32, sp, H5P_DEFAULT,
                       H5P_DEFAULT, H5P_DEFAULT);
  int ok = (d >= 0 && H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, g_vals) >= 0);
  if (d >= 0) H5Dclose(d);
  H5Sclose(sp);
  return ok;
}

static int read_dset_eq(hid_t loc, const char *name) {
  hid_t d = H5Dopen2(loc, name, H5P_DEFAULT);
  if (d < 0) return 0;
  int32_t got[N];
  memset(got, 0, sizeof(got));
  int ok = (H5Dread(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, got) >= 0 &&
            memcmp(got, g_vals, sizeof(g_vals)) == 0);
  H5Dclose(d);
  return ok;
}

/* Renaming a link, and object identity via tokens. */
static int case_link_rename_and_tokens(void) {
  remove(kFile);
  hid_t f = H5Fcreate(kFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("create");
  if (!write_dset(f, "original")) return fail("seed dataset");

  if (H5Lmove(f, "original", f, "renamed", H5P_DEFAULT, H5P_DEFAULT) < 0)
    return fail("H5Lmove");
  if (H5Lexists(f, "original", H5P_DEFAULT) > 0)
    return fail("the old link still exists after a rename");
  if (H5Lexists(f, "renamed", H5P_DEFAULT) <= 0)
    return fail("the new link is missing after a rename");
  if (!read_dset_eq(f, "renamed"))
    return fail("data changed across a rename");

  /* Two links to one object must report identical tokens; a different object
     must not. Also round-trip a token through its string form. */
  if (H5Lcreate_hard(f, "renamed", f, "alias", H5P_DEFAULT, H5P_DEFAULT) < 0)
    return fail("H5Lcreate_hard");
  if (!write_dset(f, "other")) return fail("second dataset");

  H5O_info2_t a, b, c;
  if (H5Oget_info_by_name3(f, "renamed", &a, H5O_INFO_BASIC, H5P_DEFAULT) < 0 ||
      H5Oget_info_by_name3(f, "alias", &b, H5O_INFO_BASIC, H5P_DEFAULT) < 0 ||
      H5Oget_info_by_name3(f, "other", &c, H5O_INFO_BASIC, H5P_DEFAULT) < 0)
    return fail("H5Oget_info_by_name3");

  int cmp = 1;
  if (H5Otoken_cmp(f, &a.token, &b.token, &cmp) < 0)
    return fail("H5Otoken_cmp on two links to one object");
  if (cmp != 0) return fail("two links to one object compared as different");
  if (H5Otoken_cmp(f, &a.token, &c.token, &cmp) < 0)
    return fail("H5Otoken_cmp on two different objects");
  if (cmp == 0) return fail("two different objects compared as identical");

  char *tok_str = NULL;
  H5O_token_t back;
  if (H5Otoken_to_str(f, &a.token, &tok_str) < 0 || tok_str == NULL)
    return fail("H5Otoken_to_str");
  if (H5Otoken_from_str(f, tok_str, &back) < 0)
    return fail("H5Otoken_from_str");
  H5free_memory(tok_str);
  if (H5Otoken_cmp(f, &a.token, &back, &cmp) < 0 || cmp != 0)
    return fail("a token did not survive a round-trip through its string form");

  H5Fclose(f);
  return 0;
}

/* Chunk-level reads and writes, which bypass the filter pipeline entirely. */
static int case_direct_chunk_io(void) {
  const char *path = "/tmp/clio_vol_passthrough_chunk.h5";
  remove(path);
  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("chunk: create");
  hsize_t dims[1] = {N}, chunk[1] = {N / 4};
  hid_t sp = H5Screate_simple(1, dims, NULL);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl, 1, chunk);
  hid_t d = H5Dcreate2(f, "chunked", H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl,
                       H5P_DEFAULT);
  if (d < 0) return fail("chunk: create dataset");

  int32_t raw[N / 4];
  for (int i = 0; i < N / 4; ++i) raw[i] = 0x1234 + i;
  hsize_t off[1] = {0};
  if (H5Dwrite_chunk(d, H5P_DEFAULT, 0, off, sizeof(raw), raw) < 0)
    return fail("H5Dwrite_chunk");

  int32_t back[N / 4];
  memset(back, 0, sizeof(back));
  uint32_t filters = 0;
  /* H5Dread_chunk is a VERSIONED API symbol, and the two forms differ in
   * arity. H5version.h resolves it through H5Dread_chunk_vers: version 2
   * (HDF5 2.x, which added a buffer-size argument) or version 1 (the 5-argument
   * form, which is all that exists on 1.14). CI builds against both -- HDF5
   * 2.1.1 in deps-cpu on Linux, conda-forge 1.14.6 on macOS -- and the project
   * supports HDF5 >= 1.14, so this has to compile either way.
   *
   * Branch on H5Dread_chunk_vers rather than on the library version: the API
   * version is set independently (H5_USE_114_API on a 2.x library selects
   * version 1), so the library version does not tell you which form you get.
   * Do NOT instead pin this to version 1 to get a single code path -- the
   * version-1 declaration lives inside #ifndef H5_NO_DEPRECATED_SYMBOLS, so a
   * 2.x build with deprecated symbols disabled would not have it.
   *
   * Both forms read the same chunk; version 2's extra argument is a bounds
   * check on the destination, so nothing this test asserts depends on which
   * one ran. */
#if defined(H5Dread_chunk_vers) && H5Dread_chunk_vers >= 2
  size_t back_size = sizeof(back);
  if (H5Dread_chunk(d, H5P_DEFAULT, off, &filters, back, &back_size) < 0)
    return fail("H5Dread_chunk");
#else
  if (H5Dread_chunk(d, H5P_DEFAULT, off, &filters, back) < 0)
    return fail("H5Dread_chunk");
#endif
  if (memcmp(back, raw, sizeof(raw)) != 0)
    return fail("a chunk did not survive a direct write/read");

  H5Dclose(d);
  H5Pclose(dcpl);
  H5Sclose(sp);
  H5Fclose(f);
  remove(path);
  return 0;
}

/* With caching switched off the connector is a plain pass-through. Data must
 * still be correct, and files must still be ordinary HDF5 files. */
static int case_cache_disabled(void) {
  const char *path = "/tmp/clio_vol_passthrough_nocache.h5";
  remove(path);
  setenv("CLIO_VOL_CACHE", "0", 1);

  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) { unsetenv("CLIO_VOL_CACHE"); return fail("no-cache: create"); }
  int ok = write_dset(f, "d");
  H5Fclose(f);
  if (!ok) { unsetenv("CLIO_VOL_CACHE"); return fail("no-cache: write"); }

  f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  ok = (f >= 0 && read_dset_eq(f, "d"));
  if (f >= 0) H5Fclose(f);
  unsetenv("CLIO_VOL_CACHE");
  if (!ok) return fail("no-cache: data did not round-trip with caching off");

  /* And the same file re-reads correctly with caching back on. */
  f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  ok = (f >= 0 && read_dset_eq(f, "d"));
  if (f >= 0) H5Fclose(f);
  remove(path);
  if (!ok) return fail("no-cache: a file written with caching off reads wrong "
                       "with caching on");
  return 0;
}

/* A selection read from a dataset above the serve ceiling must fall back to
 * the file and still return the right answer. */
static int case_selection_above_serve_ceiling(void) {
  const char *path = "/tmp/clio_vol_passthrough_ceiling.h5";
  remove(path);
  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return fail("ceiling: create");
  if (!write_dset(f, "big")) return fail("ceiling: write");
  H5Fclose(f);

  f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f < 0) return fail("ceiling: reopen");
  hid_t d = H5Dopen2(f, "big", H5P_DEFAULT);
  if (d < 0) return fail("ceiling: open dataset");

  /* Warm the cache with a whole read, then take a hyperslab -- the path that
     would serve from cache if the dataset were small enough to. */
  int32_t whole[N];
  if (H5Dread(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, whole) < 0)
    return fail("ceiling: warming read");

  hid_t fs = H5Dget_space(d);
  hsize_t start[1] = {N / 4}, count[1] = {N / 2};
  H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, NULL, count, NULL);
  hid_t ms = H5Screate_simple(1, count, NULL);
  int32_t part[N / 2];
  memset(part, 0, sizeof(part));
  if (H5Dread(d, H5T_NATIVE_INT32, ms, fs, H5P_DEFAULT, part) < 0)
    return fail("ceiling: hyperslab read");
  for (hsize_t i = 0; i < count[0]; ++i) {
    if (part[i] != g_vals[start[0] + i]) {
      printf("passthrough_ops: ceiling mismatch at %llu: got %d want %d\n",
             (unsigned long long)i, part[i], g_vals[start[0] + i]);
      return fail("ceiling: selection above the serve ceiling read wrong data");
    }
  }
  H5Sclose(ms);
  H5Sclose(fs);
  H5Dclose(d);
  H5Fclose(f);
  remove(path);
  return 0;
}

int main(void) {
  /* Force every dataset in this program above the cache-serving ceiling, so
     the fallback path is what runs. Must be set before the first selection
     read -- the limit is read once. */
  setenv("CLIO_VOL_MAX_SERVE_BYTES", "16", /*overwrite*/ 0);

  fill();
  int bad = 0;
  bad += case_link_rename_and_tokens();
  bad += case_direct_chunk_io();
  bad += case_cache_disabled();
  bad += case_selection_above_serve_ceiling();
  remove(kFile);
  if (bad) {
    printf("passthrough_ops: %d case(s) FAIL\n", bad);
    return 1;
  }
  printf("passthrough_ops: rename + tokens + direct chunk I/O + cache-off + "
         "serve ceiling PASS\n");
  return 0;
}
