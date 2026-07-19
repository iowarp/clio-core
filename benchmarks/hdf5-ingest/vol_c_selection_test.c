/* Isolated C test for clio VOL selection-aware READ caching (serve-only) and
 * partial-write cache invalidation.
 *
 * The connector caches whole-dataset writes as a linear image in CTE. This test
 * populates that cache (whole write + H5Fflush, which drains the async puts),
 * then reads sub-selections that the connector serves FROM the tier via HDF5
 * gather/scatter:
 *   A. file hyperslab, mem H5S_ALL  (gather only)
 *   B. file hyperslab, mem hyperslab (gather + scatter into a larger buffer)
 *   C. point selection               (gather only)
 * It then does a PARTIAL write (which must invalidate the stale linear cache)
 * and verifies a subsequent whole read reflects the new data (fresh from native,
 * not the stale cache).
 *
 * Correctness is checked against the known written values (native is the oracle;
 * a wrong serve path would return wrong data). Run with HDF5_VOL_CONNECTOR=clio
 * and a running clio_run. Exit 0 = pass.
 * Build: h5cc -o vol_c_selection_test vol_c_selection_test.c
 */
#include <hdf5.h>
#include <stdio.h>

#define R 8
#define C 6
#define VAL(r, c) ((r) * 100 + (c))

static int fail(const char *what) { fprintf(stderr, "FAIL: %s\n", what); return 0; }

int main(void) {
  const char *path = "/tmp/volc_selection.h5";
  int whole[R][C];
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) whole[r][c] = VAL(r, c);

  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[2] = {R, C};
  hid_t fs = H5Screate_simple(2, dims, NULL);
  /* Chunked {4,3} so the telemetry chunk-alignment probe has something to report:
     case D reads one whole chunk (aligned); cases A/B straddle chunk edges
     (misaligned). */
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t chunk[2] = {4, 3};
  H5Pset_chunk(dcpl, 2, chunk);
  hid_t d = H5Dcreate2(f, "m", H5T_NATIVE_INT, fs, H5P_DEFAULT, dcpl,
                       H5P_DEFAULT);
  H5Pclose(dcpl);
  if (H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, whole) < 0)
    return !fail("whole write");
  /* Drain async puts so the linear cache is populated for the reads below. */
  H5Fflush(f, H5F_SCOPE_GLOBAL);

  int ok = 1;

  /* --- A. file hyperslab rows[2..5] cols[1..4] (4x4), mem H5S_ALL --- */
  {
    hid_t sp = H5Dget_space(d);
    hsize_t start[2] = {2, 1}, count[2] = {4, 4};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, start, NULL, count, NULL);
    int got[16];
    if (H5Dread(d, H5T_NATIVE_INT, H5S_ALL, sp, H5P_DEFAULT, got) < 0)
      ok = fail("A read");
    int k = 0;
    for (int r = 2; r < 6 && ok; ++r)
      for (int c = 1; c < 5; ++c, ++k)
        if (got[k] != VAL(r, c)) ok = fail("A value");
    H5Sclose(sp);
  }

  /* --- B. file hyperslab, mem hyperslab into a full-size buffer --- */
  {
    hid_t sp = H5Dget_space(d);
    hsize_t start[2] = {2, 1}, count[2] = {4, 4};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, start, NULL, count, NULL);
    hid_t ms = H5Screate_simple(2, dims, NULL);
    H5Sselect_hyperslab(ms, H5S_SELECT_SET, start, NULL, count, NULL);
    int got[R][C];
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) got[r][c] = -1;
    if (H5Dread(d, H5T_NATIVE_INT, ms, sp, H5P_DEFAULT, got) < 0)
      ok = fail("B read");
    for (int r = 0; r < R && ok; ++r)
      for (int c = 0; c < C; ++c) {
        int inside = (r >= 2 && r < 6 && c >= 1 && c < 5);
        int want = inside ? VAL(r, c) : -1;
        if (got[r][c] != want) ok = fail("B value");
      }
    H5Sclose(ms);
    H5Sclose(sp);
  }

  /* --- C. point selection {(0,0),(7,5),(3,3)} --- */
  {
    hid_t sp = H5Dget_space(d);
    hsize_t pts[3][2] = {{0, 0}, {7, 5}, {3, 3}};
    H5Sselect_elements(sp, H5S_SELECT_SET, 3, (const hsize_t *)pts);
    int got[3];
    if (H5Dread(d, H5T_NATIVE_INT, H5S_ALL, sp, H5P_DEFAULT, got) < 0)
      ok = fail("C read");
    if (ok && (got[0] != VAL(0, 0) || got[1] != VAL(7, 5) || got[2] != VAL(3, 3)))
      ok = fail("C value");
    H5Sclose(sp);
  }

  /* --- D. aligned hyperslab rows[0..3] cols[0..2] (exactly one chunk) --- */
  {
    hid_t sp = H5Dget_space(d);
    hsize_t start[2] = {0, 0}, count[2] = {4, 3};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, start, NULL, count, NULL);
    int got[12];
    if (H5Dread(d, H5T_NATIVE_INT, H5S_ALL, sp, H5P_DEFAULT, got) < 0)
      ok = fail("D read");
    int k = 0;
    for (int r = 0; r < 4 && ok; ++r)
      for (int c = 0; c < 3; ++c, ++k)
        if (got[k] != VAL(r, c)) ok = fail("D value");
    H5Sclose(sp);
  }

  /* --- Coherence: partial write row 0 = 900+c, then whole read must reflect it
     (the partial write invalidates the stale linear cache). --- */
  {
    hid_t sp = H5Dget_space(d);
    hsize_t start[2] = {0, 0}, count[2] = {1, C};
    H5Sselect_hyperslab(sp, H5S_SELECT_SET, start, NULL, count, NULL);
    int row0[C];
    for (int c = 0; c < C; ++c) row0[c] = 900 + c;
    if (H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, sp, H5P_DEFAULT, row0) < 0)
      ok = fail("partial write");
    H5Fflush(f, H5F_SCOPE_GLOBAL);
    H5Sclose(sp);

    int got[R][C];
    if (H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, got) < 0)
      ok = fail("coherence read");
    for (int c = 0; c < C && ok; ++c)
      if (got[0][c] != 900 + c) ok = fail("coherence: stale row 0");
    for (int r = 1; r < R && ok; ++r)
      for (int c = 0; c < C; ++c)
        if (got[r][c] != VAL(r, c)) ok = fail("coherence: other rows");
  }

  H5Dclose(d);
  H5Sclose(fs);
  H5Fclose(f);
  printf("selection A/B/C/D + coherence: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
