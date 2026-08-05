/* Isolated C test for the filename-scoped file_specific ops through the clio VOL.
 *
 * H5VL_FILE_IS_ACCESSIBLE and H5VL_FILE_DELETE act on a *path*, not on an open
 * file, so HDF5 invokes the connector's file_specific callback with obj == NULL.
 * A connector that casts and dereferences that pointer segfaults; 3e8979cd
 * fixed that, and this pins it so a future edit to clio_file_specific cannot
 * quietly put the crash back.
 *
 * Two independent ways in, which is why the path is worth a dedicated probe:
 *   - HDF5 itself, via H5VL__file_open_find_connector_cb, asks EVERY plugin on
 *     HDF5_PLUGIN_PATH whether it can open a file -- so the connector did not
 *     have to be selected for a plain H5Fopen to crash.
 *   - Applications, directly: netCDF-C calls H5Fis_accessible from
 *     NC_infermodel on every nc_create/nc_open, so every netCDF-4 program hit
 *     it before any data moved.
 *
 * The h5py-based cases in this suite cannot reach either -- h5py never calls
 * H5Fis_accessible -- which is why this lives here as a C probe.
 *
 * Checks, in order:
 *   1. H5Fis_accessible on a real HDF5 file          -> true
 *   2. H5Fis_accessible on a non-HDF5 file           -> false (not a crash)
 *   3. H5Fis_accessible on a path that does not exist-> false/negative, no crash
 *   4. H5Fdelete on the real file                    -> succeeds, file is gone
 *
 * Run with HDF5_VOL_CONNECTOR=clio (and a running clio_run). Exit 0 = pass.
 * Build: h5cc -o vol_c_isaccessible_test vol_c_isaccessible_test.c
 */
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N 64

/* Build a small, valid HDF5 file through whatever VOL is active. */
static int make_file(const char *path) {
  int vals[N];
  for (int i = 0; i < N; ++i) vals[i] = i;

  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (f < 0) return 0;
  hsize_t dims[1] = {N};
  hid_t s = H5Screate_simple(1, dims, NULL);
  hid_t d = H5Dcreate2(f, "vals", H5T_NATIVE_INT, s, H5P_DEFAULT, H5P_DEFAULT,
                       H5P_DEFAULT);
  int ok = (d >= 0) &&
           (H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals) >= 0);
  if (d >= 0) H5Dclose(d);
  H5Sclose(s);
  H5Fclose(f);
  return ok;
}

int main(void) {
  const char *path = "/tmp/volc_isaccessible.h5";
  const char *junk = "/tmp/volc_isaccessible_junk.bin";
  const char *absent = "/tmp/volc_isaccessible_absent.h5";

  unlink(path);
  unlink(junk);
  unlink(absent);

  if (!make_file(path)) {
    fprintf(stderr, "setup: could not create %s\n", path);
    return 2;
  }

  FILE *fh = fopen(junk, "wb");
  if (!fh) { fprintf(stderr, "setup: could not create %s\n", junk); return 2; }
  fputs("this is not an HDF5 file", fh);
  fclose(fh);

  /* 1/2/3: the IS_ACCESSIBLE path, which is where obj == NULL arrives. */
  htri_t real = H5Fis_accessible(path, H5P_DEFAULT);
  htri_t not_hdf5 = H5Fis_accessible(junk, H5P_DEFAULT);
  /* A missing path may report false or fail outright; both are fine. What is
     not fine is crashing, so the value is only required to be "not true".
     Wrapped in H5E_BEGIN_TRY because the expected failure would otherwise
     print an error stack that reads like a test failure. */
  htri_t missing;
  H5E_BEGIN_TRY { missing = H5Fis_accessible(absent, H5P_DEFAULT); }
  H5E_END_TRY;

  int ok_real = (real > 0);
  int ok_junk = (not_hdf5 == 0);
  int ok_missing = (missing <= 0);

  /* 4: the DELETE path, the other callback invoked with obj == NULL. */
  herr_t del;
  H5E_BEGIN_TRY { del = H5Fdelete(path, H5P_DEFAULT); }
  H5E_END_TRY;
  int ok_delete = (del >= 0) && (access(path, F_OK) != 0);

  unlink(junk);

  int ok = ok_real && ok_junk && ok_missing && ok_delete;
  printf("is_accessible real=%d junk=%d missing=%d, delete=%d: %s\n",
         (int)real, (int)not_hdf5, (int)missing, (int)del, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
